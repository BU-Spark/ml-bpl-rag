"""
_unified_core
--------------
Shared building blocks for the unified-retrieval strategies. Not registered as a
strategy itself.

Runs dense-chunk + sparse-chunk + metadata-only search in parallel, then RRF-
fuses all three. No min-relevance floor (returning [] is strictly worse than
returning the top-k of what we retrieved). Optional title/newspaper diversity
cap, optional entity-expansion rewrite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai import OpenAI

from retrieval.retriever import (
    _dense_search,
    _sparse_search,
    _metadata_search,
    _graph_search,
    RetrievedDocument,
)
from database.schema import get_conn
from embedding.embedder import embedder
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, RRF_K, GRAPH_RAG_ENABLED, GRAPH_TOP_K


_client = OpenAI(api_key=OPENAI_API_KEY)


# ── Filter clause (date only) ─────────────────────────────────────────────────

def _build_filter_clause_unified(intent) -> tuple[str, list]:
    conds, params = [], []
    if intent.date_filter.year_min is not None:
        conds.append("(EXTRACT(YEAR FROM d.date_start) >= %s OR d.date_start IS NULL)")
        params.append(float(intent.date_filter.year_min))
    if intent.date_filter.year_max is not None:
        conds.append("(EXTRACT(YEAR FROM d.date_start) <= %s OR d.date_start IS NULL)")
        params.append(float(intent.date_filter.year_max))
    where = "WHERE " + " AND ".join(conds) if conds else ""
    return where, params


# ── Entity-expansion rewrite ──────────────────────────────────────────────────

_ENTITY_PROMPT = """
You are helping search a Boston Public Library / Digital Commonwealth archive
of photographs, maps, manuscripts, prints, and newspapers. For the user's
query, list 5-10 concrete proper-noun entities that would plausibly appear in
the TITLE of a catalog record relevant to this query. Include:
  - specific people (full names)
  - specific places (buildings, landmarks, neighborhoods, cities)
  - specific events (with dates when known)
  - specific object types (statue, grave, portrait, letter, map, photograph)

Output ONE line of comma-separated phrases, nothing else. No filler.

Query: {query}

Entities:
""".strip()


def entity_expand(raw_query: str) -> str:
    """Return the raw query concatenated with comma-separated catalog-style
    entity phrases. On LLM failure, fall back to raw query."""
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=0.1,
            max_tokens=120,
            messages=[{"role": "user",
                       "content": _ENTITY_PROMPT.format(query=raw_query)}],
        )
        entities = resp.choices[0].message.content.strip().splitlines()[0]
    except Exception:
        return raw_query
    return f"{raw_query}. Related: {entities}"


# ── Diversity cap ─────────────────────────────────────────────────────────────

def diversity_cap(
    rows: list,
    per_title: int = 2,
    per_newspaper: int = 3,
) -> list:
    """Cap how many rows from the same title and same newspaper survive.
    Preserves input order. Handles the failure mode where 100 distinct
    "The Boston Traveler" issues fill the pool.
    """
    kept = []
    title_counts: dict[str, int] = {}
    news_counts:  dict[str, int] = {}
    for row in rows:
        title = (row.get("title") or "").strip().lower()
        news  = (row.get("newspaper") or "").strip().lower() if hasattr(row, "get") else ""
        if title and title_counts.get(title, 0) >= per_title:
            continue
        if news and news_counts.get(news, 0) >= per_newspaper:
            continue
        kept.append(row)
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
        if news:
            news_counts[news] = news_counts.get(news, 0) + 1
    return kept


# ── RRF over N channels ───────────────────────────────────────────────────────

def rrf_n(*channels, k: int = RRF_K) -> list[tuple[str, float, dict]]:
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}
    for channel in channels:
        if not channel:
            continue
        for rank, r in enumerate(channel, start=1):
            aid = r["ark_id"]
            scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
            if aid not in rows:
                rows[aid] = r
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(aid, s, rows[aid]) for aid, s in ranked]


# ── Metadata-embedding rerank, no min-relevance floor ─────────────────────────

def metadata_rerank_no_floor(
    fused: list[tuple[str, float, dict]],
    query_dense: np.ndarray,
    top_k: int,
    content_weight:  float = 0.6,
    metadata_weight: float = 0.4,
) -> list[RetrievedDocument]:
    out: list[RetrievedDocument] = []
    if not fused:
        return out
    for ark_id, rrf_score, row in fused:
        meta_emb = row.get("metadata_embedding")
        if meta_emb is not None:
            if isinstance(meta_emb, str):
                meta_emb = json.loads(meta_emb)
            mv = np.asarray(meta_emb, dtype=np.float32)
            meta_sim = float(np.dot(query_dense, mv))
            meta_sim = max(0.0, min(1.0, meta_sim))
        else:
            meta_sim = 0.0
        final = content_weight * rrf_score + metadata_weight * meta_sim
        out.append(RetrievedDocument(
            ark_id           = ark_id,
            document_id      = row.get("document_id", 0),
            title            = row.get("title", "") or "",
            source_url       = row.get("source_url", "") or "",
            institution      = row.get("institution", "") or "",
            issue_date       = row.get("issue_date", "") or "",
            year             = row.get("year") or [],
            topics           = row.get("topics") or [],
            geography        = row.get("geography") or [],
            best_chunk_text  = row.get("chunk_text", "") or "",
            best_chunk_index = row.get("chunk_index", 0) or 0,
            exemplary_image_id = row.get("exemplary_image_id") or "",
            rrf_score        = rrf_score,
            metadata_sim     = meta_sim,
            final_score      = final,
        ))
    out.sort(key=lambda d: d.final_score, reverse=True)
    return out[:top_k]


# ── Main unified retrieval ────────────────────────────────────────────────────

@dataclass
class UnifiedConfig:
    pool_size:        int   = 200
    rewrite_mode:     str   = "raw"            # "raw" | "rewrite" | "entity_expand"
    apply_title_cap:  bool  = False
    per_title_cap:    int   = 2
    per_newspaper_cap: int  = 3
    content_weight:   float = 0.6
    metadata_weight:  float = 0.4
    use_graph:        bool  = False


def unified_retrieve(
    intent,
    top_k: int,
    cfg: UnifiedConfig,
) -> list[RetrievedDocument]:
    # Choose what text to embed
    if cfg.rewrite_mode == "entity_expand":
        qtext = entity_expand(intent.raw_query or intent.rewritten_query)
    elif cfg.rewrite_mode == "rewrite":
        qtext = intent.rewritten_query or intent.raw_query
    else:  # raw
        qtext = intent.raw_query or intent.rewritten_query

    q = embedder.encode_one_both(qtext, is_query=True)
    q_dense, q_sparse = q["dense"], q["sparse"]

    where, params = _build_filter_clause_unified(intent)

    with get_conn() as conn:
        dense_rows = _dense_search(q_dense, where, params, conn, top_k=cfg.pool_size)
        sparse_rows = _sparse_search(q_sparse, dense_rows, top_k=cfg.pool_size)
        meta_rows = _metadata_search(
            q_dense, where, params, conn, top_k=cfg.pool_size,
        )
        graph_rows: list = []
        if cfg.use_graph and GRAPH_RAG_ENABLED:
            try:
                graph_rows = _graph_search(
                    q_dense,
                    exclude_ark_ids = set(),
                    conn            = conn,
                    top_k           = GRAPH_TOP_K,
                )
            except Exception as e:
                print(f"[unified] graph search failed (non-fatal): {e}")
                graph_rows = []

    if cfg.apply_title_cap:
        dense_rows  = diversity_cap(dense_rows,  cfg.per_title_cap, cfg.per_newspaper_cap)
        sparse_rows = diversity_cap(sparse_rows, cfg.per_title_cap, cfg.per_newspaper_cap)
        meta_rows   = diversity_cap(meta_rows,   cfg.per_title_cap, cfg.per_newspaper_cap)
        if graph_rows:
            graph_rows = diversity_cap(graph_rows, cfg.per_title_cap, cfg.per_newspaper_cap)

    fused = rrf_n(dense_rows, sparse_rows, meta_rows, graph_rows)

    return metadata_rerank_no_floor(
        fused, q_dense, top_k=top_k,
        content_weight  = cfg.content_weight,
        metadata_weight = cfg.metadata_weight,
    )
