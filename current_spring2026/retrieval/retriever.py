"""
retrieval/retriever.py

Two-stage hybrid retrieval:
  Stage 1 — SQL filter by date from QueryIntent
  Stage 2 — Hybrid search:
    A) Dense search on chunk text_embedding (BGE-M3)
    B) Sparse search on chunk sparse_token_ids/sparse_weights (BGE-M3)
    C) Reciprocal Rank Fusion (RRF)
    D) Metadata embedding similarity rerank
    E) Metadata-only search for metadata_driven queries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import numpy as np

from config import (
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_FINAL,
    RRF_K,
    CONTENT_WEIGHT,
    METADATA_WEIGHT,
)
from database.schema import get_conn, get_cursor
from embedding.embedder import embedder
from retrieval.query_understanding import QueryIntent


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedDocument:
    ark_id:           str
    document_id:      int
    title:            str
    source_url:       str
    institution:      str
    issue_date:       str
    year:             List[int]
    topics:           List[str]
    geography:        List[str]
    best_chunk_text:  str
    best_chunk_index: int
    rrf_score:        float
    metadata_sim:     float
    final_score:      float
    exemplary_image_id: str = ""  # add this


# ── Filter clause ─────────────────────────────────────────────────────────────

def _build_filter_clause(intent: QueryIntent) -> tuple[str, list]:
    conditions = []
    params     = []

    if intent.date_filter.year_min is not None:
        conditions.append(
            "(EXTRACT(YEAR FROM d.date_start) >= %s OR d.date_start IS NULL)"
        )
        params.append(float(intent.date_filter.year_min))

    if intent.date_filter.year_max is not None:
        conditions.append(
            "(EXTRACT(YEAR FROM d.date_start) <= %s OR d.date_start IS NULL)"
        )
        params.append(float(intent.date_filter.year_max))

    # Only require chunks for non-metadata-driven queries
    if intent.query_type != "metadata_driven":
        conditions.append(
            "EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"
        )

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


# ── Dense search ──────────────────────────────────────────────────────────────

def _dense_search(
    query_embedding: np.ndarray,
    where_clause: str,
    where_params: list,
    conn,
    top_k: int = TOP_K_DENSE,
) -> List[dict]:
    emb_list = query_embedding.tolist()

    sql = f"""
        WITH filtered_docs AS (
            SELECT d.id AS document_id, d.ark_id
            FROM documents d
            {where_clause}
        ),
        nearest_chunks AS (
            SELECT DISTINCT ON (c.document_id)
                c.document_id,
                c.id             AS chunk_id,
                c.chunk_index,
                c.chunk_text,
                c.sparse_token_ids,    
                c.sparse_weights,      
                1 - (c.text_embedding <=> %s::vector) AS dense_score
            FROM chunks c
            INNER JOIN filtered_docs fd ON c.document_id = fd.document_id
            ORDER BY c.document_id, c.text_embedding <=> %s::vector
        )
        SELECT
            nc.*,
            d.ark_id, d.title, d.source_url, d.institution,
            d.issue_date, d.year, d.topics, d.geography,
            d.metadata_embedding,
            d.exemplary_image_id,
            d.sparse_token_ids AS doc_sparse_token_ids,
            d.sparse_weights   AS doc_sparse_weights
        FROM nearest_chunks nc
        JOIN documents d ON d.id = nc.document_id
        ORDER BY nc.dense_score DESC
        LIMIT %s
    """

    params = where_params + [emb_list, emb_list, top_k]

    with get_cursor(conn) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ── Sparse search ─────────────────────────────────────────────────────────────

def _sparse_search(
    query_sparse: dict,
    dense_results: List[dict],   # pass dense results in directly
    top_k: int = TOP_K_BM25,
) -> List[dict]:
    """
    Score sparse embeddings only on the candidates already returned
    by dense search. Avoids fetching all chunks from DB.
    """
    if not query_sparse or not dense_results:
        return []

    scored = []
    for row in dense_results:
        token_ids    = row.get("sparse_token_ids") or []
        weights      = row.get("sparse_weights")   or []
        chunk_sparse = dict(zip([str(t) for t in token_ids], weights))
        score = sum(
            float(query_sparse.get(tok, 0.0)) * float(weight)
            for tok, weight in chunk_sparse.items()
        )
        row_dict = dict(row)
        row_dict["bm25_score"] = score
        scored.append(row_dict)

    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


# ── Metadata-only search ──────────────────────────────────────────────────────

def _metadata_only_search(
    query_embedding: np.ndarray,
    query_sparse: dict,
    where_clause: str,
    where_params: list,
    conn,
    top_k: int = TOP_K_DENSE,
) -> List[dict]:
    """
    Search documents table directly by metadata embedding.
    Used for metadata_driven queries where documents may have no chunks.
    """
    emb_list = query_embedding.tolist()

    # Remove EXISTS check for this search path
    if "EXISTS" in where_clause:
        parts = where_clause.replace("WHERE ", "").split(" AND ")
        parts = [p for p in parts if "EXISTS" not in p]
        clean_where  = "WHERE " + " AND ".join(parts) if parts else ""
        clean_params = list(where_params)
    else:
        clean_where  = where_clause
        clean_params = list(where_params)

    sql = f"""
        SELECT
            d.id AS document_id,
            d.ark_id,
            d.title,
            d.source_url,
            d.institution,
            d.issue_date,
            d.year,
            d.topics,
            d.geography,
            d.metadata_embedding,
            d.exemplary_image_id,
            d.sparse_token_ids AS doc_sparse_token_ids,
            d.sparse_weights   AS doc_sparse_weights,
            '' AS chunk_text,
            0  AS chunk_index,
            1 - (d.metadata_embedding <=> %s::vector) AS dense_score
        FROM documents d
        {clean_where}
        ORDER BY d.metadata_embedding <=> %s::vector
        LIMIT %s
    """

    params = [emb_list] + clean_params + [emb_list, top_k]

    with get_cursor(conn) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ── RRF ───────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    dense_results:  List[dict],
    sparse_results: List[dict],
    k: int = RRF_K,
) -> List[tuple[str, float, dict]]:
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}

    for rank, row in enumerate(dense_results, start=1):
        aid = row["ark_id"]
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
        rows[aid]   = row

    for rank, row in enumerate(sparse_results, start=1):
        aid = row["ark_id"]
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
        if aid not in rows:
            rows[aid] = row

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(aid, score, rows[aid]) for aid, score in ranked]


# ── Metadata rerank ───────────────────────────────────────────────────────────

def _metadata_rerank(
    rrf_results:    List[tuple[str, float, dict]],
    query_embedding: np.ndarray,
    content_weight:  float = CONTENT_WEIGHT,
    metadata_weight: float = METADATA_WEIGHT,
    top_k: int             = TOP_K_FINAL,
) -> List[RetrievedDocument]:
    if not rrf_results:
        return []

    max_rrf   = rrf_results[0][1]
    min_rrf   = rrf_results[-1][1]
    rrf_range = max_rrf - min_rrf if max_rrf != min_rrf else 1.0

    final: List[RetrievedDocument] = []

    for ark_id, rrf_score, row in rrf_results:
        norm_rrf = rrf_score

        meta_emb = row.get("metadata_embedding")
        if meta_emb is not None:
            if isinstance(meta_emb, str):
                meta_emb = json.loads(meta_emb)
            meta_vec = np.array(meta_emb, dtype=np.float32)
            meta_sim = float(np.dot(query_embedding, meta_vec))
            meta_sim = max(0.0, min(1.0, meta_sim))
        else:
            meta_sim = 0.0

        final_score = content_weight * norm_rrf + metadata_weight * meta_sim

        final.append(RetrievedDocument(
            ark_id           = ark_id,
            document_id      = row["document_id"],
            title            = row.get("title", ""),
            source_url       = row.get("source_url", ""),
            institution      = row.get("institution", ""),
            issue_date       = row.get("issue_date", ""),
            year             = row.get("year") or [],
            topics           = row.get("topics") or [],
            geography        = row.get("geography") or [],
            best_chunk_text  = row.get("chunk_text", ""),
            best_chunk_index = row.get("chunk_index", 0),
            exemplary_image_id = row.get("exemplary_image_id") or "",
            rrf_score        = rrf_score,
            metadata_sim     = meta_sim,
            final_score      = final_score,
        ))

    final.sort(key=lambda x: x.final_score, reverse=True)
    # Filter out documents below minimum relevance threshold
    from config import MIN_RELEVANCE_SCORE
    final = [doc for doc in final if doc.final_score >= MIN_RELEVANCE_SCORE]
    return final[:top_k]


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(intent: QueryIntent, top_k: int = TOP_K_FINAL) -> List[RetrievedDocument]:
    where_clause, where_params = _build_filter_clause(intent)

    query_output = embedder.encode_one_both(intent.rewritten_query, is_query=True)
    query_emb    = query_output["dense"]
    query_sparse = query_output["sparse"]

    with get_conn() as conn:
        # Dense search fetches top candidates from DB
        dense_results = _dense_search(
            query_emb, where_clause, where_params, conn
        )

        # Sparse search scores only the dense candidates — no extra DB call
        sparse_results = _sparse_search(
            query_sparse, dense_results
        )

        meta_results = []
        if intent.query_type == "metadata_driven" or not dense_results:
            meta_results = _metadata_only_search(
                query_emb, query_sparse,
                where_clause, where_params, conn,
                top_k=TOP_K_DENSE,
            )

    rrf_results = _reciprocal_rank_fusion(dense_results, sparse_results)

    rrf_ark_ids = {ark for ark, _, _ in rrf_results}
    for i, row in enumerate(meta_results):
        if row["ark_id"] not in rrf_ark_ids:
            if intent.query_type == "metadata_driven":
                score = 1.0 / (RRF_K + i + 1) * 2.0
            else:
                score = 0.1
            rrf_results.append((row["ark_id"], score, row))

    results = _metadata_rerank(
        rrf_results,
        query_emb,
        content_weight  = intent.content_weight,
        metadata_weight = intent.metadata_weight,
        top_k           = top_k,
    )

    return results