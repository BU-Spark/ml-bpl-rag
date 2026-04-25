"""
retrieval/retriever.py

Hybrid retrieval — three parallel paths fused via RRF:
  A) Dense search on chunks (full-text)
  B) Sparse re-rank on dense candidates
  C) Graph retrieval via Neo4j (entity + co-occurrence)
  D) Metadata embedding search on documents
  E) RRF fusion of all result lists
  F) Final rerank by combined score
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import time

from config import (
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_FINAL,
    RRF_K,
    CONTENT_WEIGHT,
    METADATA_WEIGHT,
    MIN_RELEVANCE_SCORE,
    GRAPH_RAG_ENABLED,
    GRAPH_TOP_K,
)
from database.schema import get_conn, get_cursor
from embedding.embedder import embedder
from retrieval.query_understanding import QueryIntent


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedDocument:
    ark_id:             str
    document_id:        int
    title:              str
    source_url:         str
    institution:        str
    issue_date:         str
    year:               List[int]
    topics:             List[str]
    geography:          List[str]
    best_chunk_text:    str
    best_chunk_index:   int
    rrf_score:          float
    metadata_sim:       float
    final_score:        float
    exemplary_image_id: str = ""


# ── Filter clause ─────────────────────────────────────────────────────────────

def _build_filter_clause(intent: QueryIntent) -> tuple[str, list]:
    """Build SQL WHERE clause from date filters only."""
    # conditions = []
    # params     = []

    # if intent.date_filter.year_min is not None:
    #     conditions.append(
    #         "(EXTRACT(YEAR FROM d.date_start) >= %s OR d.date_start IS NULL)"
    #     )
    #     params.append(float(intent.date_filter.year_min))

    # if intent.date_filter.year_max is not None:
    #     conditions.append(
    #         "(EXTRACT(YEAR FROM d.date_start) <= %s OR d.date_start IS NULL)"
    #     )
    #     params.append(float(intent.date_filter.year_max))

    # where = "WHERE " + " AND ".join(conditions) if conditions else ""
    # return where, params
    return "", []


# ── Dense chunk search ────────────────────────────────────────────────────────

def _dense_search(
    query_embedding: np.ndarray,
    where_clause: str,
    where_params: list,
    conn,
    top_k: int = TOP_K_DENSE,
) -> List[dict]:
    """HNSW vector search on chunk embeddings with post-filter date constraints."""
    emb_list = query_embedding.tolist()

    # Build date filter for application AFTER HNSW search
    date_filter = ""
    if where_clause:
        # Strip "WHERE " and keep the rest
        date_filter = "WHERE " + where_clause.replace("WHERE ", "")
    
    sql = f"""
        WITH top_chunks AS (
            -- Step 1: HNSW search unrestricted - fast
            SELECT
                c.document_id,
                c.id           AS chunk_id,
                c.chunk_index,
                c.chunk_text,
                c.sparse_token_ids,
                c.sparse_weights,
                (c.text_embedding <=> %s::vector) AS distance
            FROM chunks c
            ORDER BY c.text_embedding <=> %s::vector
            LIMIT %s
        ),
        nearest_chunks AS (
            -- Step 2: Deduplicate to best chunk per document
            SELECT DISTINCT ON (document_id)
                *,
                1 - distance AS dense_score
            FROM top_chunks
            ORDER BY document_id, distance
        )
        SELECT
            nc.*,
            d.ark_id, d.title, d.source_url, d.institution,
            d.issue_date, d.year, d.topics, d.geography,
            d.metadata_embedding,
            d.exemplary_image_id,
            d.char_count,  
            d.sparse_token_ids AS doc_sparse_token_ids,
            d.sparse_weights   AS doc_sparse_weights
        FROM nearest_chunks nc
        JOIN documents d ON d.id = nc.document_id
        {date_filter}
        ORDER BY nc.dense_score DESC
    """

    # HNSW fetches top_k * 3 chunks, then date filter is applied after join
    params = [emb_list, emb_list, top_k * 3] + where_params

    with get_cursor(conn) as cur:
        cur.execute(sql, params)
        return cur.fetchall()

# ── Sparse re-ranking on dense candidates ────────────────────────────────────

def _sparse_search(
    query_sparse: dict,
    dense_results: List[dict],
    top_k: int = TOP_K_BM25,
) -> List[dict]:
    """Score sparse on dense candidates only — no extra DB call."""
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


# ── Metadata document search ──────────────────────────────────────────────────

def _metadata_search(
    query_embedding: np.ndarray,
    where_clause: str,
    where_params: list,
    conn,
    top_k: int = TOP_K_DENSE,
) -> List[dict]:
    """Search all documents by metadata embedding similarity."""
    emb_list = query_embedding.tolist()

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
            d.char_count,  
            d.exemplary_image_id,
            d.sparse_token_ids AS doc_sparse_token_ids,
            d.sparse_weights   AS doc_sparse_weights,
            '' AS chunk_text,
            0  AS chunk_index,
            1 - (d.metadata_embedding <=> %s::vector) AS dense_score
        FROM documents d
        {where_clause}
        ORDER BY d.metadata_embedding <=> %s::vector
        LIMIT %s
    """

    params = [emb_list] + where_params + [emb_list, top_k]

    with get_cursor(conn) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ── Graph retrieval ───────────────────────────────────────────────────────────

def _graph_search(
    query_embedding: np.ndarray,
    exclude_ark_ids: set,
    conn,
    top_k: int = GRAPH_TOP_K,
) -> List[dict]:
    """
    Graph retrieval via Neo4j entity matching + two-hop traversal.
    Fetches best chunk per document by embedding similarity from PostgreSQL.
    Returns results in the same dict format as dense/metadata search for RRF.
    """
    from graph.graph_retriever import retrieve_by_query

    graph_results = retrieve_by_query(
        query_embedding = query_embedding,
        exclude_ark_ids = exclude_ark_ids,
        top_k           = top_k,
    )

    if not graph_results:
        print("[retriever] Graph returned no results")
        return []

    # Fetch full document details + best chunk by embedding similarity
    graph_ark_ids = [r.ark_id for r in graph_results]
    emb_list      = query_embedding.tolist()

    sql = """
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
            c.chunk_text,
            c.chunk_index
        FROM documents d
        LEFT JOIN LATERAL (
            SELECT chunk_text, chunk_index
            FROM chunks
            WHERE document_id = d.id
            ORDER BY text_embedding <=> %s::vector
            LIMIT 1
        ) c ON true
        WHERE d.ark_id = ANY(%s)
    """

    with get_cursor(conn) as cur:
        cur.execute(sql, (emb_list, graph_ark_ids))
        rows = {row["ark_id"]: row for row in cur.fetchall()}

    # Build result dicts in same format as dense/metadata results
    results = []
    for graph_result in graph_results:
        row = rows.get(graph_result.ark_id)
        if not row:
            continue
        result = dict(row)
        result["graph_score"] = graph_result.graph_score
        results.append(result)

    return results


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    *result_lists: List[dict],
    k: int = RRF_K,
) -> List[tuple[str, float, dict]]:
    """
    Merge any number of ranked lists using RRF.
    Each list contributes 1/(k + rank) to the score.
    Empty lists are skipped silently.
    """
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}

    for result_list in result_lists:
        if not result_list:
            continue
        for rank, row in enumerate(result_list, start=1):
            aid = row["ark_id"]
            scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
            if aid not in rows:
                rows[aid] = row

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(aid, score, rows[aid]) for aid, score in ranked]


# ── Final rerank ──────────────────────────────────────────────────────────────


def _rerank(
    rrf_results:     List[tuple[str, float, dict]],
    query_embedding: np.ndarray,
    intent:          QueryIntent,                   # ← add this
    top_k:           int = TOP_K_FINAL,
) -> List[RetrievedDocument]:
    """
    Final rerank blending RRF score with metadata embedding similarity.
    """
    if not rrf_results:
        return []

    final: List[RetrievedDocument] = []

    for ark_id, rrf_score, row in rrf_results:
        meta_emb = row.get("metadata_embedding")
        if meta_emb is not None:
            if isinstance(meta_emb, str):
                meta_emb = json.loads(meta_emb)
            meta_vec = np.array(meta_emb, dtype=np.float32)
            meta_sim = float(np.dot(query_embedding, meta_vec))
            meta_sim = max(0.0, min(1.0, meta_sim))
        else:
            meta_sim = 0.0

        # ── Soft date penalty ─────────────────────────────────────────────
        date_penalty = 0.0
        if intent.date_filter.year_min is not None or intent.date_filter.year_max is not None:
            doc_years = row.get("year") or []
            doc_year  = doc_years[0] if doc_years else None
            if doc_year:
                if intent.date_filter.year_max and doc_year > intent.date_filter.year_max:
                    date_penalty = min((doc_year - intent.date_filter.year_max) / 50.0, 0.3)
                elif intent.date_filter.year_min and doc_year < intent.date_filter.year_min:
                    date_penalty = min((intent.date_filter.year_min - doc_year) / 50.0, 0.3)
        # ─────────────────────────────────────────────────────────────────

        final_score = (CONTENT_WEIGHT * rrf_score + METADATA_WEIGHT * meta_sim) * (1 - date_penalty)

        final.append(RetrievedDocument(
            ark_id             = ark_id,
            document_id        = row["document_id"],
            title              = row.get("title", ""),
            source_url         = row.get("source_url", ""),
            institution        = row.get("institution", ""),
            issue_date         = row.get("issue_date", ""),
            year               = row.get("year") or [],
            topics             = row.get("topics") or [],
            geography          = row.get("geography") or [],
            best_chunk_text    = row.get("chunk_text", ""),
            best_chunk_index   = row.get("chunk_index", 0),
            exemplary_image_id = row.get("exemplary_image_id") or "",
            rrf_score          = rrf_score,
            metadata_sim       = meta_sim,
            final_score        = final_score,
        ))

    final.sort(key=lambda x: x.final_score, reverse=True)
    final = [doc for doc in final if doc.final_score >= MIN_RELEVANCE_SCORE]
    return final[:top_k]


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(intent: QueryIntent, top_k: int = TOP_K_FINAL) -> Tuple[List[RetrievedDocument], np.ndarray]:
    """
    Delegates to the fusion_title_dedup strategy:
      1. Dual retrieval — tier1 (entity_expand + diversity caps + graph-on)
         AND unified (rewrite + no caps + graph-off), each via dense+sparse+
         metadata RRF.
      2. 1:1 interleave of the two ranked lists.
      3. Strict per-title dedup (keep first occurrence) to kill the
         "Boston Traveler x100" monoculture.

    Returns (documents, query_embedding) — embedding kept for backward compat
    with callers (pipeline.py discards it).
    """
    from retrieval.strategies.fusion_title_dedup import retrieve as fusion_retrieve

    t = time.monotonic()
    print("[retrieve] running fusion_title_dedup", flush=True)
    documents = fusion_retrieve(intent, top_k=top_k)
    print(f"[timing] fusion_title_dedup: {time.monotonic()-t:.2f}s", flush=True)

    # Embedding kept for return-tuple compatibility — strategy already used it
    # internally; recompute here only for the return contract. Cheap relative
    # to retrieval and only happens once per query.
    query_emb = embedder.encode_one_both(intent.rewritten_query, is_query=True)["dense"]
    return documents, query_emb

# def retrieve(intent: QueryIntent, top_k: int = TOP_K_FINAL) -> Tuple[List[RetrievedDocument], np.ndarray]:
#     """
#     Full hybrid retrieval:
#       1. Embed query (dense + sparse in one pass)
#       2. Dense chunk search (HNSW)
#       3. Sparse re-rank on dense candidates
#       4. Metadata document search
#       5. Graph retrieval via Neo4j (if enabled, skipped for fulltext queries)
#       6. RRF fusion of all result lists
#       7. Final rerank with soft date penalty and query-type-aware weights
#     """
#     # No hard date filter — dates handled as soft penalty in _rerank
#     where_clause, where_params = "", []

#     # Use keyword_query for fulltext, rewritten_query for metadata
#     if intent.query_type == "fulltext":
#         search_query = intent.keyword_query or intent.rewritten_query
#     else:
#         search_query = intent.rewritten_query

#     query_output = embedder.encode_one_both(search_query, is_query=True)
#     query_emb    = query_output["dense"]
#     query_sparse = query_output["sparse"]

#     with get_conn() as conn:
#         # Path A: chunk-level dense search
#         t = time.monotonic()
#         print("[retrieve] starting dense", flush=True)
#         print(f"[retrieve] query_type: '{intent.query_type}'", flush=True)
#         print(f"[retrieve] search_query: '{search_query}'", flush=True)
#         dense_results = _dense_search(query_emb, where_clause, where_params, conn)
#         print(f"[timing] dense: {time.monotonic()-t:.2f}s", flush=True)

#         # Path B: sparse re-rank on dense candidates
#         t = time.monotonic()
#         print("[retrieve] starting sparse", flush=True)
#         sparse_results = _sparse_search(query_sparse, dense_results)
#         print(f"[timing] sparse: {time.monotonic()-t:.2f}s", flush=True)

#         # Path C: document-level metadata search
#         # Skip for fulltext queries — metadata embedding of newspaper titles is not useful
#         meta_results = []
#         if intent.query_type != "fulltext":
#             t = time.monotonic()
#             print("[retrieve] starting meta", flush=True)
#             meta_results = _metadata_search(
#                 query_emb, where_clause, where_params, conn,
#                 top_k=TOP_K_DENSE,
#             )
#             print(f"[timing] meta: {time.monotonic()-t:.2f}s", flush=True)
#         else:
#             print("[retrieve] skipping meta (fulltext query)", flush=True)

#         # Path D: graph retrieval
#         # Skip for fulltext queries — graph entity matching not useful for newspaper search
#         graph_results = []
#         if GRAPH_RAG_ENABLED and intent.query_type != "fulltext":
#             t = time.monotonic()
#             print("[retrieve] starting graph", flush=True)
#             graph_results = _graph_search(
#                 query_emb,
#                 exclude_ark_ids = set(),
#                 conn            = conn,
#                 top_k           = GRAPH_TOP_K,
#             )
#             print(f"[timing] graph: {time.monotonic()-t:.2f}s", flush=True)
#         else:
#             print("[retrieve] skipping graph (fulltext query or disabled)", flush=True)

#     # Fuse all paths via RRF
#     rrf_results = _reciprocal_rank_fusion(
#         dense_results,
#         sparse_results,
#         meta_results,
#         graph_results,
#     )

#     return _rerank(rrf_results, query_emb, intent=intent, top_k=top_k), query_emb