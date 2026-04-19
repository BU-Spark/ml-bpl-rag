"""
retrieval/retriever.py

Hybrid retrieval — always runs both paths and merges via RRF:
  A) Dense + sparse search on chunks (full-text)
  B) Metadata embedding search on documents
  C) RRF fusion of all results
  D) Final rerank by combined score

No more content_driven / metadata_driven routing.
Both paths always run. Reranker decides final order.
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
    MIN_RELEVANCE_SCORE,
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

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


# ── Dense chunk search ────────────────────────────────────────────────────────

def _dense_search(
    query_embedding: np.ndarray,
    where_clause: str,
    where_params: list,
    conn,
    top_k: int = TOP_K_DENSE,
) -> List[dict]:
    """HNSW vector search on chunk embeddings."""
    emb_list = query_embedding.tolist()

    # Build exists clause correctly depending on whether where_clause exists
    if where_clause:
        exists_clause = f"{where_clause} AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"
    else:
        exists_clause = "WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"

    sql = f"""
        WITH filtered_docs AS (
            SELECT d.id AS document_id, d.ark_id
            FROM documents d
            {exists_clause}
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
    """
    Search all documents by metadata embedding similarity.
    Includes both full-text documents and metadata-only collection records.
    Always runs regardless of query type.
    """
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


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    *result_lists: List[dict],
    k: int = RRF_K,
) -> List[tuple[str, float, dict]]:
    """
    Merge any number of ranked lists using RRF.
    Each list contributes 1/(k + rank) to the score.
    """
    scores: dict[str, float] = {}
    rows:   dict[str, dict]  = {}

    for result_list in result_lists:
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
    top_k:           int   = TOP_K_FINAL,
) -> List[RetrievedDocument]:
    """
    Final rerank blending RRF score with metadata embedding similarity.
    No content/metadata weight split — uses fixed weights.
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

        # Fixed blend: RRF captures retrieval signal, metadata sim adds document-level signal
        final_score = CONTENT_WEIGHT * rrf_score + METADATA_WEIGHT * meta_sim

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

def retrieve(intent: QueryIntent, top_k: int = TOP_K_FINAL) -> List[RetrievedDocument]:
    """
    Full hybrid retrieval:
      1. Build date filter from intent
      2. Embed query (dense + sparse in one pass)
      3. Dense chunk search (HNSW)
      4. Sparse re-rank on dense candidates
      5. Metadata document search (always runs)
      6. RRF fusion of all three result lists
      7. Final rerank and threshold filter
    """
    where_clause, where_params = _build_filter_clause(intent)

    query_output = embedder.encode_one_both(intent.rewritten_query, is_query=True)
    query_emb    = query_output["dense"]
    query_sparse = query_output["sparse"]

    with get_conn() as conn:
        # Path A: chunk-level search
        dense_results  = _dense_search(
            query_emb, where_clause, where_params, conn
        )
        sparse_results = _sparse_search(
            query_sparse, dense_results
        )

        # Path B: document-level metadata search (always runs)
        meta_results = _metadata_search(
            query_emb, where_clause, where_params, conn,
            top_k=TOP_K_DENSE,
        )

    # Merge all three via RRF
    rrf_results = _reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        meta_results,
    )

    return _rerank(rrf_results, query_emb, top_k=top_k)