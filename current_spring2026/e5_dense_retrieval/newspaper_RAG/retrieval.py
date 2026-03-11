#!/usr/bin/env python3
"""
Two-stage retrieval for the historical newspaper RAG pipeline.

Stage 1 — Title/Summary search:
  Embed the query with E5 and search the Title/Summary FAISS index.
  If the maximum cosine similarity is below `threshold`, return early
  with an empty result ("no results found" gate).

Stage 2 — Full Article retrieval:
  For the doc_ids that passed Stage 1, search the Full Article FAISS
  index (also with E5). Aggregate chunk scores per document (max pooling)
  and return the top-k articles with their reconstructed full texts.

E5 convention: queries use the "query: " prefix.
"""

import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import ArticleResult

logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "


def retrieve(
    query: str,
    model: SentenceTransformer,
    title_index,
    full_index,
    title_meta: List[dict],
    chunk_meta: List[dict],
    doc_chunks_map: dict,
    threshold: float = 0.30,
    top_k: int = 10,
    full_k_multiplier: int = 5,
) -> Tuple[List[ArticleResult], float]:
    """
    Two-stage E5 dense retrieval for newspaper articles.

    Args:
        query:             User query string (no prefix needed here).
        model:             Loaded SentenceTransformer (E5) model.
        title_index:       FAISS FlatIP index over title/summary embeddings.
        full_index:        FAISS FlatIP index over full-article chunk embeddings.
        title_meta:        Metadata aligned with title_index rows.
        chunk_meta:        Metadata aligned with full_index rows.
        doc_chunks_map:    doc_id → list of chunk metadata dicts.
        threshold:         Minimum cosine similarity for Stage 1 to pass.
        top_k:             Number of top articles to return.
        full_k_multiplier: full_index search width = top_k * full_k_multiplier.

    Returns:
        (results, max_title_score)
        results is empty when max_title_score < threshold.
    """
    # ── Embed query ───────────────────────────────────────────────────────────
    q_emb = model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
    ).astype(np.float32)

    # ── Stage 1: Title/Summary search ─────────────────────────────────────────
    n_title = title_index.ntotal
    k1 = min(top_k, n_title)
    scores1, indices1 = title_index.search(q_emb, k1)

    max_score = float(scores1[0][0]) if k1 > 0 else 0.0
    logger.info(
        f"Stage 1 — max title similarity: {max_score:.4f}  (threshold: {threshold})"
    )

    if max_score < threshold:
        logger.info("Below threshold — returning no results.")
        return [], max_score

    # Collect doc_ids that passed the threshold gate
    matched_doc_ids: dict[str, float] = {}  # doc_id → title score
    for score, idx in zip(scores1[0], indices1[0]):
        if idx < 0:
            continue
        doc_id = title_meta[idx]["doc_id"]
        matched_doc_ids[doc_id] = float(score)

    logger.info(f"Stage 1 passed — {len(matched_doc_ids)} issues matched.")

    # ── Stage 2: Full Article search ─────────────────────────────────────────
    n_chunks = full_index.ntotal
    k2 = min(top_k * full_k_multiplier, n_chunks)
    scores2, indices2 = full_index.search(q_emb, k2)

    # Aggregate chunk scores per document using max pooling
    doc_best_score: dict[str, float] = {}

    for score, idx in zip(scores2[0], indices2[0]):
        if idx < 0:
            continue
        meta = chunk_meta[idx]
        doc_id = meta["doc_id"]
        if doc_id not in matched_doc_ids:
            continue  # only consider Stage-1 matched issues
        if float(score) > doc_best_score.get(doc_id, -1.0):
            doc_best_score[doc_id] = float(score)

    # Sort by retrieval score and take top-k
    ranked_doc_ids = sorted(doc_best_score, key=lambda d: doc_best_score[d], reverse=True)[:top_k]
    logger.info(f"Stage 2 — {len(ranked_doc_ids)} articles retrieved.")

    # ── Build ArticleResult objects ───────────────────────────────────────────
    results: List[ArticleResult] = []
    for doc_id in ranked_doc_ids:
        chunks = doc_chunks_map.get(doc_id, [])
        if not chunks:
            continue

        base = chunks[0]  # all chunks share the same base metadata
        # Reconstruct full text from chunks in order
        ordered = sorted(chunks, key=lambda c: c["chunk_idx"])
        full_text = base.get("full_text", " ".join(c["chunk_text"] for c in ordered))

        # Rerank text: first ~1500 chars (sufficient for Cohere cross-encoder)
        rerank_text = full_text[:1500]

        results.append(ArticleResult(
            doc_id=doc_id,
            record_id=base.get("record_id", ""),
            source_url=base.get("source_url", ""),
            newspaper=base.get("newspaper", ""),
            issue_date=base.get("issue_date", ""),
            retrieval_score=doc_best_score[doc_id],
            full_text=full_text[:8000],   # cap for LLM context
            rerank_text=rerank_text,
            topics=base.get("topics", []),
            geography=base.get("geography", []),
        ))

    return results, max_score
