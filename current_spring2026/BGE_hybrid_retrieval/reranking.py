#!/usr/bin/env python3
"""
Reranking module for the BGE-M3 hybrid RAG system.

Two-stage reranking
-------------------
Stage 1 (fast, before cross-encoder):
    Merge document chunks by source, apply metadata field weights, and apply
    a year-match boost for temporal queries.

Stage 2 (precise, optional):
    Cross-encoder reranking using BAAI/bge-reranker-v2-m3, which shares the
    same model family as the retrieval encoder.  Enable by passing
    use_cross_encoder=True to rerank().
"""

import logging
import re
import time
from collections import defaultdict
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata field weights (Stage 1)
# ---------------------------------------------------------------------------

METADATA_WEIGHTS = {
    "title_info_primary_tsi": 1.5,
    "name_role_tsim":         1.4,
    "date_tsim":              1.3,
    "abstract_tsi":           1.0,
    "note_tsim":              0.8,
    "subject_geographic_sim": 0.5,
    "genre_basic_ssim":       0.5,
    "genre_specific_ssim":    0.5,
}

YEAR_BOOST = 50.0  # applied when query year matches a document's date field


# ---------------------------------------------------------------------------
# Stage 1 — metadata-boosted reranking
# ---------------------------------------------------------------------------

def _extract_years(query: str) -> List[str]:
    return re.findall(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", query)


def _merge_chunks(docs: List[Document]) -> List[Document]:
    """Merge all chunks belonging to the same document_id."""
    grouped: dict = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source")].append(doc)

    merged = []
    for src, chunks in grouped.items():
        text = " ".join(c.page_content for c in chunks if c.page_content)
        merged.append(Document(page_content=text, metadata=chunks[0].metadata))
    return merged


def _metadata_score(doc: Document, query_years: List[str]) -> float:
    score = 1.0
    for field, weight in METADATA_WEIGHTS.items():
        if doc.metadata.get(field):
            score += weight

    date_str = str(doc.metadata.get("date_tsim", ""))
    for year in query_years:
        if re.search(rf"\b{year}\b", date_str):
            score += YEAR_BOOST
            break
    return score


def stage1_rerank(docs: List[Document], query: str, top_k: int) -> List[Document]:
    """Metadata-boosted reranking (fast, no model inference)."""
    if not docs:
        return []
    start = time.time()
    query_years = _extract_years(query)
    merged = _merge_chunks(docs)
    scored = sorted(merged, key=lambda d: _metadata_score(d, query_years), reverse=True)
    logger.info(f"Stage-1 rerank: {len(scored)} docs in {time.time()-start:.2f}s")
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Stage 2 — BGE-M3 cross-encoder reranking
# ---------------------------------------------------------------------------

class BGECrossEncoder:
    """
    Wrapper around BAAI/bge-reranker-v2-m3 for cross-encoder reranking.

    Scores each (query, passage) pair with a fine-tuned relevance score.
    """

    MODEL_ID = "BAAI/bge-reranker-v2-m3"

    def __init__(self, use_fp16: bool = True) -> None:
        from FlagEmbedding import FlagReranker  # lazy import
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.MODEL_ID} on {device}...")
        self.reranker = FlagReranker(
            self.MODEL_ID,
            use_fp16=use_fp16 and device == "cuda",
        )
        logger.info("Cross-encoder loaded.")

    def score(self, query: str, passages: List[str]) -> List[float]:
        """Return a relevance score for each (query, passage) pair."""
        pairs = [[query, p] for p in passages]
        return self.reranker.compute_score(pairs, normalize=True)


def stage2_rerank(
    docs: List[Document],
    query: str,
    cross_encoder: "BGECrossEncoder",
    top_k: int,
) -> List[Document]:
    """Cross-encoder reranking — more accurate but slower."""
    if not docs:
        return []
    start = time.time()
    passages = [d.page_content[:512] for d in docs]  # cap length for speed
    scores = cross_encoder.score(query, passages)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    logger.info(
        f"Stage-2 (cross-encoder) rerank: {len(ranked)} docs in {time.time()-start:.2f}s"
    )
    return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def rerank(
    docs: List[Document],
    query: str,
    top_k: int = 10,
    use_cross_encoder: bool = False,
    cross_encoder: "BGECrossEncoder | None" = None,
) -> List[Document]:
    """
    Rerank retrieved documents.

    Args:
        docs:               Candidate documents from hybrid retrieval.
        query:              Expanded user query.
        top_k:              Number of documents to return.
        use_cross_encoder:  If True, apply BGE cross-encoder after Stage 1.
        cross_encoder:      BGECrossEncoder instance (required if use_cross_encoder).

    Returns:
        Top-k reranked Document list.
    """
    # Stage 1 always runs; pass a larger pool to cross-encoder if enabled
    stage1_k = min(top_k * 3, len(docs)) if use_cross_encoder else top_k
    stage1_docs = stage1_rerank(docs, query, top_k=stage1_k)

    if use_cross_encoder and cross_encoder is not None:
        return stage2_rerank(stage1_docs, query, cross_encoder, top_k=top_k)

    return stage1_docs[:top_k]
