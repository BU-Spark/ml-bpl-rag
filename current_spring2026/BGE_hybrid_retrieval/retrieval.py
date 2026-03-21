#!/usr/bin/env python3
"""
Hybrid retrieval module for the BGE-M3 RAG system (FAISS-backed).

Retrieval strategy
------------------
1. Chunk-level dense retrieval — FAISS cosine similarity on full_article.faiss (top K_dense)
2. Title-level dense retrieval — FAISS cosine similarity on title_summary.faiss (top K_title)
3. RRF fusion                  — Reciprocal Rank Fusion merges both ranked lists
4. Metadata filters            — year constraints applied as post-retrieval filter

Reciprocal Rank Fusion (Cormack et al., 2009):
    rrf_score(d) = Σ  1 / (k + rank_i(d))
where k=60 is the smoothing constant and the sum is over retrieval methods.
"""

import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document

from .embedder import BGEM3Embedder
from .filters import extract_filters_with_llm
from .models import SearchFilters

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF smoothing constant

DEFAULT_INDEX_DIR = Path(__file__).parent / "indexes" / "1940"


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_indexes(index_dir: Path = DEFAULT_INDEX_DIR):
    """
    Load FAISS indexes and metadata from disk.

    Returns:
        (chunk_index, title_index, chunk_meta, title_meta, doc_chunks_map)
    """
    chunk_index = faiss.read_index(str(index_dir / "full_article.faiss"))
    title_index = faiss.read_index(str(index_dir / "title_summary.faiss"))

    with open(index_dir / "chunk_meta.pkl", "rb") as f:
        chunk_meta = pickle.load(f)
    with open(index_dir / "title_meta.pkl", "rb") as f:
        title_meta = pickle.load(f)
    with open(index_dir / "doc_chunks_map.pkl", "rb") as f:
        doc_chunks_map = pickle.load(f)

    logger.info(
        f"Loaded indexes from {index_dir}: "
        f"{chunk_index.ntotal} chunk vectors, "
        f"{title_index.ntotal} title vectors"
    )
    return chunk_index, title_index, chunk_meta, title_meta, doc_chunks_map


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def _passes_filter(meta: dict, filters: Optional[SearchFilters]) -> bool:
    """Return True if a metadata dict satisfies the given filters."""
    if filters is None:
        return True

    date_iso = meta.get("date_iso", [])
    if date_iso:
        try:
            year = int(date_iso[0][:4])
            if filters.year_exact is not None and year != filters.year_exact:
                return False
            if filters.year_start is not None and year < filters.year_start:
                return False
            if filters.year_end is not None and year > filters.year_end:
                return False
        except (ValueError, IndexError):
            pass

    return True


# ---------------------------------------------------------------------------
# Chunk-level dense retrieval
# ---------------------------------------------------------------------------

def _chunk_retrieval(
    chunk_index,
    chunk_meta: List[dict],
    dense_vec: List[float],
    filters: Optional[SearchFilters],
    k: int,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k chunks by cosine similarity via the full_article FAISS index.

    Returns list of (Document, cosine_score) sorted descending.
    """
    query = np.array([dense_vec], dtype=np.float32)
    faiss.normalize_L2(query)

    fetch_k = min(k * 3, chunk_index.ntotal)
    scores, indices = chunk_index.search(query, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_meta):
            continue
        meta = chunk_meta[idx]
        if not _passes_filter(meta, filters):
            continue
        doc = Document(
            page_content=meta.get("chunk_text", "")[:4000],
            metadata={
                "source": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_idx", 0),
                "dense_score": float(score),
                "record_id": meta.get("record_id", ""),
                "source_url": meta.get("source_url", ""),
                "newspaper": meta.get("newspaper", ""),
                "issue_date": meta.get("issue_date", ""),
                "date_iso": meta.get("date_iso", []),
                "topics": meta.get("topics", []),
                "geography": meta.get("geography", []),
            },
        )
        results.append((doc, float(score)))
        if len(results) >= k:
            break

    return results


# ---------------------------------------------------------------------------
# Title-level dense retrieval
# ---------------------------------------------------------------------------

def _title_retrieval(
    title_index,
    title_meta: List[dict],
    doc_chunks_map: Dict[str, List[dict]],
    dense_vec: List[float],
    filters: Optional[SearchFilters],
    k: int,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k documents by title-level cosine similarity via the title_summary FAISS index.
    Uses the first available chunk as the representative page content.

    Returns list of (Document, cosine_score) sorted descending.
    """
    query = np.array([dense_vec], dtype=np.float32)
    faiss.normalize_L2(query)

    fetch_k = min(k * 2, title_index.ntotal)
    scores, indices = title_index.search(query, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(title_meta):
            continue
        meta = title_meta[idx]
        if not _passes_filter(meta, filters):
            continue

        doc_id = meta.get("doc_id", "")
        chunks = doc_chunks_map.get(doc_id, [])
        page_content = chunks[0].get("chunk_text", "") if chunks else meta.get("title", "")

        doc = Document(
            page_content=page_content[:4000],
            metadata={
                "source": doc_id,
                "chunk_index": 0,
                "dense_score": float(score),
                "record_id": meta.get("record_id", ""),
                "source_url": meta.get("source_url", ""),
                "newspaper": meta.get("newspaper", ""),
                "issue_date": meta.get("issue_date", ""),
                "date_iso": meta.get("date_iso", []),
                "topics": meta.get("topics", []),
                "geography": meta.get("geography", []),
            },
        )
        results.append((doc, float(score)))
        if len(results) >= k:
            break

    return results


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = RRF_K,
) -> List[Tuple[Document, float]]:
    """
    Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a list of (Document, score) sorted by
                      relevance descending.
        k:            RRF smoothing constant (default 60).

    Returns:
        Merged list of (Document, rrf_score) sorted descending.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked, start=1):
            key = f"{doc.metadata.get('source')}::{doc.metadata.get('chunk_index')}"
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in merged]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    chunk_index,
    title_index,
    chunk_meta: List[dict],
    title_meta: List[dict],
    doc_chunks_map: Dict[str, List[dict]],
    embedder: BGEM3Embedder,
    query: str,
    llm: Any,
    k_dense: int = 150,
    k_title: int = 50,
    filters: Optional[SearchFilters] = None,
) -> Tuple[List[Document], List[float]]:
    """
    Hybrid retrieval: chunk-level dense + title-level dense → RRF fusion.

    Args:
        chunk_index:    FAISS index for full_article chunks.
        title_index:    FAISS index for title summaries.
        chunk_meta:     List of chunk metadata dicts.
        title_meta:     List of title metadata dicts.
        doc_chunks_map: Dict mapping doc_id → list of chunk dicts.
        embedder:       BGEM3Embedder instance.
        query:          Expanded user query string.
        llm:            LangChain LLM (for filter extraction).
        k_dense:        Number of chunk-level candidates to fetch.
        k_title:        Number of title-level candidates to fetch.
        filters:        Pre-computed SearchFilters (skips LLM call if provided).

    Returns:
        (fused_docs, rrf_scores) — documents sorted by RRF score descending.
    """
    start = time.time()
    logger.info("Starting hybrid retrieval (FAISS-backed)...")

    # 1. Encode query — only dense needed for FAISS search
    dense_vec, _ = embedder.embed_query(query)
    logger.info(f"Query encoded in {time.time() - start:.2f}s")

    # 2. Extract metadata filters (reuse if already computed upstream)
    if filters is None:
        filters = extract_filters_with_llm(query, llm)

    # 3. Chunk-level dense retrieval
    t1 = time.time()
    chunk_results = _chunk_retrieval(chunk_index, chunk_meta, dense_vec, filters, k_dense)
    logger.info(f"Chunk retrieval: {len(chunk_results)} results in {time.time() - t1:.2f}s")

    # 4. Title-level dense retrieval
    t2 = time.time()
    title_results = _title_retrieval(
        title_index, title_meta, doc_chunks_map, dense_vec, filters, k_title
    )
    logger.info(f"Title retrieval: {len(title_results)} results in {time.time() - t2:.2f}s")

    if not chunk_results and not title_results:
        logger.warning("No results from either retrieval path.")
        return [], []

    # 5. RRF fusion
    fused = _reciprocal_rank_fusion([chunk_results, title_results])
    docs = [doc for doc, _ in fused]
    scores = [score for _, score in fused]

    logger.info(
        f"Hybrid retrieval complete: {len(docs)} fused results in "
        f"{time.time() - start:.2f}s total."
    )
    return docs, scores
