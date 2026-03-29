#!/usr/bin/env python3
"""
Hybrid retrieval module for the BGE-M3 RAG system (FAISS + sparse).

Retrieval strategy (true hybrid dense+sparse)
----------------------------------------------
1. Chunk-level dense retrieval   — FAISS cosine on full_article.faiss
2. Chunk-level sparse retrieval  — In-memory inverted index + SPLADE dot product
3. Title-level dense retrieval   — FAISS cosine on title_summary.faiss
4. RRF fusion                    — Reciprocal Rank Fusion merges all 3 ranked lists
5. Metadata filters              — year constraints applied as post-retrieval filter

Reciprocal Rank Fusion (Cormack et al., 2009):
    rrf_score(d) = Σ  1 / (k + rank_i(d))
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

from embedder import BGEM3Embedder
from filters import extract_filters_with_llm
from models import SearchFilters

logger = logging.getLogger(__name__)

RRF_K = 60

DEFAULT_INDEX_DIR = Path(__file__).parent / "indexes" / "1940"


# ---------------------------------------------------------------------------
# Sparse inverted index for fast lexical retrieval
# ---------------------------------------------------------------------------

class SparseInvertedIndex:
    """
    In-memory inverted index over BGE-M3 SPLADE sparse vectors.
    Builds a posting list: token_id → [(chunk_idx, weight), ...].
    At query time, accumulates dot-product scores for each candidate.
    """

    def __init__(self, sparse_vecs: List[Dict[str, float]]) -> None:
        self.n = len(sparse_vecs)
        # posting lists: token_str → list of (doc_idx, weight)
        self.postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for idx, sv in enumerate(sparse_vecs):
            for tok, w in sv.items():
                if w > 0:
                    self.postings[tok].append((idx, w))

    def search(self, query_sparse: Dict[str, float], k: int) -> List[Tuple[int, float]]:
        """
        Return top-k (chunk_idx, score) by sparse dot product.
        Only iterates over tokens present in the query → very fast.
        """
        scores: Dict[int, float] = defaultdict(float)
        for tok, qw in query_sparse.items():
            if qw <= 0 or tok not in self.postings:
                continue
            for doc_idx, dw in self.postings[tok]:
                scores[doc_idx] += qw * dw

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_indexes(index_dir: Path = DEFAULT_INDEX_DIR):
    """
    Load FAISS indexes, sparse vectors, and metadata from disk.

    Returns:
        (chunk_index, title_index, chunk_meta, title_meta, doc_chunks_map,
         sparse_index)  — sparse_index is None if chunk_sparse.pkl missing.
    """
    chunk_index = faiss.read_index(str(index_dir / "full_article.faiss"))
    title_index = faiss.read_index(str(index_dir / "title_summary.faiss"))

    # Set nprobe for IVF indexes
    if hasattr(chunk_index, "nprobe"):
        chunk_index.nprobe = max(chunk_index.nprobe, 8)
    if hasattr(title_index, "nprobe"):
        title_index.nprobe = max(title_index.nprobe, 8)

    with open(index_dir / "chunk_meta.pkl", "rb") as f:
        chunk_meta = pickle.load(f)
    with open(index_dir / "title_meta.pkl", "rb") as f:
        title_meta = pickle.load(f)
    with open(index_dir / "doc_chunks_map.pkl", "rb") as f:
        doc_chunks_map = pickle.load(f)

    # Load sparse vectors (optional — backwards-compatible with old indexes)
    sparse_path = index_dir / "chunk_sparse.pkl"
    sparse_index = None
    if sparse_path.exists():
        with open(sparse_path, "rb") as f:
            sparse_vecs = pickle.load(f)
        sparse_index = SparseInvertedIndex(sparse_vecs)
        logger.info(f"Built sparse inverted index: {sparse_index.n} chunks")

    logger.info(
        f"Loaded indexes from {index_dir}: "
        f"{chunk_index.ntotal} chunk vectors, "
        f"{title_index.ntotal} title vectors, "
        f"sparse={'yes' if sparse_index else 'no'}"
    )
    return chunk_index, title_index, chunk_meta, title_meta, doc_chunks_map, sparse_index


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def _passes_filter(meta: dict, filters: Optional[SearchFilters]) -> bool:
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
# Helper: build a Document from chunk metadata
# ---------------------------------------------------------------------------

def _chunk_to_doc(meta: dict, score: float, score_field: str = "dense_score") -> Document:
    return Document(
        page_content=meta.get("chunk_text", "")[:4000],
        metadata={
            "source": meta.get("doc_id", ""),
            "chunk_index": meta.get("chunk_idx", 0),
            score_field: float(score),
            "record_id": meta.get("record_id", ""),
            "source_url": meta.get("source_url", ""),
            "newspaper": meta.get("newspaper", ""),
            "issue_date": meta.get("issue_date", ""),
            "date_iso": meta.get("date_iso", []),
            "topics": meta.get("topics", []),
            "geography": meta.get("geography", []),
        },
    )


# ---------------------------------------------------------------------------
# Dense chunk retrieval
# ---------------------------------------------------------------------------

def _chunk_dense_retrieval(
    chunk_index,
    chunk_meta: List[dict],
    query_dense: np.ndarray,
    filters: Optional[SearchFilters],
    k: int,
) -> List[Tuple[Document, float]]:
    fetch_k = min(k * 3, chunk_index.ntotal)
    scores, indices = chunk_index.search(query_dense, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_meta):
            continue
        meta = chunk_meta[idx]
        if not _passes_filter(meta, filters):
            continue
        results.append((_chunk_to_doc(meta, score, "dense_score"), float(score)))
        if len(results) >= k:
            break
    return results


# ---------------------------------------------------------------------------
# Sparse chunk retrieval
# ---------------------------------------------------------------------------

def _chunk_sparse_retrieval(
    sparse_index: SparseInvertedIndex,
    chunk_meta: List[dict],
    query_sparse: Dict[str, float],
    filters: Optional[SearchFilters],
    k: int,
) -> List[Tuple[Document, float]]:
    # Fetch more than k to allow for filtering
    raw_hits = sparse_index.search(query_sparse, k * 3)

    results = []
    for idx, score in raw_hits:
        if idx >= len(chunk_meta):
            continue
        meta = chunk_meta[idx]
        if not _passes_filter(meta, filters):
            continue
        results.append((_chunk_to_doc(meta, score, "sparse_score"), float(score)))
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
    query_dense: np.ndarray,
    filters: Optional[SearchFilters],
    k: int,
) -> List[Tuple[Document, float]]:
    fetch_k = min(k * 2, title_index.ntotal)
    scores, indices = title_index.search(query_dense, fetch_k)

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
    k_sparse: int = 100,
    k_title: int = 50,
    filters: Optional[SearchFilters] = None,
    sparse_index: Optional[SparseInvertedIndex] = None,
) -> Tuple[List[Document], List[float]]:
    """
    True hybrid retrieval: dense + sparse + title → RRF fusion.

    Three ranked lists are fused:
      1. Chunk-level dense  (FAISS cosine similarity)
      2. Chunk-level sparse (SPLADE inverted index dot product)
      3. Title-level dense  (FAISS cosine similarity)

    If sparse_index is None (old indexes without chunk_sparse.pkl),
    falls back to dense-only retrieval (2 lists).
    """
    start = time.time()
    logger.info("Starting hybrid retrieval (dense + sparse + title)...")

    # 1. Encode query — both dense and sparse in a single forward pass
    dense_vec, sparse_vec = embedder.embed_query(query)
    query_dense = np.array([dense_vec], dtype=np.float32)
    faiss.normalize_L2(query_dense)
    logger.info(f"Query encoded in {time.time() - start:.2f}s")

    # 2. Filters (reuse if already computed upstream)
    if filters is None:
        filters = extract_filters_with_llm(query, llm)

    # 3. Dense chunk retrieval
    t = time.time()
    dense_results = _chunk_dense_retrieval(chunk_index, chunk_meta, query_dense, filters, k_dense)
    logger.info(f"Dense chunk: {len(dense_results)} in {time.time() - t:.2f}s")

    # 4. Sparse chunk retrieval (if available)
    sparse_results: List[Tuple[Document, float]] = []
    if sparse_index is not None:
        t = time.time()
        sparse_results = _chunk_sparse_retrieval(sparse_index, chunk_meta, sparse_vec, filters, k_sparse)
        logger.info(f"Sparse chunk: {len(sparse_results)} in {time.time() - t:.2f}s")

    # 5. Title-level dense retrieval
    t = time.time()
    title_results = _title_retrieval(
        title_index, title_meta, doc_chunks_map, query_dense, filters, k_title,
    )
    logger.info(f"Title dense: {len(title_results)} in {time.time() - t:.2f}s")

    if not dense_results and not sparse_results and not title_results:
        logger.warning("No results from any retrieval path.")
        return [], []

    # 6. RRF fusion across all available ranked lists
    ranked_lists = [dense_results, title_results]
    if sparse_results:
        ranked_lists.append(sparse_results)
    fused = _reciprocal_rank_fusion(ranked_lists)

    docs = [doc for doc, _ in fused]
    scores = [score for _, score in fused]

    logger.info(
        f"Hybrid retrieval: {len(docs)} fused results in "
        f"{time.time() - start:.2f}s total "
        f"(dense={len(dense_results)}, sparse={len(sparse_results)}, title={len(title_results)})"
    )
    return docs, scores
