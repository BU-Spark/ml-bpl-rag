#!/usr/bin/env python3
"""
FAISS indexing pipeline for the BGE-M3 hybrid RAG system.

Reads newspaper records from a JSON file, encodes them with BGE-M3
(dense + sparse), and saves FAISS indexes, sparse vectors, and metadata
to indexes/{year}/.

Resumability
------------
Embedding progress is checkpointed after every 50 batches to:
    indexes/{year}/.checkpoint.pkl
Re-running the same command after interruption resumes from the last batch.

Output structure
----------------
indexes/{year}/
    full_article.faiss   — FAISS index for chunk-level dense vectors (IVF or Flat)
    title_summary.faiss  — FAISS index for title-level dense vectors
    chunk_sparse.pkl     — list of sparse dicts per chunk (token_id_str → weight)
    chunk_meta.pkl       — list of chunk metadata dicts
    title_meta.pkl       — list of title-level metadata dicts
    doc_chunks_map.pkl   — dict mapping doc_id → list of chunk dicts

Usage:
    python -m BGE_hybrid_retrieval.indexer --year 1940 --data path/to/data.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from embedder import BGEM3Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBED_BATCH = 12
EMBED_DIM = 1024
# If total vectors > this threshold, use IVF index for faster search
IVF_THRESHOLD = 50_000

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    try:
        return _splitter.split_text(text)
    except Exception:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start: start + CHUNK_SIZE])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def _build_faiss_index(matrix: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from an L2-normalised matrix.
    Uses IVF for large datasets (faster search), Flat for small ones (exact).
    """
    n, dim = matrix.shape
    if n > IVF_THRESHOLD:
        # IVF with ~sqrt(n) centroids, trained on the data
        nlist = max(16, min(int(np.sqrt(n)), 256))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.add(matrix)
        index.nprobe = max(4, nlist // 4)  # search ~25% of clusters
        logger.info(f"Built IVF index: {n} vectors, {nlist} clusters, nprobe={index.nprobe}")
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        logger.info(f"Built Flat index: {n} vectors")
    return index


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.rename(path)


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Corrupt checkpoint {path}, starting fresh: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes for BGE-M3 RAG.")
    parser.add_argument("--year", required=True, help="Year label (e.g. 1940)")
    parser.add_argument("--data", required=True, help="Path to year JSON file")
    args = parser.parse_args()

    index_dir = Path(__file__).parent / "indexes" / args.year
    index_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = index_dir / ".checkpoint.pkl"
    logger.info(f"Index output directory: {index_dir}")

    # Skip if already fully indexed
    if (index_dir / "full_article.faiss").exists() and not checkpoint_path.exists():
        logger.info(f"Year {args.year} already indexed. Skipping.")
        return

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info(f"Loading data from {args.data}...")
    with open(args.data) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "records" in raw:
        records = raw["records"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected JSON structure in {args.data}")
    logger.info(f"Loaded {len(records)} records.")

    embedder = BGEM3Embedder(use_fp16=True, batch_size=EMBED_BATCH)

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_meta: list = []
    title_meta: list = []
    doc_chunks_map: dict = {}

    TEXT_FIELD = "clean_text"
    ID_FIELD = "ark_id"

    for record in tqdm(records, desc="Chunking documents"):
        doc_id = record.get(ID_FIELD, record.get("doc_id", ""))
        full_text = record.get(TEXT_FIELD, record.get("full_text", ""))
        chunks = chunk_text(full_text)
        if not chunks:
            continue

        doc_chunk_list = []
        for i, chunk in enumerate(chunks):
            entry = {k: v for k, v in record.items() if k not in (TEXT_FIELD, "full_text")}
            entry["doc_id"] = doc_id
            entry["chunk_idx"] = i
            entry["chunk_text"] = chunk
            chunk_meta.append(entry)
            doc_chunk_list.append(entry)
        doc_chunks_map[doc_id] = doc_chunk_list

        title_entry = {k: v for k, v in record.items() if k not in (TEXT_FIELD, "full_text")}
        title_entry["doc_id"] = doc_id
        title_meta.append(title_entry)

    logger.info(f"Total chunks: {len(chunk_meta)}, total titles: {len(title_meta)}")

    # ── Resume from checkpoint ────────────────────────────────────────────
    ckpt = _load_checkpoint(checkpoint_path)
    if ckpt is not None:
        chunk_start = ckpt.get("chunk_batch_done", 0)
        chunk_dense_vecs = ckpt.get("chunk_dense_vecs", [])
        chunk_sparse_vecs: List[Dict[str, float]] = ckpt.get("chunk_sparse_vecs", [])
        title_start = ckpt.get("title_batch_done", 0)
        title_dense_vecs = ckpt.get("title_dense_vecs", [])
        logger.info(
            f"Resuming: {chunk_start} chunk batches, {title_start} title batches done"
        )
    else:
        chunk_start = 0
        chunk_dense_vecs: List[list] = []
        chunk_sparse_vecs: List[Dict[str, float]] = []
        title_start = 0
        title_dense_vecs: List[list] = []

    # ── Embed chunks (dense + sparse) ─────────────────────────────────────
    total_chunk_batches = (len(chunk_meta) + EMBED_BATCH - 1) // EMBED_BATCH
    if chunk_start < total_chunk_batches:
        logger.info(f"Embedding chunks (batch {chunk_start+1}/{total_chunk_batches})...")
        for batch_idx in tqdm(
            range(chunk_start, total_chunk_batches),
            initial=chunk_start, total=total_chunk_batches,
            desc="Chunk embeddings",
        ):
            i = batch_idx * EMBED_BATCH
            batch_texts = [m["chunk_text"] for m in chunk_meta[i: i + EMBED_BATCH]]
            dense, sparse = embedder.embed_passages(batch_texts)
            chunk_dense_vecs.extend(dense)
            chunk_sparse_vecs.extend(sparse)

            if (batch_idx + 1) % 50 == 0 or batch_idx == total_chunk_batches - 1:
                _save_checkpoint(checkpoint_path, {
                    "chunk_batch_done": batch_idx + 1,
                    "chunk_dense_vecs": chunk_dense_vecs,
                    "chunk_sparse_vecs": chunk_sparse_vecs,
                    "title_batch_done": title_start,
                    "title_dense_vecs": title_dense_vecs,
                })

    # ── Embed titles (dense only — titles are short, sparse adds little) ──
    total_title_batches = (len(title_meta) + EMBED_BATCH - 1) // EMBED_BATCH
    if title_start < total_title_batches:
        logger.info(f"Embedding titles (batch {title_start+1}/{total_title_batches})...")
        for batch_idx in tqdm(
            range(title_start, total_title_batches),
            initial=title_start, total=total_title_batches,
            desc="Title embeddings",
        ):
            i = batch_idx * EMBED_BATCH
            batch_texts = [
                f"{m.get('newspaper', '')} {m.get('issue_date', '')} {m.get('title', '')}".strip()
                for m in title_meta[i: i + EMBED_BATCH]
            ]
            dense, _ = embedder.embed_passages(batch_texts)
            title_dense_vecs.extend(dense)

            if (batch_idx + 1) % 50 == 0 or batch_idx == total_title_batches - 1:
                _save_checkpoint(checkpoint_path, {
                    "chunk_batch_done": total_chunk_batches,
                    "chunk_dense_vecs": chunk_dense_vecs,
                    "chunk_sparse_vecs": chunk_sparse_vecs,
                    "title_batch_done": batch_idx + 1,
                    "title_dense_vecs": title_dense_vecs,
                })

    # ── Build FAISS indexes ───────────────────────────────────────────────
    chunk_matrix = np.array(chunk_dense_vecs, dtype=np.float32)
    faiss.normalize_L2(chunk_matrix)
    chunk_index = _build_faiss_index(chunk_matrix)

    title_matrix = np.array(title_dense_vecs, dtype=np.float32)
    faiss.normalize_L2(title_matrix)
    title_index = _build_faiss_index(title_matrix)

    # ── Save everything ───────────────────────────────────────────────────
    faiss.write_index(chunk_index, str(index_dir / "full_article.faiss"))
    faiss.write_index(title_index, str(index_dir / "title_summary.faiss"))
    with open(index_dir / "chunk_sparse.pkl", "wb") as f:
        pickle.dump(chunk_sparse_vecs, f)
    with open(index_dir / "chunk_meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)
    with open(index_dir / "title_meta.pkl", "wb") as f:
        pickle.dump(title_meta, f)
    with open(index_dir / "doc_chunks_map.pkl", "wb") as f:
        pickle.dump(doc_chunks_map, f)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info(
        f"Done. {chunk_index.ntotal} chunk vectors (dense+sparse), "
        f"{title_index.ntotal} title vectors → {index_dir}"
    )


if __name__ == "__main__":
    main()
