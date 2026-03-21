#!/usr/bin/env python3
"""
FAISS indexing pipeline for the BGE-M3 RAG system.

Reads newspaper records from a JSON file, encodes them with BGE-M3 (dense),
and saves FAISS indexes + metadata pkl files to indexes/{year}/.

Output structure
----------------
indexes/{year}/
    full_article.faiss   — FAISS flat IP index for chunk-level dense vectors
    title_summary.faiss  — FAISS flat IP index for title-level dense vectors
    chunk_meta.pkl       — list of chunk metadata dicts
    title_meta.pkl       — list of title-level metadata dicts
    doc_chunks_map.pkl   — dict mapping doc_id → list of chunk dicts

Each record in the input JSON must contain at minimum:
    doc_id, full_text
Optional fields (preserved in metadata):
    record_id, source_url, newspaper, issue_date, date_iso, title, topics, geography

Usage:
    python -m BGE_hybrid_retrieval.indexer --year 1940 --data path/to/data.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .embedder import BGEM3Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBED_BATCH = 12
EMBED_DIM = 1024

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


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes for BGE-M3 RAG.")
    parser.add_argument(
        "--year", required=True,
        help="Year label for the index directory (e.g. 1940)"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to JSON file containing a list of document records"
    )
    args = parser.parse_args()

    index_dir = Path(__file__).parent / "indexes" / args.year
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Index output directory: {index_dir}")

    logger.info(f"Loading data from {args.data}...")
    with open(args.data) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} records.")

    embedder = BGEM3Embedder(use_fp16=True, batch_size=EMBED_BATCH)

    chunk_meta: list = []
    title_meta: list = []
    doc_chunks_map: dict = {}

    for record in tqdm(records, desc="Chunking documents"):
        doc_id = record.get("doc_id", "")
        full_text = record.get("full_text", "")
        chunks = chunk_text(full_text)
        if not chunks:
            continue

        doc_chunk_list = []
        for i, chunk in enumerate(chunks):
            entry = {k: v for k, v in record.items() if k != "full_text"}
            entry["chunk_idx"] = i
            entry["chunk_text"] = chunk
            chunk_meta.append(entry)
            doc_chunk_list.append(entry)
        doc_chunks_map[doc_id] = doc_chunk_list

        title_entry = {k: v for k, v in record.items() if k != "full_text"}
        title_meta.append(title_entry)

    logger.info(f"Total chunks: {len(chunk_meta)}, total titles: {len(title_meta)}")

    # Embed chunks
    chunk_dense_vecs: List[np.ndarray] = []
    logger.info("Embedding chunks...")
    for i in tqdm(range(0, len(chunk_meta), EMBED_BATCH), desc="Chunk embeddings"):
        batch_texts = [m["chunk_text"] for m in chunk_meta[i: i + EMBED_BATCH]]
        dense, _ = embedder.embed_passages(batch_texts)
        chunk_dense_vecs.extend(dense)

    # Embed titles
    title_dense_vecs: List[np.ndarray] = []
    logger.info("Embedding titles...")
    for i in tqdm(range(0, len(title_meta), EMBED_BATCH), desc="Title embeddings"):
        batch_texts = [
            f"{m.get('newspaper', '')} {m.get('issue_date', '')} {m.get('title', '')}".strip()
            for m in title_meta[i: i + EMBED_BATCH]
        ]
        dense, _ = embedder.embed_passages(batch_texts)
        title_dense_vecs.extend(dense)

    # Build FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)
    chunk_matrix = np.array(chunk_dense_vecs, dtype=np.float32)
    faiss.normalize_L2(chunk_matrix)
    chunk_index = faiss.IndexFlatIP(EMBED_DIM)
    chunk_index.add(chunk_matrix)

    title_matrix = np.array(title_dense_vecs, dtype=np.float32)
    faiss.normalize_L2(title_matrix)
    title_index = faiss.IndexFlatIP(EMBED_DIM)
    title_index.add(title_matrix)

    # Save indexes and metadata
    faiss.write_index(chunk_index, str(index_dir / "full_article.faiss"))
    faiss.write_index(title_index, str(index_dir / "title_summary.faiss"))
    with open(index_dir / "chunk_meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)
    with open(index_dir / "title_meta.pkl", "wb") as f:
        pickle.dump(title_meta, f)
    with open(index_dir / "doc_chunks_map.pkl", "wb") as f:
        pickle.dump(doc_chunks_map, f)

    logger.info(
        f"Done. Saved {chunk_index.ntotal} chunk vectors and "
        f"{title_index.ntotal} title vectors to {index_dir}"
    )


if __name__ == "__main__":
    main()
