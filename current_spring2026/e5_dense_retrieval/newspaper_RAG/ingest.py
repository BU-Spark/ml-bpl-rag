#!/usr/bin/env python3
"""
Ingestion module for historical newspaper RAG pipeline.

Reads a newspaper JSON file (e.g. 1940.json) and builds two FAISS indexes:

  1. Title/Summary Index  — one embedding per newspaper issue using
     concatenated metadata (newspaper name, date, topics, geography).
     Used for the first-stage threshold gate.

  2. Full Article Index   — chunked OCR text embeddings, one or more
     embeddings per issue. Used for second-stage dense retrieval.

Both indexes use intfloat/e5-base-v2 with normalised vectors so that
inner-product search is equivalent to cosine similarity.

E5 conventions:
  - Passages are prefixed with "passage: "
  - Queries are prefixed with "query: "  (handled in retrieval.py)
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── E5 prefix ────────────────────────────────────────────────────────────────
PASSAGE_PREFIX = "passage: "

# ── Default chunking parameters ───────────────────────────────────────────────
DEFAULT_CHUNK_WORDS = 400   # words per chunk (E5 max ≈ 512 tokens ≈ 380-420 words)
DEFAULT_OVERLAP_WORDS = 50  # word overlap between consecutive chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_title_summary_text(record: dict) -> str:
    """Concatenate metadata fields into a searchable string for the title index."""
    parts = [
        record.get("newspaper", ""),
        record.get("issue_date", ""),
        " ".join(record.get("topics", [])),
        " ".join(record.get("geography", [])),
    ]
    return " | ".join(p for p in parts if p.strip())


def _chunk_text(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_json(
    json_path: str,
    model: SentenceTransformer,
    output_dir: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    batch_size: int = 32,
) -> None:
    """
    Ingest a newspaper JSON file and write two FAISS indexes to output_dir.

    Output files written:
      title_summary.faiss   — FAISS FlatIP index (title/summary embeddings)
      title_meta.pkl        — list[dict] metadata aligned with title index rows
      full_article.faiss    — FAISS FlatIP index (chunk embeddings)
      chunk_meta.pkl        — list[dict] metadata aligned with chunk index rows
      doc_chunks_map.pkl    — dict mapping doc_id → list[chunk_meta dicts]

    Args:
        json_path:   Path to the newspaper JSON file.
        model:       Loaded SentenceTransformer (E5) model.
        output_dir:  Directory to save indexes and metadata pickles.
        chunk_words: Words per chunk for the full-article index.
        overlap_words: Overlap words between consecutive chunks.
        batch_size:  Encoding batch size.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    logger.info(f"Loaded {len(records)} records from {json_path}")

    # ── Build Title/Summary Index ─────────────────────────────────────────────
    logger.info("Building Title/Summary index...")
    title_texts: List[str] = []
    title_meta: List[dict] = []

    for rec in records:
        if not rec.get("clean_text", "").strip():
            continue  # skip issues with no OCR text
        ts_text = _build_title_summary_text(rec)
        title_texts.append(PASSAGE_PREFIX + ts_text)
        title_meta.append({
            "doc_id":     rec["ark_id"],
            "record_id":  rec["record_id"],
            "source_url": rec.get("source_url", ""),
            "newspaper":  rec.get("newspaper", ""),
            "issue_date": rec.get("issue_date", ""),
            "date_iso":   rec.get("date_iso", []),
            "title":      rec.get("title", ""),
            "topics":     rec.get("topics", []),
            "geography":  rec.get("geography", []),
            "page_count": rec.get("page_count", 0),
        })

    logger.info(f"Encoding {len(title_texts)} title/summary texts...")
    title_embs = model.encode(
        title_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = title_embs.shape[1]
    title_index = faiss.IndexFlatIP(dim)
    title_index.add(title_embs)

    faiss.write_index(title_index, str(output_path / "title_summary.faiss"))
    with open(output_path / "title_meta.pkl", "wb") as f:
        pickle.dump(title_meta, f)
    logger.info(f"Title/Summary index saved — {title_index.ntotal} entries, dim={dim}")

    # ── Build Full Article Index ──────────────────────────────────────────────
    logger.info("Building Full Article index...")
    chunk_texts: List[str] = []
    chunk_meta: List[dict] = []
    doc_chunks_map: dict = {}   # doc_id → [chunk_meta, ...]

    for rec in records:
        text = rec.get("clean_text", "").strip()
        if not text:
            continue
        doc_id = rec["ark_id"]
        base_meta = {
            "doc_id":     doc_id,
            "record_id":  rec["record_id"],
            "source_url": rec.get("source_url", ""),
            "newspaper":  rec.get("newspaper", ""),
            "issue_date": rec.get("issue_date", ""),
            "date_iso":   rec.get("date_iso", []),
            "topics":     rec.get("topics", []),
            "geography":  rec.get("geography", []),
            # Store full text here for later reconstruction (capped at 12k chars)
            "full_text":  text[:12000],
        }

        chunks = _chunk_text(text, chunk_words, overlap_words)
        doc_chunks_map[doc_id] = []

        for i, chunk in enumerate(chunks):
            chunk_entry = {**base_meta, "chunk_idx": i, "chunk_text": chunk}
            chunk_texts.append(PASSAGE_PREFIX + chunk)
            chunk_meta.append(chunk_entry)
            doc_chunks_map[doc_id].append(chunk_entry)

    logger.info(f"Encoding {len(chunk_texts)} article chunks...")
    chunk_embs = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    article_index = faiss.IndexFlatIP(dim)
    article_index.add(chunk_embs)

    faiss.write_index(article_index, str(output_path / "full_article.faiss"))
    with open(output_path / "chunk_meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)
    with open(output_path / "doc_chunks_map.pkl", "wb") as f:
        pickle.dump(doc_chunks_map, f)

    logger.info(
        f"Full Article index saved — {article_index.ntotal} chunks "
        f"from {len(doc_chunks_map)} issues, dim={dim}"
    )
    logger.info(f"All indexes written to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Index loading helper (used by pipeline / app)
# ─────────────────────────────────────────────────────────────────────────────

def load_indexes(index_dir: str) -> Tuple:
    """
    Load FAISS indexes and associated metadata pickles from disk.

    Returns:
        (title_index, full_index, title_meta, chunk_meta, doc_chunks_map)
    """
    p = Path(index_dir)
    title_index = faiss.read_index(str(p / "title_summary.faiss"))
    full_index  = faiss.read_index(str(p / "full_article.faiss"))
    with open(p / "title_meta.pkl",    "rb") as f:
        title_meta = pickle.load(f)
    with open(p / "chunk_meta.pkl",    "rb") as f:
        chunk_meta = pickle.load(f)
    with open(p / "doc_chunks_map.pkl","rb") as f:
        doc_chunks_map = pickle.load(f)

    logger.info(
        f"Loaded indexes from {index_dir}: "
        f"{title_index.ntotal} title entries, {full_index.ntotal} article chunks"
    )
    return title_index, full_index, title_meta, chunk_meta, doc_chunks_map
