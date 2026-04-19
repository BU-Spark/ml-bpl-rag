"""
ingestion/ingest.py

Ingestion pipeline for two data sources:

Source A — Full-text records (PRIMARY)
  Layout: data/fulltext/<collection_name>/<year>.json
  Structure: { "year": 1946, "records": [ {record}, ... ] }
  Fields: flat, includes clean_text (or raw_text) and all metadata
  These get both metadata + chunk embeddings (dense + sparse).

Source B — Metadata-only records (SECONDARY)
  Layout: data/metadata/metadata.jsonl
  Structure: { "data": { "id": "commonwealth:...", "attributes": { ... } } }
  These get only metadata embeddings (dense + sparse). No chunks.

Run:
    python -m ingestion.ingest
    python -m ingestion.ingest --fulltext-dir data/fulltext --skip-metadata
    python -m ingestion.ingest --metadata-file data/metadata/metadata.jsonl --skip-fulltext
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

import numpy as np
from psycopg2.extras import execute_values

from config import MIN_CHAR_COUNT, BGE_BATCH_SIZE
from database.schema import get_conn, get_cursor
from embedding.embedder import embedder
from ingestion.chunker import chunker
import re
import html

DEFAULT_FULLTEXT_DIR  = "data/fulltext"
DEFAULT_METADATA_FILE = "data/metadata/metadata.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def sparse_to_arrays(sparse: dict) -> tuple[list, list]:
    """Convert {token_id: weight} dict to (token_ids[], weights[]) arrays."""
    token_ids = [int(k) for k in sparse.keys()]
    weights   = [float(v) for v in sparse.values()]
    return token_ids, weights


def is_valid(record: dict) -> bool:
    return bool(record.get("ark_id"))


def has_fulltext(record: dict) -> bool:
    text = record.get("clean_text") or record.get("raw_text") or ""
    return (
        not record.get("_metadata_only", False)
        and len(text) >= MIN_CHAR_COUNT
    )


def parse_date_start(record: dict):
    raw = record.get("date_start", "") or ""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# ── Source A: Full-text records ───────────────────────────────────────────────

def iter_fulltext_records(fulltext_dir: str) -> Iterator[dict]:
    root = Path(fulltext_dir)
    if not root.exists():
        print(f"  [ingest] Full-text dir not found: {fulltext_dir} — skipping.")
        return

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        print(f"  [ingest] No .json files found under {fulltext_dir}")
        return

    for fpath in json_files:
        collection = fpath.parent.name
        print(f"  Reading {fpath.relative_to(root)} ...")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.get("records", []):
            if not rec.get("collection"):
                rec["collection"] = collection
            yield rec


# ── Source B: Metadata-only records ──────────────────────────────────────────

def _parse_metadata_record(raw: dict) -> dict:
    data      = raw.get("data", {})
    attrs     = data.get("attributes", {})
    record_id = data.get("id", attrs.get("id", ""))
    ark_id    = record_id.split(":")[-1] if ":" in record_id else record_id
    abstract_raw = attrs.get("abstract_tsi", "") or ""
    abstract_unescaped = html.unescape(abstract_raw)
    abstract_clean = re.sub(r"<[^>]+>", " ", abstract_unescaped).strip()
    abstract_clean = re.sub(r"\s+", " ", abstract_clean)
    

    return {
        "ark_id":         ark_id,
        "record_id":      record_id,
        "source_url":     attrs.get("identifier_uri_ss", ""),
        "iiif_manifest":  attrs.get("identifier_iiif_manifest_ss", ""),
        "newspaper":      "",
        "collection":     attrs.get("title_info_primary_tsi", ""),
        "title":          attrs.get("title_info_primary_tsi", ""),
        "issue_date":     attrs.get("title_info_partnum_tsi", ""),
        "date_iso":       attrs.get("date_edtf_ssm", []),
        "date_start":     attrs.get("date_start_dtsi", ""),
        "year":           attrs.get("date_facet_yearly_itim", []),
        "publisher":      attrs.get("publisher_tsim", []),
        "place":          attrs.get("publication_place_tsim", []),
        "language":       attrs.get("language_ssim", []),
        "institution":    attrs.get("institution_name_ssi", ""),
        "page_count":     len(attrs.get("filenames_ssim", [])),
        "pages":          attrs.get("filenames_ssim", []),
        "topics":         attrs.get("subject_topic_tsim", []),
        "geography":      attrs.get("subject_geographic_ssim", []),
        "clean_text":     "",
        "raw_text":       "",
        "char_count":     0,
        "ingested_at":    datetime.now(timezone.utc).isoformat(),
        "_metadata_only": True,
        "genre":          attrs.get("genre_basic_ssim", []),
        # Strip HTML once at parse time instead of repeatedly at embed time
        "exemplary_image_id": attrs.get("exemplary_image_ssi", ""),
        "abstract": abstract_clean,
    }


def iter_metadata_records(
    metadata_file: str,
    seen_record_ids: set,
) -> Iterator[dict]:
    fpath = Path(metadata_file)
    if not fpath.exists():
        print(f"  [ingest] Metadata file not found: {metadata_file} — skipping.")
        return

    print(f"  Reading metadata JSONL: {fpath.name} ...")
    skipped = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = _parse_metadata_record(raw)
            if rec["record_id"] in seen_record_ids:
                skipped += 1
                continue
            yield rec

    if skipped:
        print(f"  Skipped {skipped} metadata records already covered by full-text source.")


# ── DB writes ─────────────────────────────────────────────────────────────────

def upsert_documents(
    records: List[dict],
    meta_embeddings: np.ndarray,
    meta_sparse: List[dict],
    conn,
) -> dict:
    """Upsert into documents table. Returns {ark_id: db_id}."""
    rows = []
    for rec, emb, sparse in zip(records, meta_embeddings, meta_sparse):
        token_ids, weights = sparse_to_arrays(sparse)
        rows.append((
            rec["ark_id"],
            rec.get("record_id", ""),
            rec.get("source_url", ""),
            rec.get("iiif_manifest", ""),
            rec.get("newspaper", ""),
            rec.get("collection", ""),
            rec.get("institution", ""),
            rec.get("title", ""),
            rec.get("issue_date", ""),
            rec.get("date_iso") or [],
            parse_date_start(rec),
            rec.get("year") or [],
            rec.get("publisher") or [],
            rec.get("place") or [],
            rec.get("language") or [],
            rec.get("page_count", 0),
            rec.get("pages") or [],
            rec.get("topics") or [],
            rec.get("geography") or [],
            rec.get("char_count", 0),
            rec.get("genre") or [],
            rec.get("abstract", ""),
            rec.get("exemplary_image_id", ""),
            emb.tolist(),
            token_ids,
            weights,
            rec.get("ingested_at", datetime.now(timezone.utc).isoformat()),
        ))

    sql = """
        INSERT INTO documents (
            ark_id, record_id, source_url, iiif_manifest,
            newspaper, collection, institution,
            title, issue_date, date_iso, date_start, year,
            publisher, place, language,
            page_count, pages, topics, geography,
            char_count,
            genre, abstract, exemplary_image_id,
            metadata_embedding, sparse_token_ids, sparse_weights,
            ingested_at
        )
        VALUES %s
        ON CONFLICT (ark_id) DO UPDATE SET
            metadata_embedding = EXCLUDED.metadata_embedding,
            sparse_token_ids   = EXCLUDED.sparse_token_ids,
            sparse_weights     = EXCLUDED.sparse_weights,
            char_count         = EXCLUDED.char_count,
            genre               = EXCLUDED.genre,
            abstract            = EXCLUDED.abstract,
            exemplary_image_id  = EXCLUDED.exemplary_image_id,
            ingested_at        = EXCLUDED.ingested_at
        RETURNING ark_id, id
    """
    with get_cursor(conn) as cur:
        results = execute_values(cur, sql, rows, fetch=True)
    return {row["ark_id"]: row["id"] for row in results}


def insert_chunks(
    chunk_rows: List[dict],
    text_embeddings: np.ndarray,
    sparse_embeddings: List[dict],
    ark_to_doc_id: dict,
    conn,
):
    if not chunk_rows:
        return

    doc_ids = list({
        ark_to_doc_id[c["ark_id"]]
        for c in chunk_rows
        if c["ark_id"] in ark_to_doc_id
    })

    with get_cursor(conn) as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (doc_ids,))

    rows = []
    for chunk, emb, sparse in zip(chunk_rows, text_embeddings, sparse_embeddings):
        doc_id = ark_to_doc_id.get(chunk["ark_id"])
        if doc_id is None:
            continue
        token_ids, weights = sparse_to_arrays(sparse)
        rows.append((
            doc_id,
            chunk["ark_id"],
            chunk["chunk_index"],
            chunk["chunk_text"],
            emb.tolist(),
            token_ids,
            weights,
        ))

    with get_cursor(conn) as cur:
        execute_values(
            cur,
            """
            INSERT INTO chunks (
                document_id, ark_id, chunk_index, chunk_text,
                text_embedding, sparse_token_ids, sparse_weights
            )
            VALUES %s
            """,
            rows,
        )


# ── Batch flush ───────────────────────────────────────────────────────────────

def flush_batch(batch: List[dict]) -> tuple[int, int]:
    """Embed and write one batch. Returns (n_docs, n_chunks)."""

    # Single forward pass for metadata — dense + sparse together
    meta_texts  = [embedder.build_metadata_text(r) for r in batch]
    meta_output = embedder.encode_both(meta_texts)
    meta_embs   = meta_output["dense"]
    meta_sparse = meta_output["sparse"]

    # Chunk full-text records
    all_chunks: List[dict] = []
    for rec in batch:
        if has_fulltext(rec):
            all_chunks.extend(chunker.chunk_record(rec))

    # Single forward pass for chunks — dense + sparse together
    if all_chunks:
        chunk_texts  = [c["chunk_text"] for c in all_chunks]
        chunk_output = embedder.encode_both(chunk_texts)
        chunk_embs   = chunk_output["dense"]
        chunk_sparse = chunk_output["sparse"]
    else:
        chunk_embs   = np.array([])
        chunk_sparse = []

    with get_conn() as conn:
        ark_to_doc_id = upsert_documents(batch, meta_embs, meta_sparse, conn)
        if all_chunks:
            insert_chunks(all_chunks, chunk_embs, chunk_sparse, ark_to_doc_id, conn)

    return len(batch), len(all_chunks)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ingestion(
    fulltext_dir:  str  = DEFAULT_FULLTEXT_DIR,
    metadata_file: str  = DEFAULT_METADATA_FILE,
    batch_size:    int  = BGE_BATCH_SIZE,
    skip_fulltext: bool = False,
    skip_metadata: bool = False,
):
    print(f"\n{'='*60}")
    print("BPL RAG Ingestion Pipeline")
    print(f"  Full-text dir  : {fulltext_dir}")
    print(f"  Metadata file  : {metadata_file}")
    print(f"  Batch size     : {batch_size}")
    print(f"{'='*60}\n")

    total_docs = total_chunks = skipped = 0
    batch: List[dict] = []
    seen_record_ids: set[str] = set()

    def flush(b):
        nonlocal total_docs, total_chunks
        n_docs, n_chunks = flush_batch(b)
        total_docs   += n_docs
        total_chunks += n_chunks
        print(
            f"  Flushed {n_docs} docs | {n_chunks} chunks | "
            f"totals → {total_docs} docs / {total_chunks} chunks"
        )

    if not skip_fulltext:
        print("── Pass 1: Full-text records ──────────────────────────────")
        for record in iter_fulltext_records(fulltext_dir):
            if not is_valid(record):
                skipped += 1
                continue
            seen_record_ids.add(record.get("record_id", ""))
            batch.append(record)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
            batch = []

    if not skip_metadata:
        print("\n── Pass 2: Metadata-only records ──────────────────────────")
        for record in iter_metadata_records(metadata_file, seen_record_ids):
            if not is_valid(record):
                skipped += 1
                continue
            batch.append(record)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    print(f"\n✓ Ingestion complete.")
    print(f"  Documents ingested : {total_docs}")
    print(f"  Chunks created     : {total_chunks}")
    print(f"  Records skipped    : {skipped}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPL RAG ingestion pipeline")
    parser.add_argument("--fulltext-dir",   default=DEFAULT_FULLTEXT_DIR)
    parser.add_argument("--metadata-file",  default=DEFAULT_METADATA_FILE)
    parser.add_argument("--batch-size",     type=int, default=BGE_BATCH_SIZE)
    parser.add_argument("--skip-fulltext",  action="store_true")
    parser.add_argument("--skip-metadata",  action="store_true")
    args = parser.parse_args()

    run_ingestion(
        fulltext_dir  = args.fulltext_dir,
        metadata_file = args.metadata_file,
        batch_size    = args.batch_size,
        skip_fulltext = args.skip_fulltext,
        skip_metadata = args.skip_metadata,
    )
