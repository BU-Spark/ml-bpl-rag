"""
scripts/update_abstracts_and_embeddings.py

For all metadata-only documents that have a non-empty abstract in the
JSONL source file:
  1. Updates the `abstract` column in PostgreSQL with the full text
  2. Reconstructs the metadata text string (identical to ingestion)
  3. Re-computes dense + sparse embeddings using BGE-M3
  4. Updates metadata_embedding, sparse_token_ids, sparse_weights in DB

Run:
    python scripts/update_abstracts_and_embeddings.py
    python scripts/update_abstracts_and_embeddings.py --metadata-file data/metadata/metadata.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from database.schema import get_conn, get_cursor
from embedding.embedder import embedder

DEFAULT_METADATA_FILE = "data/metadata/metadata.jsonl"
BATCH_SIZE = 64


# ── HTML cleaning (identical to _parse_metadata_record) ───────────────────────

def clean_abstract(raw: str) -> str:
    """Strip HTML exactly as _parse_metadata_record does — no truncation."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ── Build metadata text (identical to build_metadata_text in embedder.py) ─────

def build_metadata_text(record: dict) -> str:
    """
    Reconstruct the metadata text string exactly as embedder.build_metadata_text()
    does — with abstract truncation removed.
    """
    parts = []

    if record.get("title"):
        parts.append(f"Title: {record['title']}")

    genres = record.get("genre") or []
    if genres:
        parts.append(f"Format: {', '.join(genres)}")

    topics = record.get("topics") or []
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")

    geography = record.get("geography") or []
    if geography:
        parts.append(f"Geography: {', '.join(geography)}")

    place = record.get("place") or []
    if place:
        parts.append(f"Place: {', '.join(place)}")

    year = record.get("year") or []
    if year:
        parts.append(f"Year: {', '.join(str(y) for y in year)}")

    collection = record.get("collection") or ""
    if collection and collection != record.get("title"):
        parts.append(f"Collection: {collection}")

    abstract = record.get("abstract") or ""
    if abstract:
        # No truncation — full abstract
        parts.append(f"Description: {abstract}")

    return " | ".join(parts)


# ── Update DB ─────────────────────────────────────────────────────────────────

def update_batch(batch: List[dict]):
    """
    For a batch of records with full abstracts:
    1. Compute new metadata text
    2. Embed dense + sparse in one forward pass
    3. Update abstract, metadata_embedding, sparse_token_ids, sparse_weights
    """
    meta_texts = [build_metadata_text(r) for r in batch]

    output     = embedder.encode_both(meta_texts)
    dense_embs = output["dense"]
    sparse_embs = output["sparse"]

    with get_conn() as conn:
        for rec, dense_emb, sparse in zip(batch, dense_embs, sparse_embs):
            token_ids = [int(k) for k in sparse.keys()]
            weights   = [float(v) for v in sparse.values()]

            with get_cursor(conn) as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET
                        abstract           = %s,
                        metadata_embedding = %s,
                        sparse_token_ids   = %s,
                        sparse_weights     = %s,
                        ingested_at        = %s
                    WHERE ark_id = %s
                    """,
                    (
                        rec["abstract"],
                        dense_emb.tolist(),
                        token_ids,
                        weights,
                        datetime.now(timezone.utc).isoformat(),
                        rec["ark_id"],
                    )
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(metadata_file: str = DEFAULT_METADATA_FILE):
    fpath = Path(metadata_file)
    if not fpath.exists():
        print(f"File not found: {metadata_file}")
        return

    print(f"\n{'='*60}")
    print("Update Abstracts + Embeddings for Metadata Records")
    print(f"  Source : {metadata_file}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"{'='*60}\n")

    print("Reading metadata JSONL...")
    records_to_update = []

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            data      = raw.get("data", {})
            attrs     = data.get("attributes", {})
            record_id = data.get("id", "")
            ark_id    = record_id.split(":")[-1] if ":" in record_id else record_id

            abstract_raw = attrs.get("abstract_tsi", "") or ""
            if not abstract_raw.strip():
                continue

            abstract = clean_abstract(abstract_raw)
            if not abstract:
                continue

            records_to_update.append({
                "ark_id":      ark_id,
                "abstract":    abstract,
                "genre":       attrs.get("genre_basic_ssim", []),
                "topics":      attrs.get("subject_topic_tsim", []),
                "geography":   attrs.get("subject_geographic_ssim", []),
                "place":       attrs.get("publication_place_tsim", []),
                "year":        attrs.get("date_facet_yearly_itim", []),
                "title":       attrs.get("title_info_primary_tsi", ""),
                "collection":  attrs.get("title_info_primary_tsi", ""),
                "institution": attrs.get("institution_name_ssi", ""),
            })

    print(f"  Found {len(records_to_update)} records with non-empty abstracts\n")

    if not records_to_update:
        print("Nothing to update.")
        return

    total_updated = 0
    start_time    = time.monotonic()

    for i in range(0, len(records_to_update), BATCH_SIZE):
        batch = records_to_update[i:i + BATCH_SIZE]
        update_batch(batch)
        total_updated += len(batch)

        elapsed   = time.monotonic() - start_time
        remaining = (elapsed / total_updated) * (len(records_to_update) - total_updated) if total_updated else 0
        print(
            f"  [{total_updated}/{len(records_to_update)}] "
            f"ETA: {remaining/60:.1f}min"
        )

    print(f"\n✓ Done.")
    print(f"  Records updated : {total_updated}")
    print(f"  Total time      : {(time.monotonic()-start_time)/60:.1f} min")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    args = parser.parse_args()
    run(metadata_file=args.metadata_file)