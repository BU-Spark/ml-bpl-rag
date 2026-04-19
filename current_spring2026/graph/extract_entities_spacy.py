"""
graph/extract_entities.py

Phase 1: Extract named entities from all documents using spaCy.
Runs on CPU only — no GPU needed.
Uses nlp.pipe() with multiple processes for parallelism.

Output: data/graph/entities_<year>.jsonl
        (or entities_all.jsonl if no year specified)

Run:
    python -m graph.extract_entities --year 1900
    python -m graph.extract_entities --all --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import spacy

from database.schema import get_conn, get_cursor
from graph.entity_extractor import RELEVANT_TYPES, TYPE_MAP

OUTPUT_DIR = Path("data/graph")


# ── Fetch documents ───────────────────────────────────────────────────────────

def fetch_documents(year: int = None) -> List[dict]:
    sql = """
        SELECT
            d.id,
            d.ark_id,
            d.title,
            d.year,
            d.institution,
            d.source_url,
            d.issue_date,
            ARRAY_AGG(c.chunk_text ORDER BY c.chunk_index) AS chunks
        FROM documents d
        JOIN chunks c ON c.document_id = d.id
    """
    params = []
    if year:
        sql += " WHERE EXTRACT(YEAR FROM d.date_start) = %s"
        params.append(year)

    sql += " GROUP BY d.id, d.ark_id, d.title, d.year, d.institution, d.source_url, d.issue_date"

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities_from_doc(doc_nlp, ark_id: str) -> List[dict]:
    """Extract entities from a spaCy doc object."""
    entity_counts = {}
    entity_raw    = {}

    for ent in doc_nlp.ents:
        if ent.label_ not in RELEVANT_TYPES:
            continue
        normalized = ent.text.strip().lower()
        if len(normalized) < 2 or len(normalized) > 100:
            continue
        if ent.label_ != "DATE" and normalized.replace(" ", "").isdigit():
            continue

        mapped_type = TYPE_MAP.get(ent.label_, ent.label_)
        key = (normalized, mapped_type)
        entity_counts[key] = entity_counts.get(key, 0) + 1
        entity_raw[key]    = ent.label_

    # Return top 40 by frequency
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:40]

    return [
        {
            "text":     text,
            "type":     type_,
            "raw_type": entity_raw[(text, type_)],
            "count":    count,
        }
        for (text, type_), count in sorted_entities
    ]


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_all(year: int = None, n_workers: int = 8):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix      = str(year) if year else "all"
    output_file = OUTPUT_DIR / f"entities_{suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"BPL Graph — Phase 1: Entity Extraction")
    print(f"  Year    : {year or 'all'}")
    print(f"  Workers : {n_workers}")
    print(f"  Output  : {output_file}")
    print(f"{'='*60}\n")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    # Disable unused pipeline components for speed
    nlp.select_pipes(enable=["ner"])

    print("Fetching documents from PostgreSQL...")
    docs = fetch_documents(year=year)
    print(f"  Found {len(docs)} documents\n")

    # Build full texts
    ark_ids    = [doc["ark_id"] for doc in docs]
    doc_metas  = {doc["ark_id"]: doc for doc in docs}
    full_texts = [" ".join(doc["chunks"] or [])[:500_000] for doc in docs]

    start_time    = time.monotonic()
    total_written = 0

    with open(output_file, "w", encoding="utf-8") as f:
        # nlp.pipe processes in parallel using n_workers processes
        for i, (ark_id, doc_nlp) in enumerate(
            zip(ark_ids, nlp.pipe(full_texts, batch_size=16, n_process=n_workers))
        ):
            entities = extract_entities_from_doc(doc_nlp, ark_id)

            if not entities:
                continue

            meta = doc_metas[ark_id]
            record = {
                "ark_id":      ark_id,
                "title":       meta["title"] or "",
                "year":        meta["year"],
                "institution": meta["institution"] or "",
                "source_url":  meta["source_url"] or "",
                "issue_date":  meta["issue_date"] or "",
                "entities":    entities,
            }
            f.write(json.dumps(record) + "\n")
            total_written += 1

            if (i + 1) % 100 == 0 or i == 0:
                elapsed   = time.monotonic() - start_time
                remaining = (elapsed / (i + 1)) * (len(docs) - i - 1)
                print(
                    f"  [{i+1}/{len(docs)}] {ark_id} | "
                    f"{len(entities)} entities | "
                    f"ETA: {remaining/60:.1f}min"
                )

    elapsed = time.monotonic() - start_time
    print(f"\n✓ Entity extraction complete.")
    print(f"  Documents processed : {total_written}")
    print(f"  Output file         : {output_file}")
    print(f"  Total time          : {elapsed/60:.1f} min")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Extract entities")
    parser.add_argument("--year",    type=int, default=None)
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    extract_all(
        year      = None if args.all else (args.year or 1900),
        n_workers = args.workers,
    )
