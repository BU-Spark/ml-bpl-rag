"""
graph/extract_entities_spacy.py

Phase 1: Extract named entities from all documents using spaCy.
Runs on CPU only — no GPU needed.
Uses nlp.pipe() with multiple processes for parallelism.
Supports both full-text and metadata-only modes.

Output:
  data/graph/entities_<year>.jsonl       (full-text mode)
  data/graph/entities_all.jsonl          (full-text mode)
  data/graph/entities_metadata.jsonl     (metadata mode)

Run:
    python -m graph.extract_entities_spacy --year 1900
    python -m graph.extract_entities_spacy --all --workers 8
    python -m graph.extract_entities_spacy --metadata-only --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import spacy

from database.schema import get_conn, get_cursor
from graph.entity_extractor import RELEVANT_TYPES, TYPE_MAP

OUTPUT_DIR = Path("data/graph")


# ── Fetch functions ───────────────────────────────────────────────────────────

def fetch_fulltext_documents(year: int = None) -> List[dict]:
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


def fetch_metadata_documents() -> List[dict]:
    """Fetch metadata-only collection records (no chunks)."""
    sql = """
        SELECT
            ark_id,
            title,
            abstract,
            topics,
            geography,
            genre,
            year,
            institution,
            source_url,
            issue_date
        FROM documents
        WHERE char_count = 0
        AND (
            (title    IS NOT NULL AND title    != '') OR
            (abstract IS NOT NULL AND abstract != '') OR
            array_length(topics,   1) > 0             OR
            array_length(geography,1) > 0
        )
    """
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql)
            return cur.fetchall()


# ── Text builders ─────────────────────────────────────────────────────────────

def build_fulltext_input(doc: dict) -> str:
    """Full document text up to 500K chars."""
    return " ".join(doc["chunks"] or [])[:500_000]


def build_metadata_input(doc: dict) -> str:
    """Combine title + abstract + topics + geography + genre."""
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    topics = doc.get("topics") or []
    if topics:
        parts.append(", ".join(topics))
    geography = doc.get("geography") or []
    if geography:
        parts.append(", ".join(geography))
    genre = doc.get("genre") or []
    if genre:
        parts.append(", ".join(genre))
    return " | ".join(parts)


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities_from_doc(doc_nlp, top_n: int = 40) -> List[dict]:
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

    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

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

def extract_all(year: int = None, metadata_only: bool = False, n_workers: int = 8):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if metadata_only:
        suffix   = "metadata"
        docs     = fetch_metadata_documents()
        get_text = build_metadata_input
        print("Fetching metadata records from PostgreSQL...")
    else:
        suffix   = str(year) if year else "all"
        docs     = fetch_fulltext_documents(year=year)
        get_text = build_fulltext_input
        print("Fetching full-text documents from PostgreSQL...")

    output_file = OUTPUT_DIR / f"entities_{suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"BPL Graph — Phase 1: Entity Extraction (spaCy)")
    year_str = str(year) if year else "all"
    mode_str = "metadata-only" if metadata_only else f"full-text (year={year_str})"
    print(f"  Mode    : {mode_str}")
    print(f"  Workers : {n_workers}")
    print(f"  Output  : {output_file}")
    print(f"{'='*60}\n")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable=["ner"])

    print(f"  Found {len(docs)} documents\n")

    ark_ids   = [doc["ark_id"] for doc in docs]
    doc_metas = {doc["ark_id"]: doc for doc in docs}
    texts     = [get_text(doc) for doc in docs]

    start_time    = time.monotonic()
    total_written = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for i, (ark_id, doc_nlp) in enumerate(
            zip(ark_ids, nlp.pipe(texts, batch_size=16, n_process=n_workers))
        ):
            entities = extract_entities_from_doc(doc_nlp)

            if not entities:
                continue

            meta = doc_metas[ark_id]
            record = {
                "ark_id":      ark_id,
                "title":       meta.get("title") or "",
                "year":        meta.get("year"),
                "institution": meta.get("institution") or "",
                "source_url":  meta.get("source_url") or "",
                "issue_date":  meta.get("issue_date") or "",
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
    parser = argparse.ArgumentParser(description="Phase 1: Extract entities (spaCy)")
    parser.add_argument("--year",          type=int, default=None)
    parser.add_argument("--all",           action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers",       type=int, default=8)
    args = parser.parse_args()

    extract_all(
        year          = None if (args.all or args.metadata_only) else (args.year or 1900),
        metadata_only = args.metadata_only,
        n_workers     = args.workers,
    )