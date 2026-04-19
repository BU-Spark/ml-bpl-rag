"""
graph/extract_entities_gpt.py

Phase 1: Extract named entities using GPT-4o-mini.
Supports two modes:

  --all / --year      Full-text newspaper records (joins with chunks)
  --metadata-only     Collection-level metadata records (no chunks)

Output:
  data/graph/entities_<year>.jsonl       (full-text mode)
  data/graph/entities_all.jsonl          (full-text mode)
  data/graph/entities_metadata.jsonl     (metadata mode)

Run:
    python -m graph.extract_entities_gpt --year 1900
    python -m graph.extract_entities_gpt --all --concurrency 50
    python -m graph.extract_entities_gpt --metadata-only --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI

from config import OPENAI_API_KEY
from database.schema import get_conn, get_cursor

OUTPUT_DIR = Path("data/graph")
client     = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ── System prompts ────────────────────────────────────────────────────────────

FULLTEXT_SYSTEM_PROMPT = """
You are analyzing historical Boston newspaper text from the early 1900s to 1940s.
Extract the most significant named entities from the text provided.

Return a JSON array with this exact structure:
[{"text": "entity name", "type": "PERSON|PLACE|ORG|EVENT"}]

Entity type rules:
- PERSON: real named individuals — politicians, criminals, athletes, business figures, public figures
- PLACE: Boston neighborhoods, streets, cities, states, countries, landmarks
- ORG: companies, government bodies, courts, sports teams, universities, newspapers, unions
- EVENT: named historical events, disasters, elections, strikes, wars, parades

Critical rules:
- Ignore OCR artifacts: single letters, abbreviations like "tel", "ap", "ho", "un", "co"
- Ignore generic words: "american", "french", "british" unless part of a proper name
- Ignore content from advertisements, weather reports, stock prices, and shipping notices
- Ignore store names, product names, and retail businesses unless historically significant
- Normalize names: "Pres. Roosevelt" → "roosevelt", "Gov. Fitzgerald" → "fitzgerald"
- All entity text must be lowercase
- Return at most 20 most significant entities from news articles only
- If text is too noisy or no entities found, return empty array []
- Return ONLY valid JSON array, no explanation, no markdown
""".strip()

METADATA_SYSTEM_PROMPT = """
You are analyzing descriptions of historical library collections from the Boston Public Library.
Extract the most significant named entities from the collection description provided.

Return a JSON array with this exact structure:
[{"text": "entity name", "type": "PERSON|PLACE|ORG|EVENT"}]

Entity type rules:
- PERSON: any named individual mentioned — collectors, donors, photographers, artists,
  authors, historical figures, politicians, scientists, activists, or any person
  referenced in the collection description or topics

- PLACE: any named location — cities, countries, regions, neighborhoods, streets,
  landmarks, geographic features, archaeological sites, or any place referenced
  in the collection description or geography fields

- ORG: any named organization — institutions, companies, government bodies,
  universities, libraries, museums, religious organizations, political parties,
  newspapers, clubs, societies, or any group referenced in the collection

- EVENT: any named occurrence — wars, movements, disasters, exhibitions, expeditions,
  legal cases, political campaigns, ceremonies, uprisings, strikes, or any
  significant historical event referenced in the collection description

Critical rules:
- Extract entities from title, description, topics, and geography fields
- Normalize all entity text to lowercase
- Return at most 20 most significant entities
- If no meaningful entities found, return empty array []
- Return ONLY valid JSON array, no explanation, no markdown
""".strip()


# ── Fetch functions ───────────────────────────────────────────────────────────

def fetch_fulltext_documents(year: int = None) -> List[dict]:
    """Fetch full-text records by joining with chunks."""
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
    """First 3000 chars of concatenated chunks."""
    return " ".join(doc["chunks"] or [])[:3000]


def build_metadata_input(doc: dict) -> str:
    """
    Combine title + abstract + topics + geography + genre
    into a single text for entity extraction.
    """
    parts = []

    if doc.get("title"):
        parts.append(f"Title: {doc['title']}")

    if doc.get("abstract"):
        parts.append(f"Description: {doc['abstract']}")

    topics = doc.get("topics") or []
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")

    geography = doc.get("geography") or []
    if geography:
        parts.append(f"Geography: {', '.join(geography)}")

    genre = doc.get("genre") or []
    if genre:
        parts.append(f"Formats: {', '.join(genre)}")

    return " | ".join(parts)


# ── Async entity extraction ───────────────────────────────────────────────────

async def extract_entities_for_doc(
    doc: dict,
    text: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Extract entities for a single document using GPT-4o-mini.
    Works for both full-text and metadata records.
    """
    if not text.strip():
        return None

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model       = "gpt-4o-mini",
                    temperature = 0,
                    max_tokens  = 500,
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": text},
                    ],
                )

                raw = response.choices[0].message.content.strip()

                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                entities = json.loads(raw)

                if not isinstance(entities, list):
                    return None

                clean_entities = []
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    name  = str(ent.get("text", "")).strip().lower()
                    etype = str(ent.get("type", "")).strip().upper()
                    if len(name) < 2 or etype not in {"PERSON", "PLACE", "ORG", "EVENT"}:
                        continue
                    clean_entities.append({
                        "text":  name,
                        "type":  etype,
                        "count": 1,
                    })

                if not clean_entities:
                    return None

                return {
                    "ark_id":      doc["ark_id"],
                    "title":       doc.get("title") or "",
                    "year":        doc.get("year"),
                    "institution": doc.get("institution") or "",
                    "source_url":  doc.get("source_url") or "",
                    "issue_date":  doc.get("issue_date") or "",
                    "entities":    clean_entities,
                }

            except json.JSONDecodeError:
                await asyncio.sleep(2 ** attempt)
                continue

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait = 2 ** (attempt + 2)
                    print(f"  Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  Error for {doc['ark_id']}: {e}")
                    return None

    return None


# ── Main extraction ───────────────────────────────────────────────────────────

async def extract_async(
    year:          int  = None,
    metadata_only: bool = False,
    concurrency:   int  = 50,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if metadata_only:
        suffix        = "metadata"
        system_prompt = METADATA_SYSTEM_PROMPT
        print("Fetching metadata records from PostgreSQL...")
        docs      = fetch_metadata_documents()
        get_text  = build_metadata_input
    else:
        suffix        = str(year) if year else "all"
        system_prompt = FULLTEXT_SYSTEM_PROMPT
        print("Fetching full-text documents from PostgreSQL...")
        docs      = fetch_fulltext_documents(year=year)
        get_text  = build_fulltext_input

    output_file = OUTPUT_DIR / f"entities_{suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"BPL Graph — Entity Extraction (GPT-4o-mini)")
    print(f"  Mode        : {'metadata-only' if metadata_only else f'full-text (year={year or \"all\"})'}")
    print(f"  Concurrency : {concurrency}")
    print(f"  Output      : {output_file}")
    print(f"{'='*60}\n")

    print(f"  Found {len(docs)} documents\n")

    semaphore  = asyncio.Semaphore(concurrency)
    start_time = time.monotonic()
    completed  = 0
    written    = 0
    BATCH_SIZE = 200

    with open(output_file, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(docs), BATCH_SIZE):
            batch     = docs[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, len(docs))

            tasks = [
                extract_entities_for_doc(
                    doc, get_text(doc), system_prompt, semaphore
                )
                for doc in batch
            ]
            results = await asyncio.gather(*tasks)

            for result in results:
                completed += 1
                if result is not None:
                    f.write(json.dumps(result) + "\n")
                    written += 1

            elapsed   = time.monotonic() - start_time
            remaining = (elapsed / completed) * (len(docs) - completed) if completed else 0
            rate      = completed / elapsed if elapsed > 0 else 0

            print(
                f"  [{batch_end}/{len(docs)}] "
                f"written={written} | "
                f"rate={rate:.1f} docs/s | "
                f"ETA: {remaining/60:.1f}min"
            )

    elapsed = time.monotonic() - start_time
    print(f"\n✓ Entity extraction complete.")
    print(f"  Documents processed     : {completed}")
    print(f"  Documents with entities : {written}")
    print(f"  Output file             : {output_file}")
    print(f"  Total time              : {elapsed/60:.1f} min")
    print(f"  Avg rate                : {completed/elapsed:.1f} docs/s")


def extract(year: int = None, metadata_only: bool = False, concurrency: int = 50):
    asyncio.run(extract_async(
        year          = year,
        metadata_only = metadata_only,
        concurrency   = concurrency,
    ))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity extraction with GPT-4o-mini")
    parser.add_argument("--year",          type=int, default=None)
    parser.add_argument("--all",           action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--concurrency",   type=int, default=50)
    args = parser.parse_args()

    if args.metadata_only:
        extract(metadata_only=True, concurrency=args.concurrency)
    else:
        extract(
            year        = None if args.all else (args.year or 1900),
            concurrency = args.concurrency,
        )