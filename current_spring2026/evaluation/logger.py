"""
evaluation/logger.py

Structured logger for every query that passes through the pipeline.
Writes to:
  1. The `query_logs` table in PostgreSQL (Railway) — for dashboarding and
     SQL analysis. Returns the inserted row's id so callers can attach
     downstream feedback events.
  2. A local CSV file (for portability and DeepEval input).

Usage:
    from evaluation.logger import log_query
    query_id = log_query(intent, retrieved_docs, generation_result,
                         latency_ms=240, session_id="...", parent_query_id=None)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from database.schema import get_conn, get_cursor
from retrieval.query_understanding import QueryIntent
from retrieval.retriever import RetrievedDocument
from generation.generator import GenerationResult

LOG_CSV_PATH = Path("logs/query_log.csv")

CSV_HEADERS = [
    "query_id",
    "queried_at",
    "session_id",
    "parent_query_id",
    "raw_query",
    "rewritten_query",
    "year_min",
    "year_max",
    "geography",
    "topics",
    "retrieved_ark_ids",
    "response",
    "latency_ms",
    "relevancy_score",
    "faithfulness_score",
]


def _ensure_csv(path: Path):
    """Create CSV with headers if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def _serialize_docs(docs: List[RetrievedDocument]) -> str:
    """JSON-encode the retrieved doc list with the fields that matter for
    replay/audit (rank, ark_id, title, url, score)."""
    payload = []
    for rank, d in enumerate(docs, start=1):
        payload.append({
            "rank":        rank,
            "ark_id":      d.ark_id,
            "title":       d.title,
            "source_url":  d.source_url,
            "institution": d.institution,
            "issue_date":  d.issue_date,
            "rrf_score":   round(d.rrf_score, 6),
            "metadata_sim": round(d.metadata_sim, 6),
            "final_score": round(d.final_score, 6),
        })
    return json.dumps(payload, ensure_ascii=False)


def log_query(
    intent:             QueryIntent,
    retrieved_docs:     List[RetrievedDocument],
    generation_result:  GenerationResult,
    latency_ms:         int = 0,
    relevancy_score:    Optional[float] = None,
    faithfulness_score: Optional[float] = None,
    session_id:         Optional[str] = None,
    parent_query_id:    Optional[int] = None,
) -> Optional[int]:
    """
    Persist one query event to the DB and CSV log.
    Returns the inserted query_logs.id, or None if the DB write failed.
    Fails silently on errors so a logging failure never breaks the search UX.
    """
    queried_at      = datetime.now(timezone.utc).isoformat()
    retrieved_arks  = [d.ark_id for d in retrieved_docs]
    filters_json    = json.dumps({
        "year_min":  intent.date_filter.year_min,
        "year_max":  intent.date_filter.year_max,
    })
    docs_json    = _serialize_docs(retrieved_docs)
    titles_json  = json.dumps(generation_result.source_titles or [], ensure_ascii=False)
    urls_json    = json.dumps(generation_result.source_urls   or [], ensure_ascii=False)

    # ── 1. Write to PostgreSQL ──────────────────────────────────────────────
    query_id: Optional[int] = None
    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (
                        queried_at, raw_query, rewritten_query,
                        filters, retrieved_ark_ids, response,
                        relevancy_score, faithfulness_score, latency_ms,
                        session_id, retrieved_docs, source_titles, source_urls,
                        parent_query_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s::jsonb, %s::jsonb, %s::jsonb, %s
                    )
                    RETURNING id
                    """,
                    (
                        queried_at,
                        intent.raw_query,
                        intent.rewritten_query,
                        filters_json,
                        retrieved_arks,
                        generation_result.response,
                        relevancy_score,
                        faithfulness_score,
                        latency_ms,
                        session_id,
                        docs_json,
                        titles_json,
                        urls_json,
                        parent_query_id,
                    ),
                )
                row = cur.fetchone()
                if row:
                    query_id = row["id"] if isinstance(row, dict) else row[0]
    except Exception as e:
        print(f"[logger] DB write failed (non-fatal): {e}")

    # ── 2. Append to CSV ────────────────────────────────────────────────────
    try:
        _ensure_csv(LOG_CSV_PATH)
        row = {
            "query_id":          query_id,
            "queried_at":        queried_at,
            "session_id":        session_id or "",
            "parent_query_id":   parent_query_id or "",
            "raw_query":         intent.raw_query,
            "rewritten_query":   intent.rewritten_query,
            "year_min":          intent.date_filter.year_min,
            "year_max":          intent.date_filter.year_max,
            "geography":         "",
            "topics":            "",
            "retrieved_ark_ids": "|".join(retrieved_arks),
            "response":          generation_result.response,
            "latency_ms":        latency_ms,
            "relevancy_score":   relevancy_score,
            "faithfulness_score": faithfulness_score,
        }
        with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)
    except Exception as e:
        print(f"[logger] CSV write failed (non-fatal): {e}")

    return query_id
