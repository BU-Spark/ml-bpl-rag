"""
evaluation/logger.py

Structured logger for every query that passes through the pipeline.
Writes to:
  1. The `query_logs` table in PostgreSQL (for dashboarding and SQL analysis)
  2. A local CSV file (for portability and DeepEval input)

Usage:
    from evaluation.logger import log_query
    log_query(intent, retrieved_docs, generation_result, latency_ms=240)
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from database.schema import get_conn, get_cursor
from retrieval.query_understanding import QueryIntent
from retrieval.retriever import RetrievedDocument
from generation.generator import GenerationResult

LOG_CSV_PATH = Path("logs/query_log.csv")

CSV_HEADERS = [
    "queried_at",
    "raw_query",
    "rewritten_query",
    "query_type",
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


def log_query(
    intent:            QueryIntent,
    retrieved_docs:    List[RetrievedDocument],
    generation_result: GenerationResult,
    latency_ms:        int = 0,
    relevancy_score:   Optional[float] = None,
    faithfulness_score: Optional[float] = None,
):
    """
    Persist one query event to the DB and CSV log.
    Fails silently on DB errors so a logging failure never breaks the search UX.
    """
    queried_at      = datetime.now(timezone.utc).isoformat()
    retrieved_arks  = [d.ark_id for d in retrieved_docs]
    filters_json    = json.dumps({
        "year_min":  intent.date_filter.year_min,
        "year_max":  intent.date_filter.year_max,
        "geography": intent.geography,
        "topics":    intent.topics,
        "doc_types": intent.doc_types,
    })

    # ── 1. Write to PostgreSQL ──────────────────────────────────────────────
    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (
                        queried_at, raw_query, rewritten_query, query_type,
                        filters, retrieved_ark_ids, response,
                        relevancy_score, faithfulness_score, latency_ms
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        queried_at,
                        intent.raw_query,
                        intent.rewritten_query,
                        intent.query_type,
                        filters_json,
                        retrieved_arks,
                        generation_result.response,
                        relevancy_score,
                        faithfulness_score,
                        latency_ms,
                    ),
                )
    except Exception as e:
        print(f"[logger] DB write failed (non-fatal): {e}")

    # ── 2. Append to CSV ────────────────────────────────────────────────────
    try:
        _ensure_csv(LOG_CSV_PATH)
        row = {
            "queried_at":        queried_at,
            "raw_query":         intent.raw_query,
            "rewritten_query":   intent.rewritten_query,
            "query_type":        intent.query_type,
            "year_min":          intent.date_filter.year_min,
            "year_max":          intent.date_filter.year_max,
            "geography":         "|".join(intent.geography),
            "topics":            "|".join(intent.topics),
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
