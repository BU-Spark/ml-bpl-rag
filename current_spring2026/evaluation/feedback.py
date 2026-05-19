"""
evaluation/feedback.py

Persists human-in-the-loop feedback events to the `feedback` table on the
Railway Postgres + a local CSV mirror. Each event ties back to a `query_logs`
row by `query_id` so future analysis can join feedback ↔ ranking.

Usage:
    from evaluation.feedback import log_feedback
    log_feedback(
        query_id   = result.query_id,
        ark_id     = "8g84ms67v",
        signal     = "up",      # or "down" or "missing"
        comment    = "exactly the photo I wanted",
        session_id = st.session_state["session_id"],
        raw_query  = "Find pictures of JFK",
    )

`signal="missing"` is used for the "none of these were what I wanted"
textbox — `ark_id` should be empty in that case.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from database.schema import get_conn, get_cursor


VALID_SIGNALS = {"up", "down", "missing"}

FEEDBACK_CSV_PATH = Path("logs/feedback_log.csv")

CSV_HEADERS = [
    "feedback_id",
    "created_at",
    "session_id",
    "query_id",
    "raw_query",
    "ark_id",
    "signal",
    "comment",
]


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()


def log_feedback(
    query_id:   Optional[int],
    ark_id:     str,
    signal:     str,
    comment:    str = "",
    session_id: Optional[str] = None,
    raw_query:  str = "",
) -> Optional[int]:
    """
    Insert one feedback row. Returns the inserted feedback.id, or None if the
    DB write failed. Mirrors to a local CSV like logger.py.
    """
    if signal not in VALID_SIGNALS:
        raise ValueError(f"signal must be one of {VALID_SIGNALS}, got {signal!r}")

    created_at = datetime.now(timezone.utc).isoformat()
    feedback_id: Optional[int] = None

    # ── DB ──────────────────────────────────────────────────────────────────
    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (
                        query_id, ark_id, signal, comment,
                        session_id, raw_query, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        query_id,
                        ark_id or "",
                        signal,
                        comment or "",
                        session_id or "",
                        raw_query or "",
                        created_at,
                    ),
                )
                row = cur.fetchone()
                if row:
                    feedback_id = row["id"] if isinstance(row, dict) else row[0]
    except Exception as e:
        print(f"[feedback] DB write failed (non-fatal): {e}")

    # ── CSV mirror ──────────────────────────────────────────────────────────
    try:
        _ensure_csv(FEEDBACK_CSV_PATH)
        with open(FEEDBACK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow({
                "feedback_id": feedback_id,
                "created_at":  created_at,
                "session_id":  session_id or "",
                "query_id":    query_id if query_id is not None else "",
                "raw_query":   raw_query,
                "ark_id":      ark_id or "",
                "signal":      signal,
                "comment":     comment or "",
            })
    except Exception as e:
        print(f"[feedback] CSV write failed (non-fatal): {e}")

    return feedback_id


def get_thumbed_up_arks(query_id: int) -> list[str]:
    """Return ark_ids the user thumbed up for this query (used by refine)."""
    if query_id is None:
        return []
    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute(
                    "SELECT DISTINCT ark_id FROM feedback "
                    "WHERE query_id = %s AND signal = 'up' AND ark_id <> ''",
                    (query_id,),
                )
                rows = cur.fetchall()
                return [r["ark_id"] if isinstance(r, dict) else r[0] for r in rows]
    except Exception as e:
        print(f"[feedback] read thumbed-up failed (non-fatal): {e}")
        return []
