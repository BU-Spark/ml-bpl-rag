"""
evaluation/promote_feedback.py

Phase-3 manual job: promote high-confidence thumbs-up signals from the
`feedback` table into ground-truth pairs for `test_queries.jsonl`.

Defaults are conservative — a (query, ark_id) pair must have:
  - at least MIN_UPS thumbs-up across distinct sessions (default 2)
  - zero thumbs-down

Anything that survives is appended to `test_queries.jsonl` with question_type
"user_promoted" so future regressions surface in eval.

Usage:
    python -m evaluation.promote_feedback                  # dry-run
    python -m evaluation.promote_feedback --apply          # actually write
    python -m evaluation.promote_feedback --min-ups 3      # raise the bar
    python -m evaluation.promote_feedback --since 2026-01-01

Run from current_spring2026/ (so test_queries.jsonl resolves correctly).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Allow `python -m evaluation.promote_feedback` from current_spring2026/
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.schema import get_conn, get_cursor


DEFAULT_QUERIES_PATH = Path(__file__).parent.parent / "test_queries.jsonl"


def fetch_feedback(since: Optional[str] = None) -> list[dict]:
    """Pull all feedback rows joined to their query for raw_query lookup.
    Prefer feedback.raw_query (denormalized) when present, else fall back to
    query_logs.raw_query."""
    sql = """
        SELECT
            COALESCE(NULLIF(f.raw_query, ''), q.raw_query) AS raw_query,
            f.ark_id,
            f.signal,
            f.session_id,
            f.created_at
        FROM feedback f
        LEFT JOIN query_logs q ON q.id = f.query_id
        WHERE f.signal IN ('up', 'down')
          AND f.ark_id <> ''
    """
    params: list = []
    if since:
        sql += " AND f.created_at >= %s"
        params.append(since)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())


def aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group by (raw_query, ark_id). Track up-count, down-count, and the set
    of distinct sessions that thumbed up."""
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "ups": 0, "downs": 0, "up_sessions": set(),
    })
    for r in rows:
        key = (r["raw_query"] or "", r["ark_id"] or "")
        if not all(key):
            continue
        bucket = agg[key]
        if r["signal"] == "up":
            bucket["ups"] += 1
            if r["session_id"]:
                bucket["up_sessions"].add(r["session_id"])
        elif r["signal"] == "down":
            bucket["downs"] += 1
    return agg


def load_existing(queries_path: Path) -> tuple[list[dict], dict[str, set[str]]]:
    """Return (entries, {raw_query: {ark_id, ...}}) so we can skip dupes."""
    if not queries_path.exists():
        return [], {}
    entries = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    existing: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        q = e.get("question", "")
        for g in e.get("ground_truths", []) or []:
            ark = g.get("ark_id", "").removeprefix("commonwealth:")
            if ark:
                existing[q].add(ark)
    return entries, existing


def main():
    parser = argparse.ArgumentParser(
        description="Promote high-confidence thumbs-up feedback to test_queries.jsonl"
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES_PATH),
        help="Path to test_queries.jsonl",
    )
    parser.add_argument(
        "--min-ups", type=int, default=2,
        help="Minimum thumbs-up across distinct sessions (default 2)",
    )
    parser.add_argument(
        "--max-downs", type=int, default=0,
        help="Maximum thumbs-down allowed (default 0)",
    )
    parser.add_argument(
        "--since", default=None,
        help="ISO timestamp; only feedback created at-or-after this is considered",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually append to test_queries.jsonl (default is dry-run)",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    print(f"[promote] reading feedback ...")
    rows = fetch_feedback(since=args.since)
    print(f"[promote] {len(rows)} thumb events fetched")

    agg = aggregate(rows)
    print(f"[promote] {len(agg)} distinct (query, ark) pairs")

    existing_entries, existing_arks = load_existing(queries_path)
    print(f"[promote] existing test_queries.jsonl: {len(existing_entries)} entries")

    # Decide which pairs to promote
    candidates: list[tuple[str, str, dict]] = []
    for (q, ark), bucket in agg.items():
        ups_distinct = len(bucket["up_sessions"]) or bucket["ups"]
        if ups_distinct < args.min_ups:
            continue
        if bucket["downs"] > args.max_downs:
            continue
        if ark in existing_arks.get(q, set()):
            continue                                    # already in eval set
        candidates.append((q, ark, bucket))

    if not candidates:
        print("[promote] nothing to promote.")
        return

    print(f"[promote] {len(candidates)} pair(s) qualify:")
    for q, ark, bucket in candidates:
        print(f"  + {q[:60]!r}  →  {ark}   "
              f"(ups={bucket['ups']} sessions={len(bucket['up_sessions'])} downs={bucket['downs']})")

    if not args.apply:
        print("\n[promote] dry-run. Re-run with --apply to write.")
        return

    # Group new ground truths by query so we extend existing entries when possible
    by_query: dict[str, list[str]] = defaultdict(list)
    for q, ark, _ in candidates:
        by_query[q].append(ark)

    existing_entries_by_q = {e["question"]: e for e in existing_entries}
    new_entries: list[dict] = []
    for q, arks in by_query.items():
        if q in existing_entries_by_q:
            entry = existing_entries_by_q[q]
            entry.setdefault("ground_truths", [])
            for ark in arks:
                entry["ground_truths"].append({
                    "title":  "(promoted from user feedback)",
                    "ark_id": f"commonwealth:{ark}",
                })
        else:
            new_entries.append({
                "question":      q,
                "question_type": "user_promoted",
                "ground_truths": [
                    {"title": "(promoted from user feedback)",
                     "ark_id": f"commonwealth:{ark}"}
                    for ark in arks
                ],
                "answer": "",
                "notes":  "auto-promoted by promote_feedback.py",
            })

    out_path = queries_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Rewrite the file: existing entries (possibly mutated) + new entries
    with open(out_path, "w", encoding="utf-8") as f:
        for e in existing_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[promote] wrote {out_path}: "
          f"{len(existing_entries)} existing + {len(new_entries)} new entries.")


if __name__ == "__main__":
    main()
