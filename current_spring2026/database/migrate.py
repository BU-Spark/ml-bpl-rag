"""
database/migrate.py

Runs database/migrations.sql against the configured Postgres (Railway).
Idempotent — uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS throughout.

Run once after pulling the human-in-the-loop changes:
    python -m database.migrate
"""

from __future__ import annotations

from pathlib import Path

from database.schema import get_conn, get_cursor


SQL_PATH = Path(__file__).parent / "migrations.sql"


def run_migrations() -> None:
    sql = SQL_PATH.read_text()
    print(f"[migrate] applying {SQL_PATH.name} ...")
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql)
    print("[migrate] done. Added: query_logs.{session_id,retrieved_docs,"
          "source_titles,source_urls,parent_query_id}, table feedback.")


if __name__ == "__main__":
    run_migrations()
