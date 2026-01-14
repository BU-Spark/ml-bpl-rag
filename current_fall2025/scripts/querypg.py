#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# ==============================================================
# 🔧 EDIT THIS QUERY
# ==============================================================

query = """
select document_id, metadata from gold.bpl_embeddings
limit 5
"""

# ==============================================================
# 🚀 DO NOT EDIT BELOW
# ==============================================================

load_dotenv()

def connect():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT"),
            database=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            sslmode=os.getenv("PGSSLMODE", "prefer"),
        )
        conn.autocommit = True
        print("✅ Connected successfully!")
        return conn
    except Exception as e:
        print("❌ Connection failed:", e)
        raise

def run_query(conn, query: str):
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            if cur.description:
                rows = cur.fetchall()
                print(f"✅ Query returned {len(rows)} row(s):\n")
                for r in rows:
                    print(dict(r))
            else:
                print("✅ Query executed successfully (no return).")
    except Exception as e:
        print("❌ Error executing query:")
        print(e)
        conn.rollback()

if __name__ == "__main__":
    conn = connect()
    run_query(conn, query)
    conn.close()
