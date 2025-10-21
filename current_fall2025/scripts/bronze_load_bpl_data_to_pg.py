import os
import json
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_batch

# Load connection info from .env
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    database=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "prefer")
)
cur = conn.cursor()

DATA_DIR = "../data/raw"  # path to your folder
insert_sql = """
INSERT INTO bronze.bpl_metadata (id, data)
VALUES (%s, %s)
ON CONFLICT (id) DO NOTHING;
"""

total_inserted = 0

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(DATA_DIR, filename)
    print(f"📄 Loading file: {filename}")

    with open(path, "r") as f:
        try:
            content = json.load(f)
            # Some files may wrap data in {"data": [...]}
            items = content["data"] if isinstance(content, dict) and "data" in content else content

            batch_data = [(item["id"], json.dumps(item)) for item in items if "id" in item]
            execute_batch(cur, insert_sql, batch_data)
            conn.commit()

            print(f"✅ Inserted {len(batch_data)} records from {filename}")
            total_inserted += len(batch_data)
        except Exception as e:
            print(f"⚠️ Skipped {filename}: {e}")

print(f"\n🎉 Done! Inserted a total of {total_inserted} records.")
cur.close()
conn.close()
