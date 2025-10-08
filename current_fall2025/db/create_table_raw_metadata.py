import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# Connect to Railway Postgres
conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    database=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "prefer")
)
cur = conn.cursor()

# SQL to create the raw_metadata table
sql = """
CREATE TABLE IF NOT EXISTS raw_metadata (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL
);
"""

# Run it
cur.execute(sql)
conn.commit()

print("✅ Table 'raw_metadata' created successfully!")

cur.close()
conn.close()
