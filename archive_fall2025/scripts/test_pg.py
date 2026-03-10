import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "require")
    )
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print("✅ Connected successfully!")
    print("PostgreSQL version:", version)
    cur.close()
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
