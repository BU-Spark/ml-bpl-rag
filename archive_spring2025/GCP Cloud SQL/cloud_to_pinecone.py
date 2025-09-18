import os
import json
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import decimal
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Pinecone client (new SDK)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("bpl-test")  # ⬅️ Replace with your actual index name

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5436")
DB_NAME = os.getenv("DB_NAME", "bpl_metadata_new")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Postgres connection
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

# Fetch a single document
cur.execute("SELECT id, embedding, metadata FROM documents LIMIT 1;")
row = cur.fetchone()

# Convert metadata safely
def safe_metadata(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

doc_id = str(row[0])
embedding = row[1]
metadata = json.loads(json.dumps(row[2], default=safe_metadata))

# Upsert to Pinecone
index.upsert([
    {
        "id": doc_id,
        "values": embedding,
        "metadata": metadata
    }
])

print(f"✅ Successfully upserted document {doc_id} to Pinecone.")

# Fetch back from Pinecone
res = index.fetch(ids=[doc_id])
print(json.dumps(res.to_dict(), indent=2))
