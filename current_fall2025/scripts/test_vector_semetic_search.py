#!/usr/bin/env python3
import os
import torch
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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
print("✅ Connected to PostgreSQL")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
print("🧠 Model loaded (sentence-transformers/all-mpnet-base-v2)")

# ---- Query embedding ----
query = "What were some important historical events that happened in Boston in 1919?"
print(f"Embedding query: {query}")
query_vec = model.encode([query])[0].tolist()

# ---- SQL similarity search ----
sql = """
SELECT document_id,
       chunk_index,
       LEFT(chunk_text, 100) AS snippet,
       1 - (embedding <=> %s::vector) AS similarity
FROM gold.bpl_embeddings
ORDER BY embedding <=> %s::vector
LIMIT 5;
"""

cur.execute(sql, (query_vec, query_vec))
rows = cur.fetchall()

print("\n📚 Top 5 relevant chunks:\n")
for doc_id, chunk_idx, snippet, sim in rows:
    print(f"🧩 {doc_id} | chunk {chunk_idx} | sim={sim:.3f}")
    print(f"   → {snippet}...\n")

cur.close()
conn.close()
print("🔒 Connection closed.")
