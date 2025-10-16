#!/usr/bin/env python3
import os
import torch
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    length_function=len, separators=["\n\n", "\n", " ", ""]
)

def chunk_text(text: str):
    if not text:
        return []
    try:
        return splitter.split_text(text)
    except Exception:
        out, start = [], 0
        while start < len(text):
            end = start + 1000
            out.append(text[start:end])
            start += 900
        return out

def main():
    cur.execute("SELECT document_id, summary_text FROM silver.bpl_combined;")
    rows = cur.fetchall()
    insert_sql = """
        INSERT INTO gold.bpl_embeddings (document_id, chunk_index, chunk_text, embedding)
        VALUES (%s, %s, %s, %s)
    """
    batch, BATCH_SIZE = [], 100
    for document_id, text in tqdm(rows, desc="Embedding"):
        if not text:
            continue
        chunks = chunk_text(text)
        embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            batch.append((document_id, idx, chunk, emb.tolist()))
            if len(batch) >= BATCH_SIZE:
                execute_batch(cur, insert_sql, batch)
                conn.commit()
                batch.clear()
    if batch:
        execute_batch(cur, insert_sql, batch)
        conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
