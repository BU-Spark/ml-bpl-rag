#!/usr/bin/env python3
import os
import torch
import psycopg2
import traceback
from tqdm import tqdm
from dotenv import load_dotenv
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def connect_db():
    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "prefer")
    )

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading model on {device}...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    print("✅ Model loaded.")
    return model

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_text(text: str):
    if not text:
        return []
    try:
        return splitter.split_text(text)
    except Exception:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start+1000])
            start += 900
        return chunks

def main():
    try:
        conn = connect_db()
        cur = conn.cursor()
        print("✅ Connected to PostgreSQL")

        model = load_model()

        print("📖 Fetching data from silver.bpl_combined...")
        cur.execute("SELECT document_id, summary_text, metadata FROM silver.bpl_combined;")
        rows = cur.fetchall()
        total_docs = len(rows)
        print(f"✅ Retrieved {total_docs:,} records")

        insert_sql = """
            INSERT INTO gold.bpl_embeddings (document_id, chunk_index, chunk_text, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id, chunk_index)
            DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;
        """

        
        batch, BATCH_SIZE = [], 100
        processed_docs, inserted_chunks = 0, 0

        for document_id, text, metadata in tqdm(rows, desc="🧠 Embedding documents"):
            if not text:
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                batch.append((document_id, idx, chunk, emb.tolist(), metadata))
                inserted_chunks += 1

                if len(batch) >= BATCH_SIZE:
                    execute_batch(cur, insert_sql, batch)
                    conn.commit()
                    batch.clear()

            processed_docs += 1
            if processed_docs % 100 == 0:
                print(f"📦 {processed_docs}/{total_docs} docs processed, {inserted_chunks:,} chunks inserted...")

        if batch:
            execute_batch(cur, insert_sql, batch)
            conn.commit()

        print(f"✅ Completed: {processed_docs:,} docs, {inserted_chunks:,} total chunks embedded.")

    except Exception as e:
        print("\n❌ ERROR OCCURRED:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
        print("🔒 Connection closed.")

if __name__ == "__main__":
    print("▶ Starting BPL embedding pipeline...")
    main()
