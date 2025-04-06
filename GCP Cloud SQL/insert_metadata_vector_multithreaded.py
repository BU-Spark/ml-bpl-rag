import os
import json
import uuid
import time
import ijson
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2.pool import ThreadedConnectionPool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import decimal


# Load environment variables
load_dotenv()

# DB settings
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bpl_metadata_new")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FILE_PATH = "/projectnb/sparkgrp/ml-bpl-rag-data/extraneous/metadata/bpl_data.json"
NUM_THREADS = 4

FIELDS_TO_EMBED = [
    'abstract_tsi',
    'title_info_primary_tsi',
    'title_info_primary_subtitle_tsi',
    'title_info_alternative_tsim'
]

# DB connection pool
DB_POOL = ThreadedConnectionPool(
    minconn=NUM_THREADS,
    maxconn=NUM_THREADS * 2,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

def default_serializer(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# Table setup
conn = DB_POOL.getconn()
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    source_id TEXT,
    field TEXT,
    content TEXT,
    embedding VECTOR(768),
    metadata JSONB
);
""")
conn.commit()

# Fetch existing (source_id, field) pairs
cur.execute("SELECT DISTINCT source_id, field FROM documents;")
inserted_pairs = set((row[0], row[1]) for row in cur.fetchall())
cur.close()
DB_POOL.putconn(conn)

# Embedder and splitter
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Worker: embed + chunk one item
def process_item(item):
    results = []
    try:
        item_id = item.get("id")
        attributes = item.get("attributes", {})
        metadata = json.dumps(attributes, default=default_serializer)

        for field in attributes:
            if (field in FIELDS_TO_EMBED) or ("note" in field) and (item_id, field) not in inserted_pairs:
                text = str(attributes[field])
                if not text.strip():
                    continue

                chunks = text_splitter.split_text(text) if len(text) > 1000 else [text]
                vectors = embeddings.embed_documents(chunks)

                for chunk, vector in zip(chunks, vectors):
                    results.append((str(uuid.uuid4()), item_id, field, chunk, vector, metadata))
    except Exception as e:
        print(f"⚠️ Error processing {item.get('id')}: {e}")
    return results

# Insert to DB
def insert_batch(batch):
    if not batch:
        return
    conn = None
    try:
        conn = DB_POOL.getconn()
        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO documents (id, source_id, field, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, batch)
        conn.commit()
        cur.close()
        print(f"Inserted {len(batch)} chunks")
    except Exception as e:
        print(f"Insert error: {e}")
    finally:
        if conn:
            DB_POOL.putconn(conn)

# Main
start = time.time()
with open(FILE_PATH, "r") as f:
    item_iterator = ijson.items(f, "Data.item.data.item")

    batch = []
    futures = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        with tqdm(desc="Embedding & Inserting", unit="doc") as pbar:
            for item in item_iterator:
                future = executor.submit(process_item, item)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                if result:
                    batch += result
                    if len(batch) >= 500:
                        insert_batch(batch)
                        batch = []
                pbar.update(1)

            if batch:
                insert_batch(batch)

DB_POOL.closeall()
elapsed = time.time() - start
print(f"Processed {total} items in {elapsed:.2f} seconds.")
