"""
database/schema.py

Creates all pgVector tables from scratch.
Run once:  python -m database.schema
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from config import PG_DSN


@contextmanager
def get_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(conn):
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cur
    finally:
        cur.close()


DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Table 1: documents ───────────────────────────────────────────────────────
-- One row per source record.
-- Holds all metadata, dense metadata embedding, and sparse metadata embedding.

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,

    -- Identity
    ark_id          TEXT UNIQUE NOT NULL,
    record_id       TEXT,
    source_url      TEXT,
    iiif_manifest   TEXT,

    -- Provenance
    newspaper       TEXT,
    collection      TEXT,
    institution     TEXT,

    -- Bibliographic
    title           TEXT,
    issue_date      TEXT,
    date_iso        TEXT[],
    date_start      TIMESTAMPTZ,
    year            INT[],
    publisher       TEXT[],
    place           TEXT[],
    language        TEXT[],

    -- Content signals
    page_count      INT,
    pages           TEXT[],
    topics          TEXT[],
    geography       TEXT[],
    char_count      INT,
    genre               TEXT[],
    abstract            TEXT,
    exemplary_image_id  TEXT,

    -- Dense metadata embedding (BGE-M3 → 1024 dims)
    metadata_embedding  VECTOR(1024),

    -- Sparse metadata embedding (two-array format, more efficient than JSONB)
    sparse_token_ids    INTEGER[],
    sparse_weights      FLOAT4[],

    -- Tracking
    ingested_at     TIMESTAMPTZ
);

-- HNSW index on metadata embedding (faster + smaller than ivfflat)
CREATE INDEX IF NOT EXISTS idx_documents_meta_emb
    ON documents
    USING hnsw (metadata_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_documents_ark      ON documents (ark_id);
CREATE INDEX IF NOT EXISTS idx_documents_year     ON documents USING GIN (year);
CREATE INDEX IF NOT EXISTS idx_documents_language ON documents USING GIN (language);
CREATE INDEX IF NOT EXISTS idx_documents_topics   ON documents USING GIN (topics);
CREATE INDEX IF NOT EXISTS idx_documents_geo      ON documents USING GIN (geography);
CREATE INDEX IF NOT EXISTS idx_documents_genre ON documents USING GIN (genre);


-- ── Table 2: chunks ──────────────────────────────────────────────────────────
-- One row per text chunk (512 tokens, 100 overlap).
-- No tsvector column — lexical search handled by BGE-M3 sparse instead.

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ark_id          TEXT NOT NULL,

    chunk_index     INT NOT NULL,
    chunk_text      TEXT NOT NULL,

    -- Dense full-text embedding (BGE-M3 → 1024 dims)
    text_embedding  VECTOR(1024),

    -- Sparse full-text embedding (two-array format)
    sparse_token_ids    INTEGER[],
    sparse_weights      FLOAT4[]
);

-- HNSW index on chunk text embedding
CREATE INDEX IF NOT EXISTS idx_chunks_text_emb
    ON chunks
    USING hnsw (text_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);


-- ── Table 3: query_logs ──────────────────────────────────────────────────────
-- Structured log of every query for evaluation and dashboarding.

CREATE TABLE IF NOT EXISTS query_logs (
    id                  BIGSERIAL PRIMARY KEY,
    queried_at          TIMESTAMPTZ DEFAULT NOW(),

    raw_query           TEXT,
    rewritten_query     TEXT,
    query_type          TEXT,

    filters             JSONB,
    retrieved_ark_ids   TEXT[],
    response            TEXT,

    relevancy_score     FLOAT,
    faithfulness_score  FLOAT,
    latency_ms          INT
);
"""


def create_schema():
    print("Creating schema ...")
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(DDL)
    print("Done. Tables: documents, chunks, query_logs")


if __name__ == "__main__":
    create_schema()
