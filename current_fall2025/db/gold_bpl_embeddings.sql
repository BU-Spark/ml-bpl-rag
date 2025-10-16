CREATE TABLE IF NOT EXISTS gold.bpl_embeddings (
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),
    PRIMARY KEY (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS bpl_embeddings_hnsw_idx
ON gold.bpl_embeddings
USING hnsw (embedding vector_cosine_ops);
