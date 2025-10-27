CREATE TABLE IF NOT EXISTS gold.bpl_embeddings (
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),
    metadata JSONB NOT NULL,
    date_start INT,
    date_end INT,
    PRIMARY KEY (document_id, chunk_index)
);

-- Date range indexes
CREATE INDEX IF NOT EXISTS bpl_embeddings_date_range_idx
ON gold.bpl_embeddings (date_start, date_end)
WHERE date_start IS NOT NULL;

CREATE INDEX IF NOT EXISTS bpl_embeddings_date_start_idx
ON gold.bpl_embeddings (date_start)
WHERE date_start IS NOT NULL;

CREATE INDEX IF NOT EXISTS bpl_embeddings_document_id_idx
ON gold.bpl_embeddings (document_id);

-- Create GIN index for fast JSONB containment queries
CREATE INDEX IF NOT EXISTS idx_bpl_embeddings_metadata_gin
ON gold.bpl_embeddings USING GIN (metadata jsonb_path_ops);