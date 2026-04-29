-- ── migrations.sql ───────────────────────────────────────────────────────────
-- Idempotent schema additions for human-in-the-loop feedback + richer query
-- logging. Safe to re-run; uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.
--
-- Run via: python -m database.migrate
-- ────────────────────────────────────────────────────────────────────────────

-- ── Enrich query_logs ───────────────────────────────────────────────────────
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS session_id        TEXT;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS retrieved_docs    JSONB;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS source_titles     JSONB;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS source_urls       JSONB;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS parent_query_id   BIGINT
    REFERENCES query_logs(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_query_logs_session
    ON query_logs (session_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_parent
    ON query_logs (parent_query_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_queried_at
    ON query_logs (queried_at DESC);

-- ── New table: feedback ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id           BIGSERIAL PRIMARY KEY,
    query_id     BIGINT REFERENCES query_logs(id) ON DELETE CASCADE,
    ark_id       TEXT,                              -- empty for "missing"
    signal       TEXT NOT NULL CHECK (signal IN ('up', 'down', 'missing')),
    comment      TEXT,
    session_id   TEXT,
    raw_query    TEXT,                              -- denormalized for fast lookup
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_query    ON feedback (query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session  ON feedback (session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_signal   ON feedback (signal);
CREATE INDEX IF NOT EXISTS idx_feedback_ark      ON feedback (ark_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created  ON feedback (created_at DESC);
