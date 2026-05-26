-- Migration: replace in-process BM25 with Postgres full-text search.
-- Safe to run multiple times (all statements are idempotent).

-- 1. Add tsvector column (Spanish dictionary for stemming/stop-words).
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS texto_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('spanish', coalesce(texto, ''))) STORED;

-- 2. GIN index — makes ts_rank queries fast even on large corpora.
CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv
    ON document_chunks USING GIN (texto_tsv);
