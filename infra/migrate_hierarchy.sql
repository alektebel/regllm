-- Hierarchical retrieval layer
-- Two complementary structures:
--   1. document_hierarchy: structural tree within each document (Chapter → Article)
--   2. validation_themes + chunk_themes: cross-document semantic groupings for validation work

-- ── Structural hierarchy ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS document_hierarchy (
    node_id     TEXT PRIMARY KEY,                   -- e.g. "eba_rts_irb__ch3"
    doc_id      TEXT NOT NULL,
    level_type  TEXT NOT NULL CHECK (level_type IN
                    ('document','part','chapter','section','subsection','article')),
    number      TEXT,                               -- "3", "3.1", "Article 9"
    label       TEXT NOT NULL,                      -- "Chapter 3: Validation function"
    parent_id   TEXT REFERENCES document_hierarchy(node_id) ON DELETE CASCADE,
    position    INT NOT NULL DEFAULT 0,             -- ordering within parent
    summary     TEXT,                               -- LLM or aggregate summary
    embedding   vector(768),                        -- semantic search at this level
    article_count INT NOT NULL DEFAULT 0,
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_dh_doc    ON document_hierarchy (doc_id);
CREATE INDEX IF NOT EXISTS idx_dh_parent ON document_hierarchy (parent_id);
CREATE INDEX IF NOT EXISTS idx_dh_level  ON document_hierarchy (level_type);
CREATE INDEX IF NOT EXISTS idx_dh_emb    ON document_hierarchy
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

-- Link chunks to their structural node
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS hierarchy_node_id TEXT
        REFERENCES document_hierarchy(node_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_dc_hierarchy ON document_chunks (hierarchy_node_id)
    WHERE hierarchy_node_id IS NOT NULL;

-- ── Validation themes ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS validation_themes (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords    JSONB NOT NULL DEFAULT '[]',        -- fast keyword pre-filter
    embedding   vector(768),                        -- theme description embedding
    parent_id   TEXT REFERENCES validation_themes(id) ON DELETE CASCADE,
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_vt_parent ON validation_themes (parent_id);
CREATE INDEX IF NOT EXISTS idx_vt_emb    ON validation_themes
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

-- Junction: each chunk can belong to multiple themes with a relevance score
CREATE TABLE IF NOT EXISTS chunk_themes (
    chunk_id    TEXT NOT NULL REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    theme_id    TEXT NOT NULL REFERENCES validation_themes(id)     ON DELETE CASCADE,
    score       FLOAT NOT NULL DEFAULT 1.0,         -- 0-1 relevance
    method      TEXT NOT NULL DEFAULT 'keyword'     -- 'keyword' | 'embedding' | 'llm'
                    CHECK (method IN ('keyword','embedding','llm')),
    PRIMARY KEY (chunk_id, theme_id)
);

CREATE INDEX IF NOT EXISTS idx_ct_theme ON chunk_themes (theme_id);
CREATE INDEX IF NOT EXISTS idx_ct_score ON chunk_themes (theme_id, score DESC);
