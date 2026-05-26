"""Pipeline tables: FTS, knowledge graph, assertions, context KB, hierarchy.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-26
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Full-text search on document_chunks ───────────────────────────────────
    op.execute(
        """
        ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS texto_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('spanish', coalesce(texto, ''))) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv
            ON document_chunks USING GIN (texto_tsv)
        """
    )

    # ── Knowledge graph ───────────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id         TEXT PRIMARY KEY,
            node_type  TEXT NOT NULL CHECK (node_type IN (
                           'regulation_article','calibration_segment','risk_parameter',
                           'portfolio','capital_metric','assertion','rating_system',
                           'db_table','db_column'
                       )),
            label      TEXT NOT NULL,
            metadata   JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_meta ON graph_nodes USING GIN (metadata)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_edges (
            id         SERIAL PRIMARY KEY,
            from_id    TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
            to_id      TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
            relation   TEXT NOT NULL CHECK (relation IN (
                           'CONSTRAINS','APPLIES_TO','FEEDS_INTO',
                           'DEPENDS_ON','TESTED_BY','VALIDATED_BY','MAPS_TO'
                       )),
            weight     FLOAT NOT NULL DEFAULT 1.0,
            metadata   JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (from_id, to_id, relation)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges (from_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_to ON graph_edges (to_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_rel ON graph_edges (relation)"
    )

    # ── Regulation assertions ─────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS regulation_assertions (
            id              TEXT PRIMARY KEY,
            regulation_id   TEXT NOT NULL,
            regulation_name TEXT NOT NULL,
            article         TEXT NOT NULL,
            passage         TEXT,
            rule_text       TEXT NOT NULL,
            check_type      TEXT NOT NULL,
            conditions      JSONB DEFAULT '[]',
            threshold       JSONB,
            materiality     TEXT NOT NULL DEFAULT 'medium',
            sql_template    TEXT,
            notes           TEXT DEFAULT '',
            active          BOOLEAN NOT NULL DEFAULT TRUE,
            regulation_date DATE,
            created_at      TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_assertions_regulation ON regulation_assertions (regulation_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_assertions_materiality ON regulation_assertions (materiality)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_assertions_active ON regulation_assertions (active) WHERE active"
    )

    # ── Context knowledge base ────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS context_chunks (
            id        SERIAL PRIMARY KEY,
            content   TEXT NOT NULL,
            metadata  JSONB DEFAULT '{}',
            embedding vector(768)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS context_chunks_embedding_idx
            ON context_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )

    # ── Hierarchical retrieval ────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_hierarchy (
            node_id       TEXT PRIMARY KEY,
            doc_id        TEXT NOT NULL,
            level_type    TEXT NOT NULL CHECK (level_type IN (
                              'document','part','chapter','section','subsection','article'
                          )),
            number        TEXT,
            label         TEXT NOT NULL,
            parent_id     TEXT REFERENCES document_hierarchy(node_id) ON DELETE CASCADE,
            position      INT NOT NULL DEFAULT 0,
            summary       TEXT,
            embedding     vector(768),
            article_count INT NOT NULL DEFAULT 0,
            metadata      JSONB NOT NULL DEFAULT '{}'
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dh_doc ON document_hierarchy (doc_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dh_parent ON document_hierarchy (parent_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dh_level ON document_hierarchy (level_type)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dh_emb ON document_hierarchy
            USING hnsw (embedding vector_cosine_ops)
            WHERE embedding IS NOT NULL
        """
    )

    op.execute(
        """
        ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS hierarchy_node_id TEXT
                REFERENCES document_hierarchy(node_id) ON DELETE SET NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dc_hierarchy ON document_chunks (hierarchy_node_id)
            WHERE hierarchy_node_id IS NOT NULL
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS validation_themes (
            id          TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            description TEXT NOT NULL,
            keywords    JSONB NOT NULL DEFAULT '[]',
            embedding   vector(768),
            parent_id   TEXT REFERENCES validation_themes(id) ON DELETE CASCADE,
            metadata    JSONB NOT NULL DEFAULT '{}'
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_vt_parent ON validation_themes (parent_id)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vt_emb ON validation_themes
            USING hnsw (embedding vector_cosine_ops)
            WHERE embedding IS NOT NULL
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_themes (
            chunk_id  TEXT NOT NULL REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
            theme_id  TEXT NOT NULL REFERENCES validation_themes(id)     ON DELETE CASCADE,
            score     FLOAT NOT NULL DEFAULT 1.0,
            method    TEXT NOT NULL DEFAULT 'keyword'
                          CHECK (method IN ('keyword','embedding','llm')),
            PRIMARY KEY (chunk_id, theme_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ct_theme ON chunk_themes (theme_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ct_score ON chunk_themes (theme_id, score DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunk_themes")
    op.execute("DROP TABLE IF EXISTS validation_themes")
    op.execute("DROP INDEX IF EXISTS idx_dc_hierarchy")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS hierarchy_node_id")
    op.execute("DROP TABLE IF EXISTS document_hierarchy")
    op.execute("DROP TABLE IF EXISTS context_chunks")
    op.execute("DROP TABLE IF EXISTS regulation_assertions")
    op.execute("DROP TABLE IF EXISTS graph_edges")
    op.execute("DROP TABLE IF EXISTS graph_nodes")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS texto_tsv")
