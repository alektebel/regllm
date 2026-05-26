-- Knowledge graph: nodes and directed edges
-- Nodes represent: regulation articles, calibration segments, risk parameters,
--                  portfolios, capital metrics, and assertions themselves.
-- Edges express the dependency/constraint relationships between them.
-- Recursive CTE traversal enables materiality propagation and root-cause clustering.

CREATE TABLE IF NOT EXISTS graph_nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL CHECK (node_type IN (
                    'regulation_article',
                    'calibration_segment',
                    'risk_parameter',
                    'portfolio',
                    'capital_metric',
                    'assertion',
                    'rating_system',
                    'db_table',
                    'db_column'
                )),
    label       TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_meta ON graph_nodes USING GIN (metadata);

CREATE TABLE IF NOT EXISTS graph_edges (
    id          SERIAL PRIMARY KEY,
    from_id     TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    to_id       TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    -- Semantics:
    --   CONSTRAINS   : article/assertion imposes a rule on a parameter or segment
    --   APPLIES_TO   : parameter applies to a portfolio or segment
    --   FEEDS_INTO   : segment/parameter output contributes to a capital metric
    --   DEPENDS_ON   : segment computation requires another segment or parameter
    --   TESTED_BY    : segment or parameter is tested by an assertion
    --   VALIDATED_BY : segment must be validated according to a guideline article
    --   MAPS_TO      : db_column maps to a risk_parameter or calibration_segment
    relation    TEXT NOT NULL CHECK (relation IN (
                    'CONSTRAINS', 'APPLIES_TO', 'FEEDS_INTO',
                    'DEPENDS_ON', 'TESTED_BY', 'VALIDATED_BY', 'MAPS_TO'
                )),
    -- weight: fraction of total RWA, capital, etc. (used for materiality scoring)
    weight      FLOAT NOT NULL DEFAULT 1.0,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (from_id, to_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges (from_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to   ON graph_edges (to_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_rel  ON graph_edges (relation);

-- regulation_assertions table (created by assertion extractor on first run,
-- but we define it here so the pipeline can reference it from startup)
CREATE TABLE IF NOT EXISTS regulation_assertions (
    id               TEXT PRIMARY KEY,
    regulation_id    TEXT NOT NULL,
    regulation_name  TEXT NOT NULL,
    article          TEXT NOT NULL,
    passage          TEXT,
    rule_text        TEXT NOT NULL,
    check_type       TEXT NOT NULL,
    conditions       JSONB DEFAULT '[]',
    threshold        JSONB,
    materiality      TEXT NOT NULL DEFAULT 'medium',
    sql_template     TEXT,
    notes            TEXT DEFAULT '',
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    regulation_date  DATE,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_assertions_regulation ON regulation_assertions (regulation_id);
CREATE INDEX IF NOT EXISTS idx_assertions_materiality ON regulation_assertions (materiality);
CREATE INDEX IF NOT EXISTS idx_assertions_active ON regulation_assertions (active) WHERE active;
