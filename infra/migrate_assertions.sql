-- Migration: regulation_assertions table.
-- Stores discrete testable rules extracted from regulation text.
-- Safe to run multiple times (all statements are idempotent).

CREATE TABLE IF NOT EXISTS regulation_assertions (
    id              TEXT PRIMARY KEY,          -- {regulation_id}_{article_slug}_{seq}
    regulation_id   TEXT NOT NULL,             -- e.g. "crr_575_2013"
    regulation_name TEXT NOT NULL,
    article         TEXT NOT NULL,             -- e.g. "Artículo 160"
    passage         TEXT NOT NULL,             -- source regulation passage (truncated)
    rule_text       TEXT NOT NULL,             -- plain-language rule in Spanish
    check_type      TEXT NOT NULL,             -- range|existence|classification|time_limit|ratio|completeness
    conditions      JSONB NOT NULL DEFAULT '[]',   -- scope pre-conditions
    threshold       JSONB,                     -- {field_hint, op, value, unit} or null
    materiality     TEXT NOT NULL DEFAULT 'medium', -- low|medium|high|critical
    sql_template    TEXT,                      -- filled in by sql_generator
    notes           TEXT,
    active          BOOLEAN NOT NULL DEFAULT true,
    regulation_date DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_assertions_regulation
    ON regulation_assertions (regulation_id);

CREATE INDEX IF NOT EXISTS idx_assertions_check_type
    ON regulation_assertions (check_type);

CREATE INDEX IF NOT EXISTS idx_assertions_materiality
    ON regulation_assertions (materiality)
    WHERE active = true;
