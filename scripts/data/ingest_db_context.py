"""
Database Context Ingestion Pipeline
=====================================
Extracts column-level context from Excel data dictionaries, SQL DDL files,
and SAS code, then embeds and stores everything in the context_chunks table.

This is the mapping layer: it connects regulation assertions to the actual
schema so the query-generation layer knows which tables/columns to target.

Usage:
    python scripts/data/ingest_db_context.py --excel data/dict/columns.xlsx
    python scripts/data/ingest_db_context.py --sql   path/to/schema.sql
    python scripts/data/ingest_db_context.py --sas   path/to/sas_jobs/
    python scripts/data/ingest_db_context.py --excel ... --sql ... --sas ...
    python scripts/data/ingest_db_context.py --excel ... --dry-run

All flags are optional — supply any combination.

DB connection (env vars, same as API):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ── DB helpers (mirrors rag_system.py pattern) ────────────────────────────────

def _conn_kwargs() -> dict:
    return dict(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def _get_conn():
    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(**_conn_kwargs())
    register_vector(conn)
    return conn


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS context_chunks (
                id        SERIAL PRIMARY KEY,
                content   TEXT NOT NULL,
                metadata  JSONB DEFAULT '{}',
                embedding vector(768)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS context_chunks_embedding_idx
                ON context_chunks USING hnsw (embedding vector_cosine_ops)
        """)
    conn.commit()


# ── Embedding ─────────────────────────────────────────────────────────────────

def _load_embedder():
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name, device="cpu")


def _embed_and_store(
    entries: list[tuple[str, dict]],
    embedder,
    conn,
    dry_run: bool = False,
    batch_size: int = 32,
) -> int:
    """Embed and insert entries into context_chunks. Returns count inserted."""
    if not entries:
        return 0

    import torch
    from psycopg2.extras import execute_values

    total = 0
    for i in range(0, len(entries), batch_size):
        batch = entries[i: i + batch_size]
        texts = [e[0] for e in batch]

        if dry_run:
            for text, meta in batch:
                logger.info("  [dry-run] %s | %s", meta.get("table_name", "?"), meta.get("column_name", "?"))
            total += len(batch)
            continue

        with torch.inference_mode():
            embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=batch_size)

        rows = [
            (text, json.dumps(meta), emb.tolist())
            for (text, meta), emb in zip(batch, embeddings)
        ]
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO context_chunks (content, metadata, embedding) VALUES %s",
                rows,
                template="(%s, %s::jsonb, %s::vector)",
            )
        conn.commit()
        total += len(batch)
        logger.info("  Stored %d/%d entries…", min(i + batch_size, len(entries)), len(entries))

    return total


# ── Source collectors ─────────────────────────────────────────────────────────

def _collect_excel(paths: list[Path]) -> list[tuple[str, dict]]:
    from src.db_context.excel_dict import extract_excel
    results = []
    for p in paths:
        logger.info("Excel: %s", p)
        results.extend(extract_excel(p))
    return results


def _collect_sql(paths: list[Path]) -> list[tuple[str, dict]]:
    from src.db_context.sql_ddl import extract_sql
    results = []
    for p in paths:
        logger.info("SQL: %s", p)
        results.extend(extract_sql(p))
    return results


def _collect_sas(paths: list[Path]) -> list[tuple[str, dict]]:
    from src.db_context.sas_labels import extract_sas_labels
    results = []
    for p in paths:
        logger.info("SAS: %s", p)
        results.extend(extract_sas_labels(p))
    return results


def _resolve_paths(raw: list[str], extensions: set[str]) -> list[Path]:
    """Expand files and directories into a flat list matching the given extensions."""
    found = []
    for r in raw:
        p = Path(r)
        if p.is_dir():
            for ext in extensions:
                found.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file() and p.suffix.lower() in extensions:
            found.append(p)
        else:
            logger.warning("Skipping %s (not a file or directory with matching extension)", r)
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest database context from Excel/SQL/SAS into pgvector")
    parser.add_argument("--excel", nargs="+", metavar="PATH",
                        help=".xlsx/.xls data dictionary files or directories")
    parser.add_argument("--sql", nargs="+", metavar="PATH",
                        help=".sql DDL files or directories")
    parser.add_argument("--sas", nargs="+", metavar="PATH",
                        help=".sas/.egp files or directories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be stored without writing to DB")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size (default: 32)")
    args = parser.parse_args()

    if not any([args.excel, args.sql, args.sas]):
        parser.error("Provide at least one of --excel, --sql, --sas")

    # ── Collect all entries before loading the model ──────────────────────────
    all_entries: list[tuple[str, dict]] = []

    if args.excel:
        paths = _resolve_paths(args.excel, {".xlsx", ".xls"})
        all_entries.extend(_collect_excel(paths))

    if args.sql:
        paths = _resolve_paths(args.sql, {".sql"})
        all_entries.extend(_collect_sql(paths))

    if args.sas:
        paths = _resolve_paths(args.sas, {".sas", ".egp"})
        all_entries.extend(_collect_sas(paths))

    logger.info("Total entries collected: %d", len(all_entries))

    if not all_entries:
        logger.warning("Nothing to ingest.")
        return

    if args.dry_run:
        for text, meta in all_entries:
            logger.info("[dry-run] %s | %s | %s",
                        meta.get("source_type"), meta.get("table_name"), meta.get("column_name"))
        logger.info("Dry run complete — %d entries would be ingested.", len(all_entries))
        return

    # ── Load model + DB after parsing (keeps peak RAM lower) ──────────────────
    import gc
    gc.collect()

    embedder = _load_embedder()
    conn = _get_conn()
    _ensure_table(conn)

    total = _embed_and_store(all_entries, embedder, conn, dry_run=False, batch_size=args.batch_size)
    conn.close()

    logger.info("Done — %d context chunks stored in context_chunks.", total)


if __name__ == "__main__":
    main()
