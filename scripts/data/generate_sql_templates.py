"""
SQL Template Generator
======================
Reads assertions from regulation_assertions where sql_template IS NULL,
calls a LOCAL Ollama model (qwen2.5-coder) to generate PostgreSQL compliance
check queries, and writes the results back to the table.

Runs entirely on the client's machine — no data leaves the network.

Prerequisites:
    ollama pull qwen2.5-coder:14b
    python scripts/data/extract_assertions.py --setup-db   # create table
    python scripts/data/ingest_db_context.py --excel ...   # populate context_chunks

Usage:
    # Generate SQL for one regulation (test first)
    python scripts/data/generate_sql_templates.py --regulation crr_575_2013 --limit 10

    # Preview without writing to DB
    python scripts/data/generate_sql_templates.py --regulation crr_575_2013 --dry-run

    # Generate for all regulations (slow — runs one LLM call per assertion)
    python scripts/data/generate_sql_templates.py --limit 0

    # Use a different model
    python scripts/data/generate_sql_templates.py --model qwen2.5-coder:32b

DB connection (env vars):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _get_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def _count_pending(conn, regulation_id: str | None) -> int:
    with conn.cursor() as cur:
        if regulation_id:
            cur.execute(
                "SELECT COUNT(*) FROM regulation_assertions WHERE sql_template IS NULL AND active = true AND regulation_id = %s",
                (regulation_id,),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) FROM regulation_assertions WHERE sql_template IS NULL AND active = true",
            )
        return cur.fetchone()[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL compliance check templates for regulation assertions (runs locally via Ollama)"
    )
    parser.add_argument("--regulation", metavar="ID",
                        help="Regulation ID to process (e.g. crr_575_2013); all if omitted")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max assertions to process (default: 100; use 0 for all)")
    parser.add_argument("--model", default="qwen2.5-coder:14b",
                        help="Ollama model to use (default: qwen2.5-coder:14b)")
    parser.add_argument("--backend", choices=["ollama", "groq"], default="ollama",
                        help="LLM backend (default: ollama — fully local)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show generated SQL but do not write to DB")
    parser.add_argument("--stats", action="store_true",
                        help="Show counts of pending/done and exit")
    args = parser.parse_args()

    conn = _get_conn()

    if args.stats:
        pending = _count_pending(conn, args.regulation)
        with conn.cursor() as cur:
            where = f"AND regulation_id = '{args.regulation}'" if args.regulation else ""
            cur.execute(f"SELECT COUNT(*) FROM regulation_assertions WHERE sql_template IS NOT NULL {where}")
            done = cur.fetchone()[0]
        print(f"Assertions with SQL template : {done}")
        print(f"Assertions pending (no SQL)  : {pending}")
        conn.close()
        return

    from src.assertions.extractor import build_llm_fn
    from src.assertions.sql_generator import SQLGenerator

    logger.info("Backend: %s | Model: %s", args.backend, args.model)
    if args.backend == "ollama":
        logger.info("Running fully locally — no data leaves this machine.")

    llm_fn = build_llm_fn(backend=args.backend, model=args.model)
    generator = SQLGenerator(llm_fn=llm_fn)

    limit = args.limit if args.limit > 0 else 10_000

    n = generator.run(
        conn=conn,
        regulation_id=args.regulation,
        limit=limit,
        dry_run=args.dry_run,
    )

    conn.close()

    if args.dry_run:
        logger.info("Dry run complete — %d SQL templates would be generated.", n)
    else:
        logger.info("Done — %d SQL templates written to regulation_assertions.sql_template.", n)


if __name__ == "__main__":
    main()
