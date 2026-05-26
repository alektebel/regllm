"""
Regulation Assertion Extractor
================================
Reads regulation chunks from document_chunks, calls a LOCAL Ollama model
to extract discrete testable assertions, and stores them in
regulation_assertions.

Runs entirely on the client's machine — no data leaves the network.

Usage:
    # Extract assertions from one regulation
    python scripts/data/extract_assertions.py --regulation crr_575_2013

    # Extract from all regulations (slow — use --limit to test first)
    python scripts/data/extract_assertions.py --limit 50

    # Dry run (shows what would be stored, no DB writes)
    python scripts/data/extract_assertions.py --regulation crr_575_2013 --dry-run

    # Use a specific Ollama model
    python scripts/data/extract_assertions.py --regulation crr_575_2013 --model qwen2.5-coder:14b

Prerequisites:
    ollama pull qwen2.5-coder:14b   # or whichever model you choose
    python scripts/data/extract_assertions.py --setup-db  # creates the table

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

# Available source regulation IDs (mirrors ingest_regulations.py)
KNOWN_REGULATIONS = [
    "crr_575_2013", "crr2_2019_876", "crr3_2024_1623",
    "crd4_2013_36", "crd5_2019_878", "crd6_2024_1619",
    "brrd_2014_59", "brrd2_2019_879", "srmr_806_2014", "srmr2_2019_877",
    "dgsd_2014_49", "emir_648_2012", "mifir_600_2014", "mifid2_2014_65",
    "eba_gl_2017_16", "eba_gl_2020_06", "eba_gl_2022_12",
    "eba_gl_2018_10", "eba_gl_2020_05", "eba_gl_2021_07",
]


def _setup_db():
    """Create the regulation_assertions table if it does not exist."""
    import psycopg2
    migration = (PROJECT_ROOT / "infra" / "migrate_assertions.sql").read_text()
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )
    with conn.cursor() as cur:
        cur.execute(migration)
    conn.commit()
    conn.close()
    logger.info("regulation_assertions table ready.")


def _progress(i: int, total: int, running: int) -> None:
    pct = 100 * i // total
    logger.info("[%3d%%] chunk %d/%d — %d assertions so far", pct, i, total, running)


def main():
    parser = argparse.ArgumentParser(
        description="Extract testable assertions from regulation corpus (runs locally via Ollama)"
    )
    parser.add_argument("--regulation", metavar="ID",
                        help=f"Regulation ID to process (one of: {', '.join(KNOWN_REGULATIONS[:5])} …)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max chunks to process (default: 200; use 0 for all)")
    parser.add_argument("--model", default="qwen2.5-coder:14b",
                        help="Ollama model to use (default: qwen2.5-coder:14b)")
    parser.add_argument("--backend", choices=["ollama", "groq"], default="ollama",
                        help="LLM backend (default: ollama — runs locally)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract but do not write to DB")
    parser.add_argument("--setup-db", action="store_true",
                        help="Create the regulation_assertions table and exit")
    parser.add_argument("--list", action="store_true",
                        help="List known regulation IDs and exit")
    args = parser.parse_args()

    if args.list:
        print("Known regulation IDs:")
        for rid in KNOWN_REGULATIONS:
            print(f"  {rid}")
        return

    if args.setup_db:
        _setup_db()
        return

    from src.assertions.extractor import AssertionExtractor, build_llm_fn

    logger.info("Backend: %s | Model: %s", args.backend, args.model)
    if args.backend == "ollama":
        logger.info("Running fully locally — no data leaves this machine.")

    llm_fn = build_llm_fn(backend=args.backend, model=args.model)
    extractor = AssertionExtractor(llm_fn=llm_fn)

    limit = args.limit if args.limit > 0 else 10_000

    total = extractor.run_from_db(
        regulation_id=args.regulation,
        limit=limit,
        dry_run=args.dry_run,
        progress_cb=_progress,
    )

    if args.dry_run:
        logger.info("Dry run complete — %d assertions would be stored.", total)
    else:
        logger.info("Done — %d assertions stored in regulation_assertions.", total)


if __name__ == "__main__":
    main()
