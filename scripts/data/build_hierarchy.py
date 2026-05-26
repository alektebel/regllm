#!/usr/bin/env python3
"""
Build hierarchical retrieval structures:
  1. Extract structural hierarchy from chunk metadata → document_hierarchy
  2. Seed validation themes → validation_themes
  3. Tag chunks with themes via keyword matching → chunk_themes
  4. (Optional) Embed themes + tag by embedding similarity

Usage:
  python scripts/data/build_hierarchy.py [--embed-themes] [--dsn DSN]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.hierarchy.extractor import HierarchyExtractor
from src.hierarchy.themes import ThemeTagger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_hierarchy")


def default_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "regllm")
    user = os.getenv("POSTGRES_USER", "regllm")
    pw   = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=None,
                        help="PostgreSQL DSN (defaults to env vars)")
    parser.add_argument("--embed-themes", action="store_true",
                        help="Compute embeddings for themes and run similarity tagging")
    parser.add_argument("--skip-structure", action="store_true",
                        help="Skip structural hierarchy extraction")
    parser.add_argument("--skip-themes", action="store_true",
                        help="Skip theme seeding and keyword tagging")
    args = parser.parse_args()

    dsn = args.dsn or default_dsn()
    log.info("Using DSN: %s", dsn.split("@")[-1])

    # ── 1. Structural hierarchy ──────────────────────────────────────────────
    if not args.skip_structure:
        log.info("=== Building structural hierarchy ===")
        extractor = HierarchyExtractor(dsn)
        counts = extractor.build()
        log.info("Hierarchy: %d nodes, %d chunk links", counts["nodes"], counts["links"])
    else:
        log.info("Skipping structural hierarchy (--skip-structure)")

    # ── 2. Validation themes ─────────────────────────────────────────────────
    if not args.skip_themes:
        tagger = ThemeTagger(dsn)

        log.info("=== Seeding validation themes ===")
        tagger.seed_themes()

        log.info("=== Keyword tagging chunks ===")
        n = tagger.tag_all_chunks(min_score=0.33)
        log.info("Keyword tagging: %d (chunk, theme) pairs", n)

        if args.embed_themes:
            log.info("=== Embedding themes ===")
            tagger.embed_themes()

            log.info("=== Embedding-based chunk tagging ===")
            n = tagger.tag_by_embedding(min_score=0.45)
            log.info("Embedding tagging: %d (chunk, theme) pairs", n)
    else:
        log.info("Skipping theme seeding/tagging (--skip-themes)")

    log.info("Done.")


if __name__ == "__main__":
    main()
