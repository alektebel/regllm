#!/usr/bin/env python3
"""Build and persist TabBERT-style embeddings for recuperatory cycles data.

Usage:
    python scripts/train_tabular_embeddings.py \\
        --csv data/samples/recuperatory_cycles.csv \\
        --strategy colon \\
        --sample 1000   # optional: limit rows for testing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("train_tabular_embeddings")

DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "recuperatory_cycles.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TabBERT-style embeddings for recuperatory cycles"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--strategy", type=str, default="colon",
        choices=["flat", "colon", "json", "col_val"],
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N rows for testing")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        raise SystemExit(1)

    from src.embeddings.tabular.embedder import (
        TabularEmbeddingSet,
        build_tabular_embeddings,
    )

    logger.info("Building embeddings from %s (strategy=%s, sample=%s)",
                args.csv, args.strategy, args.sample or "all")

    emb = build_tabular_embeddings(
        csv_path=args.csv,
        strategy=args.strategy,
        sample=args.sample,
    )

    path = emb.save(path=args.output)
    logger.info("Done — %s", emb.summary())
    print(f"Saved {len(emb)} embeddings ({emb.vectors.shape[1]} dim) → {path}")


if __name__ == "__main__":
    main()
