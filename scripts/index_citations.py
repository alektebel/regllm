#!/usr/bin/env python3
"""
Build / update the citation vector collection for RegLLM.

Usage:
    python scripts/index_citations.py --rebuild           # Full rebuild from all sources
    python scripts/index_citations.py --add-doc path.json --name "Nueva Normativa"
    python scripts/index_citations.py --status            # Show collection count
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.citation_rag import CitationRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default source paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
CITATION_TREE = PROJECT_ROOT / "data" / "citation_trees" / "eu_regulations.json"
RAW_DATA_GLOB = str(PROJECT_ROOT / "data" / "raw" / "regulation_data_*.json")


def cmd_status(crag: CitationRAG) -> None:
    count = crag.count()
    print(f"Collection '{crag.collection.name}': {count} citation vectors")


def cmd_rebuild(crag: CitationRAG) -> None:
    print("Clearing existing collection...")
    crag.clear()

    # 1. Index tree nodes
    if CITATION_TREE.exists():
        n = crag.index_citation_tree(str(CITATION_TREE))
        print(f"  Tree ({CITATION_TREE.name}): +{n} items")
    else:
        print(f"  WARNING: citation tree not found at {CITATION_TREE}")

    # 2. Index raw regulation docs
    raw_files = sorted(glob.glob(RAW_DATA_GLOB))
    if raw_files:
        for fpath in raw_files:
            name = Path(fpath).stem  # e.g. regulation_data_20260116_211936
            n = crag.index_regulation_doc(fpath, doc_name=name)
            print(f"  Doc ({Path(fpath).name}): +{n} items")
    else:
        print(f"  WARNING: no regulation docs found matching {RAW_DATA_GLOB}")

    print(f"\nDone. Total citation vectors: {crag.count()}")


def cmd_add_doc(crag: CitationRAG, doc_path: str, doc_name: str) -> None:
    n = crag.index_regulation_doc(doc_path, doc_name=doc_name)
    print(f"Added {n} items from '{doc_name}'. Total: {crag.count()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RegLLM citation index builder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rebuild", action="store_true", help="Full rebuild from all sources")
    group.add_argument("--add-doc", metavar="PATH", help="Index a single regulation JSON doc")
    group.add_argument("--status", action="store_true", help="Show collection count")
    parser.add_argument("--name", metavar="NAME", default="", help="Document name (used with --add-doc)")
    parser.add_argument("--chroma-path", default="./vector_db/chroma_db", help="ChromaDB path")

    args = parser.parse_args()

    crag = CitationRAG(chroma_path=args.chroma_path)

    if args.status:
        cmd_status(crag)
    elif args.rebuild:
        cmd_rebuild(crag)
    elif args.add_doc:
        if not args.name:
            args.name = Path(args.add_doc).stem
        cmd_add_doc(crag, args.add_doc, args.name)


if __name__ == "__main__":
    main()
