#!/usr/bin/env python3
"""
Import generated QA pairs from JSONL files into PostgreSQL database.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_db_session, QAInteraction


async def import_qa_pairs(
    jsonl_path: Path, category: str = "generated", source: str = "qa_generator"
):
    """Import QA pairs from JSONL file into database."""

    if not jsonl_path.exists():
        logger.error(f"File not found: {jsonl_path}")
        return 0

    count = 0
    skipped = 0

    async with get_db_session() as session:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Extract question and answer from messages format
                    messages = data.get("messages", [])
                    question = None
                    answer = None

                    for msg in messages:
                        if msg.get("role") == "user":
                            question = msg.get("content")
                        elif msg.get("role") == "assistant":
                            answer = msg.get("content")

                    if not question or not answer:
                        logger.warning(
                            f"Line {line_num}: Missing question or answer, skipping"
                        )
                        skipped += 1
                        continue

                    # Extract metadata
                    metadata = data.get("metadata", {})
                    source_file = metadata.get("source_file", source)
                    persona = metadata.get("persona")

                    # Create category from metadata
                    if persona:
                        cat = f"{category}_{persona.lower().replace(' ', '_')}"
                    else:
                        cat = category

                    # Create QA interaction record
                    qa = QAInteraction(
                        question=question,
                        reference_answer=None,  # We could store this separately if needed
                        model_answer=answer,
                        category=cat,
                        source=source_file,
                        metadata=metadata,
                    )

                    session.add(qa)
                    count += 1

                    # Commit in batches of 50
                    if count % 50 == 0:
                        await session.commit()
                        logger.info(f"Imported {count} QA pairs...")

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                    skipped += 1
                except Exception as e:
                    logger.error(f"Line {line_num}: Error importing - {e}")
                    skipped += 1

        # Final commit
        await session.commit()

    logger.info(f"Import complete: {count} pairs imported, {skipped} skipped")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Import QA pairs from JSONL to PostgreSQL"
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to JSONL file with QA pairs",
    )
    parser.add_argument(
        "--category",
        default="generated",
        help="Category for imported QA pairs (default: generated)",
    )
    parser.add_argument(
        "--source",
        default="qa_generator",
        help="Source identifier (default: qa_generator)",
    )

    args = parser.parse_args()

    count = asyncio.run(import_qa_pairs(args.jsonl_file, args.category, args.source))
    logger.info(f"Successfully imported {count} QA pairs from {args.jsonl_file}")


if __name__ == "__main__":
    main()
