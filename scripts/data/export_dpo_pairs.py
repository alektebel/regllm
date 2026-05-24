"""
Export DPO training pairs from user feedback.

Reads thumbs-up / thumbs-down ratings stored in PostgreSQL and produces
a JSONL file suitable for dpo_trainer.py:

  {"prompt": "...", "chosen": "...", "rejected": "..."}

Strategy:
  - thumbs_up   → chosen response
  - thumbs_down → rejected response
  Pairs are matched by question similarity (same question, different rating).
  Unmatched feedback is skipped.

Usage:
    python scripts/export_dpo_pairs.py [--out data/finetuning/dpo_pairs.jsonl]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


async def _fetch_feedback() -> list[dict]:
    from src.db import get_session, init_db
    from sqlalchemy import text

    await init_db()
    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT f.rating, f.question_text, q.model_answer "
                "FROM user_feedback f "
                "LEFT JOIN qa_interactions q ON q.id = f.question_id "
                "WHERE f.question_text IS NOT NULL "
                "  AND q.model_answer IS NOT NULL "
                "ORDER BY f.created_at DESC"
            )
        )
        return [dict(r) for r in result.mappings().all()]


def _build_pairs(rows: list[dict]) -> list[dict]:
    """
    Match chosen/rejected answers for the same question.
    Groups by question_text (exact match; close-enough for short feedback strings).
    """
    by_question: dict[str, dict] = {}

    for row in rows:
        q = (row["question_text"] or "").strip()
        if not q:
            continue
        entry = by_question.setdefault(q, {"chosen": None, "rejected": None})
        if row["rating"] == "thumbs_up" and entry["chosen"] is None:
            entry["chosen"] = row["model_answer"]
        elif row["rating"] == "thumbs_down" and entry["rejected"] is None:
            entry["rejected"] = row["model_answer"]

    pairs = []
    for question, answers in by_question.items():
        if answers["chosen"] and answers["rejected"]:
            pairs.append(
                {
                    "prompt": question,
                    "chosen": answers["chosen"],
                    "rejected": answers["rejected"],
                }
            )

    return pairs


async def export(out_path: Path) -> int:
    rows = await _fetch_feedback()
    logger.info(f"Fetched {len(rows)} feedback rows")

    pairs = _build_pairs(rows)
    logger.info(f"Built {len(pairs)} DPO pairs")

    if not pairs:
        logger.warning(
            "No pairs generated. Need at least one thumbs-up AND one thumbs-down "
            "for the same question."
        )
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(pairs)} pairs → {out_path}")
    return len(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "data/finetuning/dpo_pairs.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()
    n = asyncio.run(export(Path(args.out)))
    if n == 0:
        sys.exit(1)
    logger.info(f"Done. Run: python -m src.rlhf.dpo_trainer to fine-tune.")
