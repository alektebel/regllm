"""Merge tool-selection + golden traces with the current LITE system prompt.

Usage:
    python training/build_agent_sft.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.prompts import LITE_SYSTEM_PROMPT

SOURCES = [
    PROJECT_ROOT / "training/data/tool_sft_train.jsonl",
    PROJECT_ROOT / "training/data/golden_train.jsonl",
]
OUT_TRAIN = PROJECT_ROOT / "training/data/agent_sft_train.jsonl"
OUT_EVAL = PROJECT_ROOT / "training/data/agent_sft_eval.jsonl"
EVAL_RATIO = 0.12
SEED = 42


def _normalize(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "system":
            m["content"] = LITE_SYSTEM_PROMPT
        out.append(m)
    return out


def main() -> None:
    rows: list[dict] = []
    for path in SOURCES:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ex = json.loads(line)
                ex["messages"] = _normalize(ex["messages"])
                rows.append(ex)
        print(f"  + {path.name}: {sum(1 for _ in path.read_text().splitlines() if _.strip())}")

    random.seed(SEED)
    random.shuffle(rows)
    n_eval = max(8, int(len(rows) * EVAL_RATIO))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    OUT_TRAIN.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    OUT_EVAL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8",
    )
    print(f"→ {OUT_TRAIN} ({len(train_rows)} train, {len(eval_rows)} eval)")


if __name__ == "__main__":
    main()
