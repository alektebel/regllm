#!/usr/bin/env python3
"""Build fixed investigation eval manifest for Phase 1.

Usage:
    PYTHONPATH=. .venv/bin/python training/build_investigation_eval_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "training/data/investigation_eval_manifest.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from training.investigation_scenarios import load_scenarios_from_eval

    scenarios = load_scenarios_from_eval()
    if args.limit:
        scenarios = scenarios[: args.limit]

    rows = []
    for sc in scenarios:
        rows.append({
            "id": sc.scenario_id,
            "scenario_type": sc.scenario_type,
            "source_category": sc.source_category,
            "question": sc.question,
            "required_tools": sc.required_tools,
            "tool_steps": [{"tool": t, "args": a} for t, a in sc.tool_steps],
            "tool_context": sc.tool_context,
            "ground_truth": sc.ground_truth,
            "messages": sc.messages,
            "forbidden_phrases": sc.forbidden_phrases,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["scenario_type"]] = by_type.get(r["scenario_type"], 0) + 1
    print(f"Wrote {len(rows)} scenarios → {out}")
    print(f"  by type: {by_type}")


if __name__ == "__main__":
    main()
