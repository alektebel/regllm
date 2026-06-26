"""Evaluate tool-selection accuracy with ±1 reward on minimal questions.

Uses training/data/tool_selection_manifest.json built from eval_dataset.

Reward: +1 if the agent calls the expected tool(s), -1 otherwise.

Usage:
    python tests/eval_tool_selection.py --model qwen3:4b
    python tests/eval_tool_selection.py --model regllm-4b-sft
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

API = "http://localhost:8000"
MANIFEST = Path(__file__).resolve().parent.parent / "training" / "data" / "tool_selection_manifest.json"
SESSION = "debug_lgd"
TOOLS = ["trace_causal_chain", "get_sas_formula", "trace_dependencies", "inspect_lineage", "search_docs",
         "search_regulation", "get_field_definition"]


def run_question(question: str, model: str) -> list[str]:
    payload = {
        "question": question,
        "model": model,
        "tool_names": TOOLS,
        "max_iters": 3,
        "session_id": SESSION,
    }
    called: list[str] = []
    with httpx.stream("POST", f"{API}/agent/ask", json=payload, timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "tool_call":
                called.append(ev["tool"])
    return called


def score_item(item: dict, called: list[str]) -> tuple[int, str]:
    called_set = set(called)
    if "expected_tools" in item:
        expected = set(item["expected_tools"])
        if expected <= called_set:
            return 1, "ok"
        return -1, f"expected {sorted(expected)}, got {called}"
    expected = item["expected_tool"]
    if expected in called_set:
        return 1, "ok"
    return -1, f"expected {expected}, got {called or '(none)'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="data/tool_selection_results.json")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"Run: python training/build_tool_sft.py  (missing {MANIFEST})", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(MANIFEST.read_text())
    if args.limit:
        manifest = manifest[: args.limit]

    print(f"Tool-selection eval | model={args.model} | n={len(manifest)}")
    print("=" * 60)

    results = []
    total_reward = 0
    by_kind: dict[str, list[int]] = {}

    for i, item in enumerate(manifest):
        q = item["question"]
        t0 = time.time()
        called = run_question(q, args.model)
        reward, reason = score_item(item, called)
        total_reward += reward
        kind = item.get("kind", "?")
        by_kind.setdefault(kind, []).append(reward)

        results.append({
            **item,
            "tools_called": called,
            "reward": reward,
            "reason": reason,
            "elapsed": round(time.time() - t0, 1),
        })
        icon = "+" if reward > 0 else "-"
        print(f"[{i+1}/{len(manifest)}] [{icon}] {reason[:60]}  Q: {q[:50]}...", flush=True)

    n = len(manifest)
    accuracy = (total_reward + n) / (2 * n)  # maps [-1,1] rewards to [0,1]
    print("\n" + "=" * 60)
    print(f"Total reward: {total_reward}/{n}  accuracy: {accuracy*100:.1f}%")
    for kind, rewards in sorted(by_kind.items()):
        acc = sum(1 for r in rewards if r > 0) / len(rewards) * 100
        print(f"  {kind:10s}: {acc:.0f}% ({sum(1 for r in rewards if r>0)}/{len(rewards)})")

    out = Path(args.output)
    out.write_text(json.dumps({
        "model": args.model,
        "n": n,
        "total_reward": total_reward,
        "accuracy_pct": round(accuracy * 100, 1),
        "by_kind": {k: round(sum(1 for r in v if r > 0) / len(v) * 100, 1) for k, v in by_kind.items()},
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
