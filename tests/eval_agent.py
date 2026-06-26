"""Evaluate the agent against the golden dataset.

Usage:
    python tests/eval_agent.py [--model qwen3:4b] [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

API = "http://localhost:8000"
DATASET = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
TOOLS = ["trace_causal_chain", "get_sas_formula", "trace_dependencies", "inspect_lineage", "search_docs",
         "search_regulation", "get_field_definition"]
SESSION = "debug_lgd"


def run_question(question: str, model: str, max_iters: int = 7) -> dict:
    """Call /agent/ask and collect events."""
    payload = {
        "question": question,
        "model": model,
        "tool_names": TOOLS,
        "max_iters": max_iters,
        "session_id": SESSION,
    }
    tool_calls: list[str] = []
    answer = ""
    events: list[dict] = []
    t0 = time.time()

    with httpx.stream(
        "POST", f"{API}/agent/ask",
        json=payload, timeout=300,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            events.append(ev)
            if ev["type"] == "tool_call":
                tool_calls.append(ev["tool"])
            elif ev["type"] == "final":
                answer = ev.get("answer", "")
            elif ev["type"] == "error":
                answer = f"[ERROR] {ev.get('error', '')}"

    return {
        "tool_calls": tool_calls,
        "answer": answer,
        "elapsed": round(time.time() - t0, 1),
        "events": events,
    }


def score_tools(called: list[str], expected: list[str]) -> float:
    """Fraction of expected tools that were called at least once."""
    if not expected:
        return 1.0
    called_set = set(called)
    return sum(1 for t in expected if t in called_set) / len(expected)


def score_contains(answer: str, phrases: list[str]) -> float:
    """Fraction of expected phrases found in the answer (case-insensitive)."""
    if not phrases:
        return 1.0
    answer_upper = answer.upper()
    return sum(1 for p in phrases if p.upper() in answer_upper) / len(phrases)


def score_fields(answer: str, fields: list[str]) -> float:
    """Fraction of expected fields mentioned in the answer."""
    if not fields:
        return 1.0
    answer_upper = answer.upper()
    return sum(1 for f in fields if f.upper() in answer_upper) / len(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--limit", type=int, default=0, help="0 = run all")
    parser.add_argument("--output", default="data/eval_results.json")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text())
    if args.limit:
        dataset = dataset[:args.limit]

    print(f"Model: {args.model} | Examples: {len(dataset)} | Session: {SESSION}")
    print("=" * 70)

    results = []
    totals = {"tools": 0, "contains": 0, "fields": 0, "overall": 0}

    for i, ex in enumerate(dataset):
        eid = ex["id"]
        q = ex["question"]
        print(f"\n[{i+1}/{len(dataset)}] {eid} ({ex['difficulty']}) — {q[:60]}...")

        run = run_question(q, args.model)

        s_tools = score_tools(run["tool_calls"], ex["expected_tools"])
        s_contains = score_contains(run["answer"], ex["expected_answer_contains"])
        s_fields = score_fields(run["answer"], ex["expected_fields_mentioned"])
        s_overall = 0.3 * s_tools + 0.4 * s_contains + 0.3 * s_fields

        totals["tools"] += s_tools
        totals["contains"] += s_contains
        totals["fields"] += s_fields
        totals["overall"] += s_overall

        result = {
            "id": eid,
            "category": ex["category"],
            "difficulty": ex["difficulty"],
            "question": q,
            "tools_called": run["tool_calls"],
            "tools_expected": ex["expected_tools"],
            "answer_snippet": run["answer"][:500],
            "score_tools": round(s_tools, 2),
            "score_contains": round(s_contains, 2),
            "score_fields": round(s_fields, 2),
            "score_overall": round(s_overall, 2),
            "elapsed": run["elapsed"],
        }
        results.append(result)

        print(f"  Tools: {s_tools:.0%}  Contains: {s_contains:.0%}  "
              f"Fields: {s_fields:.0%}  Overall: {s_overall:.0%}  "
              f"({run['elapsed']}s)", flush=True)
        if run["tool_calls"]:
            print(f"  Called: {run['tool_calls']}", flush=True)

    n = len(dataset)
    print("\n" + "=" * 70)
    print(f"RESULTS — {args.model} on {n} examples")
    print(f"  Tools:    {totals['tools']/n:.1%}")
    print(f"  Contains: {totals['contains']/n:.1%}")
    print(f"  Fields:   {totals['fields']/n:.1%}")
    print(f"  Overall:  {totals['overall']/n:.1%}")

    # Breakdown by category
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, []).append(r["score_overall"])
    print("\nBy category:")
    for c, scores in sorted(cats.items()):
        avg = sum(scores) / len(scores)
        print(f"  {c:20s} {avg:.1%} ({len(scores)} examples)")

    # Breakdown by difficulty
    diffs = {}
    for r in results:
        d = r["difficulty"]
        diffs.setdefault(d, []).append(r["score_overall"])
    print("\nBy difficulty:")
    for d in ["easy", "medium", "hard"]:
        if d in diffs:
            avg = sum(diffs[d]) / len(diffs[d])
            print(f"  {d:20s} {avg:.1%} ({len(diffs[d])} examples)")

    # Save results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "session": SESSION,
        "tools_available": TOOLS,
        "n_examples": n,
        "scores": {
            "tools": round(totals["tools"] / n, 3),
            "contains": round(totals["contains"] / n, 3),
            "fields": round(totals["fields"] / n, 3),
            "overall": round(totals["overall"] / n, 3),
        },
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to {out}")


if __name__ == "__main__":
    main()
