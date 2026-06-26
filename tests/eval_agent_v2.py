"""Evaluate the agent against the golden dataset — v2.

Scoring:
  1. LLM-as-judge (qwen2.5:14b): satisfactory / not_satisfactory / deficient
  2. Embedding similarity (nomic-embed-text): cosine(answer, gold_answer_summary)
  3. Tool recall: fraction of expected tools actually called

Usage:
    python3 tests/eval_agent_v2.py [--model qwen3:4b] [--judge qwen2.5:14b-instruct-q4_K_M] [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

API = "http://localhost:8000"
OLLAMA = "http://localhost:11434"
DATASET = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
TOOLS = ["trace_causal_chain", "get_sas_formula", "trace_dependencies", "inspect_lineage", "search_docs",
         "search_regulation", "get_field_definition"]
SESSION = "debug_lgd"

# ── LLM Judge prompt ────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an expert evaluator for a banking regulation compliance agent.
You will receive:
- The QUESTION the agent was asked
- The EXPECTED answer (gold standard)
- The ACTUAL answer the agent produced
- The TOOLS the agent called vs expected

Your job: rate the agent's answer into exactly one category.

## Categories

**SATISFACTORY**: The answer correctly addresses the question. Key facts,
fields, and regulatory references match the expected answer. Minor phrasing
differences or missing non-critical details are acceptable. Tool usage was
reasonable (called relevant tools even if not exactly the expected set).

**NOT_SATISFACTORY**: The answer is partially correct but has significant
gaps — missing key fields, wrong regulatory references, incomplete lineage
trace, or failed to use the right tools leading to a shallow answer.

**DEFICIENT**: The answer is empty, completely wrong, hallucinated facts
not supported by tool results, or the agent failed to call any tools and
produced no useful information.

## Output

Respond with ONLY a JSON object:
{"verdict": "SATISFACTORY" | "NOT_SATISFACTORY" | "DEFICIENT", "reason": "1-2 sentence explanation"}
"""

JUDGE_USER = """\
QUESTION: {question}

EXPECTED ANSWER: {gold_answer}

ACTUAL ANSWER: {actual_answer}

TOOLS EXPECTED: {tools_expected}
TOOLS CALLED: {tools_called}
"""


# ── Helpers ─────────────────────────────────────────────────────────────────


def ollama_chat(model: str, system: str, user: str, timeout: float = 120) -> str:
    """Single-turn Ollama chat."""
    r = httpx.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_embed(text: str, model: str = "nomic-embed-text") -> list[float]:
    """Get embedding vector from Ollama."""
    r = httpx.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": model, "prompt": text[:2000]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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
    t0 = time.time()

    with httpx.stream(
        "POST", f"{API}/agent/ask",
        json=payload, timeout=300,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
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
    }


def judge_answer(question: str, gold: str, actual: str,
                 tools_expected: list, tools_called: list,
                 judge_model: str) -> dict:
    """Ask the judge model to rate the answer."""
    if not actual or actual.startswith("[ERROR]"):
        return {"verdict": "DEFICIENT", "reason": "Empty or error response"}

    prompt = JUDGE_USER.format(
        question=question,
        gold_answer=gold,
        actual_answer=actual[:1500],
        tools_expected=", ".join(tools_expected),
        tools_called=", ".join(tools_called) if tools_called else "(none)",
    )
    try:
        raw = ollama_chat(judge_model, JUDGE_SYSTEM, prompt)
        # Parse JSON from response — handle markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        verdict = data.get("verdict", "DEFICIENT").upper()
        if verdict not in ("SATISFACTORY", "NOT_SATISFACTORY", "DEFICIENT"):
            verdict = "DEFICIENT"
        return {"verdict": verdict, "reason": data.get("reason", "")}
    except Exception as e:
        return {"verdict": "DEFICIENT", "reason": f"Judge parse error: {e}"}


def score_tools(called: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    called_set = set(called)
    return sum(1 for t in expected if t in called_set) / len(expected)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--judge", default="qwen2.5:14b-instruct-q4_K_M")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="data/eval_results_v2.json")
    parser.add_argument("--reuse", default="", help="Reuse agent answers from previous results JSON")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text())
    if args.limit:
        dataset = dataset[:args.limit]

    # Optionally reuse previous agent runs (skip re-running agent, just re-judge)
    prev_answers: dict[str, dict] = {}
    if args.reuse:
        prev = json.loads(Path(args.reuse).read_text())
        for r in prev.get("results", []):
            prev_answers[r["id"]] = r

    print(f"Agent: {args.model} | Judge: {args.judge} | Examples: {len(dataset)}", flush=True)
    print("=" * 70, flush=True)

    results = []
    verdicts = {"SATISFACTORY": 0, "NOT_SATISFACTORY": 0, "DEFICIENT": 0}
    total_sim = 0.0
    total_tools = 0.0

    for i, ex in enumerate(dataset):
        eid = ex["id"]
        q = ex["question"]
        print(f"\n[{i+1}/{len(dataset)}] {eid} ({ex['difficulty']}) — {q[:55]}...", flush=True)

        # Run agent (or reuse)
        if eid in prev_answers:
            run = {
                "tool_calls": prev_answers[eid]["tools_called"],
                "answer": prev_answers[eid]["answer_snippet"],
                "elapsed": prev_answers[eid].get("elapsed", 0),
            }
            print("  (reusing previous answer)", flush=True)
        else:
            run = run_question(q, args.model)

        # LLM Judge
        verdict = judge_answer(
            q, ex["gold_answer_summary"], run["answer"],
            ex["expected_tools"], run["tool_calls"],
            args.judge,
        )
        verdicts[verdict["verdict"]] += 1

        # Embedding similarity
        if run["answer"] and not run["answer"].startswith("[ERROR]"):
            emb_answer = ollama_embed(run["answer"], args.embed_model)
            emb_gold = ollama_embed(ex["gold_answer_summary"], args.embed_model)
            sim = cosine_sim(emb_answer, emb_gold)
        else:
            sim = 0.0
        total_sim += sim

        # Tool recall
        t_score = score_tools(run["tool_calls"], ex["expected_tools"])
        total_tools += t_score

        result = {
            "id": eid,
            "category": ex["category"],
            "difficulty": ex["difficulty"],
            "question": q,
            "tools_called": run["tool_calls"],
            "tools_expected": ex["expected_tools"],
            "answer_snippet": run["answer"][:500],
            "gold_answer": ex["gold_answer_summary"],
            "verdict": verdict["verdict"],
            "verdict_reason": verdict["reason"],
            "embedding_similarity": round(sim, 3),
            "tool_recall": round(t_score, 2),
            "elapsed": run.get("elapsed", 0),
        }
        results.append(result)

        v = verdict["verdict"]
        icon = {"SATISFACTORY": "+", "NOT_SATISFACTORY": "~", "DEFICIENT": "-"}[v]
        print(f"  [{icon}] {v} (sim={sim:.2f}, tools={t_score:.0%}) — {verdict['reason'][:80]}", flush=True)
        if run["tool_calls"]:
            print(f"  Called: {run['tool_calls']}", flush=True)

    n = len(dataset)
    sat_pct = verdicts["SATISFACTORY"] / n * 100
    nsat_pct = verdicts["NOT_SATISFACTORY"] / n * 100
    def_pct = verdicts["DEFICIENT"] / n * 100

    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS — {args.model} judged by {args.judge}", flush=True)
    print(f"  SATISFACTORY:     {verdicts['SATISFACTORY']:3d}/{n} ({sat_pct:.1f}%)", flush=True)
    print(f"  NOT_SATISFACTORY: {verdicts['NOT_SATISFACTORY']:3d}/{n} ({nsat_pct:.1f}%)", flush=True)
    print(f"  DEFICIENT:        {verdicts['DEFICIENT']:3d}/{n} ({def_pct:.1f}%)", flush=True)
    print(f"  Avg embedding sim: {total_sim/n:.3f}", flush=True)
    print(f"  Avg tool recall:   {total_tools/n:.1%}", flush=True)

    # By category
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    print("\nBy category:", flush=True)
    for c, rs in sorted(cats.items()):
        s = sum(1 for r in rs if r["verdict"] == "SATISFACTORY")
        print(f"  {c:20s} {s}/{len(rs)} satisfactory ({s/len(rs)*100:.0f}%)", flush=True)

    # By difficulty
    diffs: dict[str, list] = {}
    for r in results:
        diffs.setdefault(r["difficulty"], []).append(r)
    print("\nBy difficulty:", flush=True)
    for d in ["easy", "medium", "hard"]:
        if d in diffs:
            rs = diffs[d]
            s = sum(1 for r in rs if r["verdict"] == "SATISFACTORY")
            print(f"  {d:20s} {s}/{len(rs)} satisfactory ({s/len(rs)*100:.0f}%)", flush=True)

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "judge": args.judge,
        "embed_model": args.embed_model,
        "session": SESSION,
        "tools_available": TOOLS,
        "n_examples": n,
        "verdicts": verdicts,
        "satisfactory_pct": round(sat_pct, 1),
        "avg_embedding_similarity": round(total_sim / n, 3),
        "avg_tool_recall": round(total_tools / n, 3),
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
