#!/usr/bin/env python3
"""Evaluate fragile retrieval precision against qwen3:4b via Ollama."""

import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b"
DATASET_PATH = "data/fragile_retrieval_dataset.json"
SYSTEM_PROMPT = (
    "Eres RegLLM, un asistente de validacion regulatoria bancaria. "
    "Responde basandote SOLO en tu conocimiento. Se preciso."
)


def query_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())["response"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def normalized(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def matches_any(response: str, expected_list: list[str]) -> bool:
    resp_norm = normalized(response)
    for exp in expected_list:
        exp_norm = normalized(exp)
        if exp_norm in resp_norm:
            return True
    return False


def evaluate() -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []
    for entry in dataset:
        qid = entry["id"]
        question = entry["question"]
        expected = entry["retrieval_expected"]
        should = expected.get("should_retrieve_exactly", [])
        should_not = expected.get("should_NOT_retrieve", [])

        response = query_ollama(question)

        retrieved_correct = matches_any(response, should)
        retrieved_wrong = matches_any(response, should_not) if should_not else False

        results.append({
            "id": qid,
            "subtype": entry["subtype"],
            "difficulty": entry["difficulty"],
            "question": question,
            "response": response,
            "should_retrieve": should,
            "should_NOT_retrieve": should_not,
            "retrieved_correct": retrieved_correct,
            "retrieved_wrong": retrieved_wrong,
        })

        status = "PASS" if retrieved_correct and not retrieved_wrong else "FAIL"
        print(f"[{status}] {qid} ({entry['subtype']}): correct={retrieved_correct}, wrong={retrieved_wrong}")

    total = len(results)
    passed = sum(1 for r in results if r["retrieved_correct"] and not r["retrieved_wrong"])
    correct_retrieval = sum(1 for r in results if r["retrieved_correct"])
    wrong_retrieval = sum(1 for r in results if r["retrieved_wrong"])

    by_subtype = defaultdict(lambda: {"total": 0, "passed": 0, "correct": 0, "wrong": 0})
    by_difficulty = defaultdict(lambda: {"total": 0, "passed": 0, "correct": 0, "wrong": 0})

    for r in results:
        st = r["subtype"]
        df = r["difficulty"]
        by_subtype[st]["total"] += 1
        by_difficulty[df]["total"] += 1
        if r["retrieved_correct"] and not r["retrieved_wrong"]:
            by_subtype[st]["passed"] += 1
            by_difficulty[df]["passed"] += 1
        if r["retrieved_correct"]:
            by_subtype[st]["correct"] += 1
            by_difficulty[df]["correct"] += 1
        if r["retrieved_wrong"]:
            by_subtype[st]["wrong"] += 1
            by_difficulty[df]["wrong"] += 1

    report = {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0,
        "correct_retrieval_rate": correct_retrieval / total if total else 0,
        "wrong_retrieval_rate": wrong_retrieval / total if total else 0,
        "by_subtype": {k: dict(v) for k, v in sorted(by_subtype.items())},
        "by_difficulty": {k: dict(v) for k, v in sorted(by_difficulty.items())},
        "results": results,
    }

    return report


def print_report(report: dict):
    print(f"\n{'='*60}")
    print(f"FRAGILE RETRIEVAL EVALUATION — {MODEL}")
    print(f"{'='*60}")
    print(f"Total: {report['total']}")
    print(f"Passed: {report['passed']} ({report['pass_rate']:.1%})")
    print(f"Correct retrieval: {report['correct_retrieval_rate']:.1%}")
    print(f"Wrong retrieval: {report['wrong_retrieval_rate']:.1%}")

    print(f"\n{'─'*60}")
    print("By difficulty:")
    for diff, stats in report["by_difficulty"].items():
        pr = stats["passed"] / stats["total"] if stats["total"] else 0
        print(f"  {diff}: {stats['passed']}/{stats['total']} passed ({pr:.1%}), "
              f"correct={stats['correct']}, wrong={stats['wrong']}")

    print(f"\n{'─'*60}")
    print("By subtype:")
    for st, stats in report["by_subtype"].items():
        pr = stats["passed"] / stats["total"] if stats["total"] else 0
        print(f"  {st}: {stats['passed']}/{stats['total']} ({pr:.1%})")

    out_path = "data/eval_fragile_retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    if "--dry" in sys.argv:
        with open(DATASET_PATH) as f:
            dataset = json.load(f)
        entry = dataset[0]
        print(f"Testing: {entry['id']} — {entry['question']}")
        resp = query_ollama(entry["question"])
        print(f"Response: {resp[:300]}...")
    else:
        report = evaluate()
        print_report(report)
