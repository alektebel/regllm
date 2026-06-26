#!/usr/bin/env python3
"""Rebuild data + continue SFT from iter1 with anti-hallucination investigation examples.

Usage:
    .venv/bin/python training/train_antihalluc.py
    .venv/bin/python training/train_antihalluc.py --skip-train  # data only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_ADAPTER = PROJECT_ROOT / "training/output/tool-sft-iter1"
OUT_ADAPTER = PROJECT_ROOT / "training/output/tool-sft-antihalluc"
TRAIN_MERGED = PROJECT_ROOT / "training/data/antihalluc_train.jsonl"
EVAL_MERGED = PROJECT_ROOT / "training/data/antihalluc_eval.jsonl"
FAILURE = PROJECT_ROOT / "training/output/moc_ecl_failure.json"


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
                print(f"  unloaded {name}", flush=True)
    except Exception as e:
        print(f"  (gpu cleanup: {e})", flush=True)


def _merge_jsonl(sources: list[Path], out_train: Path, out_eval: Path, eval_ratio: float = 0.1) -> int:
    import random
    rows: list[dict] = []
    for src in sources:
        if not src.exists():
            continue
        with open(src) as f:
            for line in f:
                rows.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_ratio))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    out_train.parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(out_train, train_rows), (out_eval, eval_rows)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-adapter", default=str(BASE_ADAPTER))
    parser.add_argument("--output", default=str(OUT_ADAPTER))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args()

    py = sys.executable
    n = 0
    if not args.skip_data:
        FAILURE.parent.mkdir(parents=True, exist_ok=True)
        FAILURE.write_text(json.dumps([{
            "source_id": "eval_020",
            "question": (
                "¿Por qué algunos ciclos tienen MoC y ECL missing en la salida? "
                "Investiga paso a paso: traza dependencias, revisa el código SAS y explica la causa raíz."
            ),
            "expected_tools": ["trace_dependencies", "inspect_lineage"],
            "tools_called": ["search_experience"],
            "kind": "investigation",
        }], indent=2, ensure_ascii=False))

        print(">>> Building tool SFT + failure replay...", flush=True)
        subprocess.run([
            py, "training/build_tool_sft.py",
            "--failure-file", str(FAILURE),
            "--failure-repeats", "15",
        ], cwd=PROJECT_ROOT, check=True)

        print(">>> Building investigation / anti-hallucination SFT...", flush=True)
        subprocess.run([py, "training/build_investigation_sft.py"], cwd=PROJECT_ROOT, check=True)

        n = _merge_jsonl([
            PROJECT_ROOT / "training/data/tool_sft_train.jsonl",
            PROJECT_ROOT / "training/data/investigation_sft.jsonl",
        ], TRAIN_MERGED, EVAL_MERGED)
        print(f"Merged {n} training rows → {TRAIN_MERGED}", flush=True)
    elif TRAIN_MERGED.exists():
        n = sum(1 for _ in open(TRAIN_MERGED))

    if args.skip_train:
        return

    _free_gpu()
    from training.sft_run import run_sft
    from training.tool_utils import eval_adapter
    base = args.resume_from or args.base_adapter
    adapter = run_sft(
        adapter_in=base,
        output_dir=args.output,
        train_file="training/data/antihalluc_train.jsonl",
        eval_file="training/data/antihalluc_eval.jsonl",
        epochs=args.epochs,
        max_seq_length=1536,
        eval_strategy="no",
    )
    print("\n>>> Evaluating tool selection...", flush=True)
    result = eval_adapter(adapter, verbose=True)
    print(f"\nAccuracy: {result['accuracy_pct']}%  by_kind={result['by_kind']}")

    inv_manifest = PROJECT_ROOT / "training/data/investigation_eval_manifest.json"
    inv_manifest.write_text(json.dumps([
        {"question": q, "expected_tools": ["trace_dependencies", "inspect_lineage"], "kind": "investigation"}
        for q in [
            "¿Por qué MoC y ECL salen missing? Trazar dependencias y explicar causa raíz.",
            "Investiga por qué ECL y MoC aparecen como missing en algunos ciclos del pipeline LGD.",
        ]
    ], indent=2, ensure_ascii=False))
    inv_result = eval_adapter(adapter, inv_manifest, verbose=True)
    print(f"Investigation tool accuracy: {inv_result['accuracy_pct']}%")

    log = PROJECT_ROOT / "training/output/iterate_log.json"
    history = json.loads(log.read_text()) if log.exists() else []
    history.append({
        "iteration": "antihalluc",
        "adapter": adapter,
        "accuracy_pct": result["accuracy_pct"],
        "investigation_pct": inv_result["accuracy_pct"],
        "note": f"Anti-hallucination SFT, {n} merged examples, {args.epochs} epochs",
    })
    log.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    (PROJECT_ROOT / "training/output/best_adapter.txt").write_text(adapter)
    print(f"\nDone. Adapter: {adapter}")


if __name__ == "__main__":
    main()
