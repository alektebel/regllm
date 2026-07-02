#!/usr/bin/env python3
"""Eval harness for LLM-generated DQC catalogs.

Scores a JSON catalog for:
  - 100% field coverage (campos_entrada union vs auditable universe)
  - Per-entry field/SQL coherence

Usage:
  # Score a saved catalog JSON
  PYTHONPATH=. .venv/bin/python training/dq/dq_catalog_eval.py \\
      --catalog path/to/catalog.json

  # Generate via LLM then score
  PYTHONPATH=. .venv/bin/python training/dq/dq_catalog_eval.py --generate

  # Regenerate manifest
  PYTHONPATH=. .venv/bin/python training/dq/build_catalog_eval_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST = Path(__file__).parent / "data" / "catalog_eval_manifest.jsonl"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest missing: {path}\n"
            "Run: PYTHONPATH=. .venv/bin/python training/dq/build_catalog_eval_manifest.py"
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _default_completion_fn(messages: list[dict]) -> str:
    from src.knowledge import get_client
    from src.knowledge.llm_client import _safe_json

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    client = get_client()
    resp = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=16384,
        json_mode=False,
    )
    parsed = _safe_json(resp.text)
    if isinstance(parsed, dict) and ("dqcs" in parsed or "checks" in parsed):
        return json.dumps(parsed, ensure_ascii=False)
    return resp.text


def eval_catalog_row(
    row: dict[str, Any],
    completion_fn: Callable[[list[dict]], str],
    *,
    min_coverage: float = 1.0,
    use_judge: bool = False,
) -> dict[str, Any]:
    from training.dq.catalog_reward import score_catalog_text
    from training.dq.catalog_schema import extract_catalog, load_field_universe

    messages = row["messages"]
    raw = completion_fn(messages)
    universe = load_field_universe()
    breakdown = score_catalog_text(raw, universe, min_coverage=min_coverage)
    catalog = extract_catalog(raw)

    gt = row.get("ground_truth", {})
    passed = (
        breakdown.components.get("r_coverage", 0) >= gt.get("min_coverage", min_coverage)
        and breakdown.components.get("r_coherence", 0) >= gt.get("min_coherence_per_entry", 0.5)
        and breakdown.components.get("r_schema", 0) >= 0.95
    )

    result: dict[str, Any] = {
        "id": row["id"],
        "passed": passed,
        "raw_completion_chars": len(raw),
        "breakdown": breakdown.to_dict(),
        "catalog_preview": {
            "n_dqcs": breakdown.n_dqcs,
            "sample_ids": [e.dqc_id for e in (catalog.dqcs[:5] if catalog else [])],
        },
    }

    if use_judge:
        from training.dq.catalog_judge import judge_catalog

        judge = judge_catalog(catalog, breakdown.to_dict(), catalog_json=raw)
        result["judge"] = judge.to_dict()
        result["judge_score"] = judge.score

    return result


def eval_manifest(
    rows: list[dict[str, Any]],
    completion_fn: Callable[[list[dict]], str],
    *,
    min_coverage: float = 1.0,
    use_judge: bool = False,
) -> dict[str, Any]:
    results = [
        eval_catalog_row(r, completion_fn, min_coverage=min_coverage, use_judge=use_judge)
        for r in rows
    ]
    n_pass = sum(1 for r in results if r["passed"])
    means = {}
    if results:
        for key in ("r_parse", "r_schema", "r_coverage", "r_coherence", "r_groups"):
            vals = [r["breakdown"]["components"].get(key, 0) for r in results]
            means[key] = round(sum(vals) / len(vals), 4)

    report: dict[str, Any] = {
        "n_rows": len(results),
        "n_passed": n_pass,
        "pass_rate": round(n_pass / len(results), 4) if results else 0.0,
        "mean_components": means,
        "mean_total": round(
            sum(r["breakdown"]["total"] for r in results) / len(results), 4
        ) if results else 0.0,
        "results": results,
    }

    if use_judge:
        scores = [r.get("judge_score", 0) for r in results if "judge_score" in r]
        report["mean_judge_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0

    return report


def score_file(catalog_path: Path, *, min_coverage: float = 1.0, use_judge: bool = False) -> dict[str, Any]:
    from training.dq.catalog_reward import score_catalog
    from training.dq.catalog_schema import extract_catalog, load_field_universe

    text = catalog_path.read_text(encoding="utf-8")
    catalog = extract_catalog(text)
    breakdown = score_catalog(catalog, load_field_universe(), min_coverage=min_coverage)
    out = breakdown.to_dict()
    out["source"] = str(catalog_path)
    out["passed"] = (
        breakdown.components.get("r_coverage", 0) >= min_coverage
        and breakdown.missing_fields == []
    )

    if use_judge:
        from training.dq.catalog_judge import judge_catalog

        judge = judge_catalog(catalog, out, catalog_json=text)
        out["judge"] = judge.to_dict()
        out["judge_score"] = judge.score

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LLM-generated DQC catalogs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog", type=Path, help="Score an existing catalog JSON file")
    parser.add_argument("--compact", action="store_true", help="Use compact grouped catalog manifest")
    parser.add_argument("--generate", action="store_true", help="Call LLM using manifest prompts")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge 0–100 rubric on output")
    parser.add_argument("--output", type=Path, help="Write results JSON here")
    parser.add_argument("--save-catalog", type=Path, help="Save raw generated catalog JSON")
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--row-id", type=str, help="Filter manifest to one row id")
    args = parser.parse_args()

    if args.catalog:
        report = score_file(args.catalog, min_coverage=args.min_coverage, use_judge=args.judge)
    elif args.generate:
        if args.compact:
            from training.dq.build_catalog_eval_manifest import build_manifest

            rows = build_manifest(compact=True)
        else:
            rows = load_manifest(args.manifest)
        if args.row_id:
            rows = [r for r in rows if r["id"] == args.row_id]

        def _gen_and_maybe_save(messages: list[dict]) -> str:
            raw = _default_completion_fn(messages)
            if args.save_catalog:
                args.save_catalog.parent.mkdir(parents=True, exist_ok=True)
                args.save_catalog.write_text(raw, encoding="utf-8")
            return raw

        report = eval_manifest(
            rows, _gen_and_maybe_save, min_coverage=args.min_coverage, use_judge=args.judge,
        )
        if args.save_catalog:
            report["catalog_path"] = str(args.save_catalog)
    else:
        parser.error("Provide --catalog PATH or --generate")

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)

    if args.catalog:
        return 0 if report.get("passed") else 1
    return 0 if report.get("pass_rate", 0) >= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
