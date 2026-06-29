"""Fixed-manifest evaluation for RL curriculum (stable metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MANIFEST = PROJECT_ROOT / "training/data/rl_eval_manifest.jsonl"


def load_eval_manifest(path: Path | str | None = None) -> list[dict[str, Any]]:
    p = Path(path or DEFAULT_MANIFEST)
    if not p.exists():
        raise FileNotFoundError(
            f"Eval manifest missing: {p}\n"
            "Run: PYTHONPATH=. .venv/bin/python training/build_rl_eval_manifest.py"
        )
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def eval_manifest_rows(
    rows: list[dict[str, Any]],
    completion_fn,
    *,
    stage_idx: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Score manifest rows with a callable(completion_fn(messages)->str)."""
    from training.curriculum import STAGES
    from training.rl_env import compute_reward
    from training.bug_generator import MutatedBug, MutationMeta

    if stage_idx is not None:
        rows = [r for r in rows if r["stage_idx"] == stage_idx]

    rewards: list[float] = []
    details: list[dict] = []

    for row in rows:
        stage = STAGES[row["stage_idx"]]
        gt = row.get("ground_truth", {})
        bug = MutatedBug(
            original_code="",
            mutated_code="",
            mutation=MutationMeta(
                mutation_type=row.get("mutation_type", ""),
                step_name=gt.get("step_name", ""),
                node_index=0,
                original_expr=gt.get("original_expr", ""),
                mutated_expr="",
                target_var=gt.get("target_var", ""),
                description="",
            ),
            correct_values=gt.get("correct_values") or {},
            buggy_values=gt.get("buggy_values") or {},
            changed_fields=row.get("changed_fields") or [],
            depth=0,
            expected_tools=row.get("expected_tools") or [],
            task_kind=row.get("task_kind", ""),
            source=row.get("source", ""),
            level_id=row.get("level_id", ""),
        )
        completion = completion_fn(row["messages"])
        bd = compute_reward(completion, bug, stage)
        rewards.append(bd.total)
        details.append({
            "id": row["id"],
            "reward": bd.total,
            "components": bd.components,
            "tools": bd.tool_calls_found,
        })

    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0
    return {
        "n": n,
        "mean_reward": round(mean, 4),
        "rewards": [round(r, 3) for r in rewards],
        "details": details,
    }


def eval_adapter_manifest(
    adapter_path: str | Path,
    manifest_path: Path | str | None = None,
    *,
    stage_idx: int | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 400,
    temperature: float = 0.0,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """Load adapter and score fixed manifest (deterministic when temperature=0)."""
    from training.tool_utils import load_adapter, release_gpu, generate_text

    rows = load_eval_manifest(manifest_path)
    if stage_idx is not None:
        rows = [r for r in rows if r["stage_idx"] == stage_idx]
    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]

    model, tokenizer = load_adapter(str(adapter_path), max_seq_length=max_seq_length)

    def _complete(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate_text(
            model, tokenizer, prompt=prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

    result = eval_manifest_rows(rows, _complete, stage_idx=None, temperature=temperature)
    del model
    release_gpu()
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Eval adapter on fixed RL manifest")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--stage", type=int, default=None, help="Filter to one stage index")
    parser.add_argument("--output", required=True, help="Write JSON results here")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap manifest rows scored (default: all for stage)")
    args = parser.parse_args()

    result = eval_adapter_manifest(
        args.adapter,
        args.manifest,
        stage_idx=args.stage,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"mean_reward={result['mean_reward']}  n={result['n']}")


if __name__ == "__main__":
    main()
