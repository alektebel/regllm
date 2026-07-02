"""Build fixed eval manifest for DQC catalog generation."""

from __future__ import annotations

import json
from pathlib import Path

from training.dq.catalog_prompt import build_catalog_messages
from training.dq.catalog_schema import load_field_universe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = Path(__file__).parent / "data" / "catalog_eval_manifest.jsonl"


def build_manifest(*, compact: bool = False) -> list[dict]:
    from training.dq.catalog_prompt import build_compact_catalog_messages, build_catalog_messages

    universe = load_field_universe()
    messages = build_compact_catalog_messages(universe) if compact else build_catalog_messages(universe)
    row = {
        "id": "eval_dqc_catalog_compact_000" if compact else "eval_dqc_catalog_full_000",
        "task": "dqc_catalog",
        "table": universe.table,
        "pk_column": universe.pk_column,
        "n_auditable_fields": len(universe.auditable_fields),
        "auditable_fields": list(universe.auditable_fields),
        "coherence_groups": [list(g) for g in universe.coherence_groups],
        "messages": messages,
        "ground_truth": {
            "min_coverage": 1.0,
            "min_coherence_per_entry": 0.6,
            "required_entry_keys": ["dqc_id", "descripcion", "campos_entrada"],
        },
    }
    return [row]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true", help="Compact 7-DQC grouped manifest for local models")
    args = parser.parse_args()
    rows = build_manifest(compact=args.compact)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} row(s) to {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
