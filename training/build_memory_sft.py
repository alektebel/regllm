"""Build procedural memory SFT datasets + optional RL prompts JSONL.

Usage:
    python training/build_memory_sft.py --n 400
    python training/build_memory_sft.py --n 200 --phase organize --out training/data/phase2.2_organizing_proc.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_LEARN = """Eres RegLLM en MODO SUEÑO. Procesas sesiones del día y decides qué merece persistirse en el grafo de conocimiento.

Usa save_insight / save_quirk / save_feedback solo para hallazgos accionables (bugs, feedback, quirks).
NO guardes logs triviales, confirmaciones obvias ni ruido operacional."""

SYSTEM_ORGANIZE = """Eres RegLLM en MODO SUEÑO — fase ORGANIZACIÓN. Mantén el grafo limpio y coherente.

Usa merge_nodes (duplicados), add_tags (nodos sin etiquetar), link_nodes (relacionados),
flag_conflict (contradicciones), delete_node (ruido)."""

SYSTEM_CONSOLIDATE = """Eres RegLLM en MODO SUEÑO — fase CONSOLIDACIÓN (EverMemOS-style).

Detecta grupos de insights relacionados (mismo tema/campo, distintos segmentos o matices)
y créales un Concept que comprima el grafo y reduzca entropía.

Usa create_concept(label, definition, covers=[ids], links_to=[campos]) cuando ≥2 insights
compartan tema. NO crees conceptos para un solo insight aislado (sobre-abstracción)."""

SYSTEM_RETRIEVE = """Eres RegLLM en MODO SUEÑO — fase RETRIEVAL. Recupera EXACTAMENTE la experiencia correcta.

Usa search_experience con queries precisas. Distingue segmento, fecha y campo cuando hay entradas similares.
Si existe un Concept consolidado, úsalo como punto de entrada antes de insights individuales."""

_SYSTEM = {
    "learn": SYSTEM_LEARN,
    "organize": SYSTEM_ORGANIZE,
    "consolidate": SYSTEM_CONSOLIDATE,
    "retrieve": SYSTEM_RETRIEVE,
}


def scenario_to_messages(scenario) -> dict:
    sys_prompt = _SYSTEM[scenario.phase]
    golden = "\n".join(scenario.golden_actions) if scenario.golden_actions else ""
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": scenario.user_prompt},
            {"role": "assistant", "content": golden},
        ],
        "id": scenario.scenario_id,
        "phase": {
            "learn": "2.1",
            "organize": "2.2",
            "consolidate": "2.4",
            "retrieve": "2.3",
        }[scenario.phase],
        "memory_phase": scenario.phase,
        "subtype": scenario.subtype,
        "difficulty": scenario.difficulty,
        "ground_truth": {
            "should_save": scenario.should_save,
            "expected_ids": scenario.expected_ids,
            "forbidden_ids": scenario.forbidden_ids,
            "merge_pairs": scenario.merge_pairs,
            "covers_ids": scenario.covers_ids,
            "concept_label": scenario.concept_label,
            "should_create_concept": scenario.should_create_concept,
            "must_contain": scenario.must_contain,
        },
    }


def scenario_to_rl_prompt(scenario) -> dict:
    """Format for future GRPO on memory env."""
    return {
        "id": scenario.scenario_id,
        "memory_phase": scenario.phase,
        "messages": [
            {"role": "system", "content": _SYSTEM[scenario.phase]},
            {"role": "user", "content": scenario.user_prompt},
        ],
        "ground_truth": scenario_to_messages(scenario)["ground_truth"],
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(rows)} examples)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400, help="Total procedural scenarios")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase", choices=["learn", "organize", "consolidate", "retrieve", "all"], default="all")
    p.add_argument("--out", default=None, help="Single output file (overrides per-phase)")
    p.add_argument("--rl-prompts", action="store_true", help="Also write memory_rl_prompts.jsonl")
    args = p.parse_args()

    from training.memory_scenario_generator import MemoryScenarioGenerator

    gen = MemoryScenarioGenerator(seed=args.seed)
    data_dir = PROJECT_ROOT / "training" / "data"

    if args.phase != "all":
        mix = {args.phase: 1.0}
    else:
        mix = {"learn": 0.30, "organize": 0.20, "consolidate": 0.15, "retrieve": 0.35}

    print(f"Generating {args.n} procedural memory scenarios (seed={args.seed})...")
    batch = gen.generate_batch(args.n, mix=mix)

    by_phase = {"learn": [], "organize": [], "consolidate": [], "retrieve": []}
    for sc in batch:
        by_phase[sc.phase].append(scenario_to_messages(sc))

    if args.out:
        write_jsonl(
            [scenario_to_messages(s) for s in batch],
            Path(args.out),
        )
    else:
        mapping = {
            "learn": data_dir / "phase2.1_learning_proc.jsonl",
            "organize": data_dir / "phase2.2_organizing_proc.jsonl",
            "consolidate": data_dir / "phase2.4_consolidating_proc.jsonl",
            "retrieve": data_dir / "phase2.3_retrieval_proc.jsonl",
        }
        for phase, rows in by_phase.items():
            if rows:
                write_jsonl(rows, mapping[phase])

        # Merged file for SFT
        merged = (
            by_phase["learn"]
            + by_phase["organize"]
            + by_phase["consolidate"]
            + by_phase["retrieve"]
        )
        write_jsonl(merged, data_dir / "memory_sft_proc.jsonl")

    if args.rl_prompts:
        rl_rows = [scenario_to_rl_prompt(s) for s in batch]
        write_jsonl(rl_rows, data_dir / "memory_rl_prompts.jsonl")

    # Quick reward sanity on golden
    from training.memory_reward import compute_memory_reward
    scores = [compute_memory_reward("\n".join(s.golden_actions), s).total for s in batch]
    print(f"  Golden mean reward: {sum(scores)/len(scores):.2f}  (expect >0.5)")

    print("Done. Merge into phase SFT with:")
    print("  cat training/data/phase2.*_proc.jsonl >> training/data/phase_sft_train.jsonl")


if __name__ == "__main__":
    main()
