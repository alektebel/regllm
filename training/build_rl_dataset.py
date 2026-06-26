"""Build GRPO dataset from toy LGD levels.

Each level becomes a prompt. The model generates investigation traces
and the reward function scores them against the known solution.

For GRPO, we need a dataset of prompts (no completions — those are
generated during training).

Usage:
    .venv/bin/python training/build_rl_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TOY_LGD = PROJECT_ROOT / "data" / "sas" / "toy_lgd"
OUT = PROJECT_ROOT / "training" / "data" / "rl_prompts.jsonl"


def _load_schema_docs() -> str:
    """Load the meta/schema docs as context for the agent."""
    meta_dir = TOY_LGD / "meta"
    if not meta_dir.exists():
        return ""
    parts = []
    for f in sorted(meta_dir.glob("*.md")):
        parts.append(f.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def _build_prompt(level: str, code: str, level_md: str, schema_docs: str) -> list[dict]:
    """Build chat messages for a level investigation."""
    system = (
        "Eres un asistente de validación regulatoria bancaria para reporting COREP/FINREP IRB.\n"
        "DEBES llamar herramientas antes de responder. Nunca respondas solo de memoria.\n\n"
        "## Herramientas disponibles\n\n"
        "- **trace_dependencies**: Cadena de dependencias BFS — \"de qué campos depende X\", fórmulas upstream.\n"
        "- **inspect_lineage**: Dónde se CALCULA un campo en SAS — línea, data step, predecesores directos.\n"
        "- **search_docs**: Documentación — esquemas, changelogs, semántica de campos.\n"
        "- **search_regulation**: Normativa — CRR, EBA, circulares BdE, suelos regulatorios.\n\n"
        "## Reglas\n\n"
        "1. Llama al menos una herramienta por pregunta.\n"
        "2. Para dependencias/cadenas: trace_dependencies. Para ubicación en código: inspect_lineage.\n"
        "3. Basa tu respuesta SOLO en resultados de herramientas. No inventes datos.\n"
        "4. Responde SIEMPRE en español.\n\n"
        "## Esquema de tablas\n\n"
        f"{schema_docs}\n"
    )

    user = (
        f"Analiza el siguiente código SAS LGD. Hay un bug que produce resultados incorrectos. "
        f"Usa las herramientas trace_dependencies e inspect_lineage para investigar. "
        f"Identifica la línea con el error, explica por qué falla, y propón la corrección.\n\n"
        f"```sas\n{code}\n```\n\n"
        f"Contexto:\n{level_md}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _augment_prompt(base_messages: list[dict], variant: int) -> list[dict]:
    """Create prompt variants by rephrasing the question."""
    rephrasings = [
        None,  # original
        "Investiga paso a paso: primero traza las dependencias del campo afectado, luego inspecciona el lineage del código SAS para encontrar el error.",
        "¿Qué está mal en este pipeline LGD? Traza las dependencias y explica la causa raíz del bug.",
        "Revisa el código SAS. Los valores de salida son incorrectos. Usa trace_dependencies e inspect_lineage para diagnosticar.",
    ]
    idx = variant % len(rephrasings)
    if idx == 0 or rephrasings[idx] is None:
        return base_messages

    msgs = list(base_messages)
    msgs[-1] = {**msgs[-1], "content": rephrasings[idx] + "\n\n" + msgs[-1]["content"]}
    return msgs


def main() -> None:
    schema_docs = _load_schema_docs()
    levels_dir = TOY_LGD / "levels"
    levels = sorted(d.name for d in levels_dir.iterdir() if d.is_dir())

    prompts = []
    for level in levels:
        level_dir = levels_dir / level
        code_path = level_dir / "lgd.sas"
        level_md_path = level_dir / "level.md"
        if not code_path.exists() or not level_md_path.exists():
            continue

        code = code_path.read_text(encoding="utf-8")
        level_md = level_md_path.read_text(encoding="utf-8")

        base_msgs = _build_prompt(level, code, level_md, schema_docs)

        # Create 4 prompt variants per level for diversity
        for v in range(4):
            msgs = _augment_prompt(base_msgs, v)
            prompts.append({
                "level": level,
                "variant": v,
                "messages": msgs,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Built {len(prompts)} RL prompts ({len(levels)} levels × 4 variants) → {OUT}")


if __name__ == "__main__":
    main()
