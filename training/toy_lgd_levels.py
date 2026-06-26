"""Curated toy_lgd levels as verifiable MutatedBug instances.

Each level under ``data/sas/toy_lgd/levels/*/`` has a known bug with
ground truth from ``solution.md`` and probe rows from stress tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sas_logic_tree import SASLogicTree, DataStepNode

from training.bug_generator import MutationMeta, MutatedBug

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEVELS_ROOT = PROJECT_ROOT / "data" / "sas" / "toy_lgd" / "levels"

_BASE_PROBE: dict[str, Any] = {
    "ID_CONTR_CICLO_LGD": "CIC_001",
    "ID_CONTRATO": "CONT_001",
    "SEGMENTO": "CORP",
    "COLATERAL_TIPO": "NINGUNA",
    "PD_ESTIMADA": 0.010,
    "LGD_ESTIMADA": 0.40,
    "EAD": 100000,
    "DPDS": 0,
    "STAGE_IFRS9": 1,
    "CURE_FLAG": 0,
    "PROVISION_PERIOD_MONTHS": 12,
    "VENTANA_CALIBRACION_YEARS": 7,
    "VENTANA_OBSERVACION_YEARS": 5,
    "SW_FUSION": 0,
    "ID_FUSION_FINAL": None,
    "PERIODO_OBSERVACION": 201402,
}


@dataclass(frozen=True)
class LevelSpec:
    level_id: str
    mutation_type: str
    step_name: str
    target_var: str
    description: str
    replacements: tuple[tuple[str, str], ...]
    probe_row: dict[str, Any]
    depth: int = 0
    focus_field: str = "ECL"


LEVEL_SPECS: list[LevelSpec] = [
    LevelSpec(
        "01_easy_varswap", "swap_vars", "work.floored", "LGD_ESTIMADA",
        "EAD used instead of LGD_ESTIMADA in HIPOTECA floor condition",
        (("AND EAD < 0.30", "AND LGD_ESTIMADA < 0.30"),),
        {**_BASE_PROBE, "ID_CONTR_CICLO_LGD": "CIC_003", "COLATERAL_TIPO": "HIPOTECA",
         "LGD_ESTIMADA": 0.20, "EAD": 150000},
        focus_field="LGD_ESTIMADA",
    ),
    LevelSpec(
        "02_easy_missing", "remove_guard", "work.ecl", "LGD_ESTIMADA",
        "Missing COALESCE/default for LGD_ESTIMADA on fusion contracts",
        (("MoC = 0.05 * LGD_ESTIMADA;", "MoC = 0.05 * COALESCE(LGD_ESTIMADA, 0.45);"),),
        {**_BASE_PROBE, "ID_CONTR_CICLO_LGD": "CIC_005", "ID_CONTRATO": "CONT_005",
         "LGD_ESTIMADA": None, "EAD": 300000, "SW_FUSION": 1, "ID_FUSION_FINAL": "FUS_7712",
         "PD_ESTIMADA": 0.020, "DPDS": 15},
        focus_field="ECL",
    ),
    LevelSpec(
        "03_easy_boundary", "wrong_op", "work.final", "STAGE_IFRS9",
        "DPDS > 30 instead of DPDS >= 30 in IFRS9 backstop",
        (("DPDS > 30", "DPDS >= 30"),),
        {**_BASE_PROBE, "ID_CONTR_CICLO_LGD": "CIC_007", "SEGMENTO": "RETAIL",
         "COLATERAL_TIPO": "PERSONAL", "LGD_ESTIMADA": 0.35, "EAD": 80000, "DPDS": 30},
        focus_field="STAGE_IFRS9",
    ),
]


def _apply_replacements(code: str, replacements: tuple[tuple[str, str], ...]) -> str:
    out = code
    for old, new in replacements:
        if old not in out:
            raise ValueError(f"replacement not found: {old!r}")
        out = out.replace(old, new, 1)
    return out


def _strip_datalines(code: str) -> str:
    """Remove inline DATALINES blocks — evaluation uses external probe rows."""
    return re.sub(
        r"(\s+DATALINES;.*?)(?=^\S|\Z)",
        "\n    /* DATALINES omitted for RL probe evaluation */\n",
        code,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


def _depth_for_step(nodes: list, step_name: str) -> int:
    data_steps = [n for n in nodes if isinstance(n, DataStepNode)]
    names = [s.output_dataset for s in data_steps]
    if step_name not in names:
        return 0
    idx = names.index(step_name)
    return max(0, len(names) - 1 - idx)


def _diff_fields(correct: dict[str, Any], buggy: dict[str, Any]) -> list[str]:
    keys = set(correct) | set(buggy)
    changed = []
    for k in sorted(keys):
        cv, bv = correct.get(k), buggy.get(k)
        if cv != bv:
            changed.append(k)
    return changed


def load_level_bug(spec: LevelSpec) -> MutatedBug | None:
    """Build a MutatedBug from a curated level directory."""
    sas_path = LEVELS_ROOT / spec.level_id / "lgd.sas"
    if not sas_path.exists():
        return None
    buggy_raw = sas_path.read_text(encoding="utf-8")
    try:
        fixed_raw = _apply_replacements(buggy_raw, spec.replacements)
    except ValueError:
        return None

    buggy_code = _strip_datalines(buggy_raw)
    original_code = _strip_datalines(fixed_raw)
    tree = SASLogicTree()
    buggy_nodes = tree.parse(buggy_code)
    correct_nodes = tree.parse(original_code)
    row = dict(spec.probe_row)

    buggy_trace = tree.evaluate(buggy_nodes, row)
    correct_trace = tree.evaluate(correct_nodes, row)
    buggy_vals = dict(buggy_trace.final)
    correct_vals = dict(correct_trace.final)

    changed = _diff_fields(correct_vals, buggy_vals)
    if not changed and spec.focus_field:
        if correct_vals.get(spec.focus_field) != buggy_vals.get(spec.focus_field):
            changed = [spec.focus_field]
    if not changed:
        return None

    orig_expr = spec.replacements[0][0] if spec.replacements else ""
    mut_expr = spec.replacements[0][1] if spec.replacements else ""
    depth = _depth_for_step(buggy_nodes, spec.step_name)

    return MutatedBug(
        original_code=original_code,
        mutated_code=buggy_code,
        mutation=MutationMeta(
            mutation_type=spec.mutation_type,
            step_name=spec.step_name,
            node_index=0,
            original_expr=orig_expr,
            mutated_expr=mut_expr,
            target_var=spec.target_var,
            description=spec.description,
        ),
        correct_values=correct_vals,
        buggy_values=buggy_vals,
        changed_fields=changed,
        depth=depth or spec.depth,
        source="curated_level",
        level_id=spec.level_id,
    )


class ToyLgdLevelCatalog:
    """All loadable curated levels."""

    def __init__(self) -> None:
        self.levels: list[tuple[str, MutatedBug]] = []
        for spec in LEVEL_SPECS:
            bug = load_level_bug(spec)
            if bug is not None:
                self.levels.append((spec.level_id, bug))

    def sample(self, n: int, seed: int = 0) -> list[tuple[str, MutatedBug]]:
        import random
        if not self.levels:
            return []
        rng = random.Random(seed)
        if n >= len(self.levels):
            out = list(self.levels)
            rng.shuffle(out)
            return out[:n]
        return rng.sample(self.levels, n)

    @property
    def level_ids(self) -> list[str]:
        return [lid for lid, _ in self.levels]


def all_curated_bugs() -> list[MutatedBug]:
    return [bug for _, bug in ToyLgdLevelCatalog().levels]
