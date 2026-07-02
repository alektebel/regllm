"""Coherence rule cards + toy DB loader for DQ-check generation RL.

A *coherence rule* is a natural-language description of a database
invariant that spans >= 2 fields/entities, plus a deterministic ``mutate``
function that introduces the violation. Mutation testing is what makes
the reward verifiable without an LLM judge (see ``dq_reward.py``).

Every rule in :data:`RULES` is empirically verified to return 0
violations on ``data/samples/recuperatory_cycles.csv`` (22K rows), so
``r_clean_zero`` is achievable. Rules whose ``visible=False`` are the
"ocultos" — internal-only checks not surfaced in the validation UI.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "recuperatory_cycles.csv"
TABLE_NAME = "recuperatory_cycles"
PK_COLUMN = "ID_CONTR_CICLO_LGD"


@dataclass(frozen=True)
class CoherenceRule:
    rule_id: str
    category: str                                   # cross_field | cross_table | cardinality
    description: str                                # natural-language spec shown to the model
    table: str
    columns: tuple[str, ...]                        # >=2 columns the predicate must combine
    severity: str                                   # HIGH | MED | LOW
    visible: bool = True                            # False = "oculto" (internal-only)
    regulation_ref: str = ""
    mutate: Callable[[dict], dict] = field(default=lambda r: dict(r), repr=False)


# ── Mutators — each injects the specific incoherence for mutation testing ──

def _valor_sin_flag(row: dict) -> dict:
    r = dict(row); r["ADJUDICACION_VALOR"] = 50000.0; r["ADJUDICACION_FLAG"] = "0"; return r

def _flag_sin_valor(row: dict) -> dict:
    r = dict(row); r["ADJUDICACION_FLAG"] = "1"; r["ADJUDICACION_VALOR"] = 0.0; return r

def _valor_sin_tipo(row: dict) -> dict:
    r = dict(row); r["ADJUDICACION_VALOR"] = 12345.0; r["ADJUDICACION_TIPO"] = ""; return r

def _tipo_sin_valor(row: dict) -> dict:
    r = dict(row); r["ADJUDICACION_TIPO"] = "SUBASTA"; r["ADJUDICACION_VALOR"] = 0.0; return r

def _stage3_low_dpd(row: dict) -> dict:
    r = dict(row); r["STAGE_IFRS9"] = "3"; r["DPDS"] = 5; return r

def _recuperacion_sobre_ead(row: dict) -> dict:
    r = dict(row)
    ead = float(row.get("EAD") or 0)
    r["RECUPERACION_ACUMULADA"] = ead * 1.8 if ead > 0 else 1_000_000.0
    r["COSTE_TOTAL_ACUMULADO"] = 0.0
    return r

def _cerrado_sin_terminacion(row: dict) -> dict:
    r = dict(row); r["ESTADO_CICLO"] = "CERRADO"; r["TERMINACION"] = ""; return r

# Single-column decoys (NOT coherence — used as negative examples so the
# model learns that a 1-column range check scores r_coherence=0).
def _decoy_lgd_neg(row: dict) -> dict:
    r = dict(row); r["LGD_REALIZADA"] = -0.5; return r

def _decoy_ead_cero(row: dict) -> dict:
    r = dict(row); r["EAD"] = 0.0; return r


# ── Rule registry ───────────────────────────────────────────────────────

RULES: list[CoherenceRule] = [
    CoherenceRule(
        rule_id="dq_coh_adjudicacion_valor_sin_flag",
        category="cross_field",
        description=(
            "Ciclo con valor de adjudicación estrictamente positivo (> 0) cuyo flag "
            "ADJUDICACION_FLAG NO está activo (= '0'). Coherencia: si existió una "
            "adjudicación, el flag debe informarse a '1'."
        ),
        table=TABLE_NAME, columns=("ADJUDICACION_VALOR", "ADJUDICACION_FLAG"),
        severity="HIGH", visible=True,
        regulation_ref="Circular 4/2017 BdE — consistencia de altas/bajas",
        mutate=_valor_sin_flag,
    ),
    CoherenceRule(
        rule_id="dq_coh_adjudicacion_flag_sin_valor",
        category="cross_field",
        description=(
            "Ciclo con ADJUDICACION_FLAG = '1' pero ADJUDICACION_VALOR = 0. "
            "Coherencia inversa: flag activo exige valor positivo asociado."
        ),
        table=TABLE_NAME, columns=("ADJUDICACION_FLAG", "ADJUDICACION_VALOR"),
        severity="MED", visible=True, mutate=_flag_sin_valor,
    ),
    CoherenceRule(
        rule_id="dq_coh_adjudicacion_valor_sin_tipo",
        category="cross_field",
        description=(
            "Ciclo con ADJUDICACION_VALOR > 0 cuyo ADJUDICACION_TIPO está vacío. "
            "Coherencia documental: toda adjudicación con valor requiere tipo."
        ),
        table=TABLE_NAME, columns=("ADJUDICACION_VALOR", "ADJUDICACION_TIPO"),
        severity="HIGH", visible=True, mutate=_valor_sin_tipo,
    ),
    CoherenceRule(
        rule_id="dq_coh_adjudicacion_tipo_sin_valor",
        category="cross_field",
        description=(
            "Ciclo con ADJUDICACION_TIPO informado pero ADJUDICACION_VALOR = 0. "
            "Coherencia: tipo informado sin valor asociado es incoherente."
        ),
        table=TABLE_NAME, columns=("ADJUDICACION_TIPO", "ADJUDICACION_VALOR"),
        severity="MED", visible=True, mutate=_tipo_sin_valor,
    ),
    CoherenceRule(
        rule_id="dq_coh_stage3_sin_dpd_suficiente",
        category="cross_field",
        description=(
            "Ciclo en STAGE_IFRS9 = '3' (default) con DPDS < 30 días. Coherencia "
            "regulatoria: la clasificación en default exige ≥ 30 días de mora."
        ),
        table=TABLE_NAME, columns=("STAGE_IFRS9", "DPDS"),
        severity="HIGH", visible=False,                       # oculto
        regulation_ref="CRR Art. 178.1(b)",
        mutate=_stage3_low_dpd,
    ),
    CoherenceRule(
        rule_id="dq_coh_recuperacion_mayor_exposicion",
        category="cross_field",
        description=(
            "Ciclo cuya (RECUPERACION_ACUMULADA + COSTE_TOTAL_ACUMULADO) supera el "
            "150% del EAD. Incoherencia: no se puede recuperar sustancialmente más "
            "de lo expuesto sin que medie un error de captura."
        ),
        table=TABLE_NAME,
        columns=("RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO", "EAD"),
        severity="MED", visible=True, mutate=_recuperacion_sobre_ead,
    ),
    CoherenceRule(
        rule_id="dq_coh_cerrado_sin_terminacion",
        category="cross_field",
        description=(
            "Ciclo con ESTADO_CICLO = 'CERRADO' y TERMINACION vacía. Coherencia: "
            "todo ciclo cerrado debe informar la causa de terminación."
        ),
        table=TABLE_NAME, columns=("ESTADO_CICLO", "TERMINACION"),
        severity="HIGH", visible=True, mutate=_cerrado_sin_terminacion,
    ),
    # ── Decoys (single-column — should score r_coherence = 0) ─────────────
    CoherenceRule(
        rule_id="dq_decoy_lgd_realizada_negativa",
        category="cross_field",                          # intentionally mislabeled
        description="LGD_REALIZADA fuera de rango [0,1] (valor negativo).",
        table=TABLE_NAME, columns=("LGD_REALIZADA",),
        severity="MED", visible=True, mutate=_decoy_lgd_neg,
    ),
    CoherenceRule(
        rule_id="dq_decoy_ead_cero",
        category="cross_field",
        description="EAD menor o igual a cero en un ciclo activo.",
        table=TABLE_NAME, columns=("EAD",),
        severity="LOW", visible=True, mutate=_decoy_ead_cero,
    ),
]

RULES_BY_ID: dict[str, CoherenceRule] = {r.rule_id: r for r in RULES}
VISIBLE_RULES: list[CoherenceRule] = [r for r in RULES if r.visible]
OCULTO_RULES: list[CoherenceRule] = [r for r in RULES if not r.visible]


# ── Toy DB ──────────────────────────────────────────────────────────────
# Type classification — keep categorical flags as TEXT so comparisons
# behave identically in SQLite and SAS PROC SQL ('0' vs 0 matters in SAS).
_REAL_COLS = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "LGD_REALIZADA", "EAD", "ECL", "PROVISION",
    "OR_DISPTO", "VALOR_COLATERAL_INICIAL", "VALOR_COLATERAL", "LTV",
    "SALDO_PENDIENTE", "INTERESES_ACUMULADOS", "FLUJO_RECUPERADO",
    "COSTE_RECUPERACION", "FLUJO_NETO", "RECUPERACION_ACUMULADA",
    "COSTE_TOTAL_ACUMULADO", "ADJUDICACION_VALOR", "TASA_DESCUENTO", "RWA",
    "TIPO_INTERES_ORIGINAL", "TIPO_INTERES_ACTUAL",
}
_INT_COLS = {"MES_CICLO", "RATING_GRADO", "DPDS"}
# STAGE_IFRS9, CURE_FLAG, ADJUDICACION_FLAG → TEXT (portable across dialects)


def _coerce(col: str, raw: str) -> Any:
    if raw is None or raw == "":
        return None
    if col in _REAL_COLS:
        try:
            return float(raw)
        except ValueError:
            return raw
    if col in _INT_COLS:
        try:
            return int(float(raw))
        except ValueError:
            return raw
    return raw


def load_toy_db(csv_path: Path = DEFAULT_CSV, table: str = TABLE_NAME) -> sqlite3.Connection:
    """Load the toy CSV into an in-memory SQLite DB with proper types.

    NULLs preserved (empty CSV cell → NULL) so SQL ``IS NULL`` works as in
    production. ``check_same_thread=False`` because TRL may call the reward
    function from worker threads.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        defs = []
        for c in cols:
            if c in _INT_COLS:
                defs.append(f'"{c}" INTEGER')
            elif c in _REAL_COLS:
                defs.append(f'"{c}" REAL')
            else:
                defs.append(f'"{c}" TEXT')
        conn.execute(f'CREATE TABLE "{table}" ({", ".join(defs)})')
        placeholders = ", ".join(["?"] * len(cols))
        ins = f'INSERT INTO "{table}" VALUES ({placeholders})'
        batch: list[list[Any]] = []
        for raw_row in reader:
            batch.append([_coerce(c, raw_row.get(c)) for c in cols])
            if len(batch) >= 1000:
                conn.executemany(ins, batch); batch.clear()
        if batch:
            conn.executemany(ins, batch)
    conn.commit()
    return conn
