"""Ground-truth defect catalog for the DQC eval harness.

Each ``Defect`` pairs a *coherence invariant* (what a good DQC must verify)
with a deterministic ``mutate(row)`` that plants exactly that incoherence into
one clean row, AND a ``dimension`` tag from the industry data-quality
taxonomies (DAMA / ISO 8000 / BCBS 239). The catalog is the oracle the eval
harness scores against:

  * **recall**    — does the agent's SQL return >=1 row on the mutated row?
  * **precision** — does the agent's SQL return 0 rows on the clean DB?
  * **oracle_sql** — the reference check that catches the defect (sanity).

Dimension coverage (so the harness can surface *model deficiencies* by area):

  completeness | validity | accuracy | consistency | timeliness
  uniqueness   | plausibility | conformity

Every oracle is verified by the harness self-test to return 0 rows on a clean
DB (built by ``generate_db.build_clean_db``) and >=1 row on its own trap.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

TABLE_NAME = "ciclos_calibrados"
PK_COLUMN = "ID_CONTR_CICLO_LGD"

# DQ dimensions (DAMA / ISO 8000 / BCBS 239)
DIMENSIONS = (
    "completeness", "validity", "accuracy", "consistency", "timeliness",
    "uniqueness", "plausibility", "conformity",
)

# Column type affinity used by generate_db when materializing SQLite.
REAL_COLS = {
    "PD_ESTIMADA", "PD_SUELO", "PD_FINAL", "PD_DOWNTURN",
    "LGD_ESTIMADA", "LGD_REALIZADA", "LGD_SUELO", "MOC", "LGD_CON_MOC", "LGD_FINAL",
    "OR_EAD", "OR_DISPTO", "OR_DISBLE", "SALDO_PENDIENTE", "EAD",
    "EAD_BALANCE", "CCF_ESTIMADO", "EAD_FUERA_BALANCE", "EAD_TOTAL",
    "VALOR_COLATERAL_INICIAL", "VALOR_COLATERAL", "HAIRCUT", "LTV",
    "K_IRB", "M_VENCIMIENTO", "RWA", "ECL", "PROVISION",
    "TIPO_INTERES_ORIGINAL", "TIPO_INTERES_ACTUAL", "INTERESES_ACUMULADOS",
    "ADJUDICACION_VALOR", "RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO",
    "TASA_DESCUENTO",
}
INT_COLS = {
    "MES_CICLO", "RATING_GRADO", "DPDS", "STAGE_IFRS9", "CURE_FLAG", "SW_FUSION",
}

FORMULA_EPS = 0.01  # tolerance for floating formula invariants
ALLOWED_SEGMENTS = ("CORP", "SME", "RETAIL_HIP", "RETAIL_CONS")
REFERENCE_PERIOD = 202412  # data beyond this is "future"/stale


@dataclass(frozen=True)
class Defect:
    defect_id: str
    dimension: str        # one of DIMENSIONS
    category: str         # formula | consistencia | referencial | cross_table | rango
    severity: str         # HIGH | MED | LOW
    description: str      # natural-language spec shown to the agent
    columns: tuple[str, ...]
    oracle_sql: str       # the reference check — returns violating rows
    mutate: Callable[[dict], dict] = field(default=lambda r: dict(r), repr=False)
    regulation_ref: str = ""
    decoy: bool = False   # True => single-column, must score r_coherence=0


# ── mutators — each plants exactly its own incoherence into a clean row ──────

def _d01_ecl(row):  # consistency — formula
    r = dict(row); r["ECL"] = r["PD_FINAL"] * r["LGD_FINAL"] * r["EAD_TOTAL"] + 1000.0; return r

def _d02_rwa(row):
    r = dict(row); r["RWA"] = r["EAD_TOTAL"] * r["LGD_FINAL"] * 12.5 * r["K_IRB"] + 5000.0; return r

def _d03_pd_floor(row):
    r = dict(row); r["PD_FINAL"] = r["PD_SUELO"] / 2; return r

def _d04_lgd_floor(row):
    r = dict(row); r["LGD_SUELO"] = 0.45; r["LGD_FINAL"] = 0.20; return r

def _d05_pd_max(row):
    r = dict(row); r["PD_FINAL"] = r["PD_ESTIMADA"] / 2; return r

def _d06_stage3_low_dpd(row):
    r = dict(row); r["STAGE_IFRS9"] = 3; r["DPDS"] = 5; return r

def _d07_valor_sin_flag(row):
    r = dict(row); r["ADJUDICACION_VALOR"] = 50000.0; r["ADJUDICACION_FLAG"] = "0"; return r

def _d08_flag_sin_valor(row):
    r = dict(row); r["ADJUDICACION_FLAG"] = "1"; r["ADJUDICACION_VALOR"] = 0.0; return r

def _d09_valor_sin_tipo(row):
    r = dict(row); r["ADJUDICACION_VALOR"] = 12345.0; r["ADJUDICACION_TIPO"] = ""; return r

def _d10_recuperacion_sobre_ead(row):  # plausibility
    r = dict(row)
    ead = float(r.get("EAD_TOTAL") or 0)
    r["RECUPERACION_ACUMULADA"] = ead * 1.8 if ead > 0 else 1_000_000.0
    r["COSTE_TOTAL_ACUMULADO"] = 0.0
    return r

def _d11_cerrado_sin_terminacion(row):
    r = dict(row); r["ESTADO_CICLO"] = "CERRADO"; r["TERMINACION"] = ""; return r

def _d12_ead_total(row):
    r = dict(row); r["EAD_TOTAL"] = r["EAD_BALANCE"] + r["EAD_FUERA_BALANCE"] + 5000.0; return r

def _d13_ead_fuerasaldo(row):
    r = dict(row); r["EAD_FUERA_BALANCE"] = r["CCF_ESTIMADO"] * r["OR_DISBLE"] + 999.0; return r

def _d14_retail_hip_sin_hipoteca(row):  # conformity (cross-table)
    r = dict(row); r["SEGMENTO"] = "RETAIL_HIP"; r["COLATERAL_TIPO"] = "NINGUNA"; return r

def _d15_hipoteca_suelo_bajo(row):
    r = dict(row); r["COLATERAL_TIPO"] = "HIPOTECA"; r["LGD_SUELO"] = 0.10; return r

def _d16_kirb(row):
    r = dict(row); r["K_IRB"] = math.sqrt(r["PD_FINAL"]) * 0.06 + r["PD_FINAL"] * 0.5 + 0.05; return r

def _d17_fusion_dup(row):  # uniqueness / cardinality
    r = dict(row); r["SW_FUSION"] = 1; r["EAD_BALANCE"] = r["OR_EAD"] * 2.0; return r

def _d18_pd_nulo_activo(row):  # completeness
    r = dict(row); r["ESTADO_CICLO"] = "ESTIMACION"; r["PD_ESTIMADA"] = None; return r

def _d19_segmento_invalido(row):  # validity
    r = dict(row); r["SEGMENTO"] = "WHOLESALE"; return r

def _d20_stage_invalido(row):  # validity
    r = dict(row); r["STAGE_IFRS9"] = 5; return r

def _d21_mes_futuro(row):  # timeliness
    r = dict(row); r["MES_CICLO"] = 210012; return r

def _d22_ead_vs_basilea(row):  # accuracy vs authoritative source
    r = dict(row); r["EAD_BALANCE"] = r["OR_DISPTO"] + 100000.0; return r

def _d24_fusion_flag_inconsistente(row):  # conformity / integrity
    r = dict(row); r["SW_FUSION"] = 0; r["ID_FUSION_FINAL"] = "FUS_GHOST"; return r

def _d25_grade_pd_no_monotono(row):  # plausibility / monotonicity
    r = dict(row); r["RATING_GRADO"] = 16; r["PD_ESTIMADA"] = 0.002; return r

def _da_ead_cero(row):  # validity decoy
    r = dict(row); r["EAD_TOTAL"] = 0.0; return r

def _db_lgd_realizada_neg(row):  # validity decoy
    r = dict(row); r["LGD_REALIZADA"] = -0.5; return r


DEFECTS: list[Defect] = [
    # ── consistency (cross-field / formula invariants) ───────────────────────
    Defect("D01", "consistency", "formula", "HIGH",
           "El ECL no coincide con la fórmula canónica ECL = PD_FINAL * LGD_FINAL * EAD_TOTAL.",
           ("ECL", "PD_FINAL", "LGD_FINAL", "EAD_TOTAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > {FORMULA_EPS}",
           _d01_ecl, "IFRS 9 / CRR Art. 158"),
    Defect("D02", "consistency", "formula", "HIGH",
           "El RWA no coincide con RWA = EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB.",
           ("RWA", "EAD_TOTAL", "LGD_FINAL", "K_IRB"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(RWA - EAD_TOTAL*LGD_FINAL*12.5*K_IRB) > 1.0",
           _d02_rwa, "CRR Art. 153"),
    Defect("D03", "consistency", "referencial", "HIGH",
           "La PD final vulnera su suelo regulatorio: PD_FINAL < PD_SUELO.",
           ("PD_FINAL", "PD_SUELO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_FINAL < PD_SUELO",
           _d03_pd_floor, "CRR Art. 160.1"),
    Defect("D04", "consistency", "referencial", "HIGH",
           "La LGD final vulnera su suelo por colateral: LGD_FINAL < LGD_SUELO.",
           ("LGD_FINAL", "LGD_SUELO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE LGD_FINAL < LGD_SUELO",
           _d04_lgd_floor, "CRR Art. 161.1"),
    Defect("D05", "consistency", "consistencia", "MED",
           "La PD final es menor que la PD estimada: el suelo (max) no se aplicó.",
           ("PD_FINAL", "PD_ESTIMADA"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_FINAL < PD_ESTIMADA",
           _d05_pd_max, "CRR Art. 160.1"),
    Defect("D06", "consistency", "consistencia", "HIGH",
           "Ciclo en STAGE_IFRS9=3 (default) con DPDS < 30 días.",
           ("STAGE_IFRS9", "DPDS"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE STAGE_IFRS9=3 AND DPDS<30",
           _d06_stage3_low_dpd, "CRR Art. 178.1(b)"),
    Defect("D07", "consistency", "consistencia", "HIGH",
           "Adjudicación con valor > 0 cuyo flag no está activo (= '0').",
           ("ADJUDICACION_VALOR", "ADJUDICACION_FLAG"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ADJUDICACION_VALOR>0 AND ADJUDICACION_FLAG='0'",
           _d07_valor_sin_flag),
    Defect("D08", "consistency", "consistencia", "MED",
           "Flag de adjudicación activo ('1') pero valor = 0.",
           ("ADJUDICACION_FLAG", "ADJUDICACION_VALOR"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ADJUDICACION_FLAG='1' AND ADJUDICACION_VALOR=0",
           _d08_flag_sin_valor),
    Defect("D09", "consistency", "consistencia", "HIGH",
           "Adjudicación con valor > 0 cuyo tipo está vacío.",
           ("ADJUDICACION_VALOR", "ADJUDICACION_TIPO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ADJUDICACION_VALOR>0 AND (ADJUDICACION_TIPO='' OR ADJUDICACION_TIPO IS NULL)",
           _d09_valor_sin_tipo),
    Defect("D11", "consistency", "consistencia", "HIGH",
           "Ciclo CERRADO sin causa de TERMINACION informada.",
           ("ESTADO_CICLO", "TERMINACION"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ESTADO_CICLO='CERRADO' AND (TERMINACION='' OR TERMINACION IS NULL)",
           _d11_cerrado_sin_terminacion),
    Defect("D12", "consistency", "formula", "MED",
           "EAD_TOTAL no coincide con EAD_BALANCE + EAD_FUERA_BALANCE.",
           ("EAD_TOTAL", "EAD_BALANCE", "EAD_FUERA_BALANCE"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(EAD_TOTAL - (EAD_BALANCE+EAD_FUERA_BALANCE)) > {FORMULA_EPS}",
           _d12_ead_total, "CRR Art. 166"),
    Defect("D13", "consistency", "formula", "MED",
           "EAD_FUERA_BALANCE no coincide con CCF_ESTIMADO * OR_DISBLE cuando hay undrawn.",
           ("EAD_FUERA_BALANCE", "CCF_ESTIMADO", "OR_DISBLE"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE OR_DISBLE>0 AND ABS(EAD_FUERA_BALANCE - CCF_ESTIMADO*OR_DISBLE) > {FORMULA_EPS}",
           _d13_ead_fuerasaldo, "CRR Art. 166.8"),
    Defect("D15", "consistency", "referencial", "HIGH",
           "Colateral HIPOTECA cuyo LGD_SUELO es inferior al mínimo del 30%.",
           ("COLATERAL_TIPO", "LGD_SUELO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE COLATERAL_TIPO='HIPOTECA' AND LGD_SUELO<0.30",
           _d15_hipoteca_suelo_bajo, "CRR Art. 161.1"),
    Defect("D16", "consistency", "formula", "MED",
           "K_IRB no coincide con SQRT(PD_FINAL)*0.06 + PD_FINAL*0.5.",
           ("K_IRB", "PD_FINAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(K_IRB - (SQRT(PD_FINAL)*0.06+PD_FINAL*0.5)) > 0.0001",
           _d16_kirb, "CRR Art. 153"),
    # ── uniqueness / cardinality ────────────────────────────────────────────
    Defect("D17", "uniqueness", "cross_table", "MED",
           "Contrato fusionado (SW_FUSION=1) cuyo EAD en balance duplica el OR_EAD "
           "original — síntoma de filas BASILEA duplicadas que sobrevivieron.",
           ("SW_FUSION", "EAD_BALANCE", "OR_EAD"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SW_FUSION=1 AND EAD_BALANCE > OR_EAD*1.5",
           _d17_fusion_dup, "BASILEA dedup"),
    # ── completeness ─────────────────────────────────────────────────────────
    Defect("D18", "completeness", "completitud", "HIGH",
           "PD_ESTIMADA vacía (NULL) en un ciclo activo (no CERRADO): campo "
           "mandatorio para el cálculo de capital.",
           ("PD_ESTIMADA", "ESTADO_CICLO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_ESTIMADA IS NULL AND ESTADO_CICLO<>'CERRADO'",
           _d18_pd_nulo_activo, "BCBS 239 P4 / EBA GL 2017/16"),
    # ── validity (domain membership) ────────────────────────────────────────
    Defect("D19", "validity", "rango", "HIGH",
           f"SEGMENTO fuera del dominio permitido {ALLOWED_SEGMENTS}.",
           ("SEGMENTO",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SEGMENTO NOT IN ('CORP','SME','RETAIL_HIP','RETAIL_CONS')",
           _d19_segmento_invalido, "CRR Art. 147"),
    Defect("D20", "validity", "rango", "HIGH",
           "STAGE_IFRS9 fuera del dominio permitido {1,2,3}.",
           ("STAGE_IFRS9",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE STAGE_IFRS9 NOT IN (1,2,3)",
           _d20_stage_invalido, "IFRS 9"),
    # ── timeliness ───────────────────────────────────────────────────────────
    Defect("D21", "timeliness", "consistencia", "MED",
           f"MES_CICLO posterior al periodo de referencia ({REFERENCE_PERIOD}): "
           "dato futuro / no valido para el corte.",
           ("MES_CICLO",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE MES_CICLO > {REFERENCE_PERIOD}",
           _d21_mes_futuro, "BCBS 239 P5"),
    # ── accuracy (vs authoritative source value) ────────────────────────────
    Defect("D22", "accuracy", "consistencia", "MED",
           "El EAD en balance difiere del importe dispuesto BASILEA (OR_DISPTO) "
           "más allá de la tolerancia: inexactitud frente a la fuente autorizada.",
           ("EAD_BALANCE", "OR_DISPTO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(EAD_BALANCE - OR_DISPTO) > 1.0",
           _d22_ead_vs_basilea, "BCBS 239 P3"),
    # ── conformity / referential integrity ──────────────────────────────────
    Defect("D14", "conformity", "cross_table", "HIGH",
           "Segmento RETAIL_HIP (hipotecario) cuyo colateral no es HIPOTECA: "
           "inconsistencia referencial entre segmento y tipo de garantía.",
           ("SEGMENTO", "COLATERAL_TIPO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SEGMENTO='RETAIL_HIP' AND COLATERAL_TIPO<>'HIPOTECA'",
           _d14_retail_hip_sin_hipoteca, "CRR Art. 147"),
    Defect("D24", "conformity", "consistencia", "MED",
           "SW_FUSION=0 (sin fusión) pero ID_FUSION_FINAL informado: flags de "
           "fusión incoherentes (integridad referencial).",
           ("SW_FUSION", "ID_FUSION_FINAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SW_FUSION=0 AND ID_FUSION_FINAL IS NOT NULL",
           _d24_fusion_flag_inconsistente),
    # ── plausibility / reasonableness ───────────────────────────────────────
    Defect("D10", "plausibility", "consistencia", "MED",
           "(RECUPERACION + COSTE_TOTAL) supera el 150% del EAD: recuperación "
           "implausible frente a la exposición (razonabilidad).",
           ("RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO", "EAD_TOTAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE (RECUPERACION_ACUMULADA+COSTE_TOTAL_ACUMULADO) > 1.5*EAD_TOTAL",
           _d10_recuperacion_sobre_ead, "EBA GL 2017/16 §135"),
    Defect("D25", "plausibility", "consistencia", "MED",
           "Peor grado de rating (>=14) con PD estimada implausible baja (<1%): "
           "violación de monotonicidad rating↔PD (razonabilidad).",
           ("RATING_GRADO", "PD_ESTIMADA"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE RATING_GRADO>=14 AND PD_ESTIMADA<0.01",
           _d25_grade_pd_no_monotono, "EBA GL 2017/16 §73"),
    # ── decoys: single-column range, must score r_coherence = 0 ──────────────
    Defect("DA", "validity", "rango", "MED",
           "EAD_TOTAL <= 0 en un ciclo activo (control de rango, no coherencia).",
           ("EAD_TOTAL",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE EAD_TOTAL<=0",
           _da_ead_cero, decoy=True),
    Defect("DB", "validity", "rango", "LOW",
           "LGD_REALIZADA fuera de rango [0,1] (control de rango, no coherencia).",
           ("LGD_REALIZADA",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE LGD_REALIZADA NOT BETWEEN 0 AND 1",
           _db_lgd_realizada_neg, decoy=True),
]

DEFECTS_BY_ID: dict[str, Defect] = {d.defect_id: d for d in DEFECTS}
COHERENCE_DEFECTS = [d for d in DEFECTS if not d.decoy]
DECOY_DEFECTS = [d for d in DEFECTS if d.decoy]


def defects_by_dimension() -> dict[str, list[Defect]]:
    out: dict[str, list[Defect]] = {d: [] for d in DIMENSIONS}
    for d in DEFECTS:
        out.setdefault(d.dimension, []).append(d)
    return out


if __name__ == "__main__":
    by_dim = defects_by_dimension()
    print(f"{len(DEFECTS)} defects across {sum(1 for v in by_dim.values() if v)} dimensions")
    for dim in DIMENSIONS:
        ds = by_dim.get(dim, [])
        if ds:
            ids = ", ".join(d.defect_id for d in ds)
            print(f"  {dim:14s} ({len(ds)}): {ids}")
