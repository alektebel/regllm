#!/usr/bin/env python3
"""Generate a structured test set for the compliance checker.

Three violation categories:
  A) CICLOS (recovery cycles) — bad data values
     - refinanciacion con provision period < 12 months (must be >= 12 for restructured)
     - dispuestos (drawn amount) < 0 (impossible)
     - LGD below regulatory floor for mortgage

  B) CONTRATOS (contracts master) — business logic inconsistencies
     - adjudicado flag set but no fecha_adjudicacion
     - tipo_producto = 'REFINANCIACION' but plazo_refinanciacion < 12 months
     - saldo_dispuesto < 0

  C) TITULARIDADES + ADJUDICADOS cross-table inconsistency
     - marca_adjudicado in titularidades doesn't match adjudicados table
     - same contrato marked as adjudicado in one table but not the other

Outputs under data/test/:
  ciclos_test.csv          — 120 rows (80 clean, 40 violations)
  contratos_test.csv       — 80 rows (50 clean, 30 violations)
  titularidades_test.csv   — 100 rows (70 clean, 30 violations)
  adjudicados_test.csv     — reference table for cross-check
  test_violations.md       — ground truth: which rows should be flagged and why
"""

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(99)

# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))

def fmt(d: date) -> str:
    return d.isoformat()

BASE_DATE = date(2024, 6, 30)

# ── A: CICLOS ─────────────────────────────────────────────────────────────────

CICLO_FIELDS = [
    "CICLO_ID", "CONTRATO", "TIPO_OPERACION", "SEGMENTO",
    "PROVISION_PERIOD_MONTHS", "DISPUESTO", "EAD",
    "LGD_ESTIMADA", "PD_ESTIMADA", "DPDS", "STAGE_IFRS9",
    "COLATERAL_TIPO", "FECHA_INICIO_CICLO", "FECHA_FIN_CICLO",
    "VENTANA_CALIBRACION_YEARS", "MOTIVO_VIOLATION",
]

def make_ciclo(i, violation=None):
    tipo = random.choice(["NORMAL", "REFINANCIACION", "ADJUDICADO"])
    seg  = random.choice(["MORTGAGE", "CORP", "RETAIL", "SME"])
    col  = {"MORTGAGE": "HIPOTECA", "CORP": "NINGUNA", "RETAIL": "PERSONAL", "SME": "FINANCIERO"}[seg]
    lgd_floor = {"MORTGAGE": 0.30, "CORP": 0.45, "RETAIL": 0.0, "SME": 0.0}[seg]

    # Clean defaults
    prov_months = random.randint(12, 36)
    dispuesto   = round(random.uniform(5_000, 500_000), 2)
    lgd         = round(lgd_floor + random.uniform(0.01, 0.25), 4)
    pd          = round(random.uniform(0.001, 0.15), 6)
    dpds        = random.choice([0, 0, 0, random.randint(1, 90), random.randint(91, 360)])
    stage       = 3 if dpds >= 90 else (2 if dpds >= 30 else 1)
    calib_years = round(random.uniform(7, 12), 1)
    start_date  = rand_date(date(2018, 1, 1), date(2023, 1, 1))
    end_date    = start_date + timedelta(days=prov_months * 30)
    motivo      = ""

    if violation == "REFIN_PERIOD_CORTO":
        tipo        = "REFINANCIACION"
        prov_months = random.randint(1, 11)   # < 12 months — VIOLATION
        end_date    = start_date + timedelta(days=prov_months * 30)
        motivo      = "REFIN_PERIOD_CORTO: provision_period < 12 meses para refinanciacion"

    elif violation == "DISPUESTO_NEGATIVO":
        dispuesto   = round(random.uniform(-50_000, -100), 2)  # < 0 — VIOLATION
        motivo      = "DISPUESTO_NEGATIVO: importe dispuesto < 0"

    elif violation == "LGD_BAJO_SUELO":
        lgd  = round(random.uniform(0.01, lgd_floor - 0.01), 4) if lgd_floor > 0 else 0.05
        seg  = "MORTGAGE"
        col  = "HIPOTECA"
        motivo = "LGD_BAJO_SUELO: LGD < 30% para hipoteca (CRR Art. 154(3))"

    return {
        "CICLO_ID":                f"CIC_{i:05d}",
        "CONTRATO":                f"CONT_{1000 + i}",
        "TIPO_OPERACION":          tipo,
        "SEGMENTO":                seg,
        "PROVISION_PERIOD_MONTHS": prov_months,
        "DISPUESTO":               dispuesto,
        "EAD":                     round(abs(dispuesto) * random.uniform(0.9, 1.1), 2),
        "LGD_ESTIMADA":            lgd,
        "PD_ESTIMADA":             pd,
        "DPDS":                    dpds,
        "STAGE_IFRS9":             stage,
        "COLATERAL_TIPO":          col,
        "FECHA_INICIO_CICLO":      fmt(start_date),
        "FECHA_FIN_CICLO":         fmt(end_date),
        "VENTANA_CALIBRACION_YEARS": calib_years,
        "MOTIVO_VIOLATION":        motivo,
    }


def gen_ciclos(n_clean=80, n_each_violation=13):
    rows = [make_ciclo(i) for i in range(n_clean)]
    i = n_clean
    for viol in ["REFIN_PERIOD_CORTO", "DISPUESTO_NEGATIVO", "LGD_BAJO_SUELO"]:
        for _ in range(n_each_violation):
            rows.append(make_ciclo(i, violation=viol))
            i += 1
    random.shuffle(rows)
    return rows


# ── B: CONTRATOS ──────────────────────────────────────────────────────────────

CONTRATO_FIELDS = [
    "CONTRATO", "TIPO_PRODUCTO", "SEGMENTO", "FECHA_CONCESION",
    "SALDO_DISPUESTO", "SALDO_LIMITE", "PLAZO_REFINANCIACION_MESES",
    "FLAG_ADJUDICADO", "FECHA_ADJUDICACION", "MARCA_ADJUDICADO",
    "MOTIVO_VIOLATION",
]

def make_contrato(i, violation=None):
    tipo    = random.choice(["HIPOTECA", "CREDITO_CONSUMO", "CREDITO_EMPRESA", "REFINANCIACION", "LINEA_CREDITO"])
    seg     = random.choice(["MORTGAGE", "RETAIL", "CORP", "SME"])
    saldo   = round(random.uniform(10_000, 800_000), 2)
    limite  = round(saldo * random.uniform(1.0, 1.5), 2)
    plazo_refin = random.randint(12, 60) if tipo == "REFINANCIACION" else 0
    adj_date    = fmt(rand_date(date(2020, 1, 1), date(2023, 12, 31)))
    flag_adj    = random.choice([0, 0, 0, 1])
    marca_adj   = "ADJUDICADO" if flag_adj else ""
    motivo      = ""

    if violation == "REFIN_PLAZO_CORTO":
        tipo        = "REFINANCIACION"
        plazo_refin = random.randint(1, 11)      # < 12 months — VIOLATION
        motivo      = "REFIN_PLAZO_CORTO: plazo_refinanciacion < 12 meses"

    elif violation == "SALDO_NEGATIVO":
        saldo  = round(random.uniform(-200_000, -1_000), 2)   # < 0 — VIOLATION
        motivo = "SALDO_NEGATIVO: saldo_dispuesto < 0"

    elif violation == "ADJUDICADO_SIN_FECHA":
        flag_adj  = 1
        marca_adj = "ADJUDICADO"
        adj_date  = ""                           # flag set but no date — VIOLATION
        motivo    = "ADJUDICADO_SIN_FECHA: flag_adjudicado=1 pero sin fecha_adjudicacion"

    return {
        "CONTRATO":                   f"CONT_{2000 + i}",
        "TIPO_PRODUCTO":              tipo,
        "SEGMENTO":                   seg,
        "FECHA_CONCESION":            fmt(rand_date(date(2010, 1, 1), date(2022, 1, 1))),
        "SALDO_DISPUESTO":            saldo,
        "SALDO_LIMITE":               limite,
        "PLAZO_REFINANCIACION_MESES": plazo_refin,
        "FLAG_ADJUDICADO":            flag_adj,
        "FECHA_ADJUDICACION":         adj_date if flag_adj else "",
        "MARCA_ADJUDICADO":           marca_adj,
        "MOTIVO_VIOLATION":           motivo,
    }


def gen_contratos(n_clean=50, n_each=10):
    rows = [make_contrato(i) for i in range(n_clean)]
    i = n_clean
    for viol in ["REFIN_PLAZO_CORTO", "SALDO_NEGATIVO", "ADJUDICADO_SIN_FECHA"]:
        for _ in range(n_each):
            rows.append(make_contrato(i, violation=viol))
            i += 1
    random.shuffle(rows)
    return rows


# ── C: TITULARIDADES + ADJUDICADOS (cross-table) ──────────────────────────────

TITULARIDAD_FIELDS = [
    "TITULARIDAD_ID", "CONTRATO", "NIF_TITULAR", "ROL_TITULAR",
    "MARCA_ADJUDICADO_TITU",   # marca in titularidades table
    "MOTIVO_VIOLATION",
]

ADJUDICADO_FIELDS = [
    "CONTRATO", "NIF_TITULAR", "MARCA_ADJUDICADO_ADJ",  # reference table
    "FECHA_ADJUDICACION", "TIPO_ADJUDICACION",
]

# Contracts that are cleanly adjudicados in both tables
CLEAN_ADJ_CONTRACTS = [f"CONT_{3000 + i}" for i in range(30)]
# Contracts that are cleanly NOT adjudicados in both tables
CLEAN_NO_ADJ_CONTRACTS = [f"CONT_{4000 + i}" for i in range(40)]
# Contracts with mismatch (violation): adjudicado in titularidades but NOT in adjudicados table
MISMATCH_CONTRACTS = [f"CONT_{5000 + i}" for i in range(15)]
# Contracts with mismatch: NOT adjudicado in titularidades but IS in adjudicados table
MISMATCH_CONTRACTS_2 = [f"CONT_{6000 + i}" for i in range(15)]

def gen_titularidades():
    rows = []
    tid = 1

    # Clean adjudicados
    for c in CLEAN_ADJ_CONTRACTS:
        for rol in ["TITULAR", "AVALISTA"][:random.randint(1, 2)]:
            rows.append({
                "TITULARIDAD_ID":        f"TIT_{tid:05d}",
                "CONTRATO":              c,
                "NIF_TITULAR":           f"NIF{random.randint(10000000, 99999999)}A",
                "ROL_TITULAR":           rol,
                "MARCA_ADJUDICADO_TITU": "ADJUDICADO",
                "MOTIVO_VIOLATION":      "",
            })
            tid += 1

    # Clean non-adjudicados
    for c in CLEAN_NO_ADJ_CONTRACTS:
        rows.append({
            "TITULARIDAD_ID":        f"TIT_{tid:05d}",
            "CONTRATO":              c,
            "NIF_TITULAR":           f"NIF{random.randint(10000000, 99999999)}B",
            "ROL_TITULAR":           "TITULAR",
            "MARCA_ADJUDICADO_TITU": "",
            "MOTIVO_VIOLATION":      "",
        })
        tid += 1

    # VIOLATION: marked adjudicado in titularidades, NOT in adjudicados table
    for c in MISMATCH_CONTRACTS:
        rows.append({
            "TITULARIDAD_ID":        f"TIT_{tid:05d}",
            "CONTRATO":              c,
            "NIF_TITULAR":           f"NIF{random.randint(10000000, 99999999)}C",
            "ROL_TITULAR":           "TITULAR",
            "MARCA_ADJUDICADO_TITU": "ADJUDICADO",
            "MOTIVO_VIOLATION":      "MARCA_MISMATCH: adjudicado en titularidades pero ausente en tabla adjudicados",
        })
        tid += 1

    # VIOLATION: NOT marked in titularidades, IS in adjudicados table
    for c in MISMATCH_CONTRACTS_2:
        rows.append({
            "TITULARIDAD_ID":        f"TIT_{tid:05d}",
            "CONTRATO":              c,
            "NIF_TITULAR":           f"NIF{random.randint(10000000, 99999999)}D",
            "ROL_TITULAR":           "TITULAR",
            "MARCA_ADJUDICADO_TITU": "",
            "MOTIVO_VIOLATION":      "MARCA_MISMATCH: contrato en tabla adjudicados pero sin marca en titularidades",
        })
        tid += 1

    return rows


def gen_adjudicados():
    """Reference table — ground truth of which contracts are adjudicados."""
    rows = []
    for c in CLEAN_ADJ_CONTRACTS:
        rows.append({
            "CONTRATO":              c,
            "NIF_TITULAR":           f"NIF{random.randint(10000000, 99999999)}A",
            "MARCA_ADJUDICADO_ADJ":  "ADJUDICADO",
            "FECHA_ADJUDICACION":    fmt(rand_date(date(2020, 1, 1), date(2023, 6, 1))),
            "TIPO_ADJUDICACION":     random.choice(["DACION", "SUBASTA", "COMPRA_DIRECTA"]),
        })
    # MISMATCH_CONTRACTS are NOT in adjudicados table (but marked in titularidades — violation)
    # MISMATCH_CONTRACTS_2 ARE in adjudicados table (but not marked in titularidades — violation)
    for c in MISMATCH_CONTRACTS_2:
        rows.append({
            "CONTRATO":              c,
            "NIF_TITULAR":           f"NIF{random.randint(10000000, 99999999)}D",
            "MARCA_ADJUDICADO_ADJ":  "ADJUDICADO",
            "FECHA_ADJUDICACION":    fmt(rand_date(date(2020, 1, 1), date(2023, 6, 1))),
            "TIPO_ADJUDICACION":     random.choice(["DACION", "SUBASTA"]),
        })
    return rows


# ── Ground-truth report ───────────────────────────────────────────────────────

VIOLATION_DOC = """# Test Set — Ground Truth Violations

## File: ciclos_test.csv

| Violation | Rule | Regulatory basis |
|-----------|------|-----------------|
| REFIN_PERIOD_CORTO | PROVISION_PERIOD_MONTHS < 12 for TIPO_OPERACION='REFINANCIACION' | EBA GL 2017/16 §5.3.1 — restructured exposures require >= 12-month observation period |
| DISPUESTO_NEGATIVO | DISPUESTO < 0 | Data integrity — drawn amount cannot be negative |
| LGD_BAJO_SUELO | LGD_ESTIMADA < 0.30 for COLATERAL_TIPO='HIPOTECA' | CRR Art. 154(3) — LGD floor 30% for mortgage-secured exposures |

Expected flagged rows: ~39 (13 per violation type)

---

## File: contratos_test.csv

| Violation | Rule | Regulatory basis |
|-----------|------|-----------------|
| REFIN_PLAZO_CORTO | PLAZO_REFINANCIACION_MESES < 12 for TIPO_PRODUCTO='REFINANCIACION' | EBA GL 2017/16 §5.4 — restructuring must have minimum 12-month probation period |
| SALDO_NEGATIVO | SALDO_DISPUESTO < 0 | Data integrity — outstanding balance cannot be negative |
| ADJUDICADO_SIN_FECHA | FLAG_ADJUDICADO=1 but FECHA_ADJUDICACION is empty | Internal model requirement — adjudicado flag requires adjudicacion date |

Expected flagged rows: ~30 (10 per violation type)

---

## File: titularidades_test.csv + adjudicados_test.csv (cross-table)

| Violation | Rule |
|-----------|------|
| MARCA_MISMATCH (type 1) | MARCA_ADJUDICADO_TITU='ADJUDICADO' in titularidades but CONTRATO is absent from adjudicados_test.csv |
| MARCA_MISMATCH (type 2) | CONTRATO present in adjudicados_test.csv but MARCA_ADJUDICADO_TITU is blank in titularidades |

Affected contracts: CONT_5000–CONT_5014 (type 1), CONT_6000–CONT_6014 (type 2)
Expected cross-table mismatches: 30

---

## How to validate

```bash
# Data validation — ciclos
python scripts/run_compliance_check.py data/test/ciclos_test.csv \\
  --table TEST.CICLOS --tiered --no-rag \\
  --grain "ciclo de recuperación — refinanciación y adjudicados"

# Data validation — contratos
python scripts/run_compliance_check.py data/test/contratos_test.csv \\
  --table TEST.CONTRATOS --tiered --no-rag \\
  --grain "contrato de crédito — refinanciación y adjudicados"

# Cross-table check (join both files and check MARCA consistency)
python scripts/check_adjudicados_consistency.py \\
  data/test/titularidades_test.csv data/test/adjudicados_test.csv
```
"""

# ── Main ──────────────────────────────────────────────────────────────────────

def write_csv(path: Path, fields: list, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", default="data/test")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ciclos        = gen_ciclos()
    contratos     = gen_contratos()
    titularidades = gen_titularidades()
    adjudicados   = gen_adjudicados()

    write_csv(out / "ciclos_test.csv",        CICLO_FIELDS,       ciclos)
    write_csv(out / "contratos_test.csv",     CONTRATO_FIELDS,    contratos)
    write_csv(out / "titularidades_test.csv", TITULARIDAD_FIELDS, titularidades)
    write_csv(out / "adjudicados_test.csv",   ADJUDICADO_FIELDS,  adjudicados)
    (out / "test_violations.md").write_text(VIOLATION_DOC, encoding="utf-8")

    # Summary
    refin  = sum(1 for r in ciclos     if r["MOTIVO_VIOLATION"].startswith("REFIN_PERIOD"))
    disp   = sum(1 for r in ciclos     if r["MOTIVO_VIOLATION"].startswith("DISPUESTO"))
    lgd    = sum(1 for r in ciclos     if r["MOTIVO_VIOLATION"].startswith("LGD"))
    mm1    = sum(1 for r in titularidades if "type 1" in r["MOTIVO_VIOLATION"] or "ausente" in r["MOTIVO_VIOLATION"])
    mm2    = sum(1 for r in titularidades if "sin marca" in r["MOTIVO_VIOLATION"])

    print(f"ciclos_test.csv        {len(ciclos):>4} rows  ({refin} REFIN_PERIOD_CORTO, {disp} DISPUESTO_NEG, {lgd} LGD_SUELO)")
    print(f"contratos_test.csv     {len(contratos):>4} rows  (10 REFIN_PLAZO_CORTO, 10 SALDO_NEG, 10 ADJ_SIN_FECHA)")
    print(f"titularidades_test.csv {len(titularidades):>4} rows  ({mm1} MARCA_MISMATCH_T1, {mm2} MARCA_MISMATCH_T2)")
    print(f"adjudicados_test.csv   {len(adjudicados):>4} rows  (reference table)")
    print(f"\nOutput: {out.resolve()}")
    print(f"Ground truth: {(out / 'test_violations.md').resolve()}")


if __name__ == "__main__":
    main()
