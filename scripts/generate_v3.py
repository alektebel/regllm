#!/usr/bin/env python3
"""Produce ``cycles_v2.csv`` and ``cycles_v3.csv`` from ``cycles_sample.csv``.

V2 is the current file copied verbatim. V3 introduces a mix of *justified*
changes (documented in ``data/changelog/*.md``) and *unjustified* changes
(that the gradient + Shapley explainer should still surface but the
GraphRAG should report as non-justified).

Justified changes
-----------------
- ``PD_ESTIMADA`` for ``RATING_GRADO ∈ {1, 2}``: ×1.15 (Q1 2025 recalibration).
- ``COLATERAL_TIPO == 'FINANCIERO'`` → ``COLATERAL_FIN`` (rename).
- ``LGD_ESTIMADA`` for ``COLATERAL_TIPO == 'HIPOTECA'``: ×0.95.

Unjustified changes (introduced for testing)
--------------------------------------------
- One out of every 50 rows: ``EAD`` ×1.20 with no documentation.

Run:
    python scripts/generate_v3.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
SRC = SAMPLES / "cycles_sample.csv"
V2 = SAMPLES / "cycles_v2.csv"
V3 = SAMPLES / "cycles_v3.csv"

NUMERIC = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9", "ECL",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
    "VENTANA_CALIBRACION_YEARS", "RATING_GRADO", "LGD_FLOOR_APLICADO",
    "MOC", "CURE_FLAG", "RWA",
}


def _to_num(v: str) -> float | str:
    if v == "":
        return v
    try:
        return float(v)
    except ValueError:
        return v


def mutate(row: dict[str, str], idx: int) -> dict[str, str]:
    """Return a mutated row representing the V3 snapshot."""
    out = dict(row)

    # Justified — Q1 2025 PD recalibration on best ratings (1,2)
    rating = _to_num(row.get("RATING_GRADO", ""))
    if isinstance(rating, float) and rating in (1.0, 2.0):
        pd = _to_num(row.get("PD_ESTIMADA", ""))
        if isinstance(pd, float):
            out["PD_ESTIMADA"] = f"{pd * 1.15:.6f}"

    # Justified — COLATERAL_TIPO rename FINANCIERO → COLATERAL_FIN
    if row.get("COLATERAL_TIPO") == "FINANCIERO":
        out["COLATERAL_TIPO"] = "COLATERAL_FIN"

    # Justified — LGD trim for HIPOTECA collateral (Q1 2025 cure-rate update)
    if row.get("COLATERAL_TIPO") == "HIPOTECA":
        lgd = _to_num(row.get("LGD_ESTIMADA", ""))
        if isinstance(lgd, float):
            out["LGD_ESTIMADA"] = f"{lgd * 0.95:.6f}"

    # Unjustified — every 50th row gets a mysterious EAD bump
    if idx % 50 == 0:
        ead = _to_num(row.get("EAD", ""))
        if isinstance(ead, float):
            out["EAD"] = f"{ead * 1.20:.2f}"

    # Recompute ECL = PD * LGD * EAD using the updated values
    pd = _to_num(out.get("PD_ESTIMADA", ""))
    lgd = _to_num(out.get("LGD_ESTIMADA", ""))
    ead = _to_num(out.get("EAD", ""))
    if all(isinstance(x, float) for x in (pd, lgd, ead)):
        out["ECL"] = f"{pd * lgd * ead:.2f}"

    return out


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source CSV not found: {SRC}")

    # V2 = exact copy
    shutil.copyfile(SRC, V2)
    print(f"wrote {V2}")

    with SRC.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    with V3.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(mutate(r, i))
    print(f"wrote {V3}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
