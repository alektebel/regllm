#!/usr/bin/env python3
"""Bootstrap data/sas/{v2,v3}/ and data/docs/* with illustrative content
so the agentic Q&A demo runs out of the box.

Run:
    python scripts/seed_docs.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAS_SRC = ROOT / "data" / "samples" / "sample_lgd.sas"
SAS_V2 = ROOT / "data" / "sas" / "v2"
SAS_V3 = ROOT / "data" / "sas" / "v3"
DOCS = ROOT / "data" / "docs"


# ── V2 / V3 SAS ──────────────────────────────────────────────────────────────


def _v3_variant(text: str) -> str:
    """Inject 2-3 plausible code-level differences for the demo:

    1. The CORP LGD floor is bumped from 45% → 50% (regulatory tightening).
    2. The PD minimum is lifted from 0.03% → 0.05%.
    3. A new step appears that derives a titulised EAD column (OR_EAD_TIT).
    """
    # 1. CORP LGD floor 0.45 → 0.50
    text = text.replace(
        "IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;",
        "IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.50 THEN LGD_ESTIMADA = 0.50;  /* V3: floor +5pp */",
    )
    # 2. PD floor 0.0003 → 0.0005
    text = text.replace(
        "IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;",
        "IF PD_ESTIMADA < 0.0005 THEN PD_ESTIMADA = 0.0005;  /* V3: PD floor 0.05% */",
    )
    text = text.replace(
        "OR PD_ESTIMADA < 0.0003;",
        "OR PD_ESTIMADA < 0.0005;",
    )
    # 3. New step: derive OR_EAD_TIT (titulised EAD)
    extra_step = """
/* V3 NEW: Cómputo del EAD titulizado para ciclos con SECURITISED='S' */
DATA work.titulizado;
    SET work.ecl_calculo;

    /* Multiplicador reglamentario para tramos titulizados: 2.0x el EAD base */
    IF SEGMENTO = 'CORP' THEN DO;
        OR_EAD_TIT = EAD * 2.0;
    END;
    ELSE IF SEGMENTO = 'RETAIL' THEN DO;
        OR_EAD_TIT = EAD * 1.5;
    END;
    ELSE DO;
        OR_EAD_TIT = EAD;
    END;
RUN;
"""
    # Insert before the PROC MEANS block
    marker = "/* Verificación ventana calibración"
    text = text.replace(marker, extra_step.lstrip() + "\n" + marker, 1)
    return text


def seed_sas() -> None:
    if not SAS_SRC.exists():
        print(f"[seed] No source SAS at {SAS_SRC}; skipping V2/V3 bootstrap")
        return
    SAS_V2.mkdir(parents=True, exist_ok=True)
    SAS_V3.mkdir(parents=True, exist_ok=True)
    v2_target = SAS_V2 / "sample_lgd.sas"
    v3_target = SAS_V3 / "sample_lgd.sas"
    if not v2_target.exists():
        shutil.copy(SAS_SRC, v2_target)
        print(f"[seed] wrote {v2_target.relative_to(ROOT)}")
    else:
        print(f"[seed] kept existing {v2_target.relative_to(ROOT)}")
    if not v3_target.exists():
        text = SAS_SRC.read_text(encoding="utf-8")
        v3_target.write_text(_v3_variant(text), encoding="utf-8")
        print(f"[seed] wrote {v3_target.relative_to(ROOT)}")
    else:
        print(f"[seed] kept existing {v3_target.relative_to(ROOT)}")


# ── Markdown corpus ──────────────────────────────────────────────────────────


_DOCS: list[tuple[str, str]] = [
    (
        "fields/OR_EAD_TIT.md",
        """\
# Field OR_EAD_TIT

`OR_EAD_TIT` is the **EAD titulizado** — Exposure At Default for tranches
sold into a securitisation programme. It is derived from the base `EAD`
by applying a regulatory multiplier that depends on the customer
`SEGMENTO`.

## Definition

| Segmento | Multiplier | Source |
|----------|-----------:|--------|
| CORP     | 2.0        | EBA GL 2020/06 §3.4 |
| RETAIL   | 1.5        | EBA GL 2020/06 §3.7 |
| Other    | 1.0        | (fall-through, no multiplier) |

## Pipeline

This field is produced by the SAS data step `work.titulizado`, which
exists in V3 of the calibration pipeline only. In V2 the field is not
computed at all — downstream consumers fell back to the raw `EAD`.

## Common pitfalls

- Comparing V2 vs V3 directly is misleading because the field did not
  exist in V2. The agent should treat a V2 row's missing `OR_EAD_TIT`
  as "not applicable" and explain that the V3 multiplier was added by
  release 2025-Q1.
""",
    ),
    (
        "fields/EAD.md",
        """\
# Field EAD

`EAD` (Exposure At Default) is the on-balance exposure used in IRB ECL
formulas: `ECL = PD × LGD × EAD`. It feeds both `ECL` and `RWA`.

## Source table

`mylib.ciclos_recuperacion` — column `EAD`, type NUMERIC (€).

## Notes

- V3 of the database refreshed `EAD` for retail mortgages with revised
  amortisation profiles (see `data/changelog/2025-q1-pd-recalibration.md`).
- The field is *input* to the SAS pipeline; it is never overwritten.
""",
    ),
    (
        "fields/PD_ESTIMADA.md",
        """\
# Field PD_ESTIMADA

`PD_ESTIMADA` is the 1-year point-in-time probability of default
estimated by the master scale.

## Floors

CRR Art. 160(1) sets a regulatory floor of **0.03 %**. The V3 pipeline
raises this floor to **0.05 %** following the 2025-Q1 master-scale
recalibration. The change is implemented in the data step
`work.ecl_calculo`.
""",
    ),
    (
        "fields/LGD_ESTIMADA.md",
        """\
# Field LGD_ESTIMADA

Loss Given Default — the proportion of `EAD` expected to be lost net of
recoveries.

## Floors

| Segment / Collateral                | V2 floor | V3 floor |
|--------------------------------------|---------:|---------:|
| HIPOTECA (CRR Art. 154(3))           | 0.30     | 0.30     |
| CORP (CRR Art. 161(1)(b))            | 0.45     | **0.50** |
| RETAIL no collateral                 | none     | none     |

The CORP floor was tightened by 5 pp in V3 (data step
`work.lgd_con_suelos`) per the 2025-Q1 supervisory dialogue.
""",
    ),
    (
        "fluxes/lgd_pipeline.md",
        """\
# LGD calibration pipeline

## Stages

1. `work.ciclos` — read raw recovery cycles, filter
   `PROVISION_PERIOD_MONTHS >= 9`.
2. `work.lgd_con_suelos` — apply regulatory LGD floors per
   `COLATERAL_TIPO` × `SEGMENTO`.
3. `work.ecl_calculo` — apply the PD floor, compute
   `ECL = PD × LGD × EAD`, reclassify IFRS 9 stage on the 30-DPD
   backstop.
4. `work.titulizado` *(V3 only)* — derive `OR_EAD_TIT` from `EAD` and
   `SEGMENTO`.
5. `work.no_conformes` — flag rows that fail one of: short calibration
   window, short observation window, mortgage with LGD < floor, PD <
   regulatory floor.

## V2 → V3 differences

- The CORP LGD floor was raised from 0.45 to 0.50.
- The PD floor was raised from 0.0003 to 0.0005.
- A new step `work.titulizado` was inserted before the descriptive
  statistics, producing `OR_EAD_TIT`.
- Downstream `no_conformes` reflects the new PD floor automatically.
""",
    ),
    (
        "tables/cycles.md",
        """\
# Table cycles_v[23]

Each row is one *recovery cycle* identified by `CICLO_ID`. Pairs across
versions share the same primary key.

## Key fields

- `CICLO_ID` (PK) — e.g. `CIC_00031`
- `SEGMENTO` — `CORP` | `RETAIL`
- `COLATERAL_TIPO` — `HIPOTECA` | `PERSONAL` | `NINGUNA` | `FINANCIERO`
- Numeric: `PD_ESTIMADA`, `LGD_ESTIMADA`, `EAD`, `DPDS`,
  `STAGE_IFRS9`, `RATING_GRADO`, `PROVISION_PERIOD_MONTHS`,
  `VENTANA_OBSERVACION_YEARS`, `VENTANA_CALIBRACION_YEARS`,
  `LGD_FLOOR_APLICADO`, `MOC`, `CURE_FLAG`, `RWA`, `ECL`
- V3-only: `OR_EAD_TIT` (computed)
""",
    ),
    (
        "fluxes/v2_to_v3_release_notes.md",
        """\
# Release notes: V2 → V3 (2025-Q1)

- **PD master-scale recalibration**: floor lifted to 0.05 %; ratings
  1–2 multiplied by 1.15. See `data/changelog/2025-q1-pd-recalibration.md`.
- **CORP LGD floor**: tightened from 45 % to 50 %.
- **Securitisation EAD**: introduced `OR_EAD_TIT` derived from `EAD` × a
  segment-dependent multiplier (CORP × 2.0, RETAIL × 1.5, otherwise × 1.0).
- **Collateral codes**: `FINANCIERO` renamed to `COLATERAL_FIN` per
  `data/changelog/2025-q2-collateral-codes.md`.
""",
    ),
]


def seed_docs() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for rel, body in _DOCS:
        target = DOCS / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"[seed] kept existing {target.relative_to(ROOT)}")
            continue
        target.write_text(body, encoding="utf-8")
        print(f"[seed] wrote {target.relative_to(ROOT)}")


# ── Build BM25 index ─────────────────────────────────────────────────────────


def reindex_docs() -> None:
    sys.path.insert(0, str(ROOT))
    from src.agent.docs_index import DocsIndex
    idx = DocsIndex().build()
    path = idx.save()
    print(f"[seed] indexed {idx.section_count()} sections → {path.relative_to(ROOT)}")


def main() -> None:
    seed_sas()
    seed_docs()
    reindex_docs()
    print("\nDone. Try:")
    print("  curl -X GET http://localhost:8000/agent/status | jq")
    print("  curl -N -X POST http://localhost:8000/agent/ask \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"question\": \"Why does CIC_00031 have OR_EAD_TIT in V3 but not V2?\"}'")


if __name__ == "__main__":
    main()
