"""Field × regulation coverage matrix for the DQC system.

Answers, *computably*, the question: "are all the PD/LGD guideline articles
and all database-coherence invariants actually checked?"

The matrix spine is deterministic — it is NOT produced by an LLM:

  * **Rows**: every field of the data dictionary (``data_dictionary.md``),
    i.e. the known schema.
  * **Columns**: every regulatory reference the dictionary attaches to a
    field (CRR articles, EBA GL 2017/16 paragraphs, BCBS 239 principles,
    IFRS 9) — the *applicability map*.
  * **Cell status** for each applicable (field, ref) pair:
      - ``covered``  — a defect in ``defect_catalog`` touches the field AND
        cites the reference, so a trap + oracle exist: coverage is
        *certifiable* (the check demonstrably fires on a planted violation).
      - ``partial``  — a defect touches the field but under a different
        reference (the invariant is exercised, the citation is not).
      - ``todo``     — nothing in the catalog exercises this pair.

  100% coverage  ⇔  no ``todo`` cells  ∧  every GL section in
  ``DQC/coverage/applicability.yaml`` is either mapped to fields or signed
  off as not-applicable by a human.

Usage:

    python DQC/eval/coverage_matrix.py                    # summary table
    python DQC/eval/coverage_matrix.py --json             # machine-readable
    python DQC/eval/coverage_matrix.py --fail-under 0.8   # CI gate
    python DQC/eval/coverage_matrix.py --emit-applicability  # (re)generate
        DQC/coverage/applicability.yaml from the ingested GL paragraphs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from defect_catalog import DEFECTS  # noqa: E402

DICTIONARY_MD = HERE / "data_dictionary.md"
GL_ARTICLES_JSON = PROJECT_ROOT / "data" / "regulation" / "eba_gl_2017_16_articles.json"
APPLICABILITY_YAML = PROJECT_ROOT / "DQC" / "coverage" / "applicability.yaml"

_FIELD_ROW = re.compile(r"^\|\s*`([A-Z0-9_]+)`\s*\|(.+)\|\s*$")


# ── reference canonicalisation ──────────────────────────────────────────────

def canon_ref(ref: str) -> str:
    """Normalise a regulatory reference for matching.

    'CRR Art. 181.1(b)' and 'CRR Art. 181.1.b' both → 'crr.art.181.1.b'.
    """
    return re.sub(r"[^0-9a-z§]+", ".", ref.strip().lower()).strip(".")


def split_refs(cell: str) -> list[str]:
    """Split a dictionary/defect ref cell into individual references."""
    cell = cell.strip()
    if not cell or cell in ("—", "-"):
        return []
    return [r.strip() for r in cell.split(" / ") if r.strip()]


# ── data dictionary parsing ─────────────────────────────────────────────────

def parse_dictionary(path: Path = DICTIONARY_MD) -> dict[str, list[str]]:
    """Extract {field: [regulatory refs]} from the markdown dictionary tables.

    The dictionary's last table column is the field's regulatory anchor; a
    '—' means the field is context-only (no article applies).
    """
    fields: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _FIELD_ROW.match(line)
        if not m:
            continue
        name = m.group(1)
        cells = [c.strip() for c in m.group(2).split("|")]
        refs = split_refs(cells[-1]) if cells else []
        fields[name] = refs
    return fields


# ── matrix ──────────────────────────────────────────────────────────────────

def build_matrix() -> dict:
    """Build the coverage matrix from the dictionary + defect catalog."""
    dictionary = parse_dictionary()

    # Index the (non-decoy) defect catalog by field and by canonical ref.
    defects_by_field: dict[str, list] = {}
    for d in DEFECTS:
        if d.decoy:
            continue
        for col in d.columns:
            defects_by_field.setdefault(col, []).append(d)

    cells: list[dict] = []
    n_applicable = n_covered = n_partial = 0
    fields_uncovered: list[str] = []
    refs_todo: dict[str, int] = {}

    for field, refs in sorted(dictionary.items()):
        field_defects = defects_by_field.get(field, [])
        field_has_cover = False
        for ref in refs:
            n_applicable += 1
            c_ref = canon_ref(ref)
            matching = [
                d for d in field_defects
                if any(canon_ref(r) == c_ref for r in split_refs(d.regulation_ref))
            ]
            if matching:
                status = "covered"
                n_covered += 1
                field_has_cover = True
            elif field_defects:
                status = "partial"
                n_partial += 1
                field_has_cover = True
            else:
                status = "todo"
                refs_todo[ref] = refs_todo.get(ref, 0) + 1
            cells.append({
                "field": field, "ref": ref, "status": status,
                "defects": [d.defect_id for d in (matching or field_defects)],
            })
        if refs and not field_has_cover:
            fields_uncovered.append(field)

    coverage_pct = (n_covered + n_partial) / n_applicable if n_applicable else 0.0
    strict_pct = n_covered / n_applicable if n_applicable else 0.0
    return {
        "n_fields": len(dictionary),
        "n_applicable_cells": n_applicable,
        "covered": n_covered,
        "partial": n_partial,
        "todo": n_applicable - n_covered - n_partial,
        "coverage": round(coverage_pct, 4),          # covered or partial
        "strict_coverage": round(strict_pct, 4),     # exact article match
        "fields_uncovered": fields_uncovered,
        "refs_todo": dict(sorted(refs_todo.items())),
        "cells": cells,
    }


# ── GL applicability skeleton ───────────────────────────────────────────────

# Keyword → schema-field mapping used to *suggest* fields per GL section.
# Suggestions are a starting point for the human review pass, never a
# conclusion: every entry is emitted with ``review: pending``.
_SECTION_FIELD_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"margen de cautela|deficiencias", re.I),
     ["MOC", "MOC_CAT_A", "MOC_CAT_B", "MOC_CAT_C", "LGD_CON_MOC"]),
    (re.compile(r"\bPD\b|tasas? de default|rating", re.I),
     ["PD_ESTIMADA", "PD_SUELO", "PD_FINAL", "RATING_GRADO", "CALIBRATION_SEGMENT"]),
    (re.compile(r"\bLGD\b|pérdida económica|realised", re.I),
     ["LGD_ESTIMADA", "LGD_REALIZADA", "LGD_SUELO", "LGD_FINAL", "LGD_DOWNTURN"]),
    (re.compile(r"garantías? reales|colateral", re.I),
     ["COLATERAL_TIPO", "VALOR_COLATERAL", "VALOR_COLATERAL_INICIAL",
      "HAIRCUT", "MES_VALORACION_COLATERAL", "LTV"]),
    (re.compile(r"tasa de descuento", re.I), ["TASA_DESCUENTO"]),
    (re.compile(r"comisiones|intereses|disposiciones", re.I),
     ["INTERESES_ACUMULADOS", "TIPO_INTERES_ORIGINAL", "TIPO_INTERES_ACTUAL"]),
    (re.compile(r"periodo histórico|observación", re.I),
     ["VENTANA_OBSERVACION_YEARS", "FLAG_NC"]),
    (re.compile(r"recuperaci", re.I),
     ["RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO", "TERMINACION",
      "ADJUDICACION_FLAG", "ADJUDICACION_TIPO", "ADJUDICACION_VALOR"]),
    (re.compile(r"calidad de los datos|requisitos de los datos", re.I),
     ["ID_CONTR_CICLO_LGD", "ID_CONTRATO", "ID_CLIENTE", "MES_CICLO"]),
    (re.compile(r"\bEL\b|en default", re.I),
     ["ECL", "PROVISION", "STAGE_IFRS9", "DPDS", "CURE_FLAG"]),
    (re.compile(r"fechas de referencia", re.I),
     ["MES_CICLO", "MES_DEFAULT", "MES_CIERRE_CICLO"]),
]

# Purely procedural GL sections (compliance obligations, addressees,
# application dates) that place no requirement on data content.
_NOT_APPLICABLE = re.compile(
    r"obligaciones de cumplimiento|objeto|ámbito de aplicación|destinatarios"
    r"|definiciones|fecha de aplicación|primera aplicación", re.I)


def emit_applicability(out: Path = APPLICABILITY_YAML) -> int:
    """Generate the GL-section → field applicability skeleton for review.

    Groups the ingested 221 GL 2017/16 paragraphs by (section, subsection)
    and heuristically suggests the schema fields each governs. Every entry
    carries ``review: pending`` — the file only becomes an authority once a
    human flips entries to ``review: approved`` (or marks them
    ``applicable: false`` with a reason).
    """
    paragraphs = json.loads(GL_ARTICLES_JSON.read_text(encoding="utf-8"))
    groups: dict[tuple[str, str], list[int]] = {}
    for p in paragraphs:
        groups.setdefault((p["section"], p.get("subsection", "")), []).append(
            p["paragraph"])

    lines = [
        "# EBA GL/2017/16 — section-level applicability map (PD & LGD estimation)",
        "#",
        "# Generated by DQC/eval/coverage_matrix.py --emit-applicability from the",
        "# 221 ingested paragraphs. `fields` are heuristic SUGGESTIONS; each entry",
        "# must be human-reviewed: set `review: approved`, adjust `fields`, or set",
        "# `applicable: false` with a `reason`. 100% article coverage means every",
        "# entry here is approved AND every applicable entry's fields appear as",
        "# covered cells in the coverage matrix.",
        "",
        "sections:",
    ]
    for (section, subsection), pars in groups.items():
        pars = sorted(pars)
        title = f"{section}" + (f" — {subsection}" if subsection else "")
        applicable = not _NOT_APPLICABLE.search(title)
        fields: list[str] = []
        if applicable:
            for pat, fs in _SECTION_FIELD_HINTS:
                if pat.search(title):
                    fields.extend(f for f in fs if f not in fields)
        lines.append(f'  - title: "{title}"')
        lines.append(f"    paragraphs: [{pars[0]}, {pars[-1]}]")
        lines.append(f"    applicable: {'true' if applicable else 'false'}")
        if not applicable:
            lines.append('    reason: "procedural section — no data requirement"')
        if fields:
            lines.append(f"    fields: [{', '.join(fields)}]")
        elif applicable:
            lines.append("    fields: []   # TODO: assign during review")
        lines.append("    review: pending")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(groups)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _print_matrix(m: dict) -> None:
    print(f"COVERAGE MATRIX: {m['n_fields']} fields, "
          f"{m['n_applicable_cells']} applicable (field × article) cells")
    print(f"  covered (exact article match): {m['covered']}  "
          f"({m['strict_coverage']:.0%})")
    print(f"  partial (field exercised, other article): {m['partial']}")
    print(f"  todo: {m['todo']}")
    print(f"  coverage (covered+partial): {m['coverage']:.0%}")
    if m["fields_uncovered"]:
        print(f"\nFields with regulatory refs but NO defect coverage:")
        for f in m["fields_uncovered"]:
            print(f"  - {f}")
    if m["refs_todo"]:
        print(f"\nReferences with todo cells:")
        for ref, n in m["refs_todo"].items():
            print(f"  - {ref}  ({n} cell(s))")


def main() -> None:
    ap = argparse.ArgumentParser(description="DQC coverage matrix")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-under", type=float, default=None, metavar="FRACTION",
                    help="exit 1 if coverage (covered+partial) is below this")
    ap.add_argument("--emit-applicability", action="store_true",
                    help="regenerate DQC/coverage/applicability.yaml")
    args = ap.parse_args()

    if args.emit_applicability:
        n = emit_applicability()
        print(f"Wrote {APPLICABILITY_YAML} ({n} GL sections)")
        return

    m = build_matrix()
    if args.json:
        print(json.dumps(m, indent=2, ensure_ascii=False))
    else:
        _print_matrix(m)
    if args.fail_under is not None and m["coverage"] < args.fail_under:
        print(f"\nFAIL: coverage {m['coverage']:.0%} < required "
              f"{args.fail_under:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
