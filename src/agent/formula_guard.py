"""Verify that agent answers cite SAS formulas grounded in tool output."""

from __future__ import annotations

import json
import re
from typing import Any

_LINEAGE_TOOLS = frozenset({
    "trace_causal_chain", "query_cycle_data",
    "trace_dependencies", "inspect_lineage", "get_sas_formula",
})

# Numeric coefficient × field (0.05 * LGD, 0.30 × LGD_FLOOR, …)
_COEF_FORMULA = re.compile(
    r"\b\d+(?:\.\d+)?\s*[×x\*]\s*[A-Z_][A-Z0-9_]*",
    re.I,
)
_MAX_FUNC = re.compile(r"\bMAX\s*\([^)]{3,120}\)", re.I)
_EQUALS_FORMULA = re.compile(
    r"\b(MoC|ECL|EAD|LGD(?:_[A-Z0-9]+)?|PD(?:_[A-Z0-9]+)?|RWA|K_IRB)\s*=\s*[^\n.]{3,120}",
    re.I,
)
_SUSPICIOUS_DATASETS = re.compile(r"\bcatalogo\.[a-z_]+", re.I)
_SUSPICIOUS_FIELDS = re.compile(r"\bEAD_BASILEG\b|\bEAD_FUSION\b", re.I)
_SAS_FILE_CLAIM = re.compile(r"\bproj_\d+_[\w]+\.sas\b", re.I)

_CLAIM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_COEF_FORMULA, "coeficiente numérico"),
    (_MAX_FUNC, "función MAX"),
    (_EQUALS_FORMULA, "asignación con fórmula"),
    (_SUSPICIOUS_DATASETS, "dataset no verificado"),
    (_SUSPICIOUS_FIELDS, "campo no presente en sesión"),
]


def _norm(s: str) -> str:
    s = s.upper().strip()
    s = s.replace("×", "*")
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=[A-Z_])", " * ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_grounding_corpus(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "tool":
            parts.append(str(m.get("content") or ""))
    return "\n".join(parts)


def sas_lineage_tools_used(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        if m.get("role") == "tool" and m.get("name") in _LINEAGE_TOOLS:
            return True
    return False


def extract_grounded_exprs(corpus: str) -> set[str]:
    exprs: set[str] = set()
    for m in re.finditer(r'"expr"\s*:\s*"((?:\\.|[^"\\])*)"', corpus):
        raw = m.group(1).encode().decode("unicode_escape", errors="replace")
        if raw.strip():
            exprs.add(_norm(raw))
    # get_sas_formula definitions[].expr
    try:
        for chunk in re.findall(r"\{[^{}]*\"definitions\"[^{}]*\}", corpus):
            if "definitions" not in chunk:
                continue
            data = json.loads(chunk)
            for d in data.get("definitions") or []:
                ex = (d.get("expr") or "").strip()
                if ex:
                    exprs.add(_norm(ex))
    except (json.JSONDecodeError, TypeError):
        pass
    return exprs


def _claim_grounded(claim: str, corpus: str, grounded_exprs: set[str]) -> bool:
    nc = _norm(claim)
    ncorpus = _norm(corpus)
    if nc in ncorpus:
        return True
    # Allow paraphrase when core expr is in tool output
    for ex in grounded_exprs:
        if not ex:
            continue
        if ex in nc or nc in ex:
            return True
        # Strip field = prefix for equals-style claims
        rhs = nc.split("=", 1)[-1].strip() if "=" in nc else nc
        if ex in rhs or rhs in ex:
            return True
    return False


def check_formula_grounding(
    answer: str,
    messages: list[dict[str, Any]],
    *,
    max_violations: int = 8,
) -> list[str]:
    """Return human-readable violations; empty list means grounded enough."""
    if not answer or not answer.strip():
        return []

    corpus = build_grounding_corpus(messages)
    grounded_exprs = extract_grounded_exprs(corpus)
    ncorpus = _norm(corpus)
    violations: list[str] = []

    has_formula_claim = False
    for pat, _label in _CLAIM_PATTERNS:
        if pat.search(answer):
            has_formula_claim = True
            break

    if has_formula_claim and not sas_lineage_tools_used(messages):
        violations.append(
            "citaste una fórmula sin llamar trace_dependencies, inspect_lineage o get_sas_formula"
        )

    for pat, label in _CLAIM_PATTERNS:
        for m in pat.finditer(answer):
            claim = m.group(0).strip()
            if _claim_grounded(claim, corpus, grounded_exprs):
                continue
            violations.append(f"{label}: «{claim}»")
            if len(violations) >= max_violations:
                return violations

    for m in _SAS_FILE_CLAIM.finditer(answer):
        fname = m.group(0)
        if fname.lower() not in corpus.lower() and _norm(fname) not in ncorpus:
            violations.append(f"archivo SAS no verificado: «{fname}»")
            if len(violations) >= max_violations:
                return violations

    return violations
