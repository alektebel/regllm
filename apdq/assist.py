"""Proposal queue: where LLM assistance plugs in — outside the trust chain.

The certification path never consults a model. What a model (the RegLLM
DQC agent, or any other) MAY do is populate *proposals* that a human then
reviews and signs:

* ``propose_manifest`` introspects an existing SQLite reporting extract
  and emits a draft binding manifest with every judgment call left as an
  explicit ``TODO`` (concept bindings, formulas, reconciliations,
  constraints). An LLM with the data dictionary and the regulation RAG
  fills the TODOs; ``load_manifest`` then refuses the draft until every
  derived column has a formula and every concept resolves — the review
  pressure is structural, not procedural.
* Requirement rows for the register are proposed the same way: extracted
  by a model, landed with ``status: pending``, unusable for certification
  until a human sets ``signed`` + ``signed_by`` (see ``register.py``).

This module is deliberately tiny: the MVP contract is the *queue format*,
not the model integration. Wire ``api/routers/dqc.py`` output into these
drafts as a follow-up (see apdq/README.md, expansion E7).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _guess_domain(sample: list) -> dict:
    types = {type(v) for v in sample if v is not None}
    if types <= {int}:
        dtype = "int"
        if sample and all(
                isinstance(v, int) and 190001 <= v <= 210012 and 1 <= v % 100 <= 12
                for v in sample if v is not None):
            dtype = "yyyymm"
    elif types <= {int, float}:
        dtype = "real"
    else:
        dtype = "text"
    domain: dict = {"type": dtype}
    non_null = [v for v in sample if v is not None]
    if len(non_null) < len(sample):
        domain["nullable"] = True
    if dtype in ("int", "real", "yyyymm") and non_null:
        domain["min"] = min(non_null)
        domain["max"] = max(non_null)
    if dtype == "text" and non_null and len(set(non_null)) <= 12:
        domain["values"] = sorted(set(non_null))
    return domain


def propose_manifest(db_path: Path | str, name: str,
                     sample_rows: int = 500) -> str:
    """Draft manifest YAML from a SQLite extract. Every binding decision
    is a TODO for review; the draft does not load until resolved."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    lines = [
        "# APDQ draft manifest — machine-proposed, NOT reviewable-free.",
        "# Every 'TODO' is a human judgment: concept binding, source-vs-",
        "# derived role, formulas, reconciliation surfaces, constraints.",
        "# apdq validate will refuse this file until they are resolved.",
        "",
        "apdq_manifest: 1",
        f"name: {name}",
        "",
        "tables:",
    ]
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'")]
    for table in tables:
        cols = list(conn.execute(f"PRAGMA table_info({table})"))
        col_names = [c[1] for c in cols]
        rows = conn.execute(
            f"SELECT {', '.join(col_names)} FROM {table} "
            f"LIMIT {sample_rows}").fetchall()
        lines.append(f"  - name: {table}")
        lines.append(f"    primary_key: TODO  # candidate: {col_names[0]}")
        lines.append("    columns:")
        for idx, col in enumerate(col_names):
            sample = [r[idx] for r in rows]
            domain = _guess_domain(sample)
            dom = ", ".join(
                f"{k}: {v}" for k, v in domain.items())
            lines.append(f"      - name: {col}")
            lines.append("        concept: TODO")
            lines.append("        role: source  # TODO: derived? then add formula")
            lines.append(f"        domain: {{{dom}}}")
        lines.append("")
    conn.close()
    return "\n".join(lines)
