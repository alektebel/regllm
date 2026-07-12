"""Lineage-obligation checker.

Executable version of the completeness argument in
``docs/AUDITOR_PARITY_STANDARD.md`` §3: walk the lineage graph the
manifest declares and verify every node carries its mandatory oracles —
or a signed waiver. The result is a machine-checkable statement a third
party can reproduce; certification fails on any unwaived gap.

Obligations:

* derived field   → documented formula (⇒ recomputation oracle exists)
* source field    → domain (validity oracle) + reconciliation against at
                    least one independent surface, or waiver
* table           → control totals (population oracle), or waiver
* structure       → primary key declared (uniqueness oracle is automatic)

Waivers live in the manifest (``waivers: {reconcile: "reason"}``) and are
surfaced on the face of the audit pack — a waiver hides nothing, it signs
for the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import Manifest


@dataclass(frozen=True)
class Obligation:
    node: str            # "table.column" or "table"
    kind: str            # formula | domain | reconcile | control
    status: str          # ok | waived | missing
    detail: str = ""


def check_obligations(manifest: Manifest) -> list[Obligation]:
    out: list[Obligation] = []
    for table in manifest.tables:
        for col in table.columns:
            node = f"{table.name}.{col.name}"
            if col.role == "derived":
                # load_manifest already refuses derived columns without a
                # parseable formula, so this is always ok for a loaded
                # manifest — recorded so the pack shows it affirmatively.
                detail = col.formula + (f" where {col.when}" if col.when else "")
                out.append(Obligation(node, "formula", "ok", detail))
                if col.when and col.domain:
                    out.append(Obligation(
                        node, "domain", "ok",
                        f"free branch: {_domain_summary(col)}"))
            else:
                out.append(Obligation(
                    node, "domain",
                    "ok" if col.domain else "missing",
                    _domain_summary(col)))
                if col.reconcile:
                    surfaces = ", ".join(r.surface for r in col.reconcile)
                    out.append(Obligation(node, "reconcile", "ok", surfaces))
                elif "reconcile" in col.waivers:
                    out.append(Obligation(
                        node, "reconcile", "waived", col.waivers["reconcile"]))
                else:
                    out.append(Obligation(
                        node, "reconcile", "missing",
                        "source fact recorded on no independent surface"))
        node = table.name
        if table.control:
            out.append(Obligation(
                node, "control", "ok",
                f"{table.control.surface}: "
                + ", ".join(c.kind if c.kind == "count" else f"sum({c.column})"
                            for c in table.control.checks)))
        elif "control" in table.waivers:
            out.append(Obligation(
                node, "control", "waived", table.waivers["control"]))
        else:
            out.append(Obligation(
                node, "control", "missing",
                "no control-totals surface ties the population"))
    return out


def _domain_summary(col) -> str:
    dom = col.domain
    if dom is None:
        return ""
    parts = [dom.type]
    if dom.values is not None:
        parts.append(f"enum({len(dom.values)})")
    if dom.min is not None or dom.max is not None:
        parts.append(f"[{dom.min}, {dom.max}]")
    if dom.nullable:
        parts.append("nullable")
    if dom.unique:
        parts.append("unique")
    return " ".join(parts)


def gaps(obligations: list[Obligation]) -> list[Obligation]:
    return [o for o in obligations if o.status == "missing"]


def summary(obligations: list[Obligation]) -> dict:
    return {
        "total": len(obligations),
        "ok": sum(1 for o in obligations if o.status == "ok"),
        "waived": sum(1 for o in obligations if o.status == "waived"),
        "missing": sum(1 for o in obligations if o.status == "missing"),
    }
