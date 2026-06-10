"""V2 vs V3 SAS code comparator.

Given two folders of SAS scripts (``data/sas/v2/*.sas`` and
``data/sas/v3/*.sas``), build a focused diff that surfaces only the data
steps which produce or consume a *target* field — so the agent can point
at the bit of code likely responsible for the discrepancy.

The diff is structured (``added_steps`` / ``removed_steps`` /
``modified_steps``) plus a unified text diff scoped to those steps.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.sas_logic_tree import (
    AnyNode,
    AssignNode,
    DataStepNode,
    FilterNode,
    IfNode,
    SASLogicTree,
    SelectNode,
)


_SAS_ROOT = Path("data/sas")


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_version(version: str, root: Path | None = None) -> str:
    """Concatenate every ``.sas`` file under ``data/sas/{version}/`` (sorted by name)."""
    root = root or _SAS_ROOT
    folder = root / version
    if not folder.exists():
        return ""
    files = sorted(folder.rglob("*.sas"))
    return "\n\n".join(f.read_text(encoding="utf-8") for f in files)


def load_pair(root: Path | None = None) -> tuple[str, str]:
    return load_version("v2", root), load_version("v3", root)


# ── AST helpers ──────────────────────────────────────────────────────────────


@dataclass
class StepFingerprint:
    """A normalised summary of a DATA step useful for diffing."""

    output: str
    inputs: tuple[str, ...]
    writes: tuple[str, ...]                 # fields assigned in this step
    reads: tuple[str, ...]                  # fields read in this step
    body_text: str                          # canonical pretty-printed body
    line_range: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "inputs": list(self.inputs),
            "writes": list(self.writes),
            "reads": list(self.reads),
            "body_text": self.body_text,
            "line_range": list(self.line_range) if self.line_range else None,
        }


def _walk_writes_reads(body: list[AnyNode]) -> tuple[set[str], set[str]]:
    writes: set[str] = set()
    reads: set[str] = set()

    def walk(nodes: list[AnyNode]) -> None:
        for n in nodes:
            if isinstance(n, AssignNode):
                writes.add(n.var.upper())
                # Anything in the expression that isn't a literal looks like a read
                for tok in _expr_idents(n.expr):
                    reads.add(tok.upper())
            elif isinstance(n, IfNode):
                for tok in _expr_idents(n.condition):
                    reads.add(tok.upper())
                walk(n.then_branch)
                walk(n.else_branch)
            elif isinstance(n, SelectNode):
                if n.select_expr:
                    for tok in _expr_idents(n.select_expr):
                        reads.add(tok.upper())
                for w in n.whens:
                    for v in w.values:
                        for tok in _expr_idents(v):
                            reads.add(tok.upper())
                    walk(w.body)
                walk(n.otherwise)
            elif isinstance(n, FilterNode):
                for tok in _expr_idents(n.condition):
                    reads.add(tok.upper())
    walk(body)
    return writes, reads


_IDENT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


def _expr_idents(expr: str) -> list[str]:
    """Cheap identifier extractor — strips literals/operators, keeps tokens."""
    if not expr:
        return []
    tokens: list[str] = []
    cur: list[str] = []
    in_str: str | None = None
    for ch in expr:
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch in _IDENT_CHARS:
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    # Drop pure-numeric tokens and a small SAS keyword stop-list
    stop = {
        "AND", "OR", "NOT", "IF", "THEN", "ELSE", "DO", "END", "SELECT",
        "WHEN", "OTHERWISE", "MIN", "MAX", "ABS", "SUM", "TRUE", "FALSE",
        "MISSING", "NULL", "IN", "LIKE", "WHERE",
    }
    out: list[str] = []
    for t in tokens:
        if t.replace(".", "", 1).isdigit():
            continue
        if t.upper() in stop:
            continue
        if not t:
            continue
        out.append(t)
    return out


def _fingerprint(step: DataStepNode) -> StepFingerprint:
    writes, reads = _walk_writes_reads(step.body)
    inputs = tuple(d.upper() for d in (step.merge_datasets or []))
    body_text = step.display(indent=0).strip()
    return StepFingerprint(
        output=step.output_dataset.upper(),
        inputs=inputs,
        writes=tuple(sorted(writes)),
        reads=tuple(sorted(reads)),
        body_text=body_text,
    )


def _data_steps(nodes: list[AnyNode]) -> list[DataStepNode]:
    return [n for n in nodes if isinstance(n, DataStepNode)]


# ── Per-target scoping ──────────────────────────────────────────────────────


def _ancestors_of(target: str, fingerprints: list[StepFingerprint]) -> set[str]:
    """Return the set of step outputs (and field names) whose values flow into ``target``.

    Walks the step graph backwards: a step S is an ancestor if it writes any
    field that another ancestor reads. Returns the union of writes covered.
    """
    target_u = target.upper()
    ancestor_steps: set[str] = set()
    needed: set[str] = {target_u}
    changed = True
    while changed:
        changed = False
        for s in fingerprints:
            if s.output in ancestor_steps:
                continue
            if needed & set(s.writes):
                ancestor_steps.add(s.output)
                # everything this step reads becomes part of the dependency set
                before = len(needed)
                needed.update(s.reads)
                if len(needed) != before:
                    changed = True
                changed = True
    return ancestor_steps


# ── Public diff API ──────────────────────────────────────────────────────────


@dataclass
class CodeDiffResult:
    target: str | None
    added_steps: list[dict[str, Any]] = field(default_factory=list)
    removed_steps: list[dict[str, Any]] = field(default_factory=list)
    modified_steps: list[dict[str, Any]] = field(default_factory=list)
    unchanged_steps: list[str] = field(default_factory=list)
    unified_diff: str = ""
    v2_present: bool = False
    v3_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "added_steps": self.added_steps,
            "removed_steps": self.removed_steps,
            "modified_steps": self.modified_steps,
            "unchanged_steps": self.unchanged_steps,
            "unified_diff": self.unified_diff,
            "v2_present": self.v2_present,
            "v3_present": self.v3_present,
        }


def compare(
    target: str | None = None,
    *,
    sas_v2: str | None = None,
    sas_v3: str | None = None,
    root: Path | None = None,
) -> CodeDiffResult:
    """Compare V2 and V3 SAS scripts; if ``target`` is given, scope to its data steps."""
    if sas_v2 is None or sas_v3 is None:
        loaded2, loaded3 = load_pair(root)
        sas_v2 = sas_v2 if sas_v2 is not None else loaded2
        sas_v3 = sas_v3 if sas_v3 is not None else loaded3

    res = CodeDiffResult(target=target.upper() if target else None)
    res.v2_present = bool(sas_v2.strip())
    res.v3_present = bool(sas_v3.strip())

    tree = SASLogicTree()
    nodes_v2 = tree.parse(sas_v2) if sas_v2 else []
    nodes_v3 = tree.parse(sas_v3) if sas_v3 else []
    fps_v2 = [_fingerprint(s) for s in _data_steps(nodes_v2)]
    fps_v3 = [_fingerprint(s) for s in _data_steps(nodes_v3)]

    # Restrict to the target's lineage when requested
    if target:
        anc_v2 = _ancestors_of(target, fps_v2)
        anc_v3 = _ancestors_of(target, fps_v3)
        scoped_v2 = [s for s in fps_v2 if s.output in anc_v2]
        scoped_v3 = [s for s in fps_v3 if s.output in anc_v3]
        # Also include any step that writes the target directly even if its
        # pretty body lacks reads (defensive)
        target_u = target.upper()
        for s in fps_v2:
            if target_u in s.writes and s.output not in anc_v2:
                scoped_v2.append(s)
        for s in fps_v3:
            if target_u in s.writes and s.output not in anc_v3:
                scoped_v3.append(s)
    else:
        scoped_v2, scoped_v3 = fps_v2, fps_v3

    by_v2 = {s.output: s for s in scoped_v2}
    by_v3 = {s.output: s for s in scoped_v3}
    all_outputs = sorted(set(by_v2) | set(by_v3))

    text_v2_lines: list[str] = []
    text_v3_lines: list[str] = []

    for out in all_outputs:
        s2 = by_v2.get(out)
        s3 = by_v3.get(out)
        if s2 and s3 and s2.body_text == s3.body_text:
            res.unchanged_steps.append(out)
            text_v2_lines.append(f"# step {out}\n{s2.body_text}\n")
            text_v3_lines.append(f"# step {out}\n{s3.body_text}\n")
            continue
        if s2 and not s3:
            res.removed_steps.append(s2.to_dict())
            text_v2_lines.append(f"# step {out}\n{s2.body_text}\n")
            text_v3_lines.append(f"# step {out}\n# (removed in V3)\n")
            continue
        if s3 and not s2:
            res.added_steps.append(s3.to_dict())
            text_v2_lines.append(f"# step {out}\n# (new in V3)\n")
            text_v3_lines.append(f"# step {out}\n{s3.body_text}\n")
            continue
        # Modified
        assert s2 and s3
        res.modified_steps.append({
            "output": out,
            "writes_v2": list(s2.writes),
            "writes_v3": list(s3.writes),
            "reads_v2": list(s2.reads),
            "reads_v3": list(s3.reads),
            "body_v2": s2.body_text,
            "body_v3": s3.body_text,
            "writes_added": sorted(set(s3.writes) - set(s2.writes)),
            "writes_removed": sorted(set(s2.writes) - set(s3.writes)),
        })
        text_v2_lines.append(f"# step {out}\n{s2.body_text}\n")
        text_v3_lines.append(f"# step {out}\n{s3.body_text}\n")

    res.unified_diff = "".join(
        difflib.unified_diff(
            "\n".join(text_v2_lines).splitlines(keepends=True),
            "\n".join(text_v3_lines).splitlines(keepends=True),
            fromfile="V2",
            tofile="V3",
            n=2,
        )
    )
    return res
