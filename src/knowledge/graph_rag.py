"""GraphRAG over the change-log graph.

For each suspect field flagged by the discrepancy explainer, we
retrieve the relevant subgraph (1–2 hop neighbourhood of the ``Field``
node), linearise it into a context block, and ask the local LLM whether
the delta is justified by a documented change.

Output is a structured per-field verdict plus an overall summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from .change_log_graph import build_graph, load_graph, save_graph
from .llm_client import LocalLLMClient, get_client

logger = logging.getLogger(__name__)


_DEFAULT_GRAPH = Path("data/changelog/graph.json")
_DEFAULT_CHANGELOG_DIR = Path("data/changelog")
_DEFAULT_SCHEMA_FILES = [Path("data/samples/irb_schema.sql")]


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class FieldJustification:
    field: str
    justified: bool
    confidence: float
    rationale: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "justified": self.justified,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }


@dataclass
class GraphRAGReport:
    overall: str
    per_field: list[FieldJustification]
    backend: str
    model: str

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "per_field": [pf.to_dict() for pf in self.per_field],
            "backend": self.backend,
            "model": self.model,
        }


# ── Subgraph retrieval ───────────────────────────────────────────────────────


def _field_node_id(field_name: str) -> str:
    return f"field:{field_name.upper()}"


def field_subgraph(g: nx.DiGraph, field_name: str, hops: int = 2) -> nx.DiGraph:
    """Return the ``hops``-hop neighbourhood of the field node (undirected)."""
    fid = _field_node_id(field_name)
    if fid not in g:
        return nx.DiGraph()
    seeds = {fid}
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        new_frontier: set[str] = set()
        for n in frontier:
            new_frontier.update(g.predecessors(n))
            new_frontier.update(g.successors(n))
        new_frontier -= visited
        visited |= new_frontier
        frontier = new_frontier
        if not frontier:
            break
    return g.subgraph(visited).copy()


def linearise_subgraph(g: nx.DiGraph, field_name: str) -> str:
    """Render the subgraph as a textual block for the LLM."""
    if g.number_of_nodes() == 0:
        return f"(no documented changes mention field {field_name})"
    lines: list[str] = []
    fid = _field_node_id(field_name)

    # 1. Sections that mention this field
    section_nodes = [
        n for n, d in g.nodes(data=True)
        if d.get("type") == "Section"
        and any(t == fid for t in g.successors(n))
    ]
    for sec in section_nodes:
        d = g.nodes[sec]
        doc = d.get("doc", "")
        head = d.get("label", "")
        body = d.get("body", "")
        lines.append(f"### {doc} → {head}")
        if body:
            lines.append(body[:600])
        # Justification edges?
        if g.has_edge(sec, fid):
            rel = g[sec][fid].get("relation", "")
            if rel == "JUSTIFIES":
                lines.append(f"  [graph] this section JUSTIFIES {field_name}")
        lines.append("")

    # 2. Rename edges
    for src, dst, d in g.edges(data=True):
        if d.get("relation") == "CHANGES_FROM_TO":
            s = g.nodes[src].get("label", src)
            t = g.nodes[dst].get("label", dst)
            if s == field_name.upper() or t == field_name.upper():
                lines.append(f"  [graph] {s}  →  {t}  (rename)")

    # 3. Schema nodes
    for n, d in g.nodes(data=True):
        if d.get("type") == "TableChange":
            cols = [g.nodes[c].get("label", c) for c in g.successors(n)
                    if g.nodes[c].get("type") == "Field"]
            if field_name.upper() in cols:
                lines.append(f"  [schema] table {d.get('label', n)} declares column {field_name}")

    if not lines:
        lines.append(f"(graph has node for {field_name} but no descriptive sections)")
    return "\n".join(lines)


def collect_evidence(g: nx.DiGraph, field_name: str) -> list[dict[str, Any]]:
    """Structured citation list for a field — used by the frontend."""
    fid = _field_node_id(field_name)
    if fid not in g:
        return []
    out: list[dict[str, Any]] = []
    for sec in g.predecessors(fid):
        d = g.nodes[sec]
        if d.get("type") != "Section":
            continue
        doc = d.get("doc", "")
        out.append({
            "section_id": sec,
            "document": doc.removeprefix("doc:") if isinstance(doc, str) else doc,
            "heading": d.get("label", ""),
            "snippet": (d.get("body", "") or "")[:300],
            "justifies": g.has_edge(sec, fid) and g[sec][fid].get("relation") == "JUSTIFIES",
        })
    return out


# ── Prompts ──────────────────────────────────────────────────────────────────


_SYSTEM = (
    "You are a data-lineage auditor. Given a numeric or categorical change in "
    "a field between table version V2 and V3, plus excerpts from the database "
    "change log, decide whether the change is JUSTIFIED by the documented "
    "changes. Reply with strict JSON only — no markdown."
)

_USER_TEMPLATE = """\
Field: {field}
Change V2 → V3: {delta}
Attribution to ΔY (the target field {target}): {contribution}
Method(s): {methods}

Change-log excerpts mentioning {field}:
{context}

Reply with JSON of the form:
{{
  "justified": true|false,
  "confidence": 0.0..1.0,
  "rationale": "1-2 sentence explanation",
  "evidence": [{{"document": "...", "heading": "...", "quote": "..."}}]
}}
"""


# ── Main API ─────────────────────────────────────────────────────────────────


class GraphRAG:
    """Combines a change-log graph with a Gemma client for grounded justifications.

    Supports two backends:
    - **NetworkX** (legacy): loads from ``graph.json``, in-memory graph.
    - **GraphStore** (KuzuDB): when ``graph_store`` is provided, queries
      are routed to the persistent knowledge graph.  The NetworkX graph
      is still built on-demand for ``linearise_subgraph`` compatibility.
    """

    def __init__(
        self,
        graph_path: Path | str = _DEFAULT_GRAPH,
        changelog_dir: Path | str = _DEFAULT_CHANGELOG_DIR,
        schema_files: list[Path | str] | None = None,
        client: LocalLLMClient | None = None,
        graph_store: Any | None = None,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.changelog_dir = Path(changelog_dir)
        self.schema_files = [Path(p) for p in (schema_files or _DEFAULT_SCHEMA_FILES)]
        self.client = client or get_client()
        self._graph: nx.DiGraph | None = None
        self._graph_store = graph_store  # GraphStore instance (optional)

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            if self._graph_store is not None:
                # Build a NetworkX view from the KuzuDB store
                self._graph = self._graph_store.to_networkx()
            elif self.graph_path.exists():
                self._graph = load_graph(self.graph_path)
            else:
                self._graph = self.reindex(persist=True)
        return self._graph

    def reindex(self, persist: bool = True) -> nx.DiGraph:
        """Rebuild the graph from the source files."""
        g = build_graph(self.changelog_dir, schema_files=self.schema_files)
        if persist:
            save_graph(g, self.graph_path)
        self._graph = g
        return g

    def justify_field(
        self,
        field_name: str,
        *,
        delta: str,
        target: str,
        contribution: float,
        methods: Iterable[str] = ("gradient", "shapley"),
        hops: int = 2,
    ) -> FieldJustification:
        """Ask Gemma whether the field's V2→V3 change is justified."""
        g = self.graph
        sub = field_subgraph(g, field_name, hops=hops)
        context = linearise_subgraph(sub, field_name)
        evidence = collect_evidence(g, field_name)

        user = _USER_TEMPLATE.format(
            field=field_name,
            delta=delta,
            target=target,
            contribution=f"{contribution:+.6g}",
            methods=", ".join(methods),
            context=context[:3000],
        )
        try:
            data = self.client.chat_json(_SYSTEM, user)
        except Exception as e:
            logger.warning("LLM call failed for %s: %s", field_name, e)
            data = {"justified": False, "confidence": 0.0,
                    "rationale": f"LLM call failed: {e}", "evidence": []}

        return FieldJustification(
            field=field_name,
            justified=bool(data.get("justified", False)),
            confidence=float(data.get("confidence", 0.0)),
            rationale=str(data.get("rationale", "")),
            evidence=data.get("evidence") or evidence,
        )

    def justify_report(
        self,
        target: str,
        attributions: list[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> GraphRAGReport:
        """Justify the top-k contributing fields and produce an overall summary."""
        per_field: list[FieldJustification] = []
        for a in attributions[:top_k]:
            field_name = a["field"]
            di = a.get("delta_input")
            if isinstance(di, list) and len(di) == 2:
                delta_str = f"{di[0]!r} → {di[1]!r}"
            elif isinstance(di, (int, float)):
                delta_str = f"{di:+.6g}"
            else:
                delta_str = str(di)
            per_field.append(self.justify_field(
                field_name=field_name,
                delta=delta_str,
                target=target,
                contribution=float(a.get("contribution", 0.0)),
                methods=[a.get("_method", "gradient/shapley")],
            ))

        # Overall summary
        n_just = sum(1 for j in per_field if j.justified)
        n_unjust = len(per_field) - n_just
        overall = (
            f"{n_just} of {len(per_field)} top contributing fields are documented "
            f"as justified V2→V3 changes; {n_unjust} are NOT."
        )

        backend = self.client.detect_backend()
        model = (
            self.client.litert_model if backend == "litert"
            else self.client.ollama_model if backend == "ollama"
            else "stub"
        )
        return GraphRAGReport(
            overall=overall,
            per_field=per_field,
            backend=backend,
            model=model,
        )
