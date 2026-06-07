"""Change-log graph builder.

Ingests:

- ``data/changelog/*.md`` — release notes, runbook entries, etc.
- ``data/samples/irb_schema.sql`` — DDL files representing schema versions.

Builds a NetworkX directed graph with these node types:

- ``Document``: one per ingested file
- ``Section``: one per H1/H2/H3 heading in markdown
- ``TableChange``: one per detected schema table mention
- ``Field``: one per detected column mention (``UPPERCASE_FIELD_NAME``)

Edges:

- ``CONTAINS``  Document → Section, Section → TableChange / Field
- ``MENTIONS_FIELD``  Section → Field
- ``CHANGES_FROM_TO``  Field → Field (rare; explicit "X renamed to Y")
- ``JUSTIFIES``  Section → Field   (when section explicitly justifies a field change)

The graph is persisted to ``data/changelog/graph.json`` so the API can
serve it without re-ingesting on every request.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


# Field names look like ALL_CAPS_WITH_UNDERSCORES, length 3+
_FIELD_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")

# Common SQL words to exclude from field detection
_FIELD_STOP = {
    "AND", "OR", "NOT", "NULL", "TRUE", "FALSE", "TABLE", "COLUMN",
    "INSERT", "UPDATE", "DELETE", "SELECT", "FROM", "WHERE", "JOIN",
    "INNER", "OUTER", "LEFT", "RIGHT", "ON", "AS", "GROUP", "ORDER",
    "BY", "HAVING", "LIMIT", "OFFSET", "INTO", "VALUES", "WITH",
    "CREATE", "DROP", "ALTER", "ADD", "REMOVE", "RENAME", "TO",
    "PRIMARY", "FOREIGN", "KEY", "REFERENCES", "INDEX", "UNIQUE",
    "DEFAULT", "CHECK", "CONSTRAINT", "VARCHAR", "INTEGER", "FLOAT",
    "DECIMAL", "NUMERIC", "TIMESTAMP", "DATE", "TIME", "BOOLEAN",
    "TEXT", "BLOB", "REAL", "DOUBLE", "BIGINT", "SMALLINT", "TINYINT",
    "ID", "NAME", "TYPE", "VERSION", "V2", "V3", "V1",
    "TODO", "FIXME", "NOTE", "WARNING", "ERROR", "DEBUG", "INFO",
    "RAG", "LLM", "API", "URL", "HTTP", "JSON", "YAML", "CSV", "SQL",
    "SAS", "ETL", "DDL", "DML", "DQL", "ACID", "OLAP", "OLTP",
    "ETC", "USA", "EU", "UK",
}

# A simple alias scheme: "rename X to Y" → CHANGES_FROM_TO edge X → Y
_RENAME_RE = re.compile(
    r"\b(?:rename(?:d|s)?|renombr(?:a|ar|ado))\s+([A-Z][A-Z0-9_]{2,})\s+(?:to|a)\s+([A-Z][A-Z0-9_]{2,})\b",
    re.IGNORECASE,
)


@dataclass
class GraphPaths:
    changelog_dir: Path
    schema_files: list[Path]
    out_path: Path


# ── Builders ─────────────────────────────────────────────────────────────────


def _split_markdown(text: str) -> list[tuple[str, int, str]]:
    """Split a markdown document into ``(heading, level, body)`` triples.

    The first chunk before any heading is given the heading "(intro)".
    """
    sections: list[tuple[str, int, list[str]]] = [("(intro)", 0, [])]
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            sections.append((m.group(2).strip(), len(m.group(1)), []))
        else:
            sections[-1][2].append(line)
    return [(h, lvl, "\n".join(body).strip()) for h, lvl, body in sections]


def _extract_fields(text: str) -> set[str]:
    fields = set()
    for m in _FIELD_RE.finditer(text):
        token = m.group(1)
        if token in _FIELD_STOP:
            continue
        if "_" not in token and len(token) < 4:
            continue  # require >=4 chars or an underscore to look field-like
        fields.add(token)
    return fields


def _extract_renames(text: str) -> list[tuple[str, str]]:
    out = []
    for m in _RENAME_RE.finditer(text):
        out.append((m.group(1).upper(), m.group(2).upper()))
    return out


def _ingest_markdown(graph: nx.DiGraph, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    doc_id = f"doc:{path.name}"
    graph.add_node(doc_id, type="Document", label=path.name, source=str(path))
    sections = _split_markdown(text)
    for idx, (heading, level, body) in enumerate(sections):
        if not body and heading == "(intro)":
            continue
        sec_id = f"sec:{path.name}:{idx}"
        graph.add_node(sec_id, type="Section", label=heading, level=level, doc=doc_id, body=body[:600])
        graph.add_edge(doc_id, sec_id, relation="CONTAINS")
        # Field mentions
        fields = _extract_fields(heading + "\n" + body)
        for f in fields:
            f_id = f"field:{f}"
            if f_id not in graph:
                graph.add_node(f_id, type="Field", label=f)
            graph.add_edge(sec_id, f_id, relation="MENTIONS_FIELD")
            # If the section text contains "justify"/"because" with a field, mark JUSTIFIES
            low = body.lower()
            if any(w in low for w in ("justif", "because", "rationale", "because of", "due to", "porque", "debido")):
                graph.add_edge(sec_id, f_id, relation="JUSTIFIES")
        # Rename edges
        for src, dst in _extract_renames(heading + "\n" + body):
            for f in (src, dst):
                fid = f"field:{f}"
                if fid not in graph:
                    graph.add_node(fid, type="Field", label=f)
            graph.add_edge(f"field:{src}", f"field:{dst}", relation="CHANGES_FROM_TO", source_section=sec_id)


def _ingest_sql(graph: nx.DiGraph, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    doc_id = f"doc:{path.name}"
    graph.add_node(doc_id, type="Document", label=path.name, source=str(path))
    # Find CREATE TABLE blocks
    table_re = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\);", re.IGNORECASE | re.DOTALL)
    for tm in table_re.finditer(text):
        table = tm.group(1)
        body = tm.group(2)
        tc_id = f"tbl:{table.lower()}"
        graph.add_node(tc_id, type="TableChange", label=table, doc=doc_id)
        graph.add_edge(doc_id, tc_id, relation="CONTAINS")
        # Each line of body usually defines a column: NAME TYPE [...]
        for line in body.split(","):
            cm = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z]+)", line)
            if cm:
                col = cm.group(1).upper()
                if col in _FIELD_STOP:
                    continue
                f_id = f"field:{col}"
                if f_id not in graph:
                    graph.add_node(f_id, type="Field", label=col)
                graph.add_edge(tc_id, f_id, relation="HAS_COLUMN")


# ── Public API ───────────────────────────────────────────────────────────────


def build_graph(
    changelog_dir: Path | str,
    schema_files: list[Path | str] | None = None,
) -> nx.DiGraph:
    """Walk ``changelog_dir/*.md`` (+ optional ``schema_files``) and return the graph."""
    g = nx.DiGraph()
    cdir = Path(changelog_dir)
    if cdir.is_dir():
        for md in sorted(cdir.glob("*.md")):
            _ingest_markdown(g, md)
    for sf in schema_files or []:
        sp = Path(sf)
        if sp.exists():
            _ingest_sql(g, sp)
    return g


def save_graph(graph: nx.DiGraph, path: Path | str) -> None:
    """Persist nodes + edges as plain JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "nodes": [{"id": n, **dict(attrs)} for n, attrs in graph.nodes(data=True)],
        "edges": [{"source": s, "target": t, **dict(attrs)}
                  for s, t, attrs in graph.edges(data=True)],
    }
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Saved change-log graph: %d nodes, %d edges → %s",
        graph.number_of_nodes(), graph.number_of_edges(), out,
    )


def load_graph(path: Path | str) -> nx.DiGraph:
    """Load a previously saved graph; returns an empty graph if missing."""
    p = Path(path)
    if not p.exists():
        return nx.DiGraph()
    data = json.loads(p.read_text(encoding="utf-8"))
    g = nx.DiGraph()
    for n in data.get("nodes", []):
        nid = n.pop("id")
        g.add_node(nid, **n)
    for e in data.get("edges", []):
        s = e.pop("source")
        t = e.pop("target")
        g.add_edge(s, t, **e)
    return g


def graph_to_payload(graph: nx.DiGraph) -> dict:
    """Frontend-friendly node/edge payload."""
    nodes = [{"id": n, **dict(d)} for n, d in graph.nodes(data=True)]
    edges = [{"source": s, "target": t, **dict(d)} for s, t, d in graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges}
