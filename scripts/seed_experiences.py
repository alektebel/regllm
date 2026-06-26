"""Seed the knowledge graph with experience/insight nodes from markdown files.

Usage:
    python -m scripts.seed_experiences [--markdown-dir data/experience]

Reads each .md file from the experience directory, parses its frontmatter,
and creates the corresponding nodes in the KuzuDB knowledge graph and
ChromaDB vector store.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import yaml

from src.knowledge.embeddings import get_embedding_service
from src.knowledge.graph_store import GraphStore
from src.knowledge.ontology import (
    EdgeType,
    Experience,
    Insight,
    KGEdge,
    Quirk,
    NodeType,
)
from src.knowledge.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


_FM_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL,
)


def parse_md(path: Path) -> dict | None:
    """Parse a markdown file with YAML frontmatter.

    Returns {id, type, priority, tags, fields, articles, source,
             feedback, body} or None if malformed.

    All metadata comes from the frontmatter; ``body`` is the remainder.
    """
    text = path.read_text(encoding="utf-8")
    m = _FM_RE.match(text)
    if not m:
        logger.warning("No frontmatter in %s, skipping", path.name)
        return None

    try:
        meta: dict = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        logger.warning("YAML parse error in %s: %s", path.name, e)
        return None

    if not isinstance(meta, dict):
        return None

    meta["body"] = m.group(2).strip()
    return meta


def seed(
    graph_store: GraphStore,
    vector_store: VectorStore,
    embed_service,
    markdown_dir: str = "data/experience",
) -> dict[str, int]:
    """Load all experience markdowns into the graph + vector store.

    Returns counts of {nodes, edges, embedded}.
    """
    md_dir = Path(markdown_dir)
    if not md_dir.is_dir():
        logger.error("Directory not found: %s", markdown_dir)
        return {"nodes": 0, "edges": 0, "embedded": 0}

    counts: dict[str, int] = {"nodes": 0, "edges": 0, "embedded": 0}
    embed_available = embed_service.available if embed_service else False

    for md_path in sorted(md_dir.glob("*.md")):
        if md_path.name == "README.md":
            continue
        parsed = parse_md(md_path)
        if not parsed:
            continue

        exp_type = parsed.get("type", "insight")
        exp_id = parsed.get("id") or md_path.stem
        priority = parsed.get("priority", 0.5)
        tags = parsed.get("tags", [])
        fields = parsed.get("fields", [])
        articles = parsed.get("articles", [])
        feedback = parsed.get("feedback") or {}
        severity = parsed.get("severity", "medium")
        body = parsed.get("body", "")
        label = body.split("\n")[0].lstrip("# ").strip() if body else md_path.stem

        if exp_type == "quirk":
            node = Quirk(
                id=f"quirk:{exp_id}",
                label=label[:100],
                description=body[:2000],
                severity=severity,
            )
        elif feedback and feedback.get("type"):
            node = Insight(
                id=f"insight:{exp_id}",
                label=f"[feedback] {label[:90]}"[:100],
                summary=body[:2000],
                tags=["feedback"] + tags,
                priority=1.0,
                feedback_type=feedback.get("type", "correction"),
                original_claim=feedback.get("original", ""),
                corrected_understanding=feedback.get("corrected", ""),
            )
        elif exp_type == "experience":
            node = Experience(
                id=f"exp:{exp_id}",
                label=label[:100],
                session_id=parsed.get("session_id", ""),
                question=label[:200],
                answer_summary=body[:2000],
            )
        else:
            node = Insight(
                id=f"insight:{exp_id}",
                label=label[:100],
                summary=body[:2000],
                tags=tags,
                priority=priority,
            )

        graph_store.add_node(node)
        counts["nodes"] += 1

        # Link to fields — search Column/Concept nodes by label substring
        for field in fields:
            candidates = graph_store.search_nodes(
                node_types=[NodeType.COLUMN, NodeType.CONCEPT],
                label_contains=field.upper(),
                limit=3,
            )
            for candidate in candidates:
                graph_store.add_edge(KGEdge(
                    src_id=node.id,
                    dst_id=candidate.id,
                    edge_type=EdgeType.RELATES_TO,
                ))
                counts["edges"] += 1

        # Link to articles
        for article in articles:
            reg_nodes = graph_store.search_nodes(
                node_types=[NodeType.REGULATION],
                label_contains=article,
                limit=1,
            )
            if reg_nodes:
                graph_store.add_edge(KGEdge(
                    src_id=node.id,
                    dst_id=reg_nodes[0].id,
                    edge_type=EdgeType.RELATES_TO,
                ))
                counts["edges"] += 1

        # Embed in vector store
        if embed_available:
            vec = embed_service.embed_one(body[:2000])
            vector_store.upsert(
                doc_id=node.id,
                text=body[:2000],
                embedding=vec,
                metadata={
                    "node_type": node.node_type.value,
                    "priority": priority,
                    "tags": ",".join(tags[:5]),
                },
            )
            counts["embedded"] += 1

        logger.info(
            "  %s %s (%s) — %d edges, embedded=%s",
            "✓" if embed_available else " ",
            node.id,
            label[:60],
            sum(1 for _ in fields + articles),
            embed_available,
        )

    logger.info("Done: %d nodes, %d edges, %d embedded", *counts.values())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed experiences into KG")
    parser.add_argument(
        "--markdown-dir",
        default="data/experience",
        help="Directory with experience .md files",
    )
    parser.add_argument(
        "--graph-path",
        default="data/knowledge/regllm.kuzu",
        help="KuzuDB graph path",
    )
    parser.add_argument(
        "--vector-path",
        default="data/knowledge/vectors",
        help="ChromaDB vector store path",
    )
    args = parser.parse_args()

    graph_store = GraphStore(args.graph_path)
    vector_store = VectorStore(args.vector_path)
    embed_service = get_embedding_service()

    counts = seed(graph_store, vector_store, embed_service, args.markdown_dir)
    print(f"Seeded {counts['nodes']} nodes, {counts['edges']} edges, {counts['embedded']} embedded")


if __name__ == "__main__":
    main()
