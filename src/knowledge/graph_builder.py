"""LLM-driven knowledge graph builder.

Uses qwen3:32b (via Ollama) to automatically extract entities,
relationships, and cross-links from three knowledge sources:

1. **Regulation text** → Regulation, Concept, Interpretation nodes + edges
2. **Database schema + context** → Column↔Regulation mappings with reasoning
3. **Agent experience** → dynamic cross-links, insights, and quirks

Each step is a single structured-JSON LLM call against the ontology.
The builder is idempotent — re-running on the same input upserts nodes.

Usage::

    from src.knowledge.graph_builder import GraphBuilder
    builder = GraphBuilder(graph_store, vector_store, embed_service)

    # Ingest a regulation article
    builder.ingest_regulation("Article 15 — Minimum provisions ...", source="circular_4_2022")

    # Map schema columns to regulations
    builder.map_schema_to_regulations(ddl_text, regulation_summaries)

    # Reflect on an agent session
    builder.reflect_on_session(session_log, session_id="abc123")
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingService, get_embedding_service
from .graph_store import GraphStore
from .llm_client import LocalLLMClient, get_client
from .ontology import (
    Column,
    Concept,
    EdgeType,
    Experience,
    Insight,
    Interpretation,
    KGEdge,
    KGNode,
    NodeType,
    Quirk,
    Regulation,
    ValidationRule,
)
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

_REGULATION_EXTRACT_SYSTEM = """\
You are a regulatory knowledge extraction engine for Spanish banking regulation \
(COREP/FINREP, EBA, BdE Circulars). Extract structured entities and \
relationships from the regulation text.

Return a JSON object with:
{
  "entities": [
    {
      "id": "short_snake_case_id",
      "type": "Regulation | Concept | Interpretation | ValidationRule",
      "label": "human-readable name",
      "props": { ... type-specific properties ... }
    }
  ],
  "relationships": [
    {
      "source": "entity_id",
      "target": "entity_id",
      "type": "GOVERNS | SUBJECT_TO | DEFINITION_IN | CLARIFIED_BY | SUPERSEDES | RELATES_TO | EXCEPTION_TO",
      "confidence": 0.0-1.0,
      "reason": "one-line justification"
    }
  ]
}

Entity type guidance:
- **Regulation**: An article, paragraph, or regulatory provision with a citable reference
- **Concept**: A defined term, threshold, ratio, or metric (e.g. "LGD floor", "PD calibration window", "materiality threshold")
- **Interpretation**: An inference about ambiguous regulation text (mark confidence < 0.8)
- **ValidationRule**: A concrete testable assertion (e.g. "LGD >= 0.45 for corporate exposures")

Relationship type guidance:
- GOVERNS: regulation constrains a concept/field
- SUBJECT_TO: regulation cross-references another regulation
- DEFINITION_IN: concept is defined within a regulation
- CLARIFIED_BY: one regulation clarifies/amends another
- SUPERSEDES: newer provision replaces older one
- RELATES_TO: semantic association (use sparingly)
- EXCEPTION_TO: an exception or carve-out to a rule

Be thorough. Extract ALL concepts mentioned, including numeric thresholds, \
time periods, asset classes, and conditions. Prefer many specific entities \
over few generic ones. Every concept should link back to its source regulation.\
"""

_SCHEMA_MAP_SYSTEM = """\
You are a regulatory compliance mapper for Spanish banking databases. \
Given database column definitions and regulation summaries, determine which \
regulations govern each column and why.

Return a JSON object with:
{
  "mappings": [
    {
      "column": "COLUMN_NAME",
      "regulations": [
        {
          "regulation_id": "id from the regulation list",
          "edge_type": "GOVERNS | IMPLEMENTS | VALIDATES",
          "confidence": 0.0-1.0,
          "reasoning": "why this regulation applies to this column"
        }
      ],
      "concepts": ["concept_name_1", "concept_name_2"],
      "validation_rules": [
        {
          "rule": "testable assertion in pseudo-SQL or plain text",
          "severity": "ERROR | WARNING",
          "source_article": "regulation reference"
        }
      ]
    }
  ]
}

Be precise about confidence:
- 1.0: regulation explicitly names the column or its exact equivalent
- 0.8: regulation clearly governs the concept this column represents
- 0.6: regulation is relevant but the mapping requires interpretation
- 0.4: tenuous connection, needs human review
- < 0.4: don't include it\
"""

_REFLECT_SYSTEM = """\
You are a self-reflective learning engine for a banking regulatory compliance \
agent. After a validation session, analyze what happened and extract:

1. **Insights**: general learnings that apply beyond this session
2. **Quirks**: unexpected behaviors, edge cases, or data anomalies
3. **New connections**: relationships between existing graph entities that \
   were revealed during the session

Return a JSON object with:
{
  "insights": [
    {
      "summary": "one-paragraph description",
      "related_fields": ["COLUMN_A", "COLUMN_B"],
      "related_articles": ["art15", "art22"],
      "tags": ["lgd", "provisioning", "edge_case"],
      "confidence": 0.0-1.0
    }
  ],
  "quirks": [
    {
      "description": "what is unexpected",
      "severity": "low | medium | high",
      "related_fields": ["COLUMN_X"],
      "suggested_validation": "how to test for this"
    }
  ],
  "new_connections": [
    {
      "source_type": "Regulation | Concept | Column | ...",
      "source_label": "name or id",
      "target_type": "...",
      "target_label": "name or id",
      "edge_type": "GOVERNS | RELATES_TO | EXCEPTION_TO | ...",
      "confidence": 0.0-1.0,
      "reason": "why this connection exists"
    }
  ]
}

Focus on what is SURPRISING or NON-OBVIOUS. Don't restate what the user asked — \
extract the deeper pattern or exception that future sessions should know about.\
"""

_CONCEPT_LINK_SYSTEM = """\
You are a knowledge graph linker. Given a list of existing graph nodes and a \
new set of entities, identify connections between them.

Return a JSON object with:
{
  "links": [
    {
      "source_id": "existing node id",
      "target_id": "new or existing node id",
      "edge_type": "GOVERNS | RELATES_TO | IMPLEMENTS | SUBJECT_TO | ...",
      "confidence": 0.0-1.0,
      "reason": "one-line justification"
    }
  ]
}

Only return links with confidence >= 0.5. Prefer specific edge types over RELATES_TO.\
"""

_VERIFY_SYSTEM = """\
You are a strict fact-checker for a regulatory knowledge graph. You will \
receive a set of extracted triples (entity–relationship–entity) alongside \
the original regulation text they were extracted from.

For EACH triple, decide:
- **accept**: the triple is clearly supported by the source text
- **revise**: the triple is partially correct but needs adjustment (provide the corrected version)
- **reject**: the triple is hallucinated or unsupported

Return a JSON object:
{
  "verdicts": [
    {
      "source": "entity_id",
      "target": "entity_id",
      "edge_type": "...",
      "verdict": "accept | revise | reject",
      "revised_confidence": 0.0-1.0,
      "reason": "why this verdict"
    }
  ],
  "missing": [
    {
      "description": "something the extraction missed",
      "suggested_entity": { "id": "...", "type": "...", "label": "..." },
      "suggested_relationship": { "source": "...", "target": "...", "type": "...", "confidence": 0.0-1.0 }
    }
  ]
}

Be strict. If the source text does not explicitly support a relationship, \
reject it. Pay special attention to:
- Scope conditions (applies only to certain asset classes, jurisdictions, time periods)
- Numeric thresholds that are approximate vs exact
- Interpretations presented as facts\
"""

_GAP_DETECTION_SYSTEM = """\
You are a knowledge graph completeness auditor for Spanish banking regulation \
(COREP/FINREP). Given the current graph statistics and a sample of nodes with \
low connectivity, identify knowledge gaps.

Return a JSON object:
{
  "gaps": [
    {
      "description": "what knowledge is missing",
      "severity": "critical | high | medium | low",
      "affected_nodes": ["node_id_1", "node_id_2"],
      "suggested_action": "what document or query would fill this gap",
      "gap_type": "missing_regulation | missing_concept | missing_link | missing_validation"
    }
  ]
}

Focus on:
- Regulation nodes with no outgoing GOVERNS edges (dead regulation — not linked to anything)
- Column nodes with no inbound GOVERNS edges (ungoverned columns)
- Concept nodes defined but never referenced by a ValidationRule
- Isolated clusters with no cross-references to other parts of the graph\
"""


# ── Builder ──────────────────────────────────────────────────────────────────


class GraphBuilder:
    """LLM-driven knowledge graph construction engine."""

    def __init__(
        self,
        store: GraphStore,
        vector_store: VectorStore | None = None,
        embed_service: EmbeddingService | None = None,
        llm: LocalLLMClient | None = None,
    ) -> None:
        self._store = store
        self._vs = vector_store
        self._embed = embed_service or get_embedding_service()
        self._llm = llm or get_client()

    # ── 1. Regulation ingestion ──────────────────────────────────────────

    def ingest_regulation(
        self,
        text: str,
        source: str = "",
        chunk_size: int = 3000,
        verify: bool = True,
    ) -> dict[str, int]:
        """Extract entities and relationships from regulation text via LLM.

        Long texts are chunked; entities are merged across chunks.
        When ``verify=True``, runs a second-pass verification (KARMA pattern)
        that challenges each triple against the source text before committing.
        """
        counts = {"nodes": 0, "edges": 0, "embedded": 0, "rejected": 0, "revised": 0}
        chunks = _chunk_text(text, chunk_size)

        all_entity_ids: list[str] = []

        for i, chunk in enumerate(chunks):
            user_msg = f"Source: {source}\nChunk {i + 1}/{len(chunks)}:\n\n{chunk}"
            extraction = self._llm.chat_json(
                _REGULATION_EXTRACT_SYSTEM, user_msg,
                temperature=0.1, max_tokens=2048,
            )

            if "error" in extraction:
                logger.warning("LLM extraction failed on chunk %d: %s", i, extraction)
                continue

            entities = extraction.get("entities", [])
            relationships = extraction.get("relationships", [])

            # Create entities (always — verification only filters relationships)
            for ent in entities:
                node_id = self._create_entity(ent, source, counts)
                if node_id:
                    all_entity_ids.append(node_id)

            # Two-pass verification
            if verify and relationships and self._llm.detect_backend() != "stub":
                vr = self.verify_extraction(chunk, entities, relationships)
                counts["rejected"] += len(vr["rejected"])
                counts["revised"] += len(vr["revised"])
                relationships = vr["accepted"]

                # Handle missing triples the verifier found
                for missing in vr.get("missing", []):
                    ent = missing.get("suggested_entity")
                    if ent:
                        node_id = self._create_entity(ent, source, counts)
                        if node_id:
                            all_entity_ids.append(node_id)
                    rel = missing.get("suggested_relationship")
                    if rel:
                        relationships.append(rel)

            # Create verified relationships
            for rel in relationships:
                self._create_relationship(rel, source, counts)

            # Embed the chunk
            if self._vs and self._embed.available:
                doc_id = f"reg:{source}:chunk{i}"
                embedding = self._embed.embed_one(chunk[:2000])
                self._vs.upsert(
                    doc_id=doc_id, text=chunk[:2000], embedding=embedding,
                    metadata={"source": source, "node_type": "Regulation", "chunk_idx": i},
                )
                counts["embedded"] += 1

        # Cross-link new entities with existing graph
        if all_entity_ids:
            self._auto_link(all_entity_ids, counts)

        logger.info("Regulation ingestion [%s]: %s", source, counts)
        return counts

    def _create_entity(
        self, ent: dict[str, Any], source: str, counts: dict[str, int],
    ) -> str | None:
        """Create a graph node from an LLM-extracted entity dict."""
        ent_type = ent.get("type", "Concept")
        raw_id = ent.get("id", "")
        label = ent.get("label", raw_id)
        props = ent.get("props", {})

        # Normalize ID
        node_id = _make_id(ent_type, raw_id or label)

        try:
            if ent_type == "Regulation":
                self._store.add_node(Regulation(
                    id=node_id, label=label,
                    article_id=props.get("article_id", raw_id),
                    source_document=source,
                    jurisdiction=props.get("jurisdiction", "ES"),
                ))
            elif ent_type == "Interpretation":
                self._store.add_node(Interpretation(
                    id=node_id, label=label,
                    confidence=props.get("confidence", 0.5),
                    source=props.get("source", "LLM_inferred"),
                    scope=props.get("scope", ""),
                ))
            elif ent_type == "ValidationRule":
                self._store.add_node(ValidationRule(
                    id=node_id, label=label,
                    rule_text=props.get("rule_text", label),
                    severity=props.get("severity", "WARNING"),
                ))
            else:  # Concept (default)
                self._store.add_node(Concept(
                    id=node_id, label=label,
                    definition=props.get("definition", ""),
                ))
            counts["nodes"] += 1
            return node_id
        except Exception as e:
            logger.warning("Failed to create node %s: %s", node_id, e)
            return None

    def _create_relationship(
        self, rel: dict[str, Any], source: str, counts: dict[str, int],
    ) -> None:
        """Create a graph edge from an LLM-extracted relationship dict."""
        src_raw = rel.get("source", "")
        dst_raw = rel.get("target", "")
        edge_type_str = rel.get("type", "RELATES_TO")
        confidence = rel.get("confidence", 0.7)

        # Try to resolve IDs — the LLM may have used either the raw id or label
        src_id = self._resolve_node_id(src_raw)
        dst_id = self._resolve_node_id(dst_raw)

        if not src_id or not dst_id:
            logger.debug("Skipping edge %s→%s: unresolved nodes", src_raw, dst_raw)
            return

        try:
            et = EdgeType(edge_type_str)
        except ValueError:
            et = EdgeType.RELATES_TO

        self._store.add_edge(KGEdge(
            src_id=src_id, dst_id=dst_id, edge_type=et,
            confidence=confidence,
            metadata={"source": source, "reason": rel.get("reason", "")},
        ))
        counts["edges"] += 1

    def _resolve_node_id(self, raw: str) -> str | None:
        """Try to find a node by various ID conventions."""
        if not raw:
            return None

        # Direct lookup with common prefixes
        for prefix in ("reg:", "concept:", "col:", "interp:", "rule:", "insight:", "quirk:"):
            candidate = f"{prefix}{raw}"
            if self._store.get_node(candidate):
                return candidate

        # Try the raw value as-is
        if self._store.get_node(raw):
            return raw

        # Search by label
        results = self._store.search_nodes(label_contains=raw, limit=1)
        if results:
            return results[0].id

        return None

    # ── 2. Schema → Regulation mapping ───────────────────────────────────

    def map_schema_to_regulations(
        self,
        ddl_text: str,
        regulation_summaries: list[dict[str, str]] | None = None,
    ) -> dict[str, int]:
        """Use the LLM to map database columns to governing regulations.

        If regulation_summaries is None, pulls existing Regulation nodes
        from the graph as context.
        """
        counts = {"edges": 0, "rules": 0, "concepts": 0}

        # Gather regulation context from the graph
        if regulation_summaries is None:
            reg_nodes = self._store.search_nodes(
                node_types=[NodeType.REGULATION], limit=50,
            )
            regulation_summaries = [
                {"id": n.id, "label": n.label, **n.props_dict()}
                for n in reg_nodes
            ]

        reg_context = json.dumps(regulation_summaries[:30], ensure_ascii=False, default=str)
        user_msg = (
            f"DATABASE SCHEMA (DDL):\n{ddl_text[:4000]}\n\n"
            f"REGULATION NODES IN GRAPH:\n{reg_context}\n\n"
            "Map each column to its governing regulations."
        )

        result = self._llm.chat_json(
            _SCHEMA_MAP_SYSTEM, user_msg,
            temperature=0.1, max_tokens=3000,
        )

        if "error" in result:
            logger.warning("Schema mapping failed: %s", result)
            return counts

        for mapping in result.get("mappings", []):
            col_name = mapping.get("column", "").upper()
            if not col_name:
                continue

            col_id = f"col:{col_name}"

            # Create regulation → column edges
            for reg in mapping.get("regulations", []):
                reg_id = reg.get("regulation_id", "")
                resolved = self._resolve_node_id(reg_id)
                if not resolved:
                    continue

                try:
                    et = EdgeType(reg.get("edge_type", "GOVERNS"))
                except ValueError:
                    et = EdgeType.GOVERNS

                self._store.add_edge(KGEdge(
                    src_id=resolved, dst_id=col_id, edge_type=et,
                    confidence=reg.get("confidence", 0.5),
                    metadata={"reasoning": reg.get("reasoning", "")},
                ))
                counts["edges"] += 1

            # Create concept nodes and link
            for concept_name in mapping.get("concepts", []):
                c_id = f"concept:{concept_name.upper()}"
                self._store.add_node(Concept(
                    id=c_id, label=concept_name,
                ))
                self._store.add_edge(KGEdge(
                    src_id=col_id, dst_id=c_id,
                    edge_type=EdgeType.IMPLEMENTS,
                ))
                counts["concepts"] += 1

            # Create validation rules
            for vr in mapping.get("validation_rules", []):
                rule_text = vr.get("rule", "")
                if not rule_text:
                    continue
                rule_id = _make_id("rule", f"{col_name}_{rule_text[:30]}")
                self._store.add_node(ValidationRule(
                    id=rule_id, label=rule_text[:100],
                    rule_text=rule_text,
                    severity=vr.get("severity", "WARNING"),
                ))
                self._store.add_edge(KGEdge(
                    src_id=rule_id, dst_id=col_id,
                    edge_type=EdgeType.VALIDATES,
                ))
                counts["rules"] += 1

        logger.info("Schema mapping: %s", counts)
        return counts

    # ── 3. Experience reflection ─────────────────────────────────────────

    def reflect_on_session(
        self,
        session_log: str,
        session_id: str = "",
    ) -> dict[str, int]:
        """Analyze an agent session log and extract insights, quirks, connections.

        This is how the graph GROWS its memory automatically.
        """
        counts = {"insights": 0, "quirks": 0, "connections": 0}

        if not session_id:
            session_id = hashlib.sha256(
                f"{session_log[:100]}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

        # Provide existing graph context so the LLM can suggest connections
        existing_nodes = self._store.search_nodes(limit=40)
        context = json.dumps(
            [{"id": n.id, "type": n.node_type.value, "label": n.label}
             for n in existing_nodes],
            ensure_ascii=False,
        )

        user_msg = (
            f"SESSION LOG:\n{session_log[:6000]}\n\n"
            f"EXISTING GRAPH NODES (sample):\n{context}\n\n"
            "Extract insights, quirks, and new connections."
        )

        result = self._llm.chat_json(
            _REFLECT_SYSTEM, user_msg,
            temperature=0.2, max_tokens=2048,
        )

        if "error" in result:
            logger.warning("Reflection failed: %s", result)
            return counts

        ts = datetime.now().isoformat()

        # Record experience node for the session
        exp_id = f"exp:{session_id}"
        self._store.add_node(Experience(
            id=exp_id,
            label=f"Session {session_id}",
            session_id=session_id,
            question=session_log[:200],
            answer_summary="",
            timestamp=ts,
        ))

        # Create insights
        for ins in result.get("insights", []):
            summary = ins.get("summary", "")
            if not summary:
                continue
            ins_id = _make_id("insight", summary[:40])

            self._store.add_node(Insight(
                id=ins_id,
                label=summary[:100],
                summary=summary,
                tags=ins.get("tags", []),
            ))

            # Link to experience
            self._store.add_edge(KGEdge(
                src_id=ins_id, dst_id=exp_id,
                edge_type=EdgeType.DISCOVERED_IN,
                confidence=ins.get("confidence", 0.7),
            ))

            # Link to fields and articles
            for field in ins.get("related_fields", []):
                self._link_to_entity(ins_id, field)
            for article in ins.get("related_articles", []):
                self._link_to_entity(ins_id, article)

            counts["insights"] += 1

            # Embed the insight
            if self._vs and self._embed.available:
                vec = self._embed.embed_one(summary)
                self._vs.upsert(
                    doc_id=ins_id, text=summary, embedding=vec,
                    metadata={"node_type": "Insight", "session": session_id},
                )

        # Create quirks
        for q in result.get("quirks", []):
            desc = q.get("description", "")
            if not desc:
                continue
            q_id = _make_id("quirk", desc[:40])

            self._store.add_node(Quirk(
                id=q_id,
                label=desc[:100],
                description=desc,
                severity=q.get("severity", "medium"),
            ))

            self._store.add_edge(KGEdge(
                src_id=q_id, dst_id=exp_id,
                edge_type=EdgeType.DISCOVERED_IN,
            ))

            for field in q.get("related_fields", []):
                resolved = self._resolve_node_id(f"col:{field.upper()}")
                if resolved:
                    self._store.add_edge(KGEdge(
                        src_id=resolved, dst_id=q_id,
                        edge_type=EdgeType.HAS_QUIRK,
                    ))

            counts["quirks"] += 1

        # Create new connections between existing nodes
        for conn in result.get("new_connections", []):
            src = self._resolve_node_id(conn.get("source_label", ""))
            dst = self._resolve_node_id(conn.get("target_label", ""))
            if not src or not dst:
                continue
            try:
                et = EdgeType(conn.get("edge_type", "RELATES_TO"))
            except ValueError:
                et = EdgeType.RELATES_TO

            self._store.add_edge(KGEdge(
                src_id=src, dst_id=dst, edge_type=et,
                confidence=conn.get("confidence", 0.6),
                metadata={"reason": conn.get("reason", ""), "session": session_id},
            ))
            counts["connections"] += 1

        logger.info("Session reflection [%s]: %s", session_id, counts)
        return counts

    # ── 4. Auto-link: connect new entities to existing graph ─────────────

    def _auto_link(
        self, new_ids: list[str], counts: dict[str, int],
    ) -> None:
        """Ask the LLM to find connections between new and existing nodes."""
        existing = self._store.search_nodes(limit=60)
        new_nodes = [self._store.get_node(nid) for nid in new_ids]
        new_nodes = [n for n in new_nodes if n is not None]

        if not existing or not new_nodes:
            return

        existing_json = json.dumps(
            [{"id": n.id, "type": n.node_type.value, "label": n.label}
             for n in existing if n.id not in set(new_ids)],
            ensure_ascii=False,
        )
        new_json = json.dumps(
            [{"id": n.id, "type": n.node_type.value, "label": n.label}
             for n in new_nodes],
            ensure_ascii=False,
        )

        user_msg = (
            f"EXISTING NODES:\n{existing_json}\n\n"
            f"NEW NODES:\n{new_json}\n\n"
            "Identify meaningful connections between new and existing nodes."
        )

        result = self._llm.chat_json(
            _CONCEPT_LINK_SYSTEM, user_msg,
            temperature=0.1, max_tokens=1500,
        )

        for link in result.get("links", []):
            src = link.get("source_id", "")
            dst = link.get("target_id", "")
            # Verify both exist
            if not self._store.get_node(src) or not self._store.get_node(dst):
                # Try to resolve
                src = self._resolve_node_id(src) or src
                dst = self._resolve_node_id(dst) or dst
                if not self._store.get_node(src) or not self._store.get_node(dst):
                    continue

            try:
                et = EdgeType(link.get("edge_type", "RELATES_TO"))
            except ValueError:
                et = EdgeType.RELATES_TO

            self._store.add_edge(KGEdge(
                src_id=src, dst_id=dst, edge_type=et,
                confidence=link.get("confidence", 0.5),
                metadata={"reason": link.get("reason", ""), "auto_linked": True},
            ))
            counts["edges"] += 1

    def _link_to_entity(self, src_id: str, ref: str) -> None:
        """Best-effort link from src_id to whatever entity matches ref."""
        target = self._resolve_node_id(ref)
        if not target:
            target = self._resolve_node_id(ref.upper())
        if target:
            self._store.add_edge(KGEdge(
                src_id=src_id, dst_id=target,
                edge_type=EdgeType.RELATES_TO,
            ))

    # ── 5. Two-pass verification (KARMA pattern) ────────────────────────

    def verify_extraction(
        self,
        source_text: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Second-pass verification: challenge extracted triples against source.

        Returns {accepted: [...], revised: [...], rejected: [...], missing: [...]}.
        """
        triples_json = json.dumps({
            "entities": entities[:30],
            "relationships": relationships[:30],
        }, ensure_ascii=False)

        user_msg = (
            f"SOURCE TEXT:\n{source_text[:4000]}\n\n"
            f"EXTRACTED TRIPLES:\n{triples_json}\n\n"
            "Verify each triple against the source text."
        )

        result = self._llm.chat_json(
            _VERIFY_SYSTEM, user_msg,
            temperature=0.1, max_tokens=2048,
        )

        if "error" in result:
            logger.warning("Verification failed: %s", result)
            return {"accepted": relationships, "revised": [], "rejected": [], "missing": []}

        accepted, revised, rejected = [], [], []
        verdict_map = {
            (v.get("source", ""), v.get("target", "")): v
            for v in result.get("verdicts", [])
        }

        for rel in relationships:
            key = (rel.get("source", ""), rel.get("target", ""))
            verdict = verdict_map.get(key)
            if verdict is None:
                accepted.append(rel)  # no verdict = pass through
            elif verdict["verdict"] == "accept":
                rel["confidence"] = verdict.get("revised_confidence", rel.get("confidence", 0.7))
                accepted.append(rel)
            elif verdict["verdict"] == "revise":
                rel["confidence"] = verdict.get("revised_confidence", 0.5)
                rel["_revision_reason"] = verdict.get("reason", "")
                revised.append(rel)
                accepted.append(rel)  # still commit, with lower confidence
            else:
                rel["_rejection_reason"] = verdict.get("reason", "")
                rejected.append(rel)

        return {
            "accepted": accepted,
            "revised": revised,
            "rejected": rejected,
            "missing": result.get("missing", []),
        }

    # ── 6. Graph gap detection (SENATOR pattern) ─────────────────────────

    def detect_gaps(self) -> dict[str, Any]:
        """Analyze the graph for knowledge gaps using structural metrics + LLM.

        Returns {gaps: [...], stats: {...}}.
        """
        stats = self._store.stats()

        # Find structurally isolated or under-connected nodes
        all_nodes = self._store.search_nodes(limit=500)
        sparse_nodes: list[dict[str, Any]] = []

        for node in all_nodes:
            edges = self._store.get_edges(node.id)
            degree = len(edges)

            is_gap = False
            if node.node_type == NodeType.REGULATION and degree == 0:
                is_gap = True  # Dead regulation — governs nothing
            elif node.node_type == NodeType.COLUMN:
                inbound = self._store.get_edges(node.id, direction="in")
                governs_in = [e for e in inbound if e.edge_type == EdgeType.GOVERNS]
                if not governs_in:
                    is_gap = True  # Ungoverned column
            elif node.node_type == NodeType.CONCEPT and degree <= 1:
                is_gap = True  # Orphan concept

            if is_gap:
                sparse_nodes.append({
                    "id": node.id,
                    "type": node.node_type.value,
                    "label": node.label,
                    "degree": degree,
                })

        if not sparse_nodes:
            return {"gaps": [], "stats": stats, "sparse_node_count": 0}

        # Ask LLM to analyze the gaps
        context = json.dumps(sparse_nodes[:40], ensure_ascii=False)
        stats_json = json.dumps(stats, ensure_ascii=False)

        user_msg = (
            f"GRAPH STATISTICS:\n{stats_json}\n\n"
            f"UNDER-CONNECTED NODES ({len(sparse_nodes)} total, showing first 40):\n{context}\n\n"
            "Identify knowledge gaps and suggest how to fill them."
        )

        result = self._llm.chat_json(
            _GAP_DETECTION_SYSTEM, user_msg,
            temperature=0.2, max_tokens=1500,
        )

        gaps = result.get("gaps", []) if "error" not in result else []

        logger.info(
            "Gap detection: %d sparse nodes, %d gaps identified",
            len(sparse_nodes), len(gaps),
        )
        return {
            "gaps": gaps,
            "stats": stats,
            "sparse_node_count": len(sparse_nodes),
            "sparse_nodes": sparse_nodes[:20],
        }

    # ── 7. Full rebuild from files ───────────────────────────────────────

    def build_from_directory(
        self,
        regulation_dir: str | Path = "data/regulation",
        schema_file: str | Path | None = "data/samples/irb_schema.sql",
    ) -> dict[str, Any]:
        """Full graph build: regulations + schema mapping.

        This is the main entry point for populating the graph from scratch.
        """
        totals: dict[str, int] = {"nodes": 0, "edges": 0, "embedded": 0,
                                   "rules": 0, "concepts": 0}
        reg_dir = Path(regulation_dir)

        # Ingest regulations
        if reg_dir.is_dir():
            for md_path in sorted(reg_dir.glob("*.md")):
                text = md_path.read_text(encoding="utf-8")
                result = self.ingest_regulation(text, source=md_path.stem)
                for k in ("nodes", "edges", "embedded"):
                    totals[k] += result.get(k, 0)

        # Also check data/docs/regulation
        docs_reg = Path("data/docs/regulation")
        if docs_reg.is_dir():
            for md_path in sorted(docs_reg.glob("*.md")):
                text = md_path.read_text(encoding="utf-8")
                result = self.ingest_regulation(text, source=md_path.stem)
                for k in ("nodes", "edges", "embedded"):
                    totals[k] += result.get(k, 0)

        # Map schema
        sf = Path(schema_file) if schema_file else None
        if sf and sf.exists():
            ddl = sf.read_text(encoding="utf-8")
            result = self.map_schema_to_regulations(ddl)
            for k in ("edges", "rules", "concepts"):
                totals[k] += result.get(k, 0)

        logger.info("Full graph build complete: %s", totals)
        return totals


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_id(type_prefix: str, raw: str) -> str:
    """Deterministic node ID from type + raw string."""
    slug = raw.lower().strip().replace(" ", "_")[:60]
    # Add short hash to avoid collisions
    h = hashlib.sha256(raw.encode()).hexdigest()[:8]
    prefix_map = {
        "Regulation": "reg", "Concept": "concept", "Interpretation": "interp",
        "ValidationRule": "rule", "Insight": "insight", "Quirk": "quirk",
        "Experience": "exp",
    }
    p = prefix_map.get(type_prefix, type_prefix.lower())
    return f"{p}:{slug}_{h}"


def _chunk_text(text: str, size: int = 3000) -> list[str]:
    """Split text into chunks, preferring paragraph boundaries."""
    if len(text) <= size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


# ── CLI entry point ──────────────────────────────────────────────────────────


def get_builder(
    graph_path: str = "data/knowledge/regllm.kuzu",
    vector_path: str = "data/knowledge/vectors",
) -> GraphBuilder:
    """Convenience factory for CLI / notebook use."""
    store = GraphStore(graph_path)
    vs = VectorStore(vector_path)
    embed = get_embedding_service()
    llm = get_client()
    return GraphBuilder(store, vs, embed, llm)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    builder = get_builder()

    if "--full" in sys.argv:
        result = builder.build_from_directory()
        print(f"Full build: {result}")
    else:
        print("Usage: python -m src.knowledge.graph_builder --full")
        print("  Builds the full knowledge graph from regulation files + schema DDL.")
