"""Ontology for the Regulatory Knowledge Graph.

Defines node types, edge types, and their Pydantic models.
Three knowledge tiers:
  1. Regulatory KB  — regulations, interpretations, concepts, validation rules
  2. Schema KB      — tables, columns from the banking reporting database
  3. Experience KB  — quirks, experiences, and insights from past validations
"""

from __future__ import annotations

import enum
import json
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Node types ───────────────────────────────────────────────────────────────


class NodeType(str, enum.Enum):
    REGULATION = "Regulation"
    INTERPRETATION = "Interpretation"
    CONCEPT = "Concept"
    TABLE = "Table"
    COLUMN = "Column"
    VALIDATION_RULE = "ValidationRule"
    EXPERIENCE = "Experience"
    QUIRK = "Quirk"
    INSIGHT = "Insight"


# ── Edge types ───────────────────────────────────────────────────────────────


class EdgeType(str, enum.Enum):
    GOVERNS = "GOVERNS"                      # Regulation → Column/Table
    IMPLEMENTS = "IMPLEMENTS"                # Column → Regulation/Concept
    EXCEPTION_TO = "EXCEPTION_TO"            # Quirk → ValidationRule/Regulation
    CLARIFIED_BY = "CLARIFIED_BY"            # Regulation → Regulation (Q&A)
    RESOLVES = "RESOLVES"                    # Interpretation → ambiguity
    SUPERSEDES = "SUPERSEDES"                # newer node → older node
    RELATES_TO = "RELATES_TO"               # generic association
    SUBJECT_TO = "SUBJECT_TO"               # Regulation → Regulation (cross-ref)
    DEFINITION_IN = "DEFINITION_IN"         # Concept → Regulation (where defined)
    HAS_QUIRK = "HAS_QUIRK"                 # Column → Quirk
    DISCOVERED_IN = "DISCOVERED_IN"         # Insight → Experience
    VALIDATES = "VALIDATES"                  # ValidationRule → Column
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"  # any → any (flagged)
    CONTAINS = "CONTAINS"                    # Document/Section hierarchy
    MENTIONS_FIELD = "MENTIONS_FIELD"        # Section → Column/Field


# ── Node models ──────────────────────────────────────────────────────────────


class KGNode(BaseModel):
    """Base for all knowledge graph nodes."""
    id: str
    node_type: NodeType
    label: str

    def props_dict(self) -> dict[str, Any]:
        """Extra properties beyond id/node_type/label, serialised as JSON."""
        exclude = {"id", "node_type", "label"}
        d = {}
        for k, v in self.model_dump().items():
            if k in exclude:
                continue
            if v is None:
                continue
            # Convert dates to ISO strings for JSON
            if isinstance(v, date):
                d[k] = v.isoformat()
            else:
                d[k] = v
        return d

    def props_json(self) -> str:
        return json.dumps(self.props_dict(), ensure_ascii=False)


class Regulation(KGNode):
    node_type: NodeType = NodeType.REGULATION
    article_id: str = ""
    source_document: str = ""
    jurisdiction: str = ""
    effective_date: date | None = None

class Interpretation(KGNode):
    node_type: NodeType = NodeType.INTERPRETATION
    confidence: float = 0.5
    source: Literal["EBA_QA", "LLM_inferred", "human_validated"] = "LLM_inferred"
    scope: str = ""
    valid_from: date | None = None
    valid_until: date | None = None

class Concept(KGNode):
    node_type: NodeType = NodeType.CONCEPT
    definition: str = ""

class Table(KGNode):
    node_type: NodeType = NodeType.TABLE
    schema_name: str = ""
    row_count: int | None = None

class Column(KGNode):
    node_type: NodeType = NodeType.COLUMN
    table_name: str = ""
    data_type: str = ""
    nullable: bool = True
    deprecated_in: str | None = None
    introduced_in: str | None = None

class ValidationRule(KGNode):
    node_type: NodeType = NodeType.VALIDATION_RULE
    severity: Literal["ERROR", "WARNING", "DISABLED"] = "ERROR"
    framework_version: str = ""
    active: bool = True
    rule_text: str = ""

class Experience(KGNode):
    node_type: NodeType = NodeType.EXPERIENCE
    session_id: str = ""
    question: str = ""
    answer_summary: str = ""
    timestamp: str = ""  # ISO datetime string

class Quirk(KGNode):
    node_type: NodeType = NodeType.QUIRK
    description: str = ""
    severity: Literal["low", "medium", "high"] = "medium"

class Insight(KGNode):
    node_type: NodeType = NodeType.INSIGHT
    summary: str = ""
    tags: list[str] = Field(default_factory=list)


# ── Edge model ───────────────────────────────────────────────────────────────


class KGEdge(BaseModel):
    """An edge in the knowledge graph."""
    src_id: str
    dst_id: str
    edge_type: EdgeType
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def props_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.metadata:
            d["metadata"] = json.dumps(self.metadata, ensure_ascii=False)
        return d

    def props_json(self) -> str:
        return json.dumps(self.props_dict(), ensure_ascii=False)


# ── Lookup ───────────────────────────────────────────────────────────────────


NODE_CLASSES: dict[NodeType, type[KGNode]] = {
    NodeType.REGULATION: Regulation,
    NodeType.INTERPRETATION: Interpretation,
    NodeType.CONCEPT: Concept,
    NodeType.TABLE: Table,
    NodeType.COLUMN: Column,
    NodeType.VALIDATION_RULE: ValidationRule,
    NodeType.EXPERIENCE: Experience,
    NodeType.QUIRK: Quirk,
    NodeType.INSIGHT: Insight,
}
