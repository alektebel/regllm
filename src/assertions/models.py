"""
Data model for a regulation assertion — a single testable compliance rule.

An assertion is parsed from a regulation article and has enough structure to:
  1. Map to schema (which tables / columns are relevant)
  2. Generate a SQL check query
  3. Report findings with the original regulatory citation
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


# Valid values for check_type and materiality
CHECK_TYPES   = {"range", "existence", "classification", "time_limit", "ratio", "completeness"}
MATERIALITIES = {"low", "medium", "high", "critical"}


@dataclass
class Threshold:
    """Numeric or date threshold embedded in an assertion."""
    field_hint: str          # column name hint, e.g. "pd", "lgd", "dias_mora"
    op: str                  # ">=", "<=", ">", "<", "=", "!="
    value: float | int | str
    unit: str = ""           # "%", "days", "EUR", etc.

    def to_dict(self) -> dict:
        return {"field_hint": self.field_hint, "op": self.op, "value": self.value, "unit": self.unit}

    @classmethod
    def from_dict(cls, d: dict) -> "Threshold":
        return cls(**d)


@dataclass
class Assertion:
    """One testable compliance rule extracted from a regulation article."""

    regulation_id:   str
    regulation_name: str
    article:         str
    passage:         str
    rule_text:       str
    check_type:      str
    conditions:      list[str]        = field(default_factory=list)
    threshold:       Threshold | None = None
    materiality:     str              = "medium"
    notes:           str              = ""
    active:          bool             = True
    sql_template:    str | None       = None
    regulation_date: str | None       = None  # ISO date string

    # Generated on first access
    _id: str | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        if self._id is None:
            slug = re.sub(r"\W+", "_", self.article.lower())[:30]
            fingerprint = hashlib.sha1(
                f"{self.regulation_id}|{self.article}|{self.rule_text}".encode()
            ).hexdigest()[:8]
            self._id = f"{self.regulation_id}__{slug}__{fingerprint}"
        return self._id

    def to_db_row(self) -> dict[str, Any]:
        return {
            "id":              self.id,
            "regulation_id":   self.regulation_id,
            "regulation_name": self.regulation_name,
            "article":         self.article,
            "passage":         self.passage[:1000],  # cap stored passage
            "rule_text":       self.rule_text,
            "check_type":      self.check_type,
            "conditions":      json.dumps(self.conditions),
            "threshold":       json.dumps(self.threshold.to_dict()) if self.threshold else None,
            "materiality":     self.materiality,
            "sql_template":    self.sql_template,
            "notes":           self.notes,
            "active":          self.active,
            "regulation_date": self.regulation_date,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "Assertion":
        threshold = None
        if row.get("threshold"):
            raw = row["threshold"] if isinstance(row["threshold"], dict) else json.loads(row["threshold"])
            threshold = Threshold.from_dict(raw)
        conditions = row.get("conditions") or []
        if isinstance(conditions, str):
            conditions = json.loads(conditions)
        a = cls(
            regulation_id=row["regulation_id"],
            regulation_name=row["regulation_name"],
            article=row["article"],
            passage=row["passage"],
            rule_text=row["rule_text"],
            check_type=row["check_type"],
            conditions=conditions,
            threshold=threshold,
            materiality=row.get("materiality", "medium"),
            notes=row.get("notes", ""),
            active=row.get("active", True),
            sql_template=row.get("sql_template"),
            regulation_date=row.get("regulation_date"),
        )
        a._id = row["id"]
        return a
