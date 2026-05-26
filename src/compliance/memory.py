"""Running memory for compliance checker — bounded at ~1K tokens regardless of table size.

Compliant rows only increment a counter.
Flagged rows are appended to findings.
When findings exceed a threshold, an LLM call compresses them into a patterns string
and findings is cleared, so memory stays bounded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_COMPRESS_THRESHOLD = 20
_COMPRESS_PROMPT = """\
Eres un analista de riesgo de crédito. Resume los siguientes hallazgos de incumplimiento \
regulatorio en un párrafo conciso (máximo 200 palabras) que capture los patrones principales, \
parámetros afectados y artículos regulatorios implicados.

Patrones previos:
{patterns}

Nuevos hallazgos:
{findings_json}

Responde SOLO con el párrafo resumen, sin JSON ni prefijos."""


@dataclass
class Finding:
    row_id: str
    column: str
    value: Any
    verdict: str          # "flagged" | "uncertain"
    flags: list[str]
    articles: list[str]
    regulatory_text: str = ""  # excerpt from RAG that grounded the diagnosis


@dataclass
class RunningMemory:
    rows_processed: int = 0
    rows_flagged: int = 0
    findings: list[Finding] = field(default_factory=list)
    patterns: str = ""
    compress_threshold: int = _COMPRESS_THRESHOLD

    def add_compliant(self) -> None:
        self.rows_processed += 1

    def add_finding(self, finding: Finding) -> None:
        self.rows_processed += 1
        self.rows_flagged += 1
        self.findings.append(finding)

    def needs_compression(self) -> bool:
        return len(self.findings) >= self.compress_threshold

    def compress(self, llm_fn) -> None:
        findings_json = json.dumps(
            [
                {
                    "row_id": f.row_id,
                    "column": f.column,
                    "value": f.value,
                    "flags": f.flags,
                    "articles": f.articles,
                }
                for f in self.findings
            ],
            ensure_ascii=False,
            indent=2,
        )
        prompt = _COMPRESS_PROMPT.format(
            patterns=self.patterns or "(ninguno)",
            findings_json=findings_json,
        )
        try:
            self.patterns = llm_fn([{"role": "user", "content": prompt}]).strip()
            logger.info("Memory compressed: %d findings → patterns (%d chars)",
                        len(self.findings), len(self.patterns))
        except Exception as e:
            logger.warning("Memory compression failed: %s", e)
            self.patterns = (
                (self.patterns + "\n" if self.patterns else "") +
                "; ".join(f"{f.column}={f.value} ({', '.join(f.articles)})" for f in self.findings)
            ).strip()
        self.findings.clear()

    def as_prompt_str(self) -> str:
        """Compact context for per-row LLM prompts."""
        parts = [f"Filas procesadas: {self.rows_processed} | Marcadas: {self.rows_flagged}"]
        if self.patterns:
            parts.append(f"Patrones detectados: {self.patterns}")
        if self.findings:
            recent = self.findings[-3:]
            parts.append(
                "Últimos hallazgos: " +
                "; ".join(f"{f.column}={f.value} → {', '.join(f.flags[:1])}" for f in recent)
            )
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "rows_processed": self.rows_processed,
            "rows_flagged": self.rows_flagged,
            "findings": [
                {"row_id": f.row_id, "column": f.column, "value": f.value,
                 "verdict": f.verdict, "flags": f.flags, "articles": f.articles}
                for f in self.findings
            ],
            "patterns": self.patterns,
        }
