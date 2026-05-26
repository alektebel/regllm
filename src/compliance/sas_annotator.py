"""SAS Code Annotator — adds inline regulation comments to SAS code.

Given a list of SASCodeBlock objects (from SASParser), annotates each
logical SAS statement with a /* comment */ citing the relevant regulation
retrieved via RAG, and a brief compliance note from the SLM.

Output: a single annotated string with comments inserted above each statement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Split SAS code into logical units at these boundaries
_STEP_BOUNDARY = re.compile(
    r"(?=\b(?:PROC|DATA\s+\w|%MACRO|%MEND|LIBNAME|FILENAME)\b)",
    re.IGNORECASE,
)

# Regulatory keywords that trigger a RAG lookup for a unit
_REG_FIELDS = re.compile(
    r"\b(PD|LGD|EAD|CCF|RATING|STAGE|ECL|RWA|DEFAULT|CURE|DOWNTURN|MARGIN"
    r"|PROVISION|CALIBR|OBSERV|PERIOD|INCUMPL|FLOOR|SUELO)\b",
    re.IGNORECASE,
)

_ANNOTATION_PROMPT = """\
Eres un experto regulatorio IRB/IFRS9. Se te proporciona un fragmento de código SAS \
y el texto regulatorio recuperado de la base de conocimiento.

Genera UNA sola línea de comentario SAS (/* ... */) que explique si la lógica del \
fragmento cumple, incumple o no está relacionada con la regulación. Máximo 120 caracteres. \
Solo devuelve el comentario, sin más texto.

FRAGMENTO SAS:
{code}

CONTEXTO REGULATORIO:
{regulation}
"""


@dataclass
class AnnotatedBlock:
    task_name: str
    annotated_code: str
    warnings: list[str]   # statements flagged as potentially non-compliant


class SASAnnotator:
    """
    Annotates SAS code blocks with inline regulation comments.

    Args:
        rag_system: RegulatoryRAGSystem instance (may be None — skips RAG, adds placeholders)
        llm_fn: callable(messages) → str (same interface as ComplianceChecker._llm_fn)
        rag_n_results: chunks to retrieve per statement
    """

    def __init__(self, rag_system=None, llm_fn=None, rag_n_results: int = 3):
        self.rag = rag_system
        self.llm_fn = llm_fn
        self.rag_n_results = rag_n_results

    def annotate(self, blocks) -> list[AnnotatedBlock]:
        """Annotate a list of SASCodeBlock objects."""
        from src.sas_parser import SASCodeBlock  # avoid circular at module level
        results = []
        for block in blocks:
            annotated = self._annotate_block(block)
            results.append(annotated)
        return results

    def render(self, annotated_blocks: list[AnnotatedBlock]) -> str:
        """Concatenate all annotated blocks into a single .sas string."""
        parts = []
        for ab in annotated_blocks:
            parts.append(f"/* ══ Task: {ab.task_name} ══ */")
            parts.append(ab.annotated_code)
            parts.append("")
        return "\n".join(parts)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _annotate_block(self, block) -> AnnotatedBlock:
        units = self._split_into_units(block.code)
        annotated_units = []
        warnings = []

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            if not _REG_FIELDS.search(unit):
                # Not regulatory — pass through unchanged
                annotated_units.append(unit)
                continue

            comment, is_warning = self._generate_comment(unit)
            annotated_units.append(f"{comment}\n{unit}")
            if is_warning:
                warnings.append(unit[:120])

        return AnnotatedBlock(
            task_name=block.task_name,
            annotated_code="\n\n".join(annotated_units),
            warnings=warnings,
        )

    def _split_into_units(self, code: str) -> list[str]:
        """Split SAS code at PROC/DATA/macro step boundaries."""
        parts = _STEP_BOUNDARY.split(code)
        # Also split large single steps at RUN; / QUIT; boundaries
        result = []
        for part in parts:
            if len(part) > 800:
                sub = re.split(r"(?<=RUN;)\s*|(?<=QUIT;)\s*", part, flags=re.IGNORECASE)
                result.extend(sub)
            else:
                result.append(part)
        return [p.strip() for p in result if p.strip()]

    def _retrieve_regulation(self, code_unit: str) -> str:
        """Query RAG for regulation relevant to this code unit."""
        if self.rag is None:
            return "(RAG no disponible)"

        # Build query from field names found in the code
        fields = list(dict.fromkeys(
            m.upper() for m in _REG_FIELDS.findall(code_unit)
        ))
        query = f"requisitos regulatorios IRB IFRS9 para: {', '.join(fields[:5])}"

        try:
            chunks = self.rag.buscar_hibrida(query, n_resultados=self.rag_n_results)
            return self.rag.formatear_contexto(chunks)
        except Exception as e:
            logger.debug("RAG error for annotation: %s", e)
            return "(error RAG)"

    def _generate_comment(self, code_unit: str) -> tuple[str, bool]:
        """Returns (comment_string, is_potential_violation)."""
        regulation = self._retrieve_regulation(code_unit)

        if self.llm_fn is None or regulation.startswith("("):
            # No LLM or no RAG — produce a placeholder comment
            fields = _REG_FIELDS.findall(code_unit)
            return f"/* [Revisar] Parámetros regulatorios detectados: {', '.join(fields[:4])} */", False

        prompt = _ANNOTATION_PROMPT.format(
            code=code_unit[:600],
            regulation=regulation[:1200],
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self.llm_fn(messages).strip()
            # Clean up: ensure it's wrapped in /* */
            if not raw.startswith("/*"):
                raw = f"/* {raw.rstrip('*/ ').strip()} */"
            is_warning = any(w in raw.lower() for w in ("incumpl", "viola", "no cumple", "inferior", "por debajo"))
            return raw, is_warning
        except Exception as e:
            logger.debug("LLM annotation error: %s", e)
            return f"/* [Error en anotación: {e}] */", False
