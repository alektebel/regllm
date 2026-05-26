"""
SAS variable label extractor.

Reads .sas / .egp files via the existing SASParser, then extracts
LABEL=, ATTRIB, and FORMAT= statements that encode business meaning.
Returns (content_text, metadata) tuples ready for embedding and storage.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# LABEL var = "description" [var2 = "..."] ;
_LABEL_STMT_RE  = re.compile(r"LABEL\b(.*?);", re.IGNORECASE | re.DOTALL)
_LABEL_PAIR_RE  = re.compile(r"""(\w+)\s*=\s*["']([^"']+)["']""")

# ATTRIB var ... LABEL="..." FORMAT=xxx.;
_ATTRIB_STMT_RE  = re.compile(r"ATTRIB\b(.*?);", re.IGNORECASE | re.DOTALL)
_ATTRIB_LABEL_RE = re.compile(r"""(\w+)\b.*?LABEL\s*=\s*["']([^"']+)["']""")
_ATTRIB_FMT_RE   = re.compile(r"""(\w+)\b.*?FORMAT\s*=\s*(\S+)""")


def _parse_code(code: str) -> dict[str, dict]:
    """Return {VAR_NAME: {label, format}} extracted from SAS code."""
    info: dict[str, dict] = {}

    for m in _LABEL_STMT_RE.finditer(code):
        for pair in _LABEL_PAIR_RE.finditer(m.group(1)):
            var = pair.group(1).upper()
            info.setdefault(var, {})["label"] = pair.group(2).strip()

    for m in _ATTRIB_STMT_RE.finditer(code):
        body = m.group(1)
        for lm in _ATTRIB_LABEL_RE.finditer(body):
            var = lm.group(1).upper()
            info.setdefault(var, {})["label"] = lm.group(2).strip()
        for fm in _ATTRIB_FMT_RE.finditer(body):
            var = fm.group(1).upper()
            info.setdefault(var, {}).setdefault("format", fm.group(2).strip())

    return info


def extract_sas_labels(path: Path | str) -> list[tuple[str, dict]]:
    """Parse a .sas or .egp file for variable labels.

    Returns [(content_text, metadata), ...] — one entry per labelled variable.
    """
    from src.sas_parser import SASParser

    path = Path(path)
    parser = SASParser()
    try:
        blocks = parser.parse(path)
    except Exception as e:
        logger.warning("SASParser failed on %s: %s", path, e)
        return []

    results: list[tuple[str, dict]] = []
    seen: set[tuple[str, str]] = set()

    for block in blocks:
        task_name = block.task_name or path.stem
        for var_name, var_info in _parse_code(block.code).items():
            label = var_info.get("label", "")
            if not label:
                continue
            key = (task_name, var_name)
            if key in seen:
                continue
            seen.add(key)

            lines = [f"Programa SAS: {task_name} | Variable: {var_name}"]
            lines.append(f"Etiqueta (LABEL): {label}")
            fmt = var_info.get("format", "")
            if fmt:
                lines.append(f"Formato SAS: {fmt}")
            lines.append(f"Fuente: {path.name} (SAS LABEL/ATTRIB)")

            results.append(("\n".join(lines), {
                "source_type": "sas_labels",
                "column_name": var_name,
                "table_name": task_name,
                "source_file": path.name,
                "confidence": "high",
            }))

    logger.info("sas_labels: %d variable labels from %s", len(results), path.name)
    return results
