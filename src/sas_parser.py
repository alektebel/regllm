"""SAS file parser — supports .sas (plain text) and .egp (SAS Enterprise Guide project).

An .egp file is a ZIP archive containing:
  - project.xml: all task definitions with embedded SAS code in <Text>/<TaskCode> elements
  - UUID-named dirs: result logs/images (skipped)
  - Optional external .sas files referenced from <CodeTask> nodes

Usage:
    parser = SASParser()
    blocks = parser.parse("project.egp")   # or "script.sas"
    for block in blocks:
        print(block.task_name, block.code[:200])
"""

from __future__ import annotations

import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# SAS code heuristic: must contain at least one of these keywords
_SAS_KEYWORDS = re.compile(
    r"\b(PROC|DATA\s|%MACRO|%LET|LIBNAME|FILENAME|ODS|TITLE|FOOTNOTE|RUN;|QUIT;)\b",
    re.IGNORECASE,
)

# Tags in project.xml that carry user-authored or generated SAS code
_CODE_TAGS = {"Text", "TaskCode", "BeginUserCode"}


@dataclass
class SASCodeBlock:
    source_file: str    # originating file (project.xml path or .sas filename)
    task_name: str      # task label from XML, or filename for plain .sas files
    code: str           # SAS source code
    block_type: str     # "embedded" | "generated" | "external"


class SASParser:
    """Parse .sas and .egp files into a list of SASCodeBlock objects."""

    def parse(self, path: str | Path) -> list[SASCodeBlock]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".egp":
            return self._parse_egp(path)
        # .txt files are treated as SAS if they contain SAS keywords
        return self._parse_sas(path)

    # ── .sas ─────────────────────────────────────────────────────────────────

    def _parse_sas(self, path: Path) -> list[SASCodeBlock]:
        code = path.read_text(encoding="utf-8", errors="replace").strip()
        if not code:
            return []
        return [SASCodeBlock(
            source_file=str(path),
            task_name=path.stem,
            code=code,
            block_type="external",
        )]

    # ── .egp ─────────────────────────────────────────────────────────────────

    def _parse_egp(self, path: Path) -> list[SASCodeBlock]:
        blocks: list[SASCodeBlock] = []
        seen_codes: set[str] = set()  # deduplicate identical code strings

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            # 1. Parse project.xml
            if "project.xml" in names:
                xml_bytes = zf.read("project.xml")
                blocks.extend(self._extract_from_xml(xml_bytes, str(path), seen_codes))

            # 2. Read any .sas files bundled in the zip (external programs)
            for name in names:
                if name.lower().endswith(".sas"):
                    code = zf.read(name).decode("utf-8", errors="replace").strip()
                    fingerprint = code[:200]
                    if code and fingerprint not in seen_codes and _SAS_KEYWORDS.search(code):
                        seen_codes.add(fingerprint)
                        blocks.append(SASCodeBlock(
                            source_file=name,
                            task_name=Path(name).stem,
                            code=code,
                            block_type="external",
                        ))

        return blocks

    def _extract_from_xml(
        self, xml_bytes: bytes, source: str, seen: set[str]
    ) -> list[SASCodeBlock]:
        blocks: list[SASCodeBlock] = []
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return blocks

        # Walk the full tree; collect code from known tags
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag not in _CODE_TAGS:
                continue
            code = (elem.text or "").strip()
            if len(code) < 30 or not _SAS_KEYWORDS.search(code):
                continue
            fingerprint = code[:200]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            # Determine task name from nearest ancestor with a Name/Label child
            task_name = self._nearest_task_name(elem, root) or tag

            block_type = "embedded" if tag == "BeginUserCode" else "generated"
            blocks.append(SASCodeBlock(
                source_file=source,
                task_name=task_name,
                code=code,
                block_type=block_type,
            ))

        return blocks

    def _nearest_task_name(self, target: ET.Element, root: ET.Element) -> str | None:
        """Walk upward to find the closest Name or Label sibling/parent text."""
        # Build parent map
        parent_map: dict[ET.Element, ET.Element] = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        elem: ET.Element | None = target
        while elem is not None:
            parent = parent_map.get(elem)
            if parent is not None:
                for child in parent:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_tag in ("Name", "Label", "TaskName") and child.text:
                        name = child.text.strip()
                        if name and len(name) < 120:
                            return name
            elem = parent
        return None
