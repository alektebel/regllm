"""Variable mapper — maps regulation concepts to database columns.

Runs ONCE per SAS version. Produces ``data/audit/mapping.json``, which is the
bridge between regulation article language and actual database column names.

Pipeline
--------
1. Parse all SAS code via ``SASLogicTree`` to extract field computation
   descriptions (lineage ancestors + data step names).
2. Read regulation articles via ``GraphRAG`` to collect every field concept
   mentioned across articles.
3. For each regulation concept, call the local LLM to find the matching
   database column name and supporting evidence.
4. Persist the mapping with confidence scores.

Confidence thresholds
---------------------
- ``>= 0.85``  High — mapping approved automatically
- ``0.70–0.84``  Medium — flag for review but usable
- ``< 0.70``   Low — **mandatory human review before any DQC query can use this**

Usage
-----
As a script::

    python -m src.audit.variable_mapper --sas-version v3

As a library::

    from src.audit.variable_mapper import build_mapping, load_mapping
    mapping = build_mapping(sas_version="v3")
    entry = load_mapping()["PROVISION_PERIOD_MONTHS"]

Filling the mapping manually
-----------------------------
See ``data/audit/mapping_guide.md`` for the full field-by-field guide.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUT = _PROJECT_ROOT / "data" / "audit" / "mapping.json"
_DEFAULT_REGULATION_DIR = _PROJECT_ROOT / "data" / "regulation"
_DEFAULT_DOCS_DIR = _PROJECT_ROOT / "data" / "docs" / "regulation"
_SAMPLES = _PROJECT_ROOT / "data" / "samples"
_SAS_ROOT = _PROJECT_ROOT / "data" / "sas"


# ── Data contract ─────────────────────────────────────────────────────────────


@dataclass
class MappingEntry:
    """One mapping between a regulation concept and a database column.

    Fields
    ------
    db_column : str
        Exact column name in ``mylib.ciclos_recuperacion``.
    regulation_concept : str
        The natural-language concept as expressed in the regulation text.
    regulation_variable : str
        The formal variable name used in the regulation (often matching
        the DB column in well-named schemas, but not always).
    articles : list[str]
        Article IDs (matching filenames in ``data/regulation/``) that
        define or constrain this variable.
    sas_variable : str
        The variable name as it appears in the SAS source code.
        May differ from ``db_column`` in legacy programmes.
    sas_lineage : list[str]
        Direct ancestor fields that feed into this variable in the SAS
        pipeline, as returned by ``trace_field_ancestors()``.
    computation_description : str
        One or two sentences describing HOW the SAS code computes this
        field — extracted from lineage + SAS comments.
    regulation_description : str
        One or two sentences describing what the regulation means by this
        concept — sourced from the article md files.
    confidence : float
        Score in [0, 1]. See module docstring for thresholds.
    needs_review : bool
        Set automatically when ``confidence < 0.70``. Also set manually
        when a human reviewer has a question about the mapping.
    notes : str
        Free-text notes from the human reviewer.
    """
    db_column: str
    regulation_concept: str
    regulation_variable: str
    articles: list[str]
    sas_variable: str
    sas_lineage: list[str]
    computation_description: str
    regulation_description: str
    confidence: float
    needs_review: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.confidence < 0.70:
            self.needs_review = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MappingEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_sas(version: str) -> str:
    folder = _SAS_ROOT / version
    if folder.exists():
        files = sorted(folder.rglob("*.sas"))
        if files:
            return "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    sample = _SAMPLES / "sample_lgd.sas"
    return sample.read_text(encoding="utf-8") if sample.exists() else ""


def _sas_lineage_for_fields(sas_code: str) -> dict[str, dict[str, Any]]:
    """Return lineage info keyed by field name (uppercase)."""
    from src.sas_logic_tree import SASLogicTree, trace_field_ancestors
    tree = SASLogicTree()
    try:
        nodes = tree.parse(sas_code)
        lg = tree.lineage(nodes)
    except Exception as e:
        logger.warning("SAS parse failed: %s", e)
        return {}

    result: dict[str, dict[str, Any]] = {}
    node_map = {n["id"]: n for n in lg.nodes}
    for field_id, node in node_map.items():
        ancestors_trace = trace_field_ancestors(lg, field_id)
        result[field_id] = {
            "sas_variable": field_id,
            "data_steps": node.get("data_steps", []),
            "direct_predecessors": [
                e["source"] for e in ancestors_trace.get("direct_predecessors", [])
            ],
            "ancestors": ancestors_trace.get("ancestors", [])[:20],
            "layer": node.get("layer", 0),
        }
    return result


def _regulation_field_mentions(regulation_dir: Path) -> dict[str, list[str]]:
    """Return {field_name → [article_id, ...]} from regulation md files."""
    import re
    _FIELD_RE = re.compile(r"`([A-Z][A-Z0-9_]{2,})`")
    result: dict[str, list[str]] = {}
    if not regulation_dir.is_dir():
        return result
    for md in sorted(regulation_dir.glob("*.md")):
        article_id = md.stem
        text = md.read_text(encoding="utf-8")
        for m in _FIELD_RE.finditer(text):
            fname = m.group(1)
            result.setdefault(fname, [])
            if article_id not in result[fname]:
                result[fname].append(article_id)
    return result


def _llm_propose_mapping(
    db_column: str,
    sas_info: dict[str, Any],
    reg_articles: list[str],
    regulation_dir: Path,
    client: Any,
) -> dict[str, Any]:
    """Ask the LLM to describe the mapping for one field. Returns a dict."""
    # Build article excerpts for context
    excerpts: list[str] = []
    for art in reg_articles[:3]:
        md = regulation_dir / f"{art}.md"
        if md.exists():
            text = md.read_text(encoding="utf-8")
            # find paragraphs mentioning the field
            lines = [ln for ln in text.splitlines() if db_column in ln]
            if lines:
                excerpts.append(f"[{art}] " + " | ".join(lines[:4]))

    ancestors_str = ", ".join(sas_info.get("ancestors", [])[:10]) or "(none)"
    steps_str = ", ".join(sas_info.get("data_steps", [])[:5]) or "(unknown)"
    excerpts_str = "\n".join(excerpts) or "(no direct mentions found)"

    system = (
        "You are a regulatory data dictionary assistant. "
        "Given information about a database column and its SAS computation context, "
        "produce a JSON object describing the mapping between the regulation concept "
        "and the database implementation. Reply with strict JSON only."
    )
    user = f"""Database column: {db_column}
SAS variable name: {sas_info.get('sas_variable', db_column)}
SAS ancestor fields: {ancestors_str}
SAS data steps: {steps_str}
Regulation articles referencing this field: {', '.join(reg_articles) or 'none'}
Regulation text excerpts:
{excerpts_str}

Produce JSON with these fields:
{{
  "regulation_concept": "natural language name of the regulatory concept",
  "regulation_variable": "formal name used in regulation (often same as db_column)",
  "computation_description": "1-2 sentences: how does SAS compute this field?",
  "regulation_description": "1-2 sentences: what does the regulation mean by this concept?",
  "confidence": 0.0..1.0
}}"""

    try:
        data = client.chat_json(system, user, max_tokens=512)
        return data
    except Exception as e:
        logger.warning("LLM call failed for %s: %s", db_column, e)
        return {}


# ── Public API ────────────────────────────────────────────────────────────────


def build_mapping(
    sas_version: str = "v3",
    regulation_dir: Path | str = _DEFAULT_REGULATION_DIR,
    out_path: Path | str = _DEFAULT_OUT,
    llm_client: Any = None,
    db_columns: list[str] | None = None,
) -> dict[str, MappingEntry]:
    """Build the variable mapping from SAS code + regulation articles.

    Parameters
    ----------
    sas_version:
        Which SAS version to parse (``"v2"`` or ``"v3"``).
    regulation_dir:
        Directory containing regulation ``*.md`` files.
    out_path:
        Where to write ``mapping.json``.
    llm_client:
        A ``LocalLLMClient`` instance. If ``None``, the module-level singleton
        is used. In stub mode, the LLM returns placeholder descriptions.
    db_columns:
        Explicit list of DB column names to map. If ``None``, the union of
        columns found in the SAS lineage and regulation article field mentions
        is used.

    Returns
    -------
    dict[str, MappingEntry]
        Mapping keyed by ``db_column`` (uppercase).
    """
    from src.knowledge.llm_client import get_client

    regulation_dir = Path(regulation_dir)
    out_path = Path(out_path)
    client = llm_client or get_client()

    logger.info("Building variable mapping for SAS version %s", sas_version)

    # Step 1 — SAS lineage
    sas_code = _load_sas(sas_version)
    lineage_map = _sas_lineage_for_fields(sas_code) if sas_code else {}
    logger.info("SAS lineage extracted: %d fields", len(lineage_map))

    # Step 2 — Regulation field mentions
    reg_mentions = _regulation_field_mentions(regulation_dir)
    logger.info("Regulation mentions: %d distinct fields", len(reg_mentions))

    # Step 3 — Determine target column set
    if db_columns is None:
        db_columns = sorted(set(lineage_map) | set(reg_mentions))

    # Step 4 — Build one entry per column
    entries: dict[str, MappingEntry] = {}
    for col in db_columns:
        sas_info = lineage_map.get(col, {"sas_variable": col, "ancestors": [], "data_steps": []})
        articles = reg_mentions.get(col, [])

        llm_data = _llm_propose_mapping(col, sas_info, articles, regulation_dir, client)

        entry = MappingEntry(
            db_column=col,
            regulation_concept=llm_data.get("regulation_concept", col.replace("_", " ").lower()),
            regulation_variable=llm_data.get("regulation_variable", col),
            articles=articles,
            sas_variable=sas_info.get("sas_variable", col),
            sas_lineage=sas_info.get("ancestors", [])[:10],
            computation_description=llm_data.get(
                "computation_description",
                f"Computed in SAS data steps: {', '.join(sas_info.get('data_steps', []))}"
                if sas_info.get("data_steps") else "Computation description not available.",
            ),
            regulation_description=llm_data.get(
                "regulation_description",
                f"Referenced in regulation articles: {', '.join(articles)}"
                if articles else "Not directly referenced in loaded regulation articles.",
            ),
            confidence=float(llm_data.get("confidence", 0.60 if articles else 0.40)),
        )
        entries[col] = entry
        logger.debug("Mapped %s → confidence %.2f", col, entry.confidence)

    # Step 5 — Persist
    _save_mapping(entries, out_path, sas_version)
    logger.info("Mapping written: %d entries → %s", len(entries), out_path)
    return entries


def _save_mapping(
    entries: dict[str, MappingEntry],
    path: Path,
    sas_version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": sas_version,
        "generated_at": date.today().isoformat(),
        "entry_count": len(entries),
        "mappings": {k: v.to_dict() for k, v in entries.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_mapping(path: Path | str = _DEFAULT_OUT) -> dict[str, MappingEntry]:
    """Load a previously saved mapping. Returns empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        logger.warning("Mapping file not found: %s. Run build_mapping() first.", p)
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: MappingEntry.from_dict(v) for k, v in data.get("mappings", {}).items()}


def mapping_report(path: Path | str = _DEFAULT_OUT) -> str:
    """Print a human-readable summary of the current mapping."""
    entries = load_mapping(path)
    if not entries:
        return "No mapping loaded."
    lines = [
        f"{'Column':<35} {'Confidence':>10}  {'Review':>6}  Articles",
        "-" * 80,
    ]
    needs_review = []
    for col, e in sorted(entries.items()):
        flag = "  YES" if e.needs_review else "   no"
        arts = ", ".join(e.articles) if e.articles else "—"
        lines.append(f"{col:<35} {e.confidence:>10.2f}  {flag}  {arts}")
        if e.needs_review:
            needs_review.append(col)
    lines.append("-" * 80)
    lines.append(f"Total: {len(entries)} entries, {len(needs_review)} flagged for review")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build the regulation → DB variable mapping.")
    parser.add_argument("--sas-version", default="v3", choices=["v2", "v3"])
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="Output path for mapping.json")
    parser.add_argument("--report", action="store_true", help="Print report of existing mapping")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.report:
        print(mapping_report(args.out))
        return

    entries = build_mapping(sas_version=args.sas_version, out_path=args.out)
    print(mapping_report(args.out))
    needs = [c for c, e in entries.items() if e.needs_review]
    if needs:
        print(f"\n⚠  {len(needs)} columns flagged for review:")
        for c in needs:
            print(f"   {c}")


if __name__ == "__main__":
    _cli()
