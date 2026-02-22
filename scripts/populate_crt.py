#!/usr/bin/env python3
"""
Populate the Regulation Citation Tree (CRT) from scraped regulation documents.

Sources used:
  - EBA Opinion on RTS on IRB Assessment Methodology (77 articles, EN)
  - EBA Annex to 180 DPD Opinion (section-based, EN)
  - BOE Ley 19/2013 Transparencia (22 articles, ES)
  - ECB SSM Supervisory Manual (chapter-based, ES)
  - CRR skeleton — articles most referenced in the QA dataset
  - EBA/GL/2017/16 skeleton — articles most referenced in the QA dataset

Outputs:
  data/citation_trees/eu_regulations.json   (machine-readable CRT)
  data/citation_trees/eu_regulations.md     (human-readable export)
"""

import json
import re
import sys
import logging
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.citation_tree import RegulationCitationTree, CitationType

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CRT_PATH = PROJECT_ROOT / "data/citation_trees/eu_regulations.json"
MD_PATH  = PROJECT_ROOT / "data/citation_trees/eu_regulations.md"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse whitespace artefacts from PDF extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_raw_docs() -> list[dict]:
    """Load all raw scraped docs, deduplicating by title."""
    raw_dir = PROJECT_ROOT / "data/raw"
    docs: list[dict] = []
    seen_titles: set[str] = set()

    for path in sorted(raw_dir.glob("*.json")):
        with open(path) as f:
            batch = json.load(f)
        for d in batch:
            title = d.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                docs.append(d)

    logger.info(f"Loaded {len(docs)} unique documents from {raw_dir}")
    return docs


# ─── Article-level parsers ────────────────────────────────────────────────────

def _parse_paragraphs(text: str) -> list[tuple[str, str]]:
    """Extract numbered paragraphs (1., 2., …) from article body.
    Returns list of (paragraph_ref, paragraph_text)."""
    paragraphs = []
    # Match '1. Text…' or '(a) Text…' at start of line
    para_pattern = re.compile(
        r"(?:^|\n)(\d+)\.\s+(.+?)(?=\n\d+\.\s+|\Z)",
        re.DOTALL
    )
    for m in para_pattern.finditer(text):
        num  = m.group(1)
        body = _clean(m.group(2))
        if len(body) > 20:
            paragraphs.append((f"Paragraph {num}", body))
    return paragraphs


def _parse_points(text: str) -> list[tuple[str, str]]:
    """Extract lettered points ((a), (b), …) from paragraph text."""
    points = []
    point_pattern = re.compile(
        r"(?:^|\n)\(([a-z])\)\s+(.+?)(?=\n\([a-z]\)\s+|\Z)",
        re.DOTALL
    )
    for m in point_pattern.finditer(text):
        letter = m.group(1)
        body   = _clean(m.group(2))
        if len(body) > 10:
            points.append((f"Point ({letter})", body))
    return points


def _add_article(tree: RegulationCitationTree, regulation_ref: str,
                 article_ref: str, title: str, body: str,
                 parse_children: bool = True):
    """Add an article node (and optionally its paragraphs/points) to the tree."""
    full_title = f"{title}" if title else article_ref
    tree.add_citation(
        reference=article_ref,
        citation_type=CitationType.ARTICLE,
        text=_clean(body[:2000]),          # cap to avoid bloat
        parent_reference=regulation_ref,
        metadata={"title": full_title, "char_count": len(body)},
    )

    if not parse_children:
        return

    paras = _parse_paragraphs(body)
    for para_ref, para_text in paras:
        full_para_ref = f"{article_ref} {para_ref}"
        tree.add_citation(
            reference=full_para_ref,
            citation_type=CitationType.PARAGRAPH,
            text=_clean(para_text[:1000]),
            parent_reference=article_ref,
        )
        # Add points under each paragraph
        for point_ref, point_text in _parse_points(para_text):
            full_point_ref = f"{full_para_ref} {point_ref}"
            tree.add_citation(
                reference=full_point_ref,
                citation_type=CitationType.POINT,
                text=_clean(point_text[:500]),
                parent_reference=full_para_ref,
            )


# ─── Per-source populators ────────────────────────────────────────────────────

def populate_eba_rts_irb(tree: RegulationCitationTree, doc: dict):
    """Parse EBA Opinion on RTS on IRB Assessment Methodology (77 EN articles)."""
    reg_ref = "EBA RTS IRB"
    tree.add_citation(
        reference=reg_ref,
        citation_type=CitationType.REGULATION,
        text=(
            "EBA Opinion on RTS on IRB Assessment Methodology. "
            "Specifies the methodology competent authorities shall follow when "
            "assessing compliance with IRB approach requirements under CRR "
            "Articles 144(2), 173(3) and 180(3)(b)."
        ),
        metadata={"source": "EBA", "url": doc.get("url", ""), "language": "en"},
    )

    text = doc["text"]
    # Article headers: "\nArticle N\nTitle\n"
    art_pattern = re.compile(r"\nArticle\s+(\d+[a-z]?)\s*\n([^\n]{0,150})\n", re.IGNORECASE)
    matches = list(art_pattern.finditer(text))

    for i, m in enumerate(matches):
        art_num   = m.group(1)
        art_title = m.group(2).strip()
        start     = m.end()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body      = text[start:end]

        _add_article(tree, reg_ref, f"Article {art_num}", art_title, body)

    logger.info(f"  {reg_ref}: {len(matches)} articles")


def populate_eba_annex_180dpd(tree: RegulationCitationTree, doc: dict):
    """Parse EBA Annex to 180 DPD Opinion (section-based, EN)."""
    reg_ref = "EBA Op 2017/17 Annex"
    tree.add_citation(
        reference=reg_ref,
        citation_type=CitationType.REGULATION,
        text=(
            "EBA Annex to Opinion on the use of the 180 days past due criterion "
            "to identify defaulted exposures under CRR Article 178."
        ),
        metadata={"source": "EBA", "url": doc.get("url", ""), "language": "en"},
    )

    text = doc["text"]
    # Numbered sections at top level: "N. Title"
    sec_pattern = re.compile(r"\n(\d+)\.\s+([A-Z][^\n]{5,80})\n")
    matches = list(sec_pattern.finditer(text))

    for i, m in enumerate(matches):
        sec_num   = m.group(1)
        sec_title = m.group(2).strip()
        start     = m.end()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body      = text[start:end]

        sec_ref = f"Section {sec_num}"
        tree.add_citation(
            reference=sec_ref,
            citation_type=CitationType.SECTION,
            text=_clean(body[:2000]),
            parent_reference=reg_ref,
            metadata={"title": sec_title},
        )

    logger.info(f"  {reg_ref}: {len(matches)} sections")


def populate_boe_ley19_2013(tree: RegulationCitationTree, doc: dict):
    """Parse BOE Ley 19/2013 Transparencia (22 Spanish articles)."""
    reg_ref = "Ley 19/2013"
    tree.add_citation(
        reference=reg_ref,
        citation_type=CitationType.DIRECTIVE,
        text=(
            "Ley 19/2013, de 9 de diciembre, de transparencia, acceso a la "
            "información pública y buen gobierno. BOE-A-2013-12887."
        ),
        metadata={"source": "BOE", "url": doc.get("url", ""), "language": "es"},
    )

    text = doc["text"]
    art_pattern = re.compile(r"\nArt[ií]culo\s+(\d+[a-z]?)\.?\s*\n([^\n]{0,150})\n", re.IGNORECASE)
    matches = list(art_pattern.finditer(text))

    for i, m in enumerate(matches):
        art_num   = m.group(1)
        art_title = m.group(2).strip()
        start     = m.end()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body      = text[start:end]

        _add_article(tree, reg_ref, f"Artículo {art_num}", art_title, body)

    logger.info(f"  {reg_ref}: {len(matches)} artículos")


def populate_ecb_ssm(tree: RegulationCitationTree, doc: dict):
    """Parse ECB SSM Supervisory Manual (chapter-based, ES)."""
    reg_ref = "ECB SSM Manual"
    tree.add_citation(
        reference=reg_ref,
        citation_type=CitationType.REGULATION,
        text=(
            "Manual de Supervisión del BCE (Mecanismo Único de Supervisión). "
            "Describe las metodologías y procesos de supervisión bancaria del BCE."
        ),
        metadata={"source": "ECB", "url": doc.get("url", ""), "language": "es"},
    )

    text = doc["text"]

    # The ECB PDF extracts as mixed-case short titles followed by body text.
    # Strategy: find lines that look like section headers (short, sentence-case,
    # not ending in punctuation, not starting with digits/parens).
    lines = text.split("\n")
    section_indices: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if (15 <= len(s) <= 80
                and not s[0].isdigit()
                and not s.startswith("(")
                and not s.endswith(".")
                and not s.endswith(",")
                and s[0].isupper()
                and sum(1 for c in s if c.isupper()) < len(s) * 0.7):  # not ALL CAPS
            section_indices.append((i, s))

    # Deduplicate consecutive repeated titles (PDF extraction artefact)
    deduped: list[tuple[int, str]] = []
    seen_titles: set[str] = set()
    for idx, title in section_indices:
        if title not in seen_titles:
            seen_titles.add(title)
            deduped.append((idx, title))

    added = 0
    for j, (line_idx, title) in enumerate(deduped):
        next_line_idx = deduped[j + 1][0] if j + 1 < len(deduped) else len(lines)
        body = "\n".join(lines[line_idx + 1:next_line_idx])

        if len(body.strip()) < 50:
            continue

        chap_ref = f"Chapter: {title[:60]}"
        tree.add_citation(
            reference=chap_ref,
            citation_type=CitationType.CHAPTER,
            text=_clean(body[:2000]),
            parent_reference=reg_ref,
        )
        added += 1

    logger.info(f"  {reg_ref}: {added} chapters")


# ─── Skeletons from QA references ────────────────────────────────────────────

# CRR articles referenced in QA pairs (Article N: human title)
CRR_ARTICLES = {
    "92":  "Own funds requirements",
    "143": "Permission to use the IRB Approach",
    "144": "Conditions for implementing the IRB Approach",
    "147": "Method for assigning exposures to exposure classes",
    "150": "Conditions for permanent partial use",
    "153": "Risk-weighted exposure amounts for corporate, institution and sovereign exposures",
    "158": "Treatment of expected loss amounts",
    "159": "Treatment of expected loss amounts",
    "169": "Principles for rating systems",
    "172": "Assignment of exposures",
    "173": "Assessment of the integrity of the assignment process",
    "178": "Default of an obligor",
    "180": "Requirements specific to PD estimation",
    "181": "Requirements specific to own-LGD estimates",
}

# EBA/GL/2017/16 articles most referenced in QA data
EBA_GL_ARTICLES = {
    "42":  "Margin of conservatism",
    "43":  "Additional margin of conservatism",
    "44":  "PD estimation – general requirements",
    "45":  "Long-run average PD",
    "46":  "Central tendency",
    "47":  "Calibration of PD estimates",
    "48":  "LGD estimation – general requirements",
    "49":  "Realised LGD",
    "50":  "Economic downturn LGD",
    "51":  "Long-run average LGD",
    "52":  "Calibration of LGD estimates",
    "93":  "Reference data sets",
    "123": "Internal validation",
    "124": "Internal validation — model performance",
    "125": "Approval process for use of models",
}


def populate_crr_skeleton(tree: RegulationCitationTree):
    """Add CRR with all articles referenced in the QA dataset."""
    reg_ref = "CRR"
    if tree.get_citation_by_reference(reg_ref) is None:
        tree.add_citation(
            reference=reg_ref,
            citation_type=CitationType.REGULATION,
            text=(
                "Regulation (EU) No 575/2013 — Capital Requirements Regulation. "
                "Lays down uniform rules concerning general prudential requirements "
                "for institutions supervised under CRD IV/V."
            ),
            metadata={"source": "EUR-Lex", "language": "en/es"},
        )

    for art_num, title in CRR_ARTICLES.items():
        art_ref = f"Article {art_num}"
        if tree.get_citation_by_reference(art_ref) is None:
            tree.add_citation(
                reference=art_ref,
                citation_type=CitationType.ARTICLE,
                text=title,
                parent_reference=reg_ref,
                metadata={"title": title},
            )

    logger.info(f"  CRR: {len(CRR_ARTICLES)} article stubs")


def populate_eba_gl_2017_16_skeleton(tree: RegulationCitationTree):
    """Add EBA/GL/2017/16 with most-referenced articles from QA data."""
    reg_ref = "EBA/GL/2017/16"
    tree.add_citation(
        reference=reg_ref,
        citation_type=CitationType.REGULATION,
        text=(
            "EBA Guidelines on PD estimation, LGD estimation and the treatment "
            "of defaulted exposures (EBA/GL/2017/16). Key reference for IRB "
            "parameter estimation under CRR."
        ),
        metadata={"source": "EBA", "language": "en/es"},
    )

    for art_num, title in EBA_GL_ARTICLES.items():
        art_ref = f"Artículo {art_num}"
        tree.add_citation(
            reference=art_ref,
            citation_type=CitationType.ARTICLE,
            text=title,
            parent_reference=reg_ref,
            metadata={"title": title},
        )

    logger.info(f"  EBA/GL/2017/16: {len(EBA_GL_ARTICLES)} article stubs")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Initialising CRT...")
    CRT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tree = RegulationCitationTree(
        name="EU Regulations",
        persist_path=str(CRT_PATH),
    )

    # ── 1. Load raw documents ──
    docs = _load_raw_docs()
    by_title = {d["title"]: d for d in docs}

    # ── 2. Populate from scraped sources ──
    logger.info("Populating from scraped documents...")

    # EBA RTS IRB (most article-rich source)
    eba_rts = next(
        (d for d in docs if "IRB" in d.get("title", "") and "RTS" in d.get("title", "")),
        None,
    )
    if eba_rts:
        populate_eba_rts_irb(tree, eba_rts)
    else:
        logger.warning("EBA RTS IRB document not found")

    # EBA 180 DPD Annex
    eba_annex = next(
        (d for d in docs if "Annex" in d.get("title", "") and "180" in d.get("title", "")),
        None,
    )
    if eba_annex:
        populate_eba_annex_180dpd(tree, eba_annex)
    else:
        logger.warning("EBA Annex 180 DPD document not found")

    # BOE Ley 19/2013
    boe = next(
        (d for d in docs if d.get("source") == "BOE" and "19/2013" in d.get("title", "")),
        None,
    )
    if boe:
        populate_boe_ley19_2013(tree, boe)
    else:
        logger.warning("BOE Ley 19/2013 document not found")

    # ECB SSM Manual
    ecb = next((d for d in docs if d.get("source") == "ECB"), None)
    if ecb:
        populate_ecb_ssm(tree, ecb)
    else:
        logger.warning("ECB SSM document not found")

    # ── 3. Skeletons for regulations cited in QA but not scraped ──
    logger.info("Adding regulation skeletons from QA references...")
    populate_crr_skeleton(tree)
    populate_eba_gl_2017_16_skeleton(tree)

    # ── 4. Persist ──
    logger.info(f"Saving CRT to {CRT_PATH}...")
    tree.save()

    # ── 5. Markdown export ──
    logger.info(f"Exporting Markdown to {MD_PATH}...")
    tree.export_markdown(str(MD_PATH))

    # ── 6. Stats ──
    stats = tree.get_statistics()
    print()
    print("=" * 55)
    print("  Citation Tree populated successfully")
    print("=" * 55)
    print(f"  Total nodes     : {stats['total_nodes']}")
    print(f"  Root nodes      : {stats['root_nodes']}")
    print(f"  Max depth       : {stats['max_depth']}")
    print(f"  Total text (ch) : {stats['total_text_length']:,}")
    print()
    print("  Nodes by type:")
    for node_type, count in sorted(stats["nodes_by_type"].items(), key=lambda x: -x[1]):
        print(f"    {node_type:<14} {count}")
    print("=" * 55)
    print(f"  Saved : {CRT_PATH}")
    print(f"  Markdown : {MD_PATH}")
    print()

    # ── 7. Quick lookup test ──
    print("=== Quick lookup test ===")
    test_refs = ["Article 1", "Artículo 42", "CRR", "EBA/GL/2017/16", "Article 178"]
    for ref in test_refs:
        node = tree.get_citation_by_reference(ref)
        if node:
            print(f"  OK  {ref!r:30s} -> {node.context}")
        else:
            print(f"  --  {ref!r:30s} -> not found")
    print()


if __name__ == "__main__":
    main()
