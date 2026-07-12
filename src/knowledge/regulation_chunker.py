"""Chunk the EBA GL/2017/16 PD & LGD guidelines for embedding-based retrieval.

The ingested guideline text (``data/regulation/eba_gl_2017_16_articles.json``)
is already segmented into 221 numbered paragraphs, each tagged with its
section/subsection heading — a citation-precise unit that regulatory text is
naturally organised around. Chunking here therefore means:

- **one chunk per paragraph** by default (preserves exact paragraph-number
  citation for every retrieved chunk — important when the downstream
  consumer, the DQC system prompt, is explicitly instructed not to invent
  regulatory references);
- **long paragraphs are split** into overlapping, sentence-aligned windows
  (never mid-sentence/mid-number) so no single chunk exceeds an embedding
  model's practical context, while the ``overlap`` keeps each split part
  self-contained enough to embed meaningfully on its own.

Paragraphs are never *merged* — a chunk always maps back to exactly one
paragraph, so a retrieval hit can always be cited as "EBA GL 2017/16 §N".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "data" / "regulation" / "eba_gl_2017_16_articles.json"
)
SOURCE_DOC = "EBA GL 2017/16"

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class RegChunk:
    """One retrievable unit of the PD/LGD guidelines."""

    chunk_id: str          # stable id, e.g. "par_73#0"
    paragraph: int         # source paragraph number (for citation)
    section: str
    subsection: str
    text: str
    part: int              # 0-based index among this paragraph's chunks
    n_parts: int            # total chunks this paragraph was split into
    source_doc: str = SOURCE_DOC

    def citation(self) -> str:
        """Human-readable regulatory citation for this chunk."""
        return f"{self.source_doc} §{self.paragraph}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "paragraph": self.paragraph,
            "section": self.section, "subsection": self.subsection,
            "text": self.text, "part": self.part, "n_parts": self.n_parts,
            "source_doc": self.source_doc,
        }


def _split_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    """Sentence-aligned sliding window — never cuts mid-sentence.

    Greedily packs sentences into windows up to ``max_chars``; each new
    window is seeded with trailing sentences from the previous window
    (up to ``overlap`` chars) so consecutive chunks share context.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    if not sentences:
        return [text] if text.strip() else []

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent) + 1
        if current and current_len + sent_len > max_chars:
            windows.append(" ".join(current))
            # seed the next window with trailing sentences within `overlap`
            carry: list[str] = []
            carry_len = 0
            for s in reversed(current):
                if carry_len + len(s) + 1 > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s) + 1
            current = carry
            current_len = carry_len
        current.append(sent)
        current_len += sent_len
    if current:
        windows.append(" ".join(current))
    return windows


def chunk_eba_guidelines(
    path: Path | str = DEFAULT_SOURCE,
    *,
    max_chars: int = 1000,
    overlap: int = 150,
) -> list[RegChunk]:
    """Chunk the ingested EBA GL/2017/16 paragraphs for embedding.

    Args:
        path: JSON file of ``{id, paragraph, section, subsection, text}``
            records (the format produced by the regulation ingestor).
        max_chars: paragraphs longer than this are split into overlapping
            sentence-aligned windows.
        overlap: characters of trailing context carried into the next
            window when a paragraph is split.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks: list[RegChunk] = []
    for rec in records:
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        base_id = rec.get("id") or f"par_{rec['paragraph']}"
        parts = (
            [text] if len(text) <= max_chars
            else _split_long_paragraph(text, max_chars, overlap)
        )
        for i, part_text in enumerate(parts):
            chunks.append(RegChunk(
                chunk_id=f"{base_id}#{i}",
                paragraph=rec["paragraph"],
                section=rec.get("section", ""),
                subsection=rec.get("subsection", ""),
                text=part_text,
                part=i,
                n_parts=len(parts),
            ))
    return chunks
