"""Markdown documentation indexer with a tiny pure-Python BM25.

Walks ``data/docs/**/*.md``, splits each file by H1/H2 headings into
``Section`` chunks, builds an Okapi-BM25 index, and persists it to
``data/docs/index.json`` for fast reload.

No external dependencies — banking glossary corpora are typically small
enough (<10k sections) that an inline BM25 is plenty. Swap in a vector
backend later behind the same ``DocsIndex.search()`` interface if needed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DOCS_ROOT = _PROJECT_ROOT / "data" / "docs"
_DEFAULT_INDEX_PATH = _DEFAULT_DOCS_ROOT / "index.json"

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ_][A-Za-zÀ-ÿ0-9_]*")

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is",
    "are", "be", "by", "with", "as", "that", "this", "from", "it",
    "el", "la", "los", "las", "de", "del", "y", "o", "en", "para",
    "por", "que", "es", "se", "un", "una", "lo",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]


# ── Markdown chunking ────────────────────────────────────────────────────────


@dataclass
class Section:
    path: str            # relative to docs root
    heading: str         # full heading text (e.g. "Field OR_EAD_TIT")
    level: int           # 1 = H1, 2 = H2, …
    body: str            # markdown body, including child paragraphs
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "heading": self.heading,
            "level": self.level,
            "body": self.body,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


def _chunk_markdown(text: str, rel_path: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []

    # Implicit "preamble" section before the first heading
    cur_heading = "(preamble)"
    cur_level = 0
    cur_start = 0
    cur_lines: list[str] = []

    def _flush(end_idx: int) -> None:
        body = "\n".join(cur_lines).strip()
        if body or cur_heading != "(preamble)":
            sections.append(Section(
                path=rel_path,
                heading=cur_heading,
                level=cur_level,
                body=body,
                line_start=cur_start + 1,
                line_end=end_idx + 1,
            ))

    for i, ln in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", ln)
        if m:
            _flush(i - 1)
            cur_level = len(m.group(1))
            cur_heading = m.group(2).strip()
            cur_start = i
            cur_lines = []
        else:
            cur_lines.append(ln)
    _flush(len(lines) - 1)

    return [s for s in sections if s.body.strip() or s.heading != "(preamble)"]


# ── BM25 ─────────────────────────────────────────────────────────────────────


@dataclass
class _BM25:
    docs: list[list[str]] = field(default_factory=list)
    doc_freqs: list[Counter] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, tokenised: list[list[str]]) -> "_BM25":
        bm = cls()
        bm.docs = tokenised
        bm.doc_freqs = [Counter(d) for d in tokenised]
        bm.doc_lens = [len(d) for d in tokenised]
        n = max(len(tokenised), 1)
        bm.avgdl = sum(bm.doc_lens) / n
        df: Counter = Counter()
        for tokens in tokenised:
            for t in set(tokens):
                df[t] += 1
        bm.idf = {
            t: math.log((n - dfi + 0.5) / (dfi + 0.5) + 1.0)
            for t, dfi in df.items()
        }
        return bm

    def score(self, query_tokens: list[str], i: int) -> float:
        freqs = self.doc_freqs[i]
        dl = self.doc_lens[i]
        s = 0.0
        for q in query_tokens:
            if q not in freqs:
                continue
            tf = freqs[q]
            idf = self.idf.get(q, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1.0))
            s += idf * tf * (self.k1 + 1) / max(denom, 1e-12)
        return s


# ── Public API ───────────────────────────────────────────────────────────────


@dataclass
class SearchHit:
    section: Section
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.section.path,
            "heading": self.section.heading,
            "level": self.section.level,
            "line_start": self.section.line_start,
            "line_end": self.section.line_end,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }


class DocsIndex:
    def __init__(self, docs_root: Path | None = None) -> None:
        self.docs_root = (docs_root or _DEFAULT_DOCS_ROOT).resolve()
        self.sections: list[Section] = []
        self._tokens: list[list[str]] = []
        self._bm25: _BM25 | None = None

    # ── Build / persist ────────────────────────────────────────────────────

    def build(self) -> "DocsIndex":
        self.sections = []
        self._tokens = []
        if not self.docs_root.exists():
            self._bm25 = _BM25.build([])
            return self
        for f in sorted(self.docs_root.rglob("*.md")):
            rel = str(f.relative_to(self.docs_root))
            text = f.read_text(encoding="utf-8")
            for sec in _chunk_markdown(text, rel):
                self.sections.append(sec)
                self._tokens.append(_tokenize(sec.heading + " " + sec.body))
        self._bm25 = _BM25.build(self._tokens)
        return self

    def save(self, path: Path | None = None) -> Path:
        path = (path or _DEFAULT_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "docs_root": str(self.docs_root),
            "sections": [s.to_dict() for s in self.sections],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DocsIndex":
        path = path or _DEFAULT_INDEX_PATH
        if not path.exists():
            return cls().build()
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(docs_root=Path(data.get("docs_root", _DEFAULT_DOCS_ROOT)))
        idx.sections = [Section(**s) for s in data.get("sections", [])]
        idx._tokens = [
            _tokenize(s.heading + " " + s.body) for s in idx.sections
        ]
        idx._bm25 = _BM25.build(idx._tokens)
        return idx

    def reindex(self) -> "DocsIndex":
        self.build()
        self.save()
        return self

    # ── Query ──────────────────────────────────────────────────────────────

    def section_count(self) -> int:
        return len(self.sections)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if self._bm25 is None or not self.sections:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[int, float]] = []
        for i in range(len(self.sections)):
            s = self._bm25.score(q_tokens, i)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[SearchHit] = []
        for i, s in scored[:k]:
            sec = self.sections[i]
            out.append(SearchHit(
                section=sec,
                score=s,
                snippet=_snippet(sec.body, q_tokens, max_chars=240),
            ))
        return out

    def get_field_definition(self, field: str) -> dict[str, Any] | None:
        """Find the section that *defines* a field. Heuristic: prefer
        sections whose heading contains the field name verbatim, then BM25."""
        field_u = field.upper()
        # Heading match
        for sec in self.sections:
            if field_u in sec.heading.upper():
                return {
                    "found": True,
                    "path": sec.path,
                    "heading": sec.heading,
                    "line_start": sec.line_start,
                    "body": sec.body[:1500],
                }
        # Fall back to BM25 with the field as the only token
        hits = self.search(field, k=1)
        if hits:
            sec = hits[0].section
            return {
                "found": True,
                "path": sec.path,
                "heading": sec.heading,
                "line_start": sec.line_start,
                "body": sec.body[:1500],
                "score": hits[0].score,
            }
        return {"found": False, "field": field_u}


# ── Snippet helper ───────────────────────────────────────────────────────────


def _snippet(body: str, query_tokens: list[str], max_chars: int = 240) -> str:
    if not body:
        return ""
    body = re.sub(r"\s+", " ", body).strip()
    lower = body.lower()
    # Find the first window that hits the query
    pos = -1
    for q in query_tokens:
        i = lower.find(q)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos < 0:
        return body[:max_chars] + ("..." if len(body) > max_chars else "")
    start = max(0, pos - max_chars // 4)
    end = min(len(body), start + max_chars)
    pre = "..." if start > 0 else ""
    post = "..." if end < len(body) else ""
    return pre + body[start:end] + post


# ── Module-level cache ───────────────────────────────────────────────────────


_default: DocsIndex | None = None


def get_default_index(rebuild_if_empty: bool = True) -> DocsIndex:
    global _default
    if _default is None:
        _default = DocsIndex.load(_DEFAULT_INDEX_PATH)
        if rebuild_if_empty and _default.section_count() == 0:
            _default = DocsIndex().build()
    return _default


def reset_default_index() -> None:
    global _default
    _default = None
