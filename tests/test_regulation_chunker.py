"""Tests for src/knowledge/regulation_chunker.py."""

from __future__ import annotations

import json

import pytest

from src.knowledge.regulation_chunker import (
    DEFAULT_SOURCE, RegChunk, chunk_eba_guidelines,
)


@pytest.fixture()
def fixture_path(tmp_path):
    records = [
        {"id": "par_1", "paragraph": 1, "section": "1 Intro", "subsection": "",
         "text": "Este es un párrafo corto sobre PD."},
        {"id": "par_2", "paragraph": 2, "section": "2 LGD", "subsection": "2.1 Suelos",
         "text": "Frase uno sobre LGD. Frase dos sobre suelos. Frase tres con más detalle."},
        {"id": "par_3", "paragraph": 3, "section": "2 LGD", "subsection": "",
         "text": ""},  # empty text — must be skipped
    ]
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return p


def test_short_paragraph_becomes_one_chunk(fixture_path):
    chunks = chunk_eba_guidelines(fixture_path, max_chars=1000, overlap=50)
    par1 = [c for c in chunks if c.paragraph == 1]
    assert len(par1) == 1
    c = par1[0]
    assert c.chunk_id == "par_1#0"
    assert c.part == 0 and c.n_parts == 1
    assert c.section == "1 Intro"
    assert c.text == "Este es un párrafo corto sobre PD."


def test_empty_paragraph_is_skipped(fixture_path):
    chunks = chunk_eba_guidelines(fixture_path)
    assert all(c.paragraph != 3 for c in chunks)


def test_citation_format(fixture_path):
    chunks = chunk_eba_guidelines(fixture_path)
    c = next(c for c in chunks if c.paragraph == 1)
    assert c.citation() == "EBA GL 2017/16 §1"


def test_long_paragraph_splits_on_sentence_boundaries(fixture_path):
    # force splitting with a tiny max_chars so paragraph 2's 3 sentences
    # cannot fit in one window
    chunks = chunk_eba_guidelines(fixture_path, max_chars=30, overlap=10)
    par2 = sorted((c for c in chunks if c.paragraph == 2), key=lambda c: c.part)
    assert len(par2) > 1
    for c in par2:
        assert c.n_parts == len(par2)
        assert c.chunk_id == f"par_2#{c.part}"
        # never cut mid-sentence: every chunk ends at a sentence terminator
        # (or is the last chunk, which may end at the source's own end)
        assert c.text.rstrip().endswith((".", ":", ";")) or c is par2[-1]
    # concatenating parts (de-duplicating the overlap) should still contain
    # every original sentence
    all_text = " ".join(c.text for c in par2)
    assert "Frase uno" in all_text
    assert "Frase dos" in all_text
    assert "Frase tres" in all_text


def test_overlap_carries_context_into_next_window(fixture_path):
    chunks = chunk_eba_guidelines(fixture_path, max_chars=30, overlap=25)
    par2 = sorted((c for c in chunks if c.paragraph == 2), key=lambda c: c.part)
    assert len(par2) >= 2
    # the tail of window 0 should reappear at the head of window 1
    tail_sentence = par2[0].text.split(". ")[-1]
    assert tail_sentence.strip(".") in par2[1].text or par2[1].text.startswith(
        tail_sentence
    )


def test_chunk_ids_are_unique(fixture_path):
    chunks = chunk_eba_guidelines(fixture_path, max_chars=30, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_to_dict_roundtrips_fields(fixture_path):
    c = chunk_eba_guidelines(fixture_path)[0]
    d = c.to_dict()
    assert d["chunk_id"] == c.chunk_id
    assert d["paragraph"] == c.paragraph
    assert d["source_doc"] == "EBA GL 2017/16"


# ── sanity check against the real, committed corpus ─────────────────────────

def test_real_corpus_chunks_cleanly():
    assert DEFAULT_SOURCE.exists(), "eba_gl_2017_16_articles.json missing"
    chunks = chunk_eba_guidelines()
    assert len(chunks) >= 221, "at least one chunk per paragraph"
    assert all(len(c.text) <= 1000 for c in chunks)
    assert all(c.paragraph >= 1 for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    # every paragraph number in the source appears in the chunks
    import json as _json
    source = _json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))
    source_paragraphs = {r["paragraph"] for r in source if (r.get("text") or "").strip()}
    chunk_paragraphs = {c.paragraph for c in chunks}
    assert source_paragraphs == chunk_paragraphs
