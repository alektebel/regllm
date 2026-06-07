"""Unit tests for src/agent/docs_index.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.docs_index import DocsIndex


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "fields").mkdir()
    (root / "fields" / "OR_EAD_TIT.md").write_text(
        "# Field OR_EAD_TIT\n\nThe titulised exposure at default.\n\n"
        "## Multiplier\nFor CORP segment use 2.0.\n",
        encoding="utf-8",
    )
    (root / "fields" / "EAD.md").write_text(
        "# Field EAD\n\nExposure at default, used in ECL = PD * LGD * EAD.\n",
        encoding="utf-8",
    )
    (root / "fluxes").mkdir()
    (root / "fluxes" / "lgd_pipeline.md").write_text(
        "# LGD pipeline\n\nThree-stage SAS calibration.\n\n"
        "## V3 changes\n- Floor raised\n- New OR_EAD_TIT step\n",
        encoding="utf-8",
    )
    return root


def test_build_chunks_by_heading(tiny_corpus: Path) -> None:
    idx = DocsIndex(docs_root=tiny_corpus).build()
    headings = sorted(s.heading for s in idx.sections)
    assert "Field OR_EAD_TIT" in headings
    assert "Field EAD" in headings
    assert "LGD pipeline" in headings
    assert "V3 changes" in headings
    assert "Multiplier" in headings


def test_search_finds_field(tiny_corpus: Path) -> None:
    idx = DocsIndex(docs_root=tiny_corpus).build()
    hits = idx.search("OR_EAD_TIT", k=3)
    assert hits, "expected at least one hit for OR_EAD_TIT"
    assert any("OR_EAD_TIT" in h.section.heading for h in hits)


def test_get_field_definition_prefers_heading_match(tiny_corpus: Path) -> None:
    idx = DocsIndex(docs_root=tiny_corpus).build()
    res = idx.get_field_definition("EAD")
    assert res and res["found"]
    assert "EAD" in res["heading"]


def test_save_and_load_round_trip(tiny_corpus: Path, tmp_path: Path) -> None:
    idx = DocsIndex(docs_root=tiny_corpus).build()
    out = tmp_path / "idx.json"
    idx.save(out)
    assert out.exists()
    loaded = DocsIndex.load(out)
    assert loaded.section_count() == idx.section_count()
    # Search works after load
    hits = loaded.search("multiplier", k=1)
    assert hits


def test_empty_corpus_returns_no_hits(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    idx = DocsIndex(docs_root=empty).build()
    assert idx.section_count() == 0
    assert idx.search("anything", k=3) == []
