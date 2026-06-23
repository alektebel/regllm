"""Unit tests for src/embeddings/* and the /embeddings API router."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.agent.docs_index import DocsIndex
from src.embeddings.embedder import build_embeddings, to_payload
from src.embeddings.projector import PROJECTION_METHODS, project


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "fields").mkdir()
    (root / "fields" / "EAD.md").write_text(
        "# Field EAD\n\nExposure at default, used in ECL = PD * LGD * EAD.\n\n"
        "## Source\nmylib.ciclos_recuperacion column EAD.\n",
        encoding="utf-8",
    )
    (root / "fields" / "PD.md").write_text(
        "# Field PD\n\nProbability of default, calibrated quarterly.\n\n"
        "## Source\nmylib.scoring column PD.\n",
        encoding="utf-8",
    )
    (root / "fields" / "LGD.md").write_text(
        "# Field LGD\n\nLoss given default, floored at regulatory minimum.\n",
        encoding="utf-8",
    )
    return root


def test_build_embeddings_shape(tiny_corpus: Path) -> None:
    idx = DocsIndex(docs_root=tiny_corpus).build()
    emb = build_embeddings(idx)
    assert len(emb) == idx.section_count()
    assert emb.vectors.shape[0] == len(emb)
    payload = to_payload(emb)
    assert {"id", "path", "heading", "snippet", "level"} <= payload[0].keys()


def test_build_embeddings_empty_corpus(tmp_path: Path) -> None:
    idx = DocsIndex(docs_root=tmp_path / "missing").build()
    emb = build_embeddings(idx)
    assert len(emb) == 0


@pytest.mark.parametrize("method", PROJECTION_METHODS)
@pytest.mark.parametrize("n_components", [2, 3])
def test_project_shapes(tiny_corpus: Path, method: str, n_components: int) -> None:
    idx = DocsIndex(docs_root=tiny_corpus).build()
    emb = build_embeddings(idx)
    coords = project(emb.vectors, method=method, n_components=n_components)
    assert coords.shape == (len(emb), n_components)


def test_project_unknown_method() -> None:
    with pytest.raises(ValueError):
        project(np.zeros((5, 4)), method="not-a-method")


def test_project_handles_too_few_samples() -> None:
    coords = project(np.zeros((1, 4)), method="pca", n_components=2)
    assert coords.shape == (1, 2)


def test_embeddings_router_smoke(monkeypatch: pytest.MonkeyPatch, tiny_corpus: Path) -> None:
    import src.agent.docs_index as docs_index_mod
    import src.embeddings.embedder as embedder_mod

    docs_index_mod.reset_default_index()
    embedder_mod.reset_default_embeddings()
    monkeypatch.setattr(docs_index_mod, "_DEFAULT_DOCS_ROOT", tiny_corpus)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_EMBEDDINGS_PATH", tiny_corpus / "embeddings.json")

    def _get_index(rebuild_if_empty: bool = True) -> DocsIndex:
        return DocsIndex(docs_root=tiny_corpus).build()

    monkeypatch.setattr(docs_index_mod, "get_default_index", _get_index)

    from api.main import app

    client = TestClient(app)

    methods_resp = client.get("/embeddings/methods")
    assert methods_resp.status_code == 200
    assert "pca" in methods_resp.json()["methods"]

    sections_resp = client.get("/embeddings/sections", params={"rebuild": True})
    assert sections_resp.status_code == 200
    assert sections_resp.json()["count"] >= 3

    project_resp = client.post(
        "/embeddings/project",
        json={"method": "pca", "n_components": 3, "rebuild": True},
    )
    assert project_resp.status_code == 200
    body = project_resp.json()
    assert body["method"] == "pca"
    assert len(body["points"]) >= 3
    assert {"x", "y", "z", "heading", "path"} <= body["points"][0].keys()

    bad_resp = client.post("/embeddings/project", json={"method": "pca", "n_components": 5})
    assert bad_resp.status_code == 400
