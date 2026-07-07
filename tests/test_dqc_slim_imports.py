"""Guards the DQC production image's dependency footprint.

The root Dockerfile installs only ``requirements-dqc.txt`` (no torch, kuzu,
chromadb, scikit-learn, umap-learn) because ``REGLLM_ROUTERS=dqc`` never
reaches the code paths that need them. These tests run in a **subprocess**
so ``sys.modules`` reflects only what the import under test actually pulled
in — checking in-process would give false negatives once another test in the
same session has already imported the heavy modules.

If one of these starts failing, something re-introduced an eager import of a
module requiring torch/kuzu/chromadb into the DQC request path, and the slim
Docker image (built from ``requirements-dqc.txt``) will fail at runtime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HEAVY = ("torch", "kuzu", "chromadb", "sklearn", "umap")


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


def test_knowledge_package_does_not_require_kuzu():
    _run(
        "import sys; import src.knowledge; "
        "heavy = [m for m in ('torch','kuzu','chromadb','sklearn','umap') "
        "if m in sys.modules]; "
        "assert not heavy, f'unexpected heavy import: {heavy}'"
    )


def test_graph_store_not_reexported_from_package_root():
    """GraphStore must stay a submodule-only import (needs kuzu); every real
    caller already does ``from src.knowledge.graph_store import GraphStore``."""
    import src.knowledge as knowledge
    assert "GraphStore" not in knowledge.__all__
    assert not hasattr(knowledge, "GraphStore")


def test_dqc_only_api_boots_without_heavy_deps():
    """The exact import graph REGLLM_ROUTERS=dqc exercises at process start
    and on a live request must not touch torch/kuzu/chromadb/sklearn/umap."""
    _run(
        "import os, sys; "
        "os.environ['REGLLM_ROUTERS'] = 'dqc'; "
        "os.environ['REGLLM_LLM'] = 'stub'; "
        "from fastapi.testclient import TestClient; "
        "from api.main import app; "
        "client = TestClient(app); "
        "r = client.get('/health'); assert r.status_code == 200, r.text; "
        "r = client.post('/dqc/generate', json={'message': 'Genera DQCs para PD_ESTIMADA'}); "
        "assert r.status_code == 200, r.text; "
        "heavy = [m for m in ('torch','kuzu','chromadb','sklearn','umap') "
        "if m in sys.modules]; "
        "assert not heavy, f'unexpected heavy import after live request: {heavy}'"
    )


def test_requirements_dqc_has_no_heavy_packages():
    lines = (PROJECT_ROOT / "requirements-dqc.txt").read_text().lower().splitlines()
    deps = "\n".join(ln for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    for pkg in ("torch", "kuzu", "chromadb", "scikit-learn", "umap-learn", "pandas"):
        assert pkg not in deps, f"{pkg} leaked into requirements-dqc.txt"


def test_dockerfile_installs_slim_requirements():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "requirements-dqc.txt" in dockerfile
    assert "requirements.txt" not in dockerfile.replace("requirements-dqc.txt", "")
    assert "pip install torch" not in dockerfile
