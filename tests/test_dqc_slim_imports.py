"""Guards the DQC production image's dependency footprint.

The root Dockerfile installs only ``requirements-dqc.txt`` (no torch, kuzu,
chromadb, scikit-learn, umap-learn, llama-cpp-python) because
``REGLLM_ROUTERS=dqc`` never reaches the code paths that need them —
``llama-cpp-python`` (the standalone GGUF backend) is opt-in, lazily
imported, and only touched when ``REGLLM_LLM=gguf``/``REGLLM_EMBED_BACKEND=
gguf`` is explicitly configured with a real weight file. These tests run in
a **subprocess** so ``sys.modules`` reflects only what the import under test
actually pulled in — checking in-process would give false negatives once
another test in the same session has already imported the heavy modules.

If one of these starts failing, something re-introduced an eager import of a
module requiring torch/kuzu/chromadb/llama_cpp into the DQC request path,
and the slim Docker image (built from ``requirements-dqc.txt``) will fail at
runtime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HEAVY = ("torch", "kuzu", "chromadb", "sklearn", "umap", "llama_cpp")
_HEAVY_CHECK = (
    f"heavy = [m for m in {_HEAVY!r} if m in sys.modules]; "
    "assert not heavy, f'unexpected heavy import: {heavy}'"
)


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


def test_knowledge_package_does_not_require_kuzu():
    _run(f"import sys; import src.knowledge; {_HEAVY_CHECK}")


def test_graph_store_not_reexported_from_package_root():
    """GraphStore must stay a submodule-only import (needs kuzu); every real
    caller already does ``from src.knowledge.graph_store import GraphStore``."""
    import src.knowledge as knowledge
    assert "GraphStore" not in knowledge.__all__
    assert not hasattr(knowledge, "GraphStore")


def test_dqc_only_api_boots_without_heavy_deps():
    """The exact import graph REGLLM_ROUTERS=dqc exercises at process start
    and on a live request — including the new regulation-RAG tool chain,
    chunker, and vector store — must not touch torch/kuzu/chromadb/sklearn/
    umap/llama_cpp."""
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
        f"{_HEAVY_CHECK}"
    )


def test_regulation_chunker_and_vector_store_import_cleanly():
    """The chunking + embedding-index modules behind search_regulation_semantic
    must themselves stay dependency-free (no numpy/chromadb) — verified by
    actually exercising build+search, not just importing."""
    _run(
        "import sys\n"
        "from src.knowledge.regulation_chunker import chunk_eba_guidelines\n"
        "from src.knowledge.regulation_vector_store import RegulationVectorStore\n"
        "chunks = chunk_eba_guidelines()\n"
        "assert len(chunks) > 200\n"
        "class _E:\n"
        "    model = 'fake'\n"
        "    resolved_backend = 'stub'\n"
        "    def embed(self, texts): return [[1.0, 0.0] for _ in texts]\n"
        "    def embed_one(self, t): return [1.0, 0.0]\n"
        "store = RegulationVectorStore.build(chunks[:5], _E())\n"
        "hits = store.search_text('pd', _E(), k=2)\n"
        "assert len(hits) == 2\n"
        f"{_HEAVY_CHECK}"
    )


def test_requirements_dqc_has_no_heavy_packages():
    lines = (PROJECT_ROOT / "requirements-dqc.txt").read_text().lower().splitlines()
    deps = "\n".join(ln for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    for pkg in ("torch", "kuzu", "chromadb", "scikit-learn", "umap-learn", "pandas",
                "llama-cpp-python"):
        assert pkg not in deps, f"{pkg} leaked into requirements-dqc.txt"


def test_dockerfile_installs_slim_requirements():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "requirements-dqc.txt" in dockerfile
    assert "requirements.txt" not in dockerfile.replace("requirements-dqc.txt", "")
    assert "pip install torch" not in dockerfile
