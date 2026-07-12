#!/usr/bin/env python3
"""Build the embedding-based regulation RAG index over the EBA GL/2017/16
PD & LGD guidelines.

Chunks ``data/regulation/eba_gl_2017_16_articles.json`` (see
``src/knowledge/regulation_chunker.py``), embeds every chunk with the
configured backend (``src/knowledge/embeddings.py`` — Ollama / standalone
GGUF / Bedrock, selected via ``REGLLM_EMBED_BACKEND``), and writes
``data/regulation/embeddings/pd_lgd_chunks.json`` — the index
``src.knowledge.regulation_vector_store`` loads at request time.

The output directory is gitignored (a generated artifact, like the DQC
eval harness's `.db` files) — rebuild it after chunker changes or after
switching the embedding backend/model (a manifest mismatch is the sign you
need to). Run:

    python scripts/build_regulation_embeddings.py
    python scripts/build_regulation_embeddings.py --max-chars 800 --overlap 100
    python scripts/build_regulation_embeddings.py --out data/regulation/embeddings/custom.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.knowledge.embeddings import get_embedding_service  # noqa: E402
from src.knowledge.regulation_chunker import (  # noqa: E402
    DEFAULT_SOURCE, chunk_eba_guidelines,
)
from src.knowledge.regulation_vector_store import (  # noqa: E402
    DEFAULT_INDEX_PATH, RegulationVectorStore,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="ingested EBA GL paragraphs JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_INDEX_PATH)
    ap.add_argument("--max-chars", type=int, default=1000,
                    help="split paragraphs longer than this")
    ap.add_argument("--overlap", type=int, default=150,
                    help="chars of trailing context carried into the next split")
    args = ap.parse_args()

    print(f"Chunking {args.source} …")
    chunks = chunk_eba_guidelines(args.source, max_chars=args.max_chars, overlap=args.overlap)
    n_split = len({c.paragraph for c in chunks if c.n_parts > 1})
    print(f"  {len(chunks)} chunks from {len({c.paragraph for c in chunks})} paragraphs "
          f"({n_split} split into multiple chunks)")

    embedder = get_embedding_service()
    backend = embedder.resolved_backend
    print(f"Embedding with backend={backend!r} model={embedder.model!r} …")
    if backend == "stub":
        print(
            "  WARNING: no embedding backend is reachable/configured — this "
            "will produce an all-zero-vector index with NO real ranking "
            "signal. Set REGLLM_EMBED_BACKEND / OLLAMA_URL / "
            "GGUF_EMBED_MODEL_PATH / BEDROCK_REGION before building for "
            "production use.",
            file=sys.stderr,
        )

    store = RegulationVectorStore.build(chunks, embedder)
    path = store.save(args.out)
    print(f"Wrote {store.size} embedded chunks (dim={store.manifest['dim']}) → {path}")


if __name__ == "__main__":
    main()
