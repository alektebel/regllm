#!/usr/bin/env python3
"""
Pipeline: EBA GL/2017/16 → KuzuDB graph + ChromaDB embeddings

Groups paragraphs by section, sends each section as one chunk to qwen3.6:27b
for entity extraction + KARMA verification, then embeds all chunks with nomic-embed-text.
"""
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline")

# Load articles
with open("data/regulation/eba_gl_2017_16_articles.json", "r") as f:
    articles = json.load(f)

# Group by section
sections = defaultdict(list)
for a in articles:
    sections[a["section"]].append(a)

logger.info("Loaded %d articles in %d sections", len(articles), len(sections))

# Initialize stores
from src.knowledge.graph_store import GraphStore
from src.knowledge.vector_store import VectorStore
from src.knowledge.embeddings import get_embedding_service
from src.knowledge.llm_client import LocalLLMClient
from src.knowledge.graph_builder import GraphBuilder

store = GraphStore("data/knowledge/regllm.kuzu")
vs = VectorStore("data/knowledge/vectors")
embed = get_embedding_service()

# Explicit high timeout for large regulation chunks
llm = LocalLLMClient(timeout=300.0)

logger.info("LLM backend: %s (model: %s)", llm.detect_backend(), llm.ollama_model)
logger.info("Embedding available: %s", embed.available)

builder = GraphBuilder(store, vs, embed, llm)

# Process each section
total_counts = {"nodes": 0, "edges": 0, "embedded": 0, "rejected": 0, "revised": 0}

for i, (section_name, section_articles) in enumerate(sorted(sections.items())):
    # Build section text from its articles
    section_text = f"# {section_name}\n\n"
    for a in section_articles:
        section_text += f"**Párrafo {a['paragraph']}.**  {a['text']}\n\n"

    source = f"EBA_GL_2017_16/{section_name.split()[0]}"

    logger.info(
        "[%d/%d] Section: %s (%d paragraphs, %d chars)",
        i + 1, len(sections), section_name[:60], len(section_articles), len(section_text),
    )

    t0 = time.time()
    try:
        # Only verify substantive regulatory sections (4+)
        section_num = section_name.split()[0] if section_name[0].isdigit() else "0"
        major = int(section_num.split(".")[0]) if section_num[0].isdigit() else 0
        verify = major >= 4

        result = builder.ingest_regulation(
            section_text,
            source=source,
            chunk_size=4000,
            verify=verify,
        )

        elapsed = time.time() - t0
        for k in total_counts:
            total_counts[k] += result.get(k, 0)

        logger.info(
            "  → nodes=%d edges=%d embedded=%d rejected=%d (%.1fs)",
            result.get("nodes", 0), result.get("edges", 0),
            result.get("embedded", 0), result.get("rejected", 0), elapsed,
        )
    except Exception as e:
        logger.error("  → FAILED: %s", e)
        import traceback
        traceback.print_exc()

# Final stats
logger.info("=" * 60)
logger.info("PIPELINE COMPLETE")
logger.info("Totals: %s", total_counts)
logger.info("Graph stats: %s", store.stats())

# Save summary
summary = {
    "source": "EBA/GL/2017/16 (PD & LGD estimation guidelines)",
    "language": "Spanish",
    "sections_processed": len(sections),
    "articles_processed": len(articles),
    "totals": total_counts,
    "graph_stats": store.stats(),
}
Path("data/knowledge").mkdir(parents=True, exist_ok=True)
with open("data/knowledge/pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

logger.info("Summary saved to data/knowledge/pipeline_summary.json")
