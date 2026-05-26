"""
Ingest regulation documents via the running FastAPI /admin/ingest endpoint.
This avoids loading the embedding model a second time.

Usage:
    ADMIN_TOKEN=secret python3 scripts/ingest_via_api.py data/raw/regulation_data_*.json
    ADMIN_TOKEN=secret python3 scripts/ingest_via_api.py  # uses default paths

Set API_URL to override the default http://localhost:8000.
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
EMBED_BATCH = int(os.getenv("EMBED_BATCH_SIZE", "16"))

DEFAULT_FILES = [
    "data/raw/regulation_data_20260116_211936.json",
    "data/raw/regulation_data_20260117_000004.json",
]


def main():
    if not ADMIN_TOKEN:
        logger.error("ADMIN_TOKEN env var is required")
        sys.exit(1)

    files = sys.argv[1:] or DEFAULT_FILES
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {ADMIN_TOKEN}"

    seen_urls: set = set()
    docs = []
    for fn in files:
        p = Path(fn)
        if not p.exists():
            logger.warning("File not found: %s — skipping", fn)
            continue
        data = json.loads(p.read_text())
        for d in data:
            url = d.get("url", "")
            text = d.get("text", "")
            if url not in seen_urls and len(text) > 200:
                seen_urls.add(url)
                docs.append({
                    "texto": text,
                    "metadata": {
                        "documento_id": url or f"doc_{len(docs)}",
                        "source": d.get("title", "Unknown"),
                        "documento": d.get("title", "Unknown"),
                        "framework": d.get("source", ""),
                        "url": url,
                        "keywords": d.get("keywords", []),
                    },
                })

    logger.info("Loaded %d unique documents", len(docs))

    total = 0
    for i, doc in enumerate(docs, 1):
        name = doc["metadata"]["source"][:60]
        chars = len(doc["texto"])
        logger.info("[%d/%d] %s (%d chars)", i, len(docs), name, chars)

        resp = session.post(
            f"{API_URL}/admin/ingest",
            json={"documents": [doc], "embed_batch_size": EMBED_BATCH},
            timeout=600,
        )
        if resp.status_code != 200:
            logger.error("  FAILED: %s %s", resp.status_code, resp.text[:200])
            continue

        n = resp.json().get("chunks_upserted", 0)
        total += n
        logger.info("  -> %d chunks (total: %d)", n, total)

    logger.info("Done — %d chunks indexed", total)


if __name__ == "__main__":
    main()
