"""
Regulatory Document Ingestion Pipeline
=======================================

Downloads EU banking regulation documents from EUR-Lex (HTML) and EBA (PDF),
parses them, and indexes them into the pgvector `document_chunks` table.

Usage:
    python scripts/ingest_regulations.py              # ingest new/changed docs
    python scripts/ingest_regulations.py --force      # re-ingest all docs
    python scripts/ingest_regulations.py --dry-run    # show what would be fetched
    python scripts/ingest_regulations.py --source crr_575_2013  # single source

DB connection (env vars, same as API):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

To run locally against production DB:
    export $(cat .env | xargs)
    POSTGRES_HOST=<rds-endpoint> python scripts/ingest_regulations.py
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKSUMS_FILE = RAW_DIR / "checksums.json"

# ── Document sources ──────────────────────────────────────────────────────────
#
# type: "eurlex_html" → parse HTML from EUR-Lex (stable CELEX URLs)
#       "pdf"         → download PDF and extract text with pdfplumber

SOURCES: list[dict] = [
    # ── Core capital regulations ──────────────────────────────────────────────
    {
        "id": "crr_575_2013",
        "name": "CRR — Reglamento (UE) 575/2013 (requisitos de capital)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32013R0575",
        "type": "eurlex_html",
        "framework": "CRR",
        "tags": ["capital", "riesgo_credito", "riesgo_mercado", "riesgo_operacional", "liquidez", "crr"],
    },
    {
        "id": "crr3_2024_1623",
        "name": "CRR3 / Basilea IV — Reglamento (UE) 2024/1623",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32024R1623",
        "type": "eurlex_html",
        "framework": "CRR3",
        "tags": ["basilea_iv", "output_floor", "sa_cr", "irb", "op_risk_sma", "capital", "crr3"],
    },
    {
        "id": "crd4_2013_36",
        "name": "CRD IV — Directiva 2013/36/UE (gobierno, ICAAP, supervisión)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32013L0036",
        "type": "eurlex_html",
        "framework": "CRD",
        "tags": ["crd", "gobierno", "supervision", "icaap", "srep", "remuneracion", "colchones"],
    },
    # ── Resolution framework ──────────────────────────────────────────────────
    {
        "id": "brrd_2014_59",
        "name": "BRRD — Directiva 2014/59/UE (resolución bancaria, MREL, bail-in)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014L0059",
        "type": "eurlex_html",
        "framework": "BRRD",
        "tags": ["resolucion", "bail_in", "mrel", "tlac", "brrd", "planes_recuperacion"],
    },
    {
        "id": "srmr_806_2014",
        "name": "SRMR — Reglamento (UE) 806/2014 (mecanismo único de resolución)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014R0806",
        "type": "eurlex_html",
        "framework": "SRM",
        "tags": ["resolucion", "jur", "mrel", "srm", "bail_in"],
    },
    # ── EBA Guidelines (PDFs) ─────────────────────────────────────────────────
    {
        "id": "eba_gl_2020_06",
        "name": "EBA/GL/2020/06 — Directrices ICAAP e ILAAP",
        "url": "https://www.eba.europa.eu/sites/default/documents/files/document_library/Publications/Guidelines/2020/EBA-GL-2020-06/872842/EBA%20GL%202020%2006%20Final%20Report%20on%20GL%20on%20ICAAP%20and%20ILAAP.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["icaap", "ilaap", "capital_interno", "pilar2", "srep"],
    },
    {
        "id": "eba_gl_2022_12",
        "name": "EBA/GL/2022/12 — Directrices IRRBB",
        "url": "https://www.eba.europa.eu/sites/default/documents/files/document_library/Publications/Guidelines/2022/EBA-GL-2022-12/1051744/Final%20Report%20on%20Guidelines%20on%20IRRBB%20and%20CSRBB.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["irrbb", "csrbb", "riesgo_tipo_interes", "pilar2"],
    },
    {
        "id": "eba_gl_2018_10",
        "name": "EBA/GL/2018/10 — Directrices NPE (créditos dudosos)",
        "url": "https://www.eba.europa.eu/sites/default/documents/files/documents/10180/2425705/b4bcf67f-5b41-4e4e-8d91-4b18e7c2a900/EBA_GL_2018_10_Final_Guidelines_on_NPEs.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["npe", "npl", "creditos_dudosos", "stage3", "ifrs9"],
    },
    {
        "id": "eba_gl_2020_07",
        "name": "EBA/GL/2020/07 — Directrices LCR (coeficiente de cobertura de liquidez)",
        "url": "https://www.eba.europa.eu/sites/default/documents/files/document_library/Publications/Guidelines/2020/EBA-GL-2020-07/873413/EBA%20GL%202020%2007%20Final%20report%20on%20LCR%20disclosure.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["lcr", "liquidez", "hqla", "divulgacion"],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "regllm-ingest/1.0 (research; contact: admin@regllm.xyz)"
})


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_checksums() -> dict:
    if CHECKSUMS_FILE.exists():
        return json.loads(CHECKSUMS_FILE.read_text())
    return {}


def _save_checksums(checksums: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_FILE.write_text(json.dumps(checksums, indent=2))


def _download(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{retries} for {url}: {e}")
            time.sleep(2 ** attempt)


def _parse_eurlex_html(content: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "lxml")

    # Remove navigation, scripts, styles, headers, footers
    for tag in soup.find_all(["script", "style", "nav", "header", "footer",
                               "noscript", "aside", "form"]):
        tag.decompose()

    # EUR-Lex puts the legal text in specific containers; try them in order
    main = (
        soup.find("div", id="document1")
        or soup.find("div", class_="tabContent")
        or soup.find("div", id="TexteOnly")
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )

    if not main:
        return soup.get_text(separator="\n")

    # Extract text, collapsing whitespace
    lines = []
    for elem in main.stripped_strings:
        line = elem.strip()
        if line:
            lines.append(line)

    text = "\n".join(lines)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(content: bytes) -> str:
    import io
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text:
                text_parts.append(page_text)

    text = "\n\n".join(text_parts)
    # Clean up common PDF artifacts
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # dehyphenate
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_source(source: dict) -> str | None:
    """Download and parse a source. Returns plain text or None on failure."""
    url = source["url"]
    logger.info(f"Fetching: {source['name']}")
    logger.info(f"  URL: {url}")

    try:
        content = _download(url)
    except Exception as e:
        logger.error(f"  FAILED to download: {e}")
        return None

    try:
        if source["type"] == "eurlex_html":
            text = _parse_eurlex_html(content)
        elif source["type"] == "pdf":
            text = _parse_pdf(content)
        else:
            logger.error(f"  Unknown type: {source['type']}")
            return None
    except Exception as e:
        logger.error(f"  FAILED to parse: {e}")
        return None

    if len(text) < 500:
        logger.warning(f"  Very short text ({len(text)} chars) — skipping")
        return None

    logger.info(f"  Parsed {len(text):,} chars")
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest regulatory documents into pgvector")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if checksum unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Download only, do not write to DB")
    parser.add_argument("--source", help="Only ingest a single source by ID")
    args = parser.parse_args()

    sources = SOURCES
    if args.source:
        sources = [s for s in SOURCES if s["id"] == args.source]
        if not sources:
            logger.error(f"Unknown source ID: {args.source}")
            logger.info(f"Available: {[s['id'] for s in SOURCES]}")
            sys.exit(1)

    checksums = _load_checksums()
    documents = []
    skipped = 0

    for source in sources:
        text = fetch_source(source)
        if text is None:
            continue

        checksum = _checksum(text)
        if not args.force and checksums.get(source["id"]) == checksum:
            logger.info(f"  No change — skipping {source['id']}")
            skipped += 1
            continue

        checksums[source["id"]] = checksum

        documents.append({
            "text": text,
            "metadata": {
                "documento_id": source["id"],
                "source": source["name"],
                "framework": source["framework"],
                "tags": source["tags"],
                "url": source["url"],
            },
        })

    logger.info(f"\nSummary: {len(documents)} to ingest, {skipped} unchanged")

    if not documents:
        logger.info("Nothing to do.")
        return

    if args.dry_run:
        logger.info("Dry run — skipping DB write")
        for doc in documents:
            logger.info(f"  Would ingest: {doc['metadata']['source']} ({len(doc['text']):,} chars)")
        return

    logger.info("Loading RAG system and embedding model…")
    from src.rag_system import RegulatoryRAGSystem
    rag = RegulatoryRAGSystem()

    logger.info(f"Indexing {len(documents)} documents…")
    total_chunks = rag.procesar_documentos(documents)
    logger.info(f"Done — {total_chunks} chunks indexed")

    _save_checksums(checksums)
    logger.info(f"Checksums saved to {CHECKSUMS_FILE}")


if __name__ == "__main__":
    main()
