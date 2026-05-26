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

PROJECT_ROOT = Path(__file__).parent.parent.parent
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
    # ── EUR-Lex: Core capital regulations ─────────────────────────────────────
    {
        "id": "crr_575_2013",
        "name": "CRR — Reglamento (UE) 575/2013",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32013R0575",
        "type": "eurlex_html",
        "framework": "CRR",
        "tags": ["capital", "riesgo_credito", "riesgo_mercado", "riesgo_operacional", "liquidez", "irb", "sa"],
    },
    {
        "id": "crr2_2019_876",
        "name": "CRR2 — Reglamento (UE) 2019/876 (NSFR, SA-CCR, FRTB, grandes exposiciones)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32019R0876",
        "type": "eurlex_html",
        "framework": "CRR2",
        "tags": ["nsfr", "sa_ccr", "frtb", "grandes_exposiciones", "capital"],
    },
    {
        "id": "crr3_2024_1623",
        "name": "CRR3 / Basilea IV — Reglamento (UE) 2024/1623",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32024R1623",
        "type": "eurlex_html",
        "framework": "CRR3",
        "tags": ["basilea_iv", "output_floor", "sa_cr", "irb", "op_risk_sma", "capital"],
    },
    # ── EUR-Lex: CRD directives ───────────────────────────────────────────────
    {
        "id": "crd4_2013_36",
        "name": "CRD IV — Directiva 2013/36/UE (gobierno, ICAAP, SREP)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32013L0036",
        "type": "eurlex_html",
        "framework": "CRD4",
        "tags": ["crd", "gobierno", "supervision", "icaap", "srep", "remuneracion", "colchones"],
    },
    {
        "id": "crd5_2019_878",
        "name": "CRD V — Directiva 2019/878/UE",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32019L0878",
        "type": "eurlex_html",
        "framework": "CRD5",
        "tags": ["crd", "pilar2", "colchones", "gobierno"],
    },
    {
        "id": "crd6_2024_1619",
        "name": "CRD VI — Directiva 2024/1619/UE",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32024L1619",
        "type": "eurlex_html",
        "framework": "CRD6",
        "tags": ["crd", "supervision", "gobierno", "capital"],
    },
    # ── EUR-Lex: Resolution framework ─────────────────────────────────────────
    {
        "id": "brrd_2014_59",
        "name": "BRRD — Directiva 2014/59/UE (resolución bancaria, MREL, bail-in)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014L0059",
        "type": "eurlex_html",
        "framework": "BRRD",
        "tags": ["resolucion", "bail_in", "mrel", "planes_recuperacion"],
    },
    {
        "id": "brrd2_2019_879",
        "name": "BRRD2 — Directiva 2019/879/UE (MREL actualización, TLAC)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32019L0879",
        "type": "eurlex_html",
        "framework": "BRRD2",
        "tags": ["mrel", "tlac", "resolucion", "bail_in"],
    },
    {
        "id": "srmr_806_2014",
        "name": "SRMR — Reglamento (UE) 806/2014 (mecanismo único de resolución)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014R0806",
        "type": "eurlex_html",
        "framework": "SRM",
        "tags": ["resolucion", "jur", "mrel", "srm"],
    },
    {
        "id": "srmr2_2019_877",
        "name": "SRMR2 — Reglamento (UE) 2019/877",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32019R0877",
        "type": "eurlex_html",
        "framework": "SRM2",
        "tags": ["mrel", "tlac", "resolucion"],
    },
    # ── EUR-Lex: Deposit guarantee & derivatives ──────────────────────────────
    {
        "id": "dgsd_2014_49",
        "name": "DGSD — Directiva 2014/49/UE (garantía de depósitos)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014L0049",
        "type": "eurlex_html",
        "framework": "DGSD",
        "tags": ["garantia_depositos", "dgsd", "fondo_garantia"],
    },
    {
        "id": "emir_648_2012",
        "name": "EMIR — Reglamento (UE) 648/2012 (derivados OTC, CCPs)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32012R0648",
        "type": "eurlex_html",
        "framework": "EMIR",
        "tags": ["derivados_otc", "ccp", "contraparte", "margen", "riesgo_contraparte"],
    },
    # ── EUR-Lex: Markets ──────────────────────────────────────────────────────
    {
        "id": "mifir_600_2014",
        "name": "MiFIR — Reglamento (UE) 600/2014 (transparencia de mercados)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014R0600",
        "type": "eurlex_html",
        "framework": "MiFIR",
        "tags": ["mercados", "transparencia", "pre_trade", "post_trade", "derivados"],
    },
    {
        "id": "mifid2_2014_65",
        "name": "MiFID II — Directiva 2014/65/UE (mercados de instrumentos financieros)",
        "url": "https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014L0065",
        "type": "eurlex_html",
        "framework": "MiFID2",
        "tags": ["mercados", "gobierno", "mejor_ejecucion", "idoneidad"],
    },
    # ── Local PDFs ────────────────────────────────────────────────────────────
    {
        "id": "ecb_ssm_guide_2025",
        "name": "ECB SSM Supervisory Guide 2025",
        "url": "local:data/raw/ssm.supervisory_guide202507.en.pdf",
        "type": "local_pdf",
        "framework": "ECB_SSM",
        "tags": ["supervision", "srep", "icaap", "ilaap", "ssm"],
    },
    {
        "id": "eba_gl_2016_07",
        "name": "EBA/GL/2016/07 — Directrices sobre definición de default (IRB)",
        "url": "local:data/raw/pdf/EBA-GL-2016-07-ES.pdf",
        "type": "local_pdf",
        "framework": "EBA_GL",
        "tags": ["irb", "default", "definicion_impago", "riesgo_credito"],
    },
    {
        "id": "eba_report_irb_modelling",
        "name": "EBA Report on IRB Modelling Practices",
        "url": "local:data/raw/pdf/EBA Report on IRB modelling practices.pdf",
        "type": "local_pdf",
        "framework": "EBA_GL",
        "tags": ["irb", "modelos", "pd", "lgd", "ead", "validacion"],
    },
    {
        "id": "eba_irb_survey_instructions",
        "name": "EBA Instructions for Qualitative Survey on IRB Models (Dec 2016)",
        "url": "local:data/raw/pdf/Instructions for the EBA qualitative survey on IRB models (Dec 2016).pdf",
        "type": "local_pdf",
        "framework": "EBA_GL",
        "tags": ["irb", "modelos", "survey", "validacion"],
    },
    {
        "id": "crr_delegated_529_2014",
        "name": "Reglamento Delegado (UE) 529/2014 — Umbrales de materialidad para enfoques internos (CRR)",
        "url": "local:data/raw/CELEX_32014R0529_ES_TXT.pdf",
        "type": "local_pdf",
        "framework": "CRR",
        "tags": ["irb", "materialidad", "crr", "enfoques_internos"],
    },
    {
        "id": "eba_rts_irb_assessment",
        "name": "EBA Final Draft RTS on Assessment Methodology for IRB",
        "url": "local:data/raw/Final Draft RTS on Assessment Methodology for IRB.pdf",
        "type": "local_pdf",
        "framework": "EBA_GL",
        "tags": ["irb", "assessment_methodology", "validacion", "rts"],
    },
    {
        "id": "eba_irb_validation_handbook",
        "name": "EBA Supervisory Handbook on Validation of IRB Rating Systems",
        "url": "local:data/raw/Supervisory handbook on the validation of IRB rating systems revised.pdf",
        "type": "local_pdf",
        "framework": "EBA_GL",
        "tags": ["irb", "validacion", "rating_systems", "supervision"],
    },
    # ── EBA Guidelines (PDFs) ─────────────────────────────────────────────────
    {
        "id": "eba_gl_2017_16",
        "name": "EBA/GL/2017/16 — Estimación de PD, LGD y tratamiento de exposiciones en default (IRB)",
        "url": "https://www.eba.europa.eu/sites/default/files/documents/10180/2033363/6b062012-45d6-4655-af04-801d26493ed0/Guidelines%20on%20PD%20and%20LGD%20estimation%20%28EBA-GL-2017-16%29.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["irb", "pd", "lgd", "ead", "riesgo_credito", "default"],
    },
    {
        "id": "eba_gl_2020_06",
        "name": "EBA/GL/2020/06 — Directrices sobre originación y seguimiento de préstamos",
        "url": "https://www.eba.europa.eu/sites/default/files/document_library/Publications/Guidelines/2020/Guidelines%20on%20loan%20origination%20and%20monitoring/884283/EBA%20GL%202020%2006%20Final%20Report%20on%20GL%20on%20loan%20origination%20and%20monitoring.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["originacion", "credito", "gobierno", "underwriting"],
    },
    {
        "id": "eba_gl_2022_14",
        "name": "EBA/GL/2022/14 — Directrices IRRBB y CSRBB",
        "url": "https://www.eba.europa.eu/sites/default/files/document_library/Publications/Guidelines/2022/EBA-GL-2022-14%20GL%20on%20IRRBB%20and%20CSRBB/1041754/Guidelines%20on%20IRRBB%20and%20CSRBB.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["irrbb", "csrbb", "riesgo_tipo_interes", "pilar2"],
    },
    {
        "id": "eba_gl_2018_10",
        "name": "EBA/GL/2018/10 — Directrices sobre créditos dudosos (NPE/NPL)",
        "url": "https://www.eba.europa.eu/sites/default/files/documents/10180/2425705/371ff4ba-d7db-4fa9-a3c7-231cb9c2a26a/Final%20Guidelines%20on%20management%20of%20non-performing%20and%20forborne%20exposures.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["npe", "npl", "creditos_dudosos", "stage3", "ifrs9"],
    },
    {
        "id": "eba_gl_2021_07",
        "name": "EBA/GL/2021/07 — Directrices sobre criterios para datos de entrada del IMA (FRTB)",
        "url": "https://www.eba.europa.eu/sites/default/files/document_library/Publications/Guidelines/2021/1017268/Guidelines%20on%20use%20of%20data%20inputs%20in%20the%20IMA.pdf",
        "type": "pdf",
        "framework": "EBA_GL",
        "tags": ["frtb", "ima", "riesgo_mercado", "datos_entrada"],
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
    del soup  # free lxml DOM (C extension memory) as early as possible

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(content: bytes) -> str:
    import io
    import pdfplumber

    buf = io.StringIO()
    first = True
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text:
                if not first:
                    buf.write("\n\n")
                buf.write(page_text)
                first = False

    text = buf.getvalue()
    buf.close()
    # Clean up common PDF artifacts
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # dehyphenate
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_source(source: dict) -> str | None:
    """Download and parse a source. Returns plain text or None on failure."""
    url = source["url"]
    logger.info(f"Fetching: {source['name']}")
    logger.info(f"  URL: {url}")

    if source["type"] == "local_pdf":
        local_path = PROJECT_ROOT / url.removeprefix("local:")
        if not local_path.exists():
            logger.error(f"  Local file not found: {local_path}")
            return None
        try:
            text = _parse_pdf(local_path.read_bytes())
        except Exception as e:
            logger.error(f"  FAILED to parse local PDF: {e}")
            return None
        if len(text) < 500:
            logger.warning(f"  Very short text ({len(text)} chars) — skipping")
            return None
        logger.info(f"  Parsed {len(text):,} chars from local file")
        return text

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
    parser.add_argument(
        "--embed-batch-size", type=int, default=8,
        help="Chunks per embedding batch (lower = less RAM, default: 8)",
    )
    parser.add_argument(
        "--from-file", metavar="PATH",
        help="Ingest from a local JSON snapshot (list of {text, source, url, ...}) instead of fetching",
    )
    parser.add_argument(
        "--llm-chunk", action="store_true",
        help="Use local Ollama model to chunk articles (cleaner, one chunk per article, no sliding window)",
    )
    parser.add_argument(
        "--chunk-model", default="qwen2.5:14b-instruct-q4_K_M",
        help="Ollama model for LLM chunking (default: qwen2.5:14b-instruct-q4_K_M)",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Insert chunks with embedding=NULL; run --embed-pending separately to fill embeddings",
    )
    parser.add_argument(
        "--embed-pending", action="store_true",
        help="Embed all document_chunks rows where embedding IS NULL, then exit",
    )
    args = parser.parse_args()

    # ── Embed-pending shortcut (no ingestion) ─────────────────────────────────
    if args.embed_pending:
        logger.info("Loading RAG system and embedding model…")
        from src.rag_system import RegulatoryRAGSystem
        rag = RegulatoryRAGSystem()
        n = rag.embed_pending(batch_size=args.embed_batch_size)
        logger.info("Done — %d chunks embedded", n)
        return

    # ── Local file ingestion path ──────────────────────────────────────────────
    if args.from_file:
        import json as _json
        path = Path(args.from_file)
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
        raw_docs = _json.loads(path.read_text())
        logger.info(f"Loaded {len(raw_docs)} docs from {path}")

        if args.source:
            raw_docs = [d for d in raw_docs if args.source.lower() in d.get("source", "").lower()
                        or args.source.lower() in d.get("url", "").lower()]
            logger.info(f"  Filtered to {len(raw_docs)} docs matching --source={args.source!r}")

        if not raw_docs:
            logger.error("No docs to ingest.")
            sys.exit(1)

        if args.dry_run:
            for d in raw_docs:
                logger.info(f"  Would ingest: {d.get('source','?')} — {len(d.get('text',''))} chars")
            return

        logger.info("Loading RAG system and embedding model…")
        from src.rag_system import RegulatoryRAGSystem
        rag = RegulatoryRAGSystem()

        total_chunks = 0
        for i, d in enumerate(raw_docs, 1):
            text = d.get("text", "")
            if len(text) < 500:
                logger.warning(f"  [{i}] Too short ({len(text)} chars), skipping")
                continue
            doc = {
                "text": text,
                "metadata": {
                    "documento_id": d.get("url", f"local_{i}"),
                    "source": d.get("source", d.get("title", f"doc_{i}")),
                    "framework": d.get("type", "scraped"),
                    "tags": d.get("keywords", []),
                    "url": d.get("url", ""),
                },
            }
            logger.info(f"[{i}/{len(raw_docs)}] Ingesting: {doc['metadata']['source']} ({len(text):,} chars)")
            if args.no_embed:
                n = rag.insert_chunks_only([doc])
            else:
                n = rag.procesar_documentos([doc], embed_batch_size=args.embed_batch_size)
            total_chunks += n
            del text
            logger.info(f"  → {n} chunks (running total: {total_chunks})")

        logger.info(f"Done — {total_chunks} total chunks ingested from {path.name}")
        return

    sources = SOURCES
    if args.source:
        sources = [s for s in SOURCES if s["id"] == args.source]
        if not sources:
            logger.error(f"Unknown source ID: {args.source}")
            logger.info(f"Available: {[s['id'] for s in SOURCES]}")
            sys.exit(1)

    checksums = _load_checksums()
    rag = None
    total_chunks = 0
    ingested = 0
    skipped = 0

    for source in sources:
        text = fetch_source(source)
        if text is None:
            continue

        checksum = _checksum(text)
        if not args.force and checksums.get(source["id"]) == checksum:
            logger.info(f"  No change — skipping {source['id']}")
            skipped += 1
            del text
            continue

        if args.dry_run:
            logger.info(f"  Would ingest: {source['name']} ({len(text):,} chars)")
            del text
            continue

        # Force GC to release BeautifulSoup/lxml DOM before loading the 1 GB model.
        # lxml uses C extensions that don't free via refcounting alone.
        import gc
        gc.collect()

        # Lazy-load RAG on first doc that needs ingesting.
        # With --no-embed we skip loading the embedding model (much faster startup).
        if rag is None:
            logger.info("Loading RAG system%s…", "" if args.no_embed else " and embedding model")
            from src.rag_system import RegulatoryRAGSystem
            rag = RegulatoryRAGSystem()

        ingested += 1
        logger.info(f"[{ingested}] Ingesting: {source['name']}")
        source_meta = {
            "documento_id": source["id"],
            "source": source["name"],
            "framework": source["framework"],
            "tags": source["tags"],
            "url": source["url"],
        }

        if args.llm_chunk:
            from src.chunking.llm_chunker import chunk_regulation
            logger.info("  LLM chunking with %s…", args.chunk_model)
            chunks = chunk_regulation(text, source_meta, model=args.chunk_model)
            logger.info("  → %d article chunks", len(chunks))
            doc = {"chunks": chunks, "metadata": source_meta}
        else:
            doc = {"text": text, "metadata": source_meta}

        if args.no_embed:
            n = rag.insert_chunks_only([doc])
        else:
            n = rag.procesar_documentos([doc], embed_batch_size=args.embed_batch_size)
        total_chunks += n
        checksums[source["id"]] = checksum
        del text  # release parsed text before fetching next doc
        logger.info(f"  → {n} chunks (running total: {total_chunks})")

    logger.info(f"\nSummary: {ingested} ingested, {skipped} unchanged")

    if rag is None:
        logger.info("Nothing to do.")
        return

    logger.info(f"Done — {total_chunks} chunks ingested")
    _save_checksums(checksums)
    logger.info(f"Checksums saved to {CHECKSUMS_FILE}")


if __name__ == "__main__":
    main()
