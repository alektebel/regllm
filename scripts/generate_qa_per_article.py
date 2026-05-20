"""
Generate Q&A pairs from document chunks using Groq (fast + cheap).
=================================================================

Reads all chunks from the `document_chunks` table, generates 1-2 questions
per chunk that a banking supervisor or risk analyst might ask, and stores the
results in data/qa_per_article.jsonl.

Usage:
    python scripts/generate_qa_per_article.py
    python scripts/generate_qa_per_article.py --limit 50   # test with first 50 chunks
    python scripts/generate_qa_per_article.py --output data/my_qa.jsonl

Requirements:
    pip install groq psycopg2-binary
    export GROQ_API_KEY=gsk_...

DB connection (env vars, same as API):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "qa_per_article.jsonl"
GROQ_MODEL = "llama-3.1-8b-instant"
SLEEP_BETWEEN_CALLS = 0.3  # seconds

PROMPT_TEMPLATE = """\
Eres un experto en regulación bancaria española. Dado el siguiente fragmento regulatorio, genera exactamente 2 preguntas concretas y técnicas que un supervisor bancario o analista de riesgos podría hacer, cuya respuesta esté contenida en el fragmento. Responde SOLO con las 2 preguntas, una por línea, sin numeración.

Fragmento:
{chunk_text}"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn_kwargs() -> dict:
    return dict(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def fetch_chunks(limit: int | None) -> list[dict]:
    """Fetch all (or up to `limit`) chunks from document_chunks."""
    query = "SELECT chunk_id, texto, metadata FROM document_chunks ORDER BY id"
    params = ()
    if limit:
        query += " LIMIT %s"
        params = (limit,)

    conn = psycopg2.connect(**_conn_kwargs())
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    chunks = []
    for chunk_id, texto, metadata in rows:
        meta = metadata if isinstance(metadata, dict) else (json.loads(metadata) if metadata else {})
        chunks.append({
            "chunk_id": chunk_id,
            "texto": texto,
            "source": meta.get("source", meta.get("documento_id", "")),
            "articulo": meta.get("articulo", ""),
        })
    return chunks


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_processed_ids(output_path: Path) -> set[str]:
    """Return the set of chunk_ids already written to the output file."""
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cid = record.get("chunk_id")
                if cid:
                    processed.add(cid)
            except json.JSONDecodeError:
                pass
    return processed


# ── Groq call ─────────────────────────────────────────────────────────────────

def generate_questions(client, chunk_text: str) -> list[str]:
    """
    Call Groq and return up to 2 questions.
    Handles rate-limit (429) with exponential backoff.
    """
    prompt = PROMPT_TEMPLATE.format(chunk_text=chunk_text[:3000])  # guard against very long chunks

    max_retries = 6
    wait = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            questions = [q.strip() for q in raw.splitlines() if q.strip()]
            return questions[:2]
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                logger.warning(f"Rate limit hit — backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                logger.error(f"Groq error: {exc}")
                return []
    logger.error("Max retries exceeded for Groq call")
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from document_chunks using Groq")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N chunks (useful for testing)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed — run: pip install groq")
        sys.exit(1)

    client = Groq(api_key=api_key)

    logger.info("Fetching chunks from database…")
    try:
        chunks = fetch_chunks(args.limit)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    logger.info(f"Found {len(chunks)} chunks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(args.output)
    logger.info(f"Already processed: {len(processed_ids)} chunks (checkpoint)")

    to_process = [c for c in chunks if c["chunk_id"] not in processed_ids]
    logger.info(f"Chunks to process: {len(to_process)}")

    if not to_process:
        logger.info("Nothing to do — all chunks already processed.")
        return

    written = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, chunk in enumerate(to_process, 1):
            chunk_text = chunk["texto"]
            if not chunk_text or len(chunk_text.strip()) < 80:
                logger.debug(f"Skipping short chunk {chunk['chunk_id']}")
                continue

            logger.info(f"[{i}/{len(to_process)}] {chunk['chunk_id']}")

            questions = generate_questions(client, chunk_text)
            if not questions:
                logger.warning(f"  No questions generated for {chunk['chunk_id']}")
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            for question in questions:
                record = {
                    "question": question,
                    "answer": chunk_text,
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "articulo": chunk["articulo"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            out_f.flush()
            time.sleep(SLEEP_BETWEEN_CALLS)

    logger.info(f"\nDone — {written} Q&A pairs written to {args.output}")


if __name__ == "__main__":
    main()
