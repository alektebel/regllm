"""
Assertion Runner
================
Executes SQL compliance check templates from regulation_assertions against
a target database and produces a structured findings report.

This is the final step of the compliance assertion pipeline:
  1. ingest_regulations.py      → document_chunks
  2. extract_assertions.py      → regulation_assertions (with rule text)
  3. ingest_db_context.py       → context_chunks (schema knowledge)
  4. generate_sql_templates.py  → regulation_assertions.sql_template filled
  5. run_assertions.py (this)   → executes SQL → findings report

The TARGET database is the bank's own data warehouse.
By default it uses the same connection as the regllm DB (for dev/demo).
In production, set TARGET_DB_* env vars to point at the bank's DB.

Usage:
    # Run all high/critical assertions (preview without writing)
    python scripts/data/run_assertions.py --materiality high,critical --dry-run

    # Run assertions for one regulation
    python scripts/data/run_assertions.py --regulation crr_575_2013

    # Run against a separate target database
    python scripts/data/run_assertions.py --target-dsn "postgresql://user:pass@host/bank_dw"

    # Write findings to JSON and CSV
    python scripts/data/run_assertions.py --out findings.json --csv findings.csv

    # Skip LLM narrative (faster, just counts)
    python scripts/data/run_assertions.py --no-llm

DB connections:
    regllm DB (reads assertions):
        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    Target DB (executes SQL):
        TARGET_DB_HOST, TARGET_DB_PORT, TARGET_DB_NAME, TARGET_DB_USER, TARGET_DB_PASSWORD
        — or — TARGET_DB_DSN = "postgresql://..."
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _get_meta_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def _print_summary(report: dict) -> None:
    findings = report.get("findings", [])
    print("\n" + "=" * 70)
    print("ASSERTION COMPLIANCE REPORT")
    print("=" * 70)
    print(f"Run date              : {report['run_date']}")
    print(f"Assertions checked    : {report['assertions_checked']}")
    print(f"Assertions with issues: {report['assertions_flagged']}")
    print(f"Compliance rate       : {report['compliance_rate']:.1%}")
    print()

    if not findings:
        print("All assertions passed — no compliance issues found.")
        return

    by_materiality: dict[str, list] = {}
    for f in findings:
        m = f.get("materiality", "medium")
        by_materiality.setdefault(m, []).append(f)

    for mat in ("critical", "high", "medium", "low"):
        group = by_materiality.get(mat, [])
        if not group:
            continue
        print(f"[{mat.upper()}] {len(group)} finding(s)")
        for f in group:
            print(f"  • {f['regulation_name']} {f['article']}")
            print(f"    {f.get('titulo') or f['rule_text'][:80]}")
            print(f"    → {f['n_failing']:,} rows failing ({f['rate']:.2%})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Execute SQL assertion checks against target database and produce findings report"
    )
    parser.add_argument("--regulation", metavar="ID",
                        help="Filter to one regulation ID (e.g. crr_575_2013)")
    parser.add_argument("--materiality", metavar="LEVELS",
                        help="Comma-separated materiality levels to run (e.g. high,critical)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max assertions to run (default: 500; use 0 for all)")
    parser.add_argument("--target-dsn", metavar="DSN",
                        help="PostgreSQL DSN for the target DB (default: regllm DB)")
    parser.add_argument("--model", default="llama3.3",
                        help="Ollama model for finding narrative (default: llama3.3)")
    parser.add_argument("--backend", choices=["ollama", "groq"], default="ollama",
                        help="LLM backend for finding narrative (default: ollama)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM narrative — output counts only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print SQL templates without executing them")
    parser.add_argument("--out", metavar="FILE",
                        help="Write JSON report to this file")
    parser.add_argument("--csv", metavar="FILE",
                        help="Write CSV summary to this file")
    args = parser.parse_args()

    meta_conn = _get_meta_conn()

    # Build LLM function for finding narrative
    llm_fn = None
    if not args.no_llm:
        from src.assertions.extractor import build_llm_fn
        llm_fn = build_llm_fn(backend=args.backend, model=args.model)
        logger.info("Finding narrative: backend=%s model=%s", args.backend, args.model)

    from src.assertions.runner import AssertionRunner

    target_dsn = args.target_dsn or os.getenv("TARGET_DB_DSN")

    runner = AssertionRunner(
        target_dsn=target_dsn,
        llm_fn=llm_fn,
        materiality_filter=args.materiality,
    )

    limit = args.limit if args.limit > 0 else 10_000

    if args.dry_run:
        runner.run_all(meta_conn, regulation_id=args.regulation, limit=limit, dry_run=True)
        meta_conn.close()
        return

    findings = runner.run_all(
        meta_conn,
        regulation_id=args.regulation,
        limit=limit,
        dry_run=False,
    )
    meta_conn.close()

    # Count how many assertions were run (approximate — runner doesn't return this directly)
    # We re-query to get the count
    meta_conn2 = _get_meta_conn()
    from src.assertions.runner import _fetch_assertions_with_sql
    total_rows = len(_fetch_assertions_with_sql(meta_conn2, args.regulation, args.materiality, limit))
    meta_conn2.close()

    report = runner.build_report(findings, total_rows)

    _print_summary(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("JSON report written to %s", out_path)

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "assertion_id", "regulation_id", "article", "rule_text",
                "materiality", "check_type", "n_failing", "n_total", "rate",
                "titulo", "descripcion",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for finding in report["findings"]:
                w.writerow({**finding, "rate": f"{finding['rate']:.4f}"})
        logger.info("CSV summary written to %s", csv_path)


if __name__ == "__main__":
    main()
