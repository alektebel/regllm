#!/usr/bin/env python3
"""CLI — SAS Parameter Compliance Checker.

Diagnoses regulatory compliance of IRB/IFRS9 parameter tables using
RAG + LLM. No hardcoded thresholds; every verdict is grounded in
retrieved regulatory text.

Usage:
  # CSV with local Ollama (default)
  python scripts/run_compliance_check.py pd_estimates.csv --table MYLIB.PD_ESTIMATES

  # Groq cloud backend
  python scripts/run_compliance_check.py pd_estimates.csv --backend groq

  # Map CSV column names to canonical regulatory parameter names
  python scripts/run_compliance_check.py data.csv --col-map PD_ESTIMATE:PD LGD_12M:LGD

  # Without DB/RAG (no DB connection needed)
  python scripts/run_compliance_check.py data.csv --no-rag

  # Annotate a SAS/EGP file with inline regulation comments
  python scripts/run_compliance_check.py --sas-file script.sas --output annotated.sas

  # Add domain context (field dictionary, emails, notes) to the checker
  python scripts/run_compliance_check.py data.csv --context-dir data/samples/

  # Ingest context documents into the knowledge base (one-time setup)
  python scripts/run_compliance_check.py --ingest-context data/samples/
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    p = argparse.ArgumentParser(description="Regulatory compliance checker for IRB/IFRS9 tables")
    p.add_argument("input", nargs="?", help="CSV or parquet file path (omit when using --sas-file or --ingest-context)")
    # SAS annotation mode
    p.add_argument("--sas-file", metavar="PATH", help="Parse and annotate a .sas or .egp file with regulation comments")
    # Context KB
    p.add_argument("--context-dir", metavar="PATH", help="Load context KB from this directory (field dicts, emails, notes)")
    p.add_argument("--ingest-context", metavar="PATH", help="Ingest context documents into the KB then exit")
    p.add_argument("--table", default="unknown", help="Table name for the report")
    p.add_argument("--output", "-o", help="Output JSON path (default: <input>_report.json)")
    p.add_argument("--backend", choices=["ollama", "groq"], default="ollama")
    p.add_argument("--ollama-model", default="qwen2.5:14b-instruct-q4_K_M")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    p.add_argument("--col-map", nargs="*", metavar="COL:KEY",
                   help="Map CSV column names to regulatory keys, e.g. PD_ESTIMATE:PD")
    p.add_argument("--row-id-col", help="Column to use as row identifier")
    p.add_argument("--no-rag", action="store_true", help="Skip RAG (no DB needed)")
    p.add_argument("--compress-threshold", type=int, default=20)
    p.add_argument("--rag-results", type=int, default=5, help="RAG chunks per row")
    p.add_argument("--db-url", help="Override DATABASE_URL env var")
    p.add_argument("--rerank", action="store_true",
                   help="Enable cross-encoder reranking (RAG-1, slower but more precise)")
    p.add_argument("--no-expand-parent", action="store_true",
                   help="Disable parent-text expansion (RAG-2 is on by default)")
    # Tiered pipeline
    p.add_argument("--tiered", action="store_true",
                   help="Use 3-tier pipeline (hard rules → cluster LLM → outlier LLM)")
    p.add_argument("--workers", type=int, default=4,
                   help="Thread-pool workers for Tier-2 cluster LLM calls (default: 4)")
    p.add_argument("--cluster-dims", nargs="*", metavar="COL",
                   help="Columns to group by in Tier-2 clustering (default: auto-detect)")
    p.add_argument("--cluster-sample", type=int, default=4,
                   help="Rows sampled per cluster for Tier-2 LLM call (default: 4)")
    p.add_argument("--zscore-threshold", type=float, default=3.0,
                   help="Z-score threshold for Tier-3 outlier detection (default: 3.0)")
    # Grain / PK
    p.add_argument("--primary-key", nargs="*", metavar="COL",
                   help="Explicit PK column(s) to build compound row IDs, e.g. CONTRATO CICLO")
    p.add_argument("--grain", dest="grain_description",
                   help="Human description of the table grain, e.g. 'ciclo de recuperación'")
    args = p.parse_args()

    column_map = {}
    if args.col_map:
        for entry in args.col_map:
            if ":" not in entry:
                sys.exit(f"Invalid --col-map entry {entry!r} — use COL:KEY format")
            csv_col, rule_key = entry.split(":", 1)
            column_map[csv_col] = rule_key.upper()

    # ── RAG setup (shared by all modes) ──────────────────────────────────────
    rag = None
    if not getattr(args, "no_rag", False):
        if getattr(args, "db_url", None):
            os.environ["DATABASE_URL"] = args.db_url
        try:
            from dotenv import load_dotenv; load_dotenv()
            from src.rag_system import RegulatoryRAGSystem
            rag = RegulatoryRAGSystem()
            logger.info("RAG system ready")
        except Exception as e:
            logger.warning("RAG unavailable (%s) — running without regulatory context", e)

    # ── Mode: ingest context documents ───────────────────────────────────────
    if args.ingest_context:
        from src.context_kb import ContextKB
        kb = ContextKB(rag_system=rag)
        n = kb.ingest_directory(args.ingest_context)
        print(f"Ingested {n} chunks from {args.ingest_context}")
        return

    # ── Mode: SAS file annotation ─────────────────────────────────────────────
    if args.sas_file:
        from src.sas_parser import SASParser
        from src.compliance.sas_annotator import SASAnnotator

        blocks = SASParser().parse(args.sas_file)
        logger.info("Parsed %d SAS code blocks from %s", len(blocks), args.sas_file)

        llm_fn = None
        if not getattr(args, "no_rag", False):
            # Build a minimal LLM function using the same backend as data mode
            _backend = getattr(args, "backend", "ollama")
            if _backend == "ollama":
                _host = getattr(args, "ollama_host", "http://localhost:11434")
                _model = getattr(args, "ollama_model", "qwen2.5:14b-instruct-q4_K_M")
                def llm_fn(messages):
                    import ollama as _ol
                    r = _ol.Client(host=_host).chat(model=_model, messages=messages, stream=False,
                                                     options={"temperature": 0.1, "num_ctx": 4096})
                    return r["message"]["content"]
            elif _backend == "groq":
                from groq import Groq
                _gmodel = getattr(args, "groq_model", "llama-3.3-70b-versatile")
                def llm_fn(messages):
                    return Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
                        model=_gmodel, messages=messages, temperature=0.1, max_tokens=256
                    ).choices[0].message.content

        annotator = SASAnnotator(rag_system=rag, llm_fn=llm_fn)
        annotated = annotator.annotate(blocks)
        output_text = annotator.render(annotated)

        warnings_total = sum(len(ab.warnings) for ab in annotated)
        output_path = args.output or args.sas_file.rsplit(".", 1)[0] + "_annotated.sas"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"\nAnnotated {len(blocks)} code block(s) — {warnings_total} potential issues flagged")
        print(f"Output: {output_path}")
        if warnings_total:
            print("\nPotential issues:")
            for ab in annotated:
                for w in ab.warnings:
                    print(f"  [{ab.task_name}] {w[:100]}")
        return

    # ── Mode: data instance validation ────────────────────────────────────────
    if not args.input:
        p.error("Provide an input file, --sas-file, or --ingest-context")

    import pandas as pd
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), args.input)

    # Context KB (optional)
    context_kb = None
    if args.context_dir:
        try:
            from src.context_kb import ContextKB
            context_kb = ContextKB(rag_system=rag)
            logger.info("Context KB ready (from %s)", args.context_dir)
        except Exception as e:
            logger.warning("Context KB unavailable (%s)", e)

    from src.compliance import ComplianceChecker
    checker = ComplianceChecker(
        rag_system=rag,
        llm_backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        groq_model=args.groq_model,
        column_map=column_map or None,
        compress_threshold=args.compress_threshold,
        rag_n_results=args.rag_results,
        rerank=args.rerank,
        expand_to_parent=not args.no_expand_parent,
        primary_key=args.primary_key or None,
        grain_description=args.grain_description,
        context_kb=context_kb,
    )

    if args.tiered:
        report = checker.run_tiered(
            df,
            table_name=args.table,
            row_id_col=args.row_id_col,
            workers=args.workers,
            cluster_dims=args.cluster_dims or None,
            cluster_sample_n=args.cluster_sample,
            zscore_threshold=args.zscore_threshold,
        )
    else:
        report = checker.run(df, table_name=args.table, row_id_col=args.row_id_col)

    output = args.output or args.input.rsplit(".", 1)[0] + "_report.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Report written to %s", output)

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Table            {report['table']}")
    print(f"  Rows processed   {report['rows_processed']}")
    print(f"  Rows flagged     {report['rows_flagged']}")
    print(f"  Rows uncertain   {report.get('rows_uncertain', 0)}")
    print(f"  Compliance rate  {report['compliance_rate']:.1%}")
    print(f"  LLM calls        {report['llm_calls']}")
    if tiers := report.get("tier_counts"):
        print(f"  Tier 1 (rules)   {tiers['tier1_flagged']} flagged")
        print(f"  Tier 2 (cluster) {tiers['tier2_rows']} rows evaluated")
        print(f"  Tier 3 (outlier) {tiers['tier3_outlier_rows']} rows individually checked")
    print(f"  Report           {output}")
    print(bar)
    if report.get("narrative"):
        n = report["narrative"]
        print(f"\n  {n.get('resumen', '')}")
        if n.get("advertencias"):
            print(f"  ⚠  {n['advertencias']}")
    print()


if __name__ == "__main__":
    main()
