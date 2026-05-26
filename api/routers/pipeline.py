"""
Pipeline router — DB context ingestion, assertion extraction, SQL generation,
audit finding execution, and knowledge graph queries.

Endpoints:
    GET  /pipeline/status                — corpus + assertion counts
    POST /pipeline/ingest-context        — upload Excel/SQL/SAS, store context_chunks
    POST /pipeline/extract-assertions    — SSE: run assertion extractor on corpus
    GET  /pipeline/assertions            — paginated assertion list
    POST /pipeline/generate-sql          — generate SQL templates for assertions
    POST /pipeline/run-assertions        — SSE: execute assertions, return findings
    GET  /pipeline/graph                 — nodes + edges for visualization
    GET  /pipeline/themes                — all validation themes with chunk counts
    GET  /pipeline/themes/{id}/chunks    — chunks for a theme, ranked by score
    GET  /pipeline/hierarchy/{doc_id}    — structural tree for a document
    GET  /pipeline/hierarchy/{doc_id}/themes — themes covered by a document
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import psycopg2
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_executor = ThreadPoolExecutor(max_workers=2)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )
    return conn


def _sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
async def pipeline_status():
    """Counts for corpus, assertions, context chunks, and graph nodes."""
    def _query():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*), COUNT(embedding) FROM document_chunks")
                chunks_total, chunks_embedded = cur.fetchone()

                cur.execute("SELECT COUNT(*) FROM regulation_assertions WHERE active")
                assertions_total = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM regulation_assertions WHERE sql_template IS NOT NULL AND active")
                assertions_with_sql = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM context_chunks")
                context_chunks = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM graph_nodes")
                graph_nodes = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM graph_edges")
                graph_edges = cur.fetchone()[0]

                cur.execute("""
                    SELECT metadata->>'documento_id', COUNT(*)
                    FROM document_chunks
                    GROUP BY metadata->>'documento_id'
                    ORDER BY COUNT(*) DESC
                """)
                corpus = [{"doc": r[0], "chunks": r[1]} for r in cur.fetchall()]

                cur.execute("""
                    SELECT materiality, COUNT(*)
                    FROM regulation_assertions WHERE active
                    GROUP BY materiality
                    ORDER BY CASE materiality
                        WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3 ELSE 4 END
                """)
                by_materiality = {r[0]: r[1] for r in cur.fetchall()}

            return {
                "corpus": {
                    "total_chunks": chunks_total,
                    "embedded": chunks_embedded,
                    "documents": corpus,
                },
                "assertions": {
                    "total": assertions_total,
                    "with_sql": assertions_with_sql,
                    "pending_sql": assertions_total - assertions_with_sql,
                    "by_materiality": by_materiality,
                },
                "context": {"chunks": context_chunks},
                "graph": {"nodes": graph_nodes, "edges": graph_edges},
            }
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)


# ── DB Context Ingestion ──────────────────────────────────────────────────────

_ALLOWED_CONTEXT_EXTS = {".xlsx", ".xls", ".sql", ".sas", ".egp"}


@router.post("/ingest-context")
async def ingest_context(files: list[UploadFile] = File(...)):
    """
    Upload one or more Excel data dictionaries, SQL DDL files, or SAS files.
    Extracts column/field context and stores embeddings in context_chunks.
    """
    saved: list[tuple[Path, str]] = []  # (tmp_path, original_name)
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in _ALLOWED_CONTEXT_EXTS:
            raise HTTPException(400, f"Unsupported file type: {f.filename}. Allowed: {sorted(_ALLOWED_CONTEXT_EXTS)}")
        content = await f.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()
        saved.append((Path(tmp.name), f.filename or tmp.name))

    def _ingest():
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.data.ingest_db_context import (
            _collect_excel, _collect_sql, _collect_sas,
            _embed_and_store, _load_embedder, _get_conn as _ctx_conn, _ensure_table,
        )

        excel_paths = [p for p, n in saved if p.suffix in {".xlsx", ".xls"}]
        sql_paths   = [p for p, n in saved if p.suffix == ".sql"]
        sas_paths   = [p for p, n in saved if p.suffix in {".sas", ".egp"}]

        entries = []
        if excel_paths: entries.extend(_collect_excel(excel_paths))
        if sql_paths:   entries.extend(_collect_sql(sql_paths))
        if sas_paths:   entries.extend(_collect_sas(sas_paths))

        if not entries:
            return {"stored": 0, "message": "No extractable content found in uploaded files"}

        embedder = _load_embedder()
        conn = _ctx_conn()
        _ensure_table(conn)
        stored = _embed_and_store(entries, embedder, conn)
        conn.close()

        # Also create db_column nodes in the knowledge graph
        _seed_graph_from_context(entries)

        return {"stored": stored, "files": [n for _, n in saved]}

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _ingest)
    finally:
        for p, _ in saved:
            p.unlink(missing_ok=True)

    return result


def _seed_graph_from_context(entries: list[tuple[str, dict]]) -> None:
    """Create db_table + db_column nodes in the knowledge graph from context entries."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            for content, meta in entries:
                table = meta.get("table_name", "")
                column = meta.get("column_name", "")
                if not table or not column:
                    continue

                table_id = f"tbl:{table.lower()}"
                col_id = f"col:{table.lower()}.{column.lower()}"

                cur.execute(
                    """
                    INSERT INTO graph_nodes (id, node_type, label, metadata)
                    VALUES (%s, 'db_table', %s, %s::jsonb)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (table_id, table, json.dumps({"source_type": meta.get("source_type", "")})),
                )
                cur.execute(
                    """
                    INSERT INTO graph_nodes (id, node_type, label, metadata)
                    VALUES (%s, 'db_column', %s, %s::jsonb)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (col_id, f"{table}.{column}", json.dumps({
                        "table": table,
                        "column": column,
                        "description": content[:300],
                        "source_type": meta.get("source_type", ""),
                    })),
                )
                cur.execute(
                    """
                    INSERT INTO graph_edges (from_id, to_id, relation)
                    VALUES (%s, %s, 'FEEDS_INTO')
                    ON CONFLICT (from_id, to_id, relation) DO NOTHING
                    """,
                    (col_id, table_id),
                )
        conn.commit()
    except Exception as e:
        logger.warning("Graph seed from context failed: %s", e)
        conn.rollback()
    finally:
        conn.close()


# ── Assertion Extraction ──────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    regulation_id: str | None = None
    limit: int = 500
    model: str = "qwen2.5:14b-instruct-q4_K_M"
    backend: str = "ollama"


@router.post("/extract-assertions")
async def extract_assertions(req: ExtractRequest):
    """
    SSE stream. Extracts testable assertions from regulation chunks and stores
    them in regulation_assertions. Also seeds article nodes in the graph.
    """
    async def _stream() -> AsyncGenerator[str, None]:
        yield _sse("start", {"message": f"Starting extraction (backend={req.backend}, limit={req.limit})"})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            from src.assertions.extractor import AssertionExtractor, build_llm_fn
            llm_fn = build_llm_fn(backend=req.backend, model=req.model)
            extractor = AssertionExtractor(llm_fn=llm_fn)

            def _progress(idx: int, total: int, stored: int):
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "progress", "chunk": idx, "total": total, "assertions_stored": stored},
                )

            total = extractor.run_from_db(
                regulation_id=req.regulation_id,
                limit=req.limit,
                progress_cb=_progress,
            )
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "done", "total": total})

        future = loop.run_in_executor(_executor, _run)

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("event") == "done":
                    # Seed regulation_article nodes into graph
                    await loop.run_in_executor(_executor, _seed_graph_from_assertions)
                    break
            except asyncio.TimeoutError:
                yield _sse("heartbeat", {})

        try:
            await future
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _seed_graph_from_assertions() -> None:
    """Create regulation_article + assertion nodes for every stored assertion."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, regulation_id, regulation_name, article, rule_text, materiality
                FROM regulation_assertions WHERE active
            """)
            rows = cur.fetchall()

        with conn.cursor() as cur:
            for (a_id, reg_id, reg_name, article, rule_text, materiality) in rows:
                art_node_id = f"art:{reg_id}__{article.lower().replace(' ', '_')[:40]}"
                assertion_node_id = f"asr:{a_id}"

                cur.execute(
                    """
                    INSERT INTO graph_nodes (id, node_type, label, metadata)
                    VALUES (%s, 'regulation_article', %s, %s::jsonb)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (art_node_id, f"{article} ({reg_id})", json.dumps({
                        "regulation_id": reg_id,
                        "regulation_name": reg_name,
                        "article": article,
                    })),
                )
                cur.execute(
                    """
                    INSERT INTO graph_nodes (id, node_type, label, metadata)
                    VALUES (%s, 'assertion', %s, %s::jsonb)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (assertion_node_id, rule_text[:80], json.dumps({
                        "assertion_id": a_id,
                        "materiality": materiality,
                        "article": article,
                        "regulation_id": reg_id,
                    })),
                )
                cur.execute(
                    """
                    INSERT INTO graph_edges (from_id, to_id, relation)
                    VALUES (%s, %s, 'CONSTRAINS')
                    ON CONFLICT (from_id, to_id, relation) DO NOTHING
                    """,
                    (art_node_id, assertion_node_id),
                )
        conn.commit()
    except Exception as e:
        logger.warning("Graph seed from assertions failed: %s", e)
        conn.rollback()
    finally:
        conn.close()


# ── List Assertions ───────────────────────────────────────────────────────────

@router.get("/assertions")
async def list_assertions(
    regulation_id: str | None = None,
    materiality: str | None = None,
    has_sql: bool | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Paginated list of extracted assertions."""
    def _query():
        conn = _get_conn()
        try:
            conditions = ["active = TRUE"]
            params: list = []
            if regulation_id:
                conditions.append("regulation_id = %s")
                params.append(regulation_id)
            if materiality:
                conditions.append("materiality = %s")
                params.append(materiality)
            if has_sql is True:
                conditions.append("sql_template IS NOT NULL")
            elif has_sql is False:
                conditions.append("sql_template IS NULL")

            where = " AND ".join(conditions)
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM regulation_assertions WHERE {where}", params)
                total = cur.fetchone()[0]

                cur.execute(
                    f"""
                    SELECT id, regulation_id, regulation_name, article, rule_text,
                           check_type, materiality, conditions, threshold, sql_template, notes
                    FROM regulation_assertions
                    WHERE {where}
                    ORDER BY CASE materiality
                        WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3 ELSE 4 END,
                        regulation_id, article
                    LIMIT %s OFFSET %s
                    """,
                    params + [limit, offset],
                )
                rows = cur.fetchall()

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [
                    {
                        "id": r[0], "regulation_id": r[1], "regulation_name": r[2],
                        "article": r[3], "rule_text": r[4], "check_type": r[5],
                        "materiality": r[6], "conditions": r[7], "threshold": r[8],
                        "sql_template": r[9], "notes": r[10],
                    }
                    for r in rows
                ],
            }
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)


# ── SQL Generation ────────────────────────────────────────────────────────────

class GenerateSQLRequest(BaseModel):
    regulation_id: str | None = None
    assertion_ids: list[str] | None = None  # specific assertions; None = all without SQL
    model: str = "qwen2.5-coder:14b"
    backend: str = "ollama"
    force: bool = False  # regenerate even if SQL already exists


@router.post("/generate-sql")
async def generate_sql(req: GenerateSQLRequest):
    """
    SSE stream. Generates SQL compliance check templates for assertions that
    don't have one yet (or all, if force=True). Requires context_chunks to be
    populated for meaningful schema mapping.
    """
    async def _stream() -> AsyncGenerator[str, None]:
        yield _sse("start", {"message": "Starting SQL generation…"})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            from src.assertions.sql_generator import SQLGenerator
            from src.assertions.extractor import build_llm_fn
            gen = SQLGenerator(llm_fn=build_llm_fn(backend=req.backend, model=req.model))

            conn = _get_conn()
            try:
                conditions = ["active = TRUE"]
                params: list = []
                if not req.force:
                    conditions.append("sql_template IS NULL")
                if req.regulation_id:
                    conditions.append("regulation_id = %s")
                    params.append(req.regulation_id)
                if req.assertion_ids:
                    placeholders = ",".join(["%s"] * len(req.assertion_ids))
                    conditions.append(f"id IN ({placeholders})")
                    params.extend(req.assertion_ids)

                where = " AND ".join(conditions)
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT id, regulation_id, regulation_name, article, passage, "
                        f"rule_text, check_type, conditions, threshold, materiality, notes "
                        f"FROM regulation_assertions WHERE {where}",
                        params,
                    )
                    rows = cur.fetchall()

                from src.assertions.models import Assertion, Threshold
                assertions = []
                for r in rows:
                    threshold = None
                    if r[8]:
                        t = r[8] if isinstance(r[8], dict) else json.loads(r[8])
                        threshold = Threshold(**t)
                    conds = r[7] if isinstance(r[7], list) else json.loads(r[7] or "[]")
                    a = Assertion(
                        regulation_id=r[1], regulation_name=r[2], article=r[3],
                        passage=r[4] or "", rule_text=r[5], check_type=r[6],
                        conditions=conds, threshold=threshold,
                        materiality=r[9], notes=r[10] or "",
                    )
                    a._id = r[0]
                    assertions.append(a)

                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "info", "message": f"Generating SQL for {len(assertions)} assertions"},
                )

                done = 0
                for a in assertions:
                    try:
                        sql = gen.generate(a, conn)
                        if sql:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE regulation_assertions SET sql_template = %s WHERE id = %s",
                                    (sql, a.id),
                                )
                            conn.commit()
                            done += 1
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {"event": "progress", "done": done, "total": len(assertions),
                                 "assertion_id": a.id, "article": a.article},
                            )
                    except Exception as e:
                        logger.warning("SQL gen failed for %s: %s", a.id, e)

                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "done", "generated": done, "total": len(assertions)},
                )
            finally:
                conn.close()

        future = loop.run_in_executor(_executor, _run)

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("event") == "done":
                    break
            except asyncio.TimeoutError:
                yield _sse("heartbeat", {})

        try:
            await future
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Run Assertions (Audit) ────────────────────────────────────────────────────

class RunRequest(BaseModel):
    # Connection to the BANK's data DB (separate from the metadata DB)
    # If omitted, runs against the metadata DB itself (useful for testing)
    target_dsn: str | None = None
    regulation_id: str | None = None
    materiality: str | None = None   # filter: critical/high/medium/low
    limit_assertions: int = 50


@router.post("/run-assertions")
async def run_assertions(req: RunRequest):
    """
    SSE stream. Executes SQL assertion templates against the target database
    and returns findings with regulatory citations and graph-aware materiality.
    """
    async def _stream() -> AsyncGenerator[str, None]:
        yield _sse("start", {"message": "Connecting to target database…"})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            # Connect to target DB (defaults to metadata DB for testing)
            if req.target_dsn:
                import psycopg2 as _pg
                target_conn = _pg.connect(req.target_dsn)
            else:
                target_conn = _get_conn()

            meta_conn = _get_conn()
            try:
                # Fetch assertions with SQL templates
                conditions = ["active = TRUE", "sql_template IS NOT NULL"]
                params: list = []
                if req.regulation_id:
                    conditions.append("regulation_id = %s")
                    params.append(req.regulation_id)
                if req.materiality:
                    conditions.append("materiality = %s")
                    params.append(req.materiality)

                where = " AND ".join(conditions)
                with meta_conn.cursor() as cur:
                    cur.execute(
                        f"SELECT id, regulation_id, regulation_name, article, rule_text, "
                        f"check_type, materiality, sql_template, threshold "
                        f"FROM regulation_assertions WHERE {where} "
                        f"LIMIT %s",
                        params + [req.limit_assertions],
                    )
                    rows = cur.fetchall()

                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "info", "message": f"Running {len(rows)} assertions…"},
                )

                findings = []
                errors = []
                for i, (a_id, reg_id, reg_name, article, rule_text, check_type, materiality, sql, threshold) in enumerate(rows):
                    try:
                        with target_conn.cursor() as cur:
                            cur.execute(sql)
                            result = cur.fetchall()
                            cols = [d[0] for d in cur.description] if cur.description else []

                        # Interpret result: look for n_failing / count / similar
                        n_failing = 0
                        result_data = []
                        if result:
                            first_row = dict(zip(cols, result[0]))
                            for key in ("n_failing", "count", "failing", "total"):
                                if key in first_row:
                                    try:
                                        n_failing = int(first_row[key])
                                    except (ValueError, TypeError):
                                        pass
                                    break
                            result_data = [dict(zip(cols, r)) for r in result[:20]]

                        # Graph materiality: traverse downstream to estimate capital impact
                        downstream = _get_downstream_count(meta_conn, f"asr:{a_id}")

                        finding = {
                            "assertion_id": a_id,
                            "regulation_id": reg_id,
                            "regulation_name": reg_name,
                            "article": article,
                            "rule_text": rule_text,
                            "check_type": check_type,
                            "materiality": materiality,
                            "n_failing": n_failing,
                            "result_sample": result_data,
                            "downstream_nodes": downstream,
                            "status": "FAIL" if n_failing > 0 else "PASS",
                        }
                        findings.append(finding)

                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"event": "finding", "index": i + 1, "total": len(rows), **finding},
                        )

                    except Exception as e:
                        err = {"assertion_id": a_id, "article": article, "error": str(e)}
                        errors.append(err)
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"event": "sql_error", **err},
                        )

                total_fail = sum(1 for f in findings if f["status"] == "FAIL")
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "done", "total": len(rows), "failures": total_fail,
                     "errors": len(errors), "pass_rate": round((len(rows) - total_fail) / max(len(rows), 1), 3)},
                )
            finally:
                target_conn.close()
                meta_conn.close()

        future = loop.run_in_executor(_executor, _run)

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("event") == "done":
                    break
            except asyncio.TimeoutError:
                yield _sse("heartbeat", {})

        try:
            await future
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _get_downstream_count(conn, node_id: str, max_depth: int = 4) -> int:
    """Count downstream graph nodes reachable from node_id (materiality proxy)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE downstream AS (
                    SELECT to_id, 1 AS depth FROM graph_edges WHERE from_id = %s
                    UNION ALL
                    SELECT e.to_id, d.depth + 1
                    FROM graph_edges e JOIN downstream d ON e.from_id = d.to_id
                    WHERE d.depth < %s
                )
                SELECT COUNT(DISTINCT to_id) FROM downstream
                """,
                (node_id, max_depth),
            )
            return cur.fetchone()[0]
    except Exception:
        return 0


# ── Knowledge Graph ───────────────────────────────────────────────────────────

@router.get("/graph")
async def get_graph(node_type: str | None = None, limit: int = 500):
    """Return graph nodes and edges for visualization."""
    def _query():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                if node_type:
                    cur.execute(
                        "SELECT id, node_type, label, metadata FROM graph_nodes WHERE node_type = %s LIMIT %s",
                        (node_type, limit),
                    )
                else:
                    cur.execute("SELECT id, node_type, label, metadata FROM graph_nodes LIMIT %s", (limit,))
                nodes = [{"id": r[0], "type": r[1], "label": r[2], "metadata": r[3]} for r in cur.fetchall()]

                node_ids = [n["id"] for n in nodes]
                if node_ids:
                    placeholders = ",".join(["%s"] * len(node_ids))
                    cur.execute(
                        f"""
                        SELECT from_id, to_id, relation, weight
                        FROM graph_edges
                        WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})
                        """,
                        node_ids * 2,
                    )
                    edges = [{"from": r[0], "to": r[1], "relation": r[2], "weight": r[3]} for r in cur.fetchall()]
                else:
                    edges = []

            return {"nodes": nodes, "edges": edges}
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)


# ── Hierarchical retrieval ────────────────────────────────────────────────────

@router.get("/themes")
async def list_themes():
    """All validation themes with chunk counts."""
    def _query():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vt.id, vt.label, vt.description,
                           COUNT(ct.chunk_id) AS chunk_count
                    FROM validation_themes vt
                    LEFT JOIN chunk_themes ct ON ct.theme_id = vt.id
                    GROUP BY vt.id, vt.label, vt.description
                    ORDER BY chunk_count DESC
                    """
                )
                return [
                    {"id": r[0], "label": r[1], "description": r[2], "chunk_count": r[3]}
                    for r in cur.fetchall()
                ]
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)


@router.get("/themes/{theme_id}/chunks")
async def theme_chunks(theme_id: str, limit: int = 20, offset: int = 0):
    """Return chunks tagged with a given theme, ranked by relevance score."""
    def _query():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dc.chunk_id, dc.texto, ct.score, ct.method,
                           dh.label AS node_label, dh.level_type
                    FROM chunk_themes ct
                    JOIN document_chunks dc ON dc.chunk_id = ct.chunk_id
                    LEFT JOIN document_hierarchy dh ON dh.node_id = dc.hierarchy_node_id
                    WHERE ct.theme_id = %s
                    ORDER BY ct.score DESC
                    LIMIT %s OFFSET %s
                    """,
                    (theme_id, limit, offset),
                )
                rows = cur.fetchall()

                cur.execute(
                    "SELECT COUNT(*) FROM chunk_themes WHERE theme_id = %s",
                    (theme_id,),
                )
                total = cur.fetchone()[0]

            return {
                "total": total,
                "items": [
                    {
                        "chunk_id": r[0],
                        "texto": r[1][:500],
                        "score": r[2],
                        "method": r[3],
                        "node_label": r[4],
                        "level_type": r[5],
                    }
                    for r in rows
                ],
            }
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)


@router.get("/hierarchy/{doc_id}")
async def doc_hierarchy(doc_id: str):
    """Structural tree for a document."""
    def _query():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE tree AS (
                        SELECT node_id, label, level_type, parent_id,
                               number, article_count, 0 AS depth, position
                        FROM document_hierarchy
                        WHERE doc_id = %s AND parent_id IS NULL
                        UNION ALL
                        SELECT dh.node_id, dh.label, dh.level_type, dh.parent_id,
                               dh.number, dh.article_count, tree.depth + 1, dh.position
                        FROM document_hierarchy dh
                        JOIN tree ON dh.parent_id = tree.node_id
                    )
                    SELECT node_id, label, level_type, parent_id,
                           number, article_count, depth
                    FROM tree ORDER BY depth, position
                    """,
                    (doc_id,),
                )
                return [
                    {
                        "node_id": r[0], "label": r[1], "level_type": r[2],
                        "parent_id": r[3], "number": r[4],
                        "article_count": r[5], "depth": r[6],
                    }
                    for r in cur.fetchall()
                ]
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)


@router.get("/hierarchy/{doc_id}/themes")
async def doc_themes(doc_id: str):
    """Which validation themes are covered by a document."""
    def _query():
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vt.id, vt.label, COUNT(ct.chunk_id) AS chunk_count,
                           ROUND(AVG(ct.score)::numeric, 3) AS avg_score
                    FROM chunk_themes ct
                    JOIN validation_themes vt ON vt.id = ct.theme_id
                    JOIN document_chunks dc ON dc.chunk_id = ct.chunk_id
                    WHERE dc.chunk_id LIKE %s
                    GROUP BY vt.id, vt.label
                    ORDER BY chunk_count DESC
                    """,
                    (f"{doc_id}_%",),
                )
                return [
                    {"id": r[0], "label": r[1], "chunk_count": r[2], "avg_score": float(r[3])}
                    for r in cur.fetchall()
                ]
        finally:
            conn.close()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _query)
