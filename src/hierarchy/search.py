"""
Hierarchical search: combines theme-based retrieval with structural context.

Given a query, this returns ranked chunks annotated with:
  - which validation theme(s) they belong to
  - their structural position (chapter → section → article)
  - cosine similarity to the query
"""

from __future__ import annotations
import logging
import psycopg2
import psycopg2.extras
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class HierarchyResult:
    chunk_id: str
    texto: str
    doc_id: str
    structural_path: str        # e.g. "Chapter 3 > Section 1 > Article 5"
    themes: list[str]           # theme labels matched
    semantic_score: float       # cosine similarity to query (0-1)
    theme_score: float          # best keyword/embedding theme score
    combined_score: float       # weighted combination


class HierarchicalSearch:
    def __init__(self, dsn: str, embed_fn=None):
        """
        dsn: PostgreSQL connection string
        embed_fn: callable(str) → list[float], produces a 768-dim embedding.
                  If None, falls back to keyword-only search.
        """
        self.dsn = dsn
        self.embed_fn = embed_fn

    # ── public API ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        theme_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
        top_k: int = 10,
        semantic_weight: float = 0.6,
    ) -> list[HierarchyResult]:
        """
        Search for chunks relevant to `query`.

        If theme_ids is given, restrict to chunks tagged with those themes.
        If doc_ids is given, restrict to those documents.
        semantic_weight controls how much vector similarity vs theme score matters.
        """
        conn = psycopg2.connect(self.dsn)
        try:
            if self.embed_fn:
                return self._semantic_search(
                    conn, query, theme_ids, doc_ids, top_k, semantic_weight
                )
            else:
                return self._keyword_search(conn, query, theme_ids, doc_ids, top_k)
        finally:
            conn.close()

    def themes_for_doc(self, doc_id: str) -> list[dict]:
        """Return all themes present in a document with chunk counts."""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT vt.id, vt.label, COUNT(ct.chunk_id) AS chunk_count,
                           AVG(ct.score) AS avg_score
                    FROM chunk_themes ct
                    JOIN validation_themes vt ON vt.id = ct.theme_id
                    JOIN document_chunks dc ON dc.chunk_id = ct.chunk_id
                    WHERE dc.chunk_id LIKE %s
                    GROUP BY vt.id, vt.label
                    ORDER BY chunk_count DESC
                    """,
                    (f"{doc_id}%",),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def hierarchy_tree(self, doc_id: str) -> list[dict]:
        """Return the full structural tree for a document as a flat list with depth."""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    WITH RECURSIVE tree AS (
                        SELECT node_id, label, level_type, parent_id, 0 AS depth
                        FROM document_hierarchy
                        WHERE doc_id = %s AND parent_id IS NULL
                        UNION ALL
                        SELECT dh.node_id, dh.label, dh.level_type, dh.parent_id,
                               tree.depth + 1
                        FROM document_hierarchy dh
                        JOIN tree ON dh.parent_id = tree.node_id
                    )
                    SELECT * FROM tree ORDER BY depth, node_id
                    """,
                    (doc_id,),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    # ── internal ─────────────────────────────────────────────────────────────

    def _semantic_search(self, conn, query, theme_ids, doc_ids, top_k, sem_w):
        query_emb = self.embed_fn(query)
        emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"

        theme_filter = ""
        args: list = [emb_str]

        if theme_ids:
            theme_filter = f"AND ct.theme_id = ANY(%s)"
            args.append(theme_ids)

        doc_filter = ""
        if doc_ids:
            doc_filter = "AND dc.chunk_id LIKE ANY(%s)"
            args.append([f"{d}%" for d in doc_ids])

        args += [top_k]

        sql = f"""
            SELECT
                dc.chunk_id,
                dc.texto,
                dc.hierarchy_node_id,
                1 - (dc.embedding <=> %s::vector) AS sem_score,
                MAX(ct.score) AS theme_score,
                ARRAY_AGG(DISTINCT vt.label) AS theme_labels
            FROM document_chunks dc
            LEFT JOIN chunk_themes ct ON ct.chunk_id = dc.chunk_id
            LEFT JOIN validation_themes vt ON vt.id = ct.theme_id
            WHERE dc.embedding IS NOT NULL
              {theme_filter}
              {doc_filter}
            GROUP BY dc.chunk_id, dc.texto, dc.hierarchy_node_id, dc.embedding
            ORDER BY sem_score DESC
            LIMIT %s
        """

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()

        return self._enrich(conn, rows, sem_w)

    def _keyword_search(self, conn, query, theme_ids, doc_ids, top_k):
        theme_filter = ""
        args: list = [query]

        if theme_ids:
            theme_filter = "AND ct.theme_id = ANY(%s)"
            args.append(theme_ids)

        doc_filter = ""
        if doc_ids:
            doc_filter = "AND dc.chunk_id LIKE ANY(%s)"
            args.append([f"{d}%" for d in doc_ids])

        args.append(top_k)

        sql = f"""
            SELECT
                dc.chunk_id,
                dc.texto,
                dc.hierarchy_node_id,
                ts_rank_cd(dc.texto_tsv, plainto_tsquery('spanish', %s)) AS sem_score,
                COALESCE(MAX(ct.score), 0) AS theme_score,
                ARRAY_AGG(DISTINCT vt.label) FILTER (WHERE vt.label IS NOT NULL) AS theme_labels
            FROM document_chunks dc
            LEFT JOIN chunk_themes ct ON ct.chunk_id = dc.chunk_id
            LEFT JOIN validation_themes vt ON vt.id = ct.theme_id
            WHERE dc.texto_tsv @@ plainto_tsquery('spanish', %s)
              {theme_filter}
              {doc_filter}
            GROUP BY dc.chunk_id, dc.texto, dc.hierarchy_node_id
            ORDER BY sem_score DESC
            LIMIT %s
        """
        # plainto_tsquery used twice in WHERE and SELECT
        args.insert(1, query)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()

        return self._enrich(conn, rows, sem_w=0.5)

    def _enrich(self, conn, rows, sem_w: float) -> list[HierarchyResult]:
        """Fetch structural path for each result and build HierarchyResult objects."""
        results = []
        for row in rows:
            path = self._structural_path(conn, row["hierarchy_node_id"])
            doc_id = row["chunk_id"].rsplit("_", 1)[0]

            sem = float(row["sem_score"] or 0)
            thm = float(row["theme_score"] or 0)
            combined = sem_w * sem + (1 - sem_w) * thm

            results.append(HierarchyResult(
                chunk_id=row["chunk_id"],
                texto=row["texto"],
                doc_id=doc_id,
                structural_path=path,
                themes=list(row["theme_labels"] or []),
                semantic_score=sem,
                theme_score=thm,
                combined_score=combined,
            ))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results

    def _structural_path(self, conn, node_id: str | None) -> str:
        if not node_id:
            return ""
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE ancestors AS (
                    SELECT node_id, label, level_type, parent_id
                    FROM document_hierarchy WHERE node_id = %s
                    UNION ALL
                    SELECT dh.node_id, dh.label, dh.level_type, dh.parent_id
                    FROM document_hierarchy dh
                    JOIN ancestors a ON dh.node_id = a.parent_id
                )
                SELECT label FROM ancestors
                WHERE level_type != 'document'
                ORDER BY level_type DESC  -- document < part < chapter < section < article
                """,
                (node_id,),
            )
            labels = [r[0] for r in cur.fetchall()]
        return " > ".join(labels) if labels else node_id
