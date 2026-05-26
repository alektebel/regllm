"""
SQL DDL extractor.

Parses CREATE TABLE statements, COMMENT ON COLUMN, and FOREIGN KEY relationships
from .sql files. Uses sqlparse for statement-level tokenisation.
Returns (content_text, metadata) tuples ready for embedding and storage.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"([`\"\[]?[\w.]+[`\"\]]?)",
    re.IGNORECASE,
)
_COMMENT_ON_COL_RE = re.compile(
    r"COMMENT\s+ON\s+COLUMN\s+(\w+)\.(\w+)\s+IS\s+'([^']*)'",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_COMMENT_RE = re.compile(r"--\s*(.+)$", re.MULTILINE)
_FK_RE = re.compile(
    r"FOREIGN\s+KEY\s*\((\w+)\)\s+REFERENCES\s+(\w+)\s*\((\w+)\)",
    re.IGNORECASE,
)
_CHECK_IN_RE = re.compile(r"IN\s*\(([^)]+)\)", re.IGNORECASE)


def _clean_name(name: str) -> str:
    return name.strip().strip("`\"[]").split(".")[-1].upper()


def _split_columns(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas that are outside parentheses."""
    depth = 0
    parts: list[str] = []
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def extract_sql(path: Path | str) -> list[tuple[str, dict]]:
    """Parse a SQL file. Returns [(content_text, metadata), ...] per column/relationship."""
    import sqlparse

    path = Path(path)
    sql_text = path.read_text(encoding="utf-8", errors="replace")
    results: list[tuple[str, dict]] = []

    # ── 1. COMMENT ON COLUMN table.col IS '...' (PostgreSQL / Oracle) ─────────
    for m in _COMMENT_ON_COL_RE.finditer(sql_text):
        table_name = _clean_name(m.group(1))
        col_name   = _clean_name(m.group(2))
        desc       = m.group(3).strip()
        content = (
            f"Tabla: {table_name} | Campo: {col_name}\n"
            f"Descripción: {desc}\n"
            f"Fuente: {path.name} (COMMENT ON COLUMN)"
        )
        results.append((content, {
            "source_type": "sql_ddl",
            "table_name": table_name,
            "column_name": col_name,
            "source_file": path.name,
            "confidence": "high",
        }))

    # ── 2. CREATE TABLE — column definitions ──────────────────────────────────
    for stmt in sqlparse.parse(sql_text):
        if stmt.get_type() != "CREATE":
            continue
        flat = str(stmt)
        tbl_m = _CREATE_TABLE_RE.search(flat)
        if not tbl_m:
            continue
        table_name = _clean_name(tbl_m.group(1))

        p_start = flat.find("(")
        p_end   = flat.rfind(")")
        if p_start == -1 or p_end == -1:
            continue
        body = flat[p_start + 1: p_end]

        for part in _split_columns(body):
            upper = part.upper().lstrip()

            # ── FK relationships ───────────────────────────────────────────────
            if "FOREIGN KEY" in upper:
                fk_m = _FK_RE.search(part)
                if fk_m:
                    col_name  = fk_m.group(1).upper()
                    ref_table = fk_m.group(2).upper()
                    ref_col   = fk_m.group(3).upper()
                    content = (
                        f"Tabla: {table_name} | Campo: {col_name}\n"
                        f"Relación: FK → {ref_table}.{ref_col}\n"
                        f"Fuente: {path.name}"
                    )
                    results.append((content, {
                        "source_type": "sql_ddl",
                        "table_name": table_name,
                        "column_name": col_name,
                        "source_file": path.name,
                        "confidence": "high",
                    }))
                continue

            # Skip other table-level constraints
            if any(upper.startswith(kw) for kw in ("PRIMARY", "UNIQUE", "INDEX", "KEY ", "CONSTRAINT", "CHECK(")):
                continue

            # ── Inline comment ─────────────────────────────────────────────────
            inline_comment = ""
            ic_m = _INLINE_COMMENT_RE.search(part)
            if ic_m:
                inline_comment = ic_m.group(1).strip()
                part = part[:ic_m.start()].strip()

            tokens = part.split()
            if not tokens:
                continue
            col_name = _clean_name(tokens[0])
            if not col_name or len(col_name) > 64:
                continue

            dtype = tokens[1] if len(tokens) > 1 else ""
            # Merge type+size e.g. VARCHAR (255)
            if len(tokens) > 2 and "(" in dtype and ")" not in dtype:
                dtype = dtype + tokens[2]

            not_null = "NOT NULL" in part.upper()

            check_m = re.search(r"CHECK\s*\(([^)]+)\)", part, re.IGNORECASE)
            valid_values = None
            if check_m:
                in_m = _CHECK_IN_RE.search(check_m.group(1))
                if in_m:
                    vals = [v.strip().strip("'\"") for v in in_m.group(1).split(",")]
                    valid_values = " | ".join(vals)

            default_m = re.search(r"DEFAULT\s+([^\s,]+)", part, re.IGNORECASE)
            default = default_m.group(1).strip("'\"") if default_m else None

            lines = [f"Tabla: {table_name} | Campo: {col_name}"]
            type_line = f"Tipo: {dtype}" if dtype else ""
            if not_null:
                type_line += " | Requerido: Sí"
            if type_line:
                lines.append(type_line)
            if inline_comment:
                lines.append(f"Descripción: {inline_comment}")
            if valid_values:
                lines.append(f"Valores válidos: {valid_values}")
            if default:
                lines.append(f"Valor por defecto: {default}")
            lines.append(f"Fuente: {path.name} (CREATE TABLE)")

            results.append(("\n".join(lines), {
                "source_type": "sql_ddl",
                "table_name": table_name,
                "column_name": col_name,
                "source_file": path.name,
                "confidence": "medium",
            }))

    logger.info("sql_ddl: %d entries from %s", len(results), path.name)
    return results
