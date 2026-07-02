"""Verifiable 5-component reward for DQ check generation.

All components are program-executed (no LLM judge):

  r_parse      — SQLite EXPLAIN succeeds on the emitted SQL
  r_template   — JSON shape valid, SELECT/FROM/WHERE present, severity &
                 category populated, query references the rule's table
  r_coherence  — WHERE references ≥ 2 columns (or query has a JOIN); this
                 is the defining property of a coherence check vs a range
                 check — without it the model is rewarded 0 here
  r_clean_zero — query returns 0 rows on the coherent toy DB (specificity)
  r_catches    — query returns ≥ 1 row on a DB with one injected violation
                 (mutation testing — proves the check actually catches the
                 incoherence it claims to)

Total is shifted to [-1, 1] for GRPO, matching training/rl_env.py:776.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from training.dq.coherence_rules import (
    CoherenceRule, PK_COLUMN, TABLE_NAME, load_toy_db,
)

# ── Reward weights (sum to 1.0) ─────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "r_parse": 0.15,
    "r_template": 0.20,
    "r_coherence": 0.20,
    "r_clean_zero": 0.20,
    "r_catches": 0.25,
}

_VALID_SEVERITY = {"HIGH", "MED", "LOW"}
_VALID_CATEGORY = {"cross_field", "cross_table", "cardinality"}
_SQL_KEYWORDS = {
    "AND", "OR", "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "SELECT",
    "FROM", "WHERE", "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE",
    "CAST", "AS", "DISTINCT", "TRUE", "FALSE", "ON", "JOIN", "INNER",
    "LEFT", "RIGHT", "OUTER", "GROUP", "ORDER", "HAVING", "BY", "LIMIT",
    "UNION", "ALL", "EXISTS",
}

_QUERY_TIMEOUT_S = 3.0
_QUERY_MAX_ROWS = 1000


# ── DB caches (one clean DB + one trap DB per rule) ─────────────────────

_clean_db_cache: sqlite3.Connection | None = None
_trap_cache: dict[str, sqlite3.Connection] = {}


def _clean_db() -> sqlite3.Connection:
    global _clean_db_cache
    if _clean_db_cache is None:
        _clean_db_cache = load_toy_db()
    return _clean_db_cache


def _trap_db(rule: CoherenceRule) -> sqlite3.Connection:
    """A copy of the clean DB plus ONE row mutated by the rule.

    The dirty row is appended last and differs from any clean row on the
    rule's mutated columns, so any non-empty query result on this DB is
    causally attributable to the injected violation (combined with
    ``r_clean_zero`` on the clean DB, this is unambiguous).
    """
    if rule.rule_id in _trap_cache:
        return _trap_cache[rule.rule_id]
    src = _clean_db()
    cur = src.execute(f'SELECT * FROM "{rule.table}" LIMIT 1')
    sample = cur.fetchone()
    if sample is None:
        # Degenerate toy DB — fall back to the clean DB.
        _trap_cache[rule.rule_id] = src
        return src
    clean_row = dict(sample)
    dirty_row = rule.mutate(clean_row)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    src.backup(conn)
    cols = list(clean_row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    # Force a fresh PK so the dirty row never collides with the control row.
    dirty_row[PK_COLUMN] = f"__DIRTY_{rule.rule_id}__"
    conn.execute(
        f'INSERT INTO "{rule.table}" ({", ".join(cols)}) VALUES ({placeholders})',
        [dirty_row[c] for c in cols],
    )
    conn.commit()
    _trap_cache[rule.rule_id] = conn
    return conn


# ── Output extraction ───────────────────────────────────────────────────

def _iter_json_objects(text: str):
    """Yield every balanced ``{...}`` substring parsed as JSON."""
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        quote = ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in ("'", '"'):
                    in_str, esc, quote = True, False, ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break


def extract_check(completion: str) -> dict | None:
    """Pull the first well-formed check JSON from the completion.

    Order of preference:
      1. ``<check>{...}</check>`` tag (the offered template).
      2. Any balanced JSON object carrying an ``sql`` key.
      3. A ```sql ... ``` fence fallback (returns minimal metadata).
    """
    for m in re.finditer(r"<check>\s*(\{.*?\})\s*</check>", completion, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and isinstance(obj.get("sql"), str):
                return obj
        except json.JSONDecodeError:
            pass
    for obj in _iter_json_objects(completion):
        if isinstance(obj, dict) and isinstance(obj.get("sql"), str):
            return obj
    for m in re.finditer(r"```(?:sql)?\s*(.*?)```", completion, re.DOTALL | re.IGNORECASE):
        body = m.group(1).strip().rstrip(";").strip()
        if body.upper().startswith("SELECT"):
            return {
                "sql": body, "name": "", "description": "",
                "severity": "MED", "category": "cross_field",
            }
    return None


# ── Component scorers ───────────────────────────────────────────────────

class _Budget:
    """sqlite progress handler that aborts once the wall-clock budget is up."""

    def __init__(self, seconds: float) -> None:
        self._deadline = time.time() + seconds

    def __call__(self) -> int:
        return 1 if time.time() > self._deadline else 0


def _safe_run(conn: sqlite3.Connection, sql: str, *, limit: int = _QUERY_MAX_ROWS):
    """SELECT-only execution with a wall-clock + row budget.

    Returns ``(columns, rows)`` on success, raises ``sqlite3.Error`` on any
    failure (parse error, timeout, non-SELECT). Mirrors the hardening in
    ``training/rl_env.py:338``.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        raise sqlite3.OperationalError("not a SELECT")
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        stripped = f"{stripped} LIMIT {limit}"
    handler = _Budget(_QUERY_TIMEOUT_S)
    conn.set_progress_handler(handler, 10000)
    try:
        cursor = conn.execute(stripped)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        rows = [dict(zip(cols, r)) for r in cursor.fetchmany(limit)]
    finally:
        conn.set_progress_handler(None, 0)
    return cols, rows


def _score_template(check: dict, rule: CoherenceRule) -> float:
    sql_upper = str(check.get("sql", "")).upper()
    has_from = re.search(r"\bFROM\b", sql_upper) is not None
    has_where = re.search(r"\bWHERE\b", sql_upper) is not None
    severity_ok = str(check.get("severity", "")).upper() in _VALID_SEVERITY
    category_ok = str(check.get("category", "")).lower() in _VALID_CATEGORY
    score = 0.25 * has_from + 0.25 * has_where + 0.25 * severity_ok + 0.25 * category_ok
    # Halve credit when the SQL doesn't even mention the rule's table — the
    # model is checking the wrong entity.
    if rule.table.upper() not in sql_upper:
        score *= 0.5
    return score


def _score_coherence(check: dict, rule: CoherenceRule) -> float:
    """Coherence ⟺ WHERE combines ≥ 2 columns OR query has a JOIN."""
    sql = str(check.get("sql", ""))
    has_join = re.search(r"\bJOIN\b", sql, re.IGNORECASE) is not None
    where_match = re.search(
        r"\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|;|$)",
        sql, re.IGNORECASE | re.DOTALL,
    )
    where_clause = where_match.group(1) if where_match else ""
    refs = set(re.findall(r"\b([A-Z_][A-Z0-9_]{2,})\b", where_clause.upper()))
    refs -= _SQL_KEYWORDS
    rule_cols = {c.upper() for c in rule.columns}
    if has_join or len(refs & rule_cols) >= 2:
        return 1.0
    if len(refs) >= 2:
        return 0.5                  # multi-column but off-target
    return 0.0


# ── Public API ──────────────────────────────────────────────────────────

@dataclass
class DQRewardBreakdown:
    total: float
    components: dict[str, float]
    check: dict | None
    error: str | None = None


def reset_caches() -> None:
    """Clear DB caches (use between independent test runs)."""
    global _clean_db_cache
    _clean_db_cache = None
    _trap_cache.clear()


def score_check(check: dict | None, rule: CoherenceRule) -> DQRewardBreakdown:
    components: dict[str, float] = {k: 0.0 for k in WEIGHTS}

    if not check or not isinstance(check.get("sql"), str) or not check["sql"].strip():
        return _finalize(components, check, error="no SQL in completion")

    sql = check["sql"].strip().rstrip(";").strip()
    if not sql.upper().startswith("SELECT"):
        return _finalize(components, check, error="not a SELECT")

    # r_parse — EXPLAIN must succeed.
    try:
        _clean_db().execute(f"EXPLAIN {sql}").fetchall()
        components["r_parse"] = 1.0
    except sqlite3.Error as e:
        return _finalize(components, check, error=f"parse: {e}")

    # r_template, r_coherence — syntactic/structural, no execution.
    components["r_template"] = _score_template(check, rule)
    components["r_coherence"] = _score_coherence(check, rule)

    # r_clean_zero — query must return 0 rows on coherent data.
    try:
        _, clean_rows = _safe_run(_clean_db(), sql)
        components["r_clean_zero"] = 1.0 if len(clean_rows) == 0 else 0.0
    except sqlite3.Error as e:
        return _finalize(components, check, error=f"clean_exec: {e}")

    # r_catches — query must return ≥ 1 row on the mutated DB. Combined
    # with r_clean_zero this is unambiguous: the only new row in trap_db
    # is the injected violation.
    try:
        _, trap_rows = _safe_run(_trap_db(rule), sql)
        components["r_catches"] = 1.0 if len(trap_rows) >= 1 else 0.0
    except sqlite3.Error as e:
        return _finalize(components, check, error=f"trap_exec: {e}")

    return _finalize(components, check)


def score_completion(completion: str, rule: CoherenceRule) -> DQRewardBreakdown:
    """End-to-end: extract check from raw completion, then score."""
    return score_check(extract_check(completion), rule)


def grpo_reward(rule: CoherenceRule):
    """Return a reward_fn suitable for TRL's GRPOTrainer (closes over rule)."""
    call_count = [0]

    def _fn(completions: list[str], **_kwargs) -> list[float]:
        out: list[float] = []
        call_count[0] += 1
        for i, text in enumerate(completions):
            bd = score_completion(text, rule)
            out.append(bd.total)
            if call_count[0] <= 3 or i == 0:
                preview = text[:160].replace("\n", " ")
                print(f"    [call {call_count[0]}] r={bd.total:.3f} "
                      f"comps={bd.components} err={bd.error}")
                print(f"    preview: {preview}...")
        return out

    return _fn


# ── Helpers ─────────────────────────────────────────────────────────────

def _finalize(components: dict[str, float], check: dict | None,
              error: str | None = None) -> DQRewardBreakdown:
    raw = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return DQRewardBreakdown(
        total=round(2.0 * raw - 1.0, 4),           # shift to [-1, 1] for GRPO
        components={k: round(v, 4) for k, v in components.items()},
        check=check,
        error=error,
    )
