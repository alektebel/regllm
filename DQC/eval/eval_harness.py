"""DQC eval harness — assess the quality of the data-quality checks produced
by the DQC agent against the 7-layer CICLOS_CALIBRADOS stress database.

It can detect **model deficiencies** by reporting recall/precision broken down
by data-quality dimension (DAMA / ISO 8000 / BCBS 239): a dimension with low
recall is an area where the agent (or trained model) fails to produce adequate
checks.

Three modes (all program-executed, no LLM judge):

  1. ``--selftest``      — validate every oracle (0 on clean, >=1 on its trap).
  2. ``--sql FILE``      — score a hand-supplied list of SQL checks against the
                           whole defect catalog (recall / precision / per-dim).
  3. ``--agent URL``     — call the DQC agent (``POST /dqc/generate``) once per
                           defect description and score the returned DQCs.

Each candidate SQL is scored with the verifiable 5-component reward (mirrors
``training/dq/dq_reward.py``): r_parse, r_template, r_coherence, r_clean_zero,
r_catches. The harness reuses the clean + per-defect trap DBs from
``generate_db``.

Example:

    python DQC/eval/eval_harness.py --selftest
    python DQC/eval/eval_harness.py --sql DQC/eval/example_checks.sql
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from defect_catalog import (  # noqa: E402
    DEFECTS, DIMENSIONS, PK_COLUMN, TABLE_NAME, defects_by_dimension,
)
from generate_db import (  # noqa: E402
    build_clean_conn, build_mixed, build_trap, build_trap_conn,
)

# Planted violations per defect. A check must detect the *invariant*, not
# memorise a single row, so sensitivity is sampled at TRAP_K points of the
# violation space and r_catches is the fraction caught.
TRAP_K = 3
# A single coherence check legitimately spans 2-3 related invariants (a floor
# breach usually implies the max() was skipped); firing on more distinct
# defects than this is a tautology smell ("overbroad").
OVERBROAD_THRESHOLD = 3

# ── reward weights (sum to 1.0); match training/dq/dq_reward.py ────────────
WEIGHTS = {"r_parse": 0.15, "r_template": 0.20, "r_coherence": 0.20,
           "r_clean_zero": 0.20, "r_catches": 0.25}

_SQL_KEYWORDS = {
    "AND", "OR", "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "SELECT",
    "FROM", "WHERE", "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE",
    "CAST", "AS", "DISTINCT", "TRUE", "FALSE", "ON", "JOIN", "INNER",
    "LEFT", "RIGHT", "OUTER", "GROUP", "ORDER", "HAVING", "BY", "LIMIT",
    "UNION", "ALL", "EXISTS", "ABS", "SQRT", "MAX", "MIN", "SUM", "COUNT",
    "AVG",
}
_TABLE_ALIASES = {
    "ciclos_recuperacion": TABLE_NAME,
    "recuperatory_cycles": TABLE_NAME,
    "mylib.ciclos_recuperacion": TABLE_NAME,
    "mylib.ciclos_calibrados": TABLE_NAME,
}
_QUERY_TIMEOUT_S = 3.0
_QUERY_MAX_ROWS = 1000


# ── SQL normalisation & safe execution ─────────────────────────────────────

def normalize_sql(sql: str) -> str:
    """Map agent table aliases to the eval table and strip trailing ';'."""
    s = sql.strip().rstrip(";").strip()
    for alias, real in _TABLE_ALIASES.items():
        s = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
                   real, s)
    return s


class _Budget:
    def __init__(self, seconds: float) -> None:
        self._deadline = time.time() + seconds

    def __call__(self) -> int:
        return 1 if time.time() > self._deadline else 0


def _safe_run(conn, sql: str, limit: int = _QUERY_MAX_ROWS):
    stripped = normalize_sql(sql)
    if not stripped.upper().startswith("SELECT"):
        raise sqlite3.OperationalError("not a SELECT")
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        stripped = f"{stripped} LIMIT {limit}"
    conn.set_progress_handler(_Budget(_QUERY_TIMEOUT_S), 10000)
    try:
        cur = conn.execute(stripped)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(limit)]
    finally:
        conn.set_progress_handler(None, 0)
    return cols, rows


# ── 5-component scoring ────────────────────────────────────────────────────

@dataclass
class Score:
    sql: str
    components: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in WEIGHTS})
    total: float = 0.0
    error: str | None = None

    @property
    def catches(self) -> bool:
        """At least one planted violation detected."""
        return self.components["r_catches"] > 0.0

    @property
    def clean(self) -> bool:
        return self.components["r_clean_zero"] >= 1.0


def _score_template(sql: str) -> float:
    u = sql.upper()
    score = 0.34 * bool(re.search(r"\bFROM\b", u)) + 0.33 * bool(re.search(r"\bWHERE\b", u))
    if any(a.upper() in u for a in _TABLE_ALIASES):
        score += 0.33
    return round(min(1.0, score), 4)


def _score_coherence(sql: str) -> float:
    has_join = re.search(r"\bJOIN\b", sql, re.IGNORECASE) is not None
    m = re.search(r"\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|;|$)",
                  sql, re.IGNORECASE | re.DOTALL)
    where = m.group(1) if m else ""
    refs = set(re.findall(r"\b([A-Z_][A-Z0-9_]{2,})\b", where.upper())) - _SQL_KEYWORDS
    if has_join or len(refs) >= 2:
        return 1.0
    return 0.0


def _caught_fraction(cols: list[str], rows: list[dict],
                     planted_pks: list[str] | None) -> float:
    """Fraction of planted violations a result set detects.

    When the check projects the PK we attribute precisely; otherwise we fall
    back to row counting (only sound for clean-zero checks, where every trap
    row returned is necessarily a planted one).
    """
    if not planted_pks:
        return 1.0 if rows else 0.0
    if PK_COLUMN in cols:
        got = {r.get(PK_COLUMN) for r in rows}
        return len(got & set(planted_pks)) / len(planted_pks)
    return min(len(rows), len(planted_pks)) / len(planted_pks)


def score_check(sql: str, defect, clean_conn, trap_conn,
                planted_pks: list[str] | None = None) -> Score:
    sc = Score(sql=sql)
    norm = normalize_sql(sql)
    if not norm.upper().startswith("SELECT"):
        sc.error = "not a SELECT"; return sc
    # r_parse
    try:
        clean_conn.execute(f"EXPLAIN {norm}").fetchall()
        sc.components["r_parse"] = 1.0
    except sqlite3.Error as e:
        sc.error = f"parse: {e}"; sc.total = _wsum(sc.components); return sc
    sc.components["r_template"] = _score_template(norm)
    sc.components["r_coherence"] = _score_coherence(norm)
    # r_clean_zero
    try:
        _, clean_rows = _safe_run(clean_conn, sql)
        sc.components["r_clean_zero"] = 1.0 if len(clean_rows) == 0 else 0.0
    except sqlite3.Error as e:
        sc.error = f"clean_exec: {e}"; sc.total = _wsum(sc.components); return sc
    # r_catches — fraction of the planted violations detected
    try:
        cols, trap_rows = _safe_run(trap_conn, sql)
        sc.components["r_catches"] = round(
            _caught_fraction(cols, trap_rows, planted_pks), 4)
    except sqlite3.Error as e:
        sc.error = f"trap_exec: {e}"; sc.total = _wsum(sc.components); return sc
    sc.total = _wsum(sc.components)
    return sc


def _wsum(components: dict[str, float]) -> float:
    return round(sum(WEIGHTS[k] * components[k] for k in WEIGHTS), 4)


# ── oracle self-test ───────────────────────────────────────────────────────

def selftest(rows: int = 1500, seed: int = 42, k: int = TRAP_K) -> dict:
    """Verify every oracle: 0 rows on clean, ALL k planted rows on its trap.

    Also reports the *oracle overlap matrix* on a mixed DB (which oracles fire
    on other defects' planted rows) — expected for nested invariants (a floor
    breach implies the max() was skipped) but should be visible, not silent.
    """
    clean = build_clean_conn(rows, seed)
    report = {"passed": 0, "failed": 0, "failures": [], "overlap": {}}
    for d in DEFECTS:
        try:
            n_clean = len(_safe_run(clean, d.oracle_sql)[1])
        except sqlite3.Error as e:
            report["failed"] += 1
            report["failures"].append({"defect": d.defect_id, "stage": "clean", "error": str(e)})
            continue
        trap, planted = build_trap(d, rows, seed, k=k)
        try:
            cols, trap_rows = _safe_run(trap, d.oracle_sql)
        except sqlite3.Error as e:
            report["failed"] += 1
            report["failures"].append({"defect": d.defect_id, "stage": "trap", "error": str(e)})
            continue
        frac = _caught_fraction(cols, trap_rows, planted)
        if n_clean == 0 and frac >= 1.0:
            report["passed"] += 1
        else:
            report["failed"] += 1
            report["failures"].append(
                {"defect": d.defect_id, "clean_rows": n_clean,
                 "caught_fraction": round(frac, 3), "dimension": d.dimension})

    # Oracle overlap on the mixed DB (informational, never a failure)
    mixed, planted_map = build_mixed(DEFECTS, rows, seed, k=k)
    pk_to_defect = {pk: did for did, pks in planted_map.items() for pk in pks}
    for d in DEFECTS:
        try:
            cols, mixed_rows = _safe_run(mixed, d.oracle_sql)
        except sqlite3.Error:
            continue
        if PK_COLUMN not in cols:
            continue
        others = sorted({
            pk_to_defect[r[PK_COLUMN]] for r in mixed_rows
            if r.get(PK_COLUMN) in pk_to_defect
            and pk_to_defect[r[PK_COLUMN]] != d.defect_id
        })
        if others:
            report["overlap"][d.defect_id] = others
    return report


# ── coverage of a free-form set of checks against the catalog ──────────────

@dataclass
class CoverageReport:
    n_checks: int
    n_valid: int                      # parse + clean
    precision: float                  # valid / n_checks
    recall: float                     # defects caught / total coherence defects
    decoy_caught: int                 # checks firing on a decoy trap (bad if coherence claim)
    overbroad_checks: int             # checks firing on > OVERBROAD_THRESHOLD defects
    per_dimension: dict[str, dict]    # dim -> {recall, caught, total}
    per_article: dict[str, dict]      # regulation_ref -> {recall, caught, total}
    uncovered_articles: list[str]     # regulation refs with zero caught defects
    caught_defects: list[str]
    missed_defects: list[str]
    per_check: list[dict]


def _per_article(caught: set[str]) -> tuple[dict[str, dict], list[str]]:
    """Recall grouped by regulation_ref over non-decoy defects."""
    per: dict[str, dict] = {}
    for d in DEFECTS:
        if d.decoy or not d.regulation_ref:
            continue
        st = per.setdefault(d.regulation_ref, {"caught": 0, "total": 0, "recall": 0.0})
        st["total"] += 1
        if d.defect_id in caught:
            st["caught"] += 1
    for st in per.values():
        st["recall"] = round(st["caught"] / st["total"], 3) if st["total"] else 0.0
    uncovered = sorted(ref for ref, st in per.items() if st["caught"] == 0)
    return per, uncovered


def coverage(checks: list[str], rows: int = 1500, seed: int = 42,
             k: int = TRAP_K) -> CoverageReport:
    """Score a list of SQL checks against the whole catalog on a MIXED DB.

    All defects are planted simultaneously (production-like) and hits are
    attributed by planted PK, so one pass per check replaces one pass per
    (check × defect). A check *covers* a defect iff it is clean on the clean
    DB and returns >= 1 of that defect's planted rows on the mixed DB.
    """
    clean = build_clean_conn(rows, seed)
    mixed, planted_map = build_mixed(DEFECTS, rows, seed, k=k)
    pk_to_defect = {pk: did for did, pks in planted_map.items() for pk in pks}
    decoy_ids = {d.defect_id for d in DEFECTS if d.decoy}

    per_dim = {dim: {"caught": 0, "total": 0, "recall": 0.0}
               for dim in DIMENSIONS}
    caught, missed, decoy_caught, overbroad = [], [], 0, 0
    per_check = []
    n_valid = 0

    for sql in checks:
        norm = normalize_sql(sql)
        if not norm.upper().startswith("SELECT"):
            per_check.append({"sql": sql[:120], "valid": False, "error": "not SELECT"})
            continue
        try:
            clean_ok = len(_safe_run(clean, sql)[1]) == 0
        except sqlite3.Error:
            clean_ok = False
        if clean_ok:
            n_valid += 1
        hits: dict[str, int] = {}
        if clean_ok:
            try:
                cols, rows_mixed = _safe_run(mixed, sql)
            except sqlite3.Error:
                cols, rows_mixed = [], []
            if PK_COLUMN in cols:
                for r in rows_mixed:
                    did = pk_to_defect.get(r.get(PK_COLUMN))
                    if did:
                        hits[did] = hits.get(did, 0) + 1
        hit_ids = sorted(hits)
        decoy_caught += sum(1 for h in hit_ids if h in decoy_ids)
        n_coherence_hits = sum(1 for h in hit_ids if h not in decoy_ids)
        is_overbroad = n_coherence_hits > OVERBROAD_THRESHOLD
        overbroad += int(is_overbroad)
        per_check.append({
            "sql": sql[:120], "valid": clean_ok, "hits": hit_ids,
            "caught_fraction": {
                did: round(n / len(planted_map[did]), 3) for did, n in hits.items()
            },
            "overbroad": is_overbroad,
        })

    # recall over coherence (non-decoy) defects
    for d in DEFECTS:
        if d.decoy:
            continue
        per_dim[d.dimension]["total"] += 1
        if any(d.defect_id in pc.get("hits", []) for pc in per_check):
            per_dim[d.dimension]["caught"] += 1
            caught.append(d.defect_id)
        else:
            missed.append(d.defect_id)
    for dim, st in per_dim.items():
        st["recall"] = round(st["caught"] / st["total"], 3) if st["total"] else 0.0
    coherence_total = sum(1 for d in DEFECTS if not d.decoy)
    recall = round(len(caught) / coherence_total, 3) if coherence_total else 0.0
    per_article, uncovered = _per_article(set(caught))

    return CoverageReport(
        n_checks=len(checks), n_valid=n_valid,
        precision=round(n_valid / len(checks), 3) if checks else 0.0,
        recall=recall, decoy_caught=decoy_caught, overbroad_checks=overbroad,
        per_dimension=per_dim, per_article=per_article,
        uncovered_articles=uncovered,
        caught_defects=caught, missed_defects=missed,
        per_check=per_check,
    )


# ── agent integration (optional) ───────────────────────────────────────────

def _parse_sql_file(path: Path) -> list[str]:
    """One SQL check per ;-separated block. Full-line ``--`` comments dropped."""
    lines = [ln for ln in path.read_text().splitlines()
             if not ln.strip().startswith("--")]
    text = "\n".join(lines)
    return [c.strip() for c in text.split(";") if c.strip()]


def call_agent(base_url: str, message: str, timeout: float = 90.0) -> list[str]:
    """POST /dqc/generate and return the list of SQL rules."""
    payload = json.dumps({"message": message, "session_id": "eval"}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/dqc/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return [d.get("regla_sql", "") for d in data.get("dqcs", []) if d.get("regla_sql")]


def run_agent_eval(base_url: str, rows: int = 1500, seed: int = 42,
                   k: int = TRAP_K) -> dict:
    """Targeted mode: ask the agent for a check per defect, score each."""
    clean = build_clean_conn(rows, seed)
    results = []
    for d in DEFECTS:
        try:
            sqls = call_agent(base_url, d.description)
        except Exception as e:
            results.append({"defect": d.defect_id, "dimension": d.dimension,
                            "error": f"agent: {e}"})
            continue
        best = None
        trap, planted = build_trap(d, rows, seed, k=k)
        for sql in sqls:
            s = score_check(sql, d, clean, trap, planted_pks=planted)
            if best is None or s.total > best.total:
                best = s
        results.append({
            "defect": d.defect_id, "dimension": d.dimension, "severity": d.severity,
            "regulation_ref": d.regulation_ref,
            "n_dqcs": len(sqls), "best_total": best.total if best else 0.0,
            "components": best.components if best else {},
            "caught": bool(best and best.catches and best.clean),
        })
    # per-dimension aggregation
    by_dim = {dim: {"caught": 0, "total": 0, "recall": 0.0, "mean_total": 0.0}
              for dim in DIMENSIONS}
    for r in results:
        dim = r["dimension"]
        by_dim[dim]["total"] += 1
        if r.get("caught"):
            by_dim[dim]["caught"] += 1
        by_dim[dim]["mean_total"] += r.get("best_total", 0.0)
    for dim, st in by_dim.items():
        st["recall"] = round(st["caught"] / st["total"], 3) if st["total"] else 0.0
        st["mean_total"] = round(st["mean_total"] / st["total"], 4) if st["total"] else 0.0
    caught_ids = {r["defect"] for r in results if r.get("caught")}
    per_article, uncovered = _per_article(caught_ids)
    return {"per_defect": results, "per_dimension": by_dim,
            "per_article": per_article, "uncovered_articles": uncovered}


# ── deficiency summary ─────────────────────────────────────────────────────

def deficiency_summary(per_dim: dict[str, dict], threshold: float = 0.5) -> list[str]:
    """Dimensions where recall < threshold are flagged as model deficiencies."""
    return [dim for dim, st in per_dim.items()
            if st.get("total", 0) > 0 and st.get("recall", 1.0) < threshold]


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_selftest(rep: dict) -> None:
    print(f"SELF-TEST: {rep['passed']} passed, {rep['failed']} failed")
    for f in rep["failures"]:
        print(f"  ✗ {f}")
    if rep.get("overlap"):
        print("\nOracle overlap on mixed DB (expected for nested invariants):")
        for did, others in sorted(rep["overlap"].items()):
            print(f"  {did} also fires on rows planted by: {', '.join(others)}")


def _print_coverage(rep: CoverageReport) -> None:
    print(f"CHECKS: {rep.n_checks} supplied, {rep.n_valid} valid+clean "
          f"(precision {rep.precision:.0%})")
    print(f"RECALL (coherence defects): {rep.recall:.0%} "
          f"({len(rep.caught_defects)}/{len(rep.caught_defects)+len(rep.missed_defects)})")
    if rep.decoy_caught:
        print(f"  ! {rep.decoy_caught} check(s) fired on a single-column DECOY "
              "(over-claiming coherence)")
    if rep.overbroad_checks:
        print(f"  ! {rep.overbroad_checks} check(s) fired on >{OVERBROAD_THRESHOLD} "
              "distinct defects (tautology smell)")
    print("\nPer-dimension recall (low = model deficiency):")
    for dim in DIMENSIONS:
        st = rep.per_dimension.get(dim, {"recall": 0, "caught": 0, "total": 0})
        if st["total"]:
            flag = "  ← DEFICIENT" if st["recall"] < 0.5 else ""
            print(f"  {dim:14s} {st['recall']:>5.0%}  ({st['caught']}/{st['total']}){flag}")
    print("\nPer-article recall (regulation_ref of the planted defects):")
    for ref, st in sorted(rep.per_article.items()):
        flag = "  ← UNCOVERED" if st["caught"] == 0 else ""
        print(f"  {ref:45.45s} {st['recall']:>5.0%}  ({st['caught']}/{st['total']}){flag}")
    if rep.missed_defects:
        print(f"\nMissed defects: {', '.join(rep.missed_defects)}")


def _print_agent(rep: dict) -> None:
    caught = sum(1 for r in rep["per_defect"] if r.get("caught"))
    total = len(rep["per_defect"])
    print(f"AGENT RECALL: {caught}/{total} defects caught")
    print("\nPer-dimension recall (low = model deficiency):")
    for dim in DIMENSIONS:
        st = rep["per_dimension"].get(dim, {"recall": 0, "caught": 0, "total": 0})
        if st["total"]:
            flag = "  ← DEFICIENT" if st["recall"] < 0.5 else ""
            print(f"  {dim:14s} {st['recall']:>5.0%}  mean_reward={st['mean_total']:.2f}{flag}")
    defs = deficiency_summary(rep["per_dimension"])
    if defs:
        print(f"\n⚠ Model deficient on: {', '.join(defs)}")
    if rep.get("uncovered_articles"):
        print(f"⚠ Articles with zero coverage: {', '.join(rep['uncovered_articles'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DQC eval harness")
    ap.add_argument("--selftest", action="store_true", help="validate all oracles")
    ap.add_argument("--sql", type=Path, help="file of SQL checks to score (free-form)")
    ap.add_argument("--agent", type=str, help="base URL of running API for targeted eval")
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=TRAP_K,
                    help="planted violations per defect (default %(default)s)")
    ap.add_argument("--fail-under", type=float, default=None, metavar="RECALL",
                    help="exit 1 if coherence-defect recall is below this "
                         "fraction (CI gate for --sql / --agent modes)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    if args.selftest:
        rep = selftest(args.rows, args.seed, args.k)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            _print_selftest(rep)
        sys.exit(0 if rep["failed"] == 0 else 1)

    if args.sql:
        checks = _parse_sql_file(args.sql)
        rep = coverage(checks, args.rows, args.seed, args.k)
        if args.json:
            print(json.dumps({k: v for k, v in asdict(rep).items()}, indent=2))
        else:
            _print_coverage(rep)
        if args.fail_under is not None and rep.recall < args.fail_under:
            print(f"\nFAIL: recall {rep.recall:.0%} < required {args.fail_under:.0%}")
            sys.exit(1)
        return

    if args.agent:
        rep = run_agent_eval(args.agent, args.rows, args.seed, args.k)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            _print_agent(rep)
        if args.fail_under is not None:
            caught = sum(1 for r in rep["per_defect"] if r.get("caught"))
            rec = caught / len(rep["per_defect"]) if rep["per_defect"] else 0.0
            if rec < args.fail_under:
                print(f"\nFAIL: recall {rec:.0%} < required {args.fail_under:.0%}")
                sys.exit(1)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
