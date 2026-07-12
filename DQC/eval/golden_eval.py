"""Golden-trace eval harness for the DQC generator.

While ``eval_harness.py`` measures *defect recall* (mutation testing), this
harness measures **end-to-end behaviour against hand-written golden traces**:
realistic user requests paired with the expected outcome — which planted
defects the produced checks must catch, which articles they must cite, which
they must NOT invent, and which quality constraints (coherence, no decoy
over-claiming) they must respect.

Every trace is scored on four criteria, then aggregated into the 8-dimension
data-quality rubric (DAMA / ISO 8000 / BCBS 239):

  c_valid      (0.15)  ≥1 produced check parses and returns 0 rows on the
                       clean DB (for adversarial traces expecting silence:
                       producing nothing IS the valid answer)
  c_catches    (0.45)  fraction of the trace's expected defects caught by a
                       clean check (≥1 planted row on that defect's trap)
  c_grounding  (0.20)  cited references include the expected article AND no
                       cited reference is a hallucination (i.e. absent from
                       the union of dictionary + catalog references)
  c_coherence  (0.20)  when the trace forbids single-column answers, at
                       least one catching check references ≥2 columns

A trace *passes* when all four criteria are fully satisfied. The rubric
report gives mean score + pass rate per dimension; ``--fail-under`` gates CI.

Modes:
    --selftest            score the gold answers themselves (must be 100%)
    --candidates FILE     score a JSON file {trace_id: [dqc, ...]} — e.g.
                          captured /dqc/generate responses
    --agent URL           call POST {URL}/dqc/generate per trace and score
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from defect_catalog import DEFECTS, DEFECTS_BY_ID, DIMENSIONS  # noqa: E402
from generate_db import build_clean_conn, build_trap  # noqa: E402
from eval_harness import (  # noqa: E402
    _caught_fraction, _safe_run, _score_coherence, call_agent, normalize_sql,
)
from coverage_matrix import canon_ref, parse_dictionary, split_refs  # noqa: E402

TRACES_PATH = HERE / "golden_traces.json"
ROWS = 400
TRAP_K = 2

WEIGHTS = {"c_valid": 0.15, "c_catches": 0.45,
           "c_grounding": 0.20, "c_coherence": 0.20}

_NO_REF_MARKERS = ("sin referencia", "no disponible", "n/a")


def load_traces(path: Path = TRACES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["traces"]


def allowed_refs() -> set[str]:
    """Canonical set of legitimate references: dictionary ∪ defect catalog."""
    out: set[str] = set()
    for refs in parse_dictionary().values():
        out.update(canon_ref(r) for r in refs)
    for d in DEFECTS:
        out.update(canon_ref(r) for r in split_refs(d.regulation_ref))
    return {r for r in out if r}


def _ref_is_known(cited: str, allowed: set[str]) -> bool:
    """A cited reference is legitimate if it matches (or is a prefix/extension
    of) something in the allowed corpus. 'Sin referencia' is always fine."""
    low = cited.strip().lower()
    if not low or any(m in low for m in _NO_REF_MARKERS):
        return True
    for part in split_refs(cited):
        c = canon_ref(part)
        if not c:
            continue
        if not any(c in a or a in c for a in allowed):
            return False
    return True


@dataclass
class TraceScore:
    trace_id: str
    dimension: str
    difficulty: str
    components: dict[str, float] = field(
        default_factory=lambda: {k: 0.0 for k in WEIGHTS})
    total: float = 0.0
    passed: bool = False
    hallucinated_refs: list[str] = field(default_factory=list)
    caught: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class GoldenEvaluator:
    """Scores candidate DQC sets against the golden traces."""

    def __init__(self, rows: int = ROWS, seed: int = 42, k: int = TRAP_K):
        self.rows, self.seed, self.k = rows, seed, k
        self.clean = build_clean_conn(rows, seed)
        self.allowed = allowed_refs()
        self._traps: dict[str, tuple] = {}

    def _trap(self, defect_id: str):
        if defect_id not in self._traps:
            self._traps[defect_id] = build_trap(
                DEFECTS_BY_ID[defect_id], self.rows, self.seed, k=self.k)
        return self._traps[defect_id]

    def _clean_zero(self, sql: str) -> bool:
        try:
            return len(_safe_run(self.clean, sql)[1]) == 0
        except Exception:
            return False

    def score_trace(self, trace: dict, dqcs: list[dict]) -> TraceScore:
        ts = TraceScore(trace["trace_id"], trace["dimension"],
                        trace.get("difficulty", "core"))
        exp = trace.get("expects", {})
        expected_defects = exp.get("defects", [])
        sqls = [d.get("regla_sql", "") for d in dqcs if d.get("regla_sql")]
        clean_sqls = [s for s in sqls if self._clean_zero(s)]

        # c_valid — for silence-expecting traces, no checks is the answer
        if exp.get("max_checks") == 0:
            ts.components["c_valid"] = 1.0 if (not sqls or clean_sqls) else 0.0
            if sqls and not clean_sqls:
                ts.notes.append("produced checks that fire on the clean DB")
        else:
            ts.components["c_valid"] = 1.0 if clean_sqls else 0.0

        # c_catches — per expected defect, does some clean check catch it?
        catching_sqls: set[str] = set()
        if expected_defects:
            for did in expected_defects:
                trap, planted = self._trap(did)
                hit = False
                for sql in clean_sqls:
                    try:
                        cols, rows = _safe_run(trap, sql)
                    except Exception:
                        continue
                    if _caught_fraction(cols, rows, planted) > 0:
                        hit = True
                        catching_sqls.add(sql)
                if hit:
                    ts.caught.append(did)
                else:
                    ts.missed.append(did)
            ts.components["c_catches"] = len(ts.caught) / len(expected_defects)
        else:
            ts.components["c_catches"] = 1.0   # nothing to catch (adversarial)

        # c_grounding — required citations present, no invented references
        refs_cited = [d.get("referencia_regulatoria", "") for d in dqcs]
        ts.hallucinated_refs = [
            r for r in refs_cited if not _ref_is_known(r, self.allowed)]
        must = exp.get("must_cite", [])
        cited_canon = " | ".join(canon_ref(p)
                                 for r in refs_cited for p in split_refs(r))
        must_ok = all(canon_ref(m) in cited_canon for m in must)
        ts.components["c_grounding"] = (
            1.0 if (must_ok and not ts.hallucinated_refs) else 0.0)
        if not must_ok:
            ts.notes.append(f"missing required citation(s): {must}")

        # c_coherence — catching checks must be multi-column when required
        if exp.get("forbid_single_column") and expected_defects:
            coherent = any(_score_coherence(normalize_sql(s)) >= 1.0
                           for s in catching_sqls)
            ts.components["c_coherence"] = 1.0 if coherent else 0.0
            if not coherent and catching_sqls:
                ts.notes.append("no catching check references >= 2 columns")
        else:
            ts.components["c_coherence"] = 1.0

        ts.total = round(sum(WEIGHTS[k] * v
                             for k, v in ts.components.items()), 4)
        ts.passed = all(v >= 1.0 for v in ts.components.values())
        return ts

    def evaluate(self, candidates: dict[str, list[dict]]) -> dict:
        """Score all traces; aggregate into the 8-dimension rubric."""
        traces = load_traces()
        scores = [self.score_trace(t, candidates.get(t["trace_id"], []))
                  for t in traces]

        rubric = {dim: {"traces": 0, "passed": 0, "mean_score": 0.0}
                  for dim in DIMENSIONS}
        for s in scores:
            r = rubric[s.dimension]
            r["traces"] += 1
            r["passed"] += int(s.passed)
            r["mean_score"] += s.total
        for r in rubric.values():
            if r["traces"]:
                r["mean_score"] = round(r["mean_score"] / r["traces"], 4)
                r["pass_rate"] = round(r["passed"] / r["traces"], 3)

        n_pass = sum(1 for s in scores if s.passed)
        return {
            "n_traces": len(scores),
            "passed": n_pass,
            "pass_rate": round(n_pass / len(scores), 3) if scores else 0.0,
            "mean_score": round(sum(s.total for s in scores) / len(scores), 4)
                          if scores else 0.0,
            "rubric": rubric,
            "traces": [asdict(s) for s in scores],
        }


# ── candidate sources ────────────────────────────────────────────────────────

def gold_candidates() -> dict[str, list[dict]]:
    return {t["trace_id"]: t.get("gold", []) for t in load_traces()}


def file_candidates(path: Path) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for tid, val in raw.items():
        out[tid] = val.get("dqcs", val) if isinstance(val, dict) else val
    return out


def agent_candidates(base_url: str) -> dict[str, list[dict]]:
    import urllib.request
    out: dict[str, list[dict]] = {}
    for t in load_traces():
        payload = json.dumps({"message": t["request"],
                              "session_id": "golden_eval"}).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/dqc/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out[t["trace_id"]] = json.loads(resp.read()).get("dqcs", [])
        except Exception as e:
            print(f"  ! agent error on {t['trace_id']}: {e}", file=sys.stderr)
            out[t["trace_id"]] = []
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_report(rep: dict) -> None:
    print(f"GOLDEN TRACES: {rep['passed']}/{rep['n_traces']} passed "
          f"(pass rate {rep['pass_rate']:.0%}, mean score {rep['mean_score']:.2f})")
    print("\n8-dimension rubric (pass rate / mean score):")
    for dim in DIMENSIONS:
        r = rep["rubric"].get(dim, {})
        if r.get("traces"):
            flag = "  ← DEFICIENT" if r["pass_rate"] < 0.5 else ""
            print(f"  {dim:14s} {r['pass_rate']:>5.0%}  "
                  f"score {r['mean_score']:.2f}  "
                  f"({r['passed']}/{r['traces']}){flag}")
    fails = [t for t in rep["traces"] if not t["passed"]]
    if fails:
        print("\nFailed traces:")
        for t in fails:
            why = (t["notes"] + ([f"missed defects: {t['missed']}"]
                                 if t["missed"] else [])
                   + ([f"hallucinated: {t['hallucinated_refs']}"]
                      if t["hallucinated_refs"] else []))
            print(f"  ✗ {t['trace_id']} ({t['dimension']}, "
                  f"score {t['total']:.2f}): {'; '.join(why) or 'see components'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DQC golden-trace eval")
    ap.add_argument("--selftest", action="store_true",
                    help="score the gold answers (must be 100%%)")
    ap.add_argument("--candidates", type=Path,
                    help="JSON file {trace_id: [dqc, ...]}")
    ap.add_argument("--agent", type=str, help="API base URL")
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail-under", type=float, default=None, metavar="RATE")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        cands = gold_candidates()
    elif args.candidates:
        cands = file_candidates(args.candidates)
    elif args.agent:
        cands = agent_candidates(args.agent)
    else:
        ap.print_help()
        return

    ev = GoldenEvaluator(rows=args.rows, seed=args.seed)
    rep = ev.evaluate(cands)
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        _print_report(rep)

    threshold = 1.0 if args.selftest else args.fail_under
    if threshold is not None and rep["pass_rate"] < threshold:
        print(f"\nFAIL: pass rate {rep['pass_rate']:.0%} < {threshold:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
