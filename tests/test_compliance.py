"""Unit tests for the compliance checker — no DB, no LLM required."""

import json
import pytest
import pandas as pd

from src.compliance.memory import RunningMemory, Finding
from src.compliance.checker import ComplianceChecker, RowVerdict, _parse_json, _ground_articles
from src.compliance.hard_rules import apply_hard_rules
from src.chat_engine import ground_articles


# ── _ground_articles (compliance module) ─────────────────────────────────────

class TestGroundArticles:
    def test_keeps_article_present_in_context(self):
        ctx = "CRR Art. 160(1) establece el suelo de PD del 0.03% para exposiciones corporativas."
        result = _ground_articles(["CRR Art. 160(1)"], ctx)
        assert result == ["CRR Art. 160(1)"]

    def test_drops_article_absent_from_context(self):
        ctx = "EBA/GL/2017/16 sección 5 sobre estimación de LGD en downturn."
        result = _ground_articles(["CRR Art. 999(99)"], ctx)
        assert result == []

    def test_empty_articles_returns_empty(self):
        assert _ground_articles([], "any context") == []

    def test_empty_context_returns_original(self):
        # When no context, cannot filter — return as-is
        arts = ["CRR Art. 160"]
        assert _ground_articles(arts, "") == arts

    def test_partial_match_accepted(self):
        # "art 160" fragment should match "Art. 160" in context
        ctx = "Según el Art. 160 del CRR, la PD mínima es 0.03%."
        result = _ground_articles(["CRR Art. 160(1)"], ctx)
        assert result == ["CRR Art. 160(1)"]

    def test_mixed_keep_and_drop(self):
        ctx = "EBA/GL/2017/16 §8.3 cubre el Margen de Conservadurismo."
        arts = ["EBA/GL/2017/16 §8.3", "CRR Art. 999"]
        result = _ground_articles(arts, ctx)
        assert "EBA/GL/2017/16 §8.3" in result
        assert "CRR Art. 999" not in result


# ── ground_articles (chat_engine module) ─────────────────────────────────────

class TestChatEngineGroundArticles:
    def test_keeps_grounded(self):
        ctx = "CRR Art. 92 establece los requerimientos de capital mínimo."
        assert ground_articles(["CRR Art. 92"], ctx) == ["CRR Art. 92"]

    def test_drops_hallucinated(self):
        ctx = "CRR Art. 92 establece los requerimientos de capital mínimo."
        assert ground_articles(["EBA/GL/2025/99 §99"], ctx) == []

    def test_none_input_returns_empty(self):
        assert ground_articles(None, "ctx") is None or ground_articles([], "ctx") == []


# ── Running memory ────────────────────────────────────────────────────────────

class TestRunningMemory:
    def _finding(self, row_id="R1"):
        return Finding(row_id=row_id, column="PD", value=0.0001,
                       verdict="flagged", flags=["PD muy baja"], articles=["CRR Art. 160"])

    def test_add_compliant(self):
        m = RunningMemory()
        m.add_compliant()
        assert m.rows_processed == 1
        assert m.rows_flagged == 0

    def test_add_finding(self):
        m = RunningMemory()
        m.add_finding(self._finding())
        assert m.rows_processed == 1
        assert m.rows_flagged == 1
        assert len(m.findings) == 1

    def test_needs_compression(self):
        m = RunningMemory(compress_threshold=2)
        m.add_finding(self._finding("R1"))
        m.add_finding(self._finding("R2"))
        assert m.needs_compression()

    def test_compress_clears_findings(self):
        m = RunningMemory(compress_threshold=1)
        m.add_finding(self._finding())
        m.compress(lambda msgs: "Patrón: PDs bajas en corporativos.")
        assert m.findings == []
        assert "PD" in m.patterns or "Patrón" in m.patterns

    def test_compress_fallback_on_error(self):
        m = RunningMemory(compress_threshold=1)
        m.add_finding(self._finding())
        m.compress(lambda msgs: (_ for _ in ()).throw(RuntimeError("LLM down")))  # type: ignore
        assert m.findings == []

    def test_as_prompt_str(self):
        m = RunningMemory()
        m.add_compliant()
        m.add_finding(self._finding())
        s = m.as_prompt_str()
        assert "2" in s
        assert "1" in s


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self):
        raw = json.dumps({"verdict": "compliant", "flags": [], "articles": []})
        assert _parse_json(raw)["verdict"] == "compliant"

    def test_markdown_fence(self):
        raw = '```json\n{"verdict": "flagged", "flags": ["PD baja"]}\n```'
        assert _parse_json(raw)["verdict"] == "flagged"

    def test_partial_json_fallback(self):
        raw = '{"verdict": "flagged", "flags": ["problema"'  # truncated
        result = _parse_json(raw)
        assert result["verdict"] == "uncertain"

    def test_plain_text_fallback(self):
        result = _parse_json("La PD está fuera de rango regulatorio.")
        assert result["verdict"] == "uncertain"
        assert result["flags"]  # non-empty


# ── ComplianceChecker ─────────────────────────────────────────────────────────

class TestComplianceChecker:
    def _mock_llm_compliant(self, messages):
        return json.dumps({
            "verdict": "compliant",
            "resumen": "El parámetro es conforme.",
            "desarrollo": "La PD está dentro del rango esperado.",
            "flags": [],
            "articles": [],
            "advertencias": None,
        })

    def _mock_llm_flagged(self, messages):
        return json.dumps({
            "verdict": "flagged",
            "resumen": "PD por debajo del suelo regulatorio.",
            "desarrollo": "La PD de 0.00005 podría estar por debajo del mínimo regulatorio según CRR Art. 160(1).",
            "flags": ["PD inferior al suelo mínimo del 0.03%"],
            "articles": ["CRR Art. 160(1)"],
            "advertencias": "Verificar con los documentos oficiales del CRR.",
        })

    def _make_checker(self, llm_fn):
        c = ComplianceChecker(rag_system=None, llm_backend="ollama")
        c._llm_fn = llm_fn
        return c

    def test_diagnose_compliant_row(self):
        checker = self._make_checker(self._mock_llm_compliant)
        verdict = checker.diagnose_row("R1", {"PD": 0.01, "LGD": 0.45})
        assert verdict.verdict == "compliant"
        assert verdict.flags == []
        assert isinstance(verdict, RowVerdict)

    def test_diagnose_flagged_row(self):
        checker = self._make_checker(self._mock_llm_flagged)
        verdict = checker.diagnose_row("R1", {"PD": 0.00005, "LGD": 0.45})
        assert verdict.verdict == "flagged"
        assert len(verdict.flags) == 1
        # articles should be empty because mock context is empty (grounded to [])
        assert verdict.articles == []

    def test_flagged_row_updates_memory(self):
        checker = self._make_checker(self._mock_llm_flagged)
        checker.diagnose_row("R1", {"PD": 0.00005})
        assert checker._memory.rows_flagged == 1

    def test_compliant_row_increments_counter_only(self):
        checker = self._make_checker(self._mock_llm_compliant)
        checker.diagnose_row("R1", {"PD": 0.01})
        assert checker._memory.rows_processed == 1
        assert checker._memory.rows_flagged == 0

    def test_run_dataframe(self):
        call_count = [0]

        def mock_llm(messages):
            call_count[0] += 1
            # Flag every other row
            if call_count[0] % 2 == 0:
                return self._mock_llm_flagged(messages)
            return self._mock_llm_compliant(messages)

        checker = self._make_checker(mock_llm)
        df = pd.DataFrame([
            {"PD": 0.01, "LGD": 0.45, "EAD": 100_000},
            {"PD": 0.00005, "LGD": 0.40, "EAD": 50_000},
            {"PD": 0.005, "LGD": 0.50, "EAD": 200_000},
        ])
        report = checker.run(df, table_name="TEST_TABLE")

        assert report["rows_processed"] == 3
        assert report["table"] == "TEST_TABLE"
        assert "compliance_rate" in report
        assert 0.0 <= report["compliance_rate"] <= 1.0
        assert "findings" in report
        assert "narrative" in report

    def test_run_uses_row_id_col(self):
        checker = self._make_checker(self._mock_llm_compliant)
        df = pd.DataFrame([{"ID": "OBL_001", "PD": 0.01}])
        report = checker.run(df, row_id_col="ID")
        assert report["rows_processed"] == 1

    def test_memory_compression_triggered(self):
        compress_calls = [0]

        def mock_llm(messages):
            content = messages[-1]["content"] if messages else ""
            # Compression prompt has no "FILA A DIAGNOSTICAR"
            if "FILA A DIAGNOSTICAR" not in content:
                compress_calls[0] += 1
                return "Patrón: múltiples PDs bajas detectadas."
            return self._mock_llm_flagged(messages)

        checker = self._make_checker(mock_llm)
        checker._memory.compress_threshold = 3

        df = pd.DataFrame([{"PD": 0.00005} for _ in range(6)])
        checker.run(df)
        assert compress_calls[0] >= 1


# ── Hard rules: provision period ─────────────────────────────────────────────

class TestProvisionPeriodRule:
    """IFRS 9 B5.5.33/B5.5.35 — provision period must be informed and stage-consistent."""

    def _run(self, rows):
        df = pd.DataFrame(rows)
        return apply_hard_rules(df)

    def _get_rule(self, report):
        return next((r for r in report.results if r.rule_id == "PROVISION_PERIOD"), None)

    def test_missing_column_skipped(self):
        """No PROVISION_PERIOD_MONTHS column → rule returns None, no crash."""
        report = self._run([{"PD": 0.01}])
        assert self._get_rule(report) is None

    def test_null_value_flagged(self):
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": None, "STAGE": 1}])
        report = apply_hard_rules(df)
        rule = self._get_rule(report)
        assert rule is not None
        assert rule.n_failing == 1

    def test_zero_value_flagged(self):
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": 0, "STAGE": 1}])
        report = apply_hard_rules(df)
        assert self._get_rule(report).n_failing == 1

    def test_stage1_correct(self):
        """Stage 1 with 12 months → compliant."""
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": 12, "IFRS9_STAGE": 1}])
        report = apply_hard_rules(df)
        assert self._get_rule(report).n_failing == 0

    def test_stage1_wrong_period_flagged(self):
        """Stage 1 with 6 months → flagged (should be 12-month ECL)."""
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": 6, "IFRS9_STAGE": 1}])
        report = apply_hard_rules(df)
        assert self._get_rule(report).n_failing == 1

    def test_stage2_lifetime_correct(self):
        """Stage 2 with 24 months → compliant (lifetime ECL > 12)."""
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": 24, "IFRS9_STAGE": 2}])
        report = apply_hard_rules(df)
        assert self._get_rule(report).n_failing == 0

    def test_stage2_twelve_months_flagged(self):
        """Stage 2 with 12 months → flagged (lifetime ECL must be > 12)."""
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": 12, "IFRS9_STAGE": 2}])
        report = apply_hard_rules(df)
        assert self._get_rule(report).n_failing == 1

    def test_stage3_short_period_flagged(self):
        """Stage 3 with 6 months → flagged."""
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": 6, "STAGE": 3}])
        report = apply_hard_rules(df)
        assert self._get_rule(report).n_failing == 1

    def test_mixed_rows(self):
        """Mix of compliant and failing rows."""
        df = pd.DataFrame([
            {"PROVISION_PERIOD_MONTHS": 12, "STAGE": 1},   # OK
            {"PROVISION_PERIOD_MONTHS": None, "STAGE": 1},  # missing
            {"PROVISION_PERIOD_MONTHS": 6, "STAGE": 1},    # wrong period
            {"PROVISION_PERIOD_MONTHS": 24, "STAGE": 2},   # OK
            {"PROVISION_PERIOD_MONTHS": 12, "STAGE": 2},   # should be lifetime
        ])
        report = apply_hard_rules(df)
        rule = self._get_rule(report)
        assert rule.n_failing == 3
        assert rule.n_total == 5

    def test_article_reference(self):
        """Rule must cite IFRS 9 B5.5.33."""
        df = pd.DataFrame([{"PROVISION_PERIOD_MONTHS": None}])
        report = apply_hard_rules(df)
        rule = self._get_rule(report)
        assert "IFRS 9" in rule.article
        assert "B5.5" in rule.article
