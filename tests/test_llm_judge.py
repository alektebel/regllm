"""
LLM-judged comprehensive test suite for RegLLM.

Tests use Qwen 2.5-7B as a judge model to evaluate if RAG responses
match ground truth semantically. Tests also validate source retrieval
and confidence thresholds.

Run:
    pytest tests/test_llm_judge.py -m llm_judge -v     # All LLM tests (GPU)
    pytest tests/test_llm_judge.py -m "not llm_judge"  # Non-LLM tests only
    pytest tests/test_llm_judge.py::TestFinancialQA -v  # Financial only
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "test_ground_truth.json"


def _load_ground_truth():
    """Load ground truth entries for parametrize (runs at collection time)."""
    if not GROUND_TRUTH_PATH.exists():
        return []
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["entries"]


def _financial_entries():
    return [e for e in _load_ground_truth() if e["category"] == "financial"]


def _regulatory_entries():
    return [e for e in _load_ground_truth() if e["category"] == "regulatory"]


def _assert_sources_match(returned_sources, expected_sources):
    """
    Assert that returned sources contain expected document references.

    Uses case-insensitive 'contains' on documento field.
    Checks exact match on articulo when specified.
    """
    returned_docs = []
    for src in returned_sources:
        doc = src.get("documento", "") if isinstance(src, dict) else getattr(src, "documento", "")
        art = src.get("articulo") if isinstance(src, dict) else getattr(src, "articulo", None)
        returned_docs.append({"documento": doc.lower(), "articulo": art})

    for expected in expected_sources:
        exp_doc = expected["documento"].lower()
        exp_art = expected.get("articulo")

        found = False
        for ret in returned_docs:
            if exp_doc in ret["documento"] or ret["documento"] in exp_doc:
                if exp_art is None or ret["articulo"] == exp_art:
                    found = True
                    break

        assert found, (
            f"Expected source '{expected['documento']}' "
            f"(articulo={exp_art}) not found in returned sources: "
            f"{[r['documento'] for r in returned_docs]}"
        )


# ============================================================================
# Financial QA Tests
# ============================================================================

@pytest.mark.llm_judge
@pytest.mark.slow
class TestFinancialQA:
    """Parametrized tests for financial QA entries using LLM judge."""

    @pytest.mark.parametrize(
        "entry",
        _financial_entries(),
        ids=[e["id"] for e in _financial_entries()],
    )
    def test_financial_qa_confidence(self, rag_test_client, entry):
        """Verify that financial queries return sufficient confidence."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": entry["pregunta"], "n_fuentes": 5},
        )
        assert response.status_code == 200, f"API error: {response.text}"

        data = response.json()
        assert data["confianza"] >= entry["umbral_confianza"], (
            f"[{entry['id']}] Confidence {data['confianza']:.2f} < "
            f"threshold {entry['umbral_confianza']} for: {entry['pregunta']}"
        )

    @pytest.mark.parametrize(
        "entry",
        _financial_entries(),
        ids=[e["id"] for e in _financial_entries()],
    )
    def test_financial_qa_sources(self, rag_test_client, entry):
        """Verify that financial queries return expected sources."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": entry["pregunta"], "n_fuentes": 5},
        )
        assert response.status_code == 200

        data = response.json()
        _assert_sources_match(data["fuentes"], entry["fuentes_esperadas"])

    @pytest.mark.parametrize(
        "entry",
        _financial_entries(),
        ids=[e["id"] for e in _financial_entries()],
    )
    def test_financial_qa_judge(self, rag_test_client, llm_judge, entry):
        """Use LLM judge to evaluate semantic adequacy of financial responses."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": entry["pregunta"], "n_fuentes": 5},
        )
        assert response.status_code == 200

        data = response.json()
        result = llm_judge.evaluate(
            pregunta=entry["pregunta"],
            respuesta_generada=data["respuesta"],
            respuesta_esperada=entry["respuesta_esperada"],
            datos_clave=entry["datos_clave"],
        )

        assert result.is_adequate, (
            f"[{entry['id']}] LLM judge: response inadequate "
            f"(score={result.score:.2f}). "
            f"Missing: {result.key_facts_missing}. "
            f"Explanation: {result.explanation}"
        )


# ============================================================================
# Regulatory QA Tests
# ============================================================================

@pytest.mark.llm_judge
@pytest.mark.slow
class TestRegulatoryQA:
    """Parametrized tests for regulatory QA entries using LLM judge."""

    @pytest.mark.parametrize(
        "entry",
        _regulatory_entries(),
        ids=[e["id"] for e in _regulatory_entries()],
    )
    def test_regulatory_qa_confidence(self, rag_test_client, entry):
        """Verify that regulatory queries return sufficient confidence."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": entry["pregunta"], "n_fuentes": 5},
        )
        assert response.status_code == 200, f"API error: {response.text}"

        data = response.json()
        assert data["confianza"] >= entry["umbral_confianza"], (
            f"[{entry['id']}] Confidence {data['confianza']:.2f} < "
            f"threshold {entry['umbral_confianza']} for: {entry['pregunta']}"
        )

    @pytest.mark.parametrize(
        "entry",
        _regulatory_entries(),
        ids=[e["id"] for e in _regulatory_entries()],
    )
    def test_regulatory_qa_sources(self, rag_test_client, entry):
        """Verify that regulatory queries return expected sources."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": entry["pregunta"], "n_fuentes": 5},
        )
        assert response.status_code == 200

        data = response.json()
        _assert_sources_match(data["fuentes"], entry["fuentes_esperadas"])

    @pytest.mark.parametrize(
        "entry",
        _regulatory_entries(),
        ids=[e["id"] for e in _regulatory_entries()],
    )
    def test_regulatory_qa_judge(self, rag_test_client, llm_judge, entry):
        """Use LLM judge to evaluate semantic adequacy of regulatory responses."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": entry["pregunta"], "n_fuentes": 5},
        )
        assert response.status_code == 200

        data = response.json()
        result = llm_judge.evaluate(
            pregunta=entry["pregunta"],
            respuesta_generada=data["respuesta"],
            respuesta_esperada=entry["respuesta_esperada"],
            datos_clave=entry["datos_clave"],
        )

        assert result.is_adequate, (
            f"[{entry['id']}] LLM judge: response inadequate "
            f"(score={result.score:.2f}). "
            f"Missing: {result.key_facts_missing}. "
            f"Explanation: {result.explanation}"
        )


# ============================================================================
# Source Retrieval Tests (no LLM judge needed)
# ============================================================================

class TestSourceRetrieval:
    """Source-only tests verifying document retrieval without LLM judge."""

    @pytest.mark.parametrize(
        "bank_name",
        ["Banco Santander", "CaixaBank", "BBVA", "Banco Sabadell", "Kutxabank"],
    )
    def test_bank_source_retrieval(self, rag_test_client, bank_name):
        """Verify that querying a bank returns sources mentioning that bank."""
        response = rag_test_client.post(
            "/consultar",
            json={"pregunta": f"¿Cuáles fueron los resultados de {bank_name} en 2023?", "n_fuentes": 5},
        )
        assert response.status_code == 200

        data = response.json()
        if not data["fuentes"]:
            pytest.skip("No sources returned (DB may be empty)")

        source_docs = [f["documento"].lower() for f in data["fuentes"]]
        bank_lower = bank_name.lower()
        assert any(
            bank_lower in doc or doc in bank_lower for doc in source_docs
        ), f"No source mentioning '{bank_name}' found in: {source_docs}"

    def test_eba_source_retrieval(self, rag_test_client):
        """Verify that EBA regulatory queries return EBA-related sources."""
        response = rag_test_client.post(
            "/consultar",
            json={
                "pregunta": "¿Cuál es el objetivo principal de las directrices EBA/GL/2017/16?",
                "n_fuentes": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        if not data["fuentes"]:
            pytest.skip("No sources returned (DB may be empty)")

        source_docs = [f["documento"].lower() for f in data["fuentes"]]
        assert any(
            "eba" in doc or "gl/2017" in doc for doc in source_docs
        ), f"No EBA-related source found in: {source_docs}"


# ============================================================================
# Confidence Threshold Tests
# ============================================================================

class TestConfidenceThresholds:
    """Boundary tests for confidence scoring."""

    def test_known_good_question_confidence(self, rag_test_client):
        """A known factual question should return reasonable confidence."""
        response = rag_test_client.post(
            "/consultar",
            json={
                "pregunta": "¿Cuáles fueron los resultados de Banco Santander en 2023?",
                "n_fuentes": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        # With data loaded, confidence should be above a minimal threshold
        if data["fuentes"]:
            assert data["confianza"] >= 0.3, (
                f"Confidence too low for known question: {data['confianza']:.2f}"
            )

    def test_gibberish_question_low_confidence(self, rag_test_client):
        """A gibberish question should return low confidence or no sources."""
        response = rag_test_client.post(
            "/consultar",
            json={
                "pregunta": "asdfghjkl qwerty zxcvbnm lorem ipsum dolor sit amet",
                "n_fuentes": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        # Either no sources or low confidence is acceptable
        if data["fuentes"]:
            assert data["confianza"] <= 0.8, (
                f"Confidence unexpectedly high for gibberish: {data['confianza']:.2f}"
            )

    def test_response_has_confianza_field(self, rag_test_client):
        """Verify the confianza field exists in API response."""
        response = rag_test_client.post(
            "/consultar",
            json={
                "pregunta": "¿Qué es el ratio de capital CET1?",
                "n_fuentes": 3,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "confianza" in data, "Response missing 'confianza' field"
        assert isinstance(data["confianza"], (int, float)), "confianza must be numeric"
        assert 0.0 <= data["confianza"] <= 1.0, f"confianza out of range: {data['confianza']}"
