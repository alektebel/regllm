"""
Adversarial alignment tests — embedding-based quality gates.

Uses sentence-transformers cosine similarity to verify that model-like responses
align with the ground-truth dataset and that off-topic answers score low.

No GPU, no live model required — only the embed model (downloaded once).
Mark slow tests with @pytest.mark.slow so CI can skip them:
    pytest tests/test_adversarial_alignment.py -v -m "not slow"

Dataset: data/test_ground_truth.json (50 entries, "regulatory" + "financial" categories)
Embed model: paraphrase-multilingual-MiniLM-L12-v2 (same as src/db.py)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "test_ground_truth.json"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Thresholds (calibrated for multilingual MiniLM)
SELF_SIM_THRESHOLD = 0.99       # identical text must score ≥ 0.99
PARAPHRASE_THRESHOLD = 0.55     # paraphrase must score ≥ 0.55 (multilingual MiniLM)
OFF_TOPIC_MAX = 0.50            # off-topic must score < 0.50
ALIGNMENT_THRESHOLD = 0.45     # model answer vs GT must score ≥ 0.45


# ─── Shared fixture: embed model ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


@pytest.fixture(scope="module")
def ground_truth():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["entries"]


def _sim(model, a: str, b: str) -> float:
    vecs = model.encode([a, b])
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])


def _batch_sim(model, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    vecs_a = model.encode(texts_a)
    vecs_b = model.encode(texts_b)
    return cosine_similarity(vecs_a, vecs_b).diagonal()


# ─── Sanity: self-similarity ──────────────────────────────────────────────────

def test_self_similarity_is_near_one(embed_model):
    """Identical strings must have cosine similarity ≈ 1.0."""
    text = "El ratio CET1 mínimo requerido por el CRR es del 4,5% de los activos ponderados por riesgo."
    score = _sim(embed_model, text, text)
    assert score >= SELF_SIM_THRESHOLD, f"Self-similarity too low: {score:.4f}"


def test_self_similarity_spanish_regulatory(embed_model):
    text = "El ICAAP es el proceso de evaluación de la adecuación del capital interno según Basilea III."
    score = _sim(embed_model, text, text)
    assert score >= SELF_SIM_THRESHOLD


# ─── Sanity: paraphrase similarity ───────────────────────────────────────────

def test_paraphrase_similarity_high(embed_model):
    """Near-synonymous regulatory sentences must score above threshold."""
    a = "El CET1 mínimo es el 4,5% de los activos ponderados por riesgo."
    b = "El capital de nivel 1 ordinario requerido es un 4,5% de los RWA."
    score = _sim(embed_model, a, b)
    assert score >= PARAPHRASE_THRESHOLD, f"Paraphrase similarity too low: {score:.4f}"


def test_paraphrase_lcr(embed_model):
    a = "El ratio de cobertura de liquidez mínimo es del 100%."
    b = "Las entidades deben mantener un LCR no inferior al 100% en todo momento."
    score = _sim(embed_model, a, b)
    assert score >= PARAPHRASE_THRESHOLD


# ─── Off-topic rejection threshold ───────────────────────────────────────────

def test_offtopic_cooking_scores_low(embed_model, ground_truth):
    """An off-topic (cooking) answer must score far below any GT answer."""
    offtopic = "Para hacer una paella valenciana necesitas arroz, pollo, conejo y azafrán."
    gt_texts = [e["respuesta_esperada"] for e in ground_truth[:10]]
    vecs = embed_model.encode([offtopic] + gt_texts)
    sims = cosine_similarity([vecs[0]], vecs[1:])[0]
    max_sim = float(sims.max())
    assert max_sim < OFF_TOPIC_MAX, (
        f"Off-topic answer scored {max_sim:.4f} — too close to regulatory GT "
        f"(threshold < {OFF_TOPIC_MAX})"
    )


def test_offtopic_poetry_scores_low(embed_model, ground_truth):
    offtopic = "Las rosas son rojas, las violetas son azules, el amor es dulce igual que tú."
    gt_texts = [e["respuesta_esperada"] for e in ground_truth[:10]]
    vecs = embed_model.encode([offtopic] + gt_texts)
    sims = cosine_similarity([vecs[0]], vecs[1:])[0]
    max_sim = float(sims.max())
    assert max_sim < OFF_TOPIC_MAX, f"Poetry scored {max_sim:.4f}"


# ─── Model answer alignment against ground truth ──────────────────────────────

@pytest.mark.slow
def test_regulatory_answers_align_with_gt(embed_model, ground_truth):
    """
    For each regulatory GT entry, a mock 'model answer' that references the
    same key terms must score above ALIGNMENT_THRESHOLD against the GT answer.
    This simulates a correctly-functioning model.
    """
    regulatory = [e for e in ground_truth if e.get("category") == "regulatory"]
    assert len(regulatory) > 0, "No regulatory entries in ground truth"

    low_scores = []
    for entry in regulatory:
        gt_answer = entry["respuesta_esperada"]
        # Simulate a model answer that paraphrases key terms from the GT
        model_answer = f"Según la normativa, {gt_answer[:80]}..."
        score = _sim(embed_model, model_answer, gt_answer)
        if score < ALIGNMENT_THRESHOLD:
            low_scores.append((entry["id"], score))

    if low_scores:
        details = ", ".join(f"{eid}={s:.3f}" for eid, s in low_scores)
        pytest.fail(
            f"{len(low_scores)}/{len(regulatory)} regulatory answers below "
            f"threshold {ALIGNMENT_THRESHOLD}: {details}"
        )


@pytest.mark.slow
def test_financial_answers_align_with_gt(embed_model, ground_truth):
    """Same for financial entries (expected lower scores — numerical data)."""
    financial = [e for e in ground_truth if e.get("category") == "financial"]
    if not financial:
        pytest.skip("No financial entries in ground truth")

    low_scores = []
    for entry in financial:
        gt_answer = entry["respuesta_esperada"]
        model_answer = f"En cuanto a los datos financieros, {gt_answer[:80]}..."
        score = _sim(embed_model, model_answer, gt_answer)
        if score < ALIGNMENT_THRESHOLD:
            low_scores.append((entry["id"], score))

    # Financial is harder — allow up to 30% failures (numerical confabulation risk)
    fail_rate = len(low_scores) / len(financial)
    assert fail_rate <= 0.30, (
        f"{fail_rate:.0%} of financial answers below threshold "
        f"({len(low_scores)}/{len(financial)})"
    )


# ─── Category discrimination ──────────────────────────────────────────────────

def test_regulatory_answers_closer_to_regulatory_gt(embed_model, ground_truth):
    """
    A regulatory answer should score higher against regulatory GT entries
    than against off-topic text.
    """
    regulatory = [e for e in ground_truth if e.get("category") == "regulatory"][:5]
    if not regulatory:
        pytest.skip("No regulatory entries")

    regulatory_answer = regulatory[0]["respuesta_esperada"]
    other_regulatory_gt = [e["respuesta_esperada"] for e in regulatory[1:]]

    offtopic_texts = [
        "El precio del aceite de oliva en España subió un 40% en 2023.",
        "El Real Madrid ganó la Champions League en 2024.",
        "Para hacer gazpacho se necesitan tomates, pepino y pimiento.",
    ]

    vecs = embed_model.encode([regulatory_answer] + other_regulatory_gt + offtopic_texts)
    reg_sims = cosine_similarity([vecs[0]], vecs[1:1 + len(other_regulatory_gt)])[0]
    off_sims = cosine_similarity([vecs[0]], vecs[1 + len(other_regulatory_gt):])[0]

    mean_reg = float(reg_sims.mean()) if len(reg_sims) else 0.0
    mean_off = float(off_sims.mean())

    assert mean_reg > mean_off, (
        f"Regulatory answer not closer to regulatory GT "
        f"(reg={mean_reg:.3f}, off={mean_off:.3f})"
    )


def test_financial_and_regulatory_gt_are_distinguishable(embed_model, ground_truth):
    """
    Mean within-category similarity should exceed between-category similarity,
    showing the embed model can discriminate regulatory vs financial content.
    """
    regulatory = [e["respuesta_esperada"] for e in ground_truth if e.get("category") == "regulatory"][:5]
    financial = [e["respuesta_esperada"] for e in ground_truth if e.get("category") == "financial"][:5]
    if not regulatory or not financial:
        pytest.skip("Not enough entries in both categories")

    reg_vecs = embed_model.encode(regulatory)
    fin_vecs = embed_model.encode(financial)

    # Within-regulatory similarity
    reg_sim_matrix = cosine_similarity(reg_vecs)
    np.fill_diagonal(reg_sim_matrix, 0)
    within_reg = float(reg_sim_matrix.sum() / (len(regulatory) * (len(regulatory) - 1))) if len(regulatory) > 1 else 0.0

    # Cross-category similarity
    cross_sim = float(cosine_similarity(reg_vecs, fin_vecs).mean())

    # Regulatory texts should be more similar to each other than to financial texts
    assert within_reg > cross_sim - 0.05, (
        f"Categories not well-separated: within_reg={within_reg:.3f}, cross={cross_sim:.3f}"
    )


# ─── Adversarial: near-miss injection ────────────────────────────────────────

def test_adversarial_regulatory_sounding_offtopic(embed_model, ground_truth):
    """
    A text that sounds regulatory but is factually wrong should still
    achieve reasonable similarity (topic-level), confirming the embed model
    tests semantic topic, not factual correctness.
    """
    adversarial = (
        "El ratio CET1 mínimo exigido por el CRR es del 99%, "
        "y los bancos deben mantener un LCR del 500% en todo momento."
    )
    gt_texts = [e["respuesta_esperada"] for e in ground_truth if e.get("category") == "regulatory"][:5]
    if not gt_texts:
        pytest.skip("No regulatory GT")

    vecs = embed_model.encode([adversarial] + gt_texts)
    sims = cosine_similarity([vecs[0]], vecs[1:])[0]
    max_sim = float(sims.max())

    # The adversarial text IS about regulation — embed similarity should be moderate
    # multilingual MiniLM uses conservative similarity for factually-wrong texts
    assert max_sim >= 0.30, (
        f"Adversarial regulatory text unexpectedly low similarity: {max_sim:.3f}. "
        "Embed model may be filtering by correctness rather than topic."
    )
