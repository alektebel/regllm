"""Tests for the Tier-1 schema-linking field filter (select_relevant_fields):
lexical default plus optional semantic (embedding) blending."""

from __future__ import annotations

import math

from api.routers.dqc_dictionary import (
    FieldEntry, select_relevant_fields, _cosine, _minmax,
)


def _fields(n: int) -> list[FieldEntry]:
    out = [FieldEntry(name=f"CAMPO_{i}", description=f"campo relleno {i}")
           for i in range(n)]
    return out


# ── behaviour without embeddings (unchanged) ─────────────────────────────────

def test_returns_all_when_under_cap():
    fs = _fields(5)
    # under the cap: no filtering, every field kept, no embedder needed
    assert select_relevant_fields(fs, ["x"], cap=60) == fs


def test_drops_ten_least_relevant_below_the_cap():
    # 40 fields (< cap 60) but with 10 clearly-irrelevant ones trimmed:
    # only 40-10 = 30 reach the prompt
    fs = _fields(40)
    chosen = select_relevant_fields(fs, ["campo relleno 3"], cap=60)
    assert len(chosen) == 30


def test_never_trims_a_small_dictionary():
    # under the MIN_FIELDS_KEPT floor: nothing is dropped
    fs = _fields(18)
    assert select_relevant_fields(fs, ["x"], cap=60) == fs
    # right at a size where the floor caps the trim (25 → keep 20, drop 5)
    assert len(select_relevant_fields(_fields(25), ["x"], cap=60)) == 20


def test_cap_and_tail_trim_together_drop_the_bottom_ten():
    # 66-field dictionary (the real reporting table size): keep 66-10 = 56
    fs = _fields(66)
    assert len(select_relevant_fields(fs, ["campo relleno 1"], cap=60)) == 56


def test_lexical_named_field_survives_the_cap():
    fs = _fields(70)
    fs.append(FieldEntry(name="PD_ESTIMADA", description="probabilidad"))
    chosen = select_relevant_fields(fs, ["la PD_ESTIMADA no supera 1"], cap=60)
    assert len(chosen) == 60
    assert any(f.name == "PD_ESTIMADA" for f in chosen)


def test_preserves_dictionary_order():
    fs = _fields(70)
    fs[65].description = "la regla habla de morosidad y morosidad"
    fs[10].description = "morosidad"
    chosen = select_relevant_fields(fs, ["morosidad"], cap=60)
    idx = [int(f.name.split("_")[1]) for f in chosen]
    assert idx == sorted(idx)


# ── semantic channel (fake embedder) ─────────────────────────────────────────

class _FakeEmbedder:
    """Embeds by a tiny hand-built semantic map so a field with no lexical
    overlap can still score high on the query."""

    VOCAB = {
        "impago": [1.0, 0.0, 0.0],
        "pd_estimada": [0.96, 0.1, 0.0],   # semantically near 'impago'
        "probabilidad": [0.9, 0.1, 0.0],
        "fecha": [0.0, 1.0, 0.0],
        "importe": [0.0, 0.0, 1.0],
    }

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        acc = [0.0, 0.0, 0.0]
        for k, v in self.VOCAB.items():
            if k in t:
                acc = [a + b for a, b in zip(acc, v)]
        return acc if any(acc) else [0.01, 0.01, 0.01]

    def embed(self, texts):
        return [self._vec(t) for t in texts]


class _StubEmbedder:
    def embed(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_semantic_channel_rescues_non_lexical_match():
    fs = _fields(70)
    # a field that shares NO words with the query 'impago'
    fs.append(FieldEntry(name="PD_ESTIMADA",
                         description="probabilidad de default"))
    query = "riesgo de impago del cliente"

    lexical_only = select_relevant_fields(fs, [query], cap=60)
    with_sem = select_relevant_fields(fs, [query], cap=60,
                                      embedder=_FakeEmbedder())

    # lexical alone cannot see it (no shared words); semantic surfaces it
    assert not any(f.name == "PD_ESTIMADA" for f in lexical_only)
    assert any(f.name == "PD_ESTIMADA" for f in with_sem)


def test_stub_embedder_falls_back_to_lexical():
    fs = _fields(70)
    fs.append(FieldEntry(name="PD_ESTIMADA", description="probabilidad"))
    query = "la PD_ESTIMADA no supera 1"
    lexical = select_relevant_fields(fs, [query], cap=60)
    # zero vectors ⇒ identical to lexical-only result
    stubbed = select_relevant_fields(fs, [query], cap=60,
                                     embedder=_StubEmbedder())
    assert [f.name for f in stubbed] == [f.name for f in lexical]


def test_embedder_exception_does_not_break_selection():
    class _Boom:
        def embed(self, texts):
            raise RuntimeError("embedding backend down")

    fs = _fields(70)
    fs.append(FieldEntry(name="PD_ESTIMADA", description="probabilidad"))
    chosen = select_relevant_fields(fs, ["PD_ESTIMADA"], cap=60,
                                    embedder=_Boom())
    assert any(f.name == "PD_ESTIMADA" for f in chosen)


# ── helpers ──────────────────────────────────────────────────────────────────

def test_cosine_and_minmax():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([0, 0], [1, 1]) == 0.0
    assert _minmax([2.0, 2.0]) == [0.0, 0.0]
    assert _minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]
