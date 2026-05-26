"""Tier-1 vectorized hard rules engine.

Applies deterministic, regulation-grounded checks on full DataFrames without
any LLM calls. Each rule returns a boolean mask of FAILING rows.

Sources:
  - CRR Art. 160(1): PD floor 0.03% for non-defaulted retail
  - CRR Art. 161(1)(b): LGD floor 45% unsecured corporate senior
  - CRR Art. 154(3): LGD floor 30% retail residential mortgage
  - EBA GL 2017/16 §5.3.2: Observation window ≥ 5 years for PD
  - EBA GL 2017/16 §6.3: Calibration period ≥ 7 years for LGD
  - IFRS 9 B5.5.12: SICR backstop 30 DPD for Stage 2

All thresholds are intentionally commented with the source so the compliance
team can audit and update as regulations change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

@dataclass
class HardRuleResult:
    rule_id: str
    description: str
    article: str
    failing_mask: pd.Series          # bool Series, True = FAILING
    failing_rows: pd.DataFrame       # sub-df of failing rows (for reporting)
    n_failing: int = 0
    n_total: int = 0

    def __post_init__(self):
        self.n_failing = int(self.failing_mask.sum())
        self.n_total   = len(self.failing_mask)

    @property
    def pass_rate(self) -> float:
        return 1.0 - self.n_failing / self.n_total if self.n_total else 1.0


@dataclass
class HardRulesReport:
    results: list[HardRuleResult] = field(default_factory=list)

    @property
    def any_failure_mask(self) -> pd.Series | None:
        if not self.results:
            return None
        combined = self.results[0].failing_mask.copy().astype(bool)
        for r in self.results[1:]:
            combined = combined | r.failing_mask
        return combined

    def to_findings(self) -> list[dict]:
        """Convert hard-rule failures to the same findings schema as LLM verdicts."""
        findings = []
        for res in self.results:
            for _, row in res.failing_rows.iterrows():
                findings.append({
                    "row_id":     str(row.name),
                    "verdict":    "flagged",
                    "resumen":    f"[Regla dura] {res.description}",
                    "flags":      [res.description],
                    "articles":   [res.article],
                    "rag_sources": [],
                    "advertencias": None,
                    "row_data":   row.to_dict(),
                    "tier":       1,
                })
        return findings

    def summary(self) -> dict:
        return {
            r.rule_id: {
                "description": r.description,
                "article":     r.article,
                "n_failing":   r.n_failing,
                "n_total":     r.n_total,
                "pass_rate":   round(r.pass_rate, 4),
            }
            for r in self.results
        }


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

def _resolve(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name found in df (case-insensitive)."""
    df_cols_upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in df_cols_upper:
            return df_cols_upper[cand.upper()]
    return None


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def _rule_pd_floor(df: pd.DataFrame, column_map: dict[str, str]) -> HardRuleResult | None:
    """CRR Art. 160(1) — PD floor 0.03 % for non-defaulted exposures."""
    inv_map = {v: k for k, v in column_map.items()}
    col = _resolve(df, [inv_map.get("PD", "PD"), "PD", "PD_ESTIMATE", "PD_CALIBRADA"])
    if col is None:
        return None
    series = pd.to_numeric(df[col], errors="coerce")
    mask = series.notna() & (series < 0.0003)
    return HardRuleResult(
        rule_id="PD_FLOOR",
        description=f"PD ({col}) < suelo regulatorio 0.03% (CRR Art. 160(1))",
        article="CRR Art. 160(1)",
        failing_mask=mask,
        failing_rows=df[mask],
    )


def _rule_lgd_floor_corporate(df: pd.DataFrame, column_map: dict[str, str]) -> HardRuleResult | None:
    """CRR Art. 161(1)(b) — LGD floor 45% unsecured corporate senior."""
    inv_map = {v: k for k, v in column_map.items()}
    lgd_col = _resolve(df, [inv_map.get("LGD", "LGD"), "LGD", "LGD_ESTIMATE", "LGD_CALIBRADA"])
    seg_col = _resolve(df, ["SEGMENT", "SEGMENTO", "CARTERA"])
    if lgd_col is None:
        return None

    series = pd.to_numeric(df[lgd_col], errors="coerce")
    if seg_col is not None:
        seg = df[seg_col].astype(str).str.upper()
        is_corporate = seg.str.contains("CORP|EMPRESA|WHOLESALE", regex=True, na=False)
        mask = is_corporate & series.notna() & (series < 0.45)
    else:
        mask = series.notna() & (series < 0.45)

    return HardRuleResult(
        rule_id="LGD_FLOOR_CORP",
        description=f"LGD ({lgd_col}) < suelo 45% para carteras corporativas (CRR Art. 161(1)(b))",
        article="CRR Art. 161(1)(b)",
        failing_mask=mask,
        failing_rows=df[mask],
    )


def _rule_lgd_floor_mortgage(df: pd.DataFrame, column_map: dict[str, str]) -> HardRuleResult | None:
    """CRR Art. 154(3) — LGD floor 30% retail residential mortgage."""
    inv_map = {v: k for k, v in column_map.items()}
    lgd_col = _resolve(df, [inv_map.get("LGD", "LGD"), "LGD", "LGD_ESTIMATE", "LGD_CALIBRADA"])
    seg_col = _resolve(df, ["SEGMENT", "SEGMENTO", "CARTERA"])
    col_col = _resolve(df, ["COLLATERAL_TYPE", "TIPO_COLATERAL", "GARANTIA"])
    if lgd_col is None:
        return None

    series = pd.to_numeric(df[lgd_col], errors="coerce")
    is_mortgage = pd.Series(False, index=df.index)
    for col in [seg_col, col_col]:
        if col is not None:
            vals = df[col].astype(str).str.upper()
            is_mortgage = is_mortgage | vals.str.contains("HIPOT|MORTGAGE|INMUEBLE|VIVIENDA", regex=True, na=False)

    mask = is_mortgage & series.notna() & (series < 0.30)
    return HardRuleResult(
        rule_id="LGD_FLOOR_MORTGAGE",
        description=f"LGD ({lgd_col}) < suelo 30% para hipotecas residenciales (CRR Art. 154(3))",
        article="CRR Art. 154(3)",
        failing_mask=mask,
        failing_rows=df[mask],
    )


def _rule_obs_window(df: pd.DataFrame, column_map: dict[str, str]) -> HardRuleResult | None:
    """EBA GL 2017/16 §5.3.2 — Observation window ≥ 5 years for PD estimation."""
    inv_map = {v: k for k, v in column_map.items()}
    col = _resolve(df, [
        inv_map.get("OBS", "OBS"),
        "OBSERVATION_WINDOW_YEARS", "VENTANA_OBSERVACION", "OBS_YEARS", "OBS_WINDOW",
    ])
    if col is None:
        return None
    series = pd.to_numeric(df[col], errors="coerce")
    mask = series.notna() & (series < 5)
    return HardRuleResult(
        rule_id="OBS_WINDOW_MIN",
        description=f"Ventana de observación ({col}) < 5 años mínimo (EBA GL 2017/16 §5.3.2)",
        article="EBA GL 2017/16 §5.3.2",
        failing_mask=mask,
        failing_rows=df[mask],
    )


def _rule_calibration_period(df: pd.DataFrame, column_map: dict[str, str]) -> HardRuleResult | None:
    """EBA GL 2017/16 §6.3 — Calibration period ≥ 7 years for LGD."""
    inv_map = {v: k for k, v in column_map.items()}
    col = _resolve(df, [
        inv_map.get("CALIB_YEARS", "CALIB_YEARS"),
        "CALIBRATION_PERIOD_YEARS", "PERIODO_CALIBRACION", "CALIB_PERIOD",
    ])
    if col is None:
        return None
    series = pd.to_numeric(df[col], errors="coerce")
    mask = series.notna() & (series < 7)
    return HardRuleResult(
        rule_id="CALIB_PERIOD_MIN",
        description=f"Período de calibración ({col}) < 7 años (EBA GL 2017/16 §6.3)",
        article="EBA GL 2017/16 §6.3",
        failing_mask=mask,
        failing_rows=df[mask],
    )


def _rule_stage2_backstop(df: pd.DataFrame, column_map: dict[str, str]) -> HardRuleResult | None:
    """IFRS 9 B5.5.12 — Stage 2 backstop: DPD ≥ 30 must be Stage 2 or 3."""
    inv_map = {v: k for k, v in column_map.items()}
    dpd_col   = _resolve(df, [inv_map.get("DPD", "DPD"), "DPD", "DIAS_MORA", "DAYS_PAST_DUE"])
    stage_col = _resolve(df, [inv_map.get("STAGE", "STAGE"), "STAGE", "ETAPA", "IFRS9_STAGE"])
    if dpd_col is None or stage_col is None:
        return None

    dpd   = pd.to_numeric(df[dpd_col], errors="coerce")
    stage = pd.to_numeric(df[stage_col], errors="coerce")
    # DPD ≥ 30 AND stage == 1 is a backstop violation
    mask = dpd.notna() & stage.notna() & (dpd >= 30) & (stage == 1)
    return HardRuleResult(
        rule_id="STAGE2_BACKSTOP",
        description=(
            f"DPD ({dpd_col}) ≥ 30 días con Stage ({stage_col}) = 1 "
            "— backstop SICR incumplido (IFRS 9 B5.5.12)"
        ),
        article="IFRS 9 B5.5.12",
        failing_mask=mask,
        failing_rows=df[mask],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RULES = [
    _rule_pd_floor,
    _rule_lgd_floor_corporate,
    _rule_lgd_floor_mortgage,
    _rule_obs_window,
    _rule_calibration_period,
    _rule_stage2_backstop,
]


def apply_hard_rules(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> HardRulesReport:
    """Apply all vectorized hard rules to *df*.

    Returns a HardRulesReport with per-rule results and an aggregate mask
    that Tier-2 uses to skip already-flagged rows.
    """
    column_map = column_map or {}
    report = HardRulesReport()
    for rule_fn in _RULES:
        try:
            result = rule_fn(df, column_map)
            if result is not None:
                report.results.append(result)
                if result.n_failing:
                    logger.info(
                        "Hard rule %s: %d/%d failing",
                        result.rule_id, result.n_failing, result.n_total,
                    )
        except Exception as e:
            logger.warning("Hard rule %s failed: %s", rule_fn.__name__, e)
    return report
