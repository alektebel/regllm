#!/usr/bin/env python3
"""Generate synthetic recuperatory cycles for tabular embedding development.

Produces a CSV with monthly snapshots for N recovery cycles, each spanning
up to 96 months (EBA minimum observation window).  Includes the full
casuistic range: segmentos, productos, terminaciones, adjudicaciones,
temporal evolution of PD/LGD/EAD/ECL, and IFRS9 staging.

Output columns are a mix of static cycle-level attributes (repeated per
snapshot) and time-varying financial / risk metrics.

Usage:
    python scripts/generate_recuperatory_cycles.py [--n-cycles 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "samples" / "recuperatory_cycles.csv"

# ── Domain constants ─────────────────────────────────────────────────────────

SEGMENTOS = ["CORP", "RETAIL", "SME", "MORTGAGE"]

PRODUCTOS_BY_SEGMENT: dict[str, list[str]] = {
    "CORP":     ["PRESTAMO", "LINEA_CREDITO", "TARJETA"],
    "RETAIL":   ["TARJETA", "PRESTAMO", "LINEA_CREDITO"],
    "SME":      ["PRESTAMO", "LINEA_CREDITO", "TARJETA"],
    "MORTGAGE": ["HIPOTECA"],
}

TERMINACIONES = [
    "PAGO_VOLUNTARIO", "SUBASTA", "DACCION_PAGO",
    "FALLIDO", "VENTA_CARTERA", "LIQUIDACION",
]

CAUSAS_DEFAULT = [
    "90_DIAS_VENCIDO", "IMPROBABLE_PAGO", "QUIEBRA",
]

COLATERAL_TIPOS = ["NINGUNA", "HIPOTECA", "PRENDA", "GARANTIA_PERSONAL", "FINANCIERO"]

ADJUDICACION_TIPOS = [
    "SUBASTA_INMOBILIARIA", "SUBASTA_MUEBLE", "DACION_PAGO",
    "VENTA_JUDICIAL", "VENTA_EXTRAJUDICIAL",
]

CALIBRATION_SEGMENTS: dict[str, list[str]] = {
    "CORP":     ["CORP_LARGE", "CORP_MID", "CORP_SMALL"],
    "RETAIL":   ["RETAIL_REVOLVING", "RETAIL_OTHER"],
    "SME":      ["SME_MICRO", "SME_SMALL", "SME_MEDIUM"],
    "MORTGAGE": ["MORTGAGE_LTV_LOW", "MORTGAGE_LTV_MED", "MORTGAGE_LTV_HIGH"],
}

# Master scale: rating grade -> PD
MASTER_SCALE = {
    1: 0.0003, 2: 0.0008, 3: 0.0025, 4: 0.0050, 5: 0.0100,
    6: 0.0180, 7: 0.0300, 8: 0.0500, 9: 0.1000, 10: 0.2500,
}

# Segment weights for random sampling
SEGMENT_WEIGHTS = [0.20, 0.25, 0.25, 0.30]

# Recovery parameters by SEGMENTO: (mean_monthly_recovery_rate, std_dev, recovery_start_month, max_recovery_rate)
RECOVERY_PARAMS: dict[str, tuple[float, float, int, float]] = {
    "MORTGAGE": (0.012, 0.015, 6, 0.70),
    "CORP":     (0.020, 0.030, 3, 0.55),
    "SME":      (0.015, 0.020, 4, 0.50),
    "RETAIL":   (0.025, 0.025, 2, 0.35),
}

CURE_PROB_BY_SEGMENT: dict[str, float] = {
    "MORTGAGE": 0.20, "CORP": 0.10, "SME": 0.12, "RETAIL": 0.15,
}

ADJUDICACION_PROB_BY_SEGMENT: dict[str, float] = {
    "MORTGAGE": 0.35, "CORP": 0.15, "SME": 0.10, "RETAIL": 0.05,
}

# OR_DISPTO ranges by segment: (min, max)
DISPTO_RANGES: dict[str, tuple[float, float]] = {
    "CORP":     (500_000, 10_000_000),
    "RETAIL":   (3_000, 60_000),
    "SME":      (50_000, 2_000_000),
    "MORTGAGE": (80_000, 500_000),
}

COLLATERAL_BY_PRODUCT: dict[str, list[str]] = {
    "HIPOTECA":      ["HIPOTECA"],
    "PRESTAMO":      ["NINGUNA", "GARANTIA_PERSONAL", "PRENDA"],
    "LINEA_CREDITO": ["NINGUNA", "GARANTIA_PERSONAL", "PRENDA", "FINANCIERO"],
    "TARJETA":       ["NINGUNA"],
}

LGD_FLOOR_BY_COLATERAL: dict[str, float] = {
    "NINGUNA":           0.0,
    "GARANTIA_PERSONAL": 0.0,
    "PRENDA":            0.10,
    "FINANCIERO":        0.10,
    "HIPOTECA":          0.30,
}

# LGD base by product
LGD_BASE_BY_PRODUCT: dict[str, float] = {
    "HIPOTECA":      0.45,
    "PRESTAMO":      0.35,
    "LINEA_CREDITO": 0.40,
    "TARJETA":       0.65,
}

# Interest rate ranges by product
INTEREST_RANGES: dict[str, tuple[float, float]] = {
    "HIPOTECA":       (0.015, 0.045),
    "PRESTAMO":       (0.035, 0.095),
    "LINEA_CREDITO":  (0.025, 0.080),
    "TARJETA":        (0.12, 0.25),
}

# Termination probability weights given segment (approximate real-world dist)
TERMINACION_WEIGHTS: dict[str, list[float]] = {
    "CORP":     [0.15, 0.25, 0.10, 0.25, 0.15, 0.10],
    "RETAIL":   [0.35, 0.10, 0.08, 0.25, 0.12, 0.10],
    "SME":      [0.25, 0.20, 0.10, 0.20, 0.15, 0.10],
    "MORTGAGE": [0.20, 0.35, 0.15, 0.10, 0.10, 0.10],
}


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _yyyymm(d: date) -> str:
    return f"{d.year}{d.month:02d}"


class RecuperatoryCycleGenerator:
    """Generates synthetic recovery cycle data with temporal evolution."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._contrato_counter = 100_000

    def _next_contrato_id(self) -> int:
        self._contrato_counter += 1
        return self._contrato_counter

    def _pick_weighted(self, items: list, weights: list | None = None):
        if weights is None:
            return self.rng.choice(items)
        return self.rng.choices(items, weights=weights, k=1)[0]

    def _sample_cycle_attributes(self) -> dict:
        """Sample static attributes for one recovery cycle."""
        segmento = self._pick_weighted(SEGMENTOS, SEGMENT_WEIGHTS)
        producto = self._pick_weighted(PRODUCTOS_BY_SEGMENT[segmento])
        terminacion = self._pick_weighted(TERMINACIONES, TERMINACION_WEIGHTS[segmento])
        causa = self._pick_weighted(CAUSAS_DEFAULT)
        calib_seg = self._pick_weighted(CALIBRATION_SEGMENTS[segmento])

        contrato_id = self._next_contrato_id()
        cliente_id = self.rng.randint(10_000, 99_999)

        fecha_default = _random_date(
            self.rng, date(2018, 1, 1), date(2023, 12, 31)
        )

        disp_min, disp_max = DISPTO_RANGES[segmento]
        or_dispto = round(self.rng.uniform(disp_min, disp_max), 2)

        # Collateral
        col_tipos = COLLATERAL_BY_PRODUCT[producto]
        col_tipo = self._pick_weighted(col_tipos)
        if col_tipo == "HIPOTECA":
            valor_colateral = round(or_dispto * self.rng.uniform(0.8, 1.5), 2)
        elif col_tipo == "PRENDA":
            valor_colateral = round(or_dispto * self.rng.uniform(0.3, 0.8), 2)
        elif col_tipo == "FINANCIERO":
            valor_colateral = round(or_dispto * self.rng.uniform(0.5, 1.0), 2)
        else:
            valor_colateral = 0.0

        # Interest rate
        ir_min, ir_max = INTEREST_RANGES[producto]
        tipo_interes = round(self.rng.uniform(ir_min, ir_max), 5)

        # Rating grade and PD
        if segmento == "CORP":
            rating = self.rng.choices(range(1, 11), weights=[5, 10, 15, 15, 15, 12, 10, 8, 5, 5])[0]
        elif segmento == "MORTGAGE":
            rating = self.rng.choices(range(1, 11), weights=[3, 8, 12, 15, 15, 13, 12, 10, 7, 5])[0]
        else:
            rating = self.rng.choices(range(1, 11), weights=[2, 5, 8, 12, 14, 15, 14, 12, 10, 8])[0]
        pd_default = MASTER_SCALE[rating]

        # EAD
        ead = round(or_dispto * self.rng.uniform(0.7, 1.0), 2)

        # LGD base and floor
        lgd_base = LGD_BASE_BY_PRODUCT[producto] * self.rng.uniform(0.8, 1.2)
        lgd_floor = LGD_FLOOR_BY_COLATERAL.get(col_tipo, 0.0)
        lgd_estimada = max(min(lgd_base, 0.95), lgd_floor)

        return {
            "cliente_id": cliente_id,
            "contrato_id": contrato_id,
            "id_fusion_final": f"{contrato_id}",
            "segmento": segmento,
            "calibration_segment": calib_seg,
            "producto": producto,
            "terminacion": terminacion,
            "causa_default": causa,
            "fecha_default": fecha_default,
            "or_dispto": or_dispto,
            "colateral_tipo": col_tipo,
            "valor_colateral_inicial": valor_colateral,
            "tipo_interes_original": tipo_interes,
            "rating_grado": rating,
            "pd_pre_default": pd_default,
            "ead": ead,
            "lgd_estimada_inicial": round(lgd_estimada, 6),
            "lgd_floor": lgd_floor,
        }

    def _generate_snapshots(self, cycle: dict) -> list[dict]:
        """Generate monthly snapshots for a cycle."""
        seg = cycle["segmento"]
        months = self._sample_cycle_length(seg, cycle["terminacion"])
        total_ead = cycle["ead"]

        # Determine if and when cure happens
        cure_prob = CURE_PROB_BY_SEGMENT[seg]
        cure_month: int | None = None
        if self.rng.random() < cure_prob:
            cure_month = self.rng.randint(3, min(24, months // 2))

        # Determine if adjudicacion occurs
        adj_prob = ADJUDICACION_PROB_BY_SEGMENT[seg]
        adj_month: int | None = None
        adj_tipo: str | None = None
        adj_valor: float = 0.0
        if self.rng.random() < adj_prob:
            adj_month = self.rng.randint(6, max(6, months - 6))
            adj_tipo = self._pick_weighted(ADJUDICACION_TIPOS)
            # Adjudication value as % of EAD
            adj_valor = round(
                total_ead * self.rng.uniform(0.15, 0.60), 2
            )

        # Recovery parameters
        mean_rec, std_rec, start_month, max_rec_rate = RECOVERY_PARAMS[seg]
        base_date = cycle["fecha_default"]
        lgd_floor = cycle["lgd_floor"]

        snapshots = []
        recuperacion_acum = 0.0
        coste_acum = 0.0
        saldo = cycle["or_dispto"]
        provision = 0.0
        tipo_interes = cycle["tipo_interes_original"]

        for mes in range(1, months + 1):
            ref_date = date(
                base_date.year + (base_date.month + mes - 1) // 12,
                (base_date.month + mes - 1) % 12 + 1,
                1,
            )

            # -- Temporal attribute evolution --

            # PD
            if cure_month is not None and mes >= cure_month:
                cure_progress = min(1.0, (mes - cure_month) / 24.0)
                pd_est = cycle["pd_pre_default"] * (1.0 - cure_progress * 0.95)
            elif mes <= 3:
                pd_est = 1.0
            else:
                pd_decay = max(0.0, 1.0 - (mes - 3) / 60.0)
                pd_est = min(1.0, cycle["pd_pre_default"] + pd_decay * 0.5)

            pd_est = max(cycle["pd_pre_default"], min(1.0, pd_est))

            # LGD estimated (converges toward realized over time)
            lgd_realizada = (
                1.0 - recuperacion_acum / total_ead if total_ead > 0 else 1.0
            )
            lgd_realizada = max(lgd_floor, min(1.0, lgd_realizada))

            blend = min(1.0, mes / 36.0)
            lgd_est = cycle["lgd_estimada_inicial"] * (
                1.0 - blend
            ) + lgd_realizada * blend
            lgd_est = max(lgd_floor, min(1.0, lgd_est))

            # ECL
            ecl = round(pd_est * lgd_est * total_ead, 2)

            # Stage IFRS9
            if cure_month is not None and mes >= cure_month + 12:
                stage = 2 if mes < cure_month + 24 else 1
            elif cure_month is not None and mes >= cure_month:
                stage = 2
            else:
                stage = 3

            # Interest
            if self.rng.random() < 0.02:  # 2% chance of rate change
                ir_min, ir_max = INTEREST_RANGES[cycle["producto"]]
                tipo_interes = round(self.rng.uniform(ir_min, ir_max), 5)

            intereses_mes = round(saldo * tipo_interes / 12, 2)
            intereses_acum = round(intereses_mes * mes, 2)

            # Recovery cash flow
            flujo_recuperado = 0.0
            coste_recuperacion = 0.0

            if mes >= start_month and stage == 3:
                # Recovery depends on termination type
                if cycle["terminacion"] == "PAGO_VOLUNTARIO" and mes < months - 3:
                    # Steady payments
                    flujo_recuperado = round(
                        abs(self.np_rng.normal(mean_rec * total_ead, std_rec * total_ead)), 2
                    )
                elif cycle["terminacion"] == "SUBASTA" and mes == adj_month:
                    flujo_recuperado = round(total_ead * self.rng.uniform(0.3, 0.6), 2)
                elif cycle["terminacion"] == "DACCION_PAGO" and mes == adj_month:
                    flujo_recuperado = round(total_ead * self.rng.uniform(0.5, 0.8), 2)
                elif cycle["terminacion"] == "VENTA_CARTERA":
                    flujo_recuperado = round(total_ead * self.rng.uniform(0.1, 0.3), 2)
                elif cycle["terminacion"] == "LIQUIDACION":
                    if mes == months:
                        flujo_recuperado = round(total_ead * self.rng.uniform(0.3, 0.6), 2)
                    elif mes >= start_month:
                        flujo_recuperado = round(
                            abs(self.np_rng.normal(mean_rec * total_ead * 0.5, std_rec * total_ead * 0.5)), 2
                        )
                elif cycle["terminacion"] == "FALLIDO":
                    if mes >= start_month and self.rng.random() < 0.1:
                        flujo_recuperado = round(
                            abs(self.np_rng.normal(mean_rec * total_ead * 0.3, std_rec * total_ead * 0.3)), 2
                        )
                else:
                    if mes >= start_month and self.rng.random() < 0.3:
                        flujo_recuperado = round(
                            abs(self.np_rng.normal(mean_rec * total_ead, std_rec * total_ead)), 2
                        )

                coste_recuperacion = round(
                    self.rng.uniform(0.01, 0.05) * flujo_recuperado, 2
                )
            elif mes < start_month and stage == 3:
                # First few months post-default, minimal recovery
                if self.rng.random() < 0.05:
                    flujo_recuperado = round(
                        abs(self.np_rng.normal(0.001 * total_ead, 0.002 * total_ead)), 2
                    )

            # Cap recovery at remaining outstanding
            remaining = max(0.0, total_ead - recuperacion_acum)
            flujo_recuperado = min(flujo_recuperado, remaining)
            flujo_neto = round(flujo_recuperado - coste_recuperacion, 2)

            recuperacion_acum += flujo_recuperado
            coste_acum += coste_recuperacion

            # Salary, adjust for recovery
            saldo = round(cycle["or_dispto"] - recuperacion_acum, 2)
            if saldo < 0:
                saldo = 0.0

            # DPDS
            dpds = mes * 30

            # Provision
            provision = round(ecl, 2)

            # RWA
            rwa = round(total_ead * 0.08 * self.rng.uniform(0.8, 1.2), 2)

            # LTV
            if cycle["valor_colateral_inicial"] > 0:
                ltv = round(saldo / cycle["valor_colateral_inicial"], 6)
            else:
                ltv = 0.0

            # Collateral value changes over time
            valor_colateral = round(
                cycle["valor_colateral_inicial"]
                * (1.0 - 0.005 * mes * self.rng.uniform(0.5, 1.5)),
                2,
            )
            if valor_colateral < 0:
                valor_colateral = 0.0

            # Estado del ciclo
            if recuperacion_acum >= total_ead * 0.95:
                estado = "CERRADO"
            elif mes >= start_month:
                estado = "RECUPERACION"
            else:
                estado = "ESTIMACION"

            # Adjudicacion event
            adj_flag = 0
            adj_tipo_str = ""
            adj_valor_actual = 0.0
            if adj_month is not None and mes == adj_month:
                adj_flag = 1
                adj_tipo_str = adj_tipo
                adj_valor_actual = adj_valor

            # Cure flag
            cure_flag = 1 if cure_month is not None and mes >= cure_month else 0

            # Discount rate (risk-free + spread)
            tasa_descuento = round(
                self.rng.uniform(0.01, 0.03) + self.rng.uniform(0.005, 0.025), 6
            )

            # Rating grade evolves with PD
            if pd_est <= 0.001:
                rating_grado = self.rng.randint(1, 2)
            elif pd_est <= 0.005:
                rating_grado = self.rng.randint(3, 4)
            elif pd_est <= 0.02:
                rating_grado = self.rng.randint(5, 6)
            elif pd_est <= 0.08:
                rating_grado = self.rng.randint(7, 8)
            else:
                rating_grado = self.rng.randint(9, 10)

            # ── Build row ──
            row = {
                # ID
                "ID_CONTR_CICLO_LGD": f"{cycle['contrato_id']}_{_yyyymm(ref_date)}",
                "ID_FUSION_FINAL": str(cycle["contrato_id"]),
                # Static
                "SEGMENTO": cycle["segmento"],
                "CALIBRATION_SEGMENT": cycle["calibration_segment"],
                "PRODUCTO": cycle["producto"],
                "TERMINACION": cycle["terminacion"],
                "CAUSA_DEFAULT": cycle["causa_default"],
                "FECHA_DEFAULT": cycle["fecha_default"].isoformat(),
                "OR_DISPTO": cycle["or_dispto"],
                "COLATERAL_TIPO": cycle["colateral_tipo"],
                "VALOR_COLATERAL_INICIAL": cycle["valor_colateral_inicial"],
                "TIPO_INTERES_ORIGINAL": cycle["tipo_interes_original"],
                # Temporal
                "FECHA_REFERENCIA": ref_date.isoformat(),
                "MES_CICLO": mes,
                "SALDO_PENDIENTE": saldo,
                "TIPO_INTERES_ACTUAL": tipo_interes,
                "INTERESES_ACUMULADOS": intereses_acum,
                "PD_ESTIMADA": round(pd_est, 8),
                "LGD_ESTIMADA": round(lgd_est, 8),
                "LGD_REALIZADA": round(lgd_realizada, 8),
                "EAD": total_ead,
                "ECL": ecl,
                "PROVISION": provision,
                "STAGE_IFRS9": stage,
                "CURE_FLAG": cure_flag,
                "DPDS": dpds,
                "RATING_GRADO": rating_grado,
                "FLUJO_RECUPERADO": flujo_recuperado,
                "COSTE_RECUPERACION": coste_recuperacion,
                "FLUJO_NETO": flujo_neto,
                "RECUPERACION_ACUMULADA": round(recuperacion_acum, 2),
                "COSTE_TOTAL_ACUMULADO": round(coste_acum, 2),
                "ADJUDICACION_FLAG": adj_flag,
                "ADJUDICACION_TIPO": adj_tipo_str,
                "ADJUDICACION_VALOR": adj_valor_actual,
                "VALOR_COLATERAL": valor_colateral,
                "LTV": ltv,
                "TASA_DESCUENTO": tasa_descuento,
                "RWA": rwa,
                "ESTADO_CICLO": estado,
            }
            snapshots.append(row)

            # Early termination if fully recovered or written off
            if estado == "CERRADO":
                break

        return snapshots

    def _sample_cycle_length(self, segmento: str, terminacion: str) -> int:
        """Sample cycle length in months based on segment and termination type."""
        base = {
            "CORP":     48,
            "RETAIL":   24,
            "SME":      36,
            "MORTGAGE": 60,
        }.get(segmento, 36)

        term_mod = {
            "PAGO_VOLUNTARIO": -6,
            "SUBASTA": 12,
            "DACCION_PAGO": -3,
            "FALLIDO": 12,
            "VENTA_CARTERA": -12,
            "LIQUIDACION": 6,
        }.get(terminacion, 0)

        length = base + term_mod + self.rng.randint(-6, 12)
        return max(6, min(96, length))

    def generate(self, n_cycles: int = 500) -> list[dict]:
        """Generate N recovery cycles and return all monthly snapshots."""
        all_rows: list[dict] = []
        for i in range(n_cycles):
            cycle = self._sample_cycle_attributes()
            snapshots = self._generate_snapshots(cycle)
            all_rows.extend(snapshots)
        return all_rows


FIELD_ORDER = [
    # Identifiers
    "ID_CONTR_CICLO_LGD",
    "ID_FUSION_FINAL",
    # Static attributes
    "SEGMENTO",
    "CALIBRATION_SEGMENT",
    "PRODUCTO",
    "TERMINACION",
    "CAUSA_DEFAULT",
    "FECHA_DEFAULT",
    "OR_DISPTO",
    "COLATERAL_TIPO",
    "VALOR_COLATERAL_INICIAL",
    "TIPO_INTERES_ORIGINAL",
    # Temporal identifiers
    "FECHA_REFERENCIA",
    "MES_CICLO",
    # Risk parameters
    "PD_ESTIMADA",
    "LGD_ESTIMADA",
    "LGD_REALIZADA",
    "EAD",
    "ECL",
    "PROVISION",
    "STAGE_IFRS9",
    "CURE_FLAG",
    "DPDS",
    "RATING_GRADO",
    # Financial
    "SALDO_PENDIENTE",
    "TIPO_INTERES_ACTUAL",
    "INTERESES_ACUMULADOS",
    # Recovery
    "FLUJO_RECUPERADO",
    "COSTE_RECUPERACION",
    "FLUJO_NETO",
    "RECUPERACION_ACUMULADA",
    "COSTE_TOTAL_ACUMULADO",
    # Adjudicaciones
    "ADJUDICACION_FLAG",
    "ADJUDICACION_TIPO",
    "ADJUDICACION_VALOR",
    # Collateral
    "VALOR_COLATERAL",
    "LTV",
    # Regulatory
    "TASA_DESCUENTO",
    "RWA",
    "ESTADO_CICLO",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic recuperatory cycles data"
    )
    parser.add_argument("--n-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    gen = RecuperatoryCycleGenerator(seed=args.seed)
    rows = gen.generate(n_cycles=args.n_cycles)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_ORDER, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_snapshots = len(rows)
    avg_per_cycle = n_snapshots / args.n_cycles
    print(
        f"Wrote {n_snapshots} snapshots across {args.n_cycles} cycles "
        f"({avg_per_cycle:.1f} avg per cycle) → {args.output}"
    )


if __name__ == "__main__":
    main()
