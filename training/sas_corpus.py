"""SAS corpus for multi-pipeline procedural bug generation.

Loads all usable SAS files in the project and pairs each with
compatible input rows. The BugGenerator then mutates each pipeline
independently, giving much more structural diversity than toy_lgd alone.

Pipelines included:
  - toy_lgd_correct.sas:     4 DATA steps, simple floors+ECL+staging
  - sample_lgd.sas (v2):     6 DATA steps, floors+MoC+RWA+ECL+staging+non-conformes
  - sample_lgd.sas (v3):     7 DATA steps, adds titulization
  - debug_lgd proj_03:       2 DATA steps, floors+MoC+ECL+RWA+LGD_REALIZADA

All share the same IRB domain (LGD/ECL/PD/EAD) but differ in structure:
different step ordering, MERGE vs SET, PROC SQL aggregations, etc.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.bug_generator import BugGenerator


# ── Input row templates ────────────────────────────────────────────────
# These cover the key scenarios: CORP/RETAIL, HIPOTECA/NINGUNA,
# fusion/non-fusion, above/below floors, boundary DPDs, cure flags.

_BASE_ROWS: list[dict[str, Any]] = [
    # CORP, no collateral, LGD below floor (0.40 < 0.45)
    {
        "CICLO_ID": "CIC_001", "CONTRATO": "CONT_001",
        "ID_CONTR_CICLO_LGD": "CIC_001", "ID_CONTRATO": "CONT_001",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.010, "LGD_ESTIMADA": 0.40, "EAD": 100000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 7, "VENTANA_OBSERVACION_YEARS": 5,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 100000, "OR_EAD_BASILEA": 100000,
        "RATING_GRADO": "A", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # CORP, HIPOTECA, LGD below floor (0.20 < 0.30)
    {
        "CICLO_ID": "CIC_003", "CONTRATO": "CONT_003",
        "ID_CONTR_CICLO_LGD": "CIC_003", "ID_CONTRATO": "CONT_003",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "HIPOTECA",
        "PD_ESTIMADA": 0.010, "LGD_ESTIMADA": 0.20, "EAD": 150000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 8, "VENTANA_OBSERVACION_YEARS": 6,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 150000, "OR_EAD_BASILEA": 150000,
        "RATING_GRADO": "B", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # RETAIL, PERSONAL, at DPD backstop boundary (DPDS=30)
    {
        "CICLO_ID": "CIC_007", "CONTRATO": "CONT_007",
        "ID_CONTR_CICLO_LGD": "CIC_007", "ID_CONTRATO": "CONT_007",
        "SEGMENTO": "RETAIL", "COLATERAL_TIPO": "PERSONAL",
        "PD_ESTIMADA": 0.030, "LGD_ESTIMADA": 0.35, "EAD": 80000,
        "DPDS": 30, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 7, "VENTANA_OBSERVACION_YEARS": 5,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 80000, "OR_EAD_BASILEA": 80000,
        "RATING_GRADO": "C", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": "2014-06-15",
    },
    # CORP, with CURE flag
    {
        "CICLO_ID": "CIC_009", "CONTRATO": "CONT_009",
        "ID_CONTR_CICLO_LGD": "CIC_009", "ID_CONTRATO": "CONT_009",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.010, "LGD_ESTIMADA": 0.45, "EAD": 120000,
        "DPDS": 5, "STAGE_IFRS9": 1, "CURE_FLAG": 1,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 9, "VENTANA_OBSERVACION_YEARS": 7,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 120000, "OR_EAD_BASILEA": 120000,
        "RATING_GRADO": "A", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # CORP, very low PD (below PD floor)
    {
        "CICLO_ID": "CIC_011", "CONTRATO": "CONT_011",
        "ID_CONTR_CICLO_LGD": "CIC_011", "ID_CONTRATO": "CONT_011",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.0001, "LGD_ESTIMADA": 0.50, "EAD": 200000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 15,
        "VENTANA_CALIBRACION_YEARS": 10, "VENTANA_OBSERVACION_YEARS": 8,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 200000, "OR_EAD_BASILEA": 200000,
        "RATING_GRADO": "AA", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # RETAIL, HIPOTECA, Stage 3 (default)
    {
        "CICLO_ID": "CIC_012", "CONTRATO": "CONT_012",
        "ID_CONTR_CICLO_LGD": "CIC_012", "ID_CONTRATO": "CONT_012",
        "SEGMENTO": "RETAIL", "COLATERAL_TIPO": "HIPOTECA",
        "PD_ESTIMADA": 0.15, "LGD_ESTIMADA": 0.55, "EAD": 250000,
        "DPDS": 90, "STAGE_IFRS9": 3, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 24,
        "VENTANA_CALIBRACION_YEARS": 7, "VENTANA_OBSERVACION_YEARS": 5,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 250000, "OR_EAD_BASILEA": 250000,
        "RATING_GRADO": "D", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": "2013-12-01",
    },
    # Fusion contract (absorbed entity — missing LGD common in level 02/05)
    {
        "CICLO_ID": "CIC_005", "CONTRATO": "CONT_005",
        "ID_CONTR_CICLO_LGD": "CIC_005", "ID_CONTRATO": "CONT_005",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.020, "LGD_ESTIMADA": None, "EAD": 300000,
        "DPDS": 15, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 9, "VENTANA_OBSERVACION_YEARS": 7,
        "SW_FUSION": 1, "ID_FUSION_FINAL": "FUS_7712",
        "OR_EAD": 300000, "OR_EAD_BASILEA": 100000,
        "RATING_GRADO": "B", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_002",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # Second fusion sibling (duplicate join scenarios)
    {
        "CICLO_ID": "CIC_006", "CONTRATO": "CONT_006",
        "ID_CONTR_CICLO_LGD": "CIC_006", "ID_CONTRATO": "CONT_006",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "NINGUNA",
        "PD_ESTIMADA": 0.025, "LGD_ESTIMADA": 0.48, "EAD": 280000,
        "DPDS": 10, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 9, "VENTANA_OBSERVACION_YEARS": 7,
        "SW_FUSION": 1, "ID_FUSION_FINAL": "FUS_7712",
        "OR_EAD": 280000, "OR_EAD_BASILEA": 100000,
        "RATING_GRADO": "B", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_003",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # Titulization / segment floor edge (retail PD near floor)
    {
        "CICLO_ID": "CIC_013", "CONTRATO": "CONT_013",
        "ID_CONTR_CICLO_LGD": "CIC_013", "ID_CONTRATO": "CONT_013",
        "SEGMENTO": "RETAIL", "COLATERAL_TIPO": "PERSONAL",
        "PD_ESTIMADA": 0.0002, "LGD_ESTIMADA": 0.42, "EAD": 95000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 9,
        "VENTANA_CALIBRACION_YEARS": 7, "VENTANA_OBSERVACION_YEARS": 5,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 95000, "OR_EAD_BASILEA": 95000,
        "RATING_GRADO": "C", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": None,
    },
    # Lowercase collateral (type coercion scenarios)
    {
        "CICLO_ID": "CIC_014", "CONTRATO": "CONT_014",
        "ID_CONTR_CICLO_LGD": "CIC_014", "ID_CONTRATO": "CONT_014",
        "SEGMENTO": "CORP", "COLATERAL_TIPO": "hipoteca",
        "PD_ESTIMADA": 0.012, "LGD_ESTIMADA": 0.18, "EAD": 160000,
        "DPDS": 0, "STAGE_IFRS9": 1, "CURE_FLAG": 0,
        "PROVISION_PERIOD_MONTHS": 12,
        "VENTANA_CALIBRACION_YEARS": 8, "VENTANA_OBSERVACION_YEARS": 6,
        "SW_FUSION": 0, "ID_FUSION_FINAL": None,
        "OR_EAD": 160000, "OR_EAD_BASILEA": 160000,
        "RATING_GRADO": "B", "PERIODO_OBSERVACION": 201402,
        "ENTIDAD_ORIGEN": "ENT_001",
        "FECHA_INCUMPLIMIENTO": None,
    },
]


# ── Pipeline definitions ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """A SAS pipeline available for bug generation."""
    name: str
    sas_path: Path
    description: str
    # Subset of _BASE_ROWS indices to use (some pipelines need specific schemas)
    row_indices: list[int] | None = None  # None = use all


_PIPELINES: list[PipelineConfig] = [
    PipelineConfig(
        name="toy_lgd",
        sas_path=PROJECT_ROOT / "data/sas/toy_lgd/toy_lgd_correct.sas",
        description="Simple 4-step LGD: floors → MoC → ECL → staging",
    ),
    PipelineConfig(
        name="sample_lgd_v2",
        sas_path=PROJECT_ROOT / "data/samples/sample_lgd.sas",
        description="6-step LGD v2: floors → segment agg → MoC → RWA → ECL → non-conformes",
    ),
    PipelineConfig(
        name="sample_lgd_v3",
        sas_path=PROJECT_ROOT / "data/sas/v3/sample_lgd.sas",
        description="7-step LGD v3: v2 + titulization step, higher floors (50%/0.05%)",
    ),
    PipelineConfig(
        name="debug_lgd_suelos",
        sas_path=PROJECT_ROOT / "data/sas/sessions/debug_lgd/proj_03_suelos_lgd.sas",
        description="2-step pipeline: floors + MoC + ECL + RWA + LGD_REALIZADA",
    ),
]


# ── Corpus class ───────────────────────────────────────────────────────

class SASCorpus:
    """Collection of SAS pipelines for multi-pipeline bug generation.

    Each pipeline is loaded, parsed, and paired with compatible input rows.
    The corpus provides BugGenerators for each pipeline and can generate
    bugs across all pipelines uniformly.
    """

    def __init__(self, seed: int = 42, max_depth: int | None = None):
        self.seed = seed
        self.max_depth = max_depth
        self.generators: list[tuple[str, BugGenerator]] = []
        self.curated_levels: list[tuple[str, Any]] = []
        self._load_all()
        self._load_curated_levels()

    def _load_curated_levels(self) -> None:
        try:
            from training.toy_lgd_levels import ToyLgdLevelCatalog
            cat = ToyLgdLevelCatalog()
            self.curated_levels = cat.levels
            if self.curated_levels:
                print(f"  loaded {len(self.curated_levels)} toy_lgd curated levels")
        except Exception as e:
            print(f"  skip curated levels: {e}")

    def _load_all(self) -> None:
        import random
        rng = random.Random(self.seed)

        for cfg in _PIPELINES:
            if not cfg.sas_path.exists():
                print(f"  skip {cfg.name}: {cfg.sas_path} not found")
                continue

            code = cfg.sas_path.read_text(encoding="utf-8")
            rows = self._rows_for_pipeline(cfg)

            try:
                gen = BugGenerator(
                    code, rows,
                    seed=rng.randint(0, 2**31),
                    max_depth=self.max_depth,
                )
                if gen._targets:
                    self.generators.append((cfg.name, gen))
                    print(f"  loaded {cfg.name}: {len(gen._targets)} targets, {len(rows)} rows")
                else:
                    print(f"  skip {cfg.name}: no mutation targets found")
            except Exception as e:
                print(f"  skip {cfg.name}: parse error: {e}")

    def _rows_for_pipeline(self, cfg: PipelineConfig) -> list[dict[str, Any]]:
        """Get input rows compatible with a pipeline."""
        if cfg.row_indices is not None:
            return [_BASE_ROWS[i] for i in cfg.row_indices]
        return list(_BASE_ROWS)

    @property
    def total_targets(self) -> int:
        return sum(len(g._targets) for _, g in self.generators)

    @property
    def pipeline_names(self) -> list[str]:
        return [name for name, _ in self.generators]

    def generate(self, max_attempts: int = 50):
        """Generate one bug from a random pipeline."""
        import random
        if not self.generators:
            return None, None
        name, gen = random.choice(self.generators)
        bug = gen.generate(max_attempts)
        return name, bug

    def generate_batch(self, n: int, balanced: bool = True):
        """Generate n bugs, balanced across pipelines if requested."""
        results = []
        if not self.generators:
            return results

        if balanced:
            per_pipeline = max(1, n // len(self.generators))
            for name, gen in self.generators:
                bugs = gen.generate_batch(per_pipeline)
                for bug in bugs:
                    results.append((name, bug))
        else:
            import random
            for _ in range(n * 3):
                if len(results) >= n:
                    break
                name, bug = self.generate()
                if bug is not None:
                    results.append((name, bug))

        return results[:n]

    def generate_curated(self, n: int, seed: int = 0) -> list[tuple[str, Any]]:
        import random
        if not self.curated_levels:
            return []
        rng = random.Random(seed)
        picked = self.curated_levels if n >= len(self.curated_levels) else rng.sample(self.curated_levels, n)
        return picked[:n]


if __name__ == "__main__":
    print("Loading SAS corpus...\n")
    corpus = SASCorpus(seed=42)

    print(f"\nTotal: {len(corpus.generators)} pipelines, {corpus.total_targets} mutation targets")
    print(f"Pipelines: {corpus.pipeline_names}\n")

    batch = corpus.generate_batch(20, balanced=True)
    print(f"Generated {len(batch)} bugs:\n")

    from collections import Counter
    by_pipeline = Counter(name for name, _ in batch)
    by_type = Counter(bug.mutation.mutation_type for _, bug in batch)
    by_depth = Counter(bug.depth for _, bug in batch)

    print(f"  by pipeline: {dict(by_pipeline)}")
    print(f"  by type:     {dict(by_type)}")
    print(f"  by depth:    {dict(by_depth)}")

    for name, bug in batch[:5]:
        print(f"\n  [{name}] {bug.mutation.mutation_type} @ {bug.mutation.step_name} depth={bug.depth}")
        print(f"    {bug.mutation.description}")
        print(f"    changed: {bug.changed_fields[:4]}")
