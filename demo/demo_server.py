"""Demo backend: the REAL FastAPI app + ReAct pipeline with the LLM client
replaced by a scripted responder, so every pipeline branch can be seen
deterministically without Ollama/Bedrock/Azure.

Run from the repo root:

    python -m uvicorn demo.demo_server:app --port 8000

then serve the frontend against it (see demo/README.md). Each rule in
demo/reglas_demo.txt is keyed to showcase one branch — the mapping table
lives in the README. DEMO_SLEEP (seconds, default 1.0) paces the LLM calls
so the live checklist phases are visible.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("REGLLM_LLM", "stub")

from api.main import app  # noqa: E402
import api.routers.dqc as dqc_router  # noqa: E402

SLEEP = float(os.getenv("DEMO_SLEEP", "1.0"))

_DQC_FIELDS = {
    "tipo": "rango", "severidad": "bloqueante", "periodicidad": "mensual",
    "referencia_regulatoria": "EBA GL/2017/16",
}


def _dqc(dqc_id, variable, descripcion, sql, condicion, campos, **extra):
    d = {**_DQC_FIELDS, "dqc_id": dqc_id, "variable": variable,
         "descripcion": descripcion, "regla_sql": sql,
         "condicion_error": condicion, "campos_entrada": campos,
         "umbral": extra.pop("umbral", ""),
         "justificacion": extra.pop("justificacion", descripcion)}
    d.update(extra)
    return {"dqcs": [d]}


def _rule_key(text: str) -> str:
    """Which demo scenario a rule belongs to (order matters: ECL mentions
    PD/LGD/EAD, so it must be checked first)."""
    t = text.lower()
    for key in ("colateral", "divisa", "ecl", "stage", "lgd", "ead", "pd"):
        if key in t:
            return key
    return "pd"


class DemoLLM:
    """Scripted responses keyed on the system prompt + the rule text."""

    def __init__(self):
        self.judge_seen: set[str] = set()

    def chat_json(self, system, user, **kwargs):
        time.sleep(SLEEP)
        u = user.lower()

        # ── workbook mapper (inspection) ──────────────────────────────────
        if "identifica diccionarios" in system:
            if "campos_2025" in u:
                # ambiguous workbook: two dictionary-shaped sheets
                return {"sheet": "CAMPOS_2025",
                        "column_mapping": {"field": "Field", "type": "Type",
                                           "description": "Description"},
                        "confidence": 0.45,
                        "question": "Hay dos hojas con estructura de "
                                    "diccionario, ¿cuál corresponde a la "
                                    "cartera vigente?"}
            return {"sheet": "DICCIONARIO",
                    "column_mapping": {"field": "Field", "type": "Type",
                                       "description": "Description",
                                       "nullable": "Null", "formula": "Formula"},
                    "confidence": 0.93, "question": None}

        # ── planner ───────────────────────────────────────────────────────
        if "SEPARARLAS" in system:
            acciones = {
                "pd": "Check de rango sobre PD_ESTIMADA (0 ≤ PD ≤ 1)",
                "ead": "Check de rango sobre EAD_TOTAL (≥ 0)",
                "lgd": "Check de rango sobre LGD_ESTIMADA (0 ≤ LGD ≤ 1)",
                "ecl": "Reperformance de la fórmula ECL = PD × LGD × EAD",
                "stage": "Check de dominio sobre STAGE_IFRS9 {1,2,3}",
                "colateral": "Identificar el campo de colateral en el diccionario",
                "divisa": "Identificar el campo de divisa/importe original",
            }
            plan = []
            for line in user.splitlines():
                line = line.strip()
                if not line or ". " not in line:
                    continue
                regla = line.split(". ", 1)[1]
                plan.append({"id": len(plan) + 1, "regla": regla,
                             "accion": acciones[_rule_key(regla)]})
            return {"plan": plan}

        # ── sufficiency ───────────────────────────────────────────────────
        if "SUFICIENTE" in system:
            rule = u.split("regla dqc:")[-1].split("diccionario de campos")[0]
            key = _rule_key(rule)
            if key == "colateral":
                # branch: the model ADMITS the info is missing → ambigua
                return {"suficiente": False, "campos": [],
                        "interpretacion": "Validar el colateral de cada operación",
                        "falta": "El diccionario no contiene ningún campo de "
                                 "colateral o garantías. Indica qué campo lo "
                                 "describe (p.ej. COLATERAL_TIPO) o añádelo."}
            if key == "divisa":
                # branch: the model CLAIMS a field that does not exist —
                # the deterministic check flips this to ambigua
                return {"suficiente": True, "campos": ["IMPORTE_DIVISA"],
                        "interpretacion": "Convertir importes en divisa a EUR",
                        "falta": ""}
            campos = {"pd": ["PD_ESTIMADA"], "ead": ["EAD_TOTAL"],
                      "lgd": ["LGD_ESTIMADA"], "stage": ["STAGE_IFRS9"],
                      "ecl": ["ECL", "PD_ESTIMADA", "LGD_ESTIMADA", "EAD_TOTAL"],
                      }[key]
            return {"suficiente": True, "campos": campos,
                    "interpretacion": f"Campos implicados: {', '.join(campos)}",
                    "falta": ""}

        # ── semantic judge ────────────────────────────────────────────────
        if "revisor experto" in system:
            if "ead_total >= 0" in u:
                # branch: query runs but checks the WRONG thing → reject
                return {"correcto": False, "confianza": 0.85,
                        "motivo": "El sentido está invertido: selecciona las "
                                  "filas que CUMPLEN la regla (EAD ≥ 0) en "
                                  "lugar de las que la violan (EAD < 0)."}
            return {"correcto": True, "confianza": 0.92,
                    "motivo": "La consulta selecciona exactamente las filas "
                              "que violan la regla."}

        # ── SAS generation (with correction-loop variants) ────────────────
        if "experto en SAS" in system:
            rule = (u.split("regla dqc:")[-1]
                    .split("valores reales")[0]
                    .split("la consulta anterior")[0])
            corrected = "la consulta anterior falló la validación" in u
            key = _rule_key(rule)
            table = "mylib.ciclos_recuperacion"

            if key == "pd":       # branch: clean first-attempt pass
                return _dqc("DQC_PD_ESTIMADA_001", "PD_ESTIMADA",
                            "La PD estimada debe estar en el rango [0, 1]",
                            f"SELECT * FROM {table} WHERE PD_ESTIMADA > 1 OR PD_ESTIMADA < 0",
                            "PD_ESTIMADA fuera de [0, 1]", ["PD_ESTIMADA"],
                            umbral="0 ≤ PD ≤ 1")
            if key == "ead":      # branch: semantically wrong → judge → fix
                sql = (f"SELECT * FROM {table} WHERE EAD_TOTAL < 0" if corrected
                       else f"SELECT * FROM {table} WHERE EAD_TOTAL >= 0")
                return _dqc("DQC_EAD_TOTAL_001", "EAD_TOTAL",
                            "El EAD total no puede ser negativo", sql,
                            "EAD_TOTAL < 0", ["EAD_TOTAL"], umbral="EAD ≥ 0")
            if key == "lgd":      # branch: hallucinated field → static → fix
                sql = (f"SELECT * FROM {table} WHERE LGD_ESTIMADA > 1 OR LGD_ESTIMADA < 0"
                       if corrected
                       else f"SELECT * FROM {table} WHERE LGD_INEXISTENTE > 1")
                return _dqc("DQC_LGD_ESTIMADA_001", "LGD_ESTIMADA",
                            "La LGD estimada debe estar en el rango [0, 1]", sql,
                            "LGD_ESTIMADA fuera de [0, 1]", ["LGD_ESTIMADA"],
                            umbral="0 ≤ LGD ≤ 1")
            if key == "ecl":      # branch: execution error → dynamic → fix
                good = (f"SELECT * FROM {table} WHERE "
                        f"ABS(ECL - PD_ESTIMADA * LGD_ESTIMADA * EAD_TOTAL) > 0.01")
                sql = good if corrected else good + " GROUP BY"
                return _dqc("DQC_ECL_001", "ECL",
                            "ECL debe reperformar como PD × LGD × EAD",
                            sql, "ECL difiere de la fórmula documentada",
                            ["ECL", "PD_ESTIMADA", "LGD_ESTIMADA", "EAD_TOTAL"],
                            tipo="formula", umbral="tolerancia 0.01")
            # key == "stage" — branch: never valid → exhausted attempts
            return _dqc("DQC_STAGE_001", "STAGE_IFRS9",
                        "El stage IFRS9 debe pertenecer al dominio {1,2,3}",
                        f"SELECT * FROM {table} WHERE CAMPO_FANTASMA NOT IN (1, 2, 3)",
                        "STAGE_IFRS9 fuera de dominio", ["STAGE_IFRS9"],
                        tipo="completitud")

        return {}


_demo = DemoLLM()
dqc_router.get_client = lambda: _demo
dqc_router.get_inspect_client = lambda: _demo
