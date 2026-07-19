# Demo kit — one rule per pipeline branch

Deterministic, LLM-free demo of the full DQC ReAct pipeline: the real
backend + real frontend, with the LLM client replaced by a scripted
responder (`demo_server.py`, same technique as the test suite).

## Run it

```bash
# 1. backend (repo root) — DEMO_SLEEP paces the phases for the eye
pip install -r requirements-dqc.txt uvicorn
DEMO_SLEEP=1.0 python -m uvicorn demo.demo_server:app --port 8000

# 2. frontend with /api proxied to the backend
cd DQC/app && npm ci
npx ng serve --proxy-config ../../demo/proxy.demo.json --port 4200
```

Open http://localhost:4200 — the demo dictionary and cases Excel are
already loaded (bundled assets). **Enable both experimental toggles**
(grounding + juez semántico), paste `reglas_demo.txt` into the textarea
(or upload it as test list) and press *Generar DQCs*.

## What each rule showcases

| # | Rule (reglas_demo.txt) | Branch exercised | Expected outcome |
|---|---|---|---|
| 1 | `DQC_PD_001: La PD…` | Clean first-attempt pass + metrics | ✓ 1 intento, 2 casos, **P 100% / R 100%**, juez ✓ |
| 2 | `DQC_EAD_002: El EAD…` | **Semantic judge rejection → correction** (query ran but checked the *compliant* rows) | ✓ en intento 2, feedback del juez visible, P/R 100% |
| 3 | `La LGD…` (sin id) | **Static validation catches a hallucinated field → correction**; metrics impossible without previous id | ✓ en intento 2, 1 caso detectado, **sin P/R** |
| 4 | `DQC_ECL_003: El ECL…` | **Execution error on the cases → correction** (reperformance formula) | ✓ en intento 2, 1 caso (ECL erróneo), P/R 100% |
| 5 | `El STAGE_IFRS9…` | **Exhausted attempts** (query never validates) | ✗ error tras 3 intentos |
| 6 | `…colateral…` | **Ambiguous — the model admits missing info** | ! ambigua con justificación |
| 7 | `…divisa…` | **Ambiguous — deterministic hallucination catch** (model claims `IMPORTE_DIVISA`, which doesn't exist) | ! ambigua citando el campo inexistente |

After the run, press **Evaluar Excel de datos** to exercise the stored-DQC
evaluation branch (per-check examples + P/R, incl. rule 5 absent because it
never persisted).

## Other branches

- **Ambiguous sheet inspection**: upload `fixtures/diccionario_ambiguo.xlsx`
  as dictionary — two dictionary-shaped sheets force the question + option
  buttons (low-confidence mapper path).
- **Unreadable workbook (400)**: rename any text file to `.xls` and upload
  it — clear Spanish error in chat instead of a 500.
- **Cases fixture design** (`fixtures/casos_demo.xlsx`): each caso violates
  exactly one known rule — see the docstring in `make_fixtures.py`;
  regenerate everything (fixtures + bundled UI assets) with
  `python demo/make_fixtures.py`.
