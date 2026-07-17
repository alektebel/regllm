# ReAct DQC pipeline — architecture, flags, and reading list

The DQC generator (`api/routers/dqc.py` + `dqc_react.py`) runs each rule
through a fresh-context Reason+Act loop, streamed live to the chat UI over
SSE (`POST /dqc/generate_stream`):

```
plan (1 agent, all rules)
└─ per rule, fresh context:
   1. suficiencia   ¿campos identificados e interpretación inequívoca?
   │                └─ campos declarados verificados contra el diccionario
   │                   (campo inexistente ⇒ regla AMBIGUA, nunca se genera)
   2. [grounding]   (opcional) muestreo de valores reales del Excel de datos
   3. generacion    consulta SAS PROC SQL (SELECT * FROM t WHERE <violación>)
   4. validacion    estática (identificadores ∈ diccionario) +
   │                dinámica (ejecución sobre los casos, SQLite en memoria)
   5. [juicio]      (opcional) juez semántico: ¿implementa ESTA regla?
   └─ cualquier fallo de 3-5 realimenta a un agente fresco (máx. 3 intentos)
```

Con Excel de datos + columna `DQC_ID` + ids previos en las reglas
(`DQC_X: regla` / `[DQC_X] regla`): ejemplos de casos detectados,
precisión/recall por DQC y medias por ejecución. `POST /dqc/evaluate`
ejecuta los DQCs ya almacenados contra un Excel de casos sin LLM.

## Experimental flags (off by default — behaviour is unchanged unless set)

| Flag (form field)  | UI toggle                      | What it does |
|--------------------|--------------------------------|--------------|
| `value_grounding`  | "Grounding con valores reales" | Samples up to 8 distinct real values per relevant field from the cases Excel into the generation prompt, so domain comparisons (`TIPO = 'HIPOTECA'`) use values that exist. Requires the data Excel. Adds a `grounding` phase to the checklist. No extra LLM calls. |
| `semantic_judge`   | "Juez semántico"               | After a query passes static+dynamic validation, an LLM-as-judge agent checks it implements *this* rule (threshold, direction of violation, edge cases). A rejection feeds the correction loop like any validation error. Adds a `juicio` phase and one agent call per attempt. |

Both patterns are the industry standard for hallucination control in
NL→SQL agents: *semantic value grounding / column exploration* and the
*unit-tester / LLM-as-judge* refinement stage (see reading list).

## Reading list — what the industry does (2025-2026)

**Agent patterns (the canon):**
- ReAct: Synergizing Reasoning and Acting in Language Models —
  <https://arxiv.org/abs/2210.03629> (the pattern this pipeline is named after)
- Reflexion: Language Agents with Verbal Reinforcement Learning —
  <https://arxiv.org/abs/2303.11366> (self-correction from feedback, our loop)

**Text-to-SQL agents with execution feedback (state of the art):**
- ReFoRCE (self-refinement, format restriction, column exploration; SOTA on
  Spider 2.0) — <https://arxiv.org/abs/2502.00675>
- MARS-SQL (multi-agent grounding/generation/validation with execution
  feedback; SOTA on BIRD) — <https://openreview.net/forum?id=EURAfiUpVJ>
- CHESS (information retriever + unit tester; evidence that *execution
  results beat model introspection* as the correction signal) —
  <https://arxiv.org/abs/2405.16755>
- Benchmarks to know: BIRD (<https://bird-bench.github.io>) and Spider 2.0
  (<https://spider2-sql.github.io>) — execution accuracy is the metric,
  exactly what our cases-Excel precision/recall measures locally.

**Data-quality platforms with LLM rule agents (commercial baseline):**
- Gartner Magic Quadrant for Augmented Data Quality 2026 (agentic rule
  creation is now a scored capability) —
  <https://www.ataccama.com/blog/whats-new-in-the-2026-gartner-magic-quadrant-for-augmented-data-quality-solutions>
- Ataccama ONE AI agent (profiling → suggested rules) —
  <https://www.ataccama.com/platform/data-quality>
- Monte Carlo "Data + AI Observability" (agent traces as first-class
  telemetry) — <https://montecarlo.ai/blog-best-ai-observability-tools>
- Great Expectations (open-source declarative expectations — the
  deterministic layer LLM suggestions compile down to) —
  <https://greatexpectations.io>
- Tool landscape survey —
  <https://www.dqlabs.ai/blog/best-data-quality-tools-for-enterprise-use-in-2026-a-practitioners-guide/>

**Banking governance (why traceability + human validation matter):**
- BCBS 239 (risk data aggregation principles — the regulatory root of DQC
  work) — <https://www.bis.org/bcbs/publ/d239.htm>
- SR 26-2 / OCC 2026-13 model risk guidance explainer —
  <https://www.databricks.com/blog/model-risk-management-2026-bankers-guide-revised-interagency-guidance>
- EBA GL/2017/16 (PD/LGD estimation guidelines — the domain rules our
  regulation RAG indexes) — see `data/regulation/`.
