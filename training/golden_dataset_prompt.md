# Prompt: Generate Golden Evaluation Dataset for RegLLM Agent

## Context

You are creating a **golden evaluation dataset** for a banking regulation compliance agent. The agent answers questions about COREP/FINREP IRB reporting by calling tools that:

1. **Trace SAS code lineage** (`trace_dependencies`, `inspect_lineage`) — BFS trace of which fields feed into a target field through SAS DATA steps and PROC SQL
2. **Search documentation** (`search_docs`, `get_field_definition`) — BM25 over markdown docs describing field semantics, table schemas, and changelog
3. **Search regulation** (`search_regulation`, `query_regulation`, `find_governing_rules`) — GraphRAG over regulatory knowledge graph (CRR, EBA GL/2017/16, Circular 6/2016 BdE)

The agent runs on a small local model (3-4B params) and has access to:
- SAS pipeline code (3 files: data loading, EAD enrichment, LGD floor application + ECL calculation)
- Markdown docs corpus (17 sections: field definitions, table schemas, changelog)
- Regulatory knowledge graph (CRR articles, EBA guidelines, BdE circulars linked to database columns)

## SAS Pipeline Summary

The pipeline has 3 steps:
1. `proj_01_carga_ciclos.sas` — Loads raw cycle data from `mylib.ciclos_recuperacion`, filters active cycles
2. `proj_02_enriquecimiento_ead.sas` — Enriches EAD with titulizacion adjustments, aggregates by cycle
3. `proj_03_suelos_lgd.sas` — Applies regulatory LGD floors (CRR Art.154/161), PD floor (Art.160), computes MoC and ECL

Key fields: `LGD_ESTIMADA`, `PD_ESTIMADA`, `EAD`, `ECL`, `RWA`, `LGD_FLOOR_APLICADO`, `MoC`, `COLATERAL_TIPO`, `SEGMENTO`

Known bug: merged contracts (SW_FUSION=1) have missing LGD_ESTIMADA from absorbed entity catalog, causing MoC=. and ECL=.

## Task

Generate a JSON array of **20-30 evaluation examples**. Each example should have:

```json
{
  "id": "eval_001",
  "question": "The natural language question the user would ask",
  "category": "lineage | regulation | field_definition | bug_detection | cross_reference",
  "difficulty": "easy | medium | hard",
  "expected_tools": ["tool_name_1", "tool_name_2"],
  "expected_answer_contains": ["key phrase 1", "key phrase 2"],
  "expected_fields_mentioned": ["FIELD_A", "FIELD_B"],
  "gold_answer_summary": "1-3 sentence reference answer",
  "notes": "Optional: what makes this a good test case"
}
```

## Category Distribution

- **lineage** (6-8): "What feeds into X?", "Where does Y come from?", "Trace the origin of Z"
- **regulation** (5-7): "What regulation governs X?", "What's the LGD floor for mortgages?", "Which CRR article sets the PD floor?"
- **field_definition** (3-4): "What is EAD?", "Define LGD_ESTIMADA", "What does STAGE_IFRS9 mean?"
- **bug_detection** (3-4): Questions that should lead the agent to discover the SW_FUSION bug or other pipeline issues
- **cross_reference** (3-5): Questions requiring both lineage + regulation, or docs + code, to answer correctly

## Difficulty Guidelines

- **easy**: Single tool call, direct answer. "What is EAD?" → `get_field_definition("EAD")`
- **medium**: 2-3 tool calls, requires combining information. "What regulatory floor applies to LGD_ESTIMADA for mortgages?"
- **hard**: Multi-step reasoning, may require the agent to discover non-obvious connections. "Why might ECL be missing for some cycles?"

## Quality Criteria

- Questions should be realistic (what a bank analyst or auditor would actually ask)
- Expected answers must be **verifiable** against the actual tool outputs
- Include some questions where the correct answer is "I don't have enough information" or requires the agent to ask for data
- Include adversarial cases: questions about fields not in the pipeline, ambiguous questions
- All regulatory references must be real (CRR Art. 154, 160, 161; EBA GL/2017/16; etc.)

## Output

Return ONLY the JSON array. No explanation needed.
