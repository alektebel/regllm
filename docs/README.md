# Documentation index

## APDQ — auditor-parity data quality (start here)

| Document | What it covers |
|---|---|
| **[`APDQ_HANDBOOK.md`](APDQ_HANDBOOK.md)** | **The consolidated reference**: proposal → market → regulation (BCBS 239/RDARR, GL/2017/16, CRR, Anejo IX, COREP/FINREP, AnaCredit/CIRBE, BIRD/IReF) → the standard → implementation reference → certification protocol → worked proof → glossary. Read this first; everything below is the deeper cut it links to. |
| [`AUDITOR_PARITY_STANDARD.md`](AUDITOR_PARITY_STANDARD.md) | The normative design spec: the auditor-parity claim, the lineage completeness argument, the 12-class taxonomy, the two-level conformance model, the gap plan. |
| [`MVP_ROADMAP.md`](MVP_ROADMAP.md) | Build plan with acceptance gates and current status; the larger-product sequence. |
| [`../apdq/README.md`](../apdq/README.md) | Implementation manual: quickstart, module map, manifest format reference, defect-class status, expansions E1–E11 with their code seams, honest status. |
| [`DATA_QUALITY_INDUSTRY_SOTA_2026.md`](DATA_QUALITY_INDUSTRY_SOTA_2026.md) | Market research: tool landscape, budgets and staffing, LLM-based DQ (vendor + academic), the regulatory bar, path to 100% coverage. |
| [`DATA_QUALITY_SPAIN_NICHE.md`](DATA_QUALITY_SPAIN_NICHE.md) | The Spanish-bank niche: market structure, Spain-specific regulation, vendor/consultancy landscape, go-to-market. |
| [`BRANCHES.md`](BRANCHES.md) | Branch map of this repository, grouped by feature, with recommended actions. |

## DQC generator & eval (the original harness)

| Document | What it covers |
|---|---|
| [`../DQC/eval/README.md`](../DQC/eval/README.md) | The mutation-testing eval harness for the DQC agent: defect catalog, trap protocol, scoring, article-coverage certification. |
| [`EVALUATION.md`](EVALUATION.md) | Evaluation methodology. |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | AWS production deployment (ECS Fargate + Bedrock). |
| [`REGULATION_RAG.md`](REGULATION_RAG.md) | The regulation knowledge base / RAG. |
| [`audit_system.md`](audit_system.md), [`audit_system_v2_ideas.md`](audit_system_v2_ideas.md) | Earlier audit-system notes. |

## SAS field-diff explainer

Documented in the repository [`README.md`](../README.md) (root).
