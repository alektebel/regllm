# Experiencias del Agente RegLLM

Almacén de experiencias aprendidas por el agente durante sesiones
de validación regulatoria. Cada archivo contiene una experiencia
(lección aprendida, bug descubierto, quirk, feedback) que el agente
puede recuperar y aplicar en sesiones futuras.

## Formato

Cada experiencia es un markdown con frontmatter YAML:

```markdown
---
id: exp_sw_fusion_bug
type: insight
priority: 0.7
tags: [bug, LGD, SW_FUSION, fusion]
fields: [LGD_ESTIMADA, MoC, ECL]
articles: []
source: "Sesión de análisis LGD (2025-06)"
feedback: false
---

# SW_FUSION=1 causa LGD_ESTIMADA missing en fusionados

Bug documentado: cuando SW_FUSION=1, el catálogo del contrato
fusionado se pierde y LGD_ESTIMADA resulta missing. Esto cascada
a MoC ( = 0.05 × . ) y ECL ( = PD × . × EAD) también missing.
```

## Index

| ID | Título | Tipo | Prioridad | Tags |
|---|---|---|---|---|
| exp_sw_fusion_bug | SW_FUSION causa LGD missing | insight | 0.7 | bug, LGD, SW_FUSION |
| exp_lgd_floor_corp | LGD floor CORP 45→50% | insight | 0.7 | LGD, floor, CORP |
| exp_hipoteca_floor | Hipoteca LGD floor 30% | insight | 0.6 | LGD, hipoteca, CRR |
| exp_pd_floor_feedback | Corrección PD floor retail | feedback | 1.0 | PD, floor, retail |
| exp_oread_join_bug | OR_EAD_TIT inflado JOIN | insight | 0.7 | bug, EAD, JOIN |
| exp_colateral_quirk | COLATERAL_FIN renombrado | quirk | 0.5 | colateral, V3 |
| exp_cure_rate_feedback | Cure rate NO aplica CORP | feedback | 1.0 | cure, LGD, CORP |
| exp_pd_flow | Flujo PD completo | insight | 0.6 | PD, floor |
| exp_stage3_conflict | Stage 3 multiplicador NO | feedback | 1.0 | LGD, Stage 3 |
| exp_ninguna_floor | LGD floor NINGUNA 0.50 | insight | 0.7 | LGD, floor |
| exp_cure_flag | CURE_FLAG hipotecas | insight | 0.5 | CURE_FLAG, LGD |
| exp_ead_ecl_feedback | EAD_TOTAL en ECL | feedback | 1.0 | EAD, ECL |
| exp_ventana_observacion | Ventana observación PD | insight | 0.6 | PD, ventana |
| exp_no_conformes | no_conformes floors | insight | 0.5 | validación |
| exp_moc | MoC 5% LGD | insight | 0.6 | MoC, LGD |
| exp_ecl_formula | ECL = PD × LGD_CON_MOC × EAD | insight | 0.6 | ECL |
| exp_provision_period_fase2 | Provision period fase 2 | insight | 0.5 | provision_period |
| bug_ead_crm_double_count | Ajuste CRM duplica reducción EAD | insight | 0.7 | bug, EAD, CRM, garantías |
| bug_staging_sicr_window | Ventana SICR fija ignora segmentación | insight | 0.7 | bug, staging, SICR, IFRS9 |
| bug_calibracion_pd_central_tendency | PD central tendency no aplicado | insight | 0.6 | bug, PD, calibración, EBA |
| dq_field_consistency_segmento | Inconsistencias SEGMENTO maestro/operativa | insight | 0.6 | data quality, segmento |
| gap_ifrs9_lgd_downturn | LGD downturn no implementado | insight | 0.6 | gap, IFRS9, LGD, downturn |
| gap_bcbs239_data_lineage | BCBS 239 lineage incompleto | insight | 0.7 | gap, BCBS239, data lineage |
| bug_tasa_recuperacion_fija | Tasa recuperación fija invalida backtesting | insight | 0.8 | bug, LGD, recovery, segmento |
| xs_bde_cir_6_2016_art_16 | Art.16 multiplicador Stage 3 | insight | 0.7 | BdE, regulation, Stage 3 |
| com_eba_gl_monitoring_2022 | EBA/GL/2022/01 monitoring IRB | insight | 0.6 | EBA, monitoring, GL |
| ops_data_freshness_alert | Data freshness insuficiente reporting | insight | 0.5 | ops, data freshness |
| edge_kirb_ead_flow | KIRB usa OR_EAD en vez de EAD_TOTAL | insight | 0.6 | KIRB, EAD, securitización |
| email_eba_qna_pd_floor | EBA Q&A confirma PD floor segmentado | insight | 0.7 | EBA, Q&A, PD floor |
| mv_pd_rating_migration_matrix | Matriz migración ratings inestable | insight | 0.6 | model validation, ratings |
| dq_lgd_outside_range | LGD fuera de rango [0,1] en 2.3% | insight | 0.7 | data quality, LGD, rango |
| excel_provision_period_matrix | Matriz provision_period no segmentada | insight | 0.6 | excel, provision_period |
| com_crr_art_178_default_definition | Default definition CRR Art.178 | insight | 0.6 | CRR, default, definition |

## Poblar el grafo

Para poblar el grafo de conocimiento con estas experiencias:

```python
from src.knowledge.graph_store import GraphStore
from src.knowledge.experience_store import ExperienceStore
import yaml

store = GraphStore("data/knowledge/regllm.kuzu")
exp_store = ExperienceStore(store)

# Cargar cada .md y crear nodos
# (script en scripts/seed_experiences.py)
```
