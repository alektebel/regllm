# Niche Analysis: Data Quality for Spanish Banks

Companion to [`DATA_QUALITY_INDUSTRY_SOTA_2026.md`](DATA_QUALITY_INDUSTRY_SOTA_2026.md),
narrowing the market to **Spanish credit institutions**. Covers the
market structure, the Spain-specific regulatory stack, who Spanish
banks buy from today, and what "100% regulation + coherence coverage"
should mean when the target regulations are the ones Spanish banks
actually answer to.

Date of research: 2026-07-11. Sources in [References](#references).

---

## 1. Why Spain is the natural niche for this repo

The codebase is already Spanish-bank-shaped, which is a real moat:

- The regulation corpus (EBA GL/2017/16 PD & LGD) is ingested **in
  Spanish** — `DQC/coverage/applicability.yaml` section titles are the
  Spanish official translation, the one Spanish validation and
  auditing teams actually work from.
- The eval database mirrors a Spanish IRB shop: `CICLOS_CALIBRADOS`,
  `BASILEA_MENSUAL`, `CONTRATOS`, `COLATERALES`, `FECHA_ADJUDICACION`
  (foreclosed-asset lifecycle — an Anejo IX concern), MoC categories
  A/B/C per EBA GL §43–44.
- **SAS parsing is the differentiator.** Spanish banks' IRB
  calibration and IFRS 9 / Anejo IX provisioning engines are
  overwhelmingly SAS batch pipelines maintained by risk-methodology
  teams. The SAS AST compiler + field-diff explainer half of RegLLM
  speaks directly to the artifact those teams live in; no
  observability vendor parses SAS lineage.

## 2. Market structure

- **10 Spanish banking groups are Significant Institutions** under
  direct ECB supervision: Santander (the only Spanish G-SIB), BBVA,
  CaixaBank, Sabadell, Bankinter, Unicaja, Abanca, Kutxabank, Ibercaja
  and Cajamar. These are the IRB users and the accounts with
  six-figure DQ budgets.
- **~73 Less Significant Institutions** (rural savings/cooperative
  banks, specialised lenders) supervised by Banco de España. Mostly
  standardised approach, thin data teams, consultancy-dependent —
  they buy outcomes, not platforms.
- Practical consequence: the addressable market is **~10–15 direct
  accounts plus 3–5 consultancies as channel**, not hundreds of
  logos. Sales motion is reference-driven; one SI validation
  department as design partner is worth more than any marketing.

## 3. The Spain-specific regulatory stack

Layered on top of the global BCBS 239 / RDARR picture from the main
memo:

| Layer | Instrument | Data-quality relevance |
|---|---|---|
| ECB SSM | RDARR guide (2024), SREP from 2025, Internal Model Investigations | TRIM ran 200 on-site model investigations at 65 SIs (2016–21); data-quality and validation-governance findings were among the recurrent severe ones. IMIs continue for model changes — every IRB change triggers a data-quality review. |
| EBA | GL/2017/16 (PD/LGD — already covered by this repo), CRR art. 174 data requirements, COREP/FINREP **DPM validation rules** | EBA publishes machine-readable validation rules quarterly; Banco de España **rejects submissions that fail them**, forcing fix-and-resubmit cycles. These rules are an importable, pre-blessed check corpus. |
| Banco de España | **Circular 4/2017 + Anejo IX** (IFRS 9 classification & provisioning, foreclosed assets; amended repeatedly, latest Circular 1/2025), Memoria de Supervisión | The 2025 Memoria names RDARR remediation a supervisory priority and announces "**more intrusive**" supervisory actions for entities under its direct watch. Anejo IX compliance is where accounting, risk and data quality collide — stages, coverage levels, collateral haircuts, adjudicados. |
| Granular registers | **CIRBE** (central credit register) + **AnaCredit** | Loan-by-loan monthly reporting; the same exposure appears in CIRBE, AnaCredit, FINREP and the IRB datamart and must reconcile — exactly the cross-table reconciliation defect family (D48–D57) the eval DB already models. |

The supervisory temperature matters commercially: BdE issued 48
requirements/recommendations in its latest cycle and has said 2025+
inspections will be more intrusive, while the ECB assesses RDARR in
SREP with escalation threats. Every one of those letters lands on a
"Calidad del Dato" office that must evidence remediation — which is an
audit-pack sale.

## 4. What Spanish banks use and pay today

- **Consultancies do most of the DQ rule authoring**: Management
  Solutions (Madrid; the reference IRB/provisioning consultancy),
  NTT Data (ex-everis), Minsait (Indra), Accenture and the Big 4.
  Rules are delivered as SQL/SAS batteries plus Excel evidence —
  hand-written, per-engagement, re-billed at every regulatory change.
  Day rates put a 3–4 person DQ-rules engagement at roughly
  €200K–€500K, recurring.
- **Platforms**: SAS is the incumbent execution engine (DQ checks
  often live inside the same SAS estate as the models). Stratio
  (Madrid-based data fabric with DQ/governance; its Datio venture was
  a BBVA alliance) is the strongest local platform player; Informatica
  and Collibra appear at governance layer in the largest groups;
  regulatory-reporting vendors (Wolters Kluwer, Regnology and similar)
  bundle EBA validation-rule execution.
- Global pricing tiers from the main memo apply to the SIs
  ($200K–$500K/yr stacks); the 73 LSIs mostly cannot buy at that level
  — they outsource to consultancies or run manual controls, which is
  precisely the gap a generated-and-certified check suite fills
  cheaply.

## 5. What "100% coverage" means for a Spanish bank

The three-artifact certification (approved applicability map ∧
zero-TODO matrix ∧ 100% harness recall) stays the same; the *corpus
set* changes. Priority order for extending `DQC/coverage/`:

1. **EBA GL/2017/16 (ES)** — done (current target; close the 58
   pending reviews and 8 partial cells per the main memo).
2. **Anejo IX / Circular 4/2017** — the highest-value Spanish
   extension. New defect families: stage-classification coherence
   (normal / vigilancia especial / dudoso), minimum coverage levels by
   collateral type and vintage, adjudicado lifecycle
   (`FECHA_ADJUDICACION`, sale discounts), refinanciación/cure-period
   rules. The eval schema already carries most of the needed fields.
3. **ECB RDARR guide** — governance-level; maps to the audit pack
   itself rather than row-level oracles (evidence of lineage,
   ownership, monitoring cadence).
4. **EBA COREP/FINREP validation rules** — import the quarterly EBA
   package as a *generated* check layer; the pitch is catching
   rejections **before** submission to BdE instead of after.
5. **CIRBE/AnaCredit ↔ datamart reconciliation** — generalise the
   D48–D57 cross-table defects to cross-*register* checks (same
   exposure, four reporting surfaces).

Each corpus gets its own `applicability.yaml` + `regulation_ref`
namespace so the coverage matrix reports per-regulation certification —
one audit pack per supervisory conversation (IMI, SREP, BdE
inspection, external audit).

## 6. Niche go-to-market implications

- **Buyer personas**: head of Calidad del Dato / CDO office (SIs),
  head of validación interna (model risk), intervención
  general/financial reporting for the Anejo IX angle. For LSIs, the
  consultancy is the buyer — position the DQC generator + harness as
  the consultancy's delivery accelerator (white-label the audit pack).
- **The pitch in local terms**: "cada párrafo aplicable de la GL/2017/16
  y del Anejo IX tiene un control que demostrablemente dispara" —
  proof-by-execution coverage, versioned in git, regenerable on every
  circular amendment (Circular 1/2025 changed Anejo IX again; a corpus
  re-hash flips affected sections to `pending` and the delta is the
  engagement).
- **Language**: keep generation, citations and reports bilingual
  ES/EN — working teams operate in Spanish, ECB JSTs read English.
- **Local-LLM deployment matters**: Spanish banks are conservative on
  data egress; the existing GGUF/Ollama path (checks generated
  on-prem, schema never leaves the bank) is a compliance selling
  point against US SaaS observability vendors.

---

## References

Market structure & supervision:
- [ECB — supervisory banking statistics on significant institutions (Q2 2025)](https://www.bankingsupervision.europa.eu/press/pr/date/2025/html/ssm.pr250917~6554cd2316.en.html)
- [Wikipedia — List of banks in Spain (SI/LSI counts)](https://en.wikipedia.org/wiki/List_of_banks_in_Spain)
- [ECB — Internal model investigations](https://www.bankingsupervision.europa.eu/activities/internal_models/imi/html/index.en.html)
- [ECB — TRIM project report (2021)](https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.trim_project_report~aa49bb624c.en.pdf)

Banco de España:
- [Memoria de Supervisión 2025 (PDF)](https://www.bde.es/f/webbe/Secciones/Publicaciones/PublicacionesAnuales/MemoriaSupervisionBancaria/25/MemoriaSupervision2025.pdf)
- [Forbes España — actuaciones supervisoras "más intrusivas" en 2025](https://forbes.es/economia/676665/el-banco-de-espana-realizara-en-2025-actuaciones-supervisoras-mas-intrusivas-en-las-entidades/)
- [BOE — Circular 4/2017 del Banco de España](https://www.boe.es/buscar/doc.php?id=BOE-A-2017-14334)
- [Management Solutions — amendments to Circular 4/2017](https://www.managementsolutions.com/en/publications-and-events/regulatory-notes/technical-notes-on-regulations/amendments-circular-42017-public-and-confidential-financial-information-standards-and-formats)
- [finReg360 — Memoria de supervisión de 2024](https://finreg360.com/alerta/el-banco-de-espana-publica-su-memoria-de-supervision-de-2024/)

EBA reporting & validation rules:
- [EBA — Reporting frameworks (DPM, validation rules)](https://www.eba.europa.eu/risk-and-data-analysis/reporting/reporting-frameworks)
- [RegReportingDesk — EBA 4.3 DPM changes for COREP/FINREP](https://regreportingdesk.com/eba-43-dpm-changes-corep-finrep/)
- [RegReportingDesk — COREP reporting & resubmission cycle](https://regreportingdesk.com/corep-reporting-explained/)

Vendor landscape:
- [Stratio BD — company profile / funding](https://www.thesaasnews.com/news/stratio-bd-raises-65-million-in-series-c)
- [Datio — BBVA–Stratio alliance](https://www.linkedin.com/company/datio-big-data)
- [Minsait — Banking](https://www.minsait.com/en/industries/banking)
- [NTT DATA — Banking and Financial Services](https://us.nttdata.com/en/industries/banking-and-financial-services)
- [Wolters Kluwer — EBA supervisory reporting (CCH Tagetik)](https://www.wolterskluwer.com/en/solutions/cch-tagetik/eba-regulatory-reporting)
