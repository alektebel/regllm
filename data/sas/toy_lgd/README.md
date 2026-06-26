# Toy LGD — Agent Backtrace Curriculum

A self-contained SAS LGD estimation codebase with progressive difficulty.
The agent reads code + documentation, requests data extracts via PROC SQL,
and backtraces through the computation graph to find planted bugs.

## How it works

1. Agent reads `meta/*.md` to understand the schema and pipeline
2. Agent reads the `lgd.sas` for a level
3. Agent requests queries (PROC SQL / PROC PRINT) to inspect intermediate tables
4. You run those queries in SAS and paste output back
5. Agent backtraces to identify the bug
6. When solved, move to next level

## Levels

| # | Level | Bug type | Tables needed |
|---|-------|----------|---------------|
| 01 | Easy | Variable swap in floor condition | CICLOS |
| 02 | Easy | Missing COALESCE for fusion contracts | CICLOS + CONTRATOS |
| 03 | Easy | Off-by-one DPD backstop | CICLOS |
| 04 | Medium | Wrong WHERE filter excluding valid rows | CICLOS |
| 05 | Medium | Fusion join duplication (OR_EAD inflated) | CICLOS + CONTRATOS + BASILEA_MENSUAL |
| 06 | Hard | Wrong BY-group in segment average for MoC | CICLOS |
| 07 | Harder | Hidden type coercion in comparison | CICLOS |
| 08 | Hardest | Compound: two independent bugs interacting | CICLOS + CONTRATOS + BASILEA_MENSUAL |

## Meta documentation

See `meta/` for table schemas and pipeline overview.
