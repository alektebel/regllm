# Static UI showcase (GitHub Pages)

A clickable demo of the DQC workflow with **no backend, no model, no AWS** —
the real Angular app driven by an in-browser fake backend.

**Live:** `https://<owner>.github.io/<repo>/` (published by
`.github/workflows/pages.yml` on every push to `main` that touches
`DQC/app/**`).

## Enabling it (one-time, repo settings)

GitHub → **Settings → Pages → Build and deployment → Source: GitHub
Actions**. Then push to `main` (or run the workflow manually from the
Actions tab). The workflow builds with `--base-href /<repo>/` and adds
`.nojekyll` so Angular's hashed assets survive.

## Demo mode

`src/app/demo/demo-backend.ts` intercepts every API call in `DqcService`
and returns canned template data with realistic delays:

- inspection proposal (sheet + column mapping),
- a replayed plan-mode run — plan, per-item phases, a judge rejection with
  a corrected second attempt, an ambiguous rule, decision traces,
- detected cases with precision/recall, and the evaluation summary,
- a mutable check store, so validar/rechazar actually work in the demo.

It activates when the page is served from `*.github.io` **or** with
`?demo=1` in the URL — so `http://localhost:4200/?demo=1` previews the
exact showcase locally, while the normal app keeps hitting the real API.

## The three UI layers

1. **Proyectos** — list of data-quality projects, `+ Nuevo proyecto`
   (name, target table, attach dictionary / cases / test-list once).
   Stored in `localStorage`, so the demo persists across reloads.
2. **Generar DQCs** — the ReAct chat: plan checklist, live decision trees,
   detected cases, `Evaluar Excel de datos`.
3. **Editar DQCs** — review layer, ordered for reading:
   **① Cómo se decidió** (the persisted decision trace: each question, its
   Sí/No answer, retries and the judge's reasons) → **② Consulta SQL** →
   **③ Casos detectados** (with P/R) → metadata. Validar / Rechazar from
   the toolbar.

## Local preview of the static build

```bash
cd DQC/app && npx ng build
cd dist/dqc-app/browser && python3 -m http.server 4300
# open http://localhost:4300/?demo=1
```
