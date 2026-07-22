# DQC frontend (Angular) — local setup

The DQC UI is an **Angular 18** app. It talks to the FastAPI backend through
a dev proxy, so `/api/*` calls are forwarded for you — no CORS setup needed.

Two proxy configs ship with the app:

| Config                  | Forwards `/api/*` to            | Use when                          |
|-------------------------|---------------------------------|-----------------------------------|
| `proxy.conf.json`       | `http://localhost:8001` (strips `/api`) | running the backend locally |
| `proxy.aws.conf.json`   | AWS API Gateway (keeps `/api`)  | using the Lambda backend          |

`npm start` uses `proxy.conf.json` (local backend).

> Set `target` in `proxy.aws.conf.json` to the API Gateway Invoke URL. See
> [`docs/AWS_LAMBDA_CONSOLE_SETUP.md`](../../docs/AWS_LAMBDA_CONSOLE_SETUP.md)
> for the console-only backend setup.

---

## Windows setup with npm

### 1. Install Node.js

Install the **LTS** build (18 or 20) from <https://nodejs.org> (or
`winget install OpenJS.NodeJS.LTS`). Verify in a fresh PowerShell / CMD:

```powershell
node --version   # v18.19+ or v20+
npm --version
```

You do **not** need a global Angular CLI — `ng` is installed locally as a
dev dependency and run via the npm scripts.

### 2. Install dependencies

```powershell
cd DQC\app
npm install
```

### 3. Start the backend (separate terminal)

The proxy expects the API on port **8001**. From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.main:app --port 8001 --reload
```

> Tip: for real (non-stub) DQC generation you need an LLM backend — a local
> Ollama, a `GEMINI_API_KEY`, or AWS Bedrock. See the root `README.md`.

### 4. Start the UI

```powershell
cd DQC\app
npm start
```

Open **http://localhost:4200**. Check the API is reachable:

```powershell
curl http://localhost:4200/api/health
# → {"status":"ok","llm_backend":"ollama"}   (or gemini / bedrock / stub)
```

---

## Point the UI at a deployed AWS backend

Skip the local backend entirely and proxy to AWS:

```powershell
cd DQC\app
npm install
npx ng serve --proxy-config proxy.aws.conf.json --port 4200 --open
```

Edit the `target` in `proxy.aws.conf.json` to change which API Gateway it hits.

## Production build

```powershell
npm run build          # → dist/dqc-app/ (served by nginx in the container)
```

## Troubleshooting

- **`/api/*` returns 404 / connection refused** — the backend isn't running
  on `:8001` (or you started it on the 8000 default). Restart uvicorn with
  `--port 8001`.
- **`ng` not recognized** — run `npm install` first; the scripts use the
  local CLI. Use `npx ng ...` for ad-hoc commands.
- **Node engine errors on `npm install`** — you're below Node 18.19; install
  the current LTS.
