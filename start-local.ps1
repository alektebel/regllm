# RegLLM — local Windows launcher (no Docker).
#
# Usage:
#   .\start-local.ps1                  # start backend + frontend
#   .\start-local.ps1 -NoBrowser       # don't open the browser automatically
#   .\start-local.ps1 -WithOllama      # also start Ollama for LLM features
#
# Prerequisites (install once):
#   1. Python 3.11+   https://www.python.org/downloads/
#   2. Node.js 18+    https://nodejs.org/
#   3. (optional) Ollama  https://ollama.com/download  (only for Q&A / GraphRAG)

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$WithOllama
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Write-Stage($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Assert-Cmd($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$name' not found. $hint" -ForegroundColor Red
        exit 1
    }
}

# ── Prerequisites ─────────────────────────────────────────────────────────────
Write-Stage "Checking prerequisites"
Assert-Cmd "python" "Install Python 3.11+ from https://www.python.org/downloads/ (tick 'Add to PATH')"
Assert-Cmd "node"   "Install Node.js 18+ from https://nodejs.org/"
Assert-Cmd "npm"    "npm should come with Node.js"

$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pyVersion -lt [version]"3.11") {
    Write-Host "ERROR: Python 3.11+ required (found $pyVersion)" -ForegroundColor Red
    exit 1
}
Write-Host "  Python $pyVersion OK"
Write-Host "  Node $(node --version)  npm $(npm --version)  OK"

# ── Python virtual environment ────────────────────────────────────────────────
$VenvDir = Join-Path $Root ".venv"
if (-not (Test-Path (Join-Path $VenvDir "Scripts\activate.ps1"))) {
    Write-Stage "Creating Python virtual environment (.venv)"
    python -m venv $VenvDir
}

Write-Stage "Installing Python dependencies (first run installs torch — may take a few minutes)"
& "$VenvDir\Scripts\python.exe" -m pip install -q --upgrade pip
& "$VenvDir\Scripts\pip.exe" install -q -r "$Root\requirements.txt"
Write-Host "  Python packages OK"

# ── Node.js / frontend ────────────────────────────────────────────────────────
$FrontendDir = Join-Path $Root "frontend"
if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Stage "Installing frontend npm packages (first run)"
    Push-Location $FrontendDir
    npm install --silent
    Pop-Location
}
Write-Host "  npm packages OK"

# ── Optional: Ollama ──────────────────────────────────────────────────────────
if ($WithOllama) {
    Assert-Cmd "ollama" "Install Ollama from https://ollama.com/download then re-run with -WithOllama"
    Write-Stage "Starting Ollama service"
    $ollamaProc = Start-Process -FilePath "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 3
    # Pull default model if not already present
    $model = "qwen2.5:7b-instruct-q4_K_M"
    Write-Host "  Pulling $model (skip if already downloaded)..."
    ollama pull $model
    Write-Host "  Ollama ready on http://localhost:11434"
}

# ── Launch backend ────────────────────────────────────────────────────────────
Write-Stage "Starting FastAPI backend on http://localhost:8000"
$backendArgs = @(
    "-NoExit", "-Command",
    "Set-Location '$Root'; & '$VenvDir\Scripts\uvicorn.exe' api.main:app --host 0.0.0.0 --port 8000 --reload"
)
$backendProc = Start-Process powershell -ArgumentList $backendArgs -PassThru

# Wait until the backend is responsive
Write-Host "  Waiting for backend..."
$timeout = 60
$elapsed = 0
while ($elapsed -lt $timeout) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
        break
    } catch {
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
}
if ($elapsed -ge $timeout) {
    Write-Host "  Backend did not start in time. Check the backend window for errors." -ForegroundColor Yellow
}
Write-Host "  Backend OK"

# ── Launch frontend ───────────────────────────────────────────────────────────
Write-Stage "Starting Next.js frontend on http://localhost:3010"
$frontendEnv  = "INTERNAL_API_URL=http://localhost:8000"
$frontendArgs = @(
    "-NoExit", "-Command",
    "Set-Location '$FrontendDir'; `$env:INTERNAL_API_URL='http://localhost:8000'; npm run dev"
)
$frontendProc = Start-Process powershell -ArgumentList $frontendArgs -PassThru

# Wait until the frontend is responsive
Write-Host "  Waiting for frontend (compiles on first request, ~15s)..."
$elapsed = 0
while ($elapsed -lt 90) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:3010" -UseBasicParsing -TimeoutSec 3
        break
    } catch {
        Start-Sleep -Seconds 3
        $elapsed += 3
    }
}

Write-Host ""
Write-Host "RegLLM is running." -ForegroundColor Green
Write-Host ""
Write-Host "  Embedding Visualizer :  http://localhost:3010/embeddings"
Write-Host "  Diff Explainer       :  http://localhost:3010/diff"
Write-Host "  API (Swagger)        :  http://localhost:8000/docs"
Write-Host ""
Write-Host "  Close the two terminal windows to stop, or press Ctrl+C in each."

if (-not $NoBrowser) {
    Start-Process "http://localhost:3010/embeddings"
}
