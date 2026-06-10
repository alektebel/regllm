# RegLLM — Windows one-command launcher.
#
# Usage:
#   .\start.ps1                                # API + Web only (no LLM)
#   .\start.ps1 -WithModel                     # include Ollama + model auto-pull
#   .\start.ps1 -Model qwen2.5:7b-instruct-q4_K_M   # specific model
#   .\start.ps1 -NoBrowser                     # skip auto-opening the UI
#   .\start.ps1 -Rebuild                       # force `docker compose build --no-cache`
#
# Prerequisites: Docker Desktop with WSL 2 backend. Add ~12 GB free disk if
# using -WithModel with the default 14B Qwen. Smaller alternatives in .env.example.

[CmdletBinding()]
param(
    [string]$Model     = "",
    [int]   $WaitSecs  = 600,
    [switch]$WithModel,
    [switch]$NoBrowser,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

function Write-Stage($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Test-Cmd($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# ── Prerequisites ───────────────────────────────────────────────────────────
Write-Stage "Checking Docker"
if (-not (Test-Cmd docker)) {
    Write-Host "Docker not found. Install Docker Desktop and re-run:" -ForegroundColor Red
    Write-Host "  https://www.docker.com/products/docker-desktop/"
    exit 1
}
try {
    docker compose version | Out-Null
} catch {
    Write-Host "`docker compose` is not available. Update Docker Desktop." -ForegroundColor Red
    exit 1
}
try {
    docker info --format '{{.ServerVersion}}' | Out-Null
} catch {
    Write-Host "Docker Desktop is installed but not running. Start it and try again." -ForegroundColor Red
    exit 1
}

# ── .env bootstrap ──────────────────────────────────────────────────────────
$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envPath)) {
    Write-Stage "Creating .env from .env.example"
    Copy-Item (Join-Path $PSScriptRoot ".env.example") $envPath
}

if ($Model) {
    $WithModel = $true
    Write-Stage "Pinning OLLAMA_MODEL=$Model in .env"
    $content = Get-Content $envPath
    $content = $content -replace '^OLLAMA_MODEL=.*', "OLLAMA_MODEL=$Model"
    if ($content -notmatch '^OLLAMA_MODEL=') {
        $content += "OLLAMA_MODEL=$Model"
    }
    Set-Content $envPath $content
}

# ── Read effective ports for nice URLs at the end ────────────────────────────
function Read-EnvVal($key, $default) {
    $line = Get-Content $envPath | Where-Object { $_ -match "^\s*$key\s*=\s*(.+?)\s*$" }
    if ($line) { return ($line -replace "^\s*$key\s*=\s*", "") }
    return $default
}
$webPort = Read-EnvVal "WEB_PORT" "3010"
$apiPort = Read-EnvVal "API_PORT" "8000"

# ── Build & start ───────────────────────────────────────────────────────────
Push-Location $PSScriptRoot
try {
    if ($Rebuild) {
        Write-Stage "Rebuilding images (no cache)"
        docker compose build --no-cache
        if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }
    }

    if ($WithModel) {
        Write-Stage "Starting Ollama and pulling the model (this can take a while on first run)"
        docker compose --profile ollama up -d --build ollama ollama-init
        if ($LASTEXITCODE -ne 0) { throw "docker compose up ollama failed" }

        Write-Host "Waiting for the model pull to finish…"
        docker compose logs -f ollama-init
        $exit = (docker inspect -f '{{.State.ExitCode}}' regllm-ollama-init 2>$null)
        if ($exit -and $exit -ne "0") {
            Write-Host "Model pull failed (exit=$exit). See `docker compose logs ollama-init`." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Stage "Skipping Ollama (no LLM — use -WithModel to enable)"
    }

    Write-Stage "Starting API and Web"
    docker compose up -d --build api web
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

    Write-Stage "Waiting for the web UI to become healthy"
    $url = "http://localhost:$webPort"
    $start = Get-Date
    do {
        Start-Sleep -Seconds 2
        $health = (docker inspect -f '{{.State.Health.Status}}' regllm-web 2>$null)
        $elapsed = (Get-Date) - $start
        Write-Host -NoNewline "."
        if ($elapsed.TotalSeconds -gt $WaitSecs) {
            Write-Host ""
            Write-Host "Timed out waiting for the web container. Run `docker compose logs web`." -ForegroundColor Red
            exit 1
        }
    } while ($health -ne "healthy")
    Write-Host ""

    Write-Host ""
    Write-Host "RegLLM is up." -ForegroundColor Green
    Write-Host "  Diff Explainer:   $url/diff"
    Write-Host "  Field Trace:      $url/trace"
    Write-Host "  API:              http://localhost:$apiPort/docs"
    Write-Host "  Logs: docker compose logs -f"
    Write-Host "  Stop: .\stop.ps1   (or `docker compose down`)"

    if (-not $NoBrowser) {
        Start-Process "$url/diff"
    }
} finally {
    Pop-Location
}
