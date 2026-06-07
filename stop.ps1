# RegLLM — stop and (optionally) clean up.
#
# Usage:
#   .\stop.ps1                    stop containers, keep data + model cache
#   .\stop.ps1 -Purge             also drop the cached Ollama models (~9 GB)
#   .\stop.ps1 -Purge -PurgeData  also wipe ./data (you'll lose uploaded SAS/docs)

[CmdletBinding()]
param(
    [switch]$Purge,
    [switch]$PurgeData
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    if ($Purge) {
        Write-Host "==> docker compose down -v  (drops ollama-models volume)" -ForegroundColor Cyan
        docker compose down -v
    } else {
        Write-Host "==> docker compose down" -ForegroundColor Cyan
        docker compose down
    }

    if ($PurgeData) {
        $dataPath = Join-Path $PSScriptRoot "data"
        if (Test-Path $dataPath) {
            Write-Host "==> removing ./data" -ForegroundColor Yellow
            Remove-Item -Recurse -Force $dataPath
        }
    }

    Write-Host "Stopped." -ForegroundColor Green
} finally {
    Pop-Location
}
