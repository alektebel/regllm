# RegLLM — stop all local processes (no Docker).
Write-Host "Stopping RegLLM local processes..." -ForegroundColor Cyan
Get-Process -Name "uvicorn"  -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "node"     -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -match "next|3010" } | Stop-Process -Force
Get-Process -Name "ollama"   -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "Done." -ForegroundColor Green
