@echo off
REM RegLLM - Windows one-command launcher (cmd.exe fallback).
REM
REM Use start.ps1 for a richer experience (live model-pull progress,
REM per-stage status, optional flags).

setlocal EnableExtensions EnableDelayedExpansion

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker not found. Install Docker Desktop:
    echo   https://www.docker.com/products/docker-desktop/
    exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
    echo "docker compose" is not available. Update Docker Desktop and try again.
    exit /b 1
)

docker info --format "{{.ServerVersion}}" >nul 2>nul
if errorlevel 1 (
    echo Docker Desktop is installed but not running. Start it and re-run.
    exit /b 1
)

if not exist .env (
    echo ==^> Creating .env from .env.example
    copy /Y .env.example .env >nul
)

echo ==^> Starting Ollama and pulling the model (first run can take a while)
docker compose up -d --build ollama ollama-init
if errorlevel 1 goto :fail

echo ==^> Waiting for the model pull to finish (Ctrl+C exits log tail only)
docker compose logs -f ollama-init

for /f "tokens=*" %%i in ('docker inspect -f "{{.State.ExitCode}}" regllm-ollama-init') do set "INIT_EXIT=%%i"
if not "%INIT_EXIT%"=="0" (
    echo Model pull failed (exit=%INIT_EXIT%). See "docker compose logs ollama-init".
    goto :fail
)

echo ==^> Starting API and Web
docker compose up -d --build api web
if errorlevel 1 goto :fail

echo ==^> Waiting for web container to become healthy (up to 5 minutes)
set /a WAITED=0
:waitloop
for /f "tokens=*" %%i in ('docker inspect -f "{{.State.Health.Status}}" regllm-web 2^>nul') do set "HEALTH=%%i"
if "%HEALTH%"=="healthy" goto :ready
timeout /t 3 /nobreak >nul
set /a WAITED+=3
if %WAITED% GEQ 300 (
    echo Timed out waiting for the web container. Try: docker compose logs web
    goto :fail
)
goto :waitloop

:ready
echo.
echo RegLLM is up.
echo   UI:   http://localhost:3010/diff
echo   API:  http://localhost:8000/docs
echo   Logs: docker compose logs -f
echo   Stop: stop.bat   (or: docker compose down)

start "" http://localhost:3010/diff
exit /b 0

:fail
echo Startup failed.
exit /b 1
