@echo off
REM RegLLM - stop the stack.
REM
REM Use stop.ps1 for purge/wipe flags.

setlocal EnableExtensions

if /I "%~1"=="--purge" (
    echo ==^> docker compose down -v  (drops ollama-models volume)
    docker compose down -v
) else (
    echo ==^> docker compose down
    docker compose down
)

echo Stopped.
exit /b 0
