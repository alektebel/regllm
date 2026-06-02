#!/usr/bin/env bash
# start.sh — start SAS API (port 8001) + Next.js frontend (port 3010)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "▶ Killing any existing processes on ports 8001 / 3010..."
fuser -k 8001/tcp 2>/dev/null || true
fuser -k 3010/tcp 2>/dev/null || true
sleep 1

echo "▶ Starting SAS API on :8001..."
cd "$ROOT"
uvicorn api.main_sas:app --host 0.0.0.0 --port 8001 &
API_PID=$!

sleep 2
if ! curl -sf http://localhost:8001/health > /dev/null; then
  echo "✗ SAS API failed to start"
  exit 1
fi
echo "✓ SAS API running (PID $API_PID)"

echo "▶ Starting Next.js frontend on :3010..."
cd "$ROOT/frontend"
npm run dev &
FE_PID=$!

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App:  http://localhost:3010/sas"
echo "  API:  http://localhost:8001/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Press Ctrl+C to stop both."

trap "kill $API_PID $FE_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
