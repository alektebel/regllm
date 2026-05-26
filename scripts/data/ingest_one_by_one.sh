#!/usr/bin/env bash
# Ingest each regulatory source in its own subprocess.
# Each run gets a fresh heap — no memory accumulation across documents.
# Already-ingested sources are skipped automatically (checksum cache).
#
# Usage:
#   bash scripts/data/ingest_one_by_one.sh
#   bash scripts/data/ingest_one_by_one.sh --force
#   bash scripts/data/ingest_one_by_one.sh --llm-chunk
#   bash scripts/data/ingest_one_by_one.sh --dry-run
set -euo pipefail

SCRIPT="scripts/data/ingest_regulations.py"
EXTRA_ARGS="${*}"

# Resolve the project venv python (works even if the shebang path is stale)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=""
for candidate in \
    "${VIRTUAL_ENV:-__none__}/bin/python" \
    "$REPO_ROOT/regllm.env/bin/python" \
    "python3" \
    "python"; do
  if [ "$candidate" = "__none__/bin/python" ]; then continue; fi
  if command -v "$candidate" &>/dev/null || [ -f "$candidate" ]; then
    # Confirm it can import sentence_transformers (i.e. it's the right env)
    if "$candidate" -c "import sentence_transformers" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: could not find a Python with sentence_transformers installed."
  echo "Activate the venv first:  source regllm.env/bin/activate"
  exit 1
fi
echo "Using python: $PYTHON"

SOURCES=(
  ecb_ssm_guide_2025
  crr_575_2013
  crr2_2019_876
  crr3_2024_1623
  crd4_2013_36
  crd5_2019_878
  crd6_2024_1619
  brrd_2014_59
  brrd2_2019_879
  srmr_806_2014
  srmr2_2019_877
  dgsd_2014_49
  emir_648_2012
  mifir_600_2014
  mifid2_2014_65
  eba_gl_2017_16
  eba_gl_2020_06
  eba_gl_2022_12
  eba_gl_2018_10
  eba_gl_2020_05
  eba_gl_2021_07
)

_ram() {
  "$PYTHON" -c "
import psutil; m=psutil.virtual_memory()
print(f'RAM {m.percent:.0f}%  ({m.used/1e9:.1f}/{m.total/1e9:.1f} GB used)')" 2>/dev/null || true
}

TOTAL=${#SOURCES[@]}
FAILED=()

echo ""
echo "Starting ingestion of $TOTAL sources."
_ram
echo ""

for i in "${!SOURCES[@]}"; do
  id="${SOURCES[$i]}"
  echo ""
  echo "=== [$((i+1))/$TOTAL] $id ==="
  _ram

  if "$PYTHON" "$SCRIPT" --source "$id" $EXTRA_ARGS; then
    echo "--- $id OK ---"
  else
    echo "--- $id FAILED (continuing) ---"
    FAILED+=("$id")
  fi
done

echo ""
echo "=== Done ==="
_ram
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All $TOTAL sources ingested successfully."
else
  echo "${#FAILED[@]} source(s) failed: ${FAILED[*]}"
  exit 1
fi
