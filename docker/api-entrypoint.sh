#!/bin/sh
# Entrypoint for the API container.
#
# Ensures a working demo on first run by:
#   1. Generating data/samples/cycles_v[23].csv if missing.
#   2. Bootstrapping data/sas/{v2,v3}/sample_lgd.sas + data/docs/* + the
#      BM25 index if any of them are missing.
#   3. Handing off to the configured CMD (uvicorn by default).
#
# This script runs AFTER the ./data bind mount is in place, so seeded
# files persist on the host filesystem between container restarts.

set -e

cd /app

# 1. Sample CSVs ---------------------------------------------------------------
if [ ! -f data/samples/cycles_v3.csv ]; then
  echo "[entrypoint] generating data/samples/cycles_v[23].csv …"
  python scripts/generate_v3.py || echo "[entrypoint] generate_v3 skipped"
fi

# 2. Agent corpus + index ------------------------------------------------------
need_seed=false
[ ! -f data/sas/v2/sample_lgd.sas ] && need_seed=true
[ ! -f data/sas/v3/sample_lgd.sas ] && need_seed=true
[ ! -f data/docs/index.json ]      && need_seed=true

if [ "$need_seed" = "true" ]; then
  echo "[entrypoint] seeding data/sas + data/docs and building the BM25 index …"
  python scripts/seed_docs.py || echo "[entrypoint] seed_docs failed (continuing)"
fi

echo "[entrypoint] starting: $*"
exec "$@"
