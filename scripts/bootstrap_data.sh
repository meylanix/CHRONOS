#!/usr/bin/env bash
# Load every source that has a working connector, plus the curated knowledge graph.
# Usage: scripts/bootstrap_data.sh [database]   (default: chronos.sqlite)
# Needs internet access to raw.githubusercontent.com (OWID, Seshat, V-Dem). Optional sources
# (World Bank, OWID Chart API, GDELT) are attempted and reported if unreachable.
set -uo pipefail
DB="${1:-chronos.sqlite}"
run() { echo "+ chronos --db $DB $*"; chronos --db "$DB" "$@"; }

run init
run fetch owid-github --dataset co2            || exit 1
run fetch owid-github --dataset energy         || exit 1
run fetch seshat                               || exit 1
if [ "${CHRONOS_SKIP_VDEM:-0}" != "1" ]; then
  run fetch vdem || echo "!! V-Dem failed (needs ~4 GB RAM for the RData file; or pass an official CSV with --file)"
fi
run kg import data/curated/tech_transitions.json || exit 1

echo "--- optional API sources (implemented, not verified live during development) ---"
run fetch worldbank --indicator IT.NET.USER.ZS --country WLD --date 1960:2025 || echo "!! World Bank API not reachable"
run fetch owid-chart --slug life-expectancy --entity World --indicator life_expectancy || echo "!! OWID Chart API not reachable"
run series --summary
