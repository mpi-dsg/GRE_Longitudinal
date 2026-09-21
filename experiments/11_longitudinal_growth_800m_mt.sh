#!/bin/bash
# Experiment 11: longitudinal growth at 800M (concurrent-safe indexes).
#
# Same never-reset 50/50 growth protocol as exp 10, but for MT indexes at a
# fixed thread count (default 8 — HotStorage concurrent panels used 8/16/32;
# start at 8 to keep wall time manageable at 800M).
#
# Usage:
#   ./experiments/11_longitudinal_growth_800m_mt.sh [dataset] [threads]
# dataset defaults to osm_800m; threads default 8.
#
# Smoke:
#   TABLE_SIZE=8000000 OPERATIONS_NUM=500000 \
#     ./experiments/11_longitudinal_growth_800m_mt.sh osm_800m 4

set -euo pipefail

DATASET="${1:-osm_800m}"
THREADS="${2:-8}"
export TABLE_SIZE="${TABLE_SIZE:-800000000}"
export OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
export INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${DATASET}"

INDEXES=(alexol artolc sali btreeolc finedex)

CSV="${RESULTS_DIR}/longitudinal_growth_800m_mt_${DATASET}_t${THREADS}_seed${SEED}.csv"
OPS_NAME="longitudinal_growth_800m_${DATASET}_r50i50"

echo "=== Experiment 11: longitudinal growth 800M MT (${DATASET}, t=${THREADS}) ==="
echo "TABLE_SIZE=${TABLE_SIZE}  INIT=${INIT_TABLE_RATIO}  OPS/batch=${OPERATIONS_NUM}"
echo "Mix: read=0.5 insert=0.5  Indexes: ${INDEXES[*]}"
echo "Output: ${CSV}"
echo ""

for idx in "${INDEXES[@]}"; do
  run_index "$idx" "$THREADS" 0.50 0.50 0 0 0 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
