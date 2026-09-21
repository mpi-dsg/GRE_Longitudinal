#!/bin/bash
# Experiment 10: longitudinal growth honeymoon at 800M (single-threaded).
#
# Scales the HotStorage / GRE longitudinal protocol to SOSD 800M keys:
#   N=800M, ρ_init=0.25 → bulk-load 200M, then never-reset batches of
#   OPERATIONS_NUM ops at 50/50 read/insert until the insert pool is drained
#   (~24 batches of 50M ops → +25M keys/batch → 200M→800M).
#
# Hypothesis (paper): first-batch thr ≫ late-batch thr; single-shot aggregate
# can invert rankings vs last-batch (esp. LIPP vs ART on moderate CDFs).
# books_800m ≈ moderate; osm_800m ≈ hard (GRE/SOSD hardness).
#
# Usage:
#   ./experiments/10_longitudinal_growth_800m_st.sh [dataset]
# dataset: osm_800m | books_800m  (default osm_800m)
#
# Smoke:
#   TABLE_SIZE=8000000 OPERATIONS_NUM=500000 \
#     ./experiments/10_longitudinal_growth_800m_st.sh osm_800m
#
# Expected wall time (full): ~2–6 h per index depending on thr; run one
# dataset at a time on a quiet core.

set -euo pipefail

DATASET="${1:-osm_800m}"
export TABLE_SIZE="${TABLE_SIZE:-800000000}"
export OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
export INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${DATASET}"

# Paper longitudinal cast: segment-fitting learned + local traditional baselines.
INDEXES=(alex lipp dili artunsync btree)

CSV="${RESULTS_DIR}/longitudinal_growth_800m_st_${DATASET}_t1_seed${SEED}.csv"
OPS_NAME="longitudinal_growth_800m_${DATASET}_r50i50"

echo "=== Experiment 10: longitudinal growth 800M ST (${DATASET}) ==="
echo "TABLE_SIZE=${TABLE_SIZE}  INIT=${INIT_TABLE_RATIO}  OPS/batch=${OPERATIONS_NUM}"
echo "Mix: read=0.5 insert=0.5  Indexes: ${INDEXES[*]}"
echo "Output: ${CSV}"
echo ""

for idx in "${INDEXES[@]}"; do
  run_index "$idx" 1 0.50 0.50 0 0 0 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
echo "Plot per-batch throughput / w_p99 / memory vs table size; mark batch-1 vs last-batch vs mean."
