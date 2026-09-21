#!/bin/bash
# Experiment 12: cross-hardness longitudinal at 800M (single-threaded).
#
# Runs the same never-reset 50/50 growth protocol on BOTH SOSD 800M datasets
# back-to-back for a compact index cast, so we can contrast moderate (books)
# vs hard (osm) degradation slopes and ranking inversions in one artifact.
#
# Usage:
#   ./experiments/12_longitudinal_cross_hardness_800m_st.sh
#
# Smoke:
#   TABLE_SIZE=8000000 OPERATIONS_NUM=500000 \
#     ./experiments/12_longitudinal_cross_hardness_800m_st.sh

set -euo pipefail

export TABLE_SIZE="${TABLE_SIZE:-800000000}"
export OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
export INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"

INDEXES=(alex lipp dili artunsync btree)
DATASETS=(books_800m osm_800m)
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

echo "=== Experiment 12: cross-hardness longitudinal 800M ST ==="
echo "Datasets: ${DATASETS[*]}"
echo "Indexes:  ${INDEXES[*]}"
echo ""

for ds in "${DATASETS[@]}"; do
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/common.sh" "${ds}"
  CSV="${RESULTS_DIR}/longitudinal_cross_hardness_800m_st_${DATASET}_t1_seed${SEED}.csv"
  OPS_NAME="longitudinal_growth_800m_${DATASET}_r50i50"
  echo "--- dataset=${DATASET} → ${CSV} ---"
  for idx in "${INDEXES[@]}"; do
    run_index "$idx" 1 0.50 0.50 0 0 0 "$CSV" "$OPS_NAME"
  done
done

echo ""
echo "Done. Compare books_800m vs osm_800m CSVs under results/longbench/longitudinal/."
