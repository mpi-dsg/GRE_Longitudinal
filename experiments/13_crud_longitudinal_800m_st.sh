#!/bin/bash
# Experiment 13: CRUD longitudinal at 800M (single-threaded, VLDB expansion).
#
# HotStorage draft is growth-only (r/i). For the VLDB expansion we need the
# same never-reset trajectory under a full CRUD mix, at longer N, on indexes
# whose update/delete/scan are audit-valid enough to interpret.
#
# Mix matches exp 01: 40/30/20/5/5. Indexes: lipp (fully valid), dili (CRUD
# ok), alex (update stubbed — kept for growth+delete+scan context), btree
# (update/scan broken — baseline thr only; interpret with CRUD_AUDIT.md).
#
# Usage:
#   ./experiments/13_crud_longitudinal_800m_st.sh [dataset]
# dataset: osm_800m | books_800m  (default books_800m — shorter SMO stress)
#
# Smoke:
#   TABLE_SIZE=8000000 OPERATIONS_NUM=500000 \
#     ./experiments/13_crud_longitudinal_800m_st.sh books_800m

set -euo pipefail

DATASET="${1:-books_800m}"
export TABLE_SIZE="${TABLE_SIZE:-800000000}"
export OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
export INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${DATASET}"

INDEXES=(lipp dili alex btree)

CSV="${RESULTS_DIR}/crud_longitudinal_800m_st_${DATASET}_t1_seed${SEED}.csv"
OPS_NAME="crud_longitudinal_800m_${DATASET}"

echo "=== Experiment 13: CRUD longitudinal 800M ST (${DATASET}) ==="
echo "TABLE_SIZE=${TABLE_SIZE}  INIT=${INIT_TABLE_RATIO}  OPS/batch=${OPERATIONS_NUM}"
echo "Mix: R=0.40 I=0.30 U=0.20 D=0.05 S=0.05"
echo "Indexes: ${INDEXES[*]}  (see CRUD_AUDIT.md for stub caveats)"
echo "Output: ${CSV}"
echo ""

for idx in "${INDEXES[@]}"; do
  run_index "$idx" 1 0.40 0.30 0.20 0.05 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
