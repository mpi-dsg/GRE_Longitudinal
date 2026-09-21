#!/bin/bash
# Experiment 1: single-threaded CRUD baseline.
#
# Runs the standard CRUD mix (40% read / 30% insert / 20% update / 5% delete /
# 5% scan) against every single-threaded-only index, one process per index,
# thread_num=1 (these indexes have no internal synchronization — never run
# them at thread_num>1). This is the same workload used in the initial
# CRUD_AUDIT.md investigation, now formalized as a standalone, re-runnable
# experiment.
#
# Usage: ./experiments/01_singlethread_crud_baseline.sh [dataset]
# dataset defaults to "libio". Known datasets: covid | planet | osm | libio
#
# Expected runtime: ~30-45 minutes total for all 6 indexes at full scale
# (200M keys, 50M ops/batch, 10 batches). Override TABLE_SIZE/OPERATIONS_NUM
# env vars for a faster smoke test, e.g.:
#   TABLE_SIZE=2000000 OPERATIONS_NUM=200000 ./experiments/01_singlethread_crud_baseline.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

CSV="${RESULTS_DIR}/crud_baseline_singlethread_${DATASET}_t1_seed${SEED}.csv"
OPS_NAME="crud_baseline_${DATASET}"

echo "=== Experiment 1: single-threaded CRUD baseline (${DATASET}) ==="
echo "Indexes: ${ST_ALL[*]}"
echo "Output:  ${CSV}"
echo ""

for idx in "${ST_ALL[@]}"; do
    run_index "$idx" 1 0.40 0.30 0.20 0.05 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
