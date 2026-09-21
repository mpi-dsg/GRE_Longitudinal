#!/bin/bash
# Experiment 2: multi-threaded CRUD scaling.
#
# Runs the standard CRUD mix (40/30/20/5/5) against every confirmed
# concurrent-safe index, sweeping thread_num across {1,4,8,16,32}, one
# process per (index, thread count) pair. hotrowex is deliberately excluded
# from MT_ALL (see common.sh / CRUD_AUDIT.md — its remove() calls exit(0)
# instead of failing gracefully, which would kill this whole sweep).
#
# All (index, thread_num) points land in ONE csv per dataset — the harness
# already records thread_num as a column, so this file alone supports a full
# scaling plot per index.
#
# Usage: ./experiments/02_multithread_scaling_crud.sh [dataset] [thread_list]
#   thread_list is a comma-separated list, default "1,4,8,16,32"
#
# Expected runtime: this is the most expensive script in the suite —
# 11 indexes x 5 thread points = 55 full-scale runs. At full scale
# (200M keys, 50M ops/batch, 10 batches) budget several hours. For a quick
# sanity pass, shrink the workload:
#   TABLE_SIZE=2000000 OPERATIONS_NUM=200000 ./experiments/02_multithread_scaling_crud.sh libio 1,4,16

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

THREAD_LIST="${2:-1,4,8,16,32}"
IFS=',' read -ra THREADS <<< "$THREAD_LIST"

CSV="${RESULTS_DIR}/crud_scaling_multithread_${DATASET}_threads-${THREAD_LIST//,/-}_seed${SEED}.csv"
OPS_NAME="crud_baseline_${DATASET}"

echo "=== Experiment 2: multi-threaded CRUD scaling (${DATASET}) ==="
echo "Indexes: ${MT_ALL[*]}"
echo "Threads: ${THREADS[*]}"
echo "Output:  ${CSV}"
echo ""

for idx in "${MT_ALL[@]}"; do
    for t in "${THREADS[@]}"; do
        run_index "$idx" "$t" 0.40 0.30 0.20 0.05 0.05 "$CSV" "$OPS_NAME"
    done
done

echo ""
echo "Done. Results: ${CSV}"
