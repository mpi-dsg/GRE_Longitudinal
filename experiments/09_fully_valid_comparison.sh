#!/bin/bash
# Experiment 9: the headline comparison — the only two indexes in the entire
# suite (18 audited) with a fully valid implementation across all five CRUD
# operations, per CRUD_AUDIT.md: lipp (single-threaded) and alexol
# (concurrent-safe). Every other index has at least one stubbed, broken, or
# unverifiable operation.
#
# lipp runs once at thread_num=1 (it has no internal synchronization).
# alexol is swept across thread_num to show how the one clean concurrent
# index scales, giving a fair single-thread comparison point (lipp vs
# alexol@1) plus a scaling curve alexol alone can't get from lipp.
#
# Standard CRUD mix: 40% read / 30% insert / 20% update / 5% delete / 5% scan.
#
# Usage: ./experiments/09_fully_valid_comparison.sh [dataset] [thread_list]
#   thread_list is a comma-separated list, default "1,4,8,16,32"

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

THREAD_LIST="${2:-1,4,8,16,32}"
IFS=',' read -ra THREADS <<< "$THREAD_LIST"

CSV="${RESULTS_DIR}/fully_valid_comparison_${DATASET}_seed${SEED}.csv"
OPS_NAME="crud_baseline_${DATASET}"

echo "=== Experiment 9: fully-valid comparison — lipp vs alexol (${DATASET}) ==="
echo "lipp:   thread_num=1 only"
echo "alexol: thread_num in ${THREADS[*]}"
echo "Output: ${CSV}"
echo ""

run_index "lipp" 1 0.40 0.30 0.20 0.05 0.05 "$CSV" "$OPS_NAME"
for t in "${THREADS[@]}"; do
    run_index "alexol" "$t" 0.40 0.30 0.20 0.05 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
