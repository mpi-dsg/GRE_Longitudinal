#!/bin/bash
# Experiment 7: scan-stress, single-threaded indexes with a VALID scan path.
#
# Per CRUD_AUDIT.md: btree's scan never populates the output buffer and its
# count formula can't match the request; artunsync's scan is entirely
# unimplemented dead code. Both excluded. dili's scan is functional but has
# different semantics (a key-VALUE range width rather than a result count,
# plus fabricated result keys) — excluded from this direct comparison since
# it isn't measuring the same thing as the other three.
#
# Scan-heavy mix: 30% read / 30% insert / 10% update / 5% delete / 25% scan.
#
# Usage: ./experiments/07_scan_stress_singlethread.sh [dataset]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

CSV="${RESULTS_DIR}/scan_stress_singlethread_${DATASET}_t1_seed${SEED}.csv"
OPS_NAME="scan_stress_${DATASET}"

echo "=== Experiment 7: scan-stress, single-threaded valid-scan indexes (${DATASET}) ==="
echo "Indexes: ${ST_VALID_SCAN[*]}"
echo "Output:  ${CSV}"
echo ""

for idx in "${ST_VALID_SCAN[@]}"; do
    run_index "$idx" 1 0.30 0.30 0.10 0.05 0.25 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
