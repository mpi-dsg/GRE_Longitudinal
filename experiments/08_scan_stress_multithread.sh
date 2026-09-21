#!/bin/bash
# Experiment 8: scan-stress, multi-threaded indexes with a VALID scan path.
#
# Per CRUD_AUDIT.md, only alexol/btreeolc/sali have a genuinely correct
# concurrent-safe scan among the MT group — the rest (artolc, wormhole_u64,
# xindex, lippol, finedex, kanva) are stubbed or broken, and hotrowex is
# excluded suite-wide for its exit(0)-on-delete bug (this workload still has
# delete=0.05, so it must stay out). dilax has valid-but-different scan
# semantics (value-range width, fabricated keys) like dili — excluded from
# this direct comparison for the same reason as experiment 7.
#
# Same scan-heavy mix as experiment 7: 30/30/10/5/25.
#
# Usage: ./experiments/08_scan_stress_multithread.sh [dataset] [threads]
#   threads defaults to 16.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

THREADS="${2:-16}"
CSV="${RESULTS_DIR}/scan_stress_multithread_${DATASET}_t${THREADS}_seed${SEED}.csv"
OPS_NAME="scan_stress_${DATASET}"

echo "=== Experiment 8: scan-stress, multi-threaded valid-scan indexes (${DATASET}, threads=${THREADS}) ==="
echo "Indexes: ${MT_VALID_SCAN[*]}"
echo "Output:  ${CSV}"
echo ""

for idx in "${MT_VALID_SCAN[@]}"; do
    run_index "$idx" "$THREADS" 0.30 0.30 0.10 0.05 0.25 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
