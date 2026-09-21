#!/bin/bash
# Experiment 6: delete-stress, multi-threaded indexes.
#
# Uses MT_ALL, which already excludes hotrowex (its remove() calls exit(0)
# instead of returning false — never include it in a workload with
# delete>0). artolc's remove() always reports true by construction (the
# underlying Tree::remove is void, same issue as artunsync) — included, not
# verified evidence of correctness. kanva's remove() is genuinely valid per
# the audit even though its update() is broken.
#
# Same delete-heavy mix as experiment 5: 30/35/10/20/5.
#
# Usage: ./experiments/06_delete_stress_multithread.sh [dataset] [threads]
#   threads defaults to 16.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

THREADS="${2:-16}"
CSV="${RESULTS_DIR}/delete_stress_multithread_${DATASET}_t${THREADS}_seed${SEED}.csv"
OPS_NAME="delete_stress_${DATASET}"

echo "=== Experiment 6: delete-stress, multi-threaded indexes (${DATASET}, threads=${THREADS}) ==="
echo "Indexes: ${MT_ALL[*]}  (artolc's remove() always reports true by construction, see CRUD_AUDIT.md)"
echo "Output:  ${CSV}"
echo ""

for idx in "${MT_ALL[@]}"; do
    run_index "$idx" "$THREADS" 0.30 0.35 0.10 0.20 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
