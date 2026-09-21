#!/bin/bash
# Experiment 5: delete-stress, all single-threaded indexes.
#
# All 6 ST indexes have SOME form of remove() (unlike update, no ST index has
# a hardcoded-false delete stub) so all are included — but artunsync's
# remove() always returns true by construction (the underlying Tree::remove
# is void), so its "success" numbers here are not verified evidence, just
# not absent either. See CRUD_AUDIT.md.
#
# Delete-heavy mix: 30% read / 35% insert / 10% update / 20% delete / 5% scan.
# insert still exceeds delete (0.35 > 0.20) so the active key set keeps
# growing net across batches, same invariant as the original protocol.
#
# Usage: ./experiments/05_delete_stress_singlethread.sh [dataset]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

CSV="${RESULTS_DIR}/delete_stress_singlethread_${DATASET}_t1_seed${SEED}.csv"
OPS_NAME="delete_stress_${DATASET}"

echo "=== Experiment 5: delete-stress, single-threaded indexes (${DATASET}) ==="
echo "Indexes: ${ST_ALL[*]}  (artunsync's remove() always reports true by construction, see CRUD_AUDIT.md)"
echo "Output:  ${CSV}"
echo ""

for idx in "${ST_ALL[@]}"; do
    run_index "$idx" 1 0.30 0.35 0.10 0.20 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
