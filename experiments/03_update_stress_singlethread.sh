#!/bin/bash
# Experiment 3: update-stress, single-threaded indexes with a VALID update path.
#
# Per CRUD_AUDIT.md, btree/alex/hot's update() are hardcoded stubs (0% real
# work) — they are excluded here rather than included and silently reporting
# nothing. artunsync's update is valid but has a small unexplained miss rate
# (see audit); included, with that caveat.
#
# Workload is update-heavy relative to the baseline mix: 30% read / 20%
# insert / 40% update / 5% delete / 5% scan (insert still exceeds delete so
# the active key set keeps growing across batches, same invariant as the
# original protocol).
#
# Usage: ./experiments/03_update_stress_singlethread.sh [dataset]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

CSV="${RESULTS_DIR}/update_stress_singlethread_${DATASET}_t1_seed${SEED}.csv"
OPS_NAME="update_stress_${DATASET}"

echo "=== Experiment 3: update-stress, single-threaded valid-update indexes (${DATASET}) ==="
echo "Indexes: ${ST_VALID_UPDATE[*]}  (artunsync has a small unexplained read/update miss rate, see CRUD_AUDIT.md)"
echo "Output:  ${CSV}"
echo ""

for idx in "${ST_VALID_UPDATE[@]}"; do
    run_index "$idx" 1 0.30 0.20 0.40 0.05 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
