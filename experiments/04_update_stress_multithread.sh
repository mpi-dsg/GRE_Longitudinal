#!/bin/bash
# Experiment 4: update-stress, multi-threaded indexes with a VALID update path.
#
# Per CRUD_AUDIT.md, hyper/btreeolc/hotrowex/kanva's update() are stubbed,
# silently redirected to insert, or don't correctly gate on delete outcome —
# excluded here. wormhole_u64 and xindex's "update" is really their normal
# in-place-upsert path (genuinely correct behavior for those libraries, not a
# fake — see audit), included as valid.
#
# Same update-heavy mix as experiment 3: 30/20/40/5/5.
#
# Usage: ./experiments/04_update_stress_multithread.sh [dataset] [threads]
#   threads defaults to 16 (half this container's 32 cores, leaves headroom).
#
# Expected runtime: 8 indexes x one thread count. At full scale, budget
# roughly 45-75 minutes total; scale down via TABLE_SIZE/OPERATIONS_NUM for a
# quick pass.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "${1:-libio}"

THREADS="${2:-16}"
CSV="${RESULTS_DIR}/update_stress_multithread_${DATASET}_t${THREADS}_seed${SEED}.csv"
OPS_NAME="update_stress_${DATASET}"

echo "=== Experiment 4: update-stress, multi-threaded valid-update indexes (${DATASET}, threads=${THREADS}) ==="
echo "Indexes: ${MT_VALID_UPDATE[*]}"
echo "Output:  ${CSV}"
echo ""

for idx in "${MT_VALID_UPDATE[@]}"; do
    run_index "$idx" "$THREADS" 0.30 0.20 0.40 0.05 0.05 "$CSV" "$OPS_NAME"
done

echo ""
echo "Done. Results: ${CSV}"
