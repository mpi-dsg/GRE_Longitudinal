#!/bin/bash
# Experiment: longitudinal Read/Insert/Update (no delete, no scan)
# Indexes: lipp, artunsync, dili  (CRUD_AUDIT ST_VALID_UPDATE subset)
# Dataset: libio (DILI-safe key domain)
# Seeds: 1866, 5, 72
# Mix: R=0.40 I=0.40 U=0.20 D=0 S=0
#   -> B = (200M-50M)/(50M*0.40) = 7 batches
# One index per process. Fail loudly if any OpCheck ALL_OPS FAIL.

set -euo pipefail

ROOT=/home/GRE_Longitudinal
cd "$ROOT"
source experiments/common.sh libio

INDEXES=(lipp artunsync dili)
SEEDS=(1866 5 72)
READ=0.40
INSERT=0.40
UPDATE=0.20
DELETE=0
SCAN=0

RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/longbench/longitudinal}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
OPS_CACHE_DIR="${OPS_CACHE_DIR:-${RESULTS_DIR}/ops_cache}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/riu_longitudinal_libio_st_r40i40u20.log"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RIU longitudinal START" | tee -a "$MASTER_LOG"
echo "indexes=${INDEXES[*]} seeds=${SEEDS[*]} R/I/U=${READ}/${INSERT}/${UPDATE}" | tee -a "$MASTER_LOG"
echo "table=${TABLE_SIZE} init=${INIT_TABLE_RATIO} ops/batch=${OPERATIONS_NUM}" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

fail_count=0
pass_count=0

for seed in "${SEEDS[@]}"; do
  export SEED="$seed"
  csv="${RESULTS_DIR}/riu_longitudinal_libio_st_r40i40u20_seed${seed}.csv"
  ops_name="riu_longitudinal_libio_r40i40u20_seed${seed}"
  rm -f "$csv"

  for idx in "${INDEXES[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "----------------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${idx} seed=${seed}" | tee -a "$MASTER_LOG"
    echo "csv=${csv}" | tee -a "$MASTER_LOG"
    echo "----------------------------------------------------------------------" | tee -a "$MASTER_LOG"

    # run_index tees to per-csv log; also mirror into master log
    run_index "$idx" 1 "$READ" "$INSERT" "$UPDATE" "$DELETE" "$SCAN" "$csv" "$ops_name" \
      2>&1 | tee -a "$MASTER_LOG"

    # Validate OpCheck lines from the per-csv log (last index section)
    per_log="${LOG_DIR}/$(basename "$csv" .csv).log"
    # Count FAIL/OK for this index run: take OpCheck lines after last START for this idx
    section=$(awk -v idx="$idx" '
      $0 ~ " "idx"  threads=" { buf=""; keep=1 }
      keep { buf=buf $0 ORS }
      END { printf "%s", buf }
    ' "$per_log")

    fails=$(printf "%s" "$section" | grep -c "OpCheck ALL_OPS FAIL" || true)
    oks=$(printf "%s" "$section" | grep -c "OpCheck ALL_OPS OK" || true)

    if [ "$fails" -gt 0 ] || [ "$oks" -eq 0 ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL_VALIDATE ${idx} seed=${seed} ALL_OPS_OK=${oks} ALL_OPS_FAIL=${fails}" | tee -a "$MASTER_LOG"
      fail_count=$((fail_count + 1))
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) PASS_VALIDATE ${idx} seed=${seed} batches_ok=${oks}" | tee -a "$MASTER_LOG"
      pass_count=$((pass_count + 1))
    fi
  done
done

echo "" | tee -a "$MASTER_LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RIU longitudinal DONE pass=${pass_count} fail=${fail_count}" | tee -a "$MASTER_LOG"
exit $(( fail_count > 0 ? 1 : 0 ))
