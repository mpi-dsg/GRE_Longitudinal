#!/bin/bash
# RID longitudinal: Read/Insert/Delete (no update, no scan), memory OFF
# Indexes with working remove: alex btree lipp dili
# (artunsync excluded — delete corruption / OpCheck misses)
# Dataset: libio
# Seeds: 1866 5 72
# Mix: R=0.40 I=0.40 D=0.20 U=0 S=0 -> B=7
set -uo pipefail
ROOT=/home/GRE_Longitudinal
cd "$ROOT"
source experiments/common.sh libio

INDEXES=(alex btree lipp dili)
SEEDS=(1866 5 72)
READ=0.40
INSERT=0.40
UPDATE=0
DELETE=0.20
SCAN=0
INDEX_TIMEOUT_SEC=${INDEX_TIMEOUT_SEC:-1800}

RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/longbench/longitudinal}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
OPS_CACHE_DIR="${OPS_CACHE_DIR:-${RESULTS_DIR}/ops_cache}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/rid_libio_st_r40i40d20.log"
: > "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RID START" | tee -a "$MASTER_LOG"
echo "indexes=${INDEXES[*]} seeds=${SEEDS[*]} R/I/D=${READ}/${INSERT}/${DELETE} memory=OFF" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

run_one() {
  local idx="$1" csv="$2" ops_name="$3"
  local ops_path="${OPS_CACHE_DIR}/${ops_name}.bin"
  local ops_flag
  if [ -f "$ops_path" ]; then ops_flag="--load_ops=${ops_path}"; else ops_flag="--save_ops=${ops_path}"; fi
  local log="${LOG_DIR}/$(basename "$csv" .csv).log"
  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${idx}  R=${READ} I=${INSERT} U=0 D=${DELETE} S=0 memory=OFF"
    echo "======================================================================"
  } | tee -a "$log" "$MASTER_LOG"

  set +e
  OMP_NUM_THREADS=1 timeout --signal=KILL "${INDEX_TIMEOUT_SEC}"     taskset -c 0 "$BIN"       --keys_file="${DATASET_PATH}" --keys_file_type=binary       --table_size="${TABLE_SIZE}" --init_table_ratio="${INIT_TABLE_RATIO}"       --operations_num="${OPERATIONS_NUM}"       --read="${READ}" --insert="${INSERT}" --update=0 --delete="${DELETE}" --scan=0       --scan_num="${SCAN_NUM}" --indexes="${idx}" --seed="${SEED}"       --operation_order="${OP_ORDER}" --thread_num=1       --output_path="${csv}"       --latency_sample       ${ops_flag} 2>&1 | tee -a "$log" "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  pkill -9 -f "--indexes=${idx} .*--output_path=${csv}" 2>/dev/null || true
  echo "DONE: ${idx} exit=${rc}" | tee -a "$log" "$MASTER_LOG"
}

pass_count=0; fail_count=0; crash_count=0
for seed in "${SEEDS[@]}"; do
  export SEED="$seed"
  csv="${RESULTS_DIR}/rid_libio_st_r40i40d20_seed${seed}.csv"
  ops_name="rid_libio_r40i40d20_seed${seed}"
  rm -f "$csv"
  for idx in "${INDEXES[@]}"; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${idx} seed=${seed}" | tee -a "$MASTER_LOG"
    run_one "$idx" "$csv" "$ops_name"
    per_log="${LOG_DIR}/$(basename "$csv" .csv).log"
    section=$(awk -v idx="$idx" '$0 ~ " "idx"  R=" { buf=""; keep=1 } keep { buf=buf $0 ORS } END { printf "%s", buf }' "$per_log")
    if printf "%s" "$section" | grep -qE "Segmentation fault|Waiting GDB|Killed|timeout:"; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) CRASH ${idx} seed=${seed}" | tee -a "$MASTER_LOG"
      crash_count=$((crash_count+1)); continue
    fi
    fails=$(printf "%s" "$section" | grep -c "OpCheck ALL_OPS FAIL" || true)
    oks=$(printf "%s" "$section" | grep -c "OpCheck ALL_OPS OK" || true)
    if [ "$fails" -gt 0 ] || [ "$oks" -eq 0 ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL_VALIDATE ${idx} seed=${seed} OK=${oks} FAIL=${fails}" | tee -a "$MASTER_LOG"
      fail_count=$((fail_count+1))
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) PASS_VALIDATE ${idx} seed=${seed} batches_ok=${oks}" | tee -a "$MASTER_LOG"
      pass_count=$((pass_count+1))
    fi
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RID DONE pass=${pass_count} fail=${fail_count} crash=${crash_count}" | tee -a "$MASTER_LOG"
exit 0
