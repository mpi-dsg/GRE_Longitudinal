#!/bin/bash
# RIUD longitudinal WITHOUT --memory (avoids artunsync Tree::size segfault)
# Indexes: lipp artunsync dili
# Dataset: libio
# Seeds: 1866 5 72
# Mix: R=0.30 I=0.40 U=0.20 D=0.10 S=0 -> B=7
set -uo pipefail
ROOT=/home/GRE_Longitudinal
cd "$ROOT"
source experiments/common.sh libio

INDEXES=(lipp artunsync dili)
SEEDS=(1866 5 72)
READ=0.30
INSERT=0.40
UPDATE=0.20
DELETE=0.10
SCAN=0
INDEX_TIMEOUT_SEC=${INDEX_TIMEOUT_SEC:-1800}

RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/longbench/longitudinal}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
OPS_CACHE_DIR="${OPS_CACHE_DIR:-${RESULTS_DIR}/ops_cache}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/riud_nomem_libio_st_r30i40u20d10.log"
: > "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RIUD nomem START" | tee -a "$MASTER_LOG"
echo "indexes=${INDEXES[*]} seeds=${SEEDS[*]} R/I/U/D=${READ}/${INSERT}/${UPDATE}/${DELETE}" | tee -a "$MASTER_LOG"
echo "memory=OFF timeout=${INDEX_TIMEOUT_SEC}s" | tee -a "$MASTER_LOG"
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
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${idx}  R=${READ} I=${INSERT} U=${UPDATE} D=${DELETE} S=0 memory=OFF"
    echo "======================================================================"
  } | tee -a "$log" "$MASTER_LOG"

  set +e
  OMP_NUM_THREADS=1 timeout --signal=KILL "${INDEX_TIMEOUT_SEC}"     taskset -c 0 "$BIN"       --keys_file="${DATASET_PATH}" --keys_file_type=binary       --table_size="${TABLE_SIZE}" --init_table_ratio="${INIT_TABLE_RATIO}"       --operations_num="${OPERATIONS_NUM}"       --read="${READ}" --insert="${INSERT}" --update="${UPDATE}" --delete="${DELETE}" --scan=0       --scan_num="${SCAN_NUM}" --indexes="${idx}" --seed="${SEED}"       --operation_order="${OP_ORDER}" --thread_num=1       --output_path="${csv}"       --latency_sample       ${ops_flag} 2>&1 | tee -a "$log" "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  pkill -9 -f "--indexes=${idx} .*--output_path=${csv}" 2>/dev/null || true
  echo "DONE: ${idx} exit=${rc}" | tee -a "$log" "$MASTER_LOG"
}

pass_count=0; fail_count=0; crash_count=0
for seed in "${SEEDS[@]}"; do
  export SEED="$seed"
  csv="${RESULTS_DIR}/riud_nomem_libio_st_r30i40u20d10_seed${seed}.csv"
  ops_name="riud_nomem_libio_r30i40u20d10_seed${seed}"
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
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RIUD nomem DONE pass=${pass_count} fail=${fail_count} crash=${crash_count}" | tee -a "$MASTER_LOG"
exit 0
