#!/bin/bash
# Phase E: prefaulted mmap arena vs alex / prefault baselines.
set -euo pipefail
cd /home/GRE_Longitudinal

OUT_DIR=./results/exploratory/alex_phase_e
LOG_DIR=./results/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

TABLE_SIZE=5000000
INIT_TABLE_RATIO=0.25
OPERATIONS_NUM=500000
SEED=1866
DATASET=./datasets/libio
OPS_BIN_OLD=./results/longbench/longitudinal/alex_localize/libio_5m_ops_seed${SEED}.bin
OPS_BIN="${OUT_DIR}/libio_5m_ops_seed${SEED}.bin"
CSV="${OUT_DIR}/libio_5m_growth_seed${SEED}.csv"

if [ -f "$OPS_BIN_OLD" ] && [ ! -f "$OPS_BIN" ]; then
  cp "$OPS_BIN_OLD" "$OPS_BIN"
fi

INDEXES=(
  alex
  alex_prefault
  alex_arena
  alex_arena_mlock
  alex_arena_1g
  alex_amort128_arena
  alex_amort128_arena_mlock
)

rm -f "$CSV"

for idx in "${INDEXES[@]}"; do
  log="${LOG_DIR}/alex_phase_e_${idx}_libio_5m_seed${SEED}.log"
  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${idx}"
    echo "======================================================================"
  } | tee -a "$log"

  if [ -f "$OPS_BIN" ]; then
    OPS_FLAGS="--load_ops=${OPS_BIN}"
  else
    OPS_FLAGS="--save_ops=${OPS_BIN}"
  fi

  OMP_NUM_THREADS=1 taskset -c 0 ./build/microbench \
      --keys_file="${DATASET}" \
      --keys_file_type=binary \
      --table_size="${TABLE_SIZE}" \
      --init_table_ratio="${INIT_TABLE_RATIO}" \
      --operations_num="${OPERATIONS_NUM}" \
      --read=0.5 --insert=0.5 --update=0 --delete=0 --scan=0 \
      --indexes="${idx}" \
      --seed="${SEED}" \
      --operation_order=shuffle \
      --thread_num=1 \
      --output_path="${CSV}" \
      --memory --latency_sample \
      ${OPS_FLAGS} 2>&1 | tee -a "$log"

  echo "DONE: ${idx} (exit ${PIPESTATUS[0]})" | tee -a "$log"
done

echo "Phase E finished. CSV=${CSV}"
