#!/bin/bash
# Quick bakeoff vs alex on libio 5M growth. Uses free core (not 0).
set -euo pipefail
cd /home/GRE_Longitudinal

OUT_DIR=./results/exploratory/beat_alex_bakeoff
LOG_DIR=./results/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

TABLE_SIZE=5000000
INIT_TABLE_RATIO=0.25
OPERATIONS_NUM=500000
SEED=1866
DATASET=./datasets/libio
OPS_BIN="${OUT_DIR}/libio_5m_ops_seed${SEED}.bin"
CSV="${OUT_DIR}/libio_5m_growth_seed${SEED}.csv"
CORE="${CORE:-1}"

INDEXES=(
  alex
  lipp
  sali
  finedex
  hyper
  dilax
  dili
  kanva
)

rm -f "$CSV"

for idx in "${INDEXES[@]}"; do
  log="${LOG_DIR}/beat_alex_bakeoff_${idx}_libio_seed${SEED}.log"
  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${idx}  core=${CORE}"
    echo "======================================================================"
  } | tee "$log"

  if [ -f "$OPS_BIN" ]; then
    OPS_FLAGS="--load_ops=${OPS_BIN}"
  else
    OPS_FLAGS="--save_ops=${OPS_BIN}"
  fi

  set +e
  OMP_NUM_THREADS=1 taskset -c "${CORE}" ./build/microbench \
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
  ec=${PIPESTATUS[0]}
  set -e
  echo "DONE: ${idx} (exit ${ec})" | tee -a "$log"
done

echo "bakeoff finished. CSV=${CSV}"
