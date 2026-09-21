#!/bin/bash
# Noise-floor calibration: repeat stock ALEX N times on same ops, free core.
set -euo pipefail
cd /home/GRE_Longitudinal

OUT_DIR=./results/exploratory/alex_noise_floor
LOG_DIR=./results/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

TABLE_SIZE=5000000
INIT_TABLE_RATIO=0.25
OPERATIONS_NUM=500000
SEED=1866
DATASET=./datasets/libio
OPS_BIN="${OUT_DIR}/libio_5m_ops_seed${SEED}.bin"
# Prefer shared cache from phase_a / bakeoff / localize
for cand in \
  ./results/longbench/longitudinal/alex_phase_a/libio_5m_ops_seed${SEED}.bin \
  ./results/longbench/longitudinal/beat_alex_bakeoff/libio_5m_ops_seed${SEED}.bin \
  ./results/longbench/longitudinal/alex_localize/libio_5m_ops_seed${SEED}.bin
do
  if [ -f "$cand" ] && [ ! -f "$OPS_BIN" ]; then
    cp -a "$cand" "$OPS_BIN"
    break
  fi
done

N_RUNS="${N_RUNS:-10}"
CORE="${CORE:-1}"
CSV="${OUT_DIR}/libio_5m_alex_repeats_seed${SEED}.csv"
rm -f "$CSV"

for i in $(seq 1 "$N_RUNS"); do
  log="${LOG_DIR}/alex_noise_floor_run${i}_libio_seed${SEED}.log"
  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  alex noise-floor run ${i}/${N_RUNS} core=${CORE}"
    echo "======================================================================"
  } | tee "$log"

  if [ -f "$OPS_BIN" ]; then
    OPS_FLAGS="--load_ops=${OPS_BIN}"
  else
    OPS_FLAGS="--save_ops=${OPS_BIN}"
  fi

  # Tag each run via a dummy env in log only; CSV index_type stays "alex".
  # We disambiguate by append order (15 rows per run).
  OMP_NUM_THREADS=1 taskset -c "${CORE}" ./build/microbench \
      --keys_file="${DATASET}" \
      --keys_file_type=binary \
      --table_size="${TABLE_SIZE}" \
      --init_table_ratio="${INIT_TABLE_RATIO}" \
      --operations_num="${OPERATIONS_NUM}" \
      --read=0.5 --insert=0.5 --update=0 --delete=0 --scan=0 \
      --indexes=alex \
      --seed="${SEED}" \
      --operation_order=shuffle \
      --thread_num=1 \
      --output_path="${CSV}" \
      --memory --latency_sample \
      ${OPS_FLAGS} 2>&1 | tee -a "$log"

  echo "DONE: run ${i} (exit ${PIPESTATUS[0]})" | tee -a "$log"
done

echo "noise_floor finished. CSV=${CSV} N=${N_RUNS}"
