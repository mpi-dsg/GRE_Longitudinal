#!/bin/bash
# Phase 2b: bulk-load fraction (ρ_init) sweep — growth-only balanced ST
# Fix N=200M, M=50M, R/I=0.5/0.5; sweep init_table_ratio ∈ {0.10, 0.50}
# ρ=0.25 (B=6) already covered by {libio,planet}_st_half_seed*.csv — do not re-run.
#
# B = floor((1-ρ)*N / (M * 0.5))
#   ρ=0.10 → bulk 20M, pool 180M, B=8
#   ρ=0.50 → bulk 100M, pool 100M, B=4
#
# Primary: libio; spot-check: planet. Seeds 1866/5/72. Cast: alex lipp dili artunsync btree.
set -uo pipefail
ROOT=/home/GRE_Longitudinal
cd "$ROOT"

export TABLE_SIZE="${TABLE_SIZE:-200000000}"
export OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
export SCAN_NUM="${SCAN_NUM:-100}"
export OP_ORDER="${OP_ORDER:-shuffle}"
export CORE="${CORE:-4}"

BIN="./build/microbench"
[ -f "$BIN" ] || { echo "ERROR: $BIN not found"; exit 1; }

INDEXES=(alex lipp dili artunsync btree)
SEEDS=(1866 5 72)
# tag|rho|need_batches
RHOS=(
  "rho010|0.10|8"
  "rho050|0.50|4"
)
# libio first, then planet spot-check
DATASETS=(libio planet)

INDEX_TIMEOUT_SEC=${INDEX_TIMEOUT_SEC:-14400}
RESULTS_DIR=${RESULTS_DIR:-${ROOT}/results/longbench/longitudinal}
LOG_DIR=${LOG_DIR:-${ROOT}/results/logs}
OPS_CACHE_DIR=${OPS_CACHE_DIR:-${ROOT}/results/ops_cache}
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/rho_init_sweep_st.log"
: > "$MASTER_LOG"

{
  echo "======================================================================"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RHO_INIT_SWEEP START"
  echo "table_size=${TABLE_SIZE} M=${OPERATIONS_NUM} R/I=0.5/0.5"
  echo "rho tags: rho010 (B=8) rho050 (B=4)  — rho025 = existing *_st_half_*"
  echo "datasets=${DATASETS[*]} indexes=${INDEXES[*]} seeds=${SEEDS[*]}"
  echo "core=${CORE} timeout_sec=${INDEX_TIMEOUT_SEC}"
  echo "======================================================================"
} | tee -a "$MASTER_LOG"

rows_for_index() {
  local csv="$1" idx="$2"
  python3 - "$csv" "$idx" <<'PY'
import csv, sys
from pathlib import Path
p, idx = Path(sys.argv[1]), sys.argv[2]
if not p.exists():
    print(0); raise SystemExit
print(sum(1 for r in csv.DictReader(open(p)) if r.get("index_type") == idx))
PY
}

bench_running() {
  pgrep -x microbench >/dev/null 2>&1 || pgrep -x longbench >/dev/null 2>&1
}

wait_idle() {
  while bench_running; do
    echo "[$(date -u +%FT%TZ)] waiting for other microbench..." | tee -a "$MASTER_LOG"
    sleep 30
  done
}

run_one() {
  local ds="$1" tag="$2" rho="$3" need="$4" seed="$5" idx="$6"
  local csv="${RESULTS_DIR}/${ds}_st_${tag}_seed${seed}.csv"
  local ops_name="rho_${ds}_${tag}_r0.50_i0.50_seed${seed}"
  local ops_path="${OPS_CACHE_DIR}/${ops_name}.bin"
  local ops_flag
  if [ -f "$ops_path" ]; then ops_flag="--load_ops=${ops_path}"; else ops_flag="--save_ops=${ops_path}"; fi
  local log="${LOG_DIR}/rho_${ds}_${tag}_seed${seed}_${idx}.log"
  local keys="./datasets/${ds}"

  if [ ! -e "$keys" ]; then
    echo "MISSING dataset $keys" | tee -a "$MASTER_LOG"
    return 1
  fi

  local n
  n=$(rows_for_index "$csv" "$idx")
  if [ "$n" -ge "$need" ]; then
    echo "SKIP ${idx} ${ds} ${tag} seed=${seed} (rows=$n)" | tee -a "$MASTER_LOG"
    return 0
  fi

  wait_idle

  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${idx} ${ds} ${tag} rho=${rho} seed=${seed} need=${need} have=${n}"
    echo "csv=${csv}"
    echo "ops=${ops_flag}"
    echo "======================================================================"
  } | tee -a "$MASTER_LOG" | tee "$log"

  set +e
  OMP_NUM_THREADS=1 timeout --signal=KILL "${INDEX_TIMEOUT_SEC}" \
    taskset -c "${CORE}" "$BIN" \
      --keys_file="${keys}" --keys_file_type=binary \
      --table_size="${TABLE_SIZE}" --init_table_ratio="${rho}" \
      --operations_num="${OPERATIONS_NUM}" \
      --read=0.50 --insert=0.50 --update=0 --delete=0 --scan=0 \
      --scan_num="${SCAN_NUM}" --indexes="${idx}" --seed="${seed}" \
      --operation_order="${OP_ORDER}" --thread_num=1 \
      --output_path="${csv}" \
      --memory --latency_sample \
      ${ops_flag} >>"$log" 2>&1
  local rc=$?
  set -e

  n=$(rows_for_index "$csv" "$idx")
  if [ "$n" -ge "$need" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) PASS ${idx} ${ds} ${tag} seed=${seed} rows=${n}" | tee -a "$MASTER_LOG"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) INCOMPLETE ${idx} ${ds} ${tag} seed=${seed} rows=${n} rc=${rc}" | tee -a "$MASTER_LOG"
  fi
}

for ds in "${DATASETS[@]}"; do
  for spec in "${RHOS[@]}"; do
    IFS='|' read -r tag rho need <<< "$spec"
    for seed in "${SEEDS[@]}"; do
      for idx in "${INDEXES[@]}"; do
        run_one "$ds" "$tag" "$rho" "$need" "$seed" "$idx" || true
      done
    done
  done
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RHO_INIT_SWEEP DONE" | tee -a "$MASTER_LOG"
