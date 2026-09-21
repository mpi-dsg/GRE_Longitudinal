#!/bin/bash
# Batch-size sensitivity (Exp 2a): balanced growth-only on libio
# Same operators as default half (R/I=0.5/0.5, ρ=0.25, N=200M)
# Sweep OPERATIONS_NUM (M) ∈ {25M, 100M}  — 50M already covered by libio_st_half_*
# Seeds: 1866 5 72
# Indexes: alex lipp artunsync btree dili
#
# B = floor(150M / (M * 0.5)) → M=25M ⇒ B=12; M=100M ⇒ B=3
set -uo pipefail
ROOT=/home/GRE_Longitudinal
cd "$ROOT"

export TABLE_SIZE="${TABLE_SIZE:-200000000}"
export INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"
export SCAN_NUM="${SCAN_NUM:-100}"
export OP_ORDER="${OP_ORDER:-shuffle}"

BIN="./build/microbench"
[ -f "$BIN" ] || { echo "ERROR: $BIN not found"; exit 1; }
[ -f "./datasets/libio" ] || { echo "ERROR: datasets/libio missing"; exit 1; }

INDEXES=(alex lipp artunsync btree dili)
SEEDS=(1866 5 72)
DATASET=libio
# tag|ops_per_batch|need_rows
BATCHES=(
  "m25m|25000000|12"
  "m100m|100000000|3"
)

INDEX_TIMEOUT_SEC=${INDEX_TIMEOUT_SEC:-14400}
RESULTS_DIR=${RESULTS_DIR:-${ROOT}/results/longbench/longitudinal}
LOG_DIR=${LOG_DIR:-${ROOT}/results/logs}
OPS_CACHE_DIR=${OPS_CACHE_DIR:-${ROOT}/results/ops_cache}
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/batchsize_sweep_libio_st.log"
if [ ! -f "$MASTER_LOG" ]; then : > "$MASTER_LOG"; fi

{
  echo "======================================================================"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) BATCHSIZE_SWEEP START"
  echo "dataset=${DATASET} table_size=${TABLE_SIZE} rho=${INIT_TABLE_RATIO} R/I=0.5/0.5"
  echo "M tags: m25m (B≈12) m100m (B≈3)  — default m50m already in libio_st_half_*"
  echo "indexes=${INDEXES[*]} seeds=${SEEDS[*]}"
  echo "timeout_sec=${INDEX_TIMEOUT_SEC} results=${RESULTS_DIR}"
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

run_one() {
  local tag="$1" ops="$2" need="$3" seed="$4" idx="$5"
  local csv="${RESULTS_DIR}/${DATASET}_st_half_${tag}_seed${seed}.csv"
  local ops_name="batchsize_${DATASET}_half_${tag}_r0.50_i0.50_seed${seed}"
  local ops_path="${OPS_CACHE_DIR}/${ops_name}.bin"
  local ops_flag
  if [ -f "$ops_path" ]; then ops_flag="--load_ops=${ops_path}"; else ops_flag="--save_ops=${ops_path}"; fi
  local log="${LOG_DIR}/batchsize_${DATASET}_half_${tag}_seed${seed}.log"

  local n
  n=$(rows_for_index "$csv" "$idx")
  if [ "$n" -ge "$need" ]; then
    echo "SKIP ${idx} ${tag} seed=${seed} (rows=$n)" | tee -a "$MASTER_LOG"
    return 0
  fi

  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${idx} ${DATASET} half ${tag} seed=${seed} M=${ops} need=${need} have=${n}"
    echo "csv=${csv}"
    echo "======================================================================"
  } | tee -a "$MASTER_LOG" | tee -a "$log"

  set +e
  OMP_NUM_THREADS=1 timeout --signal=KILL "${INDEX_TIMEOUT_SEC}" \
    taskset -c 0 "$BIN" \
      --keys_file="./datasets/${DATASET}" --keys_file_type=binary \
      --table_size="${TABLE_SIZE}" --init_table_ratio="${INIT_TABLE_RATIO}" \
      --operations_num="${ops}" \
      --read=0.50 --insert=0.50 --update=0 --delete=0 --scan=0 \
      --scan_num="${SCAN_NUM}" --indexes="${idx}" --seed="${seed}" \
      --operation_order="${OP_ORDER}" --thread_num=1 \
      --output_path="${csv}" \
      --memory --latency_sample \
      ${ops_flag} 2>&1 | tee -a "$log" | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  pkill -9 -f "--indexes=${idx} .*--output_path=${csv}" 2>/dev/null || true

  n=$(rows_for_index "$csv" "$idx")
  if [ "$n" -ge "$need" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) PASS ${idx} ${tag} seed=${seed} rows=${n}" | tee -a "$MASTER_LOG"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) INCOMPLETE ${idx} ${tag} seed=${seed} rows=${n} rc=${rc}" | tee -a "$MASTER_LOG"
  fi
}

pass_count=0
fail_count=0
for spec in "${BATCHES[@]}"; do
  IFS='|' read -r tag ops need <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    for idx in "${INDEXES[@]}"; do
      run_one "$tag" "$ops" "$need" "$seed" "$idx"
      if grep -q " PASS ${idx} ${tag} seed=${seed}" "$MASTER_LOG"; then
        pass_count=$((pass_count + 1))
      elif grep -q " INCOMPLETE ${idx} ${tag} seed=${seed}" "$MASTER_LOG"; then
        fail_count=$((fail_count + 1))
      fi
    done
  done
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) BATCHSIZE_SWEEP DONE pass≈${pass_count} incomplete≈${fail_count}" | tee -a "$MASTER_LOG"
