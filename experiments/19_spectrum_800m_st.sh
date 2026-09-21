#!/bin/bash
# Workload spectrum at 800M (growth-only): read-heavy 80/20 and write-heavy 20/80
# Indexes: alex lipp artunsync btree hyper  (dili excluded — signed-key clash @ 800M)
# Seeds: 1866 5 72
# Datasets: books_800m osm_800m
# N=800M ρ=0.25 M=50M/batch  → B=60 (RH) / B=15 (WH)
# Skip: alex on books_800m seed=72 (known segfault from Exp12)
set -uo pipefail
ROOT=/home/GRE_Longitudinal
cd "$ROOT"

export TABLE_SIZE="${TABLE_SIZE:-800000000}"
export OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
export INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"
export SCAN_NUM="${SCAN_NUM:-100}"
export OP_ORDER="${OP_ORDER:-shuffle}"

BIN="./build/microbench"
[ -f "$BIN" ] || { echo "ERROR: $BIN not found"; exit 1; }

INDEXES=(alex lipp artunsync btree hyper)
SEEDS=(1866 5 72)
DATASETS=(books_800m osm_800m)
MIXES=(
  "readheavy|0.80|0.20"
  "writeheavy|0.20|0.80"
)

# RH @ 800M is ~60 batches — allow up to 24h per index by default
INDEX_TIMEOUT_SEC=${INDEX_TIMEOUT_SEC:-86400}
RESULTS_DIR=${RESULTS_DIR:-${ROOT}/results/longbench/longitudinal}
LOG_DIR=${LOG_DIR:-${ROOT}/results/logs}
OPS_CACHE_DIR=${OPS_CACHE_DIR:-${ROOT}/results/ops_cache}
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/spectrum_800m_st.log"
: > "$MASTER_LOG"
{
  echo "======================================================================"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SPECTRUM_800M START"
  echo "table_size=${TABLE_SIZE} ops/batch=${OPERATIONS_NUM} rho=${INIT_TABLE_RATIO}"
  echo "indexes=${INDEXES[*]} seeds=${SEEDS[*]} datasets=${DATASETS[*]}"
  echo "mixes=readheavy(0.8/0.2 B≈60) writeheavy(0.2/0.8 B≈15)"
  echo "timeout_sec=${INDEX_TIMEOUT_SEC} results=${RESULTS_DIR}"
  echo "skip=alex books_800m seed72 (Exp12 segfault)"
  echo "======================================================================"
} | tee -a "$MASTER_LOG"

should_skip() {
  local ds="$1" seed="$2" idx="$3"
  if [ "$idx" = "alex" ] && [ "$ds" = "books_800m" ] && [ "$seed" = "72" ]; then
    return 0
  fi
  return 1
}

rows_for_index() {
  local csv="$1" idx="$2"
  python3 - "$csv" "$idx" <<'PY'
import csv, sys
from pathlib import Path
p, idx = Path(sys.argv[1]), sys.argv[2]
if not p.exists():
    print(0)
    raise SystemExit
print(sum(1 for r in csv.DictReader(open(p)) if r.get("index_type") == idx))
PY
}

run_one() {
  local ds="$1" mix_name="$2" read_r="$3" ins_r="$4" seed="$5" idx="$6"
  local csv="${RESULTS_DIR}/${ds}_st_${mix_name}_seed${seed}.csv"
  local ops_name="spectrum800_${ds}_${mix_name}_r${read_r}_i${ins_r}_seed${seed}"
  local ops_path="${OPS_CACHE_DIR}/${ops_name}.bin"
  local ops_flag
  if [ -f "$ops_path" ]; then ops_flag="--load_ops=${ops_path}"; else ops_flag="--save_ops=${ops_path}"; fi
  local log="${LOG_DIR}/spectrum800_${ds}_${mix_name}_seed${seed}.log"
  local ds_path="./datasets/${ds}"

  if should_skip "$ds" "$seed" "$idx"; then
    echo "SKIP ${idx} ${ds} ${mix_name} seed=${seed} (known segfault)" | tee -a "$MASTER_LOG"
    return 0
  fi

  local need=15
  [ "$mix_name" = "readheavy" ] && need=60
  local n
  n=$(rows_for_index "$csv" "$idx")
  if [ "$n" -ge "$need" ]; then
    echo "SKIP ${idx} ${ds} ${mix_name} seed=${seed} (rows=$n)" | tee -a "$MASTER_LOG"
    return 0
  fi

  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${idx} ${ds} ${mix_name} seed=${seed} R=${read_r} I=${ins_r}"
    echo "csv=${csv} need_rows=${need} have=${n}"
    echo "======================================================================"
  } | tee -a "$MASTER_LOG" | tee -a "$log"

  set +e
  OMP_NUM_THREADS=1 timeout --signal=KILL "${INDEX_TIMEOUT_SEC}" \
    taskset -c 0 "$BIN" \
      --keys_file="${ds_path}" --keys_file_type=binary \
      --table_size="${TABLE_SIZE}" --init_table_ratio="${INIT_TABLE_RATIO}" \
      --operations_num="${OPERATIONS_NUM}" \
      --read="${read_r}" --insert="${ins_r}" --update=0 --delete=0 --scan=0 \
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
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) PASS ${idx} ${ds} ${mix_name} seed=${seed} rows=${n}" | tee -a "$MASTER_LOG"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) INCOMPLETE ${idx} ${ds} ${mix_name} seed=${seed} rows=${n} rc=${rc}" | tee -a "$MASTER_LOG"
  fi
}

pass_count=0
fail_count=0
skip_count=0
for mix in "${MIXES[@]}"; do
  IFS='|' read -r mix_name read_r ins_r <<< "$mix"
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for idx in "${INDEXES[@]}"; do
        if should_skip "$ds" "$seed" "$idx"; then
          echo "SKIP ${idx} ${ds} ${mix_name} seed=${seed} (known segfault)" | tee -a "$MASTER_LOG"
          skip_count=$((skip_count + 1))
          continue
        fi
        run_one "$ds" "$mix_name" "$read_r" "$ins_r" "$seed" "$idx"
        if grep -q " PASS ${idx} ${ds} ${mix_name} seed=${seed}" "$MASTER_LOG"; then
          pass_count=$((pass_count + 1))
        elif grep -q " INCOMPLETE ${idx} ${ds} ${mix_name} seed=${seed}" "$MASTER_LOG"; then
          fail_count=$((fail_count + 1))
        elif grep -q "SKIP ${idx} ${ds} ${mix_name} seed=${seed}" "$MASTER_LOG"; then
          skip_count=$((skip_count + 1))
        fi
      done
    done
  done
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SPECTRUM_800M DONE pass≈${pass_count} incomplete≈${fail_count} skip≈${skip_count}" | tee -a "$MASTER_LOG"
