#!/bin/bash
# Workload spectrum (growth-only): read-heavy 80/20 and write-heavy 20/80
# Indexes: alex lipp artunsync btree dili
# Seeds: 1866 5 72
# Datasets: libio planet
# N=200M ρ=0.25 M=50M/batch  → B=15 (RH) / B=3 (WH)
set -uo pipefail
ROOT=/home/GRE_Longitudinal
cd "$ROOT"
source experiments/common.sh libio  # defaults; dataset overridden per run

INDEXES=(alex lipp artunsync btree dili)
SEEDS=(1866 5 72)
DATASETS=(libio planet)
# name|read|insert
MIXES=(
  "readheavy|0.80|0.20"
  "writeheavy|0.20|0.80"
)

INDEX_TIMEOUT_SEC=${INDEX_TIMEOUT_SEC:-7200}
RESULTS_DIR=${ROOT}/results/longbench/longitudinal  # next to half baselines
LOG_DIR=${LOG_DIR:-${ROOT}/results/logs}
OPS_CACHE_DIR=${OPS_CACHE_DIR:-${ROOT}/results/ops_cache}
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$OPS_CACHE_DIR"

MASTER_LOG="${LOG_DIR}/spectrum_growth_st.log"
: > "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SPECTRUM START" | tee -a "$MASTER_LOG"
echo "indexes=${INDEXES[*]} seeds=${SEEDS[*]} datasets=${DATASETS[*]}" | tee -a "$MASTER_LOG"
echo "mixes=readheavy(0.8/0.2) writeheavy(0.2/0.8)" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

already_done() {
  local csv="$1" idx="$2"
  python3 - "$csv" "$idx" <<'PY'
import sys
from pathlib import Path
csv, idx = sys.argv[1], sys.argv[2]
p = Path(csv)
if not p.exists():
    sys.exit(1)
lines = p.read_text().splitlines()
if not lines:
    sys.exit(1)
hdr = lines[0].split(",")
try:
    i = hdr.index("index_type")
except ValueError:
    i = 7
count = 0
for line in lines[1:]:
    if not line.strip():
        continue
    parts = line.split(",")
    if i < len(parts) and parts[i] == idx:
        count += 1
# expect at least 1 batch row; RH has 15, WH has 3 — require >=3 to be safe for WH, >=3 for RH start
# Use >= 3 as minimum complete signal for writeheavy; for readheavy we need 15
sys.exit(0 if count >= 3 else 1)
PY
}

run_one() {
  local ds="$1" mix_name="$2" read_r="$3" ins_r="$4" seed="$5" idx="$6"
  local csv="${RESULTS_DIR}/${ds}_st_${mix_name}_seed${seed}.csv"
  local ops_name="spectrum_${ds}_${mix_name}_r${read_r}_i${ins_r}_seed${seed}"
  local ops_path="${OPS_CACHE_DIR}/${ops_name}.bin"
  local ops_flag
  if [ -f "$ops_path" ]; then ops_flag="--load_ops=${ops_path}"; else ops_flag="--save_ops=${ops_path}"; fi
  local log="${LOG_DIR}/spectrum_${ds}_${mix_name}_seed${seed}.log"
  local ds_path="./datasets/${ds}"

  if already_done "$csv" "$idx"; then
    # For readheavy need 15 rows; refine check
    local n
    n=$(python3 -c "import csv; r=list(csv.DictReader(open('$csv'))); print(sum(1 for x in r if x['index_type']=='$idx'))")
    if [ "$mix_name" = "readheavy" ] && [ "$n" -ge 15 ]; then
      echo "SKIP ${idx} ${ds} ${mix_name} seed=${seed} (rows=$n)" | tee -a "$MASTER_LOG"
      return 0
    fi
    if [ "$mix_name" = "writeheavy" ] && [ "$n" -ge 3 ]; then
      echo "SKIP ${idx} ${ds} ${mix_name} seed=${seed} (rows=$n)" | tee -a "$MASTER_LOG"
      return 0
    fi
  fi

  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START ${idx} ${ds} ${mix_name} seed=${seed} R=${read_r} I=${ins_r}"
    echo "csv=${csv}"
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

  local n
  n=$(python3 -c "import csv,os; p='$csv';
import pathlib
path=pathlib.Path(p)
print(0 if not path.exists() else sum(1 for x in csv.DictReader(open(p)) if x['index_type']=='$idx'))")
  local need=3
  [ "$mix_name" = "readheavy" ] && need=15
  if [ "$n" -ge "$need" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) PASS ${idx} ${ds} ${mix_name} seed=${seed} rows=${n}" | tee -a "$MASTER_LOG"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) INCOMPLETE ${idx} ${ds} ${mix_name} seed=${seed} rows=${n} rc=${rc}" | tee -a "$MASTER_LOG"
  fi
}

pass_count=0; fail_count=0
for mix in "${MIXES[@]}"; do
  IFS='|' read -r mix_name read_r ins_r <<< "$mix"
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for idx in "${INDEXES[@]}"; do
        before=$(grep -c " PASS ${idx} ${ds} ${mix_name} seed=${seed}" "$MASTER_LOG" 2>/dev/null || true)
        run_one "$ds" "$mix_name" "$read_r" "$ins_r" "$seed" "$idx"
        if grep -q " PASS ${idx} ${ds} ${mix_name} seed=${seed}" "$MASTER_LOG"; then
          pass_count=$((pass_count+1))
        elif grep -q " INCOMPLETE ${idx} ${ds} ${mix_name} seed=${seed}" "$MASTER_LOG"; then
          fail_count=$((fail_count+1))
        fi
      done
    done
  done
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SPECTRUM DONE pass≈${pass_count} incomplete≈${fail_count}" | tee -a "$MASTER_LOG"
