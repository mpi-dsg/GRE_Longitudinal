#!/bin/bash
# Clean re-run of ALEX Phases A–E: one index at a time, exclusive of other GRE jobs.
# Results go to alex_phase_{a-e}_quiet/ so the old (noisy) CSVs are preserved.
set -euo pipefail
cd /home/GRE_Longitudinal

CORE="${CORE:-4}"
SEED=1866
TABLE_SIZE=5000000
INIT_TABLE_RATIO=0.25
OPERATIONS_NUM=500000
DATASET=./datasets/libio
LOG_DIR=./results/logs
SHARED_OPS=./results/longbench/longitudinal/alex_phases_quiet_shared/libio_5m_ops_seed${SEED}.bin
mkdir -p "$LOG_DIR" "$(dirname "$SHARED_OPS")"

# Prefer existing identical ops cache
if [ ! -f "$SHARED_OPS" ]; then
  for cand in \
    ./results/longbench/longitudinal/alex_localize/libio_5m_ops_seed${SEED}.bin \
    ./results/longbench/longitudinal/alex_phase_a/libio_5m_ops_seed${SEED}.bin
  do
    if [ -f "$cand" ]; then
      cp -a "$cand" "$SHARED_OPS"
      break
    fi
  done
fi

preflight() {
  # Abort if another microbench / GRE runner is already active.
  if pgrep -f './build/microbench' >/dev/null 2>&1; then
    echo "PREFLIGHT FAIL: microbench already running" >&2
    pgrep -af './build/microbench' || true
    exit 2
  fi
  if pgrep -f 'run_alex_phase_|run_blade_|run_beat_|run_lipp_|run_rail_|run_glade_|dytis' >/dev/null 2>&1; then
    # allow this script itself
    local others
    others=$(pgrep -af 'run_alex_phase_|run_blade_|run_beat_|run_lipp_|run_rail_|run_glade_|indexes=dytis' \
      | grep -v 'run_alex_phases_quiet' | grep -v "$$" || true)
    if [ -n "${others}" ]; then
      echo "PREFLIGHT FAIL: other GRE runner active:" >&2
      echo "$others" >&2
      exit 2
    fi
  fi
}

run_index() {
  local phase="$1"
  local idx="$2"
  local out_dir="$3"
  local csv="$4"
  local log="${LOG_DIR}/alex_phase_${phase}_quiet_${idx}_libio_5m_seed${SEED}.log"

  preflight
  mkdir -p "$out_dir"

  {
    echo ""
    echo "======================================================================"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  QUIET phase=${phase} index=${idx} core=${CORE}"
    echo "loadavg=$(cat /proc/loadavg)"
    echo "======================================================================"
  } | tee "$log"

  local ops_flags
  if [ -f "$SHARED_OPS" ]; then
    ops_flags="--load_ops=${SHARED_OPS}"
  else
    ops_flags="--save_ops=${SHARED_OPS}"
  fi

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
      --output_path="${csv}" \
      --memory --latency_sample \
      ${ops_flags} 2>&1 | tee -a "$log"

  local ec=${PIPESTATUS[0]}
  echo "DONE: phase=${phase} index=${idx} exit=${ec} loadavg=$(cat /proc/loadavg)" | tee -a "$log"
  if [ "$ec" -ne 0 ]; then
    echo "ABORT: ${idx} failed with ${ec}" >&2
    exit "$ec"
  fi
}

run_phase() {
  local phase="$1"
  shift
  local out_dir="./results/longbench/longitudinal/alex_phase_${phase}_quiet"
  local csv="${out_dir}/libio_5m_growth_seed${SEED}.csv"
  mkdir -p "$out_dir"
  rm -f "$csv"
  # also keep a copy of shared ops in phase dir for bookkeeping
  if [ -f "$SHARED_OPS" ]; then
    ln -sfr "$SHARED_OPS" "${out_dir}/libio_5m_ops_seed${SEED}.bin" 2>/dev/null \
      || cp -a "$SHARED_OPS" "${out_dir}/libio_5m_ops_seed${SEED}.bin"
  fi

  echo ""
  echo "########## QUIET PHASE ${phase}  core=${CORE}  $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"
  for idx in "$@"; do
    run_index "$phase" "$idx" "$out_dir" "$csv"
  done
  echo "########## QUIET PHASE ${phase} DONE  $(date -u +%Y-%m-%dT%H:%M:%SZ) ##########"
}

echo "Starting quiet A–E re-run on core ${CORE} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Shared ops: ${SHARED_OPS}"
preflight

# Phase A
run_phase a \
  alex alex_maxleaf512 alex_maxleaf256 alex_maxleaf128 alex_maxleaf64 btree

# Phase B
run_phase b \
  alex alex_maxleaf512 alex_pref_down alex_defer_root \
  alex_maxleaf512_down alex_maxleaf512_down_norooot

# Phase C
run_phase c \
  alex alex_maxleaf512 alex_defer_root \
  alex_amort64 alex_amort128 alex_amort256 \
  alex_amort128_defer alex_amort128_maxleaf512

# Phase D
run_phase d \
  alex alex_pool alex_prefault alex_pool_prefault \
  alex_amort128 alex_amort128_pool alex_amort128_pool_prefault

# Phase E
run_phase e \
  alex alex_prefault alex_arena alex_arena_mlock alex_arena_1g \
  alex_amort128_arena alex_amort128_arena_mlock

echo "ALL QUIET PHASES A–E FINISHED at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Results under results/longbench/longitudinal/alex_phase_{a,b,c,d,e}_quiet/"
