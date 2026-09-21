#!/bin/bash
# Shared config + helper sourced by every script in this folder.
# Not runnable on its own.
#
# Thread-safety groups (verified by reading each competitor's source for
# mutex/atomic/lock-free primitives, and cross-checked against README claims
# where available — see CRUD_AUDIT.md for the reasoning per index):
#
#   ST_ALL  — single-threaded only. No internal synchronization. Running these
#             with thread_num > 1 is a genuine data race (the harness's
#             #pragma omp parallel calls into the SAME index object from
#             multiple OS threads with zero locking) — never override
#             thread_num for these.
#   MT_ALL  — internally synchronized (locks, atomics, or lock-free structures
#             confirmed in source). Safe to scale thread_num up.
#
# hotrowex is intentionally excluded from MT_ALL: its remove() calls exit(0)
# instead of returning false when delete support is missing upstream, which
# silently kills the whole process (see CRUD_AUDIT.md). It is not in any
# multi-index list below. If you specifically want hotrowex's read/insert/scan
# numbers, run it alone with --delete=0 (see README.md).

ST_ALL=(btree artunsync alex lipp dili hot dytis)
MT_ALL=(alexol btreeolc artolc wormhole_u64 xindex finedex sali lippol dilax kanva hyper)

# dytis (github.com/unist-ssl/DyTIS): no mutex/atomic/lock-free primitive
# anywhere in its data-structure code (grepped Directory.h/ExtendibleHash.h/
# DyTIS.h) despite DyTIS.h importing <mutex> unused -- single-threaded only,
# same conclusion the RoBin paper's own evaluation reached (DyTIS only
# appears in their single-threaded index list, never their multi-threaded
# one). See CRUD_AUDIT.md for the full integration writeup.

# Per-audit validity subsets (see CRUD_AUDIT.md for the code-level evidence).
ST_VALID_UPDATE=(artunsync lipp dili dytis)                 # artunsync: valid but has a tiny unexplained miss rate
MT_VALID_UPDATE=(alexol artolc wormhole_u64 xindex lippol finedex sali dilax)

ST_VALID_SCAN=(alex lipp hot)                               # btree/artunsync/dytis excluded: broken/unimplemented (dytis fabricates keys, see CRUD_AUDIT.md)
MT_VALID_SCAN=(alexol btreeolc sali)                        # rest are broken/stub; dilax has different (value-range) semantics, kept out of direct comparison

# Delete stress uses the full ST_ALL / MT_ALL groups. artunsync (ST) and artolc
# (MT) always report remove()==true by construction (the underlying call is
# void) — their "success" numbers are not verified evidence, just not absent.

cd /home/GRE_Longitudinal || { echo "ERROR: /home/GRE_Longitudinal not found"; exit 1; }

BIN="./build/microbench"
[ -f "$BIN" ] || { echo "ERROR: $BIN not found. Build the project first."; exit 1; }

DATASET="${1:-libio}"
DATASET_PATH="./datasets/${DATASET}"
[ -f "$DATASET_PATH" ] || { echo "ERROR: dataset not found at $DATASET_PATH"; exit 1; }

TABLE_SIZE="${TABLE_SIZE:-200000000}"
INIT_TABLE_RATIO="${INIT_TABLE_RATIO:-0.25}"
OPERATIONS_NUM="${OPERATIONS_NUM:-50000000}"
SCAN_NUM="${SCAN_NUM:-100}"
SEED="${SEED:-1866}"
OP_ORDER="shuffle"

RESULTS_DIR="./results/longbench/longitudinal"
OPS_CACHE_DIR="./results/ops_cache"
LOG_DIR="./results/logs"
mkdir -p "$RESULTS_DIR" "$OPS_CACHE_DIR" "$LOG_DIR"

# run_index <index_name> <thread_num> <read> <insert> <update> <delete> <scan> <output_csv> <ops_cache_name>
#
# Runs exactly ONE index per process invocation (never comma-joins multiple
# indexes into a single --indexes= call) — artunsync has a confirmed,
# unresolved crash when run as the non-first index in a multi-index process
# (see CRUD_AUDIT.md); running one-index-per-process sidesteps it for every
# index, not just artunsync, since no other combination has been positively
# proven safe either. All indexes append to the same CSV (harness opens in
# append mode), so per-experiment results still land in one file.
#
# Full console output (bulk-load progress, per-batch throughput/success
# lines, any crash backtrace) is both shown live and appended to a matching
# .log file under results/logs/ — same base name as the CSV, so you can
# always go back and confirm a run actually went cleanly rather than trusting
# the CSV alone. The log is appended to, never overwritten, so re-running the
# same script (e.g. to add a missing index) keeps the full history in one
# file rather than silently dropping earlier runs.
run_index() {
    local idx="$1" threads="$2" r="$3" i="$4" u="$5" d="$6" s="$7" csv="$8" ops_name="$9"
    local ops_path="${OPS_CACHE_DIR}/${ops_name}.bin"
    local ops_flag
    if [ -f "$ops_path" ]; then
        ops_flag="--load_ops=${ops_path}"
    else
        ops_flag="--save_ops=${ops_path}"
    fi

    local core_range="0"
    [ "$threads" -gt 1 ] && core_range="0-$((threads - 1))"

    local csv_base
    csv_base="$(basename "${csv}" .csv)"
    local log="${LOG_DIR}/${csv_base}.log"

    {
      echo ""
      echo "======================================================================"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${idx}  threads=${threads}  R=${r} I=${i} U=${u} D=${d} S=${s}"
      echo "======================================================================"
    } | tee -a "$log"

    OMP_NUM_THREADS="${threads}" taskset -c "${core_range}" \
    "$BIN" \
        --keys_file="${DATASET_PATH}" \
        --keys_file_type=binary \
        --table_size="${TABLE_SIZE}" \
        --init_table_ratio="${INIT_TABLE_RATIO}" \
        --operations_num="${OPERATIONS_NUM}" \
        --read="${r}" --insert="${i}" --update="${u}" --delete="${d}" --scan="${s}" \
        --scan_num="${SCAN_NUM}" \
        --indexes="${idx}" \
        --seed="${SEED}" \
        --operation_order="${OP_ORDER}" \
        --thread_num="${threads}" \
        --output_path="${csv}" \
        --memory --latency_sample \
        ${ops_flag} 2>&1 | tee -a "$log"

    echo "DONE: ${idx} threads=${threads} (exit ${PIPESTATUS[0]})" | tee -a "$log"
}
