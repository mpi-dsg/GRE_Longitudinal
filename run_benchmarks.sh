#!/bin/bash
# Runs the longitudinal benchmark in a single invocation: operations are
# generated once and replayed identically for every index in INDEXES.

set -euo pipefail

DATASET_PATH="datasets/planet"
KEYS_FILE_TYPE="binary"
TABLE_SIZE="200000000"
INIT_TABLE_RATIO="0.25"

OPERATIONS_NUM="200000000"
SEED="1866"
OUTPUT_CSV="./results/benchmark_results.csv"
OP_ORDER="shuffle"   # iterate | shuffle | shuffle_sort
THREAD_NUM="32"

# operation ratios, must sum to 1.0
READ="0.5"
INSERT="0.5"
UPDATE="0.0"
DELETE="0.0"
SCAN="0.0"

INDEXES=( "alexol" "sali" "hyper_paper" )
INDEXES_ARG=$(IFS=,; echo "${INDEXES[*]}")

if [ ! -f "./build/longbench" ]; then
    echo "ERROR: ./build/longbench not found. Build the project first."
    exit 1
fi

mkdir -p results

echo "Longitudinal benchmark: ${DATASET_PATH}, seed ${SEED}"
echo "Indexes: ${INDEXES_ARG}"
echo "Results appended to ${OUTPUT_CSV}"

./build/longbench \
    --keys_file="${DATASET_PATH}" \
    --keys_file_type="${KEYS_FILE_TYPE}" \
    --table_size="${TABLE_SIZE}" \
    --init_table_ratio="${INIT_TABLE_RATIO}" \
    --operations_num="${OPERATIONS_NUM}" \
    --read="${READ}" \
    --insert="${INSERT}" \
    --delete="${DELETE}" \
    --update="${UPDATE}" \
    --scan="${SCAN}" \
    --indexes="${INDEXES_ARG}" \
    --seed="${SEED}" \
    --operation_order="${OP_ORDER}" \
    --thread_num="${THREAD_NUM}" \
    --output_path="${OUTPUT_CSV}" \
    --memory \
    --latency_sample

echo "Done."
