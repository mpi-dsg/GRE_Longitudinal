#!/bin/bash

# Configuration: Update these paths and dataset specifics
DATASET_PATH="datasets/covid"  # TODO: Update this dataset path
KEYS_FILE_TYPE="binary"
TABLE_SIZE="10000000"
INIT_TABLE_RATIO="0.5"

# Benchmark Dynamics
OPERATIONS_NUM="5000000"
SEED="1866" # Using the SAME seed guarantees IDENTICAL workload snapshots for every index
OUTPUT_CSV="./results/benchmark_results.csv"
OP_ORDER="iterate" # Options: 'iterate' (sort/shuffle alternating), 'shuffle', 'shuffle_sort'
THREAD_NUM="1"

# Operation Ratios (Must sum exactly to 1.0)
READ="0.2"
INSERT="0.2"
UPDATE="0.2"
DELETE="0.2"
SCAN="0.2"

# Available Competitors: Note that based on the CMake output you have various indexes linked
INDEXES=( "alexol" "xindex" "artsync" "masstree" "finedex" "kanva" "dili" "dilax" "hot" )

# Create results directory if it doesn't exist
mkdir -p results

echo "Starting Longitudinal Benchmark Experiments..."
echo "Random Seed fixed at ${SEED} ensuring identical operation generation sequences."
echo "Results will be appended to ${OUTPUT_CSV}"

# Loop through each competitor index
for INDEX in "${INDEXES[@]}"; do
    echo "=========================================================="
    echo " Evaluating Index: ${INDEX}"
    echo "=========================================================="
    
    # Run the compiled microbench tool
    ./build/microbench \
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
        --index="${INDEX}" \
        --seed="${SEED}" \
        --operation_order="${OP_ORDER}" \
        --thread_num="${THREAD_NUM}" \
        --output_path="${OUTPUT_CSV}" \
        --memory \
        --latency_sample
        # You can append --data_shift here if you want to experiment with dataset keyspace shifts!

    echo "Finished ${INDEX}."
    echo ""
done

echo "All competitor evaluation runs are completed!"
