#!/bin/bash
# Worker script for multi-sector IBP data generation on condor
# Usage: ./datagen_worker.sh <worker_id> <n_scrambles> <output_dir>

set -e

WORKER_ID=$1
N_SCRAMBLES=${2:-1000}
OUTPUT_DIR=${3:-/het/p4/dshih/jet_images-deep_learning/RL_IBPreduction_claude/data/multisector}

# Set up environment
cd /het/p4/dshih/jet_images-deep_learning/RL_IBPreduction_claude

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Calculate starting seed to ensure non-overlapping random seeds across workers
# Each worker gets seeds: worker_id * 1000000 + scramble_idx * 1000
START_SEED=$((WORKER_ID * 1000000))

OUTPUT_FILE="${OUTPUT_DIR}/multisector_data_worker${WORKER_ID}.jsonl"

echo "========================================"
echo "Multi-sector IBP Data Generation Worker"
echo "========================================"
echo "Worker ID: ${WORKER_ID}"
echo "Start seed: ${START_SEED}"
echo "N scrambles: ${N_SCRAMBLES}"
echo "Output: ${OUTPUT_FILE}"
echo "========================================"

python3 scripts/data_gen/generate_multisector_data.py \
    --n_scrambles ${N_SCRAMBLES} \
    --start_seed ${START_SEED} \
    --output ${OUTPUT_FILE} \
    --prime 1009 \
    --min_steps 5 \
    --max_steps 25 \
    --ibp_path scripts/data_gen/IBP \
    --li_path scripts/data_gen/LI

echo "Worker ${WORKER_ID} completed."
