#!/bin/bash
# ============================================================================
# SAILIR data-gen — STAGE 1: submit N Condor workers to generate the RAW (JSONL)
# training data for a topology. Edit the config block below, then run.
#
#   output:  data/${DATASET}_raw_jsonl/multisector_data_worker{0..N-1}.jsonl
#   next:    submit_preprocess_batched.sh  (Stage 2: pack to tensors)
#
# Each worker reduces n_scrambles random "scrambles" of the topology's masters
# and emits one training sample per reduction step (see README §3).
# ============================================================================
set -e
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

# ---- config: change these for your topology / dataset ----
TOPOLOGY=topology_input/pentagonbox     # topology dir (rel. to SAILIR_DIR); needs IBP, LI, masters
DATASET=pentagonbox_10x                  # dataset name  ->  data/${DATASET}_raw_jsonl
N_WORKERS=1000                           # number of Condor workers (= number of shards)
N_SCRAMBLES=1000                         # scrambles per worker (1000x1000 = 1M scrambles ~ 13M samples)
# ----------------------------------------------------------

RAW_DIR=${SAILIR_DIR}/data/${DATASET}_raw_jsonl
LOGDIR=${SAILIR_DIR}/data-gen/logs
mkdir -p "${RAW_DIR}" "${LOGDIR}"
JDL=${SAILIR_DIR}/data-gen/_datagen_${DATASET}.jdl

cat > "${JDL}" <<EOF
universe              = vanilla
executable            = ${SAILIR_DIR}/data-gen/datagen_worker.sh
arguments             = \$(Process) ${N_SCRAMBLES} ${RAW_DIR}
environment           = "SAILIR_DIR=${SAILIR_DIR} TOPOLOGY=${TOPOLOGY} DATASET=${DATASET} PYTHON=${PYTHON}"
output                = ${LOGDIR}/datagen_${DATASET}_\$(Process).out
error                 = ${LOGDIR}/datagen_${DATASET}_\$(Process).err
log                   = ${LOGDIR}/datagen_${DATASET}_\$(Process).log
request_cpus          = 1
request_memory        = 6GB
request_disk          = 1GB
should_transfer_files = NO
notification          = Error
queue ${N_WORKERS}
EOF

echo "Stage 1: ${N_WORKERS} workers x ${N_SCRAMBLES} scrambles  [${TOPOLOGY}]  ->  ${RAW_DIR}"
condor_submit "${JDL}"
