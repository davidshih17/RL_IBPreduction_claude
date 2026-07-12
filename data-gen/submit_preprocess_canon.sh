#!/bin/bash
# ============================================================================
# SAILIR data-gen — STAGE 2: preprocess all raw JSONL shards into PACKED tensor
# shards, submitted in BATCHES to throttle NFS / python-startup load. Uses
# Condor file transfer so each job reads/writes only its scratch dir. Edit the
# config block (must match the Stage-1 run), then run.
#
#   output:  data/${DATASET}_packed/shard_{0..N-1}/{train,val,test}.pt
#   next:    ../training/  (train the model on the packed shards)
# ============================================================================
set -e
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

# ---- config: must match the Stage-1 run ----
TOPOLOGY=topology_input/pentagonbox
DATASET=pentagonbox_canon10x
N_SHARDS=1000             # = N_WORKERS from Stage 1
BATCH_SIZE=200           # jobs submitted per batch
LOW_WATER=100            # drain in-flight preprocess jobs to <= this before next batch
SLEEP_SECS=30
# --------------------------------------------

RAW_DIR=${SAILIR_DIR}/data/${DATASET}_raw_jsonl
PACKED_DIR=${SAILIR_DIR}/data/${DATASET}_packed
LOGDIR=${SAILIR_DIR}/data-gen/logs
mkdir -p "${PACKED_DIR}" "${LOGDIR}"
JDL=${SAILIR_DIR}/data-gen/_preprocess_${DATASET}.jdl

OFFSET=0
while (( OFFSET < N_SHARDS )); do
    REMAINING=$((N_SHARDS - OFFSET))
    N=$(( REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE ))
    END=$((OFFSET + N - 1))

    # Throttle: wait until our in-flight preprocess jobs (idle=1, run=2) drop below LOW_WATER.
    while :; do
        IN_FLIGHT=$(condor_q dshih -af JobStatus Cmd 2>/dev/null \
            | awk '/preprocess_shard\.sh/ && ($1==1 || $1==2)' | wc -l)
        if (( IN_FLIGHT <= LOW_WATER )); then
            echo "[$(date +%T)] in_flight=${IN_FLIGHT} <= ${LOW_WATER}; submit shards [${OFFSET}..${END}]"
            break
        fi
        echo "[$(date +%T)] in_flight=${IN_FLIGHT} > ${LOW_WATER}, waiting ..."
        sleep ${SLEEP_SECS}
    done

    WIDS=$(seq ${OFFSET} ${END} | tr '\n' ' ')
    cat > "${JDL}" <<EOF
universe                 = vanilla
executable               = ${SAILIR_DIR}/data-gen/preprocess_shard.sh
arguments                = \$(WorkerID)
environment              = "SAILIR_DIR=${SAILIR_DIR} TOPOLOGY=${TOPOLOGY} PYTHON=${PYTHON}"

should_transfer_files    = YES
when_to_transfer_output  = ON_EXIT
transfer_input_files     = ${RAW_DIR}/multisector_data_worker\$(WorkerID).jsonl
transfer_output_files    = shard_\$(WorkerID)
transfer_output_remaps   = "shard_\$(WorkerID)=${PACKED_DIR}/shard_\$(WorkerID)"

output                   = ${LOGDIR}/preprocess_${DATASET}_shard_\$(WorkerID).out
error                    = ${LOGDIR}/preprocess_${DATASET}_shard_\$(WorkerID).err
log                      = ${LOGDIR}/preprocess_${DATASET}_shard_\$(WorkerID).log

request_cpus             = 1
request_memory           = 6GB
request_disk             = 1GB
notification             = Error
queue WorkerID in (${WIDS})
EOF
    condor_submit "${JDL}" 2>&1 | tail -2
    OFFSET=$((OFFSET + N))
done

echo "[$(date +%T)] all ${N_SHARDS} preprocess shards submitted  ->  ${PACKED_DIR}"
rm -f "${JDL}"
