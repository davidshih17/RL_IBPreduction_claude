#!/bin/bash
# Submit the 1000-shard preprocess in batches to throttle concurrency.
# Spreads python-startup load on NFS by submitting BATCH_SIZE jobs, waiting
# until the in-flight count is below LOW_WATER, then submitting more.
set -e

SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
JDL=${SAILIR_DIR}/scripts/data_gen/preprocess_pentagonbox_10x_batch.jdl

BATCH_SIZE=200            # jobs per batch
LOW_WATER=100             # drain to <= this many before submitting next batch
SLEEP_SECS=30             # poll interval
TOTAL_SHARDS=1000

mkdir -p ${SAILIR_DIR}/data/pentagonbox_10x_packed
mkdir -p ${SAILIR_DIR}/scripts/data_gen/logs

OFFSET=0
while (( OFFSET < TOTAL_SHARDS )); do
    REMAINING=$((TOTAL_SHARDS - OFFSET))
    N=$((REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE))
    END=$((OFFSET + N - 1))

    # Wait until current dshih queue is below LOW_WATER (count only OUR
    # preprocess jobs, not other clusters).
    while :; do
        # Count only ACTIVE (idle=1, running=2) preprocess jobs — skip
        # status=3 (X/removed zombies) so we don't get stuck.
        IN_FLIGHT=$(condor_q dshih -af JobStatus Cmd 2>/dev/null \
            | awk '/preprocess_shard_pentagonbox_10x/ && ($1==1 || $1==2)' | wc -l)
        if (( IN_FLIGHT <= LOW_WATER )); then
            echo "[$(date +%T)] in_flight=${IN_FLIGHT} <= ${LOW_WATER}; submitting batch [${OFFSET}..${END}]"
            break
        fi
        echo "[$(date +%T)] in_flight=${IN_FLIGHT} (waiting for <= ${LOW_WATER}) ..."
        sleep ${SLEEP_SECS}
    done

    # Build a per-batch jdl. Use `queue VAR in (list)` syntax with WorkerIDs
    # generated explicitly — portable across Condor versions.
    WIDS=$(seq ${OFFSET} ${END} | tr '\n' ' ')
    cat > "${JDL}" <<EOF
universe                 = vanilla
executable               = ${SAILIR_DIR}/scripts/data_gen/preprocess_shard_pentagonbox_10x.sh
arguments                = \$(WorkerID)

should_transfer_files    = YES
when_to_transfer_output  = ON_EXIT
transfer_input_files     = ${SAILIR_DIR}/data/pentagonbox_10x_raw_jsonl/multisector_data_worker\$(WorkerID).jsonl
transfer_output_files    = shard_\$(WorkerID)
transfer_output_remaps   = "shard_\$(WorkerID)=${SAILIR_DIR}/data/pentagonbox_10x_packed/shard_\$(WorkerID)"

output                   = ${SAILIR_DIR}/scripts/data_gen/logs/preprocess_pb_10x_shard_\$(WorkerID).out
error                    = ${SAILIR_DIR}/scripts/data_gen/logs/preprocess_pb_10x_shard_\$(WorkerID).err
log                      = ${SAILIR_DIR}/scripts/data_gen/logs/preprocess_pb_10x_shard_\$(WorkerID).log

request_cpus             = 1
request_memory           = 6GB
request_disk             = 1GB

notification             = Error

queue WorkerID in (${WIDS})
EOF

    condor_submit "${JDL}" 2>&1 | tail -2

    OFFSET=$((OFFSET + N))
done

echo "[$(date +%T)] All ${TOTAL_SHARDS} shards submitted in batches of ${BATCH_SIZE}."
rm -f "${JDL}"
