#!/bin/bash
# ============================================================================
# SAILIR data-gen — CANONICAL-SECTOR dataset (production design, 2026-07-11).
# Identical to submit_datagen.sh in every respect EXCEPT: scrambles are
# restricted to the 174 canonical sectors (results/canonical_sectors_tkey.txt,
# built by reduction/build_canonical_sectors_tkey.py, gated by
# verify_canonical_rep.py). This matches the locked inference pipeline: the
# hybrid router guarantees workers only ever see canonical-sector targets, so
# the model trains exactly on the sectors it will meet.
# The canonical-masters convention holds automatically: non-canonical sectors
# (incl. 152, whose Kira master moved to the 161 corner) are never scrambled,
# and canonical sectors without a Kira entry use their corner = the canonical
# master. Symmetry actions were tried and DROPPED (see --sym-actions help).
#
#   output:  data/${DATASET}_raw_jsonl/multisector_data_worker{0..N-1}.jsonl
#   next:    submit_preprocess_batched.sh  (Stage 2: pack to tensors)
# ============================================================================
set -e
SAILIR_DIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

# ---- config ----
TOPOLOGY=topology_input/pentagonbox
DATASET=pentagonbox_canon10x            # -> data/pentagonbox_canon10x_raw_jsonl
N_WORKERS=1000
N_SCRAMBLES=1000
RESTRICT_FILE=${SAILIR_DIR}/results/canonical_sectors_tkey.txt
# ----------------

if [ ! -s "${RESTRICT_FILE}" ]; then
    echo "FATAL: ${RESTRICT_FILE} missing — run reduction/build_canonical_sectors_tkey.py"
    exit 1
fi

RAW_DIR=${SAILIR_DIR}/data/${DATASET}_raw_jsonl
LOGDIR=${SAILIR_DIR}/data-gen/logs
mkdir -p "${RAW_DIR}" "${LOGDIR}"
JDL=${SAILIR_DIR}/data-gen/_datagen_${DATASET}.jdl

cat > "${JDL}" <<EOF
universe              = vanilla
executable            = ${SAILIR_DIR}/data-gen/datagen_worker.sh
arguments             = \$(Process) ${N_SCRAMBLES} ${RAW_DIR}
environment           = "SAILIR_DIR=${SAILIR_DIR} TOPOLOGY=${TOPOLOGY} DATASET=${DATASET} PYTHON=${PYTHON} RESTRICT_FILE=${RESTRICT_FILE}"
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

echo "Stage 1 (CANONICAL): ${N_WORKERS} workers x ${N_SCRAMBLES} scrambles  [${TOPOLOGY}]"
echo "  sectors: $(tr ',' '\n' < ${RESTRICT_FILE} | grep -c .) canonical  ->  ${RAW_DIR}"
condor_submit "${JDL}"
