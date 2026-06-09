#!/bin/bash
# Watch a checkpoint file. Each time it's written/updated, parse its 'step'
# field and copy it to {file}.step{N}, preserving the per-step snapshot.
# Run alongside a probe to maintain a history of thick checkpoints.
# Usage: snapshot_ckpts.sh <ckpt_path> <topology_dir> [start_step] [end_step]
set -e
CKPT=$1
TOPO=$2
START=${3:-0}
END=${4:-9999}
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
INSPECT=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/inspect_thin_ckpt.py
last_mtime=0
while true; do
    if [ -f "$CKPT" ]; then
        cur_mtime=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
        if [ "$cur_mtime" != "$last_mtime" ]; then
            STEP=$($PY $INSPECT "$CKPT" 2>/dev/null | grep "^step=" | head -1 | sed 's/step=//; s/ .*//')
            if [ -n "$STEP" ] && [ "$STEP" -ge "$START" ] && [ "$STEP" -le "$END" ]; then
                cp "$CKPT" "${CKPT}.step${STEP}"
                echo "$(date +%H:%M:%S) snapshot ${CKPT}.step${STEP}"
            fi
            last_mtime=$cur_mtime
        fi
    fi
    sleep 5
done
