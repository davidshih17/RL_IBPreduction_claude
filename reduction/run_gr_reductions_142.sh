#!/bin/bash
# Restart the 3 gravity3L benchmark arms on the COMPLETE 142-master cut-derived
# basis (topology_input/gravity3L/masters, installed 2026-07-20). Same recipe
# as run_gr_first_reductions.sh but with --resume: reuses each arm's banked
# work/results (one-step reductions are basis-independent; the new basis only
# changes which integrals are dispatched vs terminal).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/gravity3L_canon10x_nosubs/best_model.pt
TOPOLOGY=$BASE/topology_input/gravity3L
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1

run_arm () {
    local TARGET="$1"; local TAG="$2"
    local OUTDIR=$BASE/results/gr_reduce/$TAG
    if [ ! -d "$OUTDIR/work/results" ]; then
        echo "  no banked work dir for $TAG"; return 1
    fi
    PYTHONUNBUFFERED=1 setsid $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
        --topology $TOPOLOGY --integral="$TARGET" \
        --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
        --model-checkpoint $MODEL \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
        --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
        --max-concurrent 100 --check-interval 5 --use-symmetry --resume \
        > $OUTDIR/logs/orch_142.log 2>&1 < /dev/null &
    echo "  relaunched $TAG (PID=$!) -> $OUTDIR/logs/orch_142.log"
}

run_arm "1,0,1,-1,1,1,1,0,1,1,-1,0,0,0,0"    g885
run_arm "1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0"    g127
run_arm "1,0,1,1,1,1,1,0,1,1,-1,0,0,0,-1"    g893
echo "3 gravity arms relaunched on the 142-master basis."
