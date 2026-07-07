#!/bin/bash
# A/B: reduce ONE target twice via the orchestrator (Condor) -- baseline vs Design-1
# symmetry routing -- with identical settings so masters must match and worker counts
# are comparable. Uses --paper-masters-only to match the meta_reduce ground truth.
# Args: TARGET_CSV  TAG
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
TARGET="$1"; TAG="$2"
ROOT=$BASE/results/ab_symmetry/$TAG

run_one () {
    local MODE="$1"; local SYMFLAG="$2"
    local OUTDIR=$ROOT/$MODE
    mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
    PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
        --topology $TOPOLOGY --integral="$TARGET" \
        --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
        --model-checkpoint $MODEL \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
        --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
        --max-concurrent 100 --check-interval 5 $SYMFLAG \
        > $OUTDIR/logs/orch.log 2>&1 &
    echo "  launched $MODE (PID=$!) -> $OUTDIR/logs/orch.log"
}

echo "A/B target=$TARGET tag=$TAG"
run_one baseline ""
run_one design1  "--use-symmetry"
echo "both launched under $ROOT"
