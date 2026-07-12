#!/bin/bash
# A/B probes for the same-(r,s) symmetry drop detector (SAILIR_SYM_DROP=1) on top of
# the full production stack (hybrid router + sector-senior order + canonical
# masters). Compare against: m2_hybrid (identical stack, no detector), and
# m1_sectorrank_v2 / m3_sectorrank_v2.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
export SAILIR_SECTOR_RANK=1
export SAILIR_SYM_DROP=1

run_arm () {
    local TARGET="$1"; local TAG="$2"
    local OUTDIR=$BASE/results/ab_symmetry/$TAG/design1
    if [ -d "$OUTDIR" ]; then
        echo "  REFUSING to overwrite existing $OUTDIR"; return 1
    fi
    mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
    PYTHONUNBUFFERED=1 setsid $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
        --topology $TOPOLOGY --integral="$TARGET" \
        --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
        --model-checkpoint $MODEL \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
        --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
        --max-concurrent 100 --check-interval 5 --use-symmetry \
        > $OUTDIR/logs/orch.log 2>&1 < /dev/null &
    echo "  launched $TAG (PID=$!) -> $OUTDIR/logs/orch.log"
}

run_arm "2,0,1,0,1,-1,0,1,-1,-1,0"    m2_symdrop
run_arm "1,1,-1,1,1,1,-1,1,-1,0,0"    m1_symdrop
run_arm "1,1,0,1,1,0,-3,1,0,0,0"      m3_symdrop
echo "3 symdrop arms launched."
