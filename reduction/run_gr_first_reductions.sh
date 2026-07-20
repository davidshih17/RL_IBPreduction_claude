#!/bin/bash
# First end-to-end gravity3L reductions (symmetry-enhanced production stack,
# §4b recipe): three easy NONZERO benchmark targets with small FIRE answers,
# each finishing in an exact FIRE-oracle cross-check
# (cmp_gr_vs_oracle.py). None passes through the sector-767 orbit, so the
# debris-dictionary hook is not needed for these.
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

run_arm "1,0,1,-1,1,1,1,0,1,1,-1,0,0,0,0"    g885    # canonical L=7, FIRE 2 terms
run_arm "1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0"    g127    # canonical L=7, FIRE 5 terms
run_arm "1,0,1,1,1,1,1,0,1,1,-1,0,0,0,-1"    g893    # NON-canonical L=8, FIRE 2 terms
echo "3 gravity arms launched."
