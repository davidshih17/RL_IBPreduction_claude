#!/bin/bash
# Relaunch of the m1/m3 probes after the canonical-masters fix (2026-07-11), plus
# gate_small/m2 in STAGED-router mode. All arms: SAILIR_SECTOR_RANK=1 (sector-senior
# order + canonical masters, one package). m1/m3 use the same monolithic router as
# the v1 sectorrank probes (their series continues); gate_small/m2 exercise
# --symmetry-staged end-to-end (their sectorrank results already passed, so any
# change isolates the staged router).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
export SAILIR_SECTOR_RANK=1

run_arm () {
    local TARGET="$1"; local TAG="$2"; local EXTRA="$3"
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
        --max-concurrent 100 --check-interval 5 --use-symmetry $EXTRA \
        > $OUTDIR/logs/orch.log 2>&1 < /dev/null &
    echo "  launched $TAG (PID=$!) -> $OUTDIR/logs/orch.log"
}

run_arm "1,1,-1,1,1,1,-1,1,-1,0,0"    m1_sectorrank_v2  ""
run_arm "1,1,0,1,1,0,-3,1,0,0,0"      m3_sectorrank_v2  ""
run_arm "1,-1,1,0,1,1,0,0,0,0,0"      gate_small_staged "--symmetry-staged"
run_arm "2,0,1,0,1,-1,0,1,-1,-1,0"    m2_staged         "--symmetry-staged"
echo "4 arms launched."
