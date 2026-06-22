#!/bin/bash
# Round 5 of the ORIGINAL piecemeal (8,5) reduction (pentagonbox_8_5_v6), in
# PAPER-MASTERS-ONLY mode -- the exact analog of the fresh round2.
#
# Rounds 1-4 reduced the (8,5) top integral to the VERIFIED gold expression
# 262 terms = 61 PAPER masters + 201 CORNER integrals (results/pentagonbox_8_5_v6_round4/
# replay_state_all4.pkl, cache = 103,626 entries). Those 201 corners are terminal
# only because --no-paper-masters-only was used; in PAPER-MASTERS-ONLY mode they
# are non-masters and must reduce further to the 61 paper masters.
#
# Goal: push the 201 corners down to the 61 paper masters, then compare the
# resulting 61-coefficient vector against the fresh round2 result. If both land
# on the SAME 61 coefficients, the two original reductions were both correct and
# differed only by symmetry/scaleless relations (which all vanish).
#
# --resume-from needs <dir>/replay_state.pkl; the gold combined cache is named
# replay_state_all4.pkl, so we stage it via a symlink source dir.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
GOLD=$BASE/results/pentagonbox_8_5_v6_round4/replay_state_all4.pkl
OUTDIR=$BASE/results/pentagonbox_8_5_v6_round5
SRC=$OUTDIR/cache_src
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results $SRC
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
# 62-master no-symmetry basis (61 Kira masters + sector 161, the k1<->k2-symmetry
# corner SAILIR was stuck on). With 161 a master, --paper-masters-only terminates.
TOPOLOGY=$BASE/topology_input/pentagonbox_nosym
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"

if [ ! -f "$GOLD" ]; then echo "ERROR: gold cache $GOLD missing" >&2; exit 1; fi
# Stage the verified gold combined cache as the --resume-from source.
ln -sf "$GOLD" "$SRC/replay_state.pkl"
echo "staged gold cache: $SRC/replay_state.pkl -> $GOLD"

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --resume-from $SRC \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only \
    --use-v7-worker \
    --v7-cpus 1 \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
  > $OUTDIR/logs/hierarchical_round5_pmo.log 2>&1 &
ORCH_PID=$!
set +x
echo "v6 round5 (paper-masters-only) orchestrator launched PID=$ORCH_PID"
echo "  resumes gold cache from: $SRC/replay_state.pkl"
echo "  log:     $OUTDIR/logs/hierarchical_round5_pmo.log"
echo "  result:  $OUTDIR/reduction.pkl"
