#!/bin/bash
# Run v5 (8,4) reduction, save best_state path, then replay against the
# full unstripped start_expr and report the actual highest weight of
# what remains.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/logs
mkdir -p $LOGDIR
STAMP=$(date +%s)
RUN_LOG=$LOGDIR/v5_smoke_run_${STAMP}.log
REPLAY_LOG=$LOGDIR/v5_smoke_replay_${STAMP}.log
OUT=$LOGDIR/v5_smoke_${STAMP}.pkl
TOPO=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/checkpoints/pentagonbox_10x_loop_100/best_model.pt

echo "=== STEP 1: run v5 ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO \
    --model $MODEL \
    --integral='-1,2,1,0,1,2,1,1,-3,0,0' \
    --beam-width 40 \
    --max-steps 30 \
    --max-actions 900 \
    --beam-sort mixed \
    --no-paper-masters-only \
    --prime 1009 \
    --n-threads 1 \
    --device cpu \
    --output $OUT > $RUN_LOG 2>&1
echo "v5 exit=$? log=$RUN_LOG"
tail -10 $RUN_LOG

echo ""
echo "=== STEP 2: replay path against full start_expr ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/v5_verify_replay.py \
    --ckpt $OUT \
    --topology $TOPO \
    --integral='-1,2,1,0,1,2,1,1,-3,0,0' \
    --prime 1009 > $REPLAY_LOG 2>&1
echo "replay exit=$? log=$REPLAY_LOG"
echo ""
echo "=== REPLAY OUTPUT ==="
cat $REPLAY_LOG
