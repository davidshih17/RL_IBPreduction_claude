#!/bin/bash
# Round 2 of the FRESH (8,5) reduction, in PAPER-MASTERS-ONLY mode.
#
# The fresh round-1 run (results/pentagonbox_8_5_v7_fresh) reduced the (8,5) top
# integral to 253 terms = 61 PAPER masters + 192 CORNER integrals, using IBP+LI
# only (no symmetry). Those 192 corners live in sectors with no Kira master, so
# in --no-paper-masters-only mode they are TERMINAL -- which is why two different
# reduction paths (fresh vs round1-4) disagree on 42 of the 61 paper-master
# coefficients: the terminal set {papers + corners} is overcomplete.
#
# To get the unique Kira-basis answer we must push the corners all the way down
# to the 61 paper masters. This round:
#   * --resume-from the fresh round-1 cache (replay_state.pkl), so the (8,5)->253
#     reduction is reused, not redone (Hits, not new workers).
#   * --paper-masters-only: the orchestrator + workers now treat the 192 corners
#     as NON-masters and reduce them further (IBP+LI) toward the paper masters.
#
# EXPERIMENTAL: SAILIR has no symmetry/scaleless rules, so some corners may be
# irreducible under IBP+LI alone. If so the run will end "INCOMPLETE" with those
# corners remaining -- which itself tells us which sectors need symmetry.
#
# Output goes to a FRESH dir; the round-1 fresh result is NOT touched.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
R1=$BASE/results/pentagonbox_8_5_v7_fresh
OUTDIR=$BASE/results/pentagonbox_8_5_v7_fresh_round2
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
# 62-master no-symmetry basis (61 Kira masters + sector 161, the k1<->k2-symmetry
# corner SAILIR was stuck on). With 161 a master, --paper-masters-only terminates.
TOPOLOGY=$BASE/topology_input/pentagonbox_nosym
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"

if [ ! -f "$R1/replay_state.pkl" ]; then
  echo "ERROR: $R1/replay_state.pkl missing (build it with save_replay_state.py first)" >&2
  exit 1
fi

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/scripts/eval/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --resume-from $R1 \
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
  > $OUTDIR/logs/hierarchical_round2_pmo.log 2>&1 &
ORCH_PID=$!
set +x
echo "FRESH round2 (paper-masters-only) orchestrator launched PID=$ORCH_PID"
echo "  resumes cache from: $R1/replay_state.pkl"
echo "  log:     $OUTDIR/logs/hierarchical_round2_pmo.log"
echo "  result:  $OUTDIR/reduction.pkl"
