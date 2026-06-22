#!/bin/bash
# Submit v7 + v8 greedy (beam_width=1) trajectory runs on probe 74, with the
# per-step pick dump (parent_expr, target, all valid actions + model probs).
# Deterministic (1 thread / 1 worker). max-steps 50. Used to compare the two
# trajectories step-by-step and to re-score v7's actions under v8's strip.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for tag in v7 v8; do
    mkdir -p $BASE/results/traj_${tag}_74/picks
done
cd $BASE
condor_submit $BASE/results/traj_v7_74/probe.sub
condor_submit $BASE/results/traj_v8_74/probe.sub
