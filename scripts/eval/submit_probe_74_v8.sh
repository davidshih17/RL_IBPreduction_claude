#!/bin/bash
# Submit the v8 probe-74 convergence test to Condor (NOT login node).
# v8 = target + strip + raw-strip + beam-ranking ALL on the full total ordering.
# 8 threads / 8 workers (documented optimal); thread-caps auto-fire at >1.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
O=$BASE/results/probe_74_v8
mkdir -p $O
cd $BASE
condor_submit $O/probe.sub
