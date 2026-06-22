#!/bin/bash
# Parse hierarchical.log and plot non-masters count vs iteration number.
# Output: PNG plot + CSV next to the log.
set -e

LOG=${1:-/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_v3/logs/hierarchical.log}
OUTDIR=$(dirname $LOG)
CSV=$OUTDIR/nonmasters_vs_iter.csv
PNG=$OUTDIR/nonmasters_vs_iter.png

export LOG CSV PNG

PYTHON=/het/p4/dshih/conda_envs/pyg4/bin/python

$PYTHON /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval/_plot_nonmasters_vs_iter.py
