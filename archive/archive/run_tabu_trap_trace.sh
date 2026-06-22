#!/bin/bash
# Run one TABU TRAP integral with full instrumentation:
# - SAILIR_DUMP_STUCK   → forensic dump of stuck state
# - SAILIR_TRACE_BEAM_SETS → per-step beam expr_fp-set hashes
# usage: bash run_tabu_trap_trace.sh <slug> <integral_csv>
set -e
SLUG=$1
INTEGRAL_STR=$2
if [ -z "$SLUG" ] || [ -z "$INTEGRAL_STR" ]; then
    echo "usage: $0 <slug> <integral_csv>"
    exit 1
fi
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUTDIR=$BASE/results/probe_tabu_trace_$SLUG
mkdir -p $OUTDIR
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v6.py

export PYTHONUNBUFFERED=1
export MALLOC_MMAP_THRESHOLD_=67108864
export SAILIR_END_OF_STEP_TRIM=1
export SAILIR_DUMP_STUCK=$OUTDIR/stuck_state.pkl
export SAILIR_TABU_AUDIT=1

$PYTHON -u $SCRIPT \
    --topology $TOPOLOGY --model $MODEL --integral="$INTEGRAL_STR" \
    --output $OUTDIR/result.pkl --ckpt $OUTDIR/ckpt.pkl --ckpt-every 50 \
    --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
    --max-steps 5000 --max-actions 900 --beam-sort weight \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --model-batch-chunk 8 \
    > $OUTDIR/probe.out 2>&1
