#!/bin/bash
# Verify resume is bit-identical: run 1 to step 10, save ckpt; resume from
# step 5 and run to step 10; compare per-step thick ckpts at steps 6-10.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR=results/v5_verify_resume_${STAMP}
mkdir -p $DIR/scratch $DIR/resumed
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"

echo "=== Run 1 (from scratch, 10 steps, save per-step ckpts) ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 10 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 \
    --ckpt $DIR/scratch/ckpt.pkl --ckpt-every 9999 --ckpt-every-step \
    > $DIR/scratch/run.log 2>&1
echo "    scratch exit=$?"

echo "=== Run 2 (resume from step-5 ckpt, run to step 10, save per-step ckpts) ==="
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 10 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 \
    --resume-from $DIR/scratch/ckpt.pkl.step0005 \
    --ckpt $DIR/resumed/ckpt.pkl --ckpt-every 9999 --ckpt-every-step \
    > $DIR/resumed/run.log 2>&1
echo "    resumed exit=$?"

echo "=== Diff per-step state at steps 6-10 ==="
# Diff scratch vs resumed for steps 6..10. resumed wrote step6..10; compare same files.
$PYTHON -c "
import pickle, sys
def normalize(s):
    return {k: s[k] for k in ('expr','resolved_subs','sub_accum','score','path','n_non_masters')}
n_match = 0
n_total = 0
for step in range(6, 11):
    f_a = '$DIR/scratch/ckpt.pkl.step{:04d}'.format(step)
    f_b = '$DIR/resumed/ckpt.pkl.step{:04d}'.format(step)
    with open(f_a, 'rb') as f: ca = pickle.load(f)
    with open(f_b, 'rb') as f: cb = pickle.load(f)
    beam_a = [normalize(s) for s in ca['beam']]
    beam_b = [normalize(s) for s in cb['beam']]
    n_total += 1
    if beam_a == beam_b:
        n_match += 1
        print(f'  step {step}: PASS')
    else:
        print(f'  step {step}: DIFFER  (first diff: {next((i for i,(a,b) in enumerate(zip(beam_a,beam_b)) if a!=b), None)})')
print(f'\\n=== {n_match}/{n_total} steps bit-identical ===')
"
