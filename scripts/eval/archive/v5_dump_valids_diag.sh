#!/bin/bash
# Run scratch+resumed, dumping valid action lists at step 7 in each.
set -e
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
STAMP=$(date +%s)
DIR=results/v5_valids_diag_${STAMP}
mkdir -p $DIR/scratch $DIR/resumed
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=topology_input/pentagonbox
INT="-1,2,1,0,1,2,1,1,-3,0,0"

echo "=== Run scratch (7 steps, dump valids at step 7) ==="
V5_DUMP_VALIDS_AT_STEP=7 V5_DUMP_PATH=$DIR/scratch/valids.pkl \
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 7 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 \
    --ckpt $DIR/scratch/ckpt.pkl --ckpt-every 9999 --ckpt-every-step \
    > $DIR/scratch/run.log 2>&1
echo "  scratch exit=$?"

echo "=== Run resumed (from step-6 ckpt, dump valids at step 7) ==="
V5_DUMP_VALIDS_AT_STEP=7 V5_DUMP_PATH=$DIR/resumed/valids.pkl \
PYTHONUNBUFFERED=1 $PYTHON -u scripts/eval/beam_search_v5.py \
    --topology $TOPO --model $MODEL --integral="$INT" \
    --beam-width 40 --max-steps 7 --max-actions 900 --beam-sort mixed \
    --no-paper-masters-only --prime 1009 --n-threads 1 --device cpu \
    --tabu --no-exprkeyed --iraws-keep-first 50 \
    --resume-from $DIR/scratch/ckpt.pkl.step0006 \
    --ckpt $DIR/resumed/ckpt.pkl --ckpt-every 9999 --ckpt-every-step \
    > $DIR/resumed/run.log 2>&1
echo "  resumed exit=$?"

echo ""
echo "=== Diff valids ==="
$PYTHON -c "
import pickle
a = pickle.load(open('$DIR/scratch/valids.pkl', 'rb'))
b = pickle.load(open('$DIR/resumed/valids.pkl', 'rb'))
print(f'scratch tasks: {len(a)}; resumed tasks: {len(b)}')
# Compare task by task
n_task_match = 0
n_valid_match = 0
n_total = 0
for i, (ta, tb) in enumerate(zip(a, b)):
    pi_a, tgt_a, va = ta
    pi_b, tgt_b, vb = tb
    n_total += 1
    if (pi_a, tgt_a) == (pi_b, tgt_b):
        n_task_match += 1
    sa = set(va)
    sb = set(vb)
    if sa == sb:
        n_valid_match += 1
    else:
        if n_total - n_valid_match <= 3:
            print(f'  task {i}: parent={pi_a} target={tgt_a[:4]}...')
            print(f'    |valid_a|={len(va)} |valid_b|={len(vb)} '
                  f'shared={len(sa&sb)} only_a={len(sa-sb)} only_b={len(sb-sa)}')
            print(f'    same first 5: {va[:5] == vb[:5]}')
            if va[:5] != vb[:5]:
                print(f'      va[:5]={va[:5]}')
                print(f'      vb[:5]={vb[:5]}')
print(f'\\nTotal tasks: {n_total}, (parent,target) match: {n_task_match}, valid-SET match: {n_valid_match}')
"
