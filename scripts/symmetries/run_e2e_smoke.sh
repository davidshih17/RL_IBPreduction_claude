#!/bin/bash
# End-to-end smoke test: datagen -> preprocess -> model forward pass
# for both trianglebox and pentagon-box.
set -u
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
mkdir -p scripts/symmetries/logs scripts/symmetries/tmp_e2e
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

LOG=scripts/symmetries/logs/e2e_smoke.log
: > $LOG

run_topology () {
    NAME=$1; DIR=$2
    OUTDIR=scripts/symmetries/tmp_e2e/$NAME
    mkdir -p $OUTDIR

    echo "############# $NAME #############" >> $LOG

    echo ">>> datagen" >> $LOG
    PYTHONUNBUFFERED=1 "$PY" -u data-gen/generate_multisector_data.py \
        --topology $DIR \
        --n_scrambles 10 \
        --min_steps 2 --max_steps 4 \
        --output $OUTDIR/data.jsonl \
        --start_seed 42 \
        >> $LOG 2>&1
    echo "    exit=$?  $(wc -l < $OUTDIR/data.jsonl) samples" >> $LOG

    echo ">>> preprocess" >> $LOG
    PYTHONUNBUFFERED=1 "$PY" -u data-gen/preprocess_to_tensors.py \
        --topology $DIR \
        --input $OUTDIR/data.jsonl \
        --output_dir $OUTDIR/packed \
        --val_split 0.2 --test_split 0.2 \
        >> $LOG 2>&1
    echo "    exit=$?" >> $LOG

    echo ">>> model forward pass" >> $LOG
    PYTHONUNBUFFERED=1 "$PY" -u -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str(Path('training').resolve()))
import torch
from sailir.topology import Topology
from sailir.classifier import IBPActionClassifier
from sailir import ibp_env
from train_classifier import make_collate_fn, PackedDatasetV5
from torch.utils.data import DataLoader

topology = Topology.from_dir('$DIR')
ibp_env.init_from_topology(topology)
print(f'  topology: n_indices={topology.n_indices}, n_denom={topology.n_denominators}, n_actions={topology.n_actions}')

train_data = torch.load('$OUTDIR/packed/train.pt', weights_only=False)
ds = PackedDatasetV5(train_data)
print(f'  packed train: {len(ds)} samples')

collate = make_collate_fn(topology.n_indices, topology.n_denominators)
loader = DataLoader(ds, batch_size=min(4, len(ds)), collate_fn=collate)
model = IBPActionClassifier(
    embed_dim=64, n_heads=2, n_expr_layers=1, n_cross_layers=1, n_subs_layers=1,
    prime=1009,
    n_indices=topology.n_indices, n_denominators=topology.n_denominators,
    n_ibp_ops=topology.n_actions,
)
batch = next(iter(loader))
with torch.no_grad():
    logits, probs = model(
        batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
        batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'],
        batch['sub_repl_mask'], batch['sub_mask'],
        batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
        batch['sector_mask'], batch['target_integral'],
    )
print(f'  forward pass: logits.shape={tuple(logits.shape)}, probs.shape={tuple(probs.shape)}')
print('  OK')
" >> $LOG 2>&1
    echo "    exit=$?" >> $LOG
}

run_topology trianglebox  topology_input/trianglebox
run_topology pentagonbox  topology_input/pentagonbox

cat $LOG | grep -E '###|>>>|exit=|samples|topology:|packed train|forward pass|OK'
