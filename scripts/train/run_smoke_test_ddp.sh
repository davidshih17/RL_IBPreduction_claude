#!/bin/bash
# DDP smoke test: 9 train shards + 9 val shards (3 per rank), 3 epochs.
#
# Designed to catch end-of-epoch / train→val NCCL desync bugs in <5 minutes.
# Exercises the full pipeline:
#   - DDP init (3 ranks)
#   - Sharded streaming dataset with rank-stride partition
#   - Per-rank uneven sample counts (handled by Join context)
#   - Train epoch with backward + DDP gradient all-reduce
#   - Train→val boundary (the spot where the previous run hung after 2.5h)
#   - Val epoch (forward only)
#   - Post-epoch all-reduce of metrics
#   - Rank-0 checkpoint save
#   - Next epoch using same shards (tests epoch repeatability)
#
# Output is isolated: smoke_checkpoints/ + logs/smoke_test_ddp.log so it
# doesn't touch the real training state.

set -euo pipefail

cd /home/shih/work/SAILIR_phase2

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

mkdir -p smoke_checkpoints logs

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
# Lower NCCL timeout so a desync surfaces in 1 min not 10.
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Force a quick timeout so smoke test fails fast on desync.
export NCCL_TIMEOUT=60

LOG=logs/smoke_test_ddp.log
: > "$LOG"  # truncate

echo "Smoke test launched: $(date -Iseconds)" | tee -a "$LOG"

# Run in foreground (smoke test is short, want to see output and final exit code).
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    scripts/train/train_classifier.py \
    --topology       topology_input/pentagonbox \
    --shards_dir     data/pentagonbox_10x_packed \
    --max_train_shards 9 \
    --n_val_shards   9 \
    --buffer_shards  2 \
    --output_dir     smoke_checkpoints \
    --epochs         3 \
    --batch_size     64 \
    --lr             4e-4 \
    --prime          1009 \
    --device         cuda \
    --num_workers    2 \
    --checkpoint_every 1 \
    --log_every      20 \
    --seed           0 \
    2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "" | tee -a "$LOG"
echo "Smoke test exit code: $RC" | tee -a "$LOG"

# Validate expected outputs.
echo "" | tee -a "$LOG"
echo "=== Validation ===" | tee -a "$LOG"
for f in smoke_checkpoints/last.pt smoke_checkpoints/best_model.pt \
         smoke_checkpoints/checkpoint_epoch1.pt smoke_checkpoints/checkpoint_epoch2.pt \
         smoke_checkpoints/checkpoint_epoch3.pt smoke_checkpoints/final_model.pt; do
    if [ -f "$f" ]; then
        echo "  OK  $f ($(du -h "$f" | cut -f1))" | tee -a "$LOG"
    else
        echo "  MISSING  $f" | tee -a "$LOG"
        RC=99
    fi
done

# Final verdict.
echo "" | tee -a "$LOG"
if [ $RC -eq 0 ]; then
    echo "SMOKE TEST PASSED" | tee -a "$LOG"
else
    echo "SMOKE TEST FAILED (exit $RC)" | tee -a "$LOG"
fi
exit $RC
