#!/bin/bash
# DDP smoke test: 50 train shards + 9 val shards, 5 epochs (~30 min).
#
# Sized to actually exercise the failure mode that 9/3-shard smoke missed:
# scale-dependent NCCL desync at epoch boundaries shows up only after
# thousands of iterations / dozens of minutes of wall time.
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

# Detached via nohup so the job survives terminal/session disconnect.
# Use `tail -f $LOG` to watch. PID written to a file for easy kill if needed.
nohup torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    training/train_classifier.py \
    --topology       topology_input/pentagonbox \
    --shards_dir     data/pentagonbox_10x_packed \
    --max_train_shards 50 \
    --n_val_shards   9 \
    --buffer_shards  4 \
    --output_dir     smoke_checkpoints \
    --epochs         5 \
    --batch_size     64 \
    --lr             4e-4 \
    --prime          1009 \
    --device         cuda \
    --num_workers    2 \
    --checkpoint_every 1 \
    --log_every      20 \
    --seed           0 \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > logs/smoke_test_ddp.pid
echo "Started PID=$PID. Tail with: tail -f $LOG" | tee -a "$LOG"
echo "Kill with: kill \$(cat logs/smoke_test_ddp.pid)" | tee -a "$LOG"
