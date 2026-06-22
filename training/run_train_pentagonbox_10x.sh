#!/bin/bash
# Train SAILIR pentagonbox classifier on the 10x packed shards.
#
# - 3 GPUs via torchrun → DistributedDataParallel (one process per GPU)
# - 100 epochs, cosine LR schedule (state saved + restored on resume)
# - Sharded streaming dataset (~1.7 GB working set per rank, vs ~420 GB to
#   load everything). Each rank sees a disjoint set of shards.
# - Auto-resume from last.pt if present: rerunning this script picks up
#   from the latest epoch instead of starting over.
# - Per-epoch checkpoints: best_model.pt (best val), last.pt (every epoch,
#   atomic write), checkpoint_epoch{N}.pt every 5 epochs. Saved by rank 0 only.
#
# --batch_size below is PER-RANK (per-GPU). Effective batch = BS * world_size.
# BS=64 / 3 GPUs → effective bs=192.
#
# Run as:
#   ./training/run_train_pentagonbox_10x.sh
#
# To extend past 100 epochs: bump --epochs and re-run; --auto_resume picks up
# last.pt (scheduler state ensures cosine LR continues smoothly).

set -euo pipefail

cd /home/shih/work/SAILIR_phase2

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

mkdir -p checkpoints/pentagonbox logs

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
# Disable NCCL P2P over the SYS-grade interconnect (separate PCIe roots) —
# forces all-reduce through host shared memory, which is faster than the
# slow P2P fallback on this topology. Drop this on machines with NVLink/PXB.
export NCCL_P2P_DISABLE=1
# Tame NCCL chatter; bump to INFO if you need to debug comms.
export NCCL_DEBUG=WARN

BATCH_SIZE=64  # per-rank; world_size=3 → effective bs=192 (best throughput in our timing tests)
NPROC=3

LOG=logs/train_pentagonbox_10x.log
METRICS=logs/train_pentagonbox_10x_metrics.tsv

echo "Launched: $(date -Iseconds)" | tee -a "$LOG"
echo "Command: $0 $*" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC=$NPROC" | tee -a "$LOG"

nohup torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    training/train_classifier.py \
    --topology       topology_input/pentagonbox \
    --shards_dir     data/pentagonbox_10x_packed \
    --n_val_shards   50 \
    --buffer_shards  4 \
    --output_dir     checkpoints/pentagonbox \
    --log_file       "$METRICS" \
    --epochs         100 \
    --batch_size     "$BATCH_SIZE" \
    --lr             4e-4 \
    --prime          1009 \
    --device         cuda \
    --num_workers    4 \
    --checkpoint_every 5 \
    --log_every       100 \
    --auto_resume \
    >> "$LOG" 2>&1 &

PID=$!
echo "Started PID=$PID. Tail with: tail -f $LOG"
echo "$PID" > logs/train_pentagonbox_10x.pid
