#!/bin/bash
# Multi-node DDP training of pentagonbox_10x_packed on Perlmutter.
#
# Run inside an interactive salloc on 4 GPU nodes, wrapped in tmux so the
# allocation survives SSH disconnect:
#
#   tmux new -s sailir
#   salloc -N 4 -C gpu --gpus 16 -q interactive -t 04:00:00 -A m4539_g
#   bash scripts/train/perlmutter_pentagonbox_10x.sh
#   # Ctrl-B D to detach; `tmux a -t sailir` to re-attach later.
#
# Env switches:
#   SMOKE=1            — 5 epochs, 64 train shards, 16 val shards. All
#                        other env switches below are ignored in this mode;
#                        OUTPUT_DIR is forced to checkpoints/pentagonbox_10x_smoke.
#                        Multiples of world_size=16; otherwise the dataset's
#                        "drop remainder to evenly partition across ranks"
#                        logic silently zeros out splits with fewer shards
#                        than ranks. Exercises the train→val boundary and
#                        post-epoch all-reduce.
#   EPOCHS=N           — override epoch count (default 20).
#   BATCH_SIZE=N       — per-rank batch size (default 128). Effective batch
#                        is BATCH_SIZE × world_size. If you change this for
#                        the full run, scale --lr accordingly (linear or
#                        sqrt rule).
#   MAX_TRAIN_SHARDS=N — cap train shards (default: all). Useful for small
#                        end-to-end tests of the auto_resume + supervisor
#                        path without a 4-hour allocation.
#   N_VAL_SHARDS=N     — override val shard count (default 50).
#   OUTPUT_DIR=path    — checkpoint output dir (default checkpoints/pentagonbox_10x).
#                        Override to test against a non-default run dir.
#                        Must agree with the supervisor's OUTPUT_DIR when
#                        running under perlmutter_train_loop.sh.
#                        Each split needs n_shards >= world_size.
#   MODEL_VARIANT=name — which classifier class to train (default `full`,
#                        i.e. IBPActionClassifier). `nosubs` selects
#                        IBPActionClassifierNoSubs (subs encoder removed,
#                        ~40% fewer params). Checkpoints from different
#                        variants are NOT interchangeable; use a distinct
#                        OUTPUT_DIR per variant.

set -euo pipefail
cd "$(dirname "$0")/../.."

module load pytorch/2.11.0

if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
    echo "ERROR: not inside a SLURM allocation. Run salloc first." >&2
    exit 1
fi

# Rendezvous: first node in the allocation acts as master.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

SMOKE=${SMOKE:-0}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
MODEL_VARIANT=${MODEL_VARIANT:-full}
NUM_WORKERS=${NUM_WORKERS:-4}

if [[ "$SMOKE" == "1" ]]; then
    OUTPUT_DIR=checkpoints/pentagonbox_10x_smoke
    LOG_TAG=smoke
    EXTRA_ARGS=( --max_train_shards 64 --n_val_shards 16 --epochs 5 )
else
    OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/pentagonbox_10x}
    LOG_TAG=full
    N_VAL_SHARDS=${N_VAL_SHARDS:-50}
    EXTRA_ARGS=( --n_val_shards "$N_VAL_SHARDS" --epochs "$EPOCHS" --auto_resume )
    [[ -n "${MAX_TRAIN_SHARDS:-}" ]] && EXTRA_ARGS+=( --max_train_shards "$MAX_TRAIN_SHARDS" )
fi
EXTRA_ARGS+=( --model_variant "$MODEL_VARIANT" )

mkdir -p "$OUTPUT_DIR" logs
LOG=logs/pentagonbox_10x_${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log

{
  echo "[$(date -Iseconds)] launch SMOKE=$SMOKE EPOCHS=$EPOCHS"
  echo "  SLURM_JOB_ID=$SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES"
  echo "  MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
  echo "  OUTPUT_DIR=$OUTPUT_DIR"
} | tee -a "$LOG"

srun -l -u \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    --cpus-per-task=32 \
    --gpu-bind=none \
    bash scripts/train/perlmutter_srun_task.sh \
        --topology         topology_input/pentagonbox \
        --shards_dir       data/pentagonbox_10x_packed \
        --buffer_shards    4 \
        --output_dir       "$OUTPUT_DIR" \
        --batch_size       "$BATCH_SIZE" \
        --lr               4e-4 \
        --prime            1009 \
        --device           cuda \
        --num_workers      "$NUM_WORKERS" \
        --checkpoint_every 1 \
        --log_every        50 \
        --seed             0 \
        "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG"
