#!/bin/bash
# Per-rank task wrapper for Perlmutter multi-node DDP training.
#
# srun launches this script once per GPU (--ntasks-per-node=4 over N nodes).
# We translate SLURM's per-task env vars into the names train_classifier.py
# expects (which match torchrun's convention) and exec the trainer.
#
# Not intended to be run by hand — use perlmutter_pentagonbox_10x.sh.

set -euo pipefail

export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}

exec python -u scripts/train/train_classifier.py "$@"
