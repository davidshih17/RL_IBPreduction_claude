# Training SAILIR on Perlmutter (interactive queue)

Multi-node DDP training of `pentagonbox_10x_packed` on NERSC Perlmutter using
the interactive GPU queue (up to 4 nodes × 4 GPUs = 16 GPUs, max 4 hours per
allocation). The allocation is wrapped in `tmux` so it survives SSH
disconnects.

## Files

- [allocation.sh](allocation.sh) — top-level launcher; run **inside** the salloc shell.
- [srun_task.sh](srun_task.sh) — per-rank wrapper; srun invokes it once per GPU. Not run by hand.
- [train_loop.sh](train_loop.sh) — supervisor loop; runs on a **login node** in tmux, auto-resubmits 4-hour allocations until target epochs reached. See [Unattended multi-allocation runs](#unattended-multi-allocation-runs-supervisor-loop) below.

## Run procedure

### 1. SSH to Perlmutter and start tmux on the login node

```bash
ssh perlmutter.nersc.gov         # or whatever alias you use
tmux new -s sailir
```

The tmux session lives on the login node and persists across SSH
disconnects. Everything below happens inside it.

### 2. Request the interactive allocation

```bash
salloc -N 4 -C gpu --gpus 16 -q interactive -t 04:00:00 -A m4539_g
```

Notes:
- Account ends in `_g` to access GPU nodes.
- Interactive QOS cap on Perlmutter: 4 nodes, 4 hours.
- `salloc` exits after 6 minutes if resources don't show up; just re-run.
- Once allocated, you land in an interactive shell on the head compute node.

### 3. Smoke test — *do this first*

```bash
cd /global/homes/d/dshih/m4539_d/SAILIR_phase2
SMOKE=1 bash training/nersc_perlmutter/allocation.sh
```

5 epochs on 64 train shards + 16 val shards across all 16 GPUs. Counts
are multiples of `world_size=16` — the ShardedIBPDataset drops the
remainder when partitioning shards across ranks, so any split with fewer
shards than ranks is silently zeroed out. Sized to actually exercise the
train→val boundary and post-epoch all-reduce (the spots where short smokes
have missed bugs in the past). Exercises:
multi-node DDP rendezvous (`MASTER_ADDR`/`MASTER_PORT`), NCCL collectives,
sharded streaming loader, validation pass, rank-0 checkpoint save, and
epoch repeatability. Output goes to `checkpoints/pentagonbox_10x_smoke/`
and `logs/pentagonbox_10x_smoke_<timestamp>.log`.

### 4. Full training run

```bash
bash training/nersc_perlmutter/allocation.sh
```

Defaults:
- 20 epochs, all 1000 train shards, 50 val shards
- batch_size 128, lr 4e-4, num_workers 4
- `--checkpoint_every 1` (save every epoch)
- `--auto_resume` — a follow-up allocation picks up from the latest checkpoint

Output: `checkpoints/pentagonbox_10x/`, `logs/pentagonbox_10x_full_<timestamp>.log`.

Wall-time estimate (extrapolated from `run_smoke_test_ddp.sh` numbers):
~20 min/epoch on 16 GPUs → ~10 epochs per 4-hour allocation. Re-salloc and
re-run the same command to continue past 4 hours; auto_resume handles state.

### 5. Detach / re-attach

**To detach:** press `Ctrl-B`, release, then press `d`. Two separate key
presses — `Ctrl-B` is tmux's prefix, `d` is the detach command. You'll see
`[detached (from session sailir)]` and land back in your normal SSH shell.

You can now `exit` SSH, close the laptop, etc. — training keeps running
because:

- The tmux session lives on the **login node**, independent of SSH.
- The salloc shell is a child of tmux, so it's still alive.
- The actual training processes live on the **compute nodes** — they don't
  care what happens to login-node sessions at all.

**To re-attach later:**

```bash
ssh perlmutter.nersc.gov
tmux a -t sailir
```

**Other tmux commands:**

- List sessions: `tmux ls`
- Kill the session (after training is done): `tmux kill-session -t sailir`

### 6. Watch progress from outside the tmux session

From any login-node shell:

```bash
tail -f /global/homes/d/dshih/m4539_d/SAILIR_phase2/logs/pentagonbox_10x_full_*.log
```

Or check the allocation:

```bash
squeue --me
```

## Env switches on the launcher

- `SMOKE=1` — 5 epochs / 64 train shards / 16 val shards. Forces `OUTPUT_DIR=checkpoints/pentagonbox_10x_smoke`; ignores the other switches below. One-shot sanity check; **does not** auto-resume — don't combine with the supervisor loop.
- `EPOCHS=N` — override epoch count (default 20).
- `BATCH_SIZE=N` — per-rank batch size (default 128; effective batch = `BATCH_SIZE × world_size`, so 16 GPUs × 128 = 2048). Smoke-test envelope: 128 default, 192 verified-OK, 256 OOMs. If you raise it, scale `--lr` accordingly (the launcher hard-codes `--lr 4e-4`).
- `MAX_TRAIN_SHARDS=N` — cap train shards (default: all 1000). Full mode only. For small end-to-end tests of auto_resume + the supervisor without a 4-hour allocation.
- `N_VAL_SHARDS=N` — override val shard count (default 50). Full mode only.
- `OUTPUT_DIR=path` — checkpoint output dir (default `checkpoints/pentagonbox_10x`). Full mode only. Each split needs `n_shards ≥ world_size` after partitioning.

## Resuming after a 4-hour allocation ends

Just re-`salloc` and re-run `bash training/nersc_perlmutter/allocation.sh`.
`--auto_resume` is on by default, so it picks up from the last checkpoint in
`checkpoints/pentagonbox_10x/`.

## Unattended multi-allocation runs (supervisor loop)

[train_loop.sh](train_loop.sh) automates the re-salloc
cycle: it runs on a login node in tmux, requests one 4-hour interactive
allocation at a time, launches `allocation.sh` inside it with
`--auto_resume`, and repeats until `last.pt['epoch'] >= TARGET_EPOCHS`.

### Launch it

```bash
ssh perlmutter.nersc.gov
cd /global/homes/d/dshih/m4539_d/SAILIR_phase2
tmux new -s sailir-loop
bash training/nersc_perlmutter/train_loop.sh
# Ctrl-B D to detach.
```

Run it on a **login node**, *not* inside an existing `salloc` — the supervisor
itself calls `salloc`. It mostly sleeps, so it doesn't violate NERSC's
login-node policy.

### How it decides whether to keep going

After each allocation exits, the supervisor reads `last.pt['epoch']` again:

- **Progress made, target not reached** → request the next allocation.
- **Target reached** → exit 0.
- **No progress AND non-zero exit** → assume hard error, exit 1. Investigate
  `logs/pentagonbox_10x_full_*.log` before relaunching.
- **`MAX_ALLOCATIONS` reached** → exit 2 (safety brake).

### Env switches on the supervisor

- `TARGET_EPOCHS=N` — stop once `last.pt['epoch']` ≥ N (default 20).
- `MAX_ALLOCATIONS=N` — cap on allocations per run (default 6, ≈ 24h wall).
- `SLEEP_BETWEEN=N` — seconds between allocations (default 30).
- `WALLTIME=HH:MM:SS` — per-allocation walltime (default `04:00:00`, the
  interactive-queue cap). Shorten for testing the supervisor itself.
- `NODES=N` / `GPUS=N` — allocation size (defaults 4 nodes / 16 GPUs).
  Smaller allocations usually start faster.
- `OUTPUT_DIR=path` — checkpoint dir whose `last.pt` is polled (default
  `checkpoints/pentagonbox_10x`). Override when testing against a non-default
  run (e.g. the smoke output dir).
- All launcher env switches (`BATCH_SIZE`, `SMOKE`, etc.) inherit through
  `salloc` — set them in the supervisor's env:
  `BATCH_SIZE=192 bash train_loop.sh`.

### Testing the supervisor itself

To verify the loop logic without burning a full 4-hour interactive allocation,
do a **small full-mode run**: short walltime, single node, fewer shards,
isolated output dir. Don't use `SMOKE=1` — it doesn't auto-resume, so the
supervisor would never see progress accumulate across iterations.

```bash
WALLTIME=00:05:00 NODES=1 GPUS=4 \
  TARGET_EPOCHS=100 MAX_ALLOCATIONS=3 \
  MAX_TRAIN_SHARDS=50 N_VAL_SHARDS=10 \
  OUTPUT_DIR=checkpoints/pentagonbox_10x_loop_test \
  bash training/nersc_perlmutter/train_loop.sh
```

What this exercises:

- **Multi-iteration loop**: 100 target epochs can't be reached in one 5-min
  allocation, so SLURM will kill each one at `WALLTIME` and the supervisor
  will request the next. With `MAX_ALLOCATIONS=3` you'll get up to three.
- **auto_resume contract**: each iteration picks up at `last.pt['epoch']+1`,
  written to the test `OUTPUT_DIR` (the real run's checkpoints are untouched).
- **Env passthrough**: `MAX_TRAIN_SHARDS` and `N_VAL_SHARDS` propagate from
  the supervisor's env, through `salloc`, to the launcher.
- **Safety brake**: `MAX_ALLOCATIONS=3` bounds the test wall clock.

Each split needs `n_shards ≥ world_size` after partitioning. With `GPUS=4`,
the values above give ~12 train shards/rank and ~2 val shards/rank — safely
above the world_size=4 floor.

### Supervisor log

The loop's own messages go to `logs/supervisor_<timestamp>.log` (per-allocation
training logs are still written by the launcher to
`logs/pentagonbox_10x_full_<timestamp>.log`).

## Troubleshooting

- **`salloc: error: ... Invalid account`** — verify with `sacctmgr show user $USER -p`. The GPU account should be `m4539_g` (with the `_g` suffix).
- **Hangs at NCCL init** — check that `MASTER_ADDR` resolves: `getent hosts $MASTER_ADDR`. Should be the first node in `$SLURM_JOB_NODELIST`.
- **`module: command not found`** — `module load pytorch/2.11.0` requires the NERSC modules environment, which is auto-loaded in interactive shells. If running from a non-interactive shell, source `/etc/profile` first.
- **Manifest rebuild every run** — first run builds `data/pentagonbox_10x_packed/manifest.json` (one-time, ~2–3 min). Don't delete it.
