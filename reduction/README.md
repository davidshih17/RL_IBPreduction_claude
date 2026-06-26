# SAILIR — Reduction phase (async hierarchical orchestrator)

This directory is the **reduction phase** of SAILIR: it reduces a Feynman
integral to a basis of master integrals using a trained model to navigate the
integration-by-parts (IBP) + Lorentz-invariance (LI) search, parallelised across
a Condor cluster.

It is the production entry point. The other phases live in `../data_gen/`
(generate training data) and `../train/` (train the model); the shared core
library used by all three is the `../sailir/` package (the model
`IBPActionClassifier`, the IBP environment, the topology, and the GF(p) kernels).

---

## 1. What the orchestrator does

`hierarchical_reduction.py` reduces a single "top" integral by recursion:

1. It replays the start integral through whatever reductions it already knows
   (its in-memory **cache**), yielding a linear combination of integrals over
   the finite field `GF(prime)`.
2. Every term that is **not yet a master** is dispatched as a **one-step Condor
   worker** (`onestep_worker_v7.py`) that runs the beam-search engine
   (`beam_search_v7.py`) to reduce that integral by one weight level.
3. Completed workers are folded back into the cache (memoised), the expression
   is re-expanded, and any new non-masters are dispatched. This repeats until
   **0 non-masters remain** — the start integral is then fully written in the
   master basis.

The orchestrator itself is just a poller (submit + collect). It must run
**interactively on the submit node** (it submits Condor jobs); do **not** submit
the orchestrator itself to Condor.

Files in this directory:

| File | Role |
|------|------|
| `hierarchical_reduction.py` | the orchestrator (entry point) |
| `onestep_worker_v7.py`      | per-integral Condor worker (one weight level) |
| `beam_search_v7.py`         | the beam-search reduction engine |
| `beam_search_utils.py`      | shared helpers (sector mask, non-masters, …) |
| `save_replay_state.py`      | rebuild the reduced expression from a run's results |
| `print_replay_terms.py`     | classify/print the final master expression |
| `run_reduction.sh`          | **canonical launcher** — recommended config (§5), parameterized by integral |
| `launch_hier_85_v7_fresh.sh`| exact reproducer of the v7_fresh **round-1** run (`pentagonbox` + `--no-paper-masters-only`) |

---

## 2. Prerequisites (this cluster)

- **Python** (with torch):
  `/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python`
- **The compiled `sailir` Cython kernels.** Build them once per cluster /
  Python version with:
  ```bash
  python sailir/_setup_cython.py build_ext --inplace
  ```
  This produces `sailir/_*.cpython-<ver>-<arch>.so` (the kernels live inside the
  `sailir/` package). They are ABI-specific and gitignored — not portable, so
  rebuild on a new machine. If absent, the code falls back to pure Python (much
  slower) but still runs.
- **A Condor cluster** reachable via `condor_submit` / `condor_q` (the workers
  run as Condor jobs). The orchestrator runs interactively on the submit node;
  the per-integral `--work-dir` must be on a **shared filesystem** (under
  `results/` on `/het`), not node-local `/tmp`.
- **The trained model**:
  `checkpoints/pentagonbox_10x_loop_100/best_model.pt`
- **A topology** under `topology_input/` (see §4).

Paths below assume the repo lives at
`/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2` (referred to as `$BASE`).
The reduction scripts resolve `sailir` and their siblings relative to their own
location, so the repo can be moved; the model/topology paths you pass on the CLI
are yours to set.

---

## 3. The integral format

An integral is **11 comma-separated integers** — the propagator/ISP powers of
the pentagon-box (TA) family:

```
 a0  a1  a2  a3  a4  a5  a6  a7   a8  a9  a10
 └────── 8 propagators (denominators) ──────┘  └─ 3 ISPs (numerators) ─┘
```

- Positions **0–7** are the 8 physical propagators (`k1`, `k1+p1`, `k1+p1+p2`,
  `k1+p1+p2+p3`, `k2`, `k2+p1+p2+p3`, `k2+p1+p2+p3+p4`, `k1−k2`).
- Positions **8–10** are the 3 irreducible scalar products (always ≤ 0).
- A **positive** index = propagator power (dots when > 1); a **negative** index
  = numerator (ISP) power.

Example: `1,1,1,1,1,1,1,1,-5,0,0` is the top sector with an ISP to the 5th power
(the "(8,5)" integral).

---

## 3a. Kinematics (the finite-field point) — RECORD

SAILIR reduces over the finite field `GF(prime)` at a **single fixed numeric
point**: the spacetime dimension `d` and the five Mandelstam invariants are
assigned specific integers. These are **deterministic, not random** —
`Topology.from_dir` assigns the first N primes from
`[31, 47, 53, 59, 61, 67, 71, 73, 79, 83]` to the N invariants in order, with
`d = 41`. For the pentagon-box (TA) family the point is:

| symbol | value |   | symbol | value |
|--------|------:|---|--------|------:|
| `d`    | **41** | | `s34` | **53** |
| `s12`  | **31** | | `s45` | **59** |
| `s23`  | **47** | | `s51` | **61** |
| prime (`GF`) | **1009** | | | |

**The same point is used for all three phases** — data-gen reads
`topology.kinematics_values`, training learns on data generated there, and
reduction runs there — so the model and the reductions are consistent. Any
reduction result (the master coefficients in `final_expr`) is therefore valid at
exactly this phase-space point mod 1009. Reproduce/inspect the values with
`python reduction/_print_kinematics.py topology_input/pentagonbox`.

---

## 4. Choosing the master basis (important)

SAILIR's search uses **IBP + LI only — not symmetry relations**. Two topologies
ship with the repo:

- **`topology_input/pentagonbox`** — Kira's **61**-master basis (derived *with*
  sector symmetries). A handful of sub-sectors (e.g. sector 161,
  `1,0,0,0,0,1,0,1,0,0,0`) reduce to a master *only* via the k1↔k2 loop-exchange
  symmetry. Under `--paper-masters-only` against this basis, the orchestrator
  **stalls** on those corners (a worker spins forever — they are irreducible by
  IBP+LI alone).

- **`topology_input/pentagonbox_nosym`** — the **62**-master IBP+LI-only basis
  (the 61 + the symmetry-redundant corners, derived by running Kira with
  symmetries disabled). With this basis `--paper-masters-only` **terminates
  cleanly**. ← **recommended.**

If you instead use `--no-paper-masters-only` with `pentagonbox`, the run also
terminates, but the basis is overcomplete (it keeps every uncovered-sector
corner as a "master"), so the answer is not unique. For a unique, terminating
reduction, use **`pentagonbox_nosym` + `--paper-masters-only`**.

---

## 5. Quick start — reduce a new integral

```bash
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2

INTEGRAL="2,1,1,1,1,1,1,1,0,0,0"          # <-- your 11-int target
OUT=$BASE/results/my_reduction            # <-- output directory (will be created)
mkdir -p $OUT/work

PYTHONUNBUFFERED=1 $PY -u $BASE/reduction/hierarchical_reduction.py \
    --topology   $BASE/topology_input/pentagonbox_nosym \
    --integral=$INTEGRAL \
    --output     $OUT/reduction.pkl \
    --work-dir   $OUT/work \
    --model-checkpoint $BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only \
    --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --max-concurrent 1000 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --check-interval 5 --resume \
  > $OUT/hierarchical.log 2>&1 &
```

This is the **recommended configuration**: the production `pentagonbox_8_5_v7_fresh`
worker settings (1-CPU workers fanned out up to 1000-wide, straggler escalation
disabled, `--resume` for crash recovery) **but with the 62-master
`pentagonbox_nosym` topology + `--paper-masters-only`**, so the reduction
terminates at the *unique* IBP+LI master basis (no path-dependent corners). The
only deviation from the v7_fresh round-1 run is exactly those two flags — round-1
used plain `pentagonbox` + `--no-paper-masters-only` (which completes but leaves
an overcomplete, path-dependent corner basis; see §4). Watch with
`tail -f $OUT/hierarchical.log` and `condor_q`; it's done when the log prints
`ASYNC REDUCTION COMPLETE` / `SUCCESS! All integrals reduced to masters`.

`run_reduction.sh` is exactly this command, parameterized by integral — copy it
and set `INTEGRAL` / `OUT`.

---

## 6. Key options

Run `python reduction/hierarchical_reduction.py --help` for the full list. The
ones that matter:

| Flag | Meaning / recommended value |
|------|------------------------------|
| `--topology DIR` | topology dir (use `pentagonbox_nosym`, §4) |
| `--integral I` | the 11-int start integral |
| `--output FILE` | final result pickle (`final_expr` + `cache`) |
| `--work-dir DIR` | per-worker results + logs land in `DIR/results`, `DIR/logs` |
| `--model-checkpoint FILE` | the trained model (`best_model.pt`) |
| `--beam_width 40` | beam width (paper recipe) |
| `--prime 1009` | finite-field prime (must match the model's training) |
| `--paper-masters-only` | reduce to the topology's masters (default ON). §4 |
| `--use-v7-worker` | use the current engine (recommended) |
| `--v7-cpus 1` | **default 1** = cheap serial workers, `request_cpus=1`, no fork pool, single torch thread, flat 4 GB, many concurrent (best for fanning out — the production setting). `8` = an 8-way fork pool per worker (`request_cpus=10`, ~22 GB peak) for a few genuinely heavy long-runners. |
| `--worker-memory-gb 4` | per-worker memory request (cluster has soft limits) |
| `--max-concurrent 1000` | cap on simultaneous Condor workers |
| `--straggler-timeout / --straggler2-timeout` | set huge (`1000000000`) to keep every worker at `--v7-cpus`; otherwise slow workers are escalated to more CPUs |
| `--check-interval 5` | seconds between orchestrator polls |
| `--resume` | on restart, re-scan `work-dir/results` and continue (crash recovery) |
| `--resume-from DIR` | start from a **prior** run's cache (read-only); see §8 |
| `--reduce-only FILE` | restrict the round to a list of target integrals |
| `--dry-run` | write the submit files but don't submit |

---

## 7. Output & monitoring

- **`--output` pickle** (`reduction.pkl`) — a dict with:
  - `final_expr`: `{integral_tuple: coeff}` — the start integral in the master
    basis (mod `prime`). 0 non-master terms on success.
  - `cache`: every one-step reduction discovered (memo of the whole run).
  - `start_integral`, `prime`, plus timing/memory stats.
- **`work-dir/results/*.pkl`** — one pickle per reduced integral.
- **`work-dir/logs/*.out`** — per-worker beam-search logs.
- **Master log** (the file you redirected to) — one `[Iter N]` line per poll:
  ```
  [Iter 142] 156 masters, 6760 non-masters | frontier=(L=7,r=8,s=3) x 1 | Pending: 1000 | Cache: 8868 | Hits: 17459
  ```
  `masters`/`non-masters` = live status of the start reduction; `Pending` =
  in-flight workers; `Cache`/`Hits` = memoisation. Also use `condor_q`.

---

## 8. Interpreting / replaying the result

The orchestrator already prints the final master list and writes `final_expr`.
To re-derive or classify it independently from a finished run's `work-dir`:

```bash
# 1. Replay the start integral through the run's results -> replay_state.pkl
$PY $BASE/reduction/save_replay_state.py  $OUT  "$INTEGRAL"  --out $OUT/replay_state.pkl
# (pass several run dirs to chain rounds: save_replay_state.py R1 R2 R3 "$INTEGRAL")

# 2. Classify every surviving term as PAPER / CORNER master or NON-master
$PY $BASE/reduction/print_replay_terms.py  $OUT/replay_state.pkl
```

`print_replay_terms.py` ends with e.g.
`TOTAL 262 terms = 61 PAPER + 201 CORNER masters + 0 NON-masters`.

---

## 9. Resuming and multi-round reductions

- **Crash recovery:** rerun the *same* command (it already has `--resume`); it
  re-scans `work-dir/results` and continues. First stop the old orchestrator and
  `condor_rm` its workers.
- **Start from a prior run** (e.g. push a finished reduction further, or to a
  different basis): `--resume-from PRIOR_DIR` loads that run's cache read-only
  and writes new results to `--work-dir`. Combine with a different `--topology`
  (e.g. resume a `pentagonbox` run under `pentagonbox_nosym`) or with
  `--reduce-only FILE` to re-reduce a specific subset.

---

## 10. Troubleshooting

- **A worker runs forever / `Pending: 1` never drops** on a low corner integral:
  that corner reduces only via a symmetry SAILIR doesn't have (§4). Use
  `topology_input/pentagonbox_nosym` (so the corner is a master), or
  `--no-paper-masters-only`.
- **Workers get held on Condor** (CPU/memory): check `--v7-cpus` vs
  `request_cpus` and `--worker-memory-gb`; `--v7-cpus 1` requests 1 CPU / 4 GB.
- **`ModuleNotFoundError: sailir`**: run with the conda Python above and from a
  repo where `../sailir/` exists next to `reduction/`.

> Legacy/experimental scripts (other worker engines, profiling, probes, the
> Kira-symmetry tooling, one-off analyses) were moved to the top-level
> `archive/` and are no longer maintained.
