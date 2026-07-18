# SAILIR — Reduction phase (async hierarchical orchestrator)

This directory is the **reduction phase** of SAILIR: it reduces a Feynman
integral to a basis of master integrals using a trained model to navigate the
integration-by-parts (IBP) + Lorentz-invariance (LI) search, parallelised across
a Condor cluster.

The pipeline is **topology-general**: every family-specific structure (index
count, sector masks, symmetry transforms, canonical sectors, master basis) is
derived from the standard inputs under `topology_input/<family>/` and selected
at run time with `SAILIR_TOPOLOGY=<family>`. Two families ship configured:
`pentagonbox` (2 loops, 11 indices) and `gravity3L` (3 loops, 15 indices).
**The symmetry-enhanced pipeline (§4b) is the production path**; the plain
IBP+LI path (§4/§5) remains for the legacy pentagonbox baselines.

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
- **The trained model** (per topology; the `canon10x_nosubs` models are the
  ones trained on the canonical-sector datasets the symmetry-enhanced pipeline
  assumes):
  - pentagonbox: `checkpoints/pentagonbox_canon10x_nosubs/best_model.pt`
    (legacy IBP-only baselines used `checkpoints/pentagonbox_10x_loop_100/`)
  - gravity3L: `checkpoints/gravity3L_canon10x_nosubs/best_model.pt`
- **A topology** under `topology_input/` (see §4/§4b).

Paths below assume the repo lives at
`/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2` (referred to as `$BASE`).
The reduction scripts resolve `sailir` and their siblings relative to their own
location, so the repo can be moved; the model/topology paths you pass on the CLI
are yours to set.

---

## 3. The integral format

An integral is **N comma-separated integers** — the propagator/ISP powers of
the family, N = the number of propagators in
`topology_input/<family>/integralfamilies.yaml`:

- **pentagonbox**: 11 ints = 8 denominators + 3 ISPs
  (`1,1,1,1,1,1,1,1,-5,0,0` is the top sector with an ISP to the 5th power —
  the "(8,5)" integral).
- **gravity3L**: 15 ints = 10 denominators + 5 ISPs
  (e.g. `1,1,1,1,1,1,1,1,1,1,-2,0,0,0,-2` from the FIRE benchmark list).

A **positive** index = propagator power (dots when > 1); a **negative** index
= numerator (ISP) power. Denominator slots always come first (the sector mask
is the bit pattern of the positive denominator slots).

---

## 3a. Kinematics (the finite-field point) — RECORD

SAILIR reduces over the finite field `GF(prime)` at a **single fixed numeric
point**: the spacetime dimension `d` and the kinematic invariants are assigned
specific integers. These are **deterministic, not random** —
`Topology.from_dir` assigns the first N primes from
`[31, 47, 53, 59, 61, 67, 71, 73, 79, 83]` to the N invariants in order, with
`d = 41`:

- **pentagonbox** (5 invariants): `s12=31, s23=47, s34=53, s45=59, s51=61`
- **gravity3L** (1 invariant): `y=31`
- prime (`GF`) = **1009** for both (the models are trained at this prime; the
  symmetry stores are evaluated at this point).

**The same point is used for all three phases** — data-gen reads
`topology.kinematics_values`, training learns on data generated there, and
reduction runs there — so the model and the reductions are consistent. Any
reduction result (the master coefficients in `final_expr`) is therefore valid at
exactly this phase-space point mod 1009. Reproduce/inspect the values with
`python reduction/_print_kinematics.py topology_input/pentagonbox`.

---

## 4. Choosing the master basis (important)

The one-step **workers** use IBP + LI only — symmetry relations live in the
**orchestrator's routing layer** (§4b). That leaves two consistent ways to run:

- **Symmetry-enhanced (production, §4b):** the paper basis (Kira's for
  pentagonbox, FIRE's for gravity3L) is transformed into OUR canonical sectors
  (`canonical_masters.py`), so `--paper-masters-only` terminates cleanly and
  the final expression is translated back to the paper basis exactly at output
  time.

- **Legacy IBP+LI-only (pentagonbox baselines only):** without the routing
  layer, sub-sectors that reduce only via a symmetry stall under
  `--paper-masters-only` against Kira's 61-master `pentagonbox` basis (the
  m1/m3 hang, e.g. sector 161 `1,0,0,0,0,1,0,1,0,0,0`). The workaround is
  **`topology_input/pentagonbox_nosym` + `--paper-masters-only`** (the
  62-master IBP+LI-only basis — terminates, unique). `--no-paper-masters-only`
  also terminates but leaves an overcomplete, path-dependent corner basis.

---

## 4b. The symmetry-enhanced general-topology pipeline (PRODUCTION)

Design: `ORDERING.md` (the sector-senior total order) + the red DECISION
paragraphs in `../notes/symmetry_inference_routing.tex`. In one paragraph: all
sectors related by symmetry are merged into orbits with one **canonical
representative**; the total order ranks by **sector rank first**, then weight
`(r, s)`, then `|abs|`; the orchestrator's router (`canonical_monolithic_rule`
in `symmetry_route.py`) rewrites every integral into canonical sectors before
dispatch, so workers (which do NO symmetrization) only ever see canonical
sectors; the paper masters are transformed into canonical sectors and
translated back at output time.

**The symmetry engine is general** (`symmetry_engine2.py` + provider
`canonicalize2.py`): every transform is derived from
`integralfamilies.yaml` / `kinematics.yaml` / the Kira `sectormappings` files,
verified numerically on explicit finite-field vector kinematics, and stored in
`results/<family>_transforms_v2.pkl`. External symmetries (e.g. gravity's
u1↔u2, q→−q, joint u-negation) are found automatically. One physics rule is
baked into the provider and must not be relaxed: a **denominator may only map
with coefficient exactly +1** ("clean-den") — sign-flipping maps of eikonal
(linear) denominators are NOT value identities (empirically locked against
677 FIRE-table comparisons; see `canonicalize_GR.py`'s docstring).

**Run flags (set for the orchestrator AND inherited by every worker — the
orchestrator writes them into the Condor submits):**

```bash
export SAILIR_TOPOLOGY=gravity3L      # topology key (topo_config.py)
export SAILIR_SECTOR_RANK=1           # the sector-senior total order (required)
# ... and pass --use-symmetry to hierarchical_reduction.py
```

**Quick start — symmetry-enhanced gravity3L reduction:**

```bash
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1

INTEGRAL="1,0,1,-1,1,1,1,0,1,1,-1,0,0,0,0"   # a 15-int gravity target
OUT=$BASE/results/gr_reduce_example
mkdir -p $OUT/work

PYTHONUNBUFFERED=1 $PY -u $BASE/reduction/hierarchical_reduction.py \
    --topology   $BASE/topology_input/gravity3L \
    --integral=$INTEGRAL \
    --output     $OUT/reduction.pkl \
    --work-dir   $OUT/work \
    --model-checkpoint $BASE/checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only --use-symmetry \
    --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --max-concurrent 1000 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --check-interval 5 --resume \
  > $OUT/hierarchical.log 2>&1 &
```

The same command with `SAILIR_TOPOLOGY=pentagonbox`, the pentagonbox topology
dir and the `pentagonbox_canon10x_nosubs` checkpoint runs pentagonbox
symmetry-enhanced. The 27-integral gravity benchmark list (difficulty-spread
sample of the FIRE table) is `results/gr_benchmark_targets.txt`; the exact
FIRE reductions for cross-checking are extracted by
`analysis/extract_fire_oracle.py`.

**Onboarding a NEW topology** (everything below is topology-keyed; register
the family in `topo_config.py::_CFG` first — index counts, paths, store):

```bash
export SAILIR_TOPOLOGY=<family>
# 1. inputs: topology_input/<family>/ with integralfamilies.yaml,
#    kinematics.yaml, IBP, LI, masters (the paper basis), and a Kira
#    sectormappings dir (sectorSymmetries + sectorRelations)
# 2. verified transform store (gates every transform numerically):
$PY reduction/symmetry_engine2.py topology_input/<family> \
     <sectormappings_dir> results/<family>_transforms_v2.pkl 1009
# 3. canonical sectors from the CLEAN-DEN orbits
#    (template: build_canonical_sectors_GR_v2.py — edit the two path lines)
# 4. composite canonicalization maps + canonical masters (run their gates):
SAILIR_SECTOR_RANK=1 $PY reduction/sector_canon_maps.py
SAILIR_SECTOR_RANK=1 $PY reduction/canonical_masters.py
# 5. gates: reduction/sector_rank.py (rank contract) and, where an oracle
#    exists (FIRE/Kira reduction tables), the router cross-check
#    (template: smoke_gr_router_vs_oracle.py — 0 false zeros required)
```

For gravity3L all of steps 2–5 are already built, gated
(`run_engine2_behavioral_gate.sh` — ALL PASS 2026-07-18) and committed; the
per-file gates print `ALL PASS` when run directly.

---

## 5. Quick start — legacy IBP+LI-only pentagonbox run

(For the production symmetry-enhanced path — any topology — use §4b. This
section is kept for reproducing the pre-symmetry pentagonbox baselines.)

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

## 9a. Meta-orchestrator — cascading a target LIST with cache reuse

For reducing a long list of integrals (the pentagon-box `list_TA`: 1016 targets,
996 after dropping the 20 with a positive ISP slot), driving `hierarchical_reduction`
by hand one integral at a time does not scale. `meta_orchestrator.py` industrializes
it:

- **Weight-ordered cascade.** Processes the target list top-to-bottom
  (`from_federica/list_TA_ispclean_by_weight`, heaviest first), launching one
  `hierarchical_reduction.py` run per target under `results/meta_reduce/<target>/`.
- **Cache cascade (the point).** Before each launch it merges the substitution
  caches of ALL prior runs — completed *and* in-progress — into the new run's
  `--resume-from` (via `merge_caches.py`). Every reduction stands on everything
  computed so far, so lower-weight targets get progressively cheaper.
  `count_cache_coverage.py` reports how much of the list an existing cache set
  already covers (e.g. one (8,5) reduction alone covers ~19% of list_TA).
- **Load-aware continuous submission.** A new target launches only when every
  active run has fanned down to its long-pole phase (fewer than `NM_THRESHOLD`
  remaining non-masters, held stable across polls), in bursts of up to
  `LAUNCH_BATCH` while the Condor pool has more than `FREE_CPU_FLOOR` free CPUs,
  capped at `MAX_ACTIVE` simultaneously-active orchestrators (each is a ~0.6 GB
  login-node poller). All knobs sit at the top of the file.
- **Restart-safe, launch-only.** On restart it re-discovers its own runs via each
  dir's `target.txt`; it launches and monitors but never kills anything.

```bash
# run unattended in the background; everything streams to the log
nohup python reduction/meta_orchestrator.py > logs/meta_orchestrator.log 2>&1 &

# monitor the cascade
python reduction/plot_cascade_progress.py
```

Companion tooling: `build_clean_cache.py` + `launch_cache_reuse_*.sh` (curated
cache-reuse launches), `_pick_cache_target.py` (choose the next best target for
coverage), and the `replay_list_TA_*.py` / `export_list_TA_reductions.py` family
(replay every list target through the accumulated results and export the final
reductions).

**Convention note:** the settings baked into the top of `meta_orchestrator.py`
(model checkpoint, `--no-paper-masters-only`, no symmetry flags) date from the
pre-symmetry-enhanced era. To cascade under the production stack, update its
launch block to the §4b configuration (`--use-symmetry`,
`--paper-masters-only`, the canon10x checkpoint) and export
`SAILIR_TOPOLOGY=<family> SAILIR_SECTOR_RANK=1` before launching — the
master-basis choice interacts with cache merging across targets.

---

## 10. Troubleshooting

- **A worker runs forever / `Pending: 1` never drops** on a low corner integral:
  that corner reduces only via a symmetry. On the production path this is
  solved by canonical masters (§4b) — check `SAILIR_SECTOR_RANK=1` and
  `--use-symmetry` are actually set (worker logs echo the environment). On the
  legacy path use `topology_input/pentagonbox_nosym` (§4). Also note: corners
  of SCALELESS (Kira-trivial) sectors are genuinely zero but slow for the
  model to kill — they only appear as benchmark targets, not as reduction
  debris of nonzero integrals.
- **Workers get held on Condor** (CPU/memory): check `--v7-cpus` vs
  `request_cpus` and `--worker-memory-gb`; `--v7-cpus 1` requests 1 CPU / 4 GB.
- **`ModuleNotFoundError: sailir`**: run with the conda Python above and from a
  repo where `../sailir/` exists next to `reduction/`.

---

## 11. Standalone probes (one integral, one weight level)

A **probe** is a single `beam_search_v7.py` run as **one Condor job** that reduces
**one** integral by **one** weight level (`SAILIR_SUCCESS_TOTAL=1`) — *not* the
orchestrator, so there is no chaining to masters. It's the tool for benchmarking
the engine on a known-hard integral (wall-time, step count, peak memory).

```bash
# run_probe.sh <integral> <tag> [sym:0|1] [n_workers] [max_steps] [mem_gb]
bash reduction/run_probe.sh '0,1,1,1,1,1,1,1,-4,0,0'  74_off  0      # baseline
bash reduction/run_probe.sh '0,1,1,1,1,1,1,1,-4,0,0'  74_on   1      # + greedy symmetry
```

Each call writes a Condor submit + result under `results/probe_<tag>/`. Read:
- `tail results/probe_<tag>/probe.out` — `t_total` and `SUCCESS`/path length,
- `grep 'Memory (MB)' results/probe_<tag>/probe.log` — cgroup peak memory.

Defaults match the success-only recipe (`pentagonbox` + `--no-paper-masters-only`,
beam-width 40, 8/8 fork pool, `request_cpus = n_workers+2`). The four benchmark
probes: `(7,4)=0,1,1,1,1,1,1,1,-4,0,0`, `(8,4)=-1,2,1,0,1,2,1,1,-3,0,0`,
`longrunner=1,1,1,0,0,1,3,1,-2,-1,0`, `memhog=1,1,1,0,-2,1,1,1,0,0,0`.

**Greedy sector symmetry (`sym=1`)** sets `SAILIR_SYMMETRY=1`, an inference-only
pre-reduction inside the worker. **Benchmarking only — NOT production**: the
locked design (2026-07-11, measured null result; see the `SAILIR_SYM_DROP`
banner in `beam_search_v7.py`) is that symmetry lives at the orchestrator's
routing layer exclusively and workers do no symmetrization.

---

> Legacy/experimental scripts (other worker engines, profiling, one-off analyses)
> were moved to the top-level `archive/` and are no longer maintained. The probe
> recipe above supersedes the archived `probe_*.sh` scripts (which pointed at the
> pre-reorg `scripts/eval/` paths).
