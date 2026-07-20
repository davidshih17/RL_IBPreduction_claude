# master-integrals — deriving the COMPLETE master basis of a family

SAILIR component alongside `data-gen/`, training and `reduction/`. This is
**how we derive the master integral basis in general** for any topology,
before training data is generated and before any production reduction runs.

## Why this exists (the gravity3L failure, 2026-07)

Every master list we had inherited was **target-scoped**: Kira/FIRE report the
masters *of the system selected by the given targets*, not of the family.
The FIRE-68 table basis for gravity3L lacked (among others) the sector-635
masters; SAILIR cascades lawfully wander into every sector of the cone, so
workers ground 18+ hours on integrals that *are* masters but weren't in the
list. A full-cone Kira run to fix this properly is prohibitive (the
sector-1023 top-cone system at (12,2) exceeded 73 GB — and that is already
fully numeric; the cost is the ~10^6-equation system size, not arithmetic).

## The method: per-sector maximal-cut systems

For each sector S:

> (any S-sector integral) = (sum over S-sector masters) + (subsector terms)

On the maximal cut of S every subsector integral vanishes. So the masters of
S are computable from a **small in-sector system**:

1. Seed IBP+LI identities at in-sector seeds (all denominators of S at
   index >= 1, up to r_max dots and s_max numerator powers).
2. **THE CUT**: drop every generated term whose sector != S.
3. Gauss-eliminate in the sector-senior total order (ORDERING.md).
4. The *unreduced* integrals with r <= r_max-1 and s <= s_max-1 (interior of
   the seed box, excluding boundary artifacts) are the sector's masters.

The union over all canonical non-trivial sectors is the family basis.
Each sector costs seconds-to-minutes and runs independently → embarrassingly
parallel on Condor. For gravity3L (10 dens, 15 indices, 109 canonical
non-trivial sectors): ~5 min/sector, **142 masters**.

**Caveat (validate, don't assume):** cut counting can miss a master whose
maximal cut vanishes, and the strict `sector == S` truncation also drops
super-sector terms. Hence the mandatory checks below — the method is only
trusted per-family after they pass.

## Mandatory validation protocol

1. **Depth stability, fixed window**: every sector runs at two depths,
   (L+2, 2) and (L+3, 3); master sets compared over the SAME interior window.
   Any unstable sector is re-probed deeper (`cut_masters_one.py SEC 4`).
   (Comparing across *different* windows produces artifact "instabilities" —
   that bug cost a false alarm; always compare fixed-window.)
2. **Spot-check vs certified reductions**: for sectors where an independent
   answer exists (FIRE table, prior Kira run), the per-sector master count and
   identities must match. gravity3L: 10/10 sectors validated vs FIRE.
3. **Top-sector cross-check vs Kira**: a scoped Kira run on the top sector is
   affordable and independent. gravity3L: Kira(12,2) top sector = 6 masters =
   cut result, exactly. (Kira's *subsector* lists from a shallow-s run are
   NOT converged — do not use them to veto the cut result; re-probe the cut
   deeper instead.)
4. **canonical_masters gate**: after installing, `reduction/canonical_masters.py`
   must print ALL PASS (all masters in canonical sectors, relabels/merges
   consistent).
5. **Closure gate in production**: final reduction answers of known targets
   must carry net-zero coefficients on masters outside the certified subset
   (`reduction/cmp_gr_vs_oracle.py` pattern).

## Recipe for a new topology

Prerequisites: `topology_input/<fam>/` with IBP/LI templates
(HOW_TO_DERIVE_FROM_KIRA.md), canonical sectors pkl + entry in
`reduction/topo_config.py`, symmetry maps (`reduction/sector_canon_maps.py`).
Everything fully numeric, prime 1009 — framework convention.

```bash
cd SAILIR_phase2
export SAILIR_TOPOLOGY=<fam> SAILIR_SECTOR_RANK=1

# 1. generate the sweep (one Condor job per canonical non-trivial sector)
python master-integrals/make_sweep.py --outdir results/<fam>_cut_sweep
condor_submit results/<fam>_cut_sweep/sweep.sub

# 2. wait; re-run any missing/unstable sectors (deeper: worker args "SEC 4")

# 3. collect + install + gate (refuses on missing/unstable; backs up old file)
python master-integrals/collect_masters.py \
    --sweep-dir results/<fam>_cut_sweep --install
```

Then rebuild the terminal set / masters dict the reduction stack consumes and
proceed to data-gen / reduction.

## Files

| file                | role                                                        |
|---------------------|-------------------------------------------------------------|
| `cut_masters.py`    | engine: `CutMasters.sector_masters(S, rmax, smax)` (seeds → cut → eliminate → interior survivors) |
| `cut_masters_one.py`| one-sector Condor worker (`SEC [DEPTH]`), writes `S<sec>[_d<D>].pkl` |
| `make_sweep.py`     | writes `sectors.txt` / `worker.sh` / `sweep.sub` for a topology |
| `collect_masters.py`| stability check + union + `--install` (backup, family format, canonical_masters gate) |

## gravity3L record (2026-07-20)

- 109 canonical non-trivial sectors, sweep on Condor (8 GB/job), all stable
  at fixed window; **142 masters** installed as
  `topology_input/gravity3L/masters` (backups: `masters.fire68.bak` = the
  target-scoped FIRE table basis, `masters.sym64.bak` = the target-scoped
  Kira sym run).
- Sector 635 = {corner C, dotted T} — the two integrals the 18h-stuck
  workers were (correctly) unable to reduce.
- Validation: 10/10 FIRE sector checks; top sector 6 = 6 vs Kira(12,2);
  canonical_masters gate ALL PASS (142 → 142, 0 relabeled, 0 merged).
