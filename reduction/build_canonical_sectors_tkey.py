#!/usr/bin/env python
"""Build the CLEAN-orbit canonical sector map in the workers' _target_key order.

For each sector (bitmask over the 8 propagator slots), its canonical representative
is the sector of the SURVIVOR corner of the sector's clean-permutation orbit --
the _target_key-MAXIMUM, i.e. the orbit member every other member reduces TOWARD
(canonical_rep on the corner; see canonical_rep.py for the max-not-min invariant).
Sectors related by a genuine loop relabeling share a rep; affine-only relations
(corner -> combination) do NOT merge here (that is the inference routing's job,
not canonicalization).

Output: results/canonical_sectors_tkey.pkl = {
  'rep_of'      : {mask: canonical_mask},      # every 1..255 sector
  'canonical'   : sorted list of canonical sector masks,
  'order'       : '_target_key = (-r,-s,|abs|)',
}
This is the CLEAN, _target_key-consistent analog of the old min-integer
canonical_sectors.pkl (which is kept for the --canon path). Used to restrict the
symmetry-enhanced data-gen to the canonical sector set (--restrict-sectors).
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(ROOT, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from canonical_rep import canonical_rep

NPROP = 8


def sec_of(t):
    n = 0
    for k in range(NPROP):
        if t[k] > 0:
            n |= (1 << k)
    return n


def corner(mask):
    return tuple(1 if (mask >> k) & 1 else 0 for k in range(11))


rep_of = {}
for mask in range(1, 1 << NPROP):
    # Use canonical_rep as the SINGLE source of truth for "which orbit member survives"
    # (== the symmetry_route survivor). Never reimplement min/max-on-tkey here: doing so
    # once picked the anti-survivor (min instead of max) and silently mis-mapped sectors.
    rep_of[mask] = sec_of(canonical_rep(corner(mask)))

canonical = sorted(set(rep_of.values()))
out = {"rep_of": rep_of, "canonical": canonical, "order": "_target_key = (-r,-s,|abs|)"}
path = os.path.join(ROOT, "results/canonical_sectors_tkey.pkl")
with open(path, "wb") as f:
    pickle.dump(out, f)

print(f"sectors: {(1 << NPROP) - 1}")
print(f"canonical (clean-orbit, _target_key) sectors: {len(canonical)}")
print(f"zoom: {((1 << NPROP) - 1) / len(canonical):.2f}x fewer sectors for data-gen")
print(f"saved -> {path}")
# also emit the comma-separated list for --restrict-sectors
listpath = os.path.join(ROOT, "results/canonical_sectors_tkey.txt")
with open(listpath, "w") as f:
    f.write(",".join(str(s) for s in canonical) + "\n")
print(f"restrict-sectors list -> {listpath}")
print(f"first 20 canonical sectors: {canonical[:20]}")
