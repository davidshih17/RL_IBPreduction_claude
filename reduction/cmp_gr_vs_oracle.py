#!/usr/bin/env python
"""Compare finished gravity3L reductions against the FIRE oracle, exactly.

Our final_expr is in the PAPER basis via translate_to_kira (each canonical
master mapped to its designated FIRE master). FIRE's solutions use the full
REDUNDANT 68-master basis (23 masters are symmetry-duplicates). To compare:
collapse FIRE's solution through the verified merge identities
(I[dropped_paper] = co * I[kept_canonical], kept mapped to its designated
paper master), then require coefficient-for-coefficient equality mod 1009.

Usage: SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 cmp_gr_vs_oracle.py TAG...
(TAG dirs under results/gr_reduce/)."""
import os, sys, pickle
assert os.environ.get('SAILIR_TOPOLOGY') == 'gravity3L'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
P = 1009

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(P)
import canonical_masters as CM
dictionary = CM.apply_canonical_masters()
merges = CM.MERGES or {}

oracle = pickle.load(open(os.path.join(ROOT, "results/fire_oracle_GR.pkl"), "rb"))


def designated(canonical):
    """Canonical master -> its designated paper master (identity for natives).
    Only single-term dictionary entries appear here (debris entries never
    arise for merge partners)."""
    d = dictionary.get(canonical)
    if d is None:
        return canonical, 1
    assert len(d) == 2, f"debris dictionary entry in merge path: {canonical}"
    return d[0], d[1]        # I[paper] = co * I[canonical]


def collapse_fire(sol):
    """FIRE solution over the redundant 68 basis -> the designated basis."""
    out = {}
    for m, c in sol.items():
        m = tuple(m)
        mv = merges.get(m)
        if mv is None:
            out[m] = (out.get(m, 0) + c) % P
            continue
        kept, co = mv[0], mv[1]
        assert len(mv) == 2, f"debris merge for {m}"
        # I[m] = co * I[kept_canonical]; I[paper(kept)] = co2 * I[kept]
        paper, co2 = designated(kept)
        out[paper] = (out.get(paper, 0) + c * co * pow(co2, P - 2, P)) % P
    return {k: v for k, v in out.items() if v % P}


all_ok = True
for tag in sys.argv[1:]:
    d = os.path.join(ROOT, "results/gr_reduce", tag)
    pkl = os.path.join(d, "reduction.pkl")
    if not os.path.exists(pkl):
        print(f"{tag}: not finished yet")
        all_ok = False
        continue
    r = pickle.load(open(pkl, "rb"))
    ours = {tuple(k): v % P for k, v in r["final_expr"].items() if v % P}
    target = tuple(r["start_integral"])
    fire = oracle["solutions"].get(target)
    if fire is None:
        print(f"{tag}: target {list(target)} not in the FIRE table")
        all_ok = False
        continue
    fire_c = collapse_fire(fire)
    same = (ours == fire_c)
    all_ok &= same
    print(f"{tag} {list(target)}: {'EXACT MATCH vs FIRE' if same else 'MISMATCH'} "
          f"({len(ours)} terms ours, {len(fire_c)} collapsed-FIRE, "
          f"{len(fire)} raw-FIRE)")
    if not same:
        for k in sorted(set(ours) | set(fire_c)):
            a, b = ours.get(k, 0), fire_c.get(k, 0)
            if a != b:
                print(f"    {list(k)}: ours {a} vs FIRE {b}")
print("ALL PASS" if all_ok else "NOT YET / MISMATCH")
