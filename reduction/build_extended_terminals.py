#!/usr/bin/env python
"""Extended terminal set for the symmetry-enhanced gravity pipeline:
canonical-45 (FIRE, relabeled)  UNION  Kira-sym-64 (the symmetric-basis run,
kira_validate) mapped into canonical sectors via the clean composite maps.
Writes results/gr_extended_terminals.txt and prints the accounting."""
import os, sys, re, pickle
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
os.environ['SAILIR_SECTOR_RANK'] = '1'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
ibp_env.set_paper_masters_only(True)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
ours45 = {tuple(m) for m in ibp_env.MASTERS_SET}

CG = _tc.canonicalize_module()
import sector_canon_maps
maps = sector_canon_maps.load()
canon = set(pickle.load(open(_tc.CANON_PKL, "rb"))["canonical"])


def sec(t):
    return sum(1 << i for i in range(_tc.N_DEN) if t[i] > 0)


sym64 = []
for ln in open(os.path.join(_tc.TOPO_DIR, "kira_validate/results/GR/masters")):
    m = re.match(r"GR\[([0-9,\-]+)\]", ln.strip())
    if m:
        sym64.append(tuple(int(x) for x in m.group(1).split(",")))

terminals = set(ours45)
n_id = n_mapped = n_failed = 0
failed = []
for m in sym64:
    S = sec(m)
    if S in canon:
        terminals.add(m); n_id += 1
        continue
    M, c = maps[S]
    img = CG.image_unsigned(m, M, c)
    if img is None:
        n_failed += 1; failed.append(m)
        continue
    # keep the same-rank canonical term(s); lower debris integrals are
    # reducible and stay non-terminal
    nb = bin(S).count("1")
    main = [J for J in img if bin(sec(J)).count("1") == nb]
    if len(main) != 1:
        n_failed += 1; failed.append(m)
        continue
    terminals.add(main[0]); n_mapped += 1

print(f"canonical-45: {len(ours45)}")
print(f"Kira-sym-64: {len(sym64)} -> already canonical {n_id}, "
      f"mapped {n_mapped}, FAILED {n_failed}")
for m in failed:
    print(f"   FAILED to canonicalize: {list(m)} (sector {sec(m)})")
print(f"extended terminal set: {len(terminals)}")
new = sorted(terminals - ours45)
print(f"new terminals beyond the 45: {len(new)}")
for m in new[:12]:
    print(f"   + {list(m)} (sector {sec(m)})")
with open(os.path.join(ROOT, "results/gr_extended_terminals.txt"), "w") as f:
    for m in sorted(terminals):
        f.write(",".join(str(x) for x in m) + "\n")
print("saved -> results/gr_extended_terminals.txt")
