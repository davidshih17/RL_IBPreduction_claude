#!/usr/bin/env python
"""Are the truth-resistant stuck integrals masters — in ANY basis — or related
to masters by symmetry? For each: membership in FIRE-68, our canonical-45,
Kira-nosym-155 (IBP+LI-only basis); then every single-term symmetry image
(equal-value partners) checked against all three; sector orbit info."""
import os, sys, pickle, re, importlib.util
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
CG = _tc.canonicalize_module()

# bases
spec = importlib.util.spec_from_file_location(
    "grm", os.path.join(_tc.TOPO_DIR, "GR_masters_dict.py"))
grm = importlib.util.module_from_spec(spec); spec.loader.exec_module(grm)
FIRE = {tuple(m) for s, ms in grm.GR_MASTERS.items() for m in ms}
OURS = {tuple(m) for m in ibp_env.MASTERS_SET}
NOSYM = set()
for ln in open(os.path.join(_tc.TOPO_DIR, "kira_nosym_top/results/GR/masters")):
    m = re.match(r"GR\[([0-9,\-]+)\]", ln.strip())
    if m:
        NOSYM.add(tuple(int(x) for x in m.group(1).split(",")))
cm = pickle.load(open(_tc.CANON_PKL, "rb"))
CANON, REP = set(cm["canonical"]), cm["rep_of"]

TARGETS = [
    "1,1,0,1,1,1,1,0,0,1,-1,0,0,0,0",   # the resistant sector-635 case
    "1,1,0,1,1,0,1,0,0,1,0,0,0,0,0",    # 603 corner
    "1,1,1,1,1,1,0,0,0,0,0,0,0,0,0",    # 63 corner
    "1,0,1,0,1,1,0,0,1,1,0,0,0,0,0",    # 821 corner
    "1,0,1,0,0,1,1,0,1,1,0,0,0,0,0",    # 869 corner
    "1,1,1,1,0,0,0,0,0,0,0,0,0,0,0",    # 15 corner
]


def sec(t):
    return sum(1 << i for i in range(_tc.N_DEN) if t[i] > 0)


def orbit_partners(T):
    """Equal-value partners: closure under SINGLE-TERM clean images."""
    seen = {T}; frontier = [T]
    while frontier:
        K = frontier.pop()
        for (M, c) in CG._transforms(K):
            img = CG.image_unsigned(K, M, c)
            if img is None or len(img) != 1:
                continue
            (J, co), = img.items()
            if J not in seen:
                seen.add(J); frontier.append(J)
    return seen


for spec_s in TARGETS:
    T = tuple(int(x) for x in spec_s.split(","))
    S = sec(T)
    orb = orbit_partners(T)
    partners = sorted(orb - {T})
    hits = []
    for name, basis in (("FIRE68", FIRE), ("OURS45", OURS), ("NOSYM155", NOSYM)):
        if T in basis:
            hits.append(f"T IS {name} MASTER")
        for Q in partners:
            if Q in basis:
                hits.append(f"partner {list(Q)} IS {name} MASTER")
    nosym_in_sector = sorted(m for m in NOSYM if sec(m) == S)
    print(f"{spec_s} (sector {S}, canonical={S in CANON}, rep={REP[S]}):")
    print(f"   equal-value partners: {len(partners)}")
    print(f"   master hits: {hits if hits else 'NONE'}")
    print(f"   nosym masters living in sector {S}: "
          f"{[list(m) for m in nosym_in_sector] if nosym_in_sector else 'none'}")
