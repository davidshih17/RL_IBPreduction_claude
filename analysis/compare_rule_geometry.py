#!/usr/bin/env python
"""Why does the staged router cost more workers than the monolithic one?
Compare the two routers' RULES on the same integrals (run with SAILIR_SECTOR_RANK=1):
  - RHS size (number of terms)
  - number of distinct NON-MASTER RHS integrals (= future work injected)
  - depth: fraction of RHS terms at the SAME (r,s) as the source (shallow terms that
    must be routed/reduced again at this level) vs strictly lower
Sampled over integrals that BOTH routers route (the coverage-equal population)."""
import os, sys, json, random
assert os.environ.get("SAILIR_SECTOR_RANK") == "1"
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
from sailir.ibp_env import is_master
from symmetry_route import symmetry_rule, staged_rule

def rs(i): return (sum(x for x in i if x > 0), sum(-x for x in i if x < 0))

rng = random.Random(20260710)
pool = []
for S in range(1, 256):
    pr = [i for i in range(8) if S >> i & 1]
    numslots = [i for i in range(11) if i not in pr]
    for _ in range(2):
        a = [0] * 11
        for i in pr:
            a[i] = rng.randint(1, 3)
        for i in rng.sample(numslots, rng.randint(0, 2)):
            a[i] = -rng.randint(1, 2)
        pool.append(tuple(a))
with open(os.path.join(BASE, "data/pentagonbox_10x_raw_jsonl/multisector_data_worker0.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 173:
            continue
        pool.append(tuple(json.loads(line)["target"]))
        if i > 10000:
            break
pool = sorted(set(pool))

stats = {"mono": [0, 0, 0, 0], "staged": [0, 0, 0, 0]}   # nterms, nonmasters, same_rs, n_rules
for I in pool:
    rm = symmetry_rule(I); rsg = staged_rule(I)
    if rm is None or rsg is None:
        continue
    W = rs(I)
    for name, r in (("mono", rm), ("staged", rsg)):
        s = stats[name]
        s[0] += len(r)
        s[1] += sum(1 for k in r if not is_master(tuple(k)))
        s[2] += sum(1 for k in r if rs(k) == W)
        s[3] += 1

for name, (nt, nm, srs, n) in stats.items():
    print(f"{name:>7}: rules={n}  mean RHS terms={nt/n:.1f}  "
          f"mean non-master RHS={nm/n:.1f}  mean same-(r,s) RHS terms={srs/n:.2f}")
