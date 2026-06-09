"""Compare RS profile across all 40 beam states.
For each state, count how many RS entries are tier_descent / masters_only / lateral / ascent.
Then check whether the winning state(s) have noticeably more masters_only / tier_descent.
"""
import pickle, sys
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')
from sailir.ibp_env import set_prime, set_paper_masters_only, init_from_topology, weight
from sailir.topology import Topology

topo = Topology.from_dir('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox')
init_from_topology(topo); set_prime(1009); set_paper_masters_only(False)

with open(sys.argv[1], 'rb') as f:
    d = pickle.load(f)
beam = d['beam']
start_w12 = d.get('start_w12') or (8,4)
print(f'step={d["step"]} n_beam={len(beam)} start_w12={start_w12}')
print('  Per-state RS counts (tier_descent | lateral | ascent | masters_only) and nm/mw:')

def w2(itgr):
    w = weight(itgr); return (w[0], w[1])

rows = []
for i, s in enumerate(beam):
    rs = s['resolved_subs']
    td=la=ms=as_=0
    for k, v in rs.items():
        kw = w2(k)
        if not v:
            ms += 1; continue
        max_vw = max(w2(vk) for vk in v.keys())
        if max_vw < kw: td += 1
        elif max_vw == kw: la += 1
        else: as_ += 1
    nm = s.get('n_non_masters')
    mw = s.get('max_w12')
    rows.append((i, td, la, as_, ms, nm, mw, len(rs)))

# sort by (nm, then descending masters_only count)
rows.sort(key=lambda r: (r[5], -r[4]))
for r in rows:
    i, td, la, as_, ms, nm, mw, rs_n = r
    print(f'  state {i:2}  RS={rs_n:3}  td={td:3} la={la:3} as={as_:3} ms={ms:3}  nm={nm} mw={mw}')
