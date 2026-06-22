import pickle, sys
BASE='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, BASE)
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import init_from_topology, set_paper_masters_only, set_prime, is_master, weight
init_from_topology(Topology.from_dir(f'{BASE}/topology_input/pentagonbox')); set_prime(1009); set_paper_masters_only(False)
red=pickle.load(open(f'{BASE}/results/pentagonbox_8_5_v6_round3/reduction.pkl','rb'))
cache=red['cache']; start=red['start_integral']
print(f"orchestrator cache size: {len(cache)}  (my rebuild had 103621)")
# the 3 my replay left dangling
three=[(-1,2,1,0,1,1,1,1,-3,0,0),(0,0,2,1,1,-1,0,1,-1,-1,0),(0,0,2,0,0,3,-2,1,-1,0,0)]
for ig in three:
    v=cache.get(ig)
    if v is None: s="NOT in orch cache"
    elif v=={ig:1}: s="identity"
    else: s=f"reduced -> expr({len(v)} terms)"
    print(f"  I{list(ig)} w={weight(ig)[:2]}: {s}")
# replay start through orchestrator cache
def app(expr,cache,p,mx=2000000):
    ch=True;it=0
    while ch:
        ch=False;it+=1;new={}
        for ig,co in expr.items():
            if co==0: continue
            v=cache.get(ig)
            if v is not None and v!={ig:1}:
                for k,c in v.items():
                    if c: new[k]=(new.get(k,0)+co*c)%p
                ch=True
            else: new[ig]=(new.get(ig,0)+co)%p
        expr={k:v for k,v in new.items() if v}
        if it>mx: break
    return expr,it
fin,it=app({start:1},cache,1009)
nm=[i for i,c in fin.items() if c and not is_master(i)]
print(f"\nreplay start through ORCHESTRATOR cache: converged {it} iters, {len(fin)} terms, NON-MASTERS={len(nm)}")
