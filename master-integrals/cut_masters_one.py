#!/usr/bin/env python
"""One-sector cut-master job (Condor worker): argv = sector id [depth].
depth D (default 3) runs the two depths (L+D-1, D-1) and (L+D, D) for the
fixed-window stability check. D=3 writes S<sec>.pkl; deeper D writes
S<sec>_d<D>.pkl (used to re-probe sectors whose count is questioned)."""
import os, sys, pickle, time
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
sys.path.insert(0, os.path.join(ROOT, "master-integrals"))
from cut_masters import CutMasters
import topo_config as _tc

S = int(sys.argv[1])
D = int(sys.argv[2]) if len(sys.argv) > 2 else 3
L = bin(S).count("1")
eng = CutMasters(_tc.TOPO_DIR)
t0 = time.time()
mA, eA, rA = eng.sector_masters(S, L + D - 1, D - 1)
mB, eB, rB = eng.sector_masters(S, L + D, D)
out = dict(sector=S, masters=mB, depthA=mA, stable=set(mA) == set(mB),
           eqs=eB, rules=rB, time=time.time() - t0, depth=D)
SWEEP_DIR = os.environ.get(
    "SAILIR_CUT_SWEEP_DIR",
    os.path.join(ROOT, "results/gr_cut_sweep" if _tc.TOPOLOGY == "gravity3L"
                 else f"results/{_tc.TOPOLOGY}_cut_sweep"))
os.makedirs(SWEEP_DIR, exist_ok=True)
suffix = "" if D == 3 else f"_d{D}"
with open(os.path.join(SWEEP_DIR, f"S{S}{suffix}.pkl"), "wb") as f:
    pickle.dump(out, f)
print(f"S{S} (L={L}): masters={len(mB)} "
      f"{'STABLE' if out['stable'] else f'UNSTABLE(A={len(mA)})'} "
      f"eqs={eB} ({out['time']:.1f}s)")
