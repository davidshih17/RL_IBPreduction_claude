#!/usr/bin/env python
"""The FIRE master [1,1,1,1,1,1,1,0,1,1,-1,...] (sector 895, one ISP numerator)
gets a 3-term image under the BFS composite 895->767. Probe EVERY verified
transform applicable at 895 whose den behavior lands in sector 767: does any
give a SINGLE-term image? (The numerator slot 10 = k2.u2 row differs between
maps realizing the same sector move.) Also probe 2-step compositions via the
composite machinery if no direct map works."""
import os, sys
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import canonicalize_GR as CG

M895 = (1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 0, 0, 0, 0)


def sec(t):
    return sum(1 << i for i in range(CG.N_DEN) if t[i] > 0)


print(f"master {list(M895)} (sector {sec(M895)})")
n_app = n_to767 = 0
singles = []
for (M, c) in CG._transforms(M895):
    img = CG.image_unsigned(M895, M, c)
    if img is None:
        continue
    n_app += 1
    secs = {sec(J) for J in img}
    top = max(secs, key=lambda s: bin(s).count("1"))
    if 767 in secs or top == 767:
        n_to767 += 1
        if len(img) == 1:
            singles.append(img)
            print(f"  SINGLE-TERM image: {img}")
        else:
            print(f"  {len(img)}-term image, sectors {sorted(secs)}: "
                  f"{dict(list(img.items())[:3])}")
print(f"applicable transforms: {n_app}, landing in 767: {n_to767}, "
      f"single-term: {len(singles)}")
