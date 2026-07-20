#!/usr/bin/env python
"""Coefficient consistency of the g127 cycle pair: A = c1*B, B = c2*A.
If both identities are exact then c1*c2 = 1 mod 1009."""
import pickle
D = ("/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/"
     "results/gr_reduce/g127/work/results")
ra = pickle.load(open(D + "/async_864_3_2_0_0_1_0_3_0_0_0_0_0_0_-2_0.pkl", "rb"))
rb = pickle.load(open(D + "/async_878_2_2_0_0_1_1_2_0_0_0_0_0_0_-1_0.pkl", "rb"))
fa, fb = ra["final_expr"], rb["final_expr"]
(kb, c1), = fa.items()
(ka, c2), = fb.items()
print(f"A -> {list(kb)} * {c1}")
print(f"B -> {list(ka)} * {c2}")
print(f"c1*c2 mod 1009 = {c1 * c2 % 1009}   (exact pair iff 1)")
print(f"A path: {ra.get('path')}")
print(f"B path: {rb.get('path')}")
