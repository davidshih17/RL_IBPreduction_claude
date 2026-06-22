"""
Diagnose why beam_search returns 0 steps for pentagon-box integral.

Mirrors the onestep_worker setup, then probes:
  - is the target integral correctly identified as a non-master?
  - does find_candidates() return any (ibp_op, delta) pairs?
  - what does the model output when fed those candidates?
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reduction"))

import torch
from sailir.topology import Topology
from sailir import ibp_env
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_sector_mask, max_weight, get_non_masters

# Configure topology
TOPO_DIR = ROOT / "topology_input/pentagonbox"
topology = Topology.from_dir(TOPO_DIR)
ibp_env.init_from_topology(topology)
ibp_env.set_prime(1009)

target = (1, 0, 1, 0, 1, 1, -1, 0, 0, 0, -1)

print(f"target = {target}")
print(f"  N_INDICES={ibp_env.N_INDICES}  N_DENOMINATORS={ibp_env.N_DENOMINATORS}  ISP={ibp_env.ISP_POSITIONS}")
print(f"  sector_of={ibp_env.get_sector_id(target)}  is_master={ibp_env.is_master(target)}")
print(f"  is_corner={ibp_env.is_corner_integral(target)}")
print(f"  sector_mask={get_sector_mask(target)}")
print(f"  N_IBPs={len(ibp_env.IBP_TEMPLATES)}, N_LIs={len(ibp_env.LI_TEMPLATES)}")
print()

# Try one IBP application: get_raw_equation for each op
print("=== Manual enumeration: which (ibp_op, seed) -> raw equation contains the target? ===")
hits = 0
checked = 0
for ibp_op in range(topology.n_actions):
    # Try seeds offset from target by each shift
    if ibp_op < len(ibp_env.IBP_TEMPLATES):
        template = ibp_env.IBP_TEMPLATES[ibp_op]
    else:
        template = ibp_env.LI_TEMPLATES[ibp_op - len(ibp_env.IBP_TEMPLATES)]
    for shift, _ in template:
        seed = tuple(target[i] - shift[i] for i in range(topology.n_indices))
        eq = ibp_env.get_raw_equation(ibp_env.IBP_TEMPLATES, ibp_env.LI_TEMPLATES, ibp_op, seed)
        checked += 1
        if target in eq and eq[target] != 0:
            hits += 1
            if hits <= 5:
                print(f"  HIT: ibp_op={ibp_op}, shift={shift}, eq has {len(eq)} terms, target coeff={eq[target]}")
print(f"  total hits: {hits} of {checked} (ibp_op, shift) combinations")
print()

# Try enumerate_valid_actions from ibp_env
print("=== ibp_env.enumerate_valid_actions ===")
try:
    shifts = {}
    for ibp_op, terms in ibp_env.IBP_TEMPLATES.items():
        shifts[ibp_op] = [s for s, _ in terms]
    for li_idx, terms in ibp_env.LI_TEMPLATES.items():
        shifts[len(ibp_env.IBP_TEMPLATES) + li_idx] = [s for s, _ in terms]
    valid = ibp_env.enumerate_valid_actions(target, {}, ibp_env.IBP_TEMPLATES, ibp_env.LI_TEMPLATES, shifts, filter_mode='none')
    print(f"  filter_mode=none: {len(valid)} valid actions")
    valid_sub = ibp_env.enumerate_valid_actions(target, {}, ibp_env.IBP_TEMPLATES, ibp_env.LI_TEMPLATES, shifts, filter_mode='subsector')
    print(f"  filter_mode=subsector: {len(valid_sub)} valid actions")
    valid_higher = ibp_env.enumerate_valid_actions(target, {}, ibp_env.IBP_TEMPLATES, ibp_env.LI_TEMPLATES, shifts, filter_mode='higher_only')
    print(f"  filter_mode=higher_only: {len(valid_higher)} valid actions")
    if len(valid_higher) > 0:
        print(f"  first 3 actions: {valid_higher[:3]}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
