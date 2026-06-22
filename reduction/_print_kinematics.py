"""Print the EXACT finite-field kinematics a topology is loaded with — the
numeric point (dimension d + Mandelstam invariants, mod the GF prime) at which
SAILIR generates training data and runs reductions. These must match between the
training data and any reduction, and should be recorded for reproducibility.

Usage: python reduction/_print_kinematics.py [topology_input/<family>]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from sailir.topology import Topology

topo_dir = sys.argv[1] if len(sys.argv) > 1 else 'topology_input/pentagonbox'
t = Topology.from_dir(topo_dir)

print(f"topology           : {topo_dir}  (family {t.family_name})")
print(f"n_indices          : {t.n_indices}   n_denominators: {t.n_denominators}"
      f"   isp_positions: {t.isp_positions}")
print(f"kinematic_invariants: {t.kinematic_invariants}")
print(f"kinematics_values  : {t.kinematics_values}")
print()
print("=> the reduction/training run over GF(prime) at this single numeric point:")
for k, v in t.kinematics_values.items():
    print(f"     {k:>5} = {v}")
