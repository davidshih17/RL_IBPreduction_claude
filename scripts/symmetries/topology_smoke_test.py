"""
Smoke test for sailir.topology: load both trianglebox and pentagon-box
from their topology_input/ directories and print key metadata.

This touches no production code; it only exercises the new loader.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent  # SAILIR_phase2
sys.path.insert(0, str(ROOT))

from sailir.topology import Topology

TOPOLOGY_INPUT = ROOT / "topology_input"


def show(t: Topology):
    print(f"=== Topology: {t.name} (family parser name: '{t.family_name}') ===")
    print(f"  n_indices         = {t.n_indices}")
    print(f"  n_denominators    = {t.n_denominators}")
    print(f"  isp_positions     = {t.isp_positions}  (0-indexed)")
    print(f"  top_sector        = {t.top_sector}  (binary {bin(t.top_sector)})")
    print(f"  kinematic_invars  = {t.kinematic_invariants}")
    print(f"  kinematics_values = {t.kinematics_values}")
    print(f"  # IBP identities  = {t.n_ibp}")
    print(f"  # LI identities   = {t.n_li}")
    print(f"  total action size = {t.n_actions}")
    print(f"  # masters         = {len(t.masters)}")
    print(f"  # sectors w masts = {len(t.masters_by_sector)}")
    print(f"  first 3 IBP[0] terms: ")
    for shift, coeff in t.ibp_templates[0][:3]:
        print(f"    shift={shift}  coeff={coeff!r}")
    print(f"  first 3 masters:")
    for m in t.masters[:3]:
        print(f"    {m}  (sector {t.sector_of(m)})")
    # quick sanity: all masters' ISP positions are non-positive
    bad = [m for m in t.masters if any(m[i] > 0 for i in t.isp_positions)]
    if bad:
        print(f"  WARNING: {len(bad)} masters have positive index at an ISP position!")
        for m in bad[:5]:
            print(f"    {m}")
    else:
        print(f"  OK: no master violates ISP positions ({t.isp_positions})")
    # corner sanity: corner of top sector is in or near the basis
    corner = t.corner_of(t.top_sector)
    in_basis = corner in set(t.masters)
    print(f"  top-sector corner = {corner}  is_master = {in_basis}")
    print()


def main():
    print("Loading topologies from", TOPOLOGY_INPUT)
    print()
    tb = Topology.from_dir(TOPOLOGY_INPUT / "trianglebox")
    show(tb)
    pb = Topology.from_dir(TOPOLOGY_INPUT / "pentagonbox")
    show(pb)

    # Cross-check some hard-coded SAILIR truths against the loaded objects
    print("=== Cross-checks ===")
    assert tb.n_indices == 7, tb.n_indices
    assert tb.n_denominators == 6, tb.n_denominators
    assert tb.isp_positions == (6,), tb.isp_positions
    assert tb.n_ibp == 8, tb.n_ibp
    assert tb.n_li == 1, tb.n_li
    assert len(tb.masters) == 16, len(tb.masters)
    print("  trianglebox: matches SAILIR's known constants (7 indices, "
          "6 denoms, 1 ISP at pos 6, 8 IBPs + 1 LI, 16 masters)")

    assert pb.n_indices == 11, pb.n_indices
    assert pb.n_denominators == 8, pb.n_denominators
    assert pb.isp_positions == (8, 9, 10), pb.isp_positions
    assert pb.n_ibp == 12, pb.n_ibp
    assert pb.n_li == 6, pb.n_li
    assert len(pb.masters) == 61, len(pb.masters)
    print("  pentagon-box: 11 indices, 8 denoms, 3 ISPs at (8,9,10), "
          "12 IBPs + 6 LIs, 61 masters")
    print()
    print("TOPOLOGY SMOKE TEST: OK")


if __name__ == "__main__":
    main()
