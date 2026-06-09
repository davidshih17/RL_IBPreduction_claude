"""
Smoke test for the refactored ibp_env.py.

For trianglebox, load via Topology + init_from_topology and verify:
  - eval_coeff produces same values as the old hardcoded eval
  - get_raw_equation produces equations
  - is_master matches PAPER_MASTERS membership
  - parse_templates still works on the legacy IBP/LI files

For pentagon-box, just verify the module runs without errors and
gives sane output (non-trivial equations, masters loaded).
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from sailir.topology import Topology
from sailir import ibp_env


def test_trianglebox():
    print("=== trianglebox regression ===")
    tb = Topology.from_dir(ROOT / "topology_input/trianglebox")
    ibp_env.init_from_topology(tb)
    ibp_env.set_prime(1009)

    assert ibp_env.N_INDICES == 7
    assert ibp_env.N_DENOMINATORS == 6
    assert ibp_env.ISP_POSITIONS == (6,)
    assert ibp_env.FAMILY_NAME == "trianglebox"
    assert ibp_env.KINEMATICS == {"d": 41, "m1": 1, "m2": 31, "m3": 47}
    assert len(ibp_env.IBP_TEMPLATES) == 8
    assert len(ibp_env.LI_TEMPLATES) == 1
    assert len(ibp_env.MASTERS_SET) == 16
    print(f"  globals: N_INDICES={ibp_env.N_INDICES}  ISP_POSITIONS={ibp_env.ISP_POSITIONS}")
    print(f"           {len(ibp_env.IBP_TEMPLATES)} IBPs + {len(ibp_env.LI_TEMPLATES)} LIs")
    print(f"           {len(ibp_env.MASTERS_SET)} masters")

    # eval_coeff on a known coefficient: from IBP[0] terms, first term is
    #   trianglebox[0,0,0,0,0,0,0]*(-a6+d-a3-2*a0-a2)
    # With seed (2,0,2,0,1,1,0): -0 + 41 - 0 - 2*2 - 2 = 35
    seed = (2, 0, 2, 0, 1, 1, 0)
    c0 = ibp_env.eval_coeff("-a6+d-a3-2*a0-a2", seed)
    expected = (-0 + 41 - 0 - 2*2 - 2) % 1009
    assert c0 == expected, f"eval_coeff trianglebox: {c0} != {expected}"
    print(f"  eval_coeff('-a6+d-a3-2*a0-a2', {seed}) = {c0}  (expected {expected})")

    # get_raw_equation
    eq = ibp_env.get_raw_equation(ibp_env.IBP_TEMPLATES, ibp_env.LI_TEMPLATES, 0, seed)
    print(f"  IBP[0] applied to {seed}: {len(eq)} terms")
    assert len(eq) > 0

    # is_master: all 16 paper masters should pass
    masters = [
        (0, 1, 1, 1, 0, 0, 0),
        (1, 0, 1, 0, 1, 0, 0),
        (1, 1, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 1, 0, 0),
        (1, 0, 1, 1, 1, 0, 0),
        (1, -1, 1, 1, 1, 0, 0),
        (0, 1, 1, 1, 1, 0, 0),
        (-1, 1, 1, 1, 1, 0, 0),
        (1, 1, 1, 1, 1, 0, 0),
        (1, 0, 1, 0, 0, 1, 0),
        (1, 1, 0, 1, 0, 1, 0),
        (1, 0, 1, 0, 1, 1, 0),
        (1, -1, 1, 0, 1, 1, 0),
        (1, 0, 0, 1, 1, 1, 0),
        (1, 1, 0, 1, 1, 1, 0),
        (1, 0, 1, 1, 1, 1, 0),
    ]
    for m in masters:
        assert ibp_env.is_master(m), f"is_master({m}) should be True"
    print(f"  is_master: all 16 paper masters pass")

    # parse_templates on the legacy IBP_PATH still works (with new default family_name)
    legacy_ibp = ibp_env.parse_templates(ROOT / "topology_input/trianglebox/IBP")
    assert len(legacy_ibp) == 8, len(legacy_ibp)
    print(f"  parse_templates on legacy IBP file: {len(legacy_ibp)} templates")

    # is_top_sector / is_corner_integral / is_higher_sector
    assert ibp_env.is_top_sector((1, 1, 1, 1, 1, 1, 0)) is True
    assert ibp_env.is_top_sector((0, 1, 1, 1, 1, 1, 0)) is False  # missing pos 0
    assert ibp_env.is_corner_integral((1, 1, 1, 1, 1, 1, 0)) is True
    assert ibp_env.is_corner_integral((1, 1, 1, 1, 1, 1, -1)) is False  # ISP non-zero
    # is_higher_sector is hardcoded for trianglebox — should work
    assert ibp_env.is_higher_sector((1, 1, 1, 1, 1, 1, 1)) is True
    print(f"  is_top_sector/is_corner_integral/is_higher_sector: OK")
    print()


def test_pentagonbox():
    print("=== pentagonbox functional check ===")
    pb = Topology.from_dir(ROOT / "topology_input/pentagonbox")
    ibp_env.init_from_topology(pb)
    ibp_env.set_prime(1009)

    assert ibp_env.N_INDICES == 11
    assert ibp_env.N_DENOMINATORS == 8
    assert ibp_env.ISP_POSITIONS == (8, 9, 10)
    assert ibp_env.FAMILY_NAME == "TA"
    assert "s12" in ibp_env.KINEMATICS
    assert len(ibp_env.IBP_TEMPLATES) == 12
    assert len(ibp_env.LI_TEMPLATES) == 6
    assert len(ibp_env.MASTERS_SET) == 61
    print(f"  globals: N_INDICES={ibp_env.N_INDICES}  ISP_POSITIONS={ibp_env.ISP_POSITIONS}")
    print(f"           {len(ibp_env.IBP_TEMPLATES)} IBPs + {len(ibp_env.LI_TEMPLATES)} LIs")
    print(f"           {len(ibp_env.MASTERS_SET)} masters")

    # Apply IBP[0] to a seed in the top sector
    seed = (2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
    eq = ibp_env.get_raw_equation(ibp_env.IBP_TEMPLATES, ibp_env.LI_TEMPLATES, 0, seed)
    print(f"  IBP[0] on {seed}: {len(eq)} terms (non-zero coeffs)")
    assert len(eq) > 0

    # is_master: the corner of top sector 255 IS in the master basis (we verified earlier)
    assert ibp_env.is_master((1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0))
    print(f"  is_master(top-sector corner) = True")

    # is_higher_sector should raise (only valid for trianglebox)
    try:
        ibp_env.is_higher_sector((1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0))
    except NotImplementedError as e:
        print(f"  is_higher_sector raises NotImplementedError on TA  (OK)")
    else:
        raise AssertionError("is_higher_sector should raise on TA")
    print()


def main():
    test_trianglebox()
    test_pentagonbox()
    print("IBP_ENV REFACTOR SMOKE TEST: OK")


if __name__ == "__main__":
    main()
