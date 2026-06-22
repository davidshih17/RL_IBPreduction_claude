"""Stage-2 verification: packed compute_indirect_substituted_incremental_packed
produces the SAME aux as the dict compute_indirect_substituted_incremental.

Replays an identical sequence of (new_sub_int, new_resolved_sol) through both,
starting from an empty aux, on the REAL pentagonbox env (real raws via shifts /
get_raw_equation). At each step we assert:
  - cu entries equal (packed.to_dict == dict cu entry), in order
  - union bitmasks equal
  - rid maps equal
  - iraws equal
target_sector=None (exercises full Phase A + Phase B without sector projection).
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sailir.topology import Topology
import sailir.ibp_env as ibp_env
from sailir.packed_eq import IntegralRegistry
from sailir.packed_cu import compute_indirect_substituted_incremental_packed

TOPO = os.path.join(os.path.dirname(__file__), '..', '..', 'topology_input', 'pentagonbox')
SEED_74 = (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0)
PRIME = 1009


def cu_integrals(dict_cu):
    s = set()
    for eq in dict_cu:
        s.update(eq.keys())
    return s


def main():
    topology = Topology.from_dir(TOPO)
    ibp_env.init_from_topology(topology)
    ibp_env.set_prime(PRIME)
    env = ibp_env.IBPEnvironment()
    ibp_t, li_t, shifts = env.ibp_t, env.li_t, env.shifts
    N = ibp_env.N_INDICES
    rng = random.Random(20260614)

    # two independent raw caches so the runs don't share Phase-B sub_cache state
    rc_dict = {}
    rc_pack = {}
    reg = IntegralRegistry()

    dict_aux = ([], [], {}, [])
    pack_aux = ([], [], {}, [])
    resolved_subs = {}
    subbed = set()

    n_steps = 18
    for step in range(n_steps):
        # choose new_sub_int: seed for step 0, else a random not-yet-subbed
        # integral currently appearing in the cu
        if step == 0:
            new_sub_int = SEED_74
        else:
            pool = [i for i in cu_integrals(dict_aux[0]) if i not in subbed]
            if not pool:
                print(f"(stopped at step {step}: no more cu integrals to sub)")
                break
            new_sub_int = rng.choice(pool)
        if new_sub_int in subbed:
            continue

        # a flat solution: 1-3 integrals NOT in resolved_subs keys (incl. itself)
        cands = [i for i in cu_integrals(dict_aux[0])
                 if i not in subbed and i != new_sub_int]
        rng.shuffle(cands)
        sol = {k: rng.randint(1, PRIME - 1) for k in cands[:rng.randint(1, 3)]}
        # never let the solution reference a sub key (flatness invariant)
        new_resolved_sol = sol
        resolved_subs = dict(resolved_subs)
        resolved_subs[new_sub_int] = new_resolved_sol
        subbed.add(new_sub_int)

        # --- dict reference ---
        _res_d, dict_aux = ibp_env.compute_indirect_substituted_incremental(
            dict_aux, new_sub_int, new_resolved_sol, resolved_subs,
            ibp_t, li_t, shifts, rc_dict, target_sector=None)

        # --- packed ---
        _res_p, pack_aux = compute_indirect_substituted_incremental_packed(
            pack_aux, new_sub_int, new_resolved_sol, resolved_subs,
            ibp_t, li_t, shifts, rc_pack,
            reg, PRIME, N, ibp_env,
            ibp_env.get_raw_equation, target_sector=None)

        d_cu, d_ubm, d_rid, d_iraws = dict_aux
        p_cu, p_ubm, p_rid, p_iraws = pack_aux

        assert len(p_cu) == len(d_cu), \
            f"step {step}: len(cu) {len(p_cu)} != {len(d_cu)}"
        for i in range(len(d_cu)):
            got = p_cu[i].to_dict(reg)
            assert got == d_cu[i], (
                f"step {step}: cu[{i}] mismatch\n dict={d_cu[i]}\n pack={got}")
        assert p_ubm == d_ubm, f"step {step}: ubm mismatch"
        assert p_rid == d_rid, f"step {step}: rid mismatch"
        assert p_iraws == d_iraws, f"step {step}: iraws mismatch"

        # Stage 2b: packed enumerate == dict enumerate on several targets
        n_enum = 0
        if d_cu:
            from sailir.packed_cu import (
                enumerate_valid_actions_with_indirect_cache_packed as enum_packed)
            for tg in list(cu_integrals(d_cu))[:6]:
                dv = ibp_env.enumerate_valid_actions_with_indirect_cache(
                    tg, _res_d, None, resolved_subs, ibp_t, li_t, shifts,
                    'subsector', rc_dict)
                pv = enum_packed(
                    tg, _res_p, resolved_subs, ibp_t, li_t, shifts,
                    'subsector', rc_pack, reg, N)
                assert dv == pv, (
                    f"step {step} ENUMERATE mismatch target {tg}\n"
                    f" dict={dv}\n pack={pv}")
                n_enum += 1

        print(f"step {step:2d}: sub={new_sub_int[:4]}... "
              f"|cu|={len(d_cu)} |iraws|={len(d_iraws)} enum_ok={n_enum} OK")

    print(f"\nPASS: packed compute_indirect == dict over {n_steps} steps "
          f"(cu/ubm/rid/iraws all bit-identical). registry={len(reg)}")


if __name__ == '__main__':
    main()
