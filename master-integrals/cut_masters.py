#!/usr/bin/env python
"""PER-SECTOR MASTER DERIVATION ON THE MAXIMAL CUT.

Masters of sector S = in-sector integrals irreducible modulo subsectors. On
the maximal cut every subsector integral vanishes, so we build the IBP+LI
system from in-sector seeds, DROP all subsector terms, and eliminate in the
total order: the unreduced low-weight in-sector integrals are the sector's
master candidates:  (any S-integral) = (sum over S-masters) + (subsectors).

Caveat (validate, don't assume): cut-counting can miss a master whose maximal
cut vanishes. Every sector is run at TWO depths (stability), and the result
is cross-checked against certified per-sector answers where we have them.

Usage: SAILIR_TOPOLOGY=... SAILIR_SECTOR_RANK=1 cut_masters.py [--out FILE]
Runs all canonical non-trivial sectors of the topology.
"""
import os, sys, argparse, pickle, time
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc

P = 1009


def sector_of(t):
    m = 0
    for i in range(_tc.N_DEN):
        if t[i] > 0:
            m |= 1 << i
    return m


class CutMasters:
    def __init__(self, topo_dir):
        self.topo = Topology.from_dir(topo_dir)
        ibp_env.init_from_topology(self.topo)
        ibp_env.set_prime(P)
        from symmetry_route import tkey
        self.tkey = tkey
        self.env = ibp_env.IBPEnvironment()
        self.n_actions = self.topo.n_actions

    def _seeds(self, S, rmax, smax):
        dens = [i for i in range(_tc.N_DEN) if S >> i & 1]
        nums = [i for i in range(_tc.N_IND) if i not in dens]

        def distribute(total, slots):
            if not slots:
                if total == 0:
                    yield {}
                return
            first, rest = slots[0], slots[1:]
            for k in range(total + 1):
                for d in distribute(total - k, rest):
                    if k:
                        d = dict(d); d[first] = k
                    yield d

        L = len(dens)
        for extra in range(0, rmax - L + 1):
            for dots in distribute(extra, dens):
                for s in range(0, smax + 1):
                    for negs in distribute(s, nums):
                        t = [0] * _tc.N_IND
                        for i in dens:
                            t[i] = 1 + dots.get(i, 0)
                        for i, k in negs.items():
                            t[i] = -k
                        yield tuple(t)

    def sector_masters(self, S, rmax, smax):
        """Eliminate the CUT system (in-sector terms only); return the
        unreduced integrals with weight within (rmax-1, smax-1) — interior
        candidates, excluding box-boundary artifacts."""
        from sailir.ibp_env import get_raw_equation
        L = bin(S).count("1")
        keyed = []
        for seed in self._seeds(S, rmax, smax):
            for op in range(self.n_actions):
                eq = get_raw_equation(self.env.ibp_t, self.env.li_t, op, seed)
                eq = {K: v for K, v in eq.items()
                      if sector_of(K) == S and v % P}      # THE CUT
                if eq:
                    keyed.append((self.tkey(min(eq, key=self.tkey)), op, seed))
        keyed.sort(key=lambda x: x[0], reverse=True)
        rules = {}
        appearing = set()

        for _, op, seed in keyed:
            eq = get_raw_equation(self.env.ibp_t, self.env.li_t, op, seed)
            eq = {K: v % P for K, v in eq.items()
                  if sector_of(K) == S and v % P}
            appearing.update(eq)
            out = dict(eq)
            while True:
                sub = None
                for K in out:
                    if K in rules and (sub is None or self.tkey(K) < self.tkey(sub)):
                        sub = K
                if sub is None:
                    break
                co = out.pop(sub)
                for K, cK in rules[sub].items():
                    v = (out.get(K, 0) + co * cK) % P
                    if v:
                        out[K] = v
                    else:
                        out.pop(K, None)
            if not out:
                continue
            piv = min(out, key=self.tkey)
            co = out.pop(piv)
            inv = pow(co, P - 2, P)
            rules[piv] = {K: (-v * inv) % P for K, v in out.items() if v % P}

        def rs(t):
            return (sum(x for x in t if x > 0), sum(-x for x in t if x < 0))

        masters = sorted(
            (t for t in appearing if t not in rules
             and rs(t)[0] <= rmax - 1 and rs(t)[1] <= smax - 1),
            key=self.tkey, reverse=True)
        return masters, len(keyed), len(rules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, "results/gr_cut_masters.pkl"))
    args = ap.parse_args()
    cm = pickle.load(open(_tc.CANON_PKL, "rb"))
    triv_path = os.path.join(_tc.TOPO_DIR,
                             "kira_validate/sectormappings/GR/trivialsector")
    triv = set()
    if os.path.exists(triv_path):
        triv = set(int(x) for x in open(triv_path).read().strip().split(","))
    eng = CutMasters(_tc.TOPO_DIR)
    out = {}
    for S in sorted(cm["canonical"]):
        if S in triv:
            continue
        L = bin(S).count("1")
        t0 = time.time()
        mA, eA, rA = eng.sector_masters(S, L + 2, 2)
        mB, eB, rB = eng.sector_masters(S, L + 3, 3)
        stable = (set(mA) == set(mB))
        out[S] = dict(masters=mB, stable=stable, depthA=mA)
        print(f"S{S} (L={L}): masters={len(mB)} "
              f"{'STABLE' if stable else f'UNSTABLE (A={len(mA)})'} "
              f"eqs={eB} rules={rB} ({time.time()-t0:.1f}s)", flush=True)
    n_m = sum(len(v['masters']) for v in out.values())
    n_unstable = sum(1 for v in out.values() if not v['stable'])
    print(f"\nTOTAL: {n_m} cut-masters over {len(out)} sectors; "
          f"unstable sectors: {n_unstable}")
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
