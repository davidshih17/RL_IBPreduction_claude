#!/usr/bin/env python
"""Masters-equality gate for the sectorrank probes: compare final master combination
of <tag>_sectorrank/design1 against the stored <tag>_poststrip arms (design1 and
baseline). Also count dispatched worker targets in non-canonical sectors (the design
predicts exactly zero for the sectorrank arms)."""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))

cm = pickle.load(open(os.path.join(BASE, "results/canonical_sectors_tkey.pkl"), "rb"))
CANON = set(cm["canonical"])


def sector_of(t):
    return sum(1 << k for k in range(8) if t[k] > 0)


def masters(path):
    if not os.path.exists(path):
        return None
    r = pickle.load(open(path, "rb"))
    fe = r.get("final_expression", r.get("final_expr", {}))
    return {tuple(k): v for k, v in fe.items()}


def dispatched(arm):
    d = os.path.join(BASE, "results/ab_symmetry", arm, "work/results")
    if not os.path.isdir(d):
        return None, None
    tot = 0; bad = 0
    for fn in os.listdir(d):
        if fn.endswith(".pkl"):
            r = pickle.load(open(os.path.join(d, fn), "rb"))
            tot += 1
            if sector_of(tuple(r["original_integral"])) not in CANON:
                bad += 1
    return tot, bad


# stored finals: poststrip arms kept no top-level reduction.pkl; the legacy design1
# finals are the *_symcheck runs (post-strip) and the baselines are the original A/B
# tag dirs. Try each location in order.
REFS = {
    "gate_small": (["gate_small_symcheck/design1", "gate_small_poststrip/design1",
                    "gate_small/design1"],
                   ["gate_small_poststrip/baseline", "gate_small/baseline"]),
    "m1": (["m1_symcheck/design1", "m1_6prop/design1"],
           ["m1_6prop/baseline"]),
    "m2": (["m2_symcheck/design1", "m2_4prop_dots/design1"],
           ["m2_4prop_dots/baseline"]),
    "m3": (["m3_symcheck/design1", "m3_5prop_deg3/design1"],
           ["m3_5prop_deg3/baseline"]),
}


def first_masters(cands):
    for c in cands:
        m = masters(os.path.join(BASE, "results/ab_symmetry", c, "reduction.pkl"))
        if m is not None:
            return m, c
    return None, None


for tag in sys.argv[1:]:
    new = masters(os.path.join(BASE, f"results/ab_symmetry/{tag}_sectorrank/design1/reduction.pkl"))
    d1refs, basrefs = REFS.get(tag, ([f"{tag}_symcheck/design1"], []))
    old, oldsrc = first_masters(d1refs)
    bas, bassrc = first_masters(basrefs)
    tot, bad = dispatched(f"{tag}_sectorrank/design1")
    print(f"== {tag} ==")
    if new is None:
        print("   sectorrank run not finished"); continue
    for name, ref, src in (("legacy design1", old, oldsrc), ("baseline", bas, bassrc)):
        if ref is None:
            print(f"   vs {name}: (stored result missing)")
        else:
            print(f"   masters vs {name} [{src}]: {'IDENTICAL' if new == ref else 'MISMATCH'}"
                  f"  ({len(new)} vs {len(ref)} masters)")
            if new != ref:
                for k in sorted(set(new) | set(ref)):
                    if new.get(k) != ref.get(k):
                        print(f"      {list(k)}: new={new.get(k)} ref={ref.get(k)}")
    print(f"   dispatched workers: {tot}, in NON-canonical sectors: {bad}"
          + ("   <-- MUST BE 0" if bad else "   (design holds)"))
