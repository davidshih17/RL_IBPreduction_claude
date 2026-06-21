"""Directly verify the v8 subweight strip on a live/finished checkpoint.

The strip (apply_substitution_v5 -> is_active) is supposed to keep ONLY
total-ordering-active integrals in each state's expr: key(i) <= start_w12,
where key(i) = (-r, -s, |abs|-tuple). So if ANY integral in ANY beam state's
expr has key(i) > start_w12, it is subweight that LEAKED past the strip.

weight()/_target_key() are pure, so we need no topology. Usage:
  python verify_strip_v8.py <ckpt.pkl>
"""
import sys
import os
import pickle
# state-dicts hold PackedEq (in resolved_subs) -> need `sailir` importable.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))


def weight(i):
    return (sum(max(0, x) for x in i),
            -sum(min(0, x) for x in i),
            tuple(abs(x) for x in i))


def target_key(i):
    w = weight(i)
    return (-w[0], -w[1], w[2])


def main(path):
    with open(path, 'rb') as f:
        meta = pickle.load(f)
        if not (isinstance(meta, dict) and meta.get('_streamed')):
            print("not a streamed ckpt"); return
        n = meta['n_states']
        beam = [pickle.load(f) for _ in range(n)]
    start_w12 = tuple(meta['start_w12'])
    step = meta.get('step')
    print(f"ckpt step={step}  n_states={n}  start_w12={start_w12}\n")

    total_terms = 0
    total_active = 0
    total_leak = 0
    worst = []          # (leak_count, state_idx, n_terms, example_leak_integral)
    for si, s in enumerate(beam):
        expr = s['expr']
        leaks = []
        active = 0
        for integ in expr:
            k = target_key(integ)
            if k <= start_w12:
                active += 1
            else:
                leaks.append(integ)
        total_terms += len(expr)
        total_active += active
        total_leak += len(leaks)
        if leaks:
            worst.append((len(leaks), si, len(expr), leaks[0]))

    print(f"TOTAL expr terms across beam : {total_terms}")
    print(f"  active (key <= start_w12)  : {total_active}")
    print(f"  SUBWEIGHT LEAK (key > start): {total_leak}")
    if total_leak == 0:
        print("\n==> STRIP OK: zero subweight integrals present in any expr.")
        print("    (Large nm is genuinely many ACTIVE non-masters, not a leak.)")
    else:
        print("\n==> STRIP LEAK: subweight integrals are present in exprs!")
        worst.sort(reverse=True)
        for lk, si, nt, ex in worst[:5]:
            print(f"    state {si}: {lk}/{nt} terms are subweight; "
                  f"e.g. {ex} key={target_key(ex)}")

    # Also report the active-weight spread (r,s) to see how high it has climbed.
    rs_hist = {}
    for s in beam:
        for integ in s['expr']:
            w = weight(integ)
            rs_hist[(w[0], w[1])] = rs_hist.get((w[0], w[1]), 0) + 1
    print("\n(r,s) histogram over all expr terms in the beam:")
    for rs in sorted(rs_hist, key=lambda x: (-x[0], -x[1])):
        print(f"    {rs}: {rs_hist[rs]}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/probe_74_v8/ckpt.pkl')
