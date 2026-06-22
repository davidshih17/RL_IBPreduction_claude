"""Scan the v7 vs v8 beam=1 greedy trajectories step by step, find the FIRST
step where they diverge (parent expr / target / picked action), and dump detail
at and around that step."""
import os
import pickle

V7 = 'results/traj_v7_74/picks'
V8 = 'results/traj_v8_74/picks'


def weight(i):
    return (sum(max(0, x) for x in i),
            -sum(min(0, x) for x in i),
            tuple(abs(x) for x in i))


def tkey(i):
    w = weight(i)
    return (-w[0], -w[1], w[2])


def load_step(d, n):
    p = os.path.join(d, f'picks_step{n}.pkl')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        tasks = pickle.load(f)
    return tasks[0] if tasks else None


def summ(task):
    expr_items, target, n_v, picked, n_blocked, aprobs = task
    return (frozenset(expr_items), tuple(target), tuple(sorted(picked)),
            n_v, expr_items, target, picked, aprobs)


def rs_hist(expr_items):
    h = {}
    for integ, _ in expr_items:
        w = weight(integ)
        h[(w[0], w[1])] = h.get((w[0], w[1]), 0) + 1
    return dict(sorted(h.items(), key=lambda x: (-x[0][0], -x[0][1])))


def main():
    print(f"{'step':>4} | {'v7 target (r,s)':>14} {'v7 nterm':>8} | "
          f"{'v8 target (r,s)':>14} {'v8 nterm':>8} | same_expr same_tgt same_pick")
    first_div = None
    for n in range(60):
        t7 = load_step(V7, n)
        t8 = load_step(V8, n)
        if t7 is None or t8 is None:
            break
        s7 = summ(t7)
        s8 = summ(t8)
        same_expr = s7[0] == s8[0]
        same_tgt = s7[1] == s8[1]
        same_pick = s7[2] == s8[2]
        w7 = weight(s7[5])[:2]
        w8 = weight(s8[5])[:2]
        flag = "" if (same_expr and same_tgt and same_pick) else "  <-- DIVERGE"
        print(f"{n+1:>4} | {str(w7):>14} {len(s7[4]):>8} | "
              f"{str(w8):>14} {len(s8[4]):>8} | "
              f"{str(same_expr):>9} {str(same_tgt):>8} {str(same_pick):>9}{flag}")
        if first_div is None and not (same_expr and same_tgt and same_pick):
            first_div = n

    if first_div is None:
        print("\nNo divergence in the scanned window.")
        return

    print("\n" + "=" * 78)
    print(f"FIRST DIVERGENCE at run-step {first_div+1} (picks_step{first_div})")
    print("=" * 78)
    # show the PARENT expr at divergence (this is the result of step first_div-1's
    # action), and how v7 vs v8 parent exprs differ.
    t7 = summ(load_step(V7, first_div))
    t8 = summ(load_step(V8, first_div))
    e7, e8 = t7[4], t8[4]
    print(f"  v7 parent expr (r,s)-hist: {rs_hist(e7)}   n_terms={len(e7)}")
    print(f"  v8 parent expr (r,s)-hist: {rs_hist(e8)}   n_terms={len(e8)}")
    set7 = {i for i, _ in e7}
    set8 = {i for i, _ in e8}
    only7 = set7 - set8
    only8 = set8 - set7
    start = (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0)
    sk = tkey(start)
    print(f"\n  IN v7 BUT NOT v8 ({len(only7)})  [start tkey={sk}]:")
    for i in sorted(only7, key=tkey):
        w = weight(i)
        print(f"     {list(i)}  (r,s)={w[:2]}  |abs|={w[2]}  "
              f"v8-active(tkey<=start)? {tkey(i) <= sk}")
    print(f"\n  IN v8 BUT NOT v7 ({len(only8)}):")
    for i in sorted(only8, key=tkey):
        w = weight(i)
        print(f"     {list(i)}  (r,s)={w[:2]}  |abs|={w[2]}")
    print(f"\n  v7 target={list(t7[5])} (r,s)={weight(t7[5])[:2]}  picked={[ (o,list(d)) for o,d in t7[6]]}")
    print(f"  v8 target={list(t8[5])} (r,s)={weight(t8[5])[:2]}  picked={[ (o,list(d)) for o,d in t8[6]]}")


if __name__ == '__main__':
    main()
