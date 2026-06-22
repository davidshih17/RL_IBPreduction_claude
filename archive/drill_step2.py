"""Drill into the step-1 -> step-2 divergence between v7 and v8 on probe 74.

Uses the beam=1 (greedy) pick dumps: each picks_stepN.pkl is a list of
(expr_items, target, n_valid, picked, n_blocked, action_probs) per task; beam=1
=> exactly one task per step.

  picks_step0  = the run's STEP 1  (parent expr = {start})
  picks_step1  = the run's STEP 2  (parent expr = result of step-1 action)

weight()/_target_key() are pure -> no topology needed."""
import sys
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
    with open(os.path.join(d, f'picks_step{n}.pkl'), 'rb') as f:
        tasks = pickle.load(f)
    return tasks[0]   # beam=1 -> single task


def rs_hist(expr_items):
    h = {}
    for integ, _ in expr_items:
        w = weight(integ)
        h[(w[0], w[1])] = h.get((w[0], w[1]), 0) + 1
    return h


def show_picked(task, label):
    expr_items, target, n_v, picked, n_blocked, aprobs = task
    # prob of each picked action
    prob_of = {}
    for op_, d_, p in aprobs:
        prob_of[(op_, tuple(d_))] = p
    print(f"  [{label}] target={list(target)}  weight={weight(target)[:2]}  "
          f"n_valid={n_v}  n_terms={len(expr_items)}")
    print(f"       picked action(s): ", end="")
    for (op_, d_) in picked:
        print(f"(op={op_}, delta={list(d_)}, prob={prob_of.get((op_, tuple(d_)), '?'):.4f}) ", end="")
    print()
    # top-5 actions by prob
    top = sorted(aprobs, key=lambda x: -x[2])[:5]
    print(f"       model top-5 actions by prob:")
    for op_, d_, p in top:
        print(f"          op={op_:3d} delta={list(d_)} prob={p:.4f}")
    return expr_items, target, picked, prob_of


def main():
    print("=" * 78)
    print("STEP 1  (picks_step0): parent expr = {start}")
    print("=" * 78)
    t7 = load_step(V7, 0)
    t8 = load_step(V8, 0)
    e7, tg7, pk7, pr7 = show_picked(t7, "v7")
    print()
    e8, tg8, pk8, pr8 = show_picked(t8, "v8")
    print()
    print(f"  --> step-1 parent expr identical? {set(e7) == set(e8)}")
    print(f"  --> step-1 target identical?      {tg7 == tg8}")
    print(f"  --> step-1 picked action identical? {set(pk7) == set(pk8)}")

    print()
    print("=" * 78)
    print("STEP 2  (picks_step1): parent expr = result of the step-1 action")
    print("=" * 78)
    s7 = load_step(V7, 1)
    s8 = load_step(V8, 1)
    e7b, tg7b, pk7b, pr7b = show_picked(s7, "v7")
    print()
    e8b, tg8b, pk8b, pr8b = show_picked(s8, "v8")

    print()
    print("  (r,s) histogram of step-2 parent expr:")
    print(f"     v7: {dict(sorted(rs_hist(e7b).items(), key=lambda x:(-x[0][0],-x[0][1])))}")
    print(f"     v8: {dict(sorted(rs_hist(e8b).items(), key=lambda x:(-x[0][0],-x[0][1])))}")

    set7 = {integ for integ, _ in e7b}
    set8 = {integ for integ, _ in e8b}
    only7 = set7 - set8
    only8 = set8 - set7
    print()
    print(f"  terms in v7 step-2 expr but NOT in v8 (v8 stripped these): {len(only7)}")
    for integ in sorted(only7, key=tkey)[:20]:
        w = weight(integ)
        print(f"     {list(integ)}  (r,s)={w[:2]}  |abs|={w[2]}  tkey<=start? "
              f"{tkey(integ) <= tkey(tuple(tg7))}")
    print(f"  terms in v8 step-2 expr but NOT in v7: {len(only8)}")
    for integ in sorted(only8, key=tkey)[:20]:
        w = weight(integ)
        print(f"     {list(integ)}  (r,s)={w[:2]}  |abs|={w[2]}")

    print()
    print(f"  --> step-2 target identical?       {tg7b == tg8b}   "
          f"(v7={list(tg7b)}, v8={list(tg8b)})")
    print(f"  --> step-2 picked action identical? {set(pk7b) == set(pk8b)}")


if __name__ == '__main__':
    main()
