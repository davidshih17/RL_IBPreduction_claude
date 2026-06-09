"""Open every success=False pkl in <work_dir>/results/ and break down what
kind of failure produced it. Reports counts WITHOUT inferring categories
from heuristics — only direct measurements:
  * best_n_non_masters, steps, best_max_w12 from pkl
  * whether .out exists and contains 'no tasks' (the v6 STUCK pattern)
  * whether .err has a Traceback

Usage:
    python verify_fail_pkls.py <sweep_root>
"""
import argparse, glob, os, pickle, sys, time
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    args = p.parse_args()
    res_dir = os.path.join(args.sweep_root, 'work', 'results')
    log_dir = os.path.join(args.sweep_root, 'work', 'logs')

    t0 = time.time()
    n_total = 0
    by_nm_steps = Counter()
    by_stuck_pattern = Counter()
    by_err_traceback = Counter()
    sample_nonstuck_basenames = []

    for f in glob.glob(os.path.join(res_dir, '*.pkl')):
        try:
            r = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        if r.get('success'):
            continue
        n_total += 1
        base = os.path.basename(f)[:-4]  # strip .pkl
        nm = r.get('best_n_non_masters')
        steps = r.get('steps') or 0
        # Bucket by (nm, steps) categorical
        if nm is None:
            nm_bucket = 'nm=None'
        elif nm == 0:
            nm_bucket = 'nm=0'
        elif nm >= 1:
            nm_bucket = 'nm>=1'
        else:
            nm_bucket = f'nm={nm}'
        steps_bucket = 'steps=0' if steps == 0 else 'steps>0'
        by_nm_steps[(nm_bucket, steps_bucket)] += 1

        # Check .out for "no tasks" STUCK pattern (em-dash variant)
        out_path = os.path.join(log_dir, base + '.out')
        stuck = False
        if os.path.exists(out_path):
            try:
                txt = open(out_path).read()
                if 'no tasks' in txt:
                    stuck = True
            except Exception:
                pass
        by_stuck_pattern['stuck' if stuck else 'not_stuck'] += 1

        # Check .err for Traceback
        err_path = os.path.join(log_dir, base + '.err')
        has_tb = False
        if os.path.exists(err_path):
            try:
                txt = open(err_path).read(64 * 1024)
                if 'Traceback (most recent call last)' in txt:
                    has_tb = True
            except Exception:
                pass
        by_err_traceback['has_traceback' if has_tb else 'no_traceback'] += 1

        if not stuck and len(sample_nonstuck_basenames) < 10:
            sample_nonstuck_basenames.append((base, nm, steps))

    print(f'Scan time: {time.time()-t0:.1f}s')
    print(f'Total success=False pkls: {n_total}')
    print()
    print('=== by (best_nm, steps) ===')
    for k in sorted(by_nm_steps, key=lambda x: -by_nm_steps[x]):
        print(f'  {k[0]:>10s}  {k[1]:>8s}  {by_nm_steps[k]:>5d}')
    print()
    print('=== by .out STUCK pattern ===')
    for k in sorted(by_stuck_pattern, key=lambda x: -by_stuck_pattern[x]):
        print(f'  {k:>12s}  {by_stuck_pattern[k]:>5d}')
    print()
    print('=== by .err Traceback ===')
    for k in sorted(by_err_traceback, key=lambda x: -by_err_traceback[x]):
        print(f'  {k:>14s}  {by_err_traceback[k]:>5d}')
    if sample_nonstuck_basenames:
        print()
        print(f'=== sample non-STUCK success=False pkls (up to 10) ===')
        for base, nm, steps in sample_nonstuck_basenames:
            print(f'  nm={nm} steps={steps}  {base}')


if __name__ == '__main__':
    main()
