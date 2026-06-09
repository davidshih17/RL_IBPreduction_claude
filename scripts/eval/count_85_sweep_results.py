"""Scan the (8,5) v6 sweep work/results/ dir, count success vs failure,
and dump the failed integrals to a file for the second-round retry pass.
"""
import pickle, glob, sys, os, time

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_v6'
RESULTS = os.path.join(ROOT, 'work', 'results')
OUT_FAILS = os.path.join(ROOT, 'failed_integrals.txt')

n_ok = n_fail = n_err = 0
failures = []
peak_mem_kb = 0
total_steps = 0
total_time_s = 0.0
t0 = time.time()

for f in glob.glob(os.path.join(RESULTS, '*.pkl')):
    try:
        with open(f, 'rb') as fp:
            r = pickle.load(fp)
    except Exception:
        n_err += 1
        continue
    if r.get('success'):
        n_ok += 1
    else:
        n_fail += 1
        failures.append({
            'integral': r.get('original_integral'),
            'steps': r.get('steps'),
            'best_max_w12': r.get('best_max_w12'),
            'best_n_non_masters': r.get('best_n_non_masters'),
            'time': r.get('time'),
            'peak_memory_kb': r.get('peak_memory_kb'),
            'pkl': os.path.basename(f),
        })
    pk = r.get('peak_memory_kb') or 0
    if pk > peak_mem_kb:
        peak_mem_kb = pk
    total_steps += r.get('steps') or 0
    total_time_s += r.get('time') or 0.0

print(f'Scanned {n_ok + n_fail + n_err} result files in {time.time()-t0:.1f}s')
print(f'  success: {n_ok}')
print(f'  failed:  {n_fail}')
print(f'  unreadable: {n_err}')
print(f'  total worker steps: {total_steps}')
print(f'  total worker CPU-time: {total_time_s/3600:.2f} h')
print(f'  peak worker RSS observed: {peak_mem_kb/1024:.0f} MB')

if failures:
    print(f'\nFailures (first 20 of {len(failures)}):')
    for f in failures[:20]:
        i = f['integral']
        print(f'  I{list(i) if i else "?"}: steps={f["steps"]} '
              f'mw={f["best_max_w12"]} nm={f["best_n_non_masters"]} '
              f't={f["time"]:.0f}s')

    # Dump comma-separated integral strings for round-2 resubmit
    with open(OUT_FAILS, 'w') as fp:
        for f in failures:
            i = f['integral']
            if i is None:
                continue
            fp.write(','.join(str(x) for x in i) + '\n')
    print(f'\nDumped {len(failures)} integral strings to {OUT_FAILS}')
