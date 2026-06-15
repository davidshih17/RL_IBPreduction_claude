"""Careful, evidence-based provenance inventory of the 1130 round-2 non-masters.

For EACH non-master we determine why round-1 left it un-reduced, by reading the
LITERAL data — never guessing:

  - In the replay cache as fail-identity {ig:ig}  => a worker RAN TO COMPLETION
    and wrote success=False (a killed worker writes no result.pkl). We then read
    that worker's .out and classify by its terminal line:
        "no tasks — STUCK"                -> TABU_TRAP   (tabu blocked all actions)
        "no successful candidates — STUCK" -> STUCK_OTHER (apply failed, not tabu)
        anything else                      -> COMPLETED_OTHER
  - NOT in cache => no result.pkl was written. Either:
        a .out exists  -> ATTEMPTED_KILLED (ran, killed before writing: timeout/OOM/condor_rm)
        no .out        -> NEVER_ATTEMPTED  (produced by some reduction, never a sweep target)

Only TABU_TRAP integrals go into round 2 now; everything else is set aside.
"""
import os
import pickle
import collections

BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
SWEEP = f'{BASE}/results/pentagonbox_8_5_v6'
RES_DIR = f'{SWEEP}/work/results'
LOG_DIR = f'{SWEEP}/work/logs'
N_IDX = 11


def parse_integral(stem):
    parts = stem.split('_')
    nums = []
    for p in reversed(parts):
        try:
            int(p)
        except ValueError:
            break
        nums.append(p)
        if len(nums) >= N_IDX:
            break
    if len(nums) < N_IDX:
        return None
    return tuple(int(x) for x in nums[:N_IDX][::-1])


def stuck_line_of(out_path):
    """Return the terminal classification line of a worker .out, or ''."""
    try:
        with open(out_path, 'r', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return ''
    # scan from the end for a STUCK / DONE / drained line
    for ln in reversed(lines[-60:]):
        s = ln.strip()
        if '— STUCK' in s or '-- STUCK' in s or 'STUCK' in s:
            return s
        if 'DONE' in s or 'drained' in s:
            return s
    return ''


print('loading replay_state ...', flush=True)
st = pickle.load(open(f'{SWEEP}/replay_state.pkl', 'rb'))
cache = st['cache']
nm = pickle.load(open(f'{SWEEP}/round2_nonmasters.pkl', 'rb'))['nonmasters']
nm_set = set(nm)
print(f'non-masters: {len(nm)}')

# Build integral -> stem maps from FILENAMES only (no pkl loading).
print('indexing result + out filenames ...', flush=True)
result_stem = {}
for fn in os.listdir(RES_DIR):
    if fn.endswith('.pkl'):
        ig = parse_integral(fn[:-4])
        if ig is not None and ig in nm_set:
            result_stem.setdefault(ig, fn[:-4])
out_stem = {}
for fn in os.listdir(LOG_DIR):
    if fn.endswith('.out'):
        ig = parse_integral(fn[:-4])
        if ig is not None and ig in nm_set:
            out_stem.setdefault(ig, fn[:-4])
print(f'  non-masters with a result.pkl: {len(result_stem)}')
print(f'  non-masters with a .out      : {len(out_stem)}')

cat = collections.Counter()
tabu_trapped = []
stuck_other = []
killed = []
never = []
completed_other = []
detail_samples = collections.defaultdict(list)

for ig in nm:
    in_cache_fail = (ig in cache and cache[ig] == {ig: 1})
    if in_cache_fail:
        # worker completed -> read its terminal line
        stem = result_stem.get(ig)
        line = stuck_line_of(f'{LOG_DIR}/{stem}.out') if stem else ''
        if 'no tasks' in line and 'STUCK' in line:
            cat['TABU_TRAP'] += 1
            tabu_trapped.append(ig)
        elif 'no successful candidates' in line and 'STUCK' in line:
            cat['STUCK_OTHER'] += 1
            stuck_other.append(ig)
        else:
            cat['COMPLETED_OTHER'] += 1
            completed_other.append(ig)
            if len(detail_samples['COMPLETED_OTHER']) < 5:
                detail_samples['COMPLETED_OTHER'].append((ig, line[:80]))
    else:
        if ig in out_stem:
            cat['ATTEMPTED_KILLED'] += 1
            killed.append(ig)
        else:
            cat['NEVER_ATTEMPTED'] += 1
            never.append(ig)

print()
print('=== PROVENANCE TAXONOMY (1130 non-masters) ===')
for k in ('TABU_TRAP', 'STUCK_OTHER', 'COMPLETED_OTHER',
          'ATTEMPTED_KILLED', 'NEVER_ATTEMPTED'):
    print(f'  {k:18s}: {cat[k]}')
print(f'  total: {sum(cat.values())}')
print()
for k, samples in detail_samples.items():
    print(f'  sample {k}:')
    for ig, line in samples:
        print(f'    {ig}  | {line}')

out = f'{SWEEP}/round2_provenance.pkl'
pickle.dump({
    'tabu_trapped': tabu_trapped,
    'stuck_other': stuck_other,
    'completed_other': completed_other,
    'attempted_killed': killed,
    'never_attempted': never,
    'counts': dict(cat),
}, open(out, 'wb'))
print(f'\nwrote {out}')
print(f'ROUND-2 TARGETS (TABU_TRAP): {len(tabu_trapped)}')
