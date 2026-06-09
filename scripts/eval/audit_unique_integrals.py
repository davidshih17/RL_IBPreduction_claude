"""Audit dead vs completed UNIQUE integrals.

Each integral may be submitted by multiple parent reductions; only the
unique integral counts toward the orchestrator's pending frontier. An
integral is "delivered" if ANY worker for that integral produced a pkl;
otherwise it's "dead" until resubmitted.

For each unique integral suffix (the part after async_NNNN_):
  - Did any worker produce a pkl?  → DONE
  - If no pkls and no worker currently running → DEAD
  - If no pkls and some worker currently running → IN_PROGRESS

For DEAD integrals, classify by the LATEST worker's termination cause.
"Normal termination (return value 1)" is split into PYTORCH_OOM (.err
shows ENOMEM/MemoryError) vs ORCH_DUP_KILL (true dup-kill).
"""
import os
import re
import subprocess
import time
from collections import defaultdict

WORK_DIR = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_delta/work'
LOG_DIR = os.path.join(WORK_DIR, 'logs')
RES_DIR = os.path.join(WORK_DIR, 'results')

_OOM_ERR = re.compile(
    r"Cannot allocate memory|"
    r"numpy\._core\._exceptions\._ArrayMemoryError|"
    r"MemoryError|"
    r"OOMKill|"
    r"out of memory"
)

def _err_has_oom(err_path):
    """True if .err shows ENOMEM / MemoryError. Reads last 16KB only."""
    try:
        st = os.stat(err_path)
        with open(err_path, 'rb') as f:
            if st.st_size > 16384:
                f.seek(st.st_size - 16384)
            chunk = f.read().decode('latin-1', errors='replace')
        return bool(_OOM_ERR.search(chunk))
    except OSError:
        return False


# integral suffix = name minus async_NNNN_
RE_SUFFIX = re.compile(r'^async_\d+_(.+)$')

def suffix_of(name):
    m = RE_SUFFIX.match(name)
    return m.group(1) if m else None

# 1. All worker invocations: map integral -> list of worker names
print(f'[{time.strftime("%H:%M:%S")}] scanning logs...', flush=True)
integ_workers = defaultdict(list)
with os.scandir(LOG_DIR) as it:
    for entry in it:
        if not entry.name.endswith('.log'):
            continue
        name = entry.name[:-4]
        s = suffix_of(name)
        if s:
            integ_workers[s].append(name)
n_unique = len(integ_workers)
n_total = sum(len(v) for v in integ_workers.values())
print(f'  {n_unique:,} unique integrals, {n_total:,} total worker invocations', flush=True)

# 2. Map integral -> True if any pkl exists
print(f'[{time.strftime("%H:%M:%S")}] scanning pkls...', flush=True)
integ_done = set()
with os.scandir(RES_DIR) as it:
    for entry in it:
        if not entry.name.endswith('.pkl'):
            continue
        name = entry.name[:-4]
        s = suffix_of(name)
        if s:
            integ_done.add(s)
print(f'  {len(integ_done):,} unique integrals delivered at least 1 pkl', flush=True)

# 3. Currently running workers in Condor queue
print(f'[{time.strftime("%H:%M:%S")}] querying condor_q...', flush=True)
try:
    cq = subprocess.run(['condor_q', 'dshih', '-nobatch'],
                        capture_output=True, text=True, timeout=60).stdout
except Exception:
    cq = ''
running_integrals = set()
for line in cq.split('\n'):
    if 'pentagonbox_8_5_delta' not in line:
        continue
    m = re.search(r'async_\d+_([\d_\-]+)\.pkl', line)
    if m:
        running_integrals.add(m.group(1))
print(f'  {len(running_integrals):,} unique integrals currently running', flush=True)

# 4. Classify
done = 0
in_progress = 0
dead = 0
dead_list = []
for integ, workers in integ_workers.items():
    if integ in integ_done:
        done += 1
    elif integ in running_integrals:
        in_progress += 1
    else:
        dead += 1
        dead_list.append((integ, len(workers)))

print()
print('=' * 70)
print('UNIQUE-INTEGRAL accounting')
print('=' * 70)
print(f'  Unique integrals ever submitted:     {n_unique:,}')
print(f'    DONE  (at least one pkl):          {done:,}')
print(f'    IN_PROGRESS (running, no pkl yet): {in_progress:,}')
print(f'    DEAD  (no pkl, not running):       {dead:,}')
print()

# 5. For DEAD integrals, summarize how many worker invocations were tried
n_workers_dead = sum(n for (_, n) in dead_list)
print(f'  Dead integrals have {n_workers_dead:,} dead worker invocations '
      f'({n_workers_dead/max(dead,1):.1f} per dead integral on average)')
print()

# 6. Categorize the dead integrals by their MOST RECENT worker's termination
print(f'[{time.strftime("%H:%M:%S")}] classifying dead integrals by latest worker outcome...', flush=True)
cat_count = defaultdict(int)
oom_dead = []
for integ, _n in dead_list:
    workers = integ_workers[integ]
    # find most recent .log mtime
    latest = max(workers, key=lambda w: os.path.getmtime(os.path.join(LOG_DIR, w + '.log')))
    log_path = os.path.join(LOG_DIR, latest + '.log')
    try:
        with open(log_path) as f:
            txt = f.read()
    except OSError:
        cat_count['READ_ERROR'] += 1
        continue
    cat = 'UNKNOWN'
    if 'Abnormal termination (signal 9)' in txt:
        cat = 'WORKER_SIGKILL'
        mems = re.findall(r'\s+(\d+)\s+-\s+MemoryUsage of job', txt)
        mem = max(int(m) for m in mems) if mems else 0
        oom_dead.append((integ, latest, mem, 'SIGKILL'))
    elif re.search(r'evicted\. Code 1009', txt):
        cat = 'CONDOR_OOM_EVICT'
        mems = re.findall(r'\s+(\d+)\s+-\s+MemoryUsage of job', txt)
        mem = max(int(m) for m in mems) if mems else 0
        oom_dead.append((integ, latest, mem, '1009'))
    elif re.search(r'\(0\) Abnormal termination \(signal \d+\)', txt):
        cat = 'WORKER_ABNORMAL_OTHER'
    elif 'Job was aborted' in txt:
        # Distinguish ENOMEM-during-abort from clean abort
        err_path = os.path.join(LOG_DIR, latest + '.err')
        if _err_has_oom(err_path):
            cat = 'CONDOR_ABORT_OOM'
            mems = re.findall(r'\s+(\d+)\s+-\s+MemoryUsage of job', txt)
            mem = max(int(m) for m in mems) if mems else 0
            oom_dead.append((integ, latest, mem, 'abort_oom'))
        else:
            cat = 'CONDOR_ABORT_OTHER'
    elif 'Normal termination (return value 1)' in txt:
        # PYTORCH_OOM vs true ORCH_DUP_KILL via .err inspection
        err_path = os.path.join(LOG_DIR, latest + '.err')
        if _err_has_oom(err_path):
            cat = 'PYTORCH_OOM'
            mems = re.findall(r'\s+(\d+)\s+-\s+MemoryUsage of job', txt)
            mem = max(int(m) for m in mems) if mems else 0
            oom_dead.append((integ, latest, mem, 'pytorch_oom'))
        else:
            cat = 'ORCH_DUP_KILL'
    elif re.search(r'Normal termination \(return value (\d+)\)', txt):
        cat = 'NORMAL_EXIT_NONZERO'
    elif 'Job was held' in txt:
        cat = 'HELD'
    elif 'Job executing' in txt:
        cat = 'STARTED_NO_TERM'
    elif 'Job submitted' in txt:
        cat = 'NEVER_STARTED'
    cat_count[cat] += 1

print()
print('=' * 70)
print(f'DEAD integrals ({dead:,}) — taxonomy of most recent worker outcome')
print('=' * 70)
for cat, c in sorted(cat_count.items(), key=lambda x: -x[1]):
    print(f'  {cat:<24} {c:>6,}')

# 7. For DEAD integrals whose latest run was an OOM-class, dump them
print()
print('=' * 70)
print(f'DEAD integrals due to OOM (latest worker SIGKILL or Code 1009): {len(oom_dead):,}')
print('=' * 70)
for integ, name, mem, why in oom_dead:
    print(f'  {why:<8} mem={mem:>6} MB  I[{integ.replace("_", ",")}]  log: {name}')
