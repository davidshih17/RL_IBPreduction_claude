"""Worker-log audit for the (8,5) reduction.

Walks every .log in work/logs and classifies each worker by termination
cause. Cross-references the existence of a .pkl in work/results to know
whether the worker produced a usable output. Also inspects the .err file
for "Cannot allocate memory" / RuntimeError tracebacks that signal a
pytorch-caught OOM (process exits 1 cleanly with a traceback, not signal 9).

Categories (mutually exclusive, in priority order):
  STILL_RUNNING        — executed but no terminate event yet (or in queue)
  NEVER_STARTED       — submitted but never executed (still idle)
  SUCCESS_PKL         — Normal termination 0 + .pkl exists
  PYTORCH_OOM         — Normal termination 1 + .err shows
                          "Cannot allocate memory" or numpy MemoryError.
                          The pytorch CPU allocator caught ENOMEM and
                          raised RuntimeError; python exited 1.
  ORCH_DUP_KILL       — Normal termination 1 + no .pkl AND no OOM in .err
                          (true orchestrator-driven SIGTERM kill).
  NORMAL_EXIT_NONZERO — Normal termination N!=0,1 + no .pkl
  SUCCESS_NO_PKL      — Normal termination 0 but no .pkl (shouldn't happen)
  WORKER_SIGKILL      — Abnormal termination (signal 9). Kernel OOM kill.
  WORKER_ABNORMAL_OTHER — Abnormal termination (other signal). e.g. SIGSEGV
  CONDOR_OOM_EVICT    — 004 evicted Code 1009 (Condor memory enforcement)
  CONDOR_ABORT_OOM    — 009 aborted + .err shows OOM traceback (orchestrator
                          or condor_rm fired but the worker had also crashed
                          on OOM around the same time)
  CONDOR_ABORT_OTHER  — 009 aborted with no 1009 eviction, no OOM in .err
  HELD                — 012 held (rare here)
  UNKNOWN             — no parseable termination signature

Outputs:
  1. Counts per category, all-time AND past 15h.
  2. For WORKER_SIGKILL + CONDOR_OOM_EVICT (the actual OOM categories):
       list of integrals + peak memory + kill time.
  3. The "no pkl + not running" set is the real "lost jobs" bucket.

Run:  python audit_worker_logs.py [--since-hours N]
"""
import os
import re
import sys
import time
import subprocess
from collections import defaultdict, Counter
from datetime import datetime, timedelta

WORK_DIR = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_delta/work'
LOG_DIR = os.path.join(WORK_DIR, 'logs')
RES_DIR = os.path.join(WORK_DIR, 'results')

CUTOFF_HOURS = 15
if '--since-hours' in sys.argv:
    CUTOFF_HOURS = int(sys.argv[sys.argv.index('--since-hours') + 1])

CUTOFF_TS = time.time() - CUTOFF_HOURS * 3600

# Build set of completed pkls once.
print(f'[{time.strftime("%H:%M:%S")}] enumerating result pkls...', flush=True)
pkl_set = set()
for entry in os.scandir(RES_DIR):
    if entry.name.endswith('.pkl'):
        pkl_set.add(entry.name[:-4])  # strip .pkl
print(f'  {len(pkl_set):,} pkls', flush=True)

# Build set of currently-queued workers (Condor q -nobatch).
print(f'[{time.strftime("%H:%M:%S")}] querying condor_q...', flush=True)
try:
    cq = subprocess.run(
        ['condor_q', 'dshih', '-nobatch'],
        capture_output=True, text=True, timeout=60,
    ).stdout
except Exception as e:
    cq = ''
    print(f'  condor_q failed: {e}', flush=True)
running = set()
for line in cq.split('\n'):
    if 'pentagonbox_8_5_delta' not in line:
        continue
    m = re.search(r'async_\d+_[\d_\-]+\.pkl', line)
    if m:
        running.add(m.group(0)[:-4])
print(f'  {len(running):,} workers running', flush=True)

# Event regex helpers
RE_EVENT = re.compile(r'^(\d{3}) \((\d+\.\d+\.\d+)\) (\S+ \S+) (.+)$')
RE_NORMAL = re.compile(r'\(1\) Normal termination \(return value (\d+)\)')
RE_ABNORMAL = re.compile(r'\(0\) Abnormal termination \(signal (\d+)\)')
RE_EVICT_CODE = re.compile(r'Job was evicted\. Code (\d+) Subcode (\d+)')
RE_MEM = re.compile(r'^\s*(\d+)\s+-\s+MemoryUsage of job \(MB\)\s*$')

OOM_ERR_PATTERN = re.compile(
    r"Cannot allocate memory|"
    r"numpy\._core\._exceptions\._ArrayMemoryError|"
    r"MemoryError|"
    r"OOMKill|"
    r"out of memory"
)

def err_has_oom(name):
    """Return True if .err for `name` shows ENOMEM / MemoryError signature."""
    err_path = os.path.join(LOG_DIR, name + '.err')
    try:
        # tail-equivalent: read last 16KB to avoid scanning multi-MB files
        st = os.stat(err_path)
        with open(err_path, 'rb') as f:
            if st.st_size > 16384:
                f.seek(st.st_size - 16384)
            chunk = f.read().decode('latin-1', errors='replace')
        return bool(OOM_ERR_PATTERN.search(chunk))
    except OSError:
        return False


def parse_log(path):
    """Returns dict with submit_ts/exec_ts/term_ts, category, peak_mem,
    return_code, signal, eviction_code, integral (from filename).
    """
    name = os.path.basename(path)[:-4]  # strip .log
    info = {
        'name': name,
        'submit_ts': None,
        'exec_ts': None,
        'term_ts': None,
        'category': 'UNKNOWN',
        'peak_mem_mb': 0,
        'return_code': None,
        'signal': None,
        'evict_code': None,
        'aborted': False,
        'held': False,
    }
    try:
        with open(path) as f:
            for line in f:
                m_ev = RE_EVENT.match(line)
                if m_ev:
                    code = m_ev.group(1)
                    ts_str = m_ev.group(3)
                    try:
                        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').timestamp()
                    except Exception:
                        ts = None
                    if code == '000' and info['submit_ts'] is None:
                        info['submit_ts'] = ts
                    elif code == '001' and info['exec_ts'] is None:
                        info['exec_ts'] = ts
                    elif code == '005':
                        info['term_ts'] = ts
                    elif code == '004':
                        # eviction; may carry Code N
                        pass  # the code is on next continuation lines; matched below
                    elif code == '009':
                        info['aborted'] = True
                        if info['term_ts'] is None:
                            info['term_ts'] = ts
                    elif code == '012':
                        info['held'] = True
                m_mem = RE_MEM.match(line)
                if m_mem:
                    mem = int(m_mem.group(1))
                    if mem > info['peak_mem_mb']:
                        info['peak_mem_mb'] = mem
                m_norm = RE_NORMAL.search(line)
                if m_norm:
                    info['return_code'] = int(m_norm.group(1))
                m_abn = RE_ABNORMAL.search(line)
                if m_abn:
                    info['signal'] = int(m_abn.group(1))
                m_ev2 = RE_EVICT_CODE.search(line)
                if m_ev2:
                    info['evict_code'] = int(m_ev2.group(1))
    except OSError:
        info['category'] = 'READ_ERROR'
        return info

    has_pkl = name in pkl_set
    is_running = name in running

    # Classify (priority order).
    if info['signal'] == 9:
        info['category'] = 'WORKER_SIGKILL'
    elif info['signal'] is not None:
        info['category'] = 'WORKER_ABNORMAL_OTHER'
    elif info['evict_code'] == 1009:
        info['category'] = 'CONDOR_OOM_EVICT'
    elif info['aborted']:
        # Distinguish OOM-during-abort from clean abort
        info['category'] = 'CONDOR_ABORT_OOM' if err_has_oom(name) else 'CONDOR_ABORT_OTHER'
    elif info['return_code'] == 0:
        info['category'] = 'SUCCESS_PKL' if has_pkl else 'SUCCESS_NO_PKL'
    elif info['return_code'] == 1:
        if has_pkl:
            info['category'] = 'SUCCESS_PKL'
        elif err_has_oom(name):
            info['category'] = 'PYTORCH_OOM'
        else:
            info['category'] = 'ORCH_DUP_KILL'
    elif info['return_code'] is not None:
        info['category'] = 'NORMAL_EXIT_NONZERO'
    elif info['held']:
        info['category'] = 'HELD'
    elif is_running:
        info['category'] = 'STILL_RUNNING'
    elif info['exec_ts'] is not None:
        # Started but no terminate event ⇒ running (maybe queue race) or stale log
        info['category'] = 'STILL_RUNNING'
    elif info['submit_ts'] is not None:
        info['category'] = 'NEVER_STARTED'
    info['has_pkl'] = has_pkl
    info['is_running'] = is_running
    return info


def main():
    print(f'[{time.strftime("%H:%M:%S")}] scanning {LOG_DIR}...', flush=True)
    all_logs = []
    with os.scandir(LOG_DIR) as it:
        for entry in it:
            if entry.name.endswith('.log'):
                all_logs.append(entry.path)
    n = len(all_logs)
    print(f'  {n:,} logs', flush=True)

    # Parse all.
    cats_all = Counter()
    cats_recent = Counter()
    oom_kills = []  # (mtime, category, name, peak_mem_mb, term_ts)
    lost_recent = []  # (mtime, category, name)
    nopkl_total = 0
    nopkl_recent = 0

    t0 = time.time()
    last_print = t0
    for i, path in enumerate(all_logs):
        mtime = os.path.getmtime(path)
        info = parse_log(path)
        cat = info['category']
        cats_all[cat] += 1
        if mtime >= CUTOFF_TS:
            cats_recent[cat] += 1
        # OOM tracking
        if cat in ('WORKER_SIGKILL', 'CONDOR_OOM_EVICT'):
            oom_kills.append((mtime, cat, info['name'], info['peak_mem_mb'], info['term_ts']))
        # Lost-in-recent-window tracking (no pkl, not currently running)
        if mtime >= CUTOFF_TS and not info['has_pkl'] and not info['is_running']:
            nopkl_recent += 1
            lost_recent.append((mtime, cat, info['name'], info['peak_mem_mb']))
        if not info['has_pkl'] and not info['is_running']:
            nopkl_total += 1

        if time.time() - last_print > 5:
            elapsed = time.time() - t0
            rate = (i+1)/elapsed
            eta = (n - i - 1) / max(rate, 1)
            print(f'  [{time.strftime("%H:%M:%S")}] {i+1:,}/{n:,} ({rate:.0f}/s, ETA {eta:.0f}s)', flush=True)
            last_print = time.time()

    elapsed = time.time() - t0
    print(f'[{time.strftime("%H:%M:%S")}] parsed {n:,} logs in {elapsed:.1f}s\n', flush=True)

    # Report.
    print('=' * 70)
    print(f'ALL-TIME taxonomy of {n:,} worker logs')
    print('=' * 70)
    for cat, c in sorted(cats_all.items(), key=lambda x: -x[1]):
        print(f'  {cat:<24} {c:>8,}')
    print(f'  {"TOTAL":<24} {sum(cats_all.values()):>8,}')

    print()
    print('=' * 70)
    print(f'PAST {CUTOFF_HOURS}h taxonomy ({sum(cats_recent.values()):,} workers touched)')
    print('=' * 70)
    for cat, c in sorted(cats_recent.items(), key=lambda x: -x[1]):
        print(f'  {cat:<24} {c:>8,}')
    print(f'  {"TOTAL":<24} {sum(cats_recent.values()):>8,}')

    print()
    print('=' * 70)
    print('Lost-vs-completed accounting')
    print('=' * 70)
    print(f'  Logs total:                  {n:,}')
    print(f'  PKLs total:                  {len(pkl_set):,}')
    print(f'  Currently running:           {len(running):,}')
    print(f'  No-pkl AND not running:      {nopkl_total:,}  (worker invocations that never delivered)')
    print(f'  No-pkl in past {CUTOFF_HOURS}h:        {nopkl_recent:,}')

    print()
    print('=' * 70)
    print(f'OOM kills in past {CUTOFF_HOURS}h (WORKER_SIGKILL + CONDOR_OOM_EVICT)')
    print('=' * 70)
    recent_ooms = [o for o in oom_kills if o[0] >= CUTOFF_TS]
    recent_ooms.sort(key=lambda x: x[0])
    print(f'  count: {len(recent_ooms)}')
    print(f'  {"when":<19} {"category":<19} {"mem_MB":>8}  name')
    for mt, cat, name, mem, term_ts in recent_ooms:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt))
        print(f'  {when:<19} {cat:<19} {mem:>8}  {name}')

    print()
    print('=' * 70)
    print(f'OOM kills ALL-TIME')
    print('=' * 70)
    n_sig9 = cats_all.get('WORKER_SIGKILL', 0)
    n_evict = cats_all.get('CONDOR_OOM_EVICT', 0)
    print(f'  WORKER_SIGKILL  : {n_sig9:,}')
    print(f'  CONDOR_OOM_EVICT: {n_evict:,}')
    print(f'  total OOM      : {n_sig9 + n_evict:,}')


if __name__ == '__main__':
    main()
