"""Set aside (preserve checkpoint + identity-result + condor_rm) every CURRENTLY
RUNNING round-2 worker using more than THRESHOLD GB of memory.

Safety: only acts on jobs whose Args reference the round-2 work dir, and only on
JobStatus==2 (running). Writes the identity result BEFORE removing the job so the
orchestrator caches it as set-aside and never re-submits it.

Usage: python set_aside_memhogs.py [threshold_gb=10]
"""
import os
import pickle
import re
import shutil
import subprocess
import sys
import time

THRESH_MB = (float(sys.argv[1]) if len(sys.argv) > 1 else 10.0) * 1024
BASE = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
SETASIDE = f'{BASE}/results/pentagonbox_8_5_v6_round2/set_aside_memhogs'
GUARD = 'pentagonbox_8_5_v6_round2'   # only touch round-2 workers

out = subprocess.run(
    ['condor_q', 'dshih', '-af', 'ClusterId', 'ProcId', 'MemoryUsage',
     'JobStatus', 'Args'],
    capture_output=True, text=True).stdout

jobs = []
for line in out.splitlines():
    parts = line.split()
    if len(parts) < 5:
        continue
    cid, pid, mem, status = parts[0], parts[1], parts[2], parts[3]
    args = ' '.join(parts[4:])
    try:
        mem = float(mem)
    except ValueError:
        continue
    if status != '2' or mem <= THRESH_MB:
        continue
    if GUARD not in args:            # SAFETY: never touch non-round-2 jobs
        continue
    m_ig = re.search(r"--integral='?([0-9,\-]+)'?", args)
    m_out = re.search(r"--output\s+(\S+)", args)
    if not (m_ig and m_out):
        continue
    jobs.append((f'{cid}.{pid}', m_ig.group(1), m_out.group(1), mem))

print(f'Found {len(jobs)} running round-2 jobs > {THRESH_MB/1024:.0f} GB')
os.makedirs(SETASIDE, exist_ok=True)

for jobid, igstr, output, mem in jobs:
    ig = tuple(int(x) for x in igstr.split(','))
    igu = '_'.join(str(x) for x in ig)
    ckpt = output + '.checkpoint'
    saved = None
    if os.path.exists(ckpt):
        saved = os.path.join(SETASIDE, f'{igu}.checkpoint')
        shutil.copy2(ckpt, saved)
    result = {
        'original_integral': ig, 'success': False, 'final_expr': {ig: 1},
        'set_aside': True,
        'set_aside_reason': f'memhog {mem/1024:.1f}GB job {jobid}',
        'set_aside_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output, 'wb') as f:       # write identity BEFORE removing job
        pickle.dump(result, f)
    with open(os.path.join(SETASIDE, 'manifest.txt'), 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\t{ig}\t'
                f'memhog {mem/1024:.1f}GB {jobid}\tckpt={saved}\t'
                f'output={output}\n')
    rm = subprocess.run(['condor_rm', jobid], capture_output=True, text=True)
    print(f'  set aside {jobid}  {mem/1024:5.1f}GB  {ig}  '
          f'ckpt={"yes" if saved else "NO"}  rm={rm.returncode==0}')

print('done')
