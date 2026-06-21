#!/usr/bin/env python3
"""Decisive local test for the CPU-hold bug: after onestep_worker_v7's import
ORDER (beam_search_v7 first -> _cap_incidental_threads pin -> then torch), force
torch to spin up its intra-op threadpool, then read EVERY thread's CPU-affinity
from /proc/self/task/*/status. If the fix is correct, every thread is confined to
the same <=8-core set (so the process can never exceed 8.0 CPU). If torch threads
escaped the pin (old order), some threads would be allowed on all 64 cores.
"""
import os
import sys

# Replicate the worker's pre-import setup so _cap_incidental_threads sees 8/8.
sys.argv = ['test', '--n-threads', '8', '--n-workers', '8']
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

# SAME import order as the fixed worker: beam_search_v7 FIRST (pins, then imports
# torch internally), then torch.
import beam_search_v7 as bs7   # noqa: E402  -> _cap_incidental_threads() + import torch
import torch                    # noqa: E402  -> already-imported, pinned

torch.set_num_threads(8)
# Force the intra-op threadpool to actually materialize its worker threads.
a = torch.randn(2048, 2048)
for _ in range(5):
    a = a @ a
    a = a / (a.norm() + 1e-6)

import glob
from collections import Counter

main_allowed = sorted(os.sched_getaffinity(0))
print(f'main-thread sched_getaffinity: {len(main_allowed)} cores -> {main_allowed}',
      flush=True)

per_thread = {}
for t in glob.glob('/proc/self/task/*/status'):
    try:
        with open(t) as f:
            for line in f:
                if line.startswith('Cpus_allowed_list:'):
                    per_thread[t] = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass

c = Counter(per_thread.values())
print(f'total threads in process: {len(per_thread)}', flush=True)
print('Cpus_allowed_list -> #threads:', flush=True)
for cpus, n in c.items():
    print(f'  {n:3d} threads  allowed on: {cpus}', flush=True)

# Count cores in each distinct allowed-set; PASS iff every thread is on <=8 cores.
def _ncores(spec):
    n = 0
    for part in spec.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            n += int(hi) - int(lo) + 1
        else:
            n += 1
    return n

max_cores = max(_ncores(s) for s in c)
print(f'\nmax cores any thread may run on: {max_cores}', flush=True)
print('PASS — all threads confined to <=8 cores' if max_cores <= 8
      else 'FAIL — some thread can run on >8 cores (will exceed RequestCpus)',
      flush=True)
sys.exit(0 if max_cores <= 8 else 1)
