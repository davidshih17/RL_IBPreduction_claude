"""Verify: pinning the process tree to N CPUs caps total CPU at N even with
8 fork workers that each WANT ~1.2 cores (the real overshoot scenario).
Without pin -> ~8-10 cores; with pin to N -> <= N."""
import os
import sys
import time
import numpy as np

N_PIN = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = no pin
N_WORKERS = 8
CLK = os.sysconf('SC_CLK_TCK')

allowed = sorted(os.sched_getaffinity(0))
if N_PIN > 0:
    os.sched_setaffinity(0, set(allowed[:N_PIN]))   # pin self + (inherited) children


def one_cpu(pid):
    try:
        with open(f'/proc/{pid}/stat') as f:
            p = f.read().split()
        return (int(p[13]) + int(p[14])) / CLK   # this pid's utime+stime
    except FileNotFoundError:
        return 0.0


def tree_cpu(pids):
    return one_cpu(os.getpid()) + sum(one_cpu(p) for p in pids)


def worker():
    a = np.random.randint(0, 1 << 40, 2_000_000).astype(np.int64)
    b = np.random.randint(0, 100, 2_000_000).astype(np.int64)
    end = time.time() + 6
    while time.time() < end:
        np.argpartition(a, 900)
        np.lexsort((b[:50000], a[:50000]))
        a.max()


pids = []
for _ in range(N_WORKERS):
    pid = os.fork()
    if pid == 0:
        worker()
        os._exit(0)
    pids.append(pid)

t0, w0 = tree_cpu(pids), time.time()
time.sleep(4.0)
t1, w1 = tree_cpu(pids), time.time()
for pid in pids:
    os.waitpid(pid, 0)
cores = (t1 - t0) / (w1 - w0)
print(f"N_PIN={N_PIN} ({'no pin' if N_PIN == 0 else f'pinned to {N_PIN} cpus'}), "
      f"{N_WORKERS} numpy workers -> total cores = {cores:.2f}")
