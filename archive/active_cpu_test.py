"""Measure ACTIVE cores of the two things the job does, to find what exceeds 8:
(A) torch forward-like matmul with set_num_threads(8)
(B) the numpy maxweight-metric ops (argpartition / lexsort / max on big arrays)
Whole-process CPU (all threads) via /proc/self/stat. Env set by caller for A/B.
"""
import os
import time
import numpy as np
import torch

torch.set_num_threads(8)
CLK = os.sysconf('SC_CLK_TCK')


def cpu():
    with open(f'/proc/{os.getpid()}/stat') as f:
        p = f.read().split()
    return (int(p[13]) + int(p[14])) / CLK


def cores_of(fn, secs=5.0):
    fn()  # warmup
    t0, w0 = cpu(), time.time()
    end = w0 + secs
    while time.time() < end:
        fn()
    return (cpu() - t0) / (time.time() - w0)


# (A) torch matmul (model forward proxy)
xt = torch.randn(1500, 1500)
def torch_op():
    _ = xt @ xt

# (B) numpy maxweight-metric proxy: argpartition + lexsort + max on big int arrays
N = 2_000_000
a = np.random.randint(0, 1 << 40, N).astype(np.int64)
b = np.random.randint(0, 100, N).astype(np.int64)
def numpy_op():
    _ = np.argpartition(a, 900)[:900]
    _ = np.lexsort((b[:50000], a[:50000]))
    _ = a.max()

env = {k: os.environ.get(k, 'unset') for k in
       ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'KMP_BLOCKTIME', 'OMP_WAIT_POLICY')}
print(f"env={env}")
print(f"  (A) torch matmul  active cores = {cores_of(torch_op):.2f}  (set_num_threads=8)")
print(f"  (B) numpy metric  active cores = {cores_of(numpy_op):.2f}  (want ~1)")
