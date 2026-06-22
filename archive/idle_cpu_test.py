"""Definitive test: do PyTorch's intra-op threads SPIN (burn CPU) while idle?
Warm up the 8-thread OMP pool with a matmul, then sit idle 8s and measure the
WHOLE-process CPU (utime+stime over all threads, from /proc/self/stat). If idle
cores ~= 8, the pool spins (the real cause of the >8 overshoot during the
enumerate phase). If ~0, it sleeps. Env vars are set by the caller so we can A/B
the spin-control knobs."""
import os
import time
import torch

NTHREADS = 8
torch.set_num_threads(NTHREADS)
CLK = os.sysconf('SC_CLK_TCK')


def proc_cpu():
    with open(f'/proc/{os.getpid()}/stat') as f:
        p = f.read().split()
    return (int(p[13]) + int(p[14])) / CLK   # utime+stime (all threads)


# which OpenMP? (Intel libiomp5 -> KMP_BLOCKTIME; GNU libgomp -> GOMP_SPINCOUNT)
cfg = torch.__config__.parallel_info()
x = torch.randn(2000, 2000)
for _ in range(10):
    _ = x @ x                                 # warm the intra-op OMP pool

t0 = proc_cpu()
w0 = time.time()
time.sleep(8.0)                               # IDLE — pool should sleep, not spin
t1 = proc_cpu()
w1 = time.time()
idle_cores = (t1 - t0) / (w1 - w0)

env = {k: os.environ.get(k, 'unset') for k in
       ('KMP_BLOCKTIME', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_NUM_THREADS')}
print(f"env={env}")
print(f"==> IDLE CORES while pool is idle = {idle_cores:.2f}  "
      f"(set_num_threads={NTHREADS}; ~{NTHREADS} = SPINNING, ~0 = sleeping)")
print("parallel_info:", " ".join(cfg.split()))
