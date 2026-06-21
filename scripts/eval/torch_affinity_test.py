import os, sys, time
N_PIN = 4
allowed = sorted(os.sched_getaffinity(0))
if len(allowed) > N_PIN:
    os.sched_setaffinity(0, set(allowed[:N_PIN]))
import torch
torch.set_num_threads(8)
CLK = os.sysconf('SC_CLK_TCK')
def cpu():
    with open(f'/proc/{os.getpid()}/stat') as f: p=f.read().split()
    return (int(p[13])+int(p[14]))/CLK
x = torch.randn(2500,2500)
for _ in range(3): _=x@x
t0,w0=cpu(),time.time()
end=w0+6
while time.time()<end: _=x@x
cores=(cpu()-t0)/(time.time()-w0)
print(f"KMP_AFFINITY={os.environ.get('KMP_AFFINITY','unset')} -> torch(8thr) pinned-to-{N_PIN}cores uses {cores:.2f} cores (MUST be <= {N_PIN})")
