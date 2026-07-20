#!/usr/bin/env python
"""Generate a Condor cut-master sweep for the current SAILIR_TOPOLOGY.

Creates <outdir>/{sectors.txt, worker.sh, sweep.sub}: one 1-cpu job per
canonical non-trivial sector, each running cut_masters_one.py (two depths,
fixed-window stability). Submit with `condor_submit <outdir>/sweep.sub`,
then collect with collect_masters.py.

Usage:
  SAILIR_TOPOLOGY=<fam> SAILIR_SECTOR_RANK=1 python make_sweep.py \
      --outdir results/<fam>_cut_sweep [--memory-gb 8] [--depth 3]
"""
import os, sys, argparse, pickle, glob, stat
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc

PYTHON = "/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--memory-gb', type=int, default=8)
    ap.add_argument('--depth', type=int, default=3,
                    help='stability window = depths (L+D-1,D-1) and (L+D,D)')
    args = ap.parse_args()
    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)

    canon = sorted(pickle.load(open(_tc.CANON_PKL, "rb"))["canonical"])
    triv = set()
    for p in glob.glob(os.path.join(_tc.TOPO_DIR, "*/sectormappings/*/trivialsector")):
        triv |= {int(x) for x in open(p).read().strip().split(",") if x}
    sectors = [S for S in canon if S not in triv]
    with open(os.path.join(out, "sectors.txt"), "w") as f:
        f.write("\n".join(str(S) for S in sectors) + "\n")

    wsh = os.path.join(out, "worker.sh")
    with open(wsh, "w") as f:
        f.write(f"""#!/bin/bash
ulimit -v {args.memory_gb * 3 // 2 * 1000000}
export SAILIR_TOPOLOGY={_tc.TOPOLOGY} SAILIR_SECTOR_RANK=1
export SAILIR_CUT_SWEEP_DIR={out}
PYTHONUNBUFFERED=1 {PYTHON} \\
  {ROOT}/master-integrals/cut_masters_one.py "$1" "${{2:-{args.depth}}}"
""")
    os.chmod(wsh, os.stat(wsh).st_mode | stat.S_IEXEC)

    with open(os.path.join(out, "sweep.sub"), "w") as f:
        f.write(f"""universe = vanilla
executable = {wsh}
arguments = "$(sec)"
request_cpus = 1
request_memory = {args.memory_gb}GB
request_disk = 1GB
log = {out}/condor.log
output = {out}/out_$(sec).txt
error = {out}/err_$(sec).txt
queue sec from {out}/sectors.txt
""")
    print(f"{_tc.TOPOLOGY}: {len(sectors)} canonical non-trivial sectors "
          f"({len(canon)} canonical, {len(canon) - len(sectors)} trivial)")
    print(f"submit: condor_submit {out}/sweep.sub")


if __name__ == "__main__":
    main()
