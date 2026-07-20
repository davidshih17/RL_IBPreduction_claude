#!/usr/bin/env python
"""Collect a finished cut-master sweep into the family master basis.

Reads every S<sec>.pkl in the sweep dir, checks per-sector depth stability
(and any deeper _d<D> re-probes present), unions the masters, and writes:
  <sweep_dir>/masters_cut.txt            one comma-tuple per line
  <topo_dir>/masters   (with --install)  family format `FAM[i1,...,iN]  # sec`
--install backs up any existing masters file first and then runs the
reduction/canonical_masters.py gate — the basis is NOT considered installed
unless that gate prints ALL PASS.

Usage:
  SAILIR_TOPOLOGY=<fam> SAILIR_SECTOR_RANK=1 python collect_masters.py \
      --sweep-dir results/<fam>_cut_sweep [--family-prefix GR] [--install]
"""
import os, sys, re, argparse, pickle, glob, shutil, subprocess
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc


def sec_of(t):
    return sum(1 << i for i in range(_tc.N_DEN) if t[i] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', required=True)
    ap.add_argument('--family-prefix', default=None,
                    help='integral name in the masters file (default: from '
                    'the existing masters file, else the topology name)')
    ap.add_argument('--install', action='store_true')
    args = ap.parse_args()
    sd = os.path.abspath(args.sweep_dir)

    expected = set()
    sect_txt = os.path.join(sd, "sectors.txt")
    if os.path.exists(sect_txt):
        expected = {int(x) for x in open(sect_txt).read().split() if x}

    # deepest available result per sector wins
    best = {}       # sector -> (depth, record)
    for f in glob.glob(os.path.join(sd, "S*.pkl")):
        m = re.match(r"S(\d+)(?:_d(\d+))?\.pkl$", os.path.basename(f))
        if not m:
            continue
        S, D = int(m.group(1)), int(m.group(2) or 3)
        d = pickle.load(open(f, "rb"))
        if S not in best or D > best[S][0]:
            best[S] = (D, d)

    missing = sorted(expected - set(best))
    unstable = sorted(S for S, (_, d) in best.items() if not d["stable"])
    allm = sorted({tuple(m) for _, d in best.values() for m in d["masters"]})

    print(f"topology {_tc.TOPOLOGY}: {len(best)} sectors collected"
          + (f" ({len(expected)} expected)" if expected else ""))
    if missing:
        print(f"MISSING sectors: {missing}")
    if unstable:
        print(f"UNSTABLE sectors (depth-A != depth-B): {unstable}")
    print(f"union: {len(allm)} masters")
    if missing or unstable:
        print("NOT writing output — resolve missing/unstable sectors first "
              "(rerun those jobs, or re-probe deeper with cut_masters_one.py "
              "SEC D+1).")
        sys.exit(1)

    txt = os.path.join(sd, "masters_cut.txt")
    with open(txt, "w") as f:
        for m in allm:
            f.write(",".join(map(str, m)) + "\n")
    print(f"wrote {txt}")

    if not args.install:
        return
    dst = os.path.join(_tc.TOPO_DIR, "masters")
    prefix = args.family_prefix
    if prefix is None and os.path.exists(dst):
        mm = re.match(r"(\w+)\[", open(dst).readline())
        prefix = mm.group(1) if mm else None
    prefix = prefix or _tc.TOPOLOGY
    if os.path.exists(dst):
        n = 1
        while os.path.exists(f"{dst}.bak{n}"):
            n += 1
        shutil.copy2(dst, f"{dst}.bak{n}")
        print(f"backed up existing masters -> {dst}.bak{n}")
    with open(dst, "w") as f:
        for m in allm:
            f.write(f"{prefix}[{','.join(map(str, m))}]   #  {sec_of(m)}\n")
    print(f"installed {len(allm)} masters -> {dst}")
    print("running canonical_masters gate ...")
    env = dict(os.environ, SAILIR_TOPOLOGY=_tc.TOPOLOGY, SAILIR_SECTOR_RANK="1")
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "reduction/canonical_masters.py")],
                       env=env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
