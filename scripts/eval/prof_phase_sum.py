"""Sum V5_PROFILE per-step PROF lines in a probe.out into cumulative phase
totals, so we can see where the wall-clock goes across the whole reduction.

PROF lines look like:
  PROF step=12 t_step=2.30s P1=0.80s P2=1.10s P3=0.30s P4=0.05s residual=0.05s
    P1: nm_tied=.. aux=.. gv=.. tabu=..
    P2: batch_prep=.. model_fwd=..
    P3: apply=.. sort=..

Usage: python prof_phase_sum.py <probe.out> [<probe.out> ...]
"""
import re
import sys

TOP = re.compile(
    r'PROF step=(\d+) t_step=([\d.]+)s P1=([\d.]+)s P2=([\d.]+)s '
    r'P3=([\d.]+)s P4=([\d.]+)s residual=([\d.]+)s')
SUB = re.compile(r'^\s+(P[123]): (.*)$')
KV = re.compile(r'(\w+)=([\d.]+)s')


def summarize(path):
    tot = dict(t_step=0.0, P1=0.0, P2=0.0, P3=0.0, P4=0.0, residual=0.0)
    sub = {}  # e.g. 'P1.gv' -> seconds
    nsteps = 0
    with open(path) as f:
        for line in f:
            m = TOP.search(line)
            if m:
                nsteps += 1
                tot['t_step'] += float(m.group(2))
                tot['P1'] += float(m.group(3))
                tot['P2'] += float(m.group(4))
                tot['P3'] += float(m.group(5))
                tot['P4'] += float(m.group(6))
                tot['residual'] += float(m.group(7))
                continue
            ms = SUB.match(line)
            if ms:
                phase = ms.group(1)
                for k, v in KV.findall(ms.group(2)):
                    sub[f'{phase}.{k}'] = sub.get(f'{phase}.{k}', 0.0) + float(v)

    print(f'\n=== {path}  ({nsteps} steps) ===')
    t = tot['t_step'] or 1.0
    par = tot['P1'] + tot['P2'] + tot['P3'] + tot['P4']
    print(f'  sum t_step      = {tot["t_step"]:7.1f}s')
    for ph in ('P1', 'P2', 'P3', 'P4', 'residual'):
        print(f'  {ph:9s}       = {tot[ph]:7.1f}s  ({100*tot[ph]/t:4.1f}%)')
    print(f'  (P1+P2+P3+P4)   = {par:7.1f}s  ({100*par/t:4.1f}%)')
    if sub:
        print('  sub-phases:')
        for k in sorted(sub):
            print(f'    {k:16s} = {sub[k]:7.1f}s  ({100*sub[k]/t:4.1f}%)')
    return tot, sub


def main():
    results = [summarize(p) for p in sys.argv[1:]]
    if len(results) == 2:
        (ta, _), (tb, _) = results
        print('\n=== A vs B per-phase speedup (A/B) ===')
        for ph in ('t_step', 'P1', 'P2', 'P3', 'P4', 'residual'):
            a, b = ta[ph], tb[ph]
            sp = (a / b) if b > 0 else float('inf')
            print(f'  {ph:9s}: A={a:7.1f}s  B={b:7.1f}s  speedup={sp:5.2f}x  '
                  f'saved={a-b:6.1f}s')


if __name__ == '__main__':
    main()
