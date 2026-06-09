#!/usr/bin/env python3
"""Compare probe_74_10x_loop_100_optimized_v2 (bitmask+A, no dedup, 360 steps,
succeeded) vs probe_74_dedup (current run, in progress).

Reports step-by-step:
  - Step time breakdown (P1, P2, P3, total)
  - batch / cands counts
  - best beam state (n_nm, max_weight)

Shows snapshot rows at step milestones AND summary differences.
"""
import re
import sys

A_path = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/probe_74_10x_loop_100_optimized_v2/probe.out'
B_path = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/probe_74_dedup/probe.out'

step_re = re.compile(
    r'^Step (\d+): P1\(get_valid\)=([\d\.]+)s P2\(model\)=([\d\.]+)s '
    r'P3\(apply\)=([\d\.]+)s total=([\d\.]+)s batch=(\d+) cands=(\d+)'
)
beam_re = re.compile(
    r'-> beam has (\d+) states, best has (\d+) non-masters, max_weight=\(([-\d]+),\s*([-\d]+)\)'
)

def parse(p):
    out = {}
    last_step = None
    for line in open(p):
        m = step_re.match(line)
        if m:
            s = int(m.group(1))
            last_step = s
            out[s] = {'p1': float(m.group(2)), 'p2': float(m.group(3)),
                      'p3': float(m.group(4)), 'total': float(m.group(5)),
                      'batch': int(m.group(6)), 'cands': int(m.group(7))}
            continue
        m = beam_re.search(line)
        if m and last_step is not None:
            out[last_step]['n_states'] = int(m.group(1))
            out[last_step]['best_nm'] = int(m.group(2))
            out[last_step]['mw'] = (int(m.group(3)), int(m.group(4)))
    return out

A = parse(A_path)
B = parse(B_path)
print(f'A (bitmask+A no dedup): {len(A)} steps logged, max step={max(A) if A else 0}')
print(f'B (NEW dedup, in prog): {len(B)} steps logged, max step={max(B) if B else 0}')
print()

# Step-by-step comparison at milestone steps
milestones = [0, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450]
print(f"{'step':>4s} | {'A:total':>8s} {'A:P1':>6s} {'A:P3':>6s} {'A:batch':>7s} {'A:nm':>5s} {'A:mw':>9s} | "
      f"{'B:total':>8s} {'B:P1':>6s} {'B:P3':>6s} {'B:batch':>7s} {'B:nm':>5s} {'B:mw':>9s} | "
      f"slowdown")
print('-' * 130)
for s in milestones:
    if s not in A and s not in B:
        continue
    aa = A.get(s, {})
    bb = B.get(s, {})
    a_str = f"{aa.get('total', 0):8.2f} {aa.get('p1', 0):6.2f} {aa.get('p3', 0):6.2f} "
    a_str += f"{aa.get('batch', 0):7d} {aa.get('best_nm', 0):5d} {str(aa.get('mw', '')):>9s}" if 'total' in aa else " " * 38
    b_str = f"{bb.get('total', 0):8.2f} {bb.get('p1', 0):6.2f} {bb.get('p3', 0):6.2f} "
    b_str += f"{bb.get('batch', 0):7d} {bb.get('best_nm', 0):5d} {str(bb.get('mw', '')):>9s}" if 'total' in bb else " " * 38
    if aa.get('total', 0) > 0 and bb.get('total', 0) > 0:
        slow = bb['total'] / aa['total']
        slow_str = f"{slow:.2f}x"
    else:
        slow_str = ""
    print(f'{s:>4d} | {a_str} | {b_str} | {slow_str}')

# Cumulative wall time at each step
print(f'\n{"step":>4s} | {"A_cumul":>8s} {"B_cumul":>8s} {"ratio":>6s}')
print('-' * 40)
a_cumul = 0; b_cumul = 0
for s in sorted(set(list(A) + list(B))):
    if s in A: a_cumul += A[s]['total']
    if s in B: b_cumul += B[s]['total']
    if s in milestones:
        ratio = b_cumul / a_cumul if a_cumul > 0 else 0
        print(f'{s:>4d} | {a_cumul:8.0f} {b_cumul:8.0f} {ratio:5.2f}x')
print(f'\nFinal A: {a_cumul:.0f}s ({a_cumul/60:.1f}min) at step {max(A)}')
print(f'Current B: {b_cumul:.0f}s ({b_cumul/60:.1f}min) at step {max(B)}')
