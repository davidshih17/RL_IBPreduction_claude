#!/usr/bin/env python3
"""Parse orchestrator hierarchical.log status lines and plot trajectories.

Reads env vars LOG, CSV, PNG. Writes CSV + PNG.
"""
import os
import re
import sys

log = os.environ['LOG']
csv = os.environ['CSV']
png = os.environ['PNG']

# Match status lines of either format:
#   OLD: "[Iter N] M masters, K non-masters (L7:20 L6:475 ...) | Pending: P | Cache: C | Hits: H"
#   NEW: "[Iter N] M masters, K non-masters | frontier=... | work=... | Pending: P | Cache: C | Hits: H"
pat = re.compile(
    r'\[Iter\s+(\d+)\]\s+(\d+)\s+masters,\s+(\d+)\s+non-masters'
    r'[^|]*\|.*?Pending:\s+(\d+)\s+\|\s+Cache:\s+(\d+)\s+\|\s+Hits:\s+(\d+)'
)

rows = []
with open(log) as f:
    for line in f:
        m = pat.search(line)
        if m:
            it, mas, nm, pen, cac, hit = map(int, m.groups())
            rows.append((it, mas, nm, pen, cac, hit))

print(f'Parsed {len(rows)} status lines from {log}')
if not rows:
    sys.exit('No status lines matched.')

with open(csv, 'w') as f:
    f.write('iter,masters,non_masters,pending,cache,hits\n')
    for r in rows:
        f.write(','.join(str(x) for x in r) + '\n')
print(f'CSV: {csv}')

nms = [r[2] for r in rows]
print(f'Iter range: {rows[0][0]} .. {rows[-1][0]}')
print(f'Non-masters: start={nms[0]} max={max(nms)} (iter {rows[nms.index(max(nms))][0]}) '
      f'min={min(nms)} latest={nms[-1]}')
print(f'Masters: start={rows[0][1]} latest={rows[-1][1]}')
print(f'Cache: start={rows[0][4]} latest={rows[-1][4]}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
iters = [r[0] for r in rows]

axes[0].plot(iters, [r[2] for r in rows], label='non-masters', color='C0')
axes[0].plot(iters, [r[3] for r in rows], label='pending', color='C1', alpha=0.6)
axes[0].set_ylabel('count')
axes[0].set_title(f'{os.path.basename(log)} — non-masters vs iter (n_rows={len(rows)})')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, [r[1] for r in rows], label='masters', color='C2')
axes[1].plot(iters, [r[4] for r in rows], label='cache', color='C3', alpha=0.7)
axes[1].plot(iters, [r[5] for r in rows], label='hits', color='C4', alpha=0.5)
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('count')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(png, dpi=110)
print(f'PNG: {png}')
