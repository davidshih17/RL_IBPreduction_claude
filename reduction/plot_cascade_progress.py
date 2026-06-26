"""Plot the cascade's progress from the meta-orchestrator logs (current +
archived): 'still require reduction' (queue) and 'covered by cache' vs time,
plus the per-window completion RATE so we can see if it's steady, accelerating,
or decelerating.

meta status lines look like:
  [meta 2026-06-24 13:54:22] max_nm=62 active=20 queue=617 covered=337(34%) workers=350 | quiet 1/3
"""
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LOGDIR = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/meta_reduce')
TOTAL = 996
PAT = re.compile(r'\[meta (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?queue=(\d+).*?covered=(\d+)')

# gather (time -> (queue, covered)); dedup identical timestamps keeping last
pts = {}
for f in sorted(glob.glob(str(LOGDIR / 'meta*.log'))):
    for ln in open(f):
        m = PAT.search(ln)
        if m:
            t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            pts[t] = (int(m.group(2)), int(m.group(3)))

times = sorted(pts)
# Focus on the continuous active cascade: drop everything before the last >1h
# gap (idle/debugging stretches when the meta wasn't running the cascade).
cut = 0
for i in range(1, len(times)):
    if (times[i] - times[i - 1]).total_seconds() > 3600:
        cut = i
times = times[cut:]
queue = [pts[t][0] for t in times]
covered = [pts[t][1] for t in times]
# handled = launched-or-done = TOTAL - queue - (covered counted separately?).
# 'covered' is a subset already removed from queue, so launched/done = TOTAL-queue-covered.
launched = [TOTAL - q - c for q, c in zip(queue, covered)]

print(f"{len(times)} status points, {times[0]} -> {times[-1]}")
span_h = (times[-1] - times[0]).total_seconds() / 3600
print(f"span {span_h:.2f} h | require-reduction {queue[0]} -> {queue[-1]} "
      f"(down {queue[0]-queue[-1]}) | covered {covered[0]} -> {covered[-1]}")

# completion rate: targets leaving the queue per 15-min window
WIN = timedelta(minutes=15)
t0 = times[0]
binq = {}   # window index -> (queue at start, queue at end)
for t, q in zip(times, queue):
    k = int((t - t0) / WIN)
    if k not in binq:
        binq[k] = [q, q]
    binq[k][1] = q
rate_t, rate_v = [], []
for k in sorted(binq):
    drop = binq[k][0] - binq[k][1]               # targets removed in this window
    rate_t.append(t0 + k * WIN + WIN / 2)
    rate_v.append(drop * 4)                       # per 15 min -> per hour

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
ax1.plot(times, queue, '-', color='C3', lw=2, label='still require reduction')
ax1.plot(times, launched, '-', color='C0', lw=1.5, label='launched / done')
ax1.plot(times, covered, '-', color='C2', lw=1.5, label='covered by cache')
ax1.set_ylabel(f'# targets (of {TOTAL})')
ax1.legend(loc='center left')
ax1.grid(True, alpha=0.3)
ax1.set_title('SAILIR list_TA cascade progress (from meta_reduce/meta*.log)')

ax2.bar(rate_t, rate_v, width=WIN.total_seconds() / 86400 * 0.9,
        color='C3', alpha=0.7)
ax2.set_ylabel('targets removed\nfrom queue (/hour)')
ax2.set_xlabel('time')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig.autofmt_xdate()
fig.tight_layout()
out = LOGDIR / 'cascade_progress.png'
fig.savefig(out, dpi=120)
print(f"saved {out}")

# overall vs recent-hour rate (steady/accel/decel signal)
def rate_between(t_lo):
    sel = [(t, q) for t, q in zip(times, queue) if t >= t_lo]
    if len(sel) < 2:
        return None
    dt = (sel[-1][0] - sel[0][0]).total_seconds() / 3600
    return (sel[0][1] - sel[-1][1]) / dt if dt > 0 else None

overall = (queue[0] - queue[-1]) / span_h if span_h else 0
last1h = rate_between(times[-1] - timedelta(hours=1))
last30 = rate_between(times[-1] - timedelta(minutes=30))
print(f"\nrate (targets/hour leaving queue):")
print(f"  overall     : {overall:.1f}")
print(f"  last 1 hour : {last1h:.1f}" if last1h else "  last 1 hour : n/a")
print(f"  last 30 min : {last30:.1f}" if last30 else "  last 30 min : n/a")
