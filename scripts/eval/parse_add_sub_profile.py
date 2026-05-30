"""
Parse ADD_SUB_PROFILE lines from probe.out, correlate with Step boundaries,
emit per-step deltas across workers.
"""
import re
import sys
from collections import defaultdict


def parse(path):
    step_re = re.compile(r'^Step (\d+):.*total=([\d.]+)s')
    asp_re = re.compile(r'^ADD_SUB_PROFILE\t(.*)$')

    cur_step = 0
    cur_wall = 0.0
    cur_snapshot = {}
    prev_snapshot = {}
    rows = []

    def flush():
        totals = defaultdict(float)
        for pid, snap in cur_snapshot.items():
            prev = prev_snapshot.get(pid, {})
            for k, v in snap.items():
                totals[k] += v - prev.get(k, 0)
        rows.append({'step': cur_step, 'wall': cur_wall, **totals})
        for pid, snap in cur_snapshot.items():
            prev_snapshot[pid] = dict(snap)
        cur_snapshot.clear()

    with open(path) as f:
        for line in f:
            m_step = step_re.match(line)
            if m_step:
                if cur_step > 0:
                    flush()
                cur_step = int(m_step.group(1))
                cur_wall = float(m_step.group(2))
                continue
            m_asp = asp_re.match(line.rstrip('\n'))
            if m_asp:
                d = {}
                pid = None
                for p in m_asp.group(1).split('\t'):
                    if '=' not in p:
                        continue
                    k, v = p.split('=', 1)
                    if k == 'pid':
                        pid = int(v)
                    elif v.replace('.', '', 1).isdigit():
                        d[k] = float(v) if '.' in v else int(v)
                if pid is not None:
                    cur_snapshot[pid] = d

    if cur_step > 0:
        flush()
    return rows


def fmt(rows, every=10):
    print(f'{"step":>5} {"wall":>6} {"calls":>6} {"keys_it":>9} {"hits":>7} '
          f'{"hit%":>5} {"|sol|":>6} '
          f'{"t_total":>8} {"t_apply_rs":>11} {"%":>5} '
          f'{"t_outer":>8} {"%":>5} '
          f'{"t_sub_work":>11} {"%":>5}')
    for r in rows:
        if r['step'] % every != 0 and r['step'] not in (1, rows[-1]['step']):
            continue
        nc = r.get('n_calls', 0) or 1
        nk = r.get('n_keys_iterated', 0)
        nh = r.get('n_hits', 0)
        nil = r.get('n_inner_loop_iters', 0)
        tt = r.get('t_total', 0.001) or 0.001
        ta = r.get('t_apply_rs_at_entry', 0)
        to = r.get('t_outer_iter', 0)
        tsw = r.get('t_substitution_work', 0)
        hit_pct = 100*nh/max(1,nk)
        avg_inner = nil / max(1, nh)
        print(f"{r['step']:>5} {r['wall']:>6.2f} {int(nc):>6} {int(nk):>9} {int(nh):>7} "
              f"{hit_pct:>4.1f}% {avg_inner:>6.1f} "
              f"{tt:>8.3f} {ta:>11.3f} {100*ta/tt:>4.1f}% "
              f"{to:>8.3f} {100*to/tt:>4.1f}% "
              f"{tsw:>11.3f} {100*tsw/tt:>4.1f}%")


if __name__ == '__main__':
    rows = parse(sys.argv[1] if len(sys.argv) > 1
                 else '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/probe_84_prodprofile/probe.out')
    print(f'Total steps parsed: {len(rows)}')
    fmt(rows, every=15)
