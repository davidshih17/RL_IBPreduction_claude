"""
Parse probe.out, correlate PROD_PROFILE lines with Step N boundaries,
emit per-step component breakdown.

Each step bracketed by two "Step N:" lines (or start-of-file / end-of-file).
Between them: zero or more PROD_PROFILE lines (one per worker batch). The
PROD_PROFILE values are per-worker cumulative — to get per-step delta we
take, for each pid, (cumulative at end-of-step) - (cumulative at end-of-
previous-step) and sum across pids.
"""
import re
import sys
from collections import defaultdict


def parse(path):
    step_re = re.compile(r'^Step (\d+):.*total=([\d.]+)s.*batch=(\d+) cands=(\d+)')
    pp_re = re.compile(r'^PROD_PROFILE\t(.*)$')

    # Walk the file; maintain current step. For each pid maintain cumulative.
    # At each Step line, snapshot cumulative-per-pid -> compute delta vs
    # previous snapshot -> sum across pids -> emit row for this step.
    cur_step = 0
    cur_total = 0.0
    cur_batch = 0
    cur_cands = 0
    cum_per_pid = {}      # pid -> dict of last seen cumulative values
    prev_snapshot = {}    # pid -> dict (cumulative at end of previous step)
    cur_snapshot = {}     # pid -> dict (cumulative within current step)

    rows = []

    def flush_step():
        # Sum per-pid (cur - prev) into per-step totals
        totals = defaultdict(float)
        nc = h = m = 0
        for pid, snap in cur_snapshot.items():
            prev = prev_snapshot.get(pid, {})
            for k, v in snap.items():
                if k in ('n_calls', 'hits', 'miss', 'success',
                          'fail_tgt0', 'fail_solN'):
                    totals[k] += v - prev.get(k, 0)
                elif k.startswith('t_'):
                    totals[k] += v - prev.get(k, 0.0)
        rows.append({
            'step': cur_step,
            'total_wall': cur_total,
            'batch': cur_batch,
            'cands': cur_cands,
            **totals,
        })
        # Roll forward
        for pid, snap in cur_snapshot.items():
            prev_snapshot[pid] = dict(snap)
        cur_snapshot.clear()

    with open(path) as f:
        for line in f:
            m_step = step_re.match(line)
            if m_step:
                if cur_step > 0:
                    flush_step()
                cur_step = int(m_step.group(1))
                cur_total = float(m_step.group(2))
                cur_batch = int(m_step.group(3))
                cur_cands = int(m_step.group(4))
                continue
            m_pp = pp_re.match(line.rstrip('\n'))
            if m_pp:
                parts = m_pp.group(1).split('\t')
                d = {}
                for p in parts:
                    if '=' not in p:
                        continue
                    k, v = p.split('=', 1)
                    if k in ('pid', 'batch_size'):
                        d[k] = int(v)
                    elif v.replace('.', '', 1).isdigit():
                        d[k] = float(v) if '.' in v else int(v)
                pid = d.get('pid')
                if pid is None:
                    continue
                # Cumulative values from this PROD_PROFILE — these REPLACE
                # not accumulate (the worker prints its own cumulative each batch)
                snap = {k: d[k] for k in d if k not in ('pid', 'batch_size')}
                cur_snapshot[pid] = snap

    if cur_step > 0:
        flush_step()
    return rows


def fmt(rows, every=10):
    print(f'{"step":>5} {"total":>7} {"batch":>5} {"cands":>5} '
          f'{"hits/calls":>11} {"miss%":>5} '
          f'{"t_add_sub":>9} {"%add":>6} '
          f'{"t_apply_sub":>11} {"t_apply_rs":>11} '
          f'{"t_sol_target":>12} {"t_solve":>8} '
          f'{"t_get_raw":>10} {"t_other":>8} {"sum_t":>7}')
    for r in rows:
        if r['step'] % every != 0 and r['step'] not in (1, rows[-1]['step']):
            continue
        nc = r.get('n_calls', 0)
        h = r.get('hits', 0); mm = r.get('miss', 0)
        hit_rate = h / max(1, h+mm)
        miss_pct = 100 * mm / max(1, h+mm)
        t_total = r.get('t_total', 0.0) or 0.001
        sub = r.get('t_add_sub', 0)
        pct_add = 100 * sub / t_total
        sum_t = (r.get('t_add_sub',0)+r.get('t_apply_sub',0)
                 +r.get('t_apply_rs',0)+r.get('t_sol_target',0)
                 +r.get('t_solve',0)+r.get('t_get_raw',0)
                 +r.get('t_new_subs',0)+r.get('t_seed',0))
        t_other = t_total - sum_t
        print(f"{r['step']:>5} {r['total_wall']:>7.2f} {r['batch']:>5} {r['cands']:>5} "
              f"{int(h):>5}/{int(nc):>5} {miss_pct:>4.1f}% "
              f"{sub:>9.3f} {pct_add:>5.1f}% "
              f"{r.get('t_apply_sub',0):>11.3f} {r.get('t_apply_rs',0):>11.3f} "
              f"{r.get('t_sol_target',0):>12.3f} {r.get('t_solve',0):>8.3f} "
              f"{r.get('t_get_raw',0):>10.3f} {t_other:>8.3f} {sum_t:>7.3f}")


if __name__ == '__main__':
    rows = parse(sys.argv[1] if len(sys.argv) > 1
                 else '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/probe_84_prodprofile/probe.out')
    print(f'Total steps parsed: {len(rows)}')
    fmt(rows, every=10)
