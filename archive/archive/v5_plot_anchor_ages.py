#!/usr/bin/env python
"""Plot Phase 1b anchor age vs step number for the v5 tabu drain trajectory.

Reuses v5_study_action_anchors.py's classifier and replays the path.
"""
import argparse
import pickle
import sys

sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, solve_ibp_for, weight,
)
from beam_search_v5 import (
    apply_substitution_v5, add_sub_to_resolved_v5, State_v5,
)
from beam_search_utils import get_non_masters, get_sector_mask
from v5_study_action_anchors import classify_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True)
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--output', required=True, help='Output PNG path')
    p.add_argument('--prime', type=int, default=1009)
    args = p.parse_args()

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    set_prime(args.prime)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    start_int = tuple(int(x.strip("'\"")) for x in args.integral.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    target_sector = tuple(get_sector_mask(start_int))

    with open(args.result, 'rb') as f:
        d = pickle.load(f)
    if d.get('best_state') and d['best_state'].get('path'):
        path = d['best_state']['path']
        path_label = "best_state"
    else:
        path = d['beam'][0]['path']
        path_label = "beam[0]"
    print(f'Analyzing {path_label} path (len {len(path)})')

    n_idx = ibp_env.N_INDICES
    state = State_v5(
        expr={start_int: 1}, resolved_subs={}, sub_accum={},
        score=0.0, path=[], n_non_masters=1,
        max_w12=start_w12, total_w12=start_w12,
    )

    p1a_steps = []
    p1b_steps = []
    p1b_ages = []
    p1aorb_steps = []
    p1aorb_ages = []

    for step, (target, ibp_op, delta) in enumerate(path):
        rs_keys = list(state.resolved_subs.keys())
        kind, age = classify_action(target, ibp_op, delta, rs_keys, env, n_idx)
        s_no = step + 1
        if kind == 'p1a':
            p1a_steps.append(s_no)
        elif kind == 'p1b':
            p1b_steps.append(s_no)
            p1b_ages.append(age)
        elif kind == 'p1a_or_p1b':
            p1aorb_steps.append(s_no)
            p1aorb_ages.append(age)

        # advance state
        seed = tuple(target[i] + delta[i] for i in range(n_idx))
        raw = env.get_raw_equation_cached(ibp_op, seed)
        cached = apply_resolved_subs(raw, state.resolved_subs)
        if target not in cached or cached[target] == 0:
            break
        sol = solve_ibp_for(cached, target)
        new_expr, new_sub_accum = apply_substitution_v5(
            state.expr, state.sub_accum, target, sol, target_sector, start_w12,
        )
        new_rs = add_sub_to_resolved_v5(
            state.resolved_subs, target, sol, start_w12,
        )
        nm = get_non_masters(new_expr, target_sector)
        if nm:
            wl = [(weight(k)[0], weight(k)[1]) for k in nm]
            mw = max(wl); tw = (sum(w[0] for w in wl), sum(w[1] for w in wl))
        else:
            mw, tw = (0, 0), (0, 0)
        state = State_v5(
            expr=new_expr, resolved_subs=new_rs, sub_accum=new_sub_accum,
            score=0.0, path=[], n_non_masters=len(nm),
            max_w12=mw, total_w12=tw,
        )

    # Save raw data, then plot via subprocess using a python that has matplotlib.
    import json, os, subprocess
    data = {
        'path_label': path_label, 'path_len': len(path),
        'p1a_steps': p1a_steps,
        'p1b_steps': p1b_steps, 'p1b_ages': p1b_ages,
        'p1aorb_steps': p1aorb_steps, 'p1aorb_ages': p1aorb_ages,
    }
    json_path = args.output.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print(f'Wrote raw data: {json_path}')
    plot_cmd = ['/het/p4/dshih/conda_envs/pyg4/bin/python', '-c', f'''
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("{json_path}"))
fig, ax = plt.subplots(figsize=(11, 6))
if d["p1b_steps"]:
    ax.scatter(d["p1b_steps"], d["p1b_ages"], c="red", s=40,
               label="Phase 1b only", zorder=3, edgecolors="darkred")
if d["p1aorb_steps"]:
    ax.scatter(d["p1aorb_steps"], d["p1aorb_ages"], c="lightcoral", s=15,
               alpha=0.5, label="Phase 1a or 1b (1b age shown)", zorder=2)
if d["p1a_steps"]:
    ax.scatter(d["p1a_steps"], [0]*len(d["p1a_steps"]), c="blue", marker="|",
               s=80, alpha=0.4, label="Phase 1a only", zorder=1)
ax.axhline(50, color="green", linestyle="--", alpha=0.7, label="window=50")
ax.axhline(200, color="orange", linestyle="--", alpha=0.7, label="window=200")
all_s = d["p1b_steps"] + d["p1aorb_steps"] + d["p1a_steps"]
m = max(all_s) if all_s else 1
ax.plot([1, m], [0, m-1], "k:", alpha=0.4, label="age = step-1")
ax.set_xlabel("Step number")
ax.set_ylabel("Anchor age (steps since anchor sub_int added to RS)")
ax.set_title(f"v5 tabu drain ({{d[\\"path_label\\"]}}, {{d[\\"path_len\\"]}}-step path on (8,4))")
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, m + 5)
plt.tight_layout()
plt.savefig("{args.output}", dpi=100)
print("Wrote {args.output}")
''']
    subprocess.run(plot_cmd, check=True)
    return
    fig, ax = plt.subplots(figsize=(11, 6))
    # Phase 1b-only actions: scatter age vs step
    if p1b_steps:
        ax.scatter(p1b_steps, p1b_ages, c='red', s=40, label='Phase 1b only',
                   zorder=3, edgecolors='darkred')
    # Phase 1a-or-1b: also has a 1b anchor age
    if p1aorb_steps:
        ax.scatter(p1aorb_steps, p1aorb_ages, c='lightcoral', s=15,
                   alpha=0.5, label='Phase 1a or 1b (1b age plotted)',
                   zorder=2)
    # Phase 1a only: plot at age=0 on a secondary marker
    if p1a_steps:
        ax.scatter(p1a_steps, [0]*len(p1a_steps), c='blue', marker='|', s=80,
                   alpha=0.4, label='Phase 1a only (anchor = target)',
                   zorder=1)
    # Reference lines: window thresholds and "age = step" (anchor at step 1)
    ax.axhline(50, color='green', linestyle='--', alpha=0.7,
               label='window=50 cutoff')
    ax.axhline(200, color='orange', linestyle='--', alpha=0.7,
               label='window=200 cutoff')
    # Diagonal "age = step - 1" means anchor was added at step 1
    max_step = max([s for s in p1b_steps + p1aorb_steps + p1a_steps] + [1])
    ax.plot([1, max_step], [0, max_step - 1], 'k:', alpha=0.4,
            label='age = step-1 (anchor = step 1)')

    ax.set_xlabel('Step number')
    ax.set_ylabel('Anchor age (steps since anchor sub_int added to RS)')
    ax.set_title(f'v5 tabu drain ({path_label}, {len(path)}-step path on (8,4))\n'
                 f'Anchor age of each chosen action')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_step + 5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=100)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    sys.exit(main())
