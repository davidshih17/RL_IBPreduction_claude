"""Meta-orchestrator: cascade SAILIR reductions down a weight-ordered target list.

Goal: industrialize the reduction of the full list_TA (996 ISP-valid targets) so
it runs unattended instead of being driven by hand one integral at a time.

Policy
------
- Process targets top-to-bottom in `list_TA_ispclean_by_weight` (heaviest first).
- Launch the next target's reduction ONLY when every currently-active run has
  fanned down to its LONG-POLE phase: no managed run has >= NM_THRESHOLD remaining
  workers, held STABLE_CHECKS consecutive polls (so a between-waves dip doesn't
  trigger early). A COOLDOWN after each launch lets the fresh fan-out register.
- Before each launch, MERGE the caches of ALL prior reductions (completed AND
  in-progress) into the new run's --resume-from, so every reduction stands on
  everything computed so far (caches cascade; lower-weight runs get cheaper).
- MAX_ACTIVE caps simultaneously-active orchestrators (each is a ~0.6 GB login
  poller) so long-pole tails can't pile up without bound. Raise it for more
  parallelism.

Each run uses the SAME production settings as the manual runs (pentagonbox +
--no-paper-masters-only, v7-cpus 1, max-concurrent 1000), in its own dir under
META_ROOT. Restart-safe: re-discovers its own launched runs via target.txt.

Run it in the background; everything streams to its log. It launches/monitors
only -- it never kills anything.
"""
import os
import re
import subprocess
import time
from pathlib import Path

BASE = Path('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
PY = '/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python'
MODEL = BASE / 'checkpoints/pentagonbox_10x_loop_100/best_model.pt'
TOPOLOGY = BASE / 'topology_input/pentagonbox'
TARGET_LIST = BASE / 'from_federica/list_TA_ispclean_by_weight'
META_ROOT = BASE / 'results/meta_reduce'
MERGE = BASE / 'reduction/merge_caches.py'
HIER = BASE / 'reduction/hierarchical_reduction.py'

NM_THRESHOLD = 100      # "long-pole": a run with < this many non-masters (its OWN
                        # reported frontier) is winding down. This is the gating
                        # signal -- NOT the condor worker count, which momentarily
                        # reads 0 during a heavy run's in-process compute gaps.
STABLE_CHECKS = 1       # quiet polls required before launching (the masters>0
                        # guard already prevents start-phase false positives)
INTERVAL = 30           # seconds between polls
COOLDOWN = 0            # post-launch wait. 0: bursts self-pace via the masters>0
                        # guard (fresh runs read 'busy' until they fan out)
MAX_ACTIVE = 100        # cap on simultaneously-active orchestrators (~0.75 GB each)
LAUNCH_BATCH = 8        # max runs to launch per cycle. Low-weight reductions are
                        # small + finish fast, so one-at-a-time can't keep the
                        # cluster fed; launch a burst toward MAX_ACTIVE instead.
FREE_CPU_FLOOR = 100    # keep launching while the pool has > this many free CPUs
                        # (cluster-room gate). Leaves headroom for other users +
                        # turnover; MAX_ACTIVE is the other self-limit.
MAX_CONCURRENT = 1000   # per-orchestrator worker cap

# Runs launched OUTSIDE the meta whose caches we gather and whose targets we
# skip: (run_dir, target_integral).
PREEXISTING = [
    (BASE / 'results/pentagonbox_8_5_v7_fresh',         (1, 1, 1, 1, 1, 1, 1, 1, -5, 0, 0)),
    (BASE / 'results/pentagonbox_7_5_cache_from_85',    (1, 1, 1, 1, 1, 1, 1, -1, -3, -1, 0)),
    (BASE / 'results/pentagonbox_7_6_cache_from_85_75', (1, 1, 1, 1, 1, 1, 1, -1, -3, -2, 0)),
]


def log(m):
    print(f"[meta {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def weight(t):
    return (sum(x for x in t if x > 0), -sum(x for x in t if x < 0))


def integ_str(t):
    return ','.join(str(x) for x in t)


def parse_targets(p):
    out = []
    for ln in open(p):
        m = re.match(r'TA\[([0-9,\-]+)\]', ln.strip())
        if m:
            out.append(tuple(int(x) for x in m.group(1).split(',')))
    return out


def parse_keys_from_results(results_dir):
    """Cheap cache-KEY set from a run's work/results: each completed worker
    pickle's filename encodes its integral (async_<id>_<11 ints>.pkl). No pickle
    loads -- just os.listdir, so it's fine to call every poll on a live dir."""
    keys = set()
    try:
        names = os.listdir(results_dir)
    except FileNotFoundError:
        return keys
    for name in names:
        if name.startswith('async_') and name.endswith('.pkl'):
            try:
                ints = tuple(int(x) for x in name[:-4].split('_')[-11:])
            except ValueError:
                continue
            if len(ints) == 11:
                keys.add(ints)
    return keys


def load_run_keys(rundir):
    """All cache keys a FINISHED run contributes: its snapshot replay_state.pkl
    if present (one fast load), else its work/results filenames."""
    rs = rundir / 'replay_state.pkl'
    if rs.exists():
        import pickle
        with open(rs, 'rb') as f:
            st = pickle.load(f)
        cache = st['cache'] if (isinstance(st, dict) and 'cache' in st) else st
        return set(cache.keys())
    return parse_keys_from_results(rundir / 'work' / 'results')


def run_nonmasters(rundir):
    """A run's remaining-work signal for long-pole detection, from its OWN log
    ('[Iter N] X masters, Y non-masters'). Returns Y (the frontier) -- EXCEPT it
    returns a big sentinel ('busy') while the run is still on its START integral.

    The start phase (a single heavy one-step reduction that hasn't fanned out
    yet) shows '0 masters, 1 non-master' -- which is small, but is NOT winding
    down. Gating on Y alone mistook that for long-pole and launched the next
    target prematurely. A run is only 'winding down' once it has produced
    masters (fanned out past its start) AND its frontier is small, so require
    masters > 0. Reliable, unlike the condor worker count (reads 0 in gaps)."""
    log_path = rundir / 'logs' / 'hierarchical.log'
    try:
        with open(log_path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode('utf-8', 'replace')
    except FileNotFoundError:
        return 10 ** 9
    masters = nm = None
    for ln in tail.splitlines():
        m = re.search(r'\] (\d+) masters, (\d+) non-masters', ln)
        if m:
            masters, nm = int(m.group(1)), int(m.group(2))
    if nm is None:
        return 10 ** 9          # no status line yet -> busy
    if masters == 0:
        return 10 ** 9          # still reducing the START integral -> busy
    return nm


def free_cpus():
    """Unclaimed CPUs in the Condor pool -- room for more workers."""
    try:
        out = subprocess.run(['condor_status', '-af', 'State', 'Cpus'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        log(f"condor_status failed: {e}")
        return 0
    n = 0
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 2 and p[0] == 'Unclaimed':
            try:
                n += int(p[1])
            except ValueError:
                pass
    return n


def worker_counts():
    """{run_dir_name: n_workers} from condor_q (dir just before /work/results/)."""
    try:
        out = subprocess.run(['condor_q', '-nobatch'],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        log(f"condor_q failed: {e}")
        return None
    counts = {}
    for ln in out.splitlines():
        if 'onestep_worker_v7' not in ln:
            continue
        m = re.search(r'/([^/ ]+)/work/results/', ln)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def merge(sources, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, '-u', str(MERGE), str(out_dir)] + [str(s) for s in sources]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / 'merge.log').write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        log(f"merge FAILED rc={r.returncode}; see {out_dir/'merge.log'}")
    return r.returncode == 0


def snapshot(rundir):
    """Freeze a finished run's work/results into rundir/replay_state.pkl so future
    merges load one file instead of re-scanning thousands of pickles. Idempotent."""
    if (rundir / 'replay_state.pkl').exists():
        return
    log(f"snapshot {rundir.name} -> replay_state.pkl")
    merge([rundir / 'work' / 'results'], rundir)


def is_done(rundir):
    return (rundir / 'reduction.pkl').exists() or (rundir / 'replay_state.pkl').exists()


def launch(t, rundir, cache_dir, rank):
    for sub in ['logs', 'work/logs', 'work/results']:
        (rundir / sub).mkdir(parents=True, exist_ok=True)
    (rundir / 'target.txt').write_text(f"{rank} {integ_str(t)}\n")
    cmd = [PY, '-u', str(HIER),
           '--topology', str(TOPOLOGY), f'--integral={integ_str(t)}',
           '--output', str(rundir / 'reduction.pkl'),
           '--work-dir', str(rundir / 'work'),
           '--resume-from', str(cache_dir),
           '--model-checkpoint', str(MODEL),
           '--beam_width', '40', '--max_steps', '1000000', '--prime', '1009',
           '--no-paper-masters-only', '--use-v7-worker', '--v7-cpus', '1',
           '--worker-memory-gb', '4',
           '--straggler-timeout', '1000000000',
           '--straggler2-timeout', '1000000000',
           '--check-interval', '5', '--max-concurrent', str(MAX_CONCURRENT),
           '--resume']
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    logf = open(rundir / 'logs/hierarchical.log', 'w')
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)


def main():
    META_ROOT.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(TARGET_LIST)
    log(f"{len(targets)} targets in {TARGET_LIST.name}")

    # managed = every run whose cache we gather: pre-existing + meta-launched.
    managed = [{'target': t, 'rundir': rd, 'done': is_done(rd)} for rd, t in PREEXISTING]
    launched_targets = {t for _, t in PREEXISTING}

    # restart safety: re-adopt meta runs we launched in a previous incarnation.
    for marker in sorted(META_ROOT.glob('tgt*/target.txt')):
        rd = marker.parent
        parts = marker.read_text().split()
        t = tuple(int(x) for x in parts[1].split(','))
        if t not in launched_targets:
            managed.append({'target': t, 'rundir': rd, 'done': is_done(rd)})
            launched_targets.add(t)
            log(f"re-adopted {rd.name} ({integ_str(t)}) done={managed[-1]['done']}")

    queue = [(rank, t) for rank, t in enumerate(targets, 1)
             if t not in launched_targets]
    log(f"{len(launched_targets)} already handled; {len(queue)} queued; "
        f"NM_THRESHOLD={NM_THRESHOLD} STABLE={STABLE_CHECKS} MAX_ACTIVE={MAX_ACTIVE}")

    stability = 0
    cooldown_until = 0.0

    # coverage tracking: targets whose reduction the accumulated cache already
    # resolves (a direct cache key => trivially reduces, no dedicated run needed).
    covered_by_cache = []                       # [(rank, target)]
    done_keys = set()                           # cumulative keys of FINISHED runs
    for m in managed:
        if m['done']:
            done_keys |= load_run_keys(m['rundir'])
    REQ_FILE = META_ROOT / 'require_reduction.txt'
    COV_FILE = META_ROOT / 'covered_by_cache.txt'

    while True:
        counts = worker_counts()
        if counts is None:
            time.sleep(INTERVAL)
            continue

        for m in managed:                       # mark + snapshot newly finished
            if not m['done'] and is_done(m['rundir']):
                m['done'] = True
                snapshot(m['rundir'])
                done_keys |= load_run_keys(m['rundir'])

        active = [m for m in managed if not m['done']]

        # Use the UPDATED cache (finished runs + the in-progress runs' completed
        # workers) to split the remaining queue into "trivially cached" (a direct
        # cache key -> reduces by substitution alone) vs "still require reduction".
        # Every run reduces the FULL subtree of everything it touches, so a target
        # that is a key today is guaranteed fully reducible once all runs finish;
        # drop it from the launch queue and record it as cache-covered.
        in_progress_keys = set()
        for m in active:
            in_progress_keys |= parse_keys_from_results(m['rundir'] / 'work' / 'results')
        all_keys = done_keys | in_progress_keys
        newly = [(rk, t) for (rk, t) in queue if t in all_keys]
        if newly:
            cset = {t for _, t in newly}
            queue[:] = [(rk, t) for (rk, t) in queue if t not in cset]
            covered_by_cache.extend(newly)
            REQ_FILE.write_text(''.join(f"{rk}\tTA[{integ_str(t)}]\n" for rk, t in queue))
            COV_FILE.write_text(''.join(f"{rk}\tTA[{integ_str(t)}]\n"
                                        for rk, t in sorted(covered_by_cache)))
            log(f"COVERAGE +{len(newly)}: {len(covered_by_cache)} of {len(targets)} "
                f"targets now trivially cached "
                f"({100.0 * len(covered_by_cache) / len(targets):.1f}%); "
                f"{len(queue)} still require reduction "
                f"(cache keys: {len(all_keys)})")

        # Gate on CLUSTER ROOM, not per-run winddown: at the low-weight tail each
        # reduction is tiny, so winddown-gating starves the cluster. Keep filling
        # concurrent reductions while the pool has free CPUs, up to MAX_ACTIVE.
        # (Self-limits: MAX_ACTIVE + the free-CPU floor; batched submits keep the
        # schedd calm even at high concurrency.)
        free = free_cpus()
        roomy = free > FREE_CPU_FLOOR
        log(f"active={len(active)} queue={len(queue)} "
            f"covered={len(covered_by_cache)}"
            f"({100.0 * len(covered_by_cache) / len(targets):.0f}%) "
            f"workers={sum(counts.values())} free_cpus={free} "
            f"{'ROOM' if roomy else 'cluster FULL'}")

        if queue and len(active) < MAX_ACTIVE and roomy:
            # Merge the accumulated cache ONCE this cycle -- the runs we launch
            # now add nothing to it yet, so the whole burst resumes from the same
            # snapshot (avoids re-merging the same ~170k-entry cache N times).
            shared = META_ROOT / '_burst_cache'
            sources = [m['rundir'] for m in managed]
            log(f"--- BURST: merging {len(sources)} caches into _burst_cache ---")
            if merge(sources, shared):
                n_active = len(active)
                launched_now = 0
                while queue and n_active < MAX_ACTIVE and launched_now < LAUNCH_BATCH:
                    rank, t = queue.pop(0)
                    r, s = weight(t)
                    rundir = META_ROOT / f"tgt{rank:04d}_w{r}_{s}"
                    p = launch(t, rundir, shared, rank)
                    managed.append({'target': t, 'rundir': rundir, 'done': False})
                    launched_targets.add(t)
                    n_active += 1
                    launched_now += 1
                    log(f"--- LAUNCHED rank {rank} PID={p.pid} dir={rundir.name} "
                        f"(active~{n_active}) ---")
                if launched_now:
                    stability = 0
                    cooldown_until = time.time() + COOLDOWN

        if not queue and not active:
            log("ALL DONE: queue empty, no active runs.")
            break
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
