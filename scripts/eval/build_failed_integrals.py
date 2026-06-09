"""Fast classifier of v6 sweep failures. Output is a CSV of failed integrals
plus one per-category text file for round-2 retry dispatch.

Speed strategy (vs the slow v1 of this script):
  - Don't open *.log or *.out for every worker (those are big NFS files).
  - Use result.pkl PRESENCE as "did the worker complete its main()?"
  - Use condor_q's currently-queued integrals as "is it still running?"
  - For each crash victim (worker that ran but produced no result.pkl AND
    is no longer queued), open just its .err (small) to categorize.

Categories:
    OOM_PYTORCH             PyTorch CPU allocator died (Cannot allocate memory)
    OOM_OTHER               Some other memory failure (MemoryError, std::bad_alloc, …)
    OOM_SILENT_SIGKILL      .log shows "Abnormal termination (signal 9)" with no
                            allocation error in .err — OOM-killer caught it
                            before PyTorch could raise (typically at 4-6× the
                            memory request).
    TRACEBACK               Python traceback in .err that's not OOM
    KILLED                  Condor-side hold / SIGKILL pattern in .err
    TABU_TRAP_PROGRESS      v6 worker STUCK with best_nm >= 1 AND steps > 0.
                            Worker found at least one strict improvement over
                            start before tabu blocked all further actions.
    TABU_TRAP_NO_PROGRESS   v6 worker STUCK with best_nm >= 1 AND steps == 0.
                            No beam state ever beat the initial (max_w12,
                            n_non_masters); best_state stayed pinned to start.
                            Likely model failure: every action explored
                            locally degraded the cost function.
    SUCCESS_FALSE_OTHER     success=False but best_nm < 1 (shouldn't happen
                            normally — best_nm < 1 means "no non-masters" which
                            should be success=True).
    ORCH_KILL_ZOMBIE        .log shows "via condor_rm (by user dshih)" — the
                            orchestrator killed this worker (its integral was
                            substituted out of expr); not a real failure.
    NO_RESULT_NO_ERR        .err exists but has no diagnostic + no result.pkl
                            AND not condor_rm'd AND no SIGKILL.
    NO_RESULT_MISSING       neither .err nor result.pkl (likely test/non-worker)

Usage:
    python build_failed_integrals.py <sweep_root> [--out failed_integrals.csv]
"""
import argparse, os, pickle, re, subprocess, sys, time


def parse_integral_from_basename(base):
    """Take an 11-int tuple from the END of an 'async_<seq>_<11_ints>' name."""
    parts = base.split('_')
    nums = []
    for p in reversed(parts):
        try:
            int(p)
        except ValueError:
            break
        nums.append(p)
        if len(nums) >= 11:
            break
    if len(nums) < 11:
        return None
    nums = nums[:11][::-1]
    return tuple(int(x) for x in nums)


def classify_err(err_text):
    """Open .err contents → category, details. Used only for crash victims."""
    if 'Cannot allocate memory' in err_text or 'alloc_cpu.cpp' in err_text:
        return ('OOM_PYTORCH', 'PyTorch DefaultCPUAllocator alloc fail')
    for kw, label in [('MemoryError', 'MemoryError'),
                      ('OutOfMemory', 'OutOfMemory'),
                      ('std::bad_alloc', 'std::bad_alloc')]:
        if kw in err_text:
            return ('OOM_OTHER', label)
    if any(p in err_text for p in ('Killed', 'SIGKILL', 'Aborted')):
        return ('KILLED', 'kill signal pattern in .err')
    if 'Traceback (most recent call last)' in err_text:
        last = [l for l in err_text.splitlines() if l.strip()]
        return ('TRACEBACK', (last[-1] if last else 'unknown')[:160])
    if not err_text.strip():
        return ('NO_RESULT_NO_ERR', '.err is empty')
    # has SOMETHING but nothing matched — flag for manual inspection
    return ('NO_RESULT_NO_ERR',
            'crashed without recognised diagnostic; first line: '
            + err_text.splitlines()[0][:160])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_root')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    log_dir = os.path.join(args.sweep_root, 'work', 'logs')
    res_dir = os.path.join(args.sweep_root, 'work', 'results')
    out_path = args.out or os.path.join(args.sweep_root, 'failed_integrals.csv')

    t0 = time.time()

    # ── Read persistent manual-OOM kill list (set by kill_hog_workers.py) ──
    manual_oom_basenames = set()
    manual_oom_csv = os.path.join(args.sweep_root, 'manual_oom_killed.csv')
    if os.path.exists(manual_oom_csv):
        with open(manual_oom_csv) as fp:
            next(fp, None)  # header
            for line in fp:
                cols = line.rstrip('\n').split(',')
                if len(cols) >= 6:
                    manual_oom_basenames.add(cols[5])
        print(f'[{time.time()-t0:5.1f}s] manual-OOM kills on record: '
              f'{len(manual_oom_basenames)}', flush=True)

    # ── PASS 1: inventory log dir (.err / .out / .log basenames) ─────────
    print(f'[{time.time()-t0:5.1f}s] listdir({log_dir}) ...', flush=True)
    entries = os.listdir(log_dir)
    err_set = {fn[:-4] for fn in entries if fn.endswith('.err')}
    print(f'[{time.time()-t0:5.1f}s]   .err basenames: {len(err_set)}',
          flush=True)

    # ── PASS 2: result.pkl inventory ─────────────────────────────────────
    print(f'[{time.time()-t0:5.1f}s] listdir({res_dir}) ...', flush=True)
    res_entries = os.listdir(res_dir)
    result_set = {fn[:-4] for fn in res_entries
                  if fn.endswith('.pkl') and not fn.endswith('.checkpoint')}
    print(f'[{time.time()-t0:5.1f}s]   result.pkl basenames: {len(result_set)}',
          flush=True)

    # ── PASS 3: condor_q to find currently in-flight integrals ──────────
    print(f'[{time.time()-t0:5.1f}s] condor_q -af Args ...', flush=True)
    queued_integrals = set()
    try:
        cp = subprocess.run(
            ['condor_q', '-af', 'Args'],
            capture_output=True, text=True, timeout=60,
        )
        for line in cp.stdout.splitlines():
            m = re.search(r"--integral=['\"]?([\d,\-]+)", line)
            if m:
                queued_integrals.add(tuple(int(x) for x in m.group(1).split(',')))
    except Exception as e:
        print(f'  condor_q failed: {e}; treating queue as empty', flush=True)
    print(f'[{time.time()-t0:5.1f}s]   queued integrals: {len(queued_integrals)}',
          flush=True)

    # ── PASS 4: categorise every worker that has a .err ──────────────────
    # The .err is created at job-start, so its presence == "the cluster
    # actually started running on a node at some point".
    # Categories that are NOT real failures — accounted separately, not
    # written to failed_integrals.csv and not eligible for round-2 retry.
    NON_FAILURE_CATS = {'ORCH_KILL_ZOMBIE'}

    counts = {}                # real failures only
    non_failure_counts = {}    # ORCH_KILL_ZOMBIE etc — informational
    rows = []                  # failure rows (written to CSV)
    n_skipped_running = 0
    n_processed = 0

    def bump(cat, integ_str, details, base):
        if cat in NON_FAILURE_CATS:
            non_failure_counts[cat] = non_failure_counts.get(cat, 0) + 1
            return
        counts[cat] = counts.get(cat, 0) + 1
        rows.append((integ_str, cat, details, base))

    print(f'[{time.time()-t0:5.1f}s] classifying {len(err_set)} workers ...',
          flush=True)
    for base in err_set:
        n_processed += 1
        integ = parse_integral_from_basename(base)
        if integ is None:
            continue
        integ_str = ','.join(str(x) for x in integ)

        if base in result_set:
            # Finished workers — read result.pkl, only care about success=False
            pkl_path = os.path.join(res_dir, base + '.pkl')
            try:
                r = pickle.load(open(pkl_path, 'rb'))
            except Exception:
                bump('NO_RESULT_MISSING', integ_str,
                     f'result.pkl unreadable', base)
                continue
            if r.get('success'):
                continue   # not a failure
            best_mw = r.get('best_max_w12')
            best_nm = r.get('best_n_non_masters')
            steps = r.get('steps') or 0
            details = f'best_mw={best_mw} best_nm={best_nm} steps={steps}'
            # steps = len(best_state.path) — actions from start to best_state.
            # steps=0 means best_state == initial_state (no beam state ever
            # beat the start tuple). steps>0 means the worker made at least
            # one strict improvement before tabu blocked all further moves.
            if best_nm is not None and best_nm >= 1:
                if steps > 0:
                    bump('TABU_TRAP_PROGRESS', integ_str, details, base)
                else:
                    bump('TABU_TRAP_NO_PROGRESS', integ_str, details, base)
            else:
                bump('SUCCESS_FALSE_OTHER', integ_str, details, base)
            continue

        # No result.pkl — is it still running?
        if integ in queued_integrals:
            n_skipped_running += 1
            continue

        # Crash victim — check Condor .log first (it's authoritative for
        # signals/abort cause), then fall back to .err pattern matching.
        log_path = os.path.join(log_dir, base + '.log')
        log_text = ''
        try:
            with open(log_path) as fp:
                log_text = fp.read(32 * 1024)
        except Exception:
            pass
        if 'via condor_rm' in log_text:
            # If we recorded a manual OOM kill for this basename, that
            # explanation wins — the worker was over the memory budget and
            # we condor_rm'd it deliberately. Otherwise it's the
            # orchestrator's obsolete-cancel path.
            if base in manual_oom_basenames:
                bump('MANUAL_OOM', integ_str,
                     'condor_rm by user; worker exceeded memory budget',
                     base)
            else:
                bump('ORCH_KILL_ZOMBIE', integ_str,
                     'condor_rm by user dshih (orchestrator obsolete-cancel)',
                     base)
            continue
        if 'Abnormal termination (signal 9)' in log_text:
            # silent OOM-killer kill — .err typically has no allocation msg
            # because torch was SIGKILL'd before it could raise
            bump('OOM_SILENT_SIGKILL', integ_str,
                 'SIGKILL from OOM-killer; no diagnostic in .err', base)
            continue

        err_path = os.path.join(log_dir, base + '.err')
        try:
            with open(err_path) as fp:
                err_text = fp.read(64 * 1024)
        except Exception as e:
            bump('NO_RESULT_MISSING', integ_str, f'.err unreadable: {e}', base)
            continue
        cat, details = classify_err(err_text)
        bump(cat, integ_str, details, base)

        if n_processed % 5000 == 0:
            print(f'[{time.time()-t0:5.1f}s]   {n_processed}/{len(err_set)} '
                  f'classified, {sum(counts.values())} failures so far',
                  flush=True)

    # ── Report and write ─────────────────────────────────────────────────
    n_fail = sum(counts.values())
    print(f'\n[{time.time()-t0:5.1f}s] Failures by category ({n_fail} total):')
    for k in sorted(counts, key=lambda c: -counts[c]):
        print(f'  {k:22s}  {counts[k]}')
    print()
    print(f'[{time.time()-t0:5.1f}s] Non-failure events (NOT retried):')
    for k in sorted(non_failure_counts, key=lambda c: -non_failure_counts[c]):
        print(f'  {k:22s}  {non_failure_counts[k]}')
    print(f'  {"still running":22s}  {n_skipped_running}')

    print(f'\n[{time.time()-t0:5.1f}s] writing {len(rows)} rows -> {out_path}',
          flush=True)
    with open(out_path, 'w') as fp:
        fp.write('integral,category,details,log_basename\n')
        for integ, cat, det, base in rows:
            det_q = (det or '').replace(',', ';')
            fp.write(f'{integ},{cat},{det_q},{base}\n')

    by_cat_dir = os.path.dirname(out_path) or '.'
    for cat in counts:
        per_cat = os.path.join(by_cat_dir, f'failed_integrals_{cat}.txt')
        with open(per_cat, 'w') as fp:
            for integ, c, _, _ in rows:
                if c == cat:
                    fp.write(integ + '\n')
        print(f'  per-category list: {per_cat}  ({counts[cat]})')

    print(f'\n[{time.time()-t0:5.1f}s] done')


if __name__ == '__main__':
    main()
