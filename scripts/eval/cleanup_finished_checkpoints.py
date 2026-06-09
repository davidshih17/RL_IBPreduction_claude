"""One-time cleanup of leftover v6 worker checkpoints.

For each `*.pkl.checkpoint` file under <results_dir>:
  - if the matching `*.pkl` is on disk AND it loads AND `success` is True,
    delete the checkpoint
  - otherwise leave it alone (still-running worker, success=False worker
    we might want to inspect, or unreadable result.pkl).

Usage:
    python cleanup_finished_checkpoints.py <results_dir> [--dry-run]
"""
import argparse, os, pickle, sys, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results_dir')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    t0 = time.time()
    n_ckpt = n_deleted = n_skipped_running = n_skipped_failed = n_skipped_err = 0
    bytes_freed = 0

    for fn in os.listdir(args.results_dir):
        if not fn.endswith('.pkl.checkpoint'):
            continue
        n_ckpt += 1
        ck_path = os.path.join(args.results_dir, fn)
        parent_pkl = ck_path[:-len('.checkpoint')]
        if not os.path.exists(parent_pkl):
            n_skipped_running += 1
            continue
        try:
            with open(parent_pkl, 'rb') as fp:
                r = pickle.load(fp)
        except Exception:
            n_skipped_err += 1
            continue
        if not r.get('success'):
            n_skipped_failed += 1
            continue
        sz = os.path.getsize(ck_path)
        if not args.dry_run:
            try:
                os.remove(ck_path)
            except OSError:
                n_skipped_err += 1
                continue
        n_deleted += 1
        bytes_freed += sz
        # report progress every 1000 deletions
        if n_deleted % 1000 == 0:
            print(f'  ... {n_deleted} deleted ({bytes_freed/1e9:.1f} GB) in '
                  f'{time.time()-t0:.0f}s', flush=True)

    print()
    print(f'scanned    : {n_ckpt} checkpoint files in {time.time()-t0:.1f}s')
    print(f'deleted    : {n_deleted}  ({bytes_freed/1e9:.1f} GB freed'
          f'{" — dry-run, no actual deletes" if args.dry_run else ""})')
    print(f'skipped    :')
    print(f'  no result.pkl (worker still running/crashed): {n_skipped_running}')
    print(f'  result.pkl had success=False:                {n_skipped_failed}')
    print(f'  result.pkl unreadable / unlink error:        {n_skipped_err}')


if __name__ == '__main__':
    main()
