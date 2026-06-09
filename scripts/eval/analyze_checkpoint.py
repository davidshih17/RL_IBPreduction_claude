"""Break down a v6 worker checkpoint to identify which structure is bloating.

Loads <ckpt>.pkl.checkpoint and reports:
  - file size on disk vs in-memory pickled object byte-size
  - tabu_dict: total entries, per-expr breakdown
  - beam: per-slot expr size, path length, aux_flat size if present
  - per-beam-slot resolved_subs / rs / iraws sizes
  - dominant contributor to total bytes

Usage:
    python analyze_checkpoint.py <path_to.pkl.checkpoint>
"""
import argparse, os, pickle, sys
from collections import Counter


def deep_size_bytes(obj, seen=None):
    """Recursive object size — rough but useful for relative comparison.

    Uses pickle.dumps to estimate, since the on-disk size IS the pickled form.
    For per-slot breakdown we re-pickle each container; this is approximate
    (shared objects get double-counted) but good enough for ranking.
    """
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_path')
    p.add_argument('--per-slot', action='store_true',
                   help='Dump per-beam-slot details, not just totals.')
    args = p.parse_args()

    disk_size = os.path.getsize(args.ckpt_path)
    print(f'file: {args.ckpt_path}')
    print(f'on-disk size: {disk_size/1e6:.1f} MB')
    print()
    with open(args.ckpt_path, 'rb') as fp:
        ck = pickle.load(fp)
    print(f'top-level keys: {list(ck.keys())}')
    print(f'  step          : {ck.get("step")}')
    print(f'  target_sector : {ck.get("target_sector")}')
    print(f'  start_w12     : {ck.get("start_w12")}')
    print()

    # --- tabu_dict ---
    tabu = ck.get('tabu_dict')
    if tabu is None:
        print('tabu_dict: None')
    else:
        n_buckets = len(tabu)
        n_entries = sum(len(e[1]) for e in tabu)
        tabu_bytes = deep_size_bytes(tabu)
        print(f'tabu_dict:')
        print(f'  expr-fingerprint buckets: {n_buckets}')
        print(f'  total tabu actions:       {n_entries}')
        print(f'  pickled bytes:            {tabu_bytes/1e6:.1f} MB '
              f'({tabu_bytes/disk_size*100:.1f}% of file)')
        if n_buckets > 0:
            sizes = [len(e[1]) for e in tabu]
            sizes.sort(reverse=True)
            print(f'  top-10 bucket sizes:      {sizes[:10]}')
            print(f'  mean / median / max:      {sum(sizes)/len(sizes):.1f} / '
                  f'{sorted(sizes)[len(sizes)//2]} / {max(sizes)}')

    print()

    # --- beam ---
    beam = ck.get('beam', [])
    print(f'beam: {len(beam)} slots')
    beam_bytes = deep_size_bytes(beam)
    print(f'  pickled bytes total:      {beam_bytes/1e6:.1f} MB '
          f'({beam_bytes/disk_size*100:.1f}% of file)')

    # Per-slot subfield breakdown — sum over all slots
    subfield_totals = Counter()
    subfield_counts = Counter()
    sample_slot = None
    for i, sd in enumerate(beam):
        for k, v in sd.items():
            sz = deep_size_bytes(v)
            subfield_totals[k] += sz
            if isinstance(v, (list, tuple, dict, set)):
                subfield_counts[k] += len(v)
        if sample_slot is None:
            sample_slot = sd
    print()
    print('beam — per-field totals across all slots:')
    print(f'  {"field":<28s} {"bytes_MB":>10s} {"% of beam":>10s} '
          f'{"sum(len)":>10s}')
    for k in sorted(subfield_totals, key=lambda x: -subfield_totals[x]):
        print(f'  {k:<28s} {subfield_totals[k]/1e6:>10.1f} '
              f'{subfield_totals[k]/beam_bytes*100:>9.1f}% '
              f'{subfield_counts.get(k, 0):>10d}')

    if args.per_slot:
        print()
        print('per-slot detail:')
        for i, sd in enumerate(beam):
            print(f'\nslot {i}:')
            for k, v in sd.items():
                if isinstance(v, (list, tuple, dict, set)):
                    print(f'  {k:<28s} len={len(v):<8d} '
                          f'bytes={deep_size_bytes(v)/1e6:.2f} MB')
                else:
                    print(f'  {k:<28s} {repr(v)[:80]}')


if __name__ == '__main__':
    main()
