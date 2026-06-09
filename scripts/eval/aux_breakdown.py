"""Per-beam-slot breakdown of aux_flat to see if iraws-keep-first is actually
capping things. Aux is (cached_unique, union_bms, raw_id_to_idx, indirect_raws).
After _prune_aux_by_recency with keep_first=50, len(iraws) should be ~ 50, and
cached_unique / union_bms should be ~50.

Usage:
    python aux_breakdown.py <ckpt.pkl.checkpoint>
"""
import argparse, os, pickle, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_path')
    args = p.parse_args()
    with open(args.ckpt_path, 'rb') as fp:
        ck = pickle.load(fp)
    beam = ck['beam']
    print(f'step={ck.get("step")} beam_size={len(beam)}')
    print()
    print(f'{"slot":>4s} {"path":>6s} {"rs":>6s} {"sub_acc":>8s} '
          f'{"aux=None":>8s} {"iraws":>6s} {"cu_len":>6s} {"ubm_len":>7s} '
          f'{"cu_bytes":>10s}')
    for i, sd in enumerate(beam):
        n_path = len(sd.get('path', []))
        n_rs = len(sd.get('resolved_subs') or {})
        n_sa = len(sd.get('sub_accum') or {})
        aux = sd.get('aux_flat')
        if aux is None:
            print(f'{i:>4d} {n_path:>6d} {n_rs:>6d} {n_sa:>8d} '
                  f'{"y":>8s} {0:>6d} {0:>6d} {0:>7d} {0:>10d}')
            continue
        # picklable form is (cu, ubm, stable_rid, iraws, marker)
        if len(aux) == 5:
            cu, ubm, _rid, iraws, _ = aux
        else:
            cu, ubm, _rid, iraws = aux
        n_iraws = len(iraws)
        n_cu = len(cu)
        n_ubm = len(ubm)
        cu_bytes = len(pickle.dumps(cu, protocol=pickle.HIGHEST_PROTOCOL))
        print(f'{i:>4d} {n_path:>6d} {n_rs:>6d} {n_sa:>8d} '
              f'{"n":>8s} {n_iraws:>6d} {n_cu:>6d} {n_ubm:>7d} '
              f'{cu_bytes:>10d}')


if __name__ == '__main__':
    main()
