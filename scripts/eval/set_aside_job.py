"""Set aside a running round-2 integral for later inspection.

Preserves the worker's checkpoint (copied, not moved) into a set-aside dir, and
writes an identity result.pkl at the worker's output path so the orchestrator
caches it as a (failed) identity and will NOT re-submit it. The caller is
responsible for condor_rm-ing the actual job AFTER this runs.

Usage:
  python set_aside_job.py --integral 1,1,1,0,-2,1,1,1,0,0,0 \
      --output <work/results/async_NNNN_...pkl> \
      --setaside-dir <dir> --reason "memhog 21GB job 1652814"
"""
import argparse
import os
import pickle
import shutil
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--integral', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--setaside-dir', required=True)
    p.add_argument('--reason', default='')
    args = p.parse_args()

    ig = tuple(int(x) for x in args.integral.split(','))
    igstr = '_'.join(str(x) for x in ig)
    os.makedirs(args.setaside_dir, exist_ok=True)

    # 1. Preserve the checkpoint (copy, leave original in place).
    ckpt = args.output + '.checkpoint'
    saved_ckpt = None
    if os.path.exists(ckpt):
        saved_ckpt = os.path.join(args.setaside_dir, f'{igstr}.checkpoint')
        shutil.copy2(ckpt, saved_ckpt)
        print(f'preserved checkpoint -> {saved_ckpt} '
              f'({os.path.getsize(saved_ckpt)/1e6:.0f} MB)')
    else:
        print(f'WARNING: no checkpoint at {ckpt}')

    # 2. Write an identity result so the orchestrator sets it aside (skips it).
    result = {
        'original_integral': ig,
        'success': False,
        'final_expr': {ig: 1},
        'set_aside': True,
        'set_aside_reason': args.reason,
        'set_aside_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)
    print(f'wrote identity result -> {args.output}')

    # 3. Manifest line.
    with open(os.path.join(args.setaside_dir, 'manifest.txt'), 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\t{ig}\t{args.reason}\t'
                f'ckpt={saved_ckpt}\toutput={args.output}\n')
    print(f'manifest updated in {args.setaside_dir}')


if __name__ == '__main__':
    main()
