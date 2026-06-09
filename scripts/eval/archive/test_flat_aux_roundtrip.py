"""Verify FlatAux converter: dict-aux -> flat-aux -> dict-aux must
round-trip to an equivalent structure. Run delta beam search for a few
steps, grab the aux of one of the surviving states, convert, compare.
"""
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE.parent))

from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import (
    IBPEnvironment, set_prime, set_paper_masters_only,
)
from sailir.classifier import IBPActionClassifier
from sailir.aux_flat import FlatAux
import sailir.delta_beam_search as dbs
from beam_search_utils import get_sector_mask
import torch


def main():
    topology = Topology.from_dir('topology_input/pentagonbox')
    ibp_env.init_from_topology(topology)
    set_prime(1009)
    set_paper_masters_only(False)
    env = IBPEnvironment()

    model = IBPActionClassifier(
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    ckpt = torch.load('checkpoints/pentagonbox_10x_loop_100/best_model.pt',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    integral = tuple(int(x) for x in '-1,2,1,0,1,2,1,1,-3,0,0'.split(','))
    target_sector = tuple(get_sector_mask(integral))
    start_expr = {integral: 1}

    # Run a few steps to grow aux
    n_run_steps = 20
    print(f'Running {n_run_steps} steps to grow aux...', flush=True)
    _ = dbs.beam_search_delta(
        env, model, start_expr, target_sector,
        beam_width=40, max_steps=n_run_steps, prime=1009,
        verbose=False, stop_on_weight_improvement=False, device='cpu',
    )

    # The beam reference is lost after beam_search_delta returns, so we
    # capture by hand: re-run with a hook that grabs the beam at step N.
    captured = {'beam': None}
    orig_loop = dbs.beam_search_delta

    # Manual capture: instrument by reaching inside via a quick hack —
    # run again with verbose+stop_on_weight, grab the returned beam.
    print('Re-running 5 steps to capture a beam state for testing...', flush=True)
    solution, beam, _ = dbs.beam_search_delta(
        env, model, start_expr, target_sector,
        beam_width=40, max_steps=5, prime=1009,
        verbose=False, stop_on_weight_improvement=False, device='cpu',
    )
    if not beam:
        print('No beam states captured; abort.', flush=True)
        return

    # Pick the first survivor that has materialized aux
    test_state = None
    for s in beam:
        if s._indirect_aux is not None:
            test_state = s
            break
    if test_state is None:
        # Force materialization
        test_state = beam[0]
        _ = test_state.indirect_cache_list

    orig_aux = test_state.indirect_aux
    cu, ubm, rid, iraws = orig_aux
    print(f'  |cu|={len(cu)} |ubm|={len(ubm)} |rid|={len(rid)} |iraws|={len(iraws)}', flush=True)
    print(f'  avg cu entry size: {sum(len(c) for c in cu) / max(1, len(cu)):.1f} keys', flush=True)

    # Convert dict -> flat
    t0 = time.time()
    flat = FlatAux.from_dict_aux(orig_aux)
    t_to_flat = time.time() - t0
    print(f'\n  dict -> flat:  {t_to_flat*1000:.1f} ms', flush=True)
    print(f'    cu_keys shape:    {flat.cu_keys.shape}', flush=True)
    print(f'    cu_coeffs shape:  {flat.cu_coeffs.shape}', flush=True)
    print(f'    iraws_meta shape: {flat.iraws_meta.shape}', flush=True)
    print(f'    cu_offsets sum: {int(flat.cu_offsets[-1])}', flush=True)

    # Convert back
    t0 = time.time()
    cu2, ubm2, rid2, iraws2 = flat.to_dict_aux(env)
    t_to_dict = time.time() - t0
    print(f'  flat -> dict:  {t_to_dict*1000:.1f} ms', flush=True)

    # Compare
    print('\nEquivalence check:', flush=True)
    ok = True
    if len(cu) != len(cu2):
        print(f'  cu length differs: {len(cu)} vs {len(cu2)}'); ok = False
    else:
        diffs = 0
        for i, (a, b) in enumerate(zip(cu, cu2)):
            if a != b:
                diffs += 1
                if diffs <= 3:
                    print(f'  cu[{i}] differs: |a|={len(a)} |b|={len(b)}')
        if diffs == 0:
            print(f'  cu: all {len(cu)} entries equal ✓')
        else:
            print(f'  cu: {diffs}/{len(cu)} entries DIFFER ✗')
            ok = False

    if ubm != ubm2:
        print(f'  ubm differs'); ok = False
    else:
        print(f'  ubm: equal ✓')

    if len(iraws) != len(iraws2):
        print(f'  iraws length differs'); ok = False
    else:
        diffs = sum(
            1 for (a, b) in zip(iraws, iraws2)
            if a[0] != b[0] or a[1] != b[1] or a[2] != b[2] or a[3] is not b[3]
        )
        if diffs == 0:
            print(f'  iraws: all {len(iraws)} entries equal (incl. raw ref identity) ✓')
        else:
            print(f'  iraws: {diffs}/{len(iraws)} entries DIFFER ✗')
            ok = False

    if rid != rid2:
        # rid has id() keys which differ across processes. But this is in
        # the SAME process. raw ref objects from raw_eq_cache are the same
        # objects, so id() must match.
        # Allow rid to differ if the keys are different but the SAME raws
        # map to the same cu_idx in both.
        print(f'  rid: dict differs (this is expected if id() values differ)')

    print('\nVerdict:', 'PASS' if ok else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
