"""Compare scratch step6 aux (saved picklable) vs the same file loaded via
_aux_from_picklable into an env. Are they EQUIVALENT in content?

scratch saves ckpt.pkl.step0006 with aux already picklable-converted.
This script loads it twice:
  (a) raw load: just unpickle; aux is in stable-picklable form
  (b) env-load: depickle then _aux_from_picklable(aux, env=env)

Then we check: for the same state i, do cu/ubm have the same content? do
iraws have the same (sub_int, op, shift, seed=sub_int-shift) tuples? does
rid map each iraws entry to the same cu index?
"""
import pickle, sys, os
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/scripts/eval')
from sailir.ibp_env import (set_prime, set_paper_masters_only,
                             init_from_topology, IBPEnvironment)
from sailir.topology import Topology
from beam_search_v5 import _aux_from_picklable, _aux_to_result

topo_dir = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/topology_input/pentagonbox'
topology = Topology.from_dir(topo_dir)
init_from_topology(topology); set_prime(1009); set_paper_masters_only(False)
env = IBPEnvironment()

ckpt_path = sys.argv[1]
print(f'Loading {ckpt_path}')
with open(ckpt_path, 'rb') as f:
    d = pickle.load(f)
print(f"saved step: {d['step']}, n_beam={len(d['beam'])}, start_w12={d.get('start_w12')}")

# Take state 0
s = d['beam'][0]
saved_aux = s['aux_flat']
print(f"saved aux: type={type(saved_aux).__name__} "
      f"len={len(saved_aux) if saved_aux else 'None'}")

if saved_aux is not None:
    cu_s, ubm_s, srid, iraws_s = saved_aux[0], saved_aux[1], saved_aux[2], saved_aux[3]
    marker = saved_aux[4] if len(saved_aux) >= 5 else None
    print(f"  picklable form: cu={len(cu_s)} ubm={len(ubm_s)} "
          f"stable_rid={len(srid)} iraws={len(iraws_s)} marker={marker!r}")

    # First 5 iraws
    print('\nFirst 5 iraws (saved):')
    for i, e in enumerate(iraws_s[:5]):
        sub_int, op, shift, raw = e
        seed = tuple(sub_int[k] - shift[k] for k in range(len(sub_int)))
        rsize = len(raw) if hasattr(raw, '__len__') else 'n/a'
        print(f'  [{i}] sub_int={sub_int[:4]}... op={op} '
              f'shift={shift[:4]}... seed={seed[:4]}... raw_terms={rsize}')

    # Now depickle-load via _aux_from_picklable
    print('\n\nDepickle via _aux_from_picklable(env)…')
    loaded_aux = _aux_from_picklable(saved_aux, env=env)
    cu_l, ubm_l, rid_l, iraws_l = loaded_aux
    print(f'  loaded: cu={len(cu_l)} ubm={len(ubm_l)} rid={len(rid_l)} '
          f'iraws={len(iraws_l)}')

    # Compare content
    print('\n=== Content comparison ===')
    print(f'iraws count match: {len(iraws_s) == len(iraws_l)}')

    # iraws should have SAME (sub_int, op, shift) at each position
    n_mismatch_meta = 0
    for i, (a, b) in enumerate(zip(iraws_s, iraws_l)):
        if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
            n_mismatch_meta += 1
            if n_mismatch_meta <= 3:
                print(f'  iraws[{i}] meta mismatch:')
                print(f'    saved sub_int={a[0][:4]} op={a[1]} shift={a[2][:4]}')
                print(f'    loaded sub_int={b[0][:4]} op={b[1]} shift={b[2][:4]}')
    print(f'iraws meta (sub_int/op/shift) mismatches: {n_mismatch_meta}/{len(iraws_s)}')

    # cu content: each cu[i] is a substituted-raw dict; compare termwise
    cu_match = (cu_s == cu_l)
    print(f'cu content equal: {cu_match}')
    if not cu_match:
        diffs = sum(1 for a, b in zip(cu_s, cu_l) if a != b)
        print(f'  cu entries that differ: {diffs}/{len(cu_s)}')

    ubm_match = (ubm_s == ubm_l)
    print(f'ubm content equal: {ubm_match}')

    # rid mapping: for each iraws_l[i], does id(raw_l) → cu_idx equal srid[(op, seed)]?
    n_rid_mismatch = 0
    for i in range(len(iraws_l)):
        sub_int, op, shift, raw = iraws_l[i]
        seed = tuple(sub_int[k] - shift[k] for k in range(len(sub_int)))
        cu_idx_saved = srid.get((op, seed))
        cu_idx_loaded = rid_l.get(id(raw))
        if cu_idx_saved != cu_idx_loaded:
            n_rid_mismatch += 1
            if n_rid_mismatch <= 3:
                print(f'  rid[{i}] mismatch: saved cu_idx={cu_idx_saved} '
                      f'loaded cu_idx={cu_idx_loaded}')
    print(f'rid mappings inconsistent: {n_rid_mismatch}/{len(iraws_l)}')

    # Now reconstruct the indirect_cache via _aux_to_result and inspect
    print('\n=== _aux_to_result reconstruction ===')
    res = _aux_to_result(loaded_aux)
    print(f'reconstructed indirect_cache: {len(res)} entries')
    # Show first 3
    for i, r in enumerate(res[:3]):
        sub_int, op, shift, raw, cached, ubm = r
        print(f'  [{i}] sub_int={sub_int[:4]} op={op} '
              f'cached_terms={len(cached)} ubm={ubm}')
