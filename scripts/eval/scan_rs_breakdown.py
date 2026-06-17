"""Per-structure breakdown of a beam_search_v7 checkpoint: how many integral
(integral->coeff) dict ENTRIES live in resolved_subs vs cu vs sub_accum vs expr
vs tabu, with a memory estimate. Answers 'how much is resolved_subs taking?'.

Global visited set (by id) so COW-shared dicts are counted once, attributed to
whichever top-level structure reaches them first. resolved_subs values are not
shared with cu, so the split is clean for the structures we care about.

NOTE: in the live RUN cu is PACKED (not dict); the checkpoint stores cu as dict.
So 'cu' here is informational; the run's dict footprint is resolved_subs + expr
+ sub_accum + tabu.

Usage: python scan_rs_breakdown.py <ckpt.pkl>
"""
import pickle
import sys
from collections import namedtuple

N = 11
State_v5 = namedtuple(
    'State_v5',
    ['expr', 'resolved_subs', 'sub_accum', 'score', 'path',
     'n_non_masters', 'max_w12', 'total_w12', 'aux_flat'])


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'State_v5':
            return State_v5
        return super().find_class(module, name)


def is_integral(k):
    return (isinstance(k, tuple) and len(k) == N
            and all(isinstance(x, int) and not isinstance(x, bool) for x in k))


# global visited set across ALL structures (so shared dicts counted once)
visited = set()


def count_entries(obj):
    """Count integral->coeff dict entries reachable from obj (dedup by id).
    Returns (n_entries, n_dicts)."""
    n_entries = 0
    n_dicts = 0
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, (dict, list, tuple, set, frozenset)):
            oid = id(o)
            if oid in visited:
                continue
            visited.add(oid)
        if isinstance(o, dict):
            is_eq = False
            for k, v in o.items():
                if is_integral(k):
                    n_entries += 1
                    is_eq = True
                    stack.append(v)
                else:
                    stack.append(k)
                    stack.append(v)
            if is_eq:
                n_dicts += 1
        elif isinstance(o, (list, tuple)):
            for x in o:
                stack.append(x)
        elif isinstance(o, (set, frozenset)):
            for x in o:
                stack.append(x)
    return n_entries, n_dicts


def fmt(n_entries, n_dicts):
    # ~70 B/entry (32 slot + ~28 int + shared tuple) + ~64 B/dict base
    lo = (n_entries * 60 + n_dicts * 64) / 1e9
    hi = (n_entries * 100 + n_dicts * 64) / 1e9
    return (f"{n_entries:>14,} entries  {n_dicts:>10,} dicts  "
            f"~{lo:.2f}-{hi:.2f} GB")


def main():
    path = sys.argv[1]
    print(f"loading {path} ...", flush=True)
    with open(path, 'rb') as f:
        ckpt = _Unpickler(f).load()
    print("loaded; counting per structure (global id-dedup) ...\n", flush=True)

    # introspect the checkpoint structure (don't assume)
    print(f"ckpt type: {type(ckpt).__name__}", flush=True)
    if isinstance(ckpt, dict):
        print(f"ckpt keys: {list(ckpt.keys())}", flush=True)
    beam = ckpt.get('beam') if isinstance(ckpt, dict) else (
        ckpt if isinstance(ckpt, list) else [])
    print(f"beam states: {len(beam)}", flush=True)
    if beam:
        s0 = beam[0]
        print(f"state[0] type: {type(s0).__name__}", flush=True)
        if isinstance(s0, dict):
            print(f"state[0] keys: {list(s0.keys())}", flush=True)
        elif hasattr(s0, '_fields'):
            print(f"state[0] fields: {s0._fields}", flush=True)
    print(flush=True)

    def get_field(st, field):
        if isinstance(st, dict):
            return st.get(field)
        return getattr(st, field, None)

    totals = {}
    # Order matters for attribution under sharing; do resolved_subs first
    # (the thing we care about), then the rest.
    for field in ['resolved_subs', 'aux_flat', 'sub_accum', 'expr']:
        te = td = 0
        for st in beam:
            val = get_field(st, field)
            if val is None:
                continue
            e, d = count_entries(val)
            te += e
            td += d
        totals[field] = (te, td)
        print(f"  {field:14s}: {fmt(te, td)}", flush=True)

    # tabu_dict (top-level)
    tb = ckpt.get('tabu_dict') if isinstance(ckpt, dict) else None
    if tb is not None:
        e, d = count_entries(tb)
        totals['tabu_dict'] = (e, d)
        print(f"  {'tabu_dict':14s}: {fmt(e, d)}", flush=True)

    print("\n  (cu/aux_flat is PACKED in the live run -> not dict footprint there)")
    rs_e, rs_d = totals.get('resolved_subs', (0, 0))
    print(f"\n==> resolved_subs (the Stage-3 target): "
          f"{rs_e:,} entries across {rs_d:,} solution dicts")
    print(f"    estimated dict memory: ~{rs_e*60/1e9:.2f}-{rs_e*100/1e9:.2f} GB")
    print(f"    packed equivalent (int32 id + int16 coeff = 6 B): "
          f"~{rs_e*6/1e9:.3f} GB + ~0.005 GB registry")


if __name__ == '__main__':
    main()
