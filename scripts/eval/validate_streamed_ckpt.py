"""Validate a STREAMED checkpoint (from _stream_dump_ckpt): meta dict frame
(_streamed=True, n_states) followed by n_states independent state-dict frames.
Confirms it is readable exactly the way the resume path reads it, and the
states look structurally sane. Usage: python validate_streamed_ckpt.py <ckpt>
"""
import pickle
import sys


def main():
    path = sys.argv[1]
    with open(path, 'rb') as f:
        meta = pickle.load(f)
        if not (isinstance(meta, dict) and meta.get('_streamed')):
            print(f"NOT streamed format (top-level keys: "
                  f"{list(meta.keys()) if isinstance(meta, dict) else type(meta)})")
            # old format: meta IS the full dict with 'beam'
            beam = meta.get('beam') if isinstance(meta, dict) else None
            print(f"  legacy beam states: {len(beam) if beam else 0}")
            return
        n = meta['n_states']
        print(f"streamed=True  step={meta['step']}  n_states={n}  "
              f"tabu_dict={'present' if meta.get('tabu_dict') is not None else 'None'}")
        beam = []
        for i in range(n):
            beam.append(pickle.load(f))
        # confirm no trailing garbage / exactly n frames
        trailing = f.read(1)
    print(f"loaded {len(beam)} state frames (expected {n}): "
          f"{'OK' if len(beam) == n else 'MISMATCH'}")
    print(f"trailing bytes after {n} frames: "
          f"{len(trailing)} ({'clean EOF' if not trailing else 'UNEXPECTED DATA'})")
    s0 = beam[0]
    keys = list(s0.keys()) if isinstance(s0, dict) else None
    print(f"state[0] keys: {keys}")
    aux = s0.get('aux_flat') if isinstance(s0, dict) else None
    if aux is not None:
        print(f"state[0] aux_flat: tuple len={len(aux)} "
              f"(cu entries={len(aux[0])}, last elem marker={aux[-1]!r})")
    nrs = sum(len(s.get('resolved_subs') or {}) for s in beam)
    print(f"total resolved_subs entries across beam: {nrs}")
    print("VALIDATION PASS" if len(beam) == n and not trailing else "VALIDATION FAIL")


if __name__ == '__main__':
    main()
