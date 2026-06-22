"""Empirically verify the streaming-checkpoint mechanism caps the peak RSS at
~ONE state's serialized dict, not the whole beam. Mimics the exact pattern:

  OLD: out = [big_dict(s) for s in beam]; pickle.dump({'beam': out}, f)
       -> all N big dicts live simultaneously -> peak ~ N x dict
  NEW: for s in beam: d = big_dict(s); pickle.dump(d, f); del d
       -> one big dict at a time (fresh memo per pickle.dump, del frees) -> peak ~ 1 x dict

Each big_dict simulates one state's to_dict-materialized cu (~the 5.2GB/29
per-state share). VmHWM (peak RSS) is reset between the two methods via
/proc/self/clear_refs so each peak is isolated. Output is the verdict.
"""
import os
import pickle


def vmhwm_mb():
    for l in open('/proc/self/status'):
        if l.startswith('VmHWM:'):
            return int(l.split()[1]) // 1024
    return -1


def vmrss_mb():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS:'):
            return int(l.split()[1]) // 1024
    return -1


def reset_peak():
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
    except OSError:
        pass


N = 20
PER = 1_200_000  # entries per state-dict -> ~120-150 MB each


def big_dict(seed):
    # ~ one state's serialized cu: {int_key: int_val}, sized to be a real chunk
    return {(seed, i): (i * 2 + 1) for i in range(PER)}


def main():
    beam = list(range(N))          # the "states" (small handles), stay alive
    devnull = open(os.devnull, 'wb')

    base = vmrss_mb()
    print(f"baseline RSS = {base} MB ; N={N} states, ~{PER:,} entries each")

    # ---- OLD: materialize all N at once ----
    reset_peak()
    out = []
    for s in beam:
        out.append(big_dict(s))
    pickle.dump({'beam': out}, devnull)
    old_peak = vmhwm_mb()
    del out
    print(f"OLD (all-at-once): peak RSS = {old_peak} MB  (delta over base = {old_peak-base} MB)")

    # ---- NEW: stream one at a time ----
    reset_peak()
    pickle.dump({'_streamed': True, 'n_states': N}, devnull)
    for s in beam:
        d = big_dict(s)
        pickle.dump(d, devnull)
        del d
    new_peak = vmhwm_mb()
    print(f"NEW (streamed):    peak RSS = {new_peak} MB  (delta over base = {new_peak-base} MB)")

    od = old_peak - base
    nd = new_peak - base
    print()
    print(f"per-state dict ~ {nd} MB ; old held ~{od/max(nd,1):.1f}x that ; "
          f"streaming cut the transient {od/max(nd,1):.1f}x")
    ok = nd > 0 and od > nd * (N * 0.4)   # old should be many-x the new
    print(f"VERDICT: streaming caps transient at ~1 state: "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()
