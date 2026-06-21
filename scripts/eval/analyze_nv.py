"""Per-step n_valid (number of actions fed to the model) for v7 and v8 greedy
runs. The dump's n_v = min(len(valid), ~900), so n_v == 900 => the 900-action
cap is biting (there were >= 900 valid actions and the rest were truncated)."""
import pickle


def w(i):
    return (sum(max(0, x) for x in i), -sum(min(0, x) for x in i))


def load(tag, n):
    with open(f'results/traj_{tag}_74/picks/picks_step{n}.pkl', 'rb') as f:
        return pickle.load(f)


print(f"{'step':>4} | v7: per-task (target_rs, n_valid)            "
      f"| v8: (target_rs, n_valid)   cap@900?")
for n in range(50):
    try:
        t7 = load('v7', n)
        t8 = load('v8', n)
    except FileNotFoundError:
        break
    v7info = [(w(t[1]), t[2]) for t in t7]   # (target (r,s), n_valid)
    v8info = [(w(t[1]), t[2]) for t in t8]
    cap8 = any(t[2] >= 900 for t in t8)
    cap7 = any(t[2] >= 900 for t in t7)
    mark = "  <-- v8 CAPPED" if cap8 else ("  (v7 capped)" if cap7 else "")
    print(f"{n+1:>4} | {str(v7info):>44} | {str(v8info):>26}{mark}")
