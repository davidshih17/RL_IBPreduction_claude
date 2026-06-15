"""Stage-1 verification: sailir/packed_eq kernels are BIT-IDENTICAL to the
dict-based ibp_env kernels they replace.

Oracle = verbatim copies of ibp_env._substitute_one_in_cached and
cached_union_bitmask. We fuzz random GF(1009) equations + substitutions
(including the absent-key fast path, collisions, and exact cancellations) and
assert the packed result's to_dict() equals the dict oracle, and that
union_bitmask matches.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sailir.packed_eq import IntegralRegistry, PackedEq, substitute_one, union_bitmask

PRIME = 1009
N_INDICES = 11


# ---- ORACLE: verbatim from sailir/ibp_env.py ----
def oracle_substitute_one(cached, sub_key, replacement):
    if sub_key not in cached:
        return cached
    new_cached = dict(cached)
    coeff = new_cached.pop(sub_key)
    for k, v in replacement.items():
        nc = (coeff * v) % PRIME
        if k in new_cached:
            s = (new_cached[k] + nc) % PRIME
            if s == 0:
                del new_cached[k]
            else:
                new_cached[k] = s
        elif nc != 0:
            new_cached[k] = nc
    return new_cached


def oracle_union_bitmask(cached_eq):
    u = 0
    for k in cached_eq:
        bm = 0
        for i in range(N_INDICES):
            if k[i] > 0:
                bm |= (1 << i)
        u |= bm
    return u


def rand_integral(rng):
    return tuple(rng.randint(-5, 5) for _ in range(N_INDICES))


def rand_eq(rng, n, pool):
    return {pool[rng.randrange(len(pool))]: rng.randint(1, PRIME - 1)
            for _ in range(n)}


def packed_substitute(eq_dict, sub_int, rep_dict, reg):
    peq = PackedEq.from_dict(eq_dict, reg)
    sub_id = reg.get_id(sub_int)
    rep_ids = [reg.get_id(k) for k in rep_dict]
    rep_coeffs = [rep_dict[k] for k in rep_dict]
    out = substitute_one(peq, sub_id, rep_ids, rep_coeffs, PRIME)
    return out, (out is peq)


def main():
    rng = random.Random(20260614)
    reg = IntegralRegistry()
    n_trials = 30000
    n_unchanged_checked = 0
    n_cancel_checked = 0

    for t in range(n_trials):
        pool = [rand_integral(rng) for _ in range(rng.randint(2, 25))]
        eq = rand_eq(rng, rng.randint(0, 15), pool)
        rep = rand_eq(rng, rng.randint(0, 8), pool)

        # 70% pick a sub_key that IS in eq (exercise substitution); else random
        if eq and rng.random() < 0.7:
            sub_int = rng.choice(list(eq.keys()))
        else:
            sub_int = pool[rng.randrange(len(pool))]

        want = oracle_substitute_one(eq, sub_int, rep)
        got, was_unchanged = packed_substitute(eq, sub_int, rep, reg)

        if sub_int not in eq:
            assert was_unchanged, f"trial {t}: absent-key should be unchanged (is)"
            n_unchanged_checked += 1

        got_dict = got.to_dict(reg)
        assert got_dict == want, (
            f"trial {t}: MISMATCH\n eq={eq}\n sub={sub_int}\n rep={rep}\n"
            f" want={want}\n got ={got_dict}")

        # union bitmask check on the result
        assert union_bitmask(got, reg) == oracle_union_bitmask(got_dict), \
            f"trial {t}: union_bitmask mismatch"

    # --- targeted edge cases ---
    # (a) exact cancellation to empty: eq = {A: 5}, sub A -> {A: ... } no;
    #     use eq={A:c, B:d}, sub A -> {B: e} chosen so B cancels.
    A = rand_integral(rng); B = rand_integral(rng)
    while B == A:
        B = rand_integral(rng)
    # eq = c*A + d*B ; sub A -> r*B  =>  result (d + c*r)*B ; pick r so d+c*r=0
    c, d = 3, 7
    r = (-d * pow(c, -1, PRIME)) % PRIME      # c*r = -d  => coefficient of B = 0
    eq = {A: c, B: d}
    rep = {B: r}
    want = oracle_substitute_one(dict(eq), A, rep)
    got, _ = packed_substitute(eq, A, rep, reg)
    assert got.to_dict(reg) == want and len(want) == 0, \
        f"cancellation-to-empty failed: want={want} got={got.to_dict(reg)}"
    n_cancel_checked += 1

    # (b) empty replacement: sub A -> {} removes A entirely
    eq = {A: c, B: d}
    want = oracle_substitute_one(dict(eq), A, {})
    got, _ = packed_substitute(eq, A, {}, reg)
    assert got.to_dict(reg) == want == {B: d}, "empty-replacement failed"

    # (c) empty equation: unchanged
    got, unchanged = packed_substitute({}, A, {B: 5}, reg)
    assert unchanged and got.to_dict(reg) == {}, "empty-eq failed"

    print(f"PASS: {n_trials} random trials bit-identical "
          f"(absent-key fast path checked {n_unchanged_checked}x, "
          f"explicit cancellation {n_cancel_checked}x). "
          f"registry size={len(reg)}")


if __name__ == '__main__':
    main()
