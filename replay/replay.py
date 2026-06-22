#!/usr/bin/env python3
"""
SAILIR pentagonbox (8,5) reduction -- STANDALONE replay
=======================================================

Replays the (8,5) "top" Feynman integral

    I[1, 1, 1, 1, 1, 1, 1, 1, -5, 0, 0]

down to a basis of MASTER + CORNER integrals, using a precomputed cache of
one-step reductions (the combined output of SAILIR reduction rounds 1-4).

This script is fully self-contained: it depends ONLY on the Python standard
library (pickle, json, argparse) and the two data files shipped alongside it.
It does NOT import any part of the SAILIR codebase.

----------------------------------------------------------------------------
WHAT IT DOES (the math, briefly)
----------------------------------------------------------------------------
An integral is written as an 11-component integer tuple
    (a0, a1, ..., a7, a8, a9, a10)
The first 8 entries are propagator powers (denominators); the last 3 (the
"ISP" positions 8,9,10) are irreducible scalar products (numerators).

The reduction `cache` maps an integral to a linear combination of other
integrals over the finite field GF(prime):
    cache[I] = { J1: c1, J2: c2, ... }      (meaning  I = sum_k c_k * J_k)
A "fail/identity" entry cache[I] = {I: 1} means I was left unreduced.

Replay = start from the single term {start_integral: 1} and repeatedly
substitute every integral that appears in the cache, accumulating
coefficients mod `prime`, until nothing changes (a fixpoint). What remains
is the start integral expressed purely in terms of integrals that are NOT in
the cache -- i.e. the master/corner basis.

Each surviving term is classified as:
  * PAPER  -- a member of the master basis (topology_pentagonbox.json:masters)
  * CORNER -- a corner integral (all denominators in {0,1}, all ISPs 0) whose
              sector is not covered by the master basis (still a valid basis
              element)
  * NON    -- anything else; a NON term means the reduction is INCOMPLETE.
A successful, complete reduction leaves 0 NON terms.

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
  python replay.py                      # uses the bundled data files
  python replay.py --verbose            # also print every surviving term
  python replay.py --out result.json    # save the reduced expression
  python replay.py --cache other.pkl --topology other.json

Exit status is 0 on a complete reduction (0 NON terms), 1 otherwise.
"""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _die(msg, code=2):
    """Print a single actionable error line to stderr and exit."""
    sys.stderr.write(f"error: {msg}\n")
    sys.exit(code)


# ===========================================================================
# Master / corner classification -- pure functions, no external dependencies.
# (Reshaped from sailir.ibp_env.weight / get_sector / is_corner_integral /
#  is_master; semantics preserved exactly, with PAPER_MASTERS_ONLY = False.)
# ===========================================================================
def weight(i):
    """Integral weight used for ordering: (total dots, total numerator, |abs|).
      w1 = sum of positive entries      (raised-propagator "dots")
      w2 = sum of |negative entries|    (numerator powers)
      w3 = tuple of absolute values     (tie-break)
    """
    return (sum(x for x in i if x > 0),
            -sum(x for x in i if x < 0),
            tuple(abs(x) for x in i))


def get_sector(i, isp_positions, n_indices):
    """Sector tuple: 1 at each denominator slot with a positive power."""
    return tuple(1 if i[p] > 0 else 0
                 for p in range(n_indices) if p not in isp_positions)


def is_corner_integral(i, isp_positions, n_indices):
    """Corner integral: every denominator in {0,1}; every ISP exactly 0."""
    for j in range(n_indices):
        if j in isp_positions:
            if i[j] != 0:
                return False
        else:
            if i[j] not in (0, 1):
                return False
    return True


def make_is_master(masters_set, isp_positions, n_indices):
    """Build is_master(i) closing over the topology constants.

    Masters are (PAPER_MASTERS_ONLY = False convention):
      1. members of the master basis `masters_set`, OR
      2. corner integrals whose sector is NOT covered by the master basis.
    """
    master_sectors = frozenset(get_sector(m, isp_positions, n_indices)
                               for m in masters_set)

    def is_master(i):
        if i in masters_set:
            return True
        if get_sector(i, isp_positions, n_indices) not in master_sectors:
            return is_corner_integral(i, isp_positions, n_indices)
        return False

    return is_master


# ===========================================================================
# Replay -- expand the start term through the cache to a fixpoint (mod prime).
# (Reshaped from save_replay_state.apply_subs; identical algorithm.)
# ===========================================================================
def apply_substitutions(start_integral, cache, prime, max_iter=200000,
                        log=lambda *_: None):
    """Return {integral: coeff} after replaying {start: 1} through `cache`."""
    expr = {start_integral: 1}
    iters = 0
    changed = True
    while changed:
        changed = False
        iters += 1
        new_expr = {}
        for ig, coeff in expr.items():
            if coeff == 0:
                continue
            sub = cache.get(ig)
            if sub is None or sub == {ig: 1}:
                # Leaf (a master/corner) or an explicit unreduced identity:
                # carry the term unchanged.
                new_expr[ig] = (new_expr.get(ig, 0) + coeff) % prime
                continue
            for k, c in sub.items():
                if c == 0:
                    continue
                new_expr[k] = (new_expr.get(k, 0) + coeff * c) % prime
            changed = True
        expr = {k: v for k, v in new_expr.items() if v != 0}
        log(f"  iter {iters:>3}: {len(expr)} live terms")
        if iters > max_iter:
            log(f"  WARNING: hit max_iter={max_iter}; stopping")
            break
    return expr


# ===========================================================================
def load_topology(path):
    with open(path) as fp:
        t = json.load(fp)
    for key in ('n_indices', 'isp_positions', 'masters'):
        if key not in t:
            raise KeyError(key)
    masters_set = frozenset(tuple(m) for m in t['masters'])
    return t, masters_set, int(t['n_indices']), tuple(t['isp_positions'])


def load_cache(path):
    with open(path, 'rb') as fp:
        b = pickle.load(fp)
    for key in ('start_integral', 'prime', 'cache'):
        if key not in b:
            raise KeyError(key)
    return (tuple(b['start_integral']), int(b['prime']), b['cache'])


def main():
    ap = argparse.ArgumentParser(
        description="Standalone replay of the SAILIR pentagonbox (8,5) reduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', default=os.path.join(HERE, 'reduction_cache.pkl'),
                    help='reduction cache pickle (default: bundled reduction_cache.pkl)')
    ap.add_argument('--topology', default=os.path.join(HERE, 'topology_pentagonbox.json'),
                    help='topology JSON (default: bundled topology_pentagonbox.json)')
    ap.add_argument('--start', default=None,
                    help='override start integral, comma-separated 11 ints '
                         '(default: the one stored in the cache)')
    ap.add_argument('--out', default=None,
                    help='also save the reduced expression to this JSON file')
    ap.add_argument('--summary-only', action='store_true',
                    help='print only the term counts, not the full expression')
    ap.add_argument('--progress', action='store_true',
                    help='print the per-iteration live-term count during replay')
    args = ap.parse_args()

    # Fail fast on an unwritable --out directory BEFORE the (expensive) replay,
    # so a path typo doesn't waste the whole computation.
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or '.'
        if not os.path.isdir(out_dir):
            _die(f"--out directory does not exist: {out_dir}")

    # Load the bundled data with friendly errors for the common mistakes
    # (a data file not extracted next to replay.py, or the wrong file passed).
    try:
        topo, masters_set, n_indices, isp_positions = load_topology(args.topology)
    except FileNotFoundError:
        _die(f"topology file not found: {args.topology}\n"
             f"       (it must sit next to replay.py, or pass --topology PATH)")
    except (ValueError, KeyError, UnicodeDecodeError) as e:
        _die(f"could not parse topology JSON '{args.topology}': {e!r}")
    try:
        start, prime, cache = load_cache(args.cache)
    except FileNotFoundError:
        _die(f"cache file not found: {args.cache}\n"
             f"       (make sure reduction_cache.pkl was extracted next to replay.py)")
    except (pickle.UnpicklingError, KeyError, EOFError, ValueError) as e:
        _die(f"could not read cache '{args.cache}': {e!r}\n"
             f"       (is this really reduction_cache.pkl, and was it fully copied?)")

    # Optional start override, validated against the topology width.
    if args.start is not None:
        try:
            start = tuple(int(x) for x in args.start.split(','))
        except ValueError:
            _die(f"--start must be comma-separated integers (got {args.start!r})")
        if len(start) != n_indices:
            _die(f"--start must have exactly {n_indices} integers, got {len(start)}")

    print("=" * 70)
    print("SAILIR pentagonbox (8,5) reduction -- standalone replay")
    print("=" * 70)
    print(f"family         : {topo['family_name']}")
    print(f"n_indices      : {n_indices}   isp_positions: {isp_positions}")
    print(f"master basis   : {len(masters_set)} integrals")
    print(f"cache entries  : {len(cache):,}")
    print(f"prime (GF)     : {prime}")
    print(f"start integral : I{list(start)}   weight(r,s)={weight(start)[:2]}")
    print("-" * 70)
    print("replaying ...")

    log = print if args.progress else (lambda *_: None)
    expr = apply_substitutions(start, cache, prime, log=log)

    is_master = make_is_master(masters_set, isp_positions, n_indices)

    # Classify and order heaviest-first (same key as the reference tool).
    terms = [(ig, c) for ig, c in expr.items() if c != 0]
    terms.sort(key=lambda kv: (-weight(kv[0])[0], -weight(kv[0])[1], kv[0]))

    n_paper = n_corner = n_non = 0
    classified = []
    for ig, c in terms:
        if ig in masters_set:
            cat = 'PAPER'
            n_paper += 1
        elif is_master(ig):
            cat = 'CORNER'
            n_corner += 1
        else:
            cat = 'NON'
            n_non += 1
        classified.append((ig, c, cat))

    # The reduced expression IS the result: start = sum_k coeff_k * I_k (mod prime).
    # Printed by DEFAULT; --summary-only suppresses the per-term listing.
    if not args.summary_only:
        print("=" * 70)
        print(f"REDUCED EXPRESSION  (start = sum of coeff * master, mod {prime})")
        print("=" * 70)
        print(f"I{list(start)}  =")
        print(f'{"#":>4} {"coeff":>5} {"cat":>6} {"w(r,s)":>8}  master / corner integral')
        for n, (ig, c, cat) in enumerate(classified, 1):
            w = weight(ig)
            print(f'{n:>4} {c:>5} {cat:>6} ({w[0]},{w[1]})  I{list(ig)}')

    print("-" * 70)
    print(f"RESULT: {len(terms)} terms = {n_paper} PAPER + {n_corner} CORNER "
          f"masters + {n_non} NON-masters")
    if n_non == 0:
        print("SUCCESS: the (8,5) integral is fully reduced to the "
              "master + corner basis.")
    else:
        print(f"INCOMPLETE: {n_non} non-master term(s) remain unreduced.")

    if args.out:
        result = {
            'start_integral': list(start),
            'prime': prime,
            'n_terms': len(terms),
            'n_paper': n_paper, 'n_corner': n_corner, 'n_non': n_non,
            'terms': [{'integral': list(ig), 'coeff': c, 'category': cat}
                      for ig, c, cat in classified],
        }
        try:
            with open(args.out, 'w') as fp:
                json.dump(result, fp, indent=1)
            print(f"saved reduced expression -> {args.out}")
        except OSError as e:
            # The reduction already succeeded; don't discard it over a write error.
            sys.stderr.write(f"warning: could not write --out file "
                             f"{args.out!r}: {e}\n")

    return 0 if n_non == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
