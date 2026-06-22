# SAILIR pentagonbox (8,5) reduction — standalone replay

This package replays a precomputed integration-by-parts (IBP) reduction of the
pentagonbox **(8,5) top integral**

```
I[1, 1, 1, 1, 1, 1, 1, 1, -5, 0, 0]
```

all the way down to a basis of **master + corner integrals**, and prints /
saves the result. It reproduces, byte-for-byte, the reduction produced by the
full SAILIR pipeline over reduction rounds 1–4 — but with **zero dependencies**
beyond the Python standard library.

---

## 1. Requirements

- **Python ≥ 3.6** (CPython). Nothing else — no `pip install`, no NumPy, no
  SAILIR. Only `pickle`, `json`, `argparse` from the standard library are used.
  Invoke it as `python3` (on minimal/RHEL systems there is no bare `python`).
- ~400 MB free RAM (the replay expands to ~76k live terms at its peak — measured
  peak RSS ~370 MB — before draining to the 262-term basis).
- ~53 MB disk for the bundled cache.

## 2. Contents

| File                        | What it is                                                            |
|-----------------------------|-----------------------------------------------------------------------|
| `replay.py`                 | The standalone replay program (pure standard library).                |
| `reduction_cache.pkl`       | The combined round 1–4 one-step reduction cache (data, ~53 MB).       |
| `topology_pentagonbox.json` | Topology constants + the 61-integral master basis (human-readable).   |
| `README.md`                 | This file.                                                            |

The two data files contain only plain Python tuples and integers — the pickle
unpickles without any SAILIR code on the path.

## 3. Quick start

```bash
python3 replay.py
```

(or `./replay.py` — the file is executable and has a `python3` shebang.)

**By default it prints the full reduced expression** — the start integral
written as a linear combination of master/corner integrals (mod 1009), all 262
terms heaviest-first — followed by the summary:

```
======================================================================
REDUCED EXPRESSION  (start = sum of coeff * master, mod 1009)
======================================================================
I[1, 1, 1, 1, 1, 1, 1, 1, -5, 0, 0]  =
   # coeff    cat   w(r,s)  master / corner integral
   1   166  PAPER (8,1)  I[1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0]
   2   867  PAPER (8,1)  I[1, 1, 1, 1, 1, 1, 1, 1, 0, -1, 0]
 ...
 262   227 CORNER (0,0)  I[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
----------------------------------------------------------------------
RESULT: 262 terms = 61 PAPER + 201 CORNER masters + 0 NON-masters
SUCCESS: the (8,5) integral is fully reduced to the master + corner basis.
```

Each row is one term `coeff * I[...]` of the reduced expression; `cat` is its
class (PAPER/CORNER, see §4) and `w(r,s)` its weight. Row 262 (the all-zeros
corner) is the lightest term.

The program exits `0` on a complete reduction (0 NON-master terms), `1` if any
NON-master term remains, and `2` on a usage error (missing data file, bad
`--start`, unwritable `--out`, etc. — each reported as a single `error:` line).

### Save it / quiet it

```bash
python3 replay.py --out result.json     # ALSO save the expression as JSON
python3 replay.py --summary-only        # print only the counts, not the 262 terms
python3 replay.py --progress            # show the per-iteration replay progress
```

## 4. What it computes (the math, briefly)

An integral is an 11-component integer tuple
`(a0,…,a7, a8,a9,a10)`. The first 8 entries are **propagator powers**
(denominators); the last 3 (the *ISP* positions 8, 9, 10) are **irreducible
scalar products** (numerators).

The reduction `cache` maps an integral to a linear combination of other
integrals over the finite field `GF(prime)` with `prime = 1009`:

```
cache[I] = { J1: c1, J2: c2, ... }     means   I = c1*J1 + c2*J2 + ...  (mod 1009)
```

A *fail/identity* entry `cache[I] = {I: 1}` marks `I` as left unreduced.

**Replay** starts from the single term `{start: 1}` and repeatedly substitutes
every integral found in the cache, accumulating coefficients mod `prime`, until
nothing changes (a fixpoint). What survives is the start integral written purely
in terms of integrals that are **not** in the cache — the master/corner basis.

Each surviving term is classified as:

- **PAPER** — a member of the master basis (`topology_pentagonbox.json:masters`,
  61 integrals).
- **CORNER** — a corner integral (all denominators in `{0,1}`, all ISPs `0`)
  whose sector is *not* covered by the master basis. Still a legitimate basis
  element.
- **NON** — anything else. A NON term means the reduction is **incomplete**.

A complete reduction leaves **0 NON terms**.

## 5. Command-line options

```
python3 replay.py [--cache FILE] [--topology FILE] [--start I] [--out FILE]
                  [--summary-only] [--progress]

  --cache FILE      reduction cache pickle      (default: bundled reduction_cache.pkl)
  --topology FILE   topology JSON               (default: bundled topology_pentagonbox.json)
  --start I         override start integral, comma-separated 11 ints
                    (default: the start stored in the cache)
  --out FILE        ALSO save the reduced expression as JSON
  --summary-only    print only the term counts, not the full expression
  --progress        print the per-iteration live-term count during replay
```

By default the full reduced expression (all 262 terms) is printed to stdout.

## 6. Data formats

**`reduction_cache.pkl`** (Python pickle, protocol 4) — a dict:

```python
{
  'start_integral': (1,1,1,1,1,1,1,1,-5,0,0),   # tuple[int] length 11
  'prime': 1009,                                 # int
  'cache': { I_tuple: { J_tuple: coeff_int, ... }, ... },
  'description': "...", 'source': "...",
}
```

**`topology_pentagonbox.json`**:

```json
{
  "family_name": "TA",
  "n_indices": 11,
  "n_denominators": 8,
  "isp_positions": [8, 9, 10],
  "n_masters": 61,
  "masters": [[...11 ints...], ...]
}
```

**`result.json`** (written by `--out`): `start_integral`, `prime`, the term
counts (`n_paper`, `n_corner`, `n_non`), and `terms`, a list of
`{"integral": [...], "coeff": int, "category": "PAPER|CORNER|NON"}`.

## 7. Provenance & verification

- The cache is the union of the per-integral one-step reductions produced by
  SAILIR reduction rounds 1–4 of the pentagonbox (8,5) sector (103,626 cache
  entries).
- `replay.py`'s classification functions (`weight`, `get_sector`,
  `is_corner_integral`, `is_master`) and its `apply_substitutions` replay are
  faithful reshapings of the corresponding SAILIR routines, with the
  "paper-masters-only" option **off** (corner integrals in uncovered sectors
  count as masters).
- The standalone output was verified **term-by-term and coefficient-by-
  coefficient identical** to the reduced expression computed by the full SAILIR
  codebase: 262 terms, 0 differing keys, 0 differing coefficients.
