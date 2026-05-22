"""
Parser for Kira's sectorSymmetries and sectorRelations files.

Format reverse-engineered from Kira source: src/kira/trivial_sym.cpp,
function write_symmetries (lines 1781-1809 in v3.1).

Per-line layout (whitespace-separated tokens):

    source_sector
    [loop_substs: 2*n_loops tokens, lhs1 rhs1 lhs2 rhs2 ... ]
    det                     # +1 or -1, the overall sign
    target_sector
    n_props                 # number of propagators in source sector
    external_symmetry       # index into externalTransf
    kin_subst               # GiNaC-printed list like {p1==p1,p2==p2,...}
    momentum_elim           # GiNaC-printed list like {p5==-p1-p2-p3-p4}
    ing[0..n_indices-1]     # permutation array; ing[g] = target position
                            # for source position g, or -1 if absent in target
    sym_dots                # 1 if "dots/placeholder" symmetry, 0 if direct
    target_topology         # 0 for same-family

Total tokens per record = 9 + 2*n_loops + n_indices.

Semantics of a single record (when applicable, see below):

    I_{source_topology}[a_0, ..., a_{n-1}]
        ==  I_{target_topology}[b_0, ..., b_{n-1}]
    where b_{ing[g]} = a_g for every g with ing[g] != -1,
    and b_j = 0 for every target position j not in the image of ing.

NOTE on the `det` / sign field: Kira writes a signed determinant per
record, computed as the Jacobian of the loop-momentum substitution in
src/kira/trivial_sym.cpp:test(). Empirically (see scripts/symmetries/
parser_smoke_test.py) the value equals the parity of the propagator
permutation sigma. This sign is *not* applied at the integral-value
level: SAILIR's existing trianglebox pipeline treats integrals such
as I[(0,1,1,1,0,0,0)] as nonzero masters even though some
sectorSymmetries rows with that integral fixed by sigma carry
det = -1 (which, if applied literally, would force I = 0). Hence the
record stores `sign` for completeness but `apply_record` does not
multiply by it. The field is exposed for diagnostics and for any
future use that wants to opt in.

Applicability: the symmetry is only valid for integrals where
a_g == 0 for every g with ing[g] == -1 (i.e., source positions that
are dropped by the permutation must carry no index), AND the integral
must lie in the source sector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class SymmetryRecord:
    source_sector: int
    target_sector: int
    target_topology: int
    sign: int                            # +1 or -1
    n_props: int                         # propagator count in source sector
    external_symmetry: int               # index into externalTransf
    sym_dots: int                        # 1 if placeholder/dots, 0 otherwise
    ing: Tuple[int, ...]                 # length n_indices, target position for each source g
    loop_substs: Tuple[Tuple[str, str], ...]  # raw (lhs, rhs) strings per loop momentum
    kin_subst: str                       # raw {...} token
    momentum_elim: str                   # raw {...} token


def parse_symmetries(path: str, n_indices: int, n_loops: int = 2) -> List[SymmetryRecord]:
    """Parse a Kira sectorSymmetries or sectorRelations file.

    Args:
        path: Path to the file.
        n_indices: Number of indices/propagators in the family (jule in Kira).
        n_loops: Number of loop momenta.

    Returns:
        List of SymmetryRecord, one per non-blank line, in file order.

    Raises:
        ValueError if any line has the wrong token count, an out-of-range
        permutation entry, or an unexpected sign.
    """
    expected = 9 + 2 * n_loops + n_indices
    records: List[SymmetryRecord] = []
    with open(path, "r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) != expected:
                raise ValueError(
                    f"{path}:{line_no}: expected {expected} tokens "
                    f"(n_loops={n_loops}, n_indices={n_indices}), "
                    f"got {len(toks)}: {line!r}"
                )
            idx = 0
            source_sector = int(toks[idx]); idx += 1
            substs: List[Tuple[str, str]] = []
            for _ in range(n_loops):
                lhs = toks[idx]; idx += 1
                rhs = toks[idx]; idx += 1
                substs.append((lhs, rhs))
            sign = int(toks[idx]); idx += 1
            if sign not in (+1, -1):
                raise ValueError(f"{path}:{line_no}: sign must be +/-1, got {sign}")
            target_sector = int(toks[idx]); idx += 1
            n_props = int(toks[idx]); idx += 1
            external_symmetry = int(toks[idx]); idx += 1
            kin_subst = toks[idx]; idx += 1
            momentum_elim = toks[idx]; idx += 1
            ing = tuple(int(toks[idx + g]) for g in range(n_indices))
            idx += n_indices
            for g, v in enumerate(ing):
                if v != -1 and not (0 <= v < n_indices):
                    raise ValueError(
                        f"{path}:{line_no}: ing[{g}]={v} out of range [0,{n_indices})"
                    )
            sym_dots = int(toks[idx]); idx += 1
            target_topology = int(toks[idx]); idx += 1
            assert idx == expected
            records.append(SymmetryRecord(
                source_sector=source_sector,
                target_sector=target_sector,
                target_topology=target_topology,
                sign=sign,
                n_props=n_props,
                external_symmetry=external_symmetry,
                sym_dots=sym_dots,
                ing=ing,
                loop_substs=tuple(substs),
                kin_subst=kin_subst,
                momentum_elim=momentum_elim,
            ))
    return records


def sector_of(integral: Tuple[int, ...]) -> int:
    """Sector bitmask: bit i set iff integral[i] > 0 (Kira convention)."""
    mask = 0
    for i, a in enumerate(integral):
        if a > 0:
            mask |= (1 << i)
    return mask


def apply_record(integral: Tuple[int, ...], rec: SymmetryRecord) -> Tuple[int, ...] | None:
    """Apply a single SymmetryRecord to an integral.

    Returns the transformed integral if the record is applicable, else None.

    The integral identity is sign-free: I_source[a] == I_target[apply_ing(a)].
    See the module docstring for the rationale on why rec.sign is not applied.

    Applicability:
      * sector_of(integral) must equal rec.source_sector
      * integral[g] must be 0 for every g with rec.ing[g] == -1
        (such source positions are dropped by the permutation).
    """
    n = len(integral)
    assert len(rec.ing) == n, f"ing length {len(rec.ing)} != integral length {n}"
    if sector_of(integral) != rec.source_sector:
        return None
    for g, t in enumerate(rec.ing):
        if t == -1 and integral[g] != 0:
            return None
    new = [0] * n
    for g, t in enumerate(rec.ing):
        if t == -1:
            continue
        new[t] = integral[g]
    return tuple(new)


# --------------------------------------------------------------------------
# Total order and orbit / canonicalize
# --------------------------------------------------------------------------

OrderKey = Callable[[Tuple[int, ...]], tuple]


def default_order_key(integral: Tuple[int, ...]) -> tuple:
    """Default total order for picking orbit representatives.

    Sort key: (t, r, s, d, sector_id, integral).
      t = number of positive indices (propagators present)
      r = sum of positive indices
      s = -sum of non-positive indices  (numerator weight)
      d = sum of (a_i - 1) for a_i > 1   (dots = repeated propagators)

    Lower is "simpler"; the orbit minimum under this order is the
    canonical representative. Matches the Kira convention that masters
    are placed in the simpler sector of an orbit (paper line 224-227).
    """
    pos = [a for a in integral if a > 0]
    neg = [a for a in integral if a <= 0]
    t = len(pos)
    r = sum(pos)
    s = -sum(neg)
    d = sum(a - 1 for a in pos if a > 1)
    return (t, r, s, d, sector_of(integral), integral)


@dataclass
class SymmetryGroup:
    """Collection of symmetry records, indexed by source sector for lookup.

    Built from one or more lists of SymmetryRecord (typically the parsed
    sectorSymmetries and sectorRelations files). By default `sym_dots`=1
    records ("placeholder" symmetries Kira emits when the explicit
    momentum substitution couldn't be resolved) are included; pass
    `include_dots=False` to exclude them.

    The canonicalize() method is memoized via an internal `_canon_cache`
    so repeated lookups are O(1). Call clear_cache() if you mutate the
    record list after construction.
    """

    records: List[SymmetryRecord] = field(default_factory=list)
    by_source: Dict[int, List[SymmetryRecord]] = field(default_factory=dict, repr=False)
    _canon_cache: Dict[Tuple[int, ...], Tuple[int, ...]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_records(cls, *record_lists: Iterable[SymmetryRecord],
                     include_dots: bool = True) -> "SymmetryGroup":
        recs: List[SymmetryRecord] = []
        for rl in record_lists:
            for r in rl:
                if not include_dots and r.sym_dots == 1:
                    continue
                recs.append(r)
        idx: Dict[int, List[SymmetryRecord]] = {}
        for r in recs:
            idx.setdefault(r.source_sector, []).append(r)
        return cls(records=recs, by_source=idx)

    def applicable_records(self, integral: Tuple[int, ...]) -> List[SymmetryRecord]:
        return self.by_source.get(sector_of(integral), [])

    def neighbors(self, integral: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Images of `integral` under every applicable record (excluding self)."""
        out: List[Tuple[int, ...]] = []
        for r in self.applicable_records(integral):
            img = apply_record(integral, r)
            if img is not None and img != integral:
                out.append(img)
        return out

    def orbit(self, integral: Tuple[int, ...]) -> Set[Tuple[int, ...]]:
        """All integrals reachable from `integral` by repeatedly applying
        applicable records.

        Records are 1-directional (source_sector -> target_sector); for
        cross-sector relations, only the "higher -> lower" direction is
        present in the file. The orbit therefore naturally "flows downhill"
        in sector ordering: starting from any element, the reachable set
        always contains the orbit minimum.
        """
        visited: Set[Tuple[int, ...]] = {integral}
        frontier: List[Tuple[int, ...]] = [integral]
        while frontier:
            x = frontier.pop()
            for y in self.neighbors(x):
                if y not in visited:
                    visited.add(y)
                    frontier.append(y)
        return visited

    def canonicalize(self, integral: Tuple[int, ...],
                     order_key: OrderKey = default_order_key) -> Tuple[int, ...]:
        """Return the orbit minimum of `integral` under `order_key`.

        Memoized when `order_key is default_order_key` (the common case).
        For a non-default order_key, the result is recomputed each call.
        """
        if order_key is default_order_key:
            hit = self._canon_cache.get(integral)
            if hit is not None:
                return hit
            rep = min(self.orbit(integral), key=order_key)
            self._canon_cache[integral] = rep
            return rep
        return min(self.orbit(integral), key=order_key)

    def clear_cache(self) -> None:
        self._canon_cache.clear()


# --------------------------------------------------------------------------
# Expression-level canonicalization (sum of integrals -> sum over reps)
# --------------------------------------------------------------------------

# SAILIR represents an expression (a linear combination of integrals) as a
# dict { integral_tuple : coeff } with `coeff` an int in [0, PRIME).
ExprDict = Dict[Tuple[int, ...], int]


def canonicalize_expr(
    expr: ExprDict,
    group: SymmetryGroup,
    prime: int,
    order_key: OrderKey = default_order_key,
) -> ExprDict:
    """Replace each integral in `expr` with its orbit rep and merge.

    Since the integral identity I_source[a] == I_target[apply_ing(a)] is
    sign-free (Kira's `det` is metadata, not an integral-level sign), the
    coefficient passes through unchanged when we substitute the rep:

        sum_i  c_i * I[a_i]
            ==  sum_i  c_i * I[canon(a_i)]
            ==  sum_r  (sum_{i: canon(a_i)=r}  c_i)  *  I[r]      (mod prime)

    Zero-coefficient terms in the result are dropped. Input is not mutated.
    """
    out: ExprDict = {}
    for integral, coeff in expr.items():
        c = coeff % prime
        if c == 0:
            continue
        rep = group.canonicalize(integral, order_key=order_key)
        prev = out.get(rep, 0)
        out[rep] = (prev + c) % prime
    return {k: v for k, v in out.items() if v != 0}
