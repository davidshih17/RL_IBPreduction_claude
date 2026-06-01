"""Flat-array representation of the indirect-aux 4-tuple for shared-memory
parallel P4.

Aux layout (the dict-based original):
  (cu, ubm, rid, iraws)
  cu      : list of dict[integral_tuple -> int_coeff]
  ubm     : list of int (sector union bitmask per cu entry)
  rid     : dict[id(raw_eq_dict) -> cu_idx]   (process-local: id() is)
  iraws   : list of (sub_int, ibp_op, shift, raw_eq_dict_ref)

Flat layout:
  cu_offsets  : int64 (n_cu + 1,) — cu[i] occupies cu_keys[cu_offsets[i] : cu_offsets[i+1]]
  cu_lengths  : int64 (n_cu,)     — number of LIVE entries in cu[i] (a row may be shorter than its allocated slot if substitution shrank it)
  cu_keys     : int32 (n_keys_max, N_INDICES)  — concatenated integral keys
  cu_coeffs   : int64 (n_keys_max,)            — coefficients aligned to cu_keys
  ubm         : int64 (n_cu,)                  — sector union bitmask
  iraws_meta  : int32 (n_iraws, 2 + 2*N_INDICES) — (sub_int, ibp_op, shift) per iraws entry
                                                  layout: [sub_int(N_INDICES), ibp_op(1), shift(N_INDICES)]
                                                  raw_eq_dict is RECOMPUTED locally via env.get_raw_equation_cached((ibp_op, sub_int - shift))

  cu_keys / cu_coeffs are pre-allocated with EXTRA capacity beyond
  n_keys_total so substitutions can append new entries at the end without
  reshuffling earlier slots. cu_offsets / cu_lengths are updated to point
  the new region; the old slot becomes garbage and gets reclaimed at the
  next compact.

  rid is NOT stored in the flat aux because it's process-local (keyed by
  id()). Each process derives rid from iraws_meta + raw_eq_cache on first
  use.

  Rationale for flat:
  1. Numpy arrays in multiprocessing.shared_memory are zero-copy across
     processes. Workers map shm regions directly; no pickle.
  2. Phase A's hot loop can run in Cython with `nogil`, enabling real
     threading parallelism with no GIL contention.
  3. Substitutions only touch entries where sub_key is present; can be
     batched into a small "diff" payload (changed_cu_idx, new keys+coeffs)
     instead of shipping the whole aux back.
"""
from __future__ import annotations

import numpy as np

from sailir import ibp_env


class FlatAux:
    """Flat-array representation of the indirect-aux 4-tuple.

    Memory layout:
        cu_offsets:  int64 (n_cu+1,)   — start of cu[i] in cu_keys; cu_offsets[-1] is total used
        cu_lengths:  int64 (n_cu,)     — live entry count for cu[i]
        cu_keys:     int32 (capacity, n_indices)
        cu_coeffs:   int64 (capacity,)
        ubm:         int64 (n_cu,)
        iraws_meta:  int32 (n_iraws, 2*n_indices + 1)
                     columns: [sub_int (n_indices), ibp_op (1), shift (n_indices)]

    The cu_offsets array points to a sub-block of cu_keys/cu_coeffs for
    each cu entry. cu_lengths[i] is the number of LIVE entries (may be
    less than cu_offsets[i+1] - cu_offsets[i] if shrunk by substitution).
    """
    __slots__ = (
        'n_cu', 'n_iraws', 'capacity',
        'cu_offsets', 'cu_lengths',
        'cu_keys', 'cu_coeffs',
        'ubm', 'iraws_meta',
    )

    def __init__(self, n_cu, n_iraws, capacity,
                 cu_offsets, cu_lengths, cu_keys, cu_coeffs, ubm, iraws_meta):
        self.n_cu = n_cu
        self.n_iraws = n_iraws
        self.capacity = capacity
        self.cu_offsets = cu_offsets
        self.cu_lengths = cu_lengths
        self.cu_keys = cu_keys
        self.cu_coeffs = cu_coeffs
        self.ubm = ubm
        self.iraws_meta = iraws_meta

    @classmethod
    def from_dict_aux(cls, aux_tuple, n_indices=None):
        """Convert a dict-based aux 4-tuple to flat representation."""
        cu, ubm, _rid, iraws = aux_tuple
        if n_indices is None:
            n_indices = ibp_env.N_INDICES
        n_cu = len(cu)
        n_iraws = len(iraws)

        # Pass 1: compute total live entries
        cu_lengths_list = [len(c) for c in cu]
        n_keys_total = sum(cu_lengths_list)

        # Pre-allocate with 50% extra capacity for future substitutions.
        capacity = max(n_keys_total + n_keys_total // 2, 16)

        cu_offsets = np.zeros(n_cu + 1, dtype=np.int64)
        cu_lengths = np.array(cu_lengths_list, dtype=np.int64)
        cu_keys = np.zeros((capacity, n_indices), dtype=np.int32)
        cu_coeffs = np.zeros(capacity, dtype=np.int64)
        ubm_arr = np.array(ubm, dtype=np.int64) if len(ubm) > 0 else np.zeros(0, dtype=np.int64)

        # Pass 2: fill cu_keys/cu_coeffs
        offset = 0
        for i, c in enumerate(cu):
            cu_offsets[i] = offset
            for k_tuple, v in c.items():
                for j in range(n_indices):
                    cu_keys[offset, j] = k_tuple[j]
                cu_coeffs[offset] = v
                offset += 1
        cu_offsets[n_cu] = offset

        # iraws_meta
        iraws_meta = np.zeros((n_iraws, 2 * n_indices + 1), dtype=np.int32)
        for j, (sub_int, ibp_op, shift, _raw) in enumerate(iraws):
            for k in range(n_indices):
                iraws_meta[j, k] = sub_int[k]
            iraws_meta[j, n_indices] = ibp_op
            for k in range(n_indices):
                iraws_meta[j, n_indices + 1 + k] = shift[k]

        return cls(
            n_cu=n_cu, n_iraws=n_iraws, capacity=capacity,
            cu_offsets=cu_offsets, cu_lengths=cu_lengths,
            cu_keys=cu_keys, cu_coeffs=cu_coeffs,
            ubm=ubm_arr, iraws_meta=iraws_meta,
        )

    def to_dict_aux(self, env, n_indices=None):
        """Convert flat representation back to dict-based aux 4-tuple.

        Reconstructs raws + rid locally via env.get_raw_equation_cached.
        """
        if n_indices is None:
            n_indices = ibp_env.N_INDICES

        # Reconstruct cu as list of dicts
        cu = []
        for i in range(self.n_cu):
            off = int(self.cu_offsets[i])
            n = int(self.cu_lengths[i])
            d = {}
            for j in range(n):
                k = tuple(int(x) for x in self.cu_keys[off + j])
                v = int(self.cu_coeffs[off + j])
                d[k] = v
            cu.append(d)

        ubm = [int(x) for x in self.ubm]

        # Reconstruct iraws + rid from iraws_meta
        iraws = []
        rid = {}
        for j in range(self.n_iraws):
            row = self.iraws_meta[j]
            sub_int = tuple(int(x) for x in row[:n_indices])
            ibp_op = int(row[n_indices])
            shift = tuple(int(x) for x in row[n_indices + 1:])
            seed = tuple(sub_int[k] - shift[k] for k in range(n_indices))
            raw = env.get_raw_equation_cached(ibp_op, seed)
            iraws.append((sub_int, ibp_op, shift, raw))
            rid_key = id(raw)
            if rid_key not in rid:
                # Worker MUST assign cu_idx in iraws-iteration order to
                # match the original incremental code's order.
                rid[rid_key] = len(rid)

        return (cu, ubm, rid, iraws)
