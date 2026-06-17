"""Shared-memory CSR packing of resolved_subs for cross-process parallelism.

resolved_subs is {sub_int: {integral: coeff}} (dict-of-dicts, ~0.5-0.9GB at
depth, the thing that blocks process-pool parallelism because shipping it per
step is too expensive). Here it is packed into 4 flat numpy arrays (CSR layout)
over an IntegralRegistry:

  sub_ids   : int32[n_subs]      sorted sub-integral ids
  offsets   : int64[n_subs+1]    CSR offsets into the flat pools
  flat_ids  : int32[total_terms] integral ids of every solution, concatenated
  flat_coeffs: int16[total_terms] coeffs, parallel to flat_ids

lookup(sub_id) = binary-search sub_ids -> slice flat pools. Membership and
solution lookup are both O(log n_subs). The 4 arrays can live in
multiprocessing.shared_memory so N worker processes attach READ-ONLY with zero
copy (no pickle, no fork-COW-refcount storm — the refcount is on 4 array objects,
not millions of dict/tuple objects).

Bit-identical to the dict form by construction (same ids+coeffs); verified via
to_dict() round-trip.
"""
import numpy as np
from multiprocessing import shared_memory


class PackedResolvedSubs:
    __slots__ = ('sub_ids', 'offsets', 'flat_ids', 'flat_coeffs', '_shms')

    def __init__(self, sub_ids, offsets, flat_ids, flat_coeffs, shms=None):
        self.sub_ids = sub_ids
        self.offsets = offsets
        self.flat_ids = flat_ids
        self.flat_coeffs = flat_coeffs
        self._shms = shms or []        # SharedMemory handles to keep alive

    # ---- build from the dict form ----
    @classmethod
    def from_dict(cls, resolved_subs, reg):
        items = sorted((reg.get_id(k), v) for k, v in resolved_subs.items())
        n = len(items)
        sub_ids = np.empty(n, np.int32)
        offsets = np.empty(n + 1, np.int64)
        offsets[0] = 0
        id_chunks = []
        co_chunks = []
        for i, (sid, sol) in enumerate(items):
            sub_ids[i] = sid
            m = len(sol)
            ids = np.fromiter((reg.get_id(k) for k in sol), np.int32, m)
            cos = np.fromiter((sol[k] for k in sol), np.int16, m)
            order = np.argsort(ids, kind='stable')
            id_chunks.append(ids[order])
            co_chunks.append(cos[order])
            offsets[i + 1] = offsets[i] + m
        flat_ids = (np.concatenate(id_chunks) if id_chunks
                    else np.empty(0, np.int32))
        flat_coeffs = (np.concatenate(co_chunks) if co_chunks
                       else np.empty(0, np.int16))
        return cls(sub_ids, offsets, flat_ids, flat_coeffs)

    def __len__(self):
        return len(self.sub_ids)

    @property
    def n_terms(self):
        return len(self.flat_ids)

    # ---- lookup ----
    def get(self, sub_id):
        """(ids, coeffs) views for sub_id, or None if not a sub key."""
        sids = self.sub_ids
        pos = int(np.searchsorted(sids, sub_id))
        if pos >= len(sids) or sids[pos] != sub_id:
            return None
        a = int(self.offsets[pos])
        b = int(self.offsets[pos + 1])
        return self.flat_ids[a:b], self.flat_coeffs[a:b]

    def contains(self, sub_id):
        sids = self.sub_ids
        pos = int(np.searchsorted(sids, sub_id))
        return pos < len(sids) and sids[pos] == sub_id

    # ---- verification: reconstruct the dict form ----
    def to_dict(self, reg):
        out = {}
        for i in range(len(self.sub_ids)):
            sub_int = reg.get_tuple(int(self.sub_ids[i]))
            a = int(self.offsets[i])
            b = int(self.offsets[i + 1])
            out[sub_int] = {reg.get_tuple(int(self.flat_ids[j])):
                            int(self.flat_coeffs[j]) for j in range(a, b)}
        return out

    # ---- shared memory ----
    def to_shared(self, tag):
        """Copy the 4 arrays into shared memory. Returns a descriptor dict
        (names/shapes/dtypes) that attach() turns back into a read-only view in
        a worker process. Keeps the SharedMemory handles alive on self._shms."""
        desc = {}
        shms = []
        for name, arr in (('sub_ids', self.sub_ids), ('offsets', self.offsets),
                          ('flat_ids', self.flat_ids),
                          ('flat_coeffs', self.flat_coeffs)):
            nbytes = max(arr.nbytes, 1)
            shm = shared_memory.SharedMemory(create=True, size=nbytes,
                                             name=f"{tag}_{name}")
            view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            view[:] = arr[:]
            desc[name] = (shm.name, arr.shape, str(arr.dtype))
            shms.append(shm)
        self._shms = shms
        return desc

    @classmethod
    def attach(cls, desc):
        """Attach (read-only view) to shared arrays in a worker process. The
        returned object holds the SharedMemory handles so the buffers stay
        mapped; caller must keep it alive for the duration of use."""
        arrs = {}
        shms = []
        for name in ('sub_ids', 'offsets', 'flat_ids', 'flat_coeffs'):
            shm_name, shape, dtype = desc[name]
            shm = shared_memory.SharedMemory(name=shm_name)
            arrs[name] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
            shms.append(shm)
        return cls(arrs['sub_ids'], arrs['offsets'], arrs['flat_ids'],
                   arrs['flat_coeffs'], shms=shms)

    def close_shared(self, unlink=True):
        for shm in self._shms:
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except (FileNotFoundError, BufferError):
                pass
        self._shms = []
