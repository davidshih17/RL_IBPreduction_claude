# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated inner of add_sub_to_resolved.

apply_sub_inner(value, target, resolved_sol, PRIME)
    Equivalent to:
        coeff = value.pop(target)
        for k, v in resolved_sol.items():
            new_coeff = (coeff * v) % PRIME
            if new_coeff == 0: continue
            if k in value:
                s = (value[k] + new_coeff) % PRIME
                if s == 0: del value[k]
                else:     value[k] = s
            else:
                value[k] = new_coeff
    where value, resolved_sol are dict[tuple, int], target is tuple,
    PRIME is small int. Pure dict ops + integer arithmetic.

    Returns: value (same dict, modified in place).
"""


def apply_sub_inner(dict value, target, dict resolved_sol, long PRIME):
    cdef long coeff
    cdef long v
    cdef long new_coeff
    cdef long s

    coeff = value.pop(target)
    for k in resolved_sol:
        v = <long> resolved_sol[k]
        new_coeff = (coeff * v) % PRIME
        if new_coeff == 0:
            continue
        if k in value:
            s = ((<long> value[k]) + new_coeff) % PRIME
            if s == 0:
                del value[k]
            else:
                value[k] = s
        else:
            value[k] = new_coeff
    return value
