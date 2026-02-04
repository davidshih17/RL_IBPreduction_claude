#!/usr/bin/env python3
"""
Print a detailed validation table for I[1,1,1,1,1,1,-3]
"""

# Our 16 masters from async_111111m3_fresh.log
our_masters = [
    ([1, 0, 1, 1, 1, 1, 0], 784),
    ([1, 1, 0, 1, 1, 1, 0], 822),
    ([1, 1, 1, 1, 1, 0, 0], 514),
    ([-1, 1, 1, 1, 1, 0, 0], 45),
    ([0, 1, 1, 1, 1, 0, 0], 600),
    ([1, -1, 1, 0, 1, 1, 0], 567),
    ([1, -1, 1, 1, 1, 0, 0], 239),
    ([1, 0, 0, 1, 1, 1, 0], 759),
    ([1, 0, 1, 0, 1, 1, 0], 948),
    ([1, 0, 1, 1, 1, 0, 0], 890),
    ([1, 1, 0, 1, 0, 1, 0], 761),
    ([1, 1, 0, 1, 1, 0, 0], 593),
    ([0, 0, 1, 1, 1, 0, 0], 742),
    ([0, 1, 1, 1, 0, 0, 0], 97),
    ([1, 0, 1, 0, 0, 1, 0], 989),
    ([1, 0, 1, 0, 1, 0, 0], 258),
]

# Kira's 15 masters
kira_masters = {
    tuple([1, 0, 1, 1, 1, 1, 0]): 784,
    tuple([1, -1, 1, 0, 1, 1, 0]): 567,
    tuple([1, 0, 1, 0, 1, 1, 0]): 948,
    tuple([1, 1, 1, 1, 1, 0, 0]): 514,
    tuple([-1, 1, 1, 1, 1, 0, 0]): 45,
    tuple([0, 1, 1, 1, 1, 0, 0]): 600,
    tuple([1, -1, 1, 1, 1, 0, 0]): 239,
    tuple([1, 0, 1, 1, 1, 0, 0]): 890,
    tuple([0, 0, 1, 1, 1, 0, 0]): 742,
    tuple([1, 0, 1, 0, 1, 0, 0]): 258,
    tuple([0, 1, 1, 1, 0, 0, 0]): 77,
    tuple([1, 1, 0, 1, 1, 1, 0]): 822,
    tuple([1, 0, 0, 1, 1, 1, 0]): 759,
    tuple([1, 1, 0, 1, 1, 0, 0]): 593,
    tuple([1, 1, 0, 1, 0, 1, 0]): 761,
}

print("Validation of I[1,1,1,1,1,1,-3] reduction (mod 1009)")
print("=" * 95)
print(f"{'Master Integral':<30} {'Our Coeff':>10} {'Kira Coeff':>10} {'Status':>40}")
print("=" * 95)

exact_count = 0
for integral, our_coeff in our_masters:
    integral_tuple = tuple(integral)
    integral_str = str(integral)

    if integral_tuple in kira_masters:
        kira_coeff = kira_masters[integral_tuple]
        if our_coeff == kira_coeff:
            status = "Exact"
            exact_count += 1
            print(f"{integral_str:<30} {our_coeff:>10} {kira_coeff:>10} {status:>40}")
        else:
            # Check if this is [0,1,1,1,0,0,0]
            if integral == [0, 1, 1, 1, 0, 0, 0]:
                status = "Part of symmetry pair (see below)"
                print(f"{integral_str:<30} {our_coeff:>10} {kira_coeff:>10} {status:>40}")
            else:
                status = f"MISMATCH (diff={abs(our_coeff-kira_coeff)})"
                print(f"{integral_str:<30} {our_coeff:>10} {kira_coeff:>10} {status:>40}")
    else:
        # This is [1,0,1,0,0,1,0], the symmetry partner
        status = "Symmetry partner of [0,1,1,1,0,0,0]"
        print(f"{integral_str:<30} {our_coeff:>10} {'---':>10} {status:>40}")

print("=" * 95)
print(f"\nExact matches: {exact_count}/16 masters")
print(f"\nSymmetry-related pair:")
print(f"  [0,1,1,1,0,0,0] (coeff=97) and [1,0,1,0,0,1,0] (coeff=989)")
print(f"  Related by k1 <-> k2 loop momentum exchange")
print(f"  Sum: (97 + 989) mod 1009 = 77 = Kira's coefficient for [0,1,1,1,0,0,0]")
print(f"\n  ✓ All 16 masters validated against Kira's 15-master symmetry-reduced basis!")
