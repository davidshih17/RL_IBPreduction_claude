#!/usr/bin/env python3
"""
Compare coefficients for I[1,1,1,1,1,1,-3] between our async reduction and Kira
"""

# Our 16 masters from async_111111m3_fresh.log
our_masters = {
    (1, 0, 1, 1, 1, 1, 0): 784,
    (1, 1, 0, 1, 1, 1, 0): 822,
    (1, 1, 1, 1, 1, 0, 0): 514,
    (-1, 1, 1, 1, 1, 0, 0): 45,
    (0, 1, 1, 1, 1, 0, 0): 600,
    (1, -1, 1, 0, 1, 1, 0): 567,
    (1, -1, 1, 1, 1, 0, 0): 239,
    (1, 0, 0, 1, 1, 1, 0): 759,
    (1, 0, 1, 0, 1, 1, 0): 948,
    (1, 0, 1, 1, 1, 0, 0): 890,
    (1, 1, 0, 1, 0, 1, 0): 761,
    (1, 1, 0, 1, 1, 0, 0): 593,
    (0, 0, 1, 1, 1, 0, 0): 742,
    (0, 1, 1, 1, 0, 0, 0): 97,  # Note: Kira has 77 for this!
    (1, 0, 1, 0, 0, 1, 0): 989,  # Kira doesn't have this
    (1, 0, 1, 0, 1, 0, 0): 258,
}

# Kira's 15 masters from numerics_1009_0.m
kira_masters = {
    (1, 0, 1, 1, 1, 1, 0): 784,
    (1, -1, 1, 0, 1, 1, 0): 567,
    (1, 0, 1, 0, 1, 1, 0): 948,
    (1, 1, 1, 1, 1, 0, 0): 514,
    (-1, 1, 1, 1, 1, 0, 0): 45,
    (0, 1, 1, 1, 1, 0, 0): 600,
    (1, -1, 1, 1, 1, 0, 0): 239,
    (1, 0, 1, 1, 1, 0, 0): 890,
    (0, 0, 1, 1, 1, 0, 0): 742,
    (1, 0, 1, 0, 1, 0, 0): 258,
    (0, 1, 1, 1, 0, 0, 0): 77,  # Note: We have 97 for this!
    (1, 1, 0, 1, 1, 1, 0): 822,
    (1, 0, 0, 1, 1, 1, 0): 759,
    (1, 1, 0, 1, 1, 0, 0): 593,
    (1, 1, 0, 1, 0, 1, 0): 761,
}

print("=" * 80)
print("Comparing I[1,1,1,1,1,1,-3] coefficients: Our result vs Kira")
print("=" * 80)

# Find exact matches
exact_matches = []
for integral, our_coeff in our_masters.items():
    if integral in kira_masters:
        kira_coeff = kira_masters[integral]
        if our_coeff == kira_coeff:
            exact_matches.append(integral)

print(f"\nExact matches: {len(exact_matches)}/{len(kira_masters)}")
for integral in sorted(exact_matches):
    print(f"  {list(integral)}: {our_masters[integral]}")

# Find discrepancies
print("\n" + "=" * 80)
print("Discrepancies:")
print("=" * 80)

discrepancies = []
for integral in our_masters:
    if integral in kira_masters:
        if our_masters[integral] != kira_masters[integral]:
            discrepancies.append(integral)
            print(f"  {list(integral)}")
            print(f"    Our coeff:  {our_masters[integral]}")
            print(f"    Kira coeff: {kira_masters[integral]}")
            print(f"    Difference: {(our_masters[integral] - kira_masters[integral]) % 1009}")

if not discrepancies:
    print("  None among common integrals!")

# Integrals only in our result
print("\n" + "=" * 80)
print("Integrals only in our result:")
print("=" * 80)
only_ours = [i for i in our_masters if i not in kira_masters]
for integral in sorted(only_ours):
    print(f"  {list(integral)}: {our_masters[integral]}")

# Integrals only in Kira's result
print("\n" + "=" * 80)
print("Integrals only in Kira's result:")
print("=" * 80)
only_kira = [i for i in kira_masters if i not in our_masters]
for integral in sorted(only_kira):
    print(f"  {list(integral)}: {kira_masters[integral]}")

# Check if [1,0,1,0,0,1,0] (ours) and [0,1,1,1,0,0,0] (both) might be related
print("\n" + "=" * 80)
print("Investigating potential symmetry:")
print("=" * 80)

our_unique = (1, 0, 1, 0, 0, 1, 0)
disputed = (0, 1, 1, 1, 0, 0, 0)

print(f"\nOur unique master: {list(our_unique)} with coeff {our_masters[our_unique]}")
print(f"Disputed master:   {list(disputed)}")
print(f"  Our coeff:  {our_masters[disputed]}")
print(f"  Kira coeff: {kira_masters[disputed]}")

print("\nChecking if they're related by k1 <-> k2 symmetry...")
# For k1 <-> k2: indices 0,1,3,5 swap
def k1_k2_swap(indices):
    result = list(indices)
    result[0], result[1] = result[1], result[0]  # k1^2 <-> k2^2
    result[3], result[5] = result[5], result[3]  # (k1+p1)^2 <-> (k2-p1)^2
    return tuple(result)

swapped_unique = k1_k2_swap(our_unique)
swapped_disputed = k1_k2_swap(disputed)

print(f"  {list(our_unique)} -> {list(swapped_unique)}")
print(f"  {list(disputed)} -> {list(swapped_disputed)}")

if swapped_unique == disputed:
    print(f"\n*** {list(our_unique)} IS related to {list(disputed)} by k1 <-> k2!")
    combined = (our_masters[our_unique] + our_masters[disputed]) % 1009
    print(f"    Our coeff for {list(disputed)}: {our_masters[disputed]}")
    print(f"    Our coeff for {list(our_unique)}: {our_masters[our_unique]}")
    print(f"    Sum: ({our_masters[disputed]} + {our_masters[our_unique]}) mod 1009 = {combined}")
    print(f"    Kira coeff for {list(disputed)}: {kira_masters[disputed]}")
    if combined == kira_masters[disputed]:
        print(f"\n    ✓ PERFECT MATCH! Kira uses symmetry-reduced basis.")
        print(f"      Our two symmetry-related masters sum to Kira's single master!")
