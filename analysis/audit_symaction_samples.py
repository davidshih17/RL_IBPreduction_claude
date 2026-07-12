#!/usr/bin/env python
"""Audit a --sym-actions data-gen output: how many samples offer/choose symmetry
actions, sector coverage, and basic sanity (sym op indices, zero deltas).
Usage: audit_symaction_samples.py <jsonl>"""
import sys, json

n = n_offer = n_chosen = bad_delta = 0
sectors = set(); sym_offer_counts = []
for line in open(sys.argv[1]):
    s = json.loads(line)
    n += 1
    sectors.add(s["sector_id"])
    base = s.get("n_base_ops", 0)
    if not base:
        continue
    sym_acts = [a for a in s["valid_actions"] if a[0] >= base]
    if sym_acts:
        n_offer += 1
        sym_offer_counts.append(len(sym_acts))
        # zero delta = DIRECT (seeded at the target); nonzero = INDIRECT (seeded
        # at a substituted integral, like indirect IBP actions) — both legitimate
        bad_delta += sum(1 for a in sym_acts if any(x != 0 for x in a[1]))
    if s["chosen_action"][0] >= base:
        n_chosen += 1

print(f"samples: {n}, sectors covered: {len(sectors)}")
print(f"samples OFFERING >=1 symmetry action: {n_offer} ({100*n_offer/max(1,n):.1f}%)")
if sym_offer_counts:
    sym_offer_counts.sort()
    print(f"  sym actions offered per such sample: median "
          f"{sym_offer_counts[len(sym_offer_counts)//2]}, max {sym_offer_counts[-1]}")
print(f"samples where the CHOSEN action is a symmetry op: {n_chosen} "
      f"({100*n_chosen/max(1,n):.1f}%)")
print(f"symmetry actions that are INDIRECT (nonzero delta, legitimate): {bad_delta}")
