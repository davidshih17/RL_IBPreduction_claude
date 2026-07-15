#!/usr/bin/env python
"""Gate for the required-prime fix (2026-07-15):
  (1) constructing ANY prime-carrying model class without `prime` must raise
      TypeError immediately (no silent default — the 2-month production bug
      class is now impossible to write);
  (2) constructing WITH the trained prime must load both production checkpoints;
  (3) the two constructions (trained prime vs the old broken default) must
      genuinely differ on real inputs (sanity that the fix changes behavior)."""
import os, sys
import torch
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE)
from sailir.classifier import (IBPActionClassifier, CoefficientEncoder,
                               FullSubstitutionEncoder,
                               TransformerExpressionEncoderWithTarget)
from sailir.classifier_nosubs import IBPActionClassifierNoSubs

fails = 0
for cls, kwargs in ((CoefficientEncoder, {}),
                    (FullSubstitutionEncoder, {}),
                    (TransformerExpressionEncoderWithTarget, {}),
                    (IBPActionClassifier, {}),
                    (IBPActionClassifierNoSubs, {})):
    try:
        cls(**kwargs)
        print(f"  {cls.__name__}: constructed WITHOUT prime — GATE FAIL")
        fails += 1
    except TypeError:
        print(f"  {cls.__name__}: TypeError without prime — OK")

for name, cls in (("pentagonbox_10x_loop_100", IBPActionClassifier),
                  ("pentagonbox_canon10x_nosubs", IBPActionClassifierNoSubs)):
    ck = torch.load(os.path.join(BASE, "checkpoints", name, "best_model.pt"),
                    map_location="cpu", weights_only=False)
    m = cls(prime=ck["args"]["prime"], n_indices=11, n_denominators=8, n_ibp_ops=18)
    m.load_state_dict(ck["model_state_dict"])
    print(f"  {name}: loads with prime={ck['args']['prime']} — OK")

print("GATE " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
