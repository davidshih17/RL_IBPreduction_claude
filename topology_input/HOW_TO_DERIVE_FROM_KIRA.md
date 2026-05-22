# Generating SAILIR Inputs from Kira

SAILIR's data generation needs three files per integral family:

| File      | What it is                                          | SAILIR usage                           |
|-----------|-----------------------------------------------------|----------------------------------------|
| `IBP`     | Integration-by-parts identity templates             | Action space (one action per identity) |
| `LI`      | Lorentz invariance identity templates               | Action space (one action per identity) |
| `masters` | Minimal basis of master integrals for the family    | Reduction targets / training anchors   |

All three are produced by **Kira** from the family's `integralfamilies.yaml` +
`kinematics.yaml`. This document describes how to do that end-to-end and how
to drop the outputs into a SAILIR `topology_input/` subdirectory.

---

## 1. Inputs you must provide

For a new integral family you need two YAML files:

### `integralfamilies.yaml`

Defines propagators, loop momenta, sector convention. Example
(pentagon-box, 11 propagators, 8 denominators + 3 ISPs):

```yaml
integralfamilies:
  - name: "TA"
    loop_momenta: [k1, k2]
    top_level_sectors: [255]      # all 8 denominators in denom = bits 0..7
    propagators:
      - ["k1",                0]   # D1
      - ["k1+p1",             0]   # D2
      - ["k1+p1+p2",          0]   # D3
      - ["k1+p1+p2+p3",       0]   # D4
      - ["k2",                0]   # D5
      - ["k2+p1+p2+p3",       0]   # D6
      - ["k2+p1+p2+p3+p4",    0]   # D7
      - ["k1-k2",             0]   # D8
      - ["k1+p1+p2+p3+p4",    0]   # D9  (ISP)
      - ["k2+p1",             0]   # D10 (ISP)
      - ["k2+p1+p2",          0]   # D11 (ISP)
```

Notes:
- The first N propagators listed are interpreted as physical denominators
  and form the basis of the **top sector**. `top_level_sectors` is the
  bitmask of the top sector (e.g., `255 = 0b11111111` = first 8 in denom).
- Auxiliary propagators (ISPs) come after. They should appear only with
  non-positive index in the basis, but in IBP intermediates they can
  transiently take positive values; Kira reduces those back.
- If you want Kira to enforce ISP positions globally, add e.g.
  `symbolic_ibp: [9, 10, 11]` (1-indexed positions). For the pentagon-box,
  we don't currently enforce this — Federica's setup omits it — but the
  resulting masters never put D9..D11 in the denominator anyway.

### `kinematics.yaml`

Defines external momenta and Mandelstam invariants. Example:

```yaml
kinematics:
  incoming_momenta: [p1, p2, p3, p4, p5]
  outgoing_momenta: []
  momentum_conservation: [p5, "-p1 - p2 - p3 - p4"]
  kinematic_invariants:
    - [s12, 2]
    - [s23, 2]
    - [s34, 2]
    - [s45, 2]
    - [s51, 2]
  scalarproduct_rules:
    - [[p1, p1], 0]
    - [[p2, p2], 0]
    - [[p3, p3], 0]
    - [[p4, p4], 0]
    - [[p1+p2, p1+p2], s12]
    - [[p1+p3, p1+p3], s45-s12-s23]
    # ...
```

The kinematic invariant names (here `s12 ... s51`) become symbol names in
the coefficients of the IBP/LI templates SAILIR consumes.

---

## 2. Picking the Kira job to actually run

To get the master basis, you need a **non-trivial** reduction. The exact
choice trades runtime for completeness. The pattern that worked for
trianglebox (and now pentagon-box) is:

```yaml
# jobs.yaml
jobs:
  - reduce_sectors:
      reduce:
        - {sectors: [TOP_SECTOR], r: TOP_T + 2, s: 4}
      select_integrals:
        select_mandatory_list:
          - [FAMILY_NAME, target_integrals]
      run_initiate: true
      run_triangular: true
      run_back_substitution: true
```

with a single target integral in the top sector that exercises a dot, e.g.:

```
# target_integrals  (file referenced above)
TA[2,1,1,1,1,1,1,1,0,0,0]
```

Key design rules:
- **The target must live in the top sector** (`t = N_DENOMINATORS`).
  Otherwise the reduction won't traverse all sub-sectors and masters
  in other branches will be missed.
- **The target must have at least one dot** (an index = 2). Without it,
  the corner integral is its own master (Kira returns 1 master, useless).
- **Seed bounds**: `r = target_r + 1` and `s = 4` are the values that
  reproduced the trianglebox paper's 15-master basis. They tend to be
  enough for the master basis to converge — Kira will keep iterating
  ("Regenerate" phase) until no new relations are found.

For pentagon-box this gave 61 masters in ~49 minutes on a single core.

A faster variant (only the basis count, no reduction coefficients):
omit `run_back_substitution` — but the masters file is still written
only after the full pipeline completes, so the savings depend on Kira's
implementation. The full pipeline is the safe choice.

---

## 3. Running Kira

```bash
export FERMATPATH=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/fire/FIRE7/extra/fuel/extra/ferl64/fer64

cd /path/to/family/work_dir   # contains config/, jobs.yaml, target_integrals
/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/kira-3.1 jobs.yaml
```

Always run in the background with unbuffered logging:

```bash
nohup bash run_kira.sh > kira_run.log 2>&1 &
```

Outputs of interest:
- `sectormappings/<FAMILY>/IBP`            — IBP templates
- `sectormappings/<FAMILY>/LI`             — LI templates
- `sectormappings/<FAMILY>/nonTrivialSector` — list of non-trivial sectors with t
- `results/<FAMILY>/masters`               — final master basis

---

## 4. Dropping into SAILIR

Create `topology_input/<family_name>/`, copy the three files plus the
config used:

```
topology_input/<family_name>/
  ├── IBP                          ← from Kira sectormappings/<FAMILY>/
  ├── LI                           ← from Kira sectormappings/<FAMILY>/
  ├── masters                      ← from Kira results/<FAMILY>/
  ├── integralfamilies.yaml        ← copy of Kira config (for reproducibility)
  ├── kinematics.yaml              ← copy of Kira config
  ├── kira_jobs.yaml               ← copy of the jobs.yaml that produced these
  └── kira_target_integrals        ← copy of the target list used
```

Then build the SAILIR-side master dict (sector_id → list of master tuples):

```bash
python build_<FAMILY>_masters_dict.py    # writes <FAMILY>_masters_dict.py
```

The dict shape mirrors trianglebox's `PAPER_MASTERS` — a flat mapping
from sector id to a list of 11-tuples (or whatever index count the
family has). It's used by `generate_multisector_data.py` to anchor
training trajectories.

---

## 5. Per-family parameters SAILIR needs

Beyond the three files, SAILIR needs to know a few topology constants
to dimension the model and the data structures correctly. These should
be set in the family's input directory (e.g., `topology_input/<FAMILY>/<FAMILY>_masters_dict.py`):

```python
ISP_POSITIONS  = (8, 9, 10)   # 0-indexed positions that are always ISPs
N_INDICES      = 11           # total index tuple length
N_DENOMINATORS = 8            # number of "physical" propagators
```

For trianglebox: `N_INDICES=7, N_DENOMINATORS=6, ISP_POSITIONS=(6,)`.

---

## 6. Sanity checks before running data-gen

1. `wc -l IBP` should give roughly `(# blank lines + 1) ≈ # of identities`.
   For pentagon-box: 13 identities.
2. `wc -l LI` similarly for LI identities. Pentagon-box: 7.
3. `wc -l masters` = number of master integrals in the basis. Cross-check
   against a literature reference or independent reduction.
4. Verify no master has positive index at any `ISP_POSITIONS` slot:
   ```bash
   python -c "
   import re
   bad = 0
   for ln in open('masters'):
       m = re.match(r'\\w+\\[([^\\]]+)\\]', ln)
       if not m: continue
       v = [int(x) for x in m.group(1).split(',')]
       if any(v[i] > 0 for i in (8,9,10)): bad += 1
   print('masters violating ISP convention:', bad)
   "
   ```
   Should print 0.

If any check fails, fix the Kira setup before generating training data —
SAILIR will silently produce garbage if the family definitions are
inconsistent.
