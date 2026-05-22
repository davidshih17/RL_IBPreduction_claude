"""
Topology configuration for SAILIR.

A Topology object captures everything family-specific that SAILIR's
data generation, training, and evaluation pipelines need:
  * n_indices, n_denominators, isp_positions      — index / sector geometry
  * family_name, kinematic_invariants             — for parsing IBP/LI files
  * kinematics_values                              — default numeric values
  * ibp_templates, li_templates                    — action templates
  * masters, masters_by_sector                     — basis of master integrals

A topology is loaded from a topology_input/<family>/ directory by
Topology.from_dir(path). The directory must contain:
    integralfamilies.yaml  (Kira-format family definition)
    kinematics.yaml         (Kira-format kinematics)
    IBP                     (Kira's IBP templates)
    LI                      (Kira's LI templates)
    masters                 (Kira's master basis)

See topology_input/HOW_TO_DERIVE_FROM_KIRA.md for how to produce these
from Kira for a new integral family.

Module-level convenience: configure(topology) sets a process-wide
current() topology that ibp_env and friends consult, so we don't need
to thread the topology argument through every function signature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# A single identity is a list of (shift_tuple, coeff_str) pairs.
Identity = List[Tuple[Tuple[int, ...], str]]


@dataclass
class Topology:
    name: str                                    # directory name (e.g. "trianglebox")
    family_name: str                              # parser symbol (e.g. "trianglebox", "TA")
    n_indices: int                                # length of integral tuple
    n_denominators: int                           # bits in sector mask (= popcount of top sector)
    isp_positions: Tuple[int, ...]                # 0-indexed positions that are ISPs
    kinematic_invariants: List[str]              # ['m2','m3'] or ['s12','s23','s34','s45','s51']
    kinematics_values: Dict[str, int]             # default numeric values for evaluation
    ibp_templates: List[Identity]                 # parsed IBP identities (order matters)
    li_templates: List[Identity]                  # parsed LI identities
    masters: List[Tuple[int, ...]]                # flat list of master tuples
    masters_by_sector: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)

    # -- derived properties --
    @property
    def n_ibp(self) -> int:
        return len(self.ibp_templates)

    @property
    def n_li(self) -> int:
        return len(self.li_templates)

    @property
    def n_actions(self) -> int:
        return self.n_ibp + self.n_li

    @property
    def top_sector(self) -> int:
        """Bitmask of the top sector (all denominators present)."""
        mask = 0
        for i in range(self.n_indices):
            if i not in self.isp_positions:
                mask |= (1 << i)
        return mask

    # -- common helpers (avoid duplicated logic across other modules) --
    def sector_of(self, integral: Tuple[int, ...]) -> int:
        mask = 0
        for i, a in enumerate(integral):
            if a > 0:
                mask |= (1 << i)
        return mask

    def corner_of(self, sector_id: int) -> Tuple[int, ...]:
        return tuple(1 if (sector_id >> i) & 1 else 0 for i in range(self.n_indices))

    def is_corner(self, integral: Tuple[int, ...]) -> bool:
        for i, a in enumerate(integral):
            if i in self.isp_positions:
                if a != 0:
                    return False
            else:
                if a != 0 and a != 1:
                    return False
        return True

    @classmethod
    def from_dir(cls, path, kinematics_values: Optional[Dict[str, int]] = None) -> "Topology":
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"topology dir not found: {path}")
        intfam = _read_integralfamilies_yaml(path / "integralfamilies.yaml")
        family_name = intfam["name"]
        propagators = intfam["propagators"]
        n_indices = len(propagators)

        top_level_sectors = intfam.get("top_level_sectors", [])
        if top_level_sectors:
            top = _parse_sector_literal(top_level_sectors[0])
        else:
            top = (1 << n_indices) - 1  # fallback: all positions denom
        n_denominators = bin(top).count("1")
        isp_positions = tuple(i for i in range(n_indices) if not ((top >> i) & 1))

        kin_invariants, replace_by_one = _read_kinematic_invariants(path / "kinematics.yaml")

        if kinematics_values is None:
            # Reasonable defaults: d=41 and a few small primes for invariants;
            # if Kira's config set symbol_to_replace_by_one, pin that to 1.
            primes = [31, 47, 53, 59, 61, 67, 71, 73, 79, 83]
            kinematics_values = {"d": 41}
            j = 0
            for name in kin_invariants:
                if name == replace_by_one:
                    kinematics_values[name] = 1
                else:
                    kinematics_values[name] = primes[j % len(primes)]
                    j += 1

        ibp_templates = parse_identity_file(path / "IBP", family_name)
        li_templates = parse_identity_file(path / "LI", family_name)
        masters_list = parse_masters_file(path / "masters", family_name)
        masters_by_sector = _group_by_sector(masters_list)

        return cls(
            name=path.name,
            family_name=family_name,
            n_indices=n_indices,
            n_denominators=n_denominators,
            isp_positions=isp_positions,
            kinematic_invariants=kin_invariants,
            kinematics_values=kinematics_values,
            ibp_templates=ibp_templates,
            li_templates=li_templates,
            masters=masters_list,
            masters_by_sector=masters_by_sector,
        )


# ---------------------------------------------------------------------------
# YAML parsing — light-weight, doesn't require pyyaml. We only need a few
# specific fields and the file format is small and predictable.
# ---------------------------------------------------------------------------

def _read_integralfamilies_yaml(path: Path) -> dict:
    """Minimal parser for the bits of integralfamilies.yaml we need.

    Extracts:
      name (str), top_level_sectors (list[int|str]), propagators (list)
    """
    text = path.read_text()
    name_match = re.search(r'-\s*name\s*:\s*"?([^"\s]+)"?', text)
    name = name_match.group(1) if name_match else "unknown"

    # top_level_sectors: e.g. "top_level_sectors: [63]" or "top_level_sectors: [b011111100]"
    tls_match = re.search(r'top_level_sectors\s*:\s*\[([^\]]*)\]', text)
    top_level_sectors: List = []
    if tls_match:
        for tok in tls_match.group(1).split(','):
            tok = tok.strip().strip('"').strip("'")
            if not tok or tok.startswith('#'):
                continue
            top_level_sectors.append(tok)

    # Count propagators by scanning lines: after "propagators:" header, every
    # subsequent line that (a) is more indented than the header and (b) starts
    # with "- [" is a propagator entry. We stop on the first dedented non-blank
    # line. This is robust to oddities like quoted masses ("0") or missing
    # trailing spaces before ].
    propagators: List = []
    in_props = False
    header_indent = -1
    for line in text.splitlines():
        stripped = line.strip()
        if not in_props:
            if re.match(r'^\s*propagators\s*:\s*$', line):
                in_props = True
                header_indent = len(line) - len(line.lstrip())
            continue
        if not stripped:
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= header_indent and not stripped.startswith('-'):
            # dedented to a sibling key, stop
            break
        if stripped.startswith('- ['):
            propagators.append(stripped)

    return {
        "name": name,
        "top_level_sectors": top_level_sectors,
        "propagators": propagators,
    }


def _parse_sector_literal(s) -> int:
    """Accept Kira sector forms: int 63, str "63", or "b00111111"."""
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if s.startswith('b'):
        return int(s[1:], 2)
    return int(s)


def _read_kinematic_invariants(path: Path) -> Tuple[List[str], Optional[str]]:
    """Extract invariant symbol names and the replace-by-one symbol.

    Returns (names, replace_by_one_symbol_or_None). For SAILIR's
    finite-field evaluation we set the replace-by-one symbol to value 1.
    """
    text = path.read_text()
    names: List[str] = []
    in_block = False
    header_indent = -1
    for line in text.splitlines():
        if not in_block:
            if re.match(r'^\s*kinematic_invariants\s*:\s*$', line):
                in_block = True
                header_indent = len(line) - len(line.lstrip())
            continue
        if not line.strip():
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= header_indent and not line.strip().startswith('-'):
            break
        m = re.search(r'-\s*\[\s*([A-Za-z_]\w*)', line)
        if m:
            names.append(m.group(1))

    rep = None
    rep_m = re.search(r'symbol_to_replace_by_one\s*:\s*([A-Za-z_]\w*)', text)
    if rep_m:
        rep = rep_m.group(1)
    return names, rep


# ---------------------------------------------------------------------------
# IBP / LI / masters parsers
# ---------------------------------------------------------------------------

def parse_identity_file(path: Path, family_name: str) -> List[Identity]:
    """Parse Kira-format IBP or LI file.

    Each blank-line-separated block is one identity; each line is
    `<family_name>[i0,i1,...,in-1]*(coeff_str)`.
    """
    pattern = re.compile(
        rf'^\s*{re.escape(family_name)}\s*\[\s*([^\]]+?)\s*\]\s*\*\s*\(\s*(.+?)\s*\)\s*$'
    )
    identities: List[Identity] = []
    current: Identity = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line.strip():
                if current:
                    identities.append(current)
                    current = []
                continue
            m = pattern.match(line)
            if m:
                shift = tuple(int(x.strip()) for x in m.group(1).split(','))
                coeff = m.group(2).strip()
                current.append((shift, coeff))
    if current:
        identities.append(current)
    return identities


def parse_masters_file(path: Path, family_name: str) -> List[Tuple[int, ...]]:
    """Parse Kira's masters file (one `<family_name>[a,b,...]  # sec` per line)."""
    pattern = re.compile(rf'^\s*{re.escape(family_name)}\s*\[\s*([^\]]+?)\s*\]')
    out: List[Tuple[int, ...]] = []
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                out.append(tuple(int(x.strip()) for x in m.group(1).split(',')))
    return out


def _group_by_sector(masters: List[Tuple[int, ...]]) -> Dict[int, List[Tuple[int, ...]]]:
    by_sec: Dict[int, List[Tuple[int, ...]]] = {}
    for m in masters:
        s = sum((1 << i) for i, a in enumerate(m) if a > 0)
        by_sec.setdefault(s, []).append(m)
    return by_sec


# ---------------------------------------------------------------------------
# Module-level current topology
# ---------------------------------------------------------------------------

_current: Optional[Topology] = None


def configure(topology: Topology) -> None:
    global _current
    _current = topology


def current() -> Topology:
    if _current is None:
        raise RuntimeError(
            "No topology configured. Call sailir.topology.configure(...) at "
            "process startup with a Topology loaded from topology_input/<family>/."
        )
    return _current
