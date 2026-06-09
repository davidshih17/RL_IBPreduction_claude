"""
Smoke test for the topology-parametrized classifier.

For each topology, instantiate IBPActionClassifier with (n_indices,
n_denominators) from the topology and run a forward pass with random
tensors of correct shape. Verify the model runs without error and
returns logits of expected shape.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from sailir.topology import Topology
from sailir.classifier import IBPActionClassifier


def forward_check(topo_name, n_indices, n_denominators, n_actions):
    print(f"=== {topo_name} (n_indices={n_indices}, n_denom={n_denominators}, "
          f"n_actions={n_actions}) ===")
    model = IBPActionClassifier(
        embed_dim=64, n_heads=2, n_expr_layers=1, n_cross_layers=1, n_subs_layers=1,
        prime=1009,
        n_indices=n_indices, n_denominators=n_denominators,
        n_ibp_ops=n_actions,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}")

    B, T, S, R, A = 2, 4, 3, 2, 5  # batch, expr terms, subs, repls, actions
    expr_integrals = torch.zeros(B, T, n_indices, dtype=torch.long)
    expr_coeffs = torch.zeros(B, T, dtype=torch.long)
    expr_mask = torch.ones(B, T, dtype=torch.bool)
    sub_keys = torch.zeros(B, S, n_indices, dtype=torch.long)
    sub_repl_ints = torch.zeros(B, S, R, n_indices, dtype=torch.long)
    sub_repl_coeffs = torch.zeros(B, S, R, dtype=torch.long)
    sub_repl_mask = torch.ones(B, S, R, dtype=torch.bool)
    sub_mask = torch.ones(B, S, dtype=torch.bool)
    action_ibp_ops = torch.zeros(B, A, dtype=torch.long)
    action_deltas = torch.zeros(B, A, n_indices, dtype=torch.long)
    action_mask = torch.ones(B, A, dtype=torch.bool)
    sector_mask = torch.zeros(B, n_denominators, dtype=torch.long)
    target_integral = torch.zeros(B, n_indices, dtype=torch.long)

    with torch.no_grad():
        logits, probs = model(
            expr_integrals, expr_coeffs, expr_mask,
            sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
            action_ibp_ops, action_deltas, action_mask,
            sector_mask, target_integral,
        )
    assert logits.shape == (B, A), logits.shape
    assert probs.shape == (B, A), probs.shape
    print(f"  forward pass OK: logits.shape={tuple(logits.shape)}, "
          f"probs.shape={tuple(probs.shape)}")
    print()


def main():
    tb = Topology.from_dir(ROOT / "topology_input/trianglebox")
    forward_check("trianglebox", tb.n_indices, tb.n_denominators, tb.n_actions)

    pb = Topology.from_dir(ROOT / "topology_input/pentagonbox")
    forward_check("pentagonbox", pb.n_indices, pb.n_denominators, pb.n_actions)

    print("CLASSIFIER REFACTOR SMOKE TEST: OK")


if __name__ == "__main__":
    main()
