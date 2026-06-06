"""SAILIR v3: action = post-sub equation; no subs as direct input anywhere.

Architecture summary:
  Inputs to the policy pi(a|s):
    s = (expression terms, target_integral, sector_mask)
        — same as `nosubs`; subs encoder removed entirely.
    a = the transitively-resolved post-substitution equation that this action
        produces, i.e. the set of (integral, coefficient) terms after applying
        all accumulated subs to the raw IBP relation. The (ibp_op, delta)
        handle is NOT used.

  Motivation: in v1/v2 (`full`, `subs_xattn`), each action is given to the
  model as a compact handle (ibp_op, delta). The model has no way to know
  *what equation that action actually produces* without integrating subs.
  v3 hands the model the resolved equation directly: the action representation
  IS the equation. Subs are no longer needed as a separate input because the
  resolved action already reflects all transitively-applied substitutions.

  Equation encoding mirrors how `FullSubstitutionEncoder.encode_single_substitution`
  pools a sub's replacement-term set: encode each (integral, coefficient) term,
  attention-pool to one vector per action.

Trained via `--model_variant v3`. The on-the-fly action-equation packing
lives in `scripts/train/train_classifier.py` (and parallel in
`scripts/eval/ablate_subs.py`), not in this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sailir.classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        CrossAttentionScorer,
        IntegralEncoder,
        CoefficientEncoder,
    )
except ImportError:
    from classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        CrossAttentionScorer,
        IntegralEncoder,
        CoefficientEncoder,
    )


class ActionEquationEncoder(nn.Module):
    """Encode an action's post-sub equation as a set of (integral, coefficient) terms.

    Pooling pattern mirrors `FullSubstitutionEncoder.encode_single_substitution`:
    encode each term, then attention-pool with a learned query to one vector
    per action. Returns (B, A, embed_dim).
    """

    def __init__(self, embed_dim=256, max_index=20, min_index=-10, prime=2147483647,
                 n_heads=4, n_indices=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.integral_enc = IntegralEncoder(
            embed_dim // 2, max_index, min_index, n_indices=n_indices,
        )
        self.coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)
        self.term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, eq_integrals, eq_coeffs, eq_mask):
        # eq_integrals: (B, A, T, n_indices) long
        # eq_coeffs:    (B, A, T) long
        # eq_mask:      (B, A, T) bool
        B, A, T = eq_coeffs.shape
        device = eq_integrals.device

        int_emb = self.integral_enc(eq_integrals)         # (B, A, T, ed/2)
        coeff_emb = self.coeff_enc(eq_coeffs)             # (B, A, T, ed/2)
        term_emb = self.term_proj(
            torch.cat([int_emb, coeff_emb], dim=-1)
        )                                                  # (B, A, T, ed)

        flat_term = term_emb.view(B * A, T, self.embed_dim)
        flat_mask = eq_mask.view(B * A, T)
        query = self.pool_query.expand(B * A, -1, -1)

        # Attention-pool only over actions that have at least one valid term.
        # Actions with no terms get a zero embedding (avoids NaN in attention).
        has_terms = flat_mask.any(dim=1)
        out = torch.zeros(B * A, self.embed_dim, device=device)
        if has_terms.any():
            valid_idx = has_terms.nonzero(as_tuple=True)[0]
            pooled, _ = self.pool_attn(
                query[valid_idx], flat_term[valid_idx], flat_term[valid_idx],
                key_padding_mask=~flat_mask[valid_idx],
            )
            out[valid_idx] = pooled.squeeze(1)
        return out.view(B, A, self.embed_dim)


class IBPActionClassifierV3(nn.Module):
    """SAILIR v3 action classifier.

    No subs encoder. No (ibp_op, delta) handle. Action embedding comes from
    the post-substitution equation directly.

    Constructor accepts the same kwargs as the other variants for symmetry;
    `n_subs_layers` and `n_ibp_ops` are accepted but not used.
    """

    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 n_subs_layers=2, prime=2147483647, n_indices=7, n_denominators=6,
                 n_ibp_ops=9, **kwargs):
        super().__init__()
        self.prime = prime
        self.embed_dim = embed_dim
        self.n_indices = n_indices
        self.n_denominators = n_denominators
        del n_subs_layers, n_ibp_ops  # accepted for API symmetry, unused

        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers,
            n_indices=n_indices, **kwargs,
        )
        self.sector_enc = SectorEncoder(embed_dim, n_denominators=n_denominators)
        self.action_enc = ActionEquationEncoder(
            embed_dim, prime=prime, n_heads=n_heads, n_indices=n_indices, **kwargs,
        )

        # state_combine: cls + target + sector -> embed_dim (no subs channel).
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral,
                action_eq_integrals, action_eq_coeffs, action_eq_mask):
        # Accepted (for call-site parity) but unused: sub_* and the (ibp_op,
        # delta) action handle. Action embedding comes from the post-sub
        # equation; subs are not used as direct input.
        del sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask
        del action_ibp_ops, action_deltas

        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True,
        )
        sector_emb = self.sector_enc(sector_mask)

        state_emb = self.state_combine(
            torch.cat([cls_pooled, target_pooled, sector_emb], dim=-1)
        )
        action_emb = self.action_enc(
            action_eq_integrals, action_eq_coeffs, action_eq_mask,
        )
        logits = self.scorer(state_emb, action_emb, expr_terms, expr_mask, action_mask)
        return logits, F.softmax(logits, dim=-1)

    def predict(self, *args, **kwargs):
        logits, _ = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)
