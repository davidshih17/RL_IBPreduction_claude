"""No-subs variant of the SAILIR action classifier.

Mirrors `IBPActionClassifier` from `sailir/classifier.py` but drops the
`FullSubstitutionEncoder` entirely. The state vector that gets broadcast to
the cross-attention scorer is built from `cls_pooled`, `target_pooled`, and
`sector_emb` only (3 channels instead of 4), so `state_combine` is
`3*embed_dim -> embed_dim`.

Motivation: the trained pentagonbox-10x classifier was found to be
content-insensitive to subs (zero-content random subs and shuffled subs both
produce ~bit-identical predictions). The subs encoder is ~38% of total
parameters and most of the cost beyond `expr_enc`. This variant strips it out
so we can quantify size, memory, and step-time savings, and provide a
candidate replacement architecture to retrain.

Shares the encoder building blocks (TransformerExpressionEncoderWithTarget,
SectorEncoder, ActionEncoder, CrossAttentionScorer) with the full model by
importing them from `sailir.classifier`. The shared modules are not duplicated
or modified.

forward() takes a strict subset of the full model's arguments — the sub_*
tensors are NOT in the signature. Drive selection through the
`--model_variant` flag in train_classifier.py rather than by trying to load
weights across variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# train_classifier.py loads this module after inserting `sailir/` onto
# sys.path, in which case `classifier` is the right name. The eval/benchmark
# scripts add the project root, in which case `sailir.classifier` is the
# discoverable name. Handle both.
try:
    from sailir.classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        ActionEncoder,
        CrossAttentionScorer,
    )
except ImportError:
    from classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        ActionEncoder,
        CrossAttentionScorer,
    )


class IBPActionClassifierNoSubs(nn.Module):
    """SAILIR action classifier without the subs encoder.

    Constructor accepts the same kwargs as `IBPActionClassifier` for symmetry;
    `n_subs_layers` is accepted but ignored.
    """

    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 n_subs_layers=2, *, prime, n_indices=7, n_denominators=6,
                 n_ibp_ops=9, **kwargs):
        super().__init__()
        self.prime = prime
        self.embed_dim = embed_dim
        self.n_indices = n_indices
        self.n_denominators = n_denominators
        del n_subs_layers  # explicitly noted as ignored

        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers,
            n_indices=n_indices, **kwargs,
        )
        self.sector_enc = SectorEncoder(embed_dim, n_denominators=n_denominators)
        self.action_enc = ActionEncoder(embed_dim, n_indices=n_indices,
                                         n_ibp_ops=n_ibp_ops, **kwargs)

        # State combine: cls + target + sector -> embed_dim (no subs channel).
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
        # sub_* positional args accepted for call-site parity with
        # IBPActionClassifier; they are NOT read.
        del sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask

        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True,
        )
        sector_emb = self.sector_enc(sector_mask)

        state_emb = self.state_combine(
            torch.cat([cls_pooled, target_pooled, sector_emb], dim=-1)
        )
        action_emb = self.action_enc(action_ibp_ops, action_deltas)
        logits = self.scorer(state_emb, action_emb, expr_terms, expr_mask, action_mask)
        return logits, F.softmax(logits, dim=-1)

    def predict(self, *args, **kwargs):
        logits, _ = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)
