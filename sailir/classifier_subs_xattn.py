"""SAILIR classifier with subs entering only through per-action cross-attention.

Motivation: in `IBPActionClassifier`, the subs path produces a single 256-d
pooled vector that gets concatenated with cls/target/sector into a global
`state_emb`, broadcast identically to all actions. Inspection of the trained
weights (see [[project-sailir-phase2-subs-finding]]) shows that the subs
column of `state_combine` collapsed to an effectively rank-1 readout — the
model uses ~1 scalar feature from subs and discards the rest. The
`ablate_subs.py` content-randomization test confirms this empirically: random
sub contents produce bit-identical predictions.

Hypothesis: subs information is being squashed by the architecture, not
because it's redundant. Specifically, (a) all actions share the same pooled
summary, and (b) actions can't look up *specific* subs that matter to them.

This variant changes the architecture so each action can cross-attend over
the per-sub embeddings directly. The pooled-into-state pathway is removed
entirely; all sub information has to flow through the per-action attention
surface. Together with a single per-token type embedding to distinguish
expression terms from subs, this gives subs the same per-action retrieval
interface that expression terms already have.

If after a few epochs the random-subs ablation shows meaningful KL > 0 (vs
the existing pentagonbox_10x_loop_100 checkpoint's KL ≈ 0), then the model
is using sub content. If KL stays ~0, the sub information is genuinely
redundant given the rest of the input — switch to `nosubs` permanently.

Wires up at the call site identically to `IBPActionClassifier` (same forward
signature). Train via `--model_variant subs_xattn`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same dual-context import dance as classifier_nosubs.py.
try:
    from sailir.classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        ActionEncoder,
        FullSubstitutionEncoder,
    )
except ImportError:
    from classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        ActionEncoder,
        FullSubstitutionEncoder,
    )


class SubsEncoderPerSub(FullSubstitutionEncoder):
    """Variant of `FullSubstitutionEncoder` that returns per-sub embeddings.

    Identical to the parent up through the transformer over the sub sequence;
    skips the final attention pool. Returns (B, max_subs, embed_dim).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We never call these — drop them so they don't bloat the param count
        # or DDP all-reduce traffic.
        del self.final_pool_query
        del self.final_pool_attn

    def forward(self, sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask):
        batch_size, max_subs, max_repl, _ = sub_repl_ints.shape
        device = sub_mask.device

        if not sub_mask.any():
            return torch.zeros(batch_size, max_subs, self.embed_dim, device=device)

        n = self.n_indices
        flat_sub_embs = self.encode_single_substitution(
            sub_keys.view(batch_size * max_subs, n),
            sub_repl_ints.view(batch_size * max_subs, max_repl, n),
            sub_repl_coeffs.view(batch_size * max_subs, max_repl),
            sub_repl_mask.view(batch_size * max_subs, max_repl),
        )
        all_sub_embs = flat_sub_embs.view(batch_size, max_subs, self.embed_dim)
        all_sub_embs = all_sub_embs * sub_mask.unsqueeze(-1).float()
        all_sub_embs = all_sub_embs + self.pos_encoding[:max_subs].unsqueeze(0)

        encoded = self.transformer(all_sub_embs, src_key_padding_mask=~sub_mask)
        return encoded


class CrossAttentionScorerSubsXAttn(nn.Module):
    """Like `CrossAttentionScorer`, but the KV set is the concatenation of
    per-term expression embeddings and per-sub embeddings (each tagged with a
    learned type embedding so the model can distinguish them).
    """

    def __init__(self, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.state_proj = nn.Linear(embed_dim, embed_dim)
        self.action_proj = nn.Linear(embed_dim, embed_dim)
        self.expr_term_proj = nn.Linear(embed_dim, embed_dim)
        self.sub_term_proj = nn.Linear(embed_dim, embed_dim)
        # Type embedding: 0 = expression token, 1 = sub token.
        self.token_type_emb = nn.Embedding(2, embed_dim)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.cross_attn_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(0.1),
            ) for _ in range(n_layers)
        ])
        self.cross_attn_ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, state_emb, action_emb,
                expr_terms, expr_mask,
                sub_terms, sub_mask,
                action_mask):
        batch_size, n_actions, _ = action_emb.shape

        state_proj = self.state_proj(state_emb)
        action_proj = self.action_proj(action_emb)
        expr_proj = self.expr_term_proj(expr_terms) + self.token_type_emb.weight[0]
        sub_proj = self.sub_term_proj(sub_terms) + self.token_type_emb.weight[1]

        # KV is [expr terms ; sub terms]. Padding mask is OR-ed; an action
        # cross-attends to all valid expr + valid sub tokens in one shot.
        kv = torch.cat([expr_proj, sub_proj], dim=1)
        kv_valid = torch.cat([expr_mask, sub_mask], dim=1)
        key_pad_mask = ~kv_valid

        attended = action_proj
        for i in range(len(self.cross_attn_layers)):
            attn_out, _ = self.cross_attn_layers[i](
                attended, kv, kv, key_padding_mask=key_pad_mask,
            )
            attended = self.cross_attn_norms[i](attended + attn_out)
            attended = self.cross_attn_ffn_norms[i](
                attended + self.cross_attn_ffn[i](attended)
            )

        state_expanded = state_proj.unsqueeze(1).expand(-1, n_actions, -1)
        logits = self.scorer(torch.cat([state_expanded, attended], dim=-1)).squeeze(-1)
        return logits.masked_fill(~action_mask, float('-inf'))


class IBPActionClassifierSubsXAttn(nn.Module):
    """SAILIR classifier where subs enter only through cross-attention.

    Differences from `IBPActionClassifier`:
      - The subs encoder returns per-sub embeddings (no final pool).
      - `state_combine` takes 3 channels (cls + target + sector), NOT 4
        — the pooled-subs channel is removed entirely.
      - The cross-attention scorer attends over expr+sub tokens concatenated.
    """

    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 n_subs_layers=2, prime=2147483647, n_indices=7, n_denominators=6,
                 n_ibp_ops=9, **kwargs):
        super().__init__()
        self.prime = prime
        self.embed_dim = embed_dim
        self.n_indices = n_indices
        self.n_denominators = n_denominators

        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers,
            n_indices=n_indices, **kwargs,
        )
        self.subs_enc = SubsEncoderPerSub(
            embed_dim, n_heads=n_heads, n_layers=n_subs_layers, prime=prime,
            n_indices=n_indices, **kwargs,
        )
        self.sector_enc = SectorEncoder(embed_dim, n_denominators=n_denominators)
        self.action_enc = ActionEncoder(embed_dim, n_indices=n_indices,
                                         n_ibp_ops=n_ibp_ops, **kwargs)

        # state_combine: cls + target + sector -> embed_dim (no subs channel).
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.scorer = CrossAttentionScorerSubsXAttn(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True,
        )
        sub_terms = self.subs_enc(
            sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
        )
        sector_emb = self.sector_enc(sector_mask)

        # NO pooled subs channel in state — all sub info flows through cross-attn.
        state_emb = self.state_combine(
            torch.cat([cls_pooled, target_pooled, sector_emb], dim=-1)
        )
        action_emb = self.action_enc(action_ibp_ops, action_deltas)
        logits = self.scorer(
            state_emb, action_emb,
            expr_terms, expr_mask,
            sub_terms, sub_mask,
            action_mask,
        )
        return logits, F.softmax(logits, dim=-1)

    def predict(self, *args, **kwargs):
        logits, _ = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)
