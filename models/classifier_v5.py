"""
IBP Action Classifier v5

Improvements over v3:
1. Target as input - target integral is a [TARGET] token in transformer (from v4)
2. Full substitution encoding - encodes the COMPLETE substitution structure:
   - The integral being substituted
   - ALL replacement terms with their coefficients
3. Cross-attention scoring (from v3) - actions attend to expression terms

The key insight is that the substitution history contains crucial information:
not just WHAT was substituted, but WHAT it was replaced with.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse components from v2
from classifier_v2 import IntegralEncoder, CoefficientEncoder, ActionEncoder


class FullSubstitutionEncoder(nn.Module):
    """
    Encode the COMPLETE substitution structure.

    Each substitution is: [integral_being_substituted, [[integral1, coeff1], [integral2, coeff2], ...]]

    We encode:
    1. The integral being substituted (the "key")
    2. Each replacement term (integral + coefficient)
    3. Pool the replacement terms
    4. Combine key and replacement into a single embedding per substitution
    5. Use Transformer to encode the sequence of substitutions
    """
    def __init__(self, embed_dim=256, max_index=20, min_index=-10, prime=2147483647,
                 n_heads=4, n_layers=2, max_subs=50, max_replacement_terms=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_replacement_terms = max_replacement_terms

        # Encoder for the integral being substituted (the "key")
        self.key_integral_enc = IntegralEncoder(embed_dim, max_index, min_index)

        # Encoder for replacement terms (integral + coefficient)
        self.replacement_integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)
        self.replacement_coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)

        # Project replacement term to embed_dim
        self.replacement_term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Attention pooling for replacement terms
        self.replacement_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.replacement_pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # Combine key and replacement into single substitution embedding
        self.sub_combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Positional encoding for substitution sequence (order matters)
        self.pos_encoding = nn.Parameter(torch.randn(max_subs, embed_dim) * 0.02)

        # Transformer to encode sequence of substitutions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final pooling
        self.final_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.final_pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def encode_single_substitution(self, key_integral, replacement_integrals, replacement_coeffs, replacement_mask):
        """
        Encode a single substitution.

        Args:
            key_integral: (batch, 7) the integral being substituted
            replacement_integrals: (batch, max_terms, 7) replacement terms
            replacement_coeffs: (batch, max_terms) replacement coefficients
            replacement_mask: (batch, max_terms) True for valid terms
        Returns:
            sub_emb: (batch, embed_dim)
        """
        batch_size = key_integral.size(0)

        # Encode the key integral
        key_emb = self.key_integral_enc(key_integral)  # (batch, embed_dim)

        # Encode replacement terms
        repl_int_emb = self.replacement_integral_enc(replacement_integrals)
        repl_coeff_emb = self.replacement_coeff_enc(replacement_coeffs)
        repl_combined = torch.cat([repl_int_emb, repl_coeff_emb], dim=-1)
        repl_term_emb = self.replacement_term_proj(repl_combined)

        # Pool replacement terms using attention
        query = self.replacement_pool_query.expand(batch_size, -1, -1)
        key_pad_mask = ~replacement_mask if replacement_mask is not None else None

        # Handle case where all terms are masked
        if replacement_mask is not None:
            has_terms = replacement_mask.any(dim=1)
            if not has_terms.all():
                repl_pooled = torch.zeros(batch_size, self.embed_dim, device=key_integral.device)
                if has_terms.any():
                    valid_idx = has_terms.nonzero(as_tuple=True)[0]
                    valid_pooled, _ = self.replacement_pool_attn(
                        query[valid_idx], repl_term_emb[valid_idx], repl_term_emb[valid_idx],
                        key_padding_mask=key_pad_mask[valid_idx]
                    )
                    repl_pooled[valid_idx] = valid_pooled.squeeze(1)
            else:
                repl_pooled, _ = self.replacement_pool_attn(query, repl_term_emb, repl_term_emb, key_padding_mask=key_pad_mask)
                repl_pooled = repl_pooled.squeeze(1)
        else:
            repl_pooled, _ = self.replacement_pool_attn(query, repl_term_emb, repl_term_emb)
            repl_pooled = repl_pooled.squeeze(1)

        # Combine key and replacement
        combined = torch.cat([key_emb, repl_pooled], dim=-1)
        sub_emb = self.sub_combine(combined)

        return sub_emb

    def forward(self, sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask):
        """
        Encode the full substitution history (batched version).

        Args:
            sub_keys: (batch, max_subs, 7) key integrals being substituted
            sub_repl_ints: (batch, max_subs, max_repl, 7) replacement integrals
            sub_repl_coeffs: (batch, max_subs, max_repl) replacement coefficients
            sub_repl_mask: (batch, max_subs, max_repl) mask for valid replacement terms
            sub_mask: (batch, max_subs) mask for valid substitutions
        Returns:
            (batch, embed_dim) pooled substitution history embedding
        """
        batch_size, max_subs, max_repl, _ = sub_repl_ints.shape
        device = sub_mask.device

        # Handle case where no samples have substitutions
        if not sub_mask.any():
            return torch.zeros(batch_size, self.embed_dim, device=device)

        # Flatten to (batch * max_subs, ...) for batched encoding
        flat_keys = sub_keys.view(batch_size * max_subs, 7)
        flat_repl_ints = sub_repl_ints.view(batch_size * max_subs, max_repl, 7)
        flat_repl_coeffs = sub_repl_coeffs.view(batch_size * max_subs, max_repl)
        flat_repl_mask = sub_repl_mask.view(batch_size * max_subs, max_repl)

        # Encode all substitutions in one batched call
        flat_sub_embs = self.encode_single_substitution(flat_keys, flat_repl_ints, flat_repl_coeffs, flat_repl_mask)

        # Reshape back to (batch, max_subs, embed_dim)
        all_sub_embs = flat_sub_embs.view(batch_size, max_subs, self.embed_dim)

        # Zero out invalid substitutions
        all_sub_embs = all_sub_embs * sub_mask.unsqueeze(-1).float()

        # Add positional encoding
        all_sub_embs = all_sub_embs + self.pos_encoding[:max_subs].unsqueeze(0)

        # Apply transformer
        attn_mask = ~sub_mask
        encoded = self.transformer(all_sub_embs, src_key_padding_mask=attn_mask)

        # Final pooling
        query = self.final_pool_query.expand(batch_size, -1, -1)

        result = torch.zeros(batch_size, self.embed_dim, device=device)
        has_subs = sub_mask.any(dim=1)

        if has_subs.any():
            valid_idx = has_subs.nonzero(as_tuple=True)[0]
            valid_pooled, _ = self.final_pool_attn(
                query[valid_idx], encoded[valid_idx], encoded[valid_idx],
                key_padding_mask=attn_mask[valid_idx]
            )
            result[valid_idx] = valid_pooled.squeeze(1)

        return result


class SectorEncoder(nn.Module):
    """Encode sector as a 6-bit mask into an embedding."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.position_embeddings = nn.ModuleList([
            nn.Embedding(2, embed_dim // 6 + 1) for _ in range(6)
        ])
        self.proj = nn.Sequential(
            nn.Linear(6 * (embed_dim // 6 + 1), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, sector_mask):
        pos_embs = [self.position_embeddings[i](sector_mask[:, i].long()) for i in range(6)]
        return self.proj(torch.cat(pos_embs, dim=-1))


class TransformerExpressionEncoderWithTarget(nn.Module):
    """
    Encode expression AND target using Transformer self-attention.
    Sequence: [CLS] [TARGET] term1 term2 ... termN
    """
    def __init__(self, embed_dim=256, max_index=20, min_index=-10, prime=2147483647,
                 n_heads=4, n_layers=2, max_terms=512):
        super().__init__()
        self.embed_dim = embed_dim

        self.integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)
        self.coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)
        self.target_integral_enc = IntegralEncoder(embed_dim, max_index, min_index)

        self.term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.target_output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, integrals, coeffs, target_integral, mask=None, return_per_term=False):
        batch_size, max_terms, _ = integrals.shape

        int_emb = self.integral_enc(integrals)
        coeff_emb = self.coeff_enc(coeffs)
        term_emb = self.term_proj(torch.cat([int_emb, coeff_emb], dim=-1))

        target_emb = self.target_proj(self.target_integral_enc(target_integral)).unsqueeze(1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, target_emb, term_emb], dim=1)

        if mask is not None:
            special_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=mask.device)
            attn_mask = ~torch.cat([special_mask, mask], dim=1)
        else:
            attn_mask = None

        encoded = self.transformer(seq, src_key_padding_mask=attn_mask)

        cls_pooled = self.cls_output_proj(encoded[:, 0])
        target_pooled = self.target_output_proj(encoded[:, 1])
        per_term = encoded[:, 2:]

        if return_per_term:
            return cls_pooled, target_pooled, per_term
        return cls_pooled, target_pooled


class CrossAttentionScorer(nn.Module):
    """Score actions by attending to expression terms."""
    def __init__(self, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.state_proj = nn.Linear(embed_dim, embed_dim)
        self.action_proj = nn.Linear(embed_dim, embed_dim)
        self.expr_term_proj = nn.Linear(embed_dim, embed_dim)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.cross_attn_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(0.1)
            ) for _ in range(n_layers)
        ])
        self.cross_attn_ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, state_emb, action_emb, expr_terms, expr_mask, action_mask):
        batch_size, n_actions, _ = action_emb.shape

        state_proj = self.state_proj(state_emb)
        action_proj = self.action_proj(action_emb)
        expr_proj = self.expr_term_proj(expr_terms)

        key_pad_mask = ~expr_mask if expr_mask is not None else None

        attended = action_proj
        for i in range(len(self.cross_attn_layers)):
            attn_out, _ = self.cross_attn_layers[i](attended, expr_proj, expr_proj, key_padding_mask=key_pad_mask)
            attended = self.cross_attn_norms[i](attended + attn_out)
            attended = self.cross_attn_ffn_norms[i](attended + self.cross_attn_ffn[i](attended))

        state_expanded = state_proj.unsqueeze(1).expand(-1, n_actions, -1)
        logits = self.scorer(torch.cat([state_expanded, attended], dim=-1)).squeeze(-1)
        return logits.masked_fill(~action_mask, float('-inf'))


class IBPActionClassifierV5(nn.Module):
    """
    Classifier v5 with:
    - Full substitution encoding (key + replacement terms)
    - Target as input
    - Cross-attention scoring
    - Sector conditioning
    """
    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 n_subs_layers=2, prime=2147483647, **kwargs):
        super().__init__()
        self.prime = prime
        self.embed_dim = embed_dim

        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers, **kwargs
        )
        self.subs_enc = FullSubstitutionEncoder(
            embed_dim, n_heads=n_heads, n_layers=n_subs_layers, prime=prime, **kwargs
        )
        self.sector_enc = SectorEncoder(embed_dim)
        self.action_enc = ActionEncoder(embed_dim, **kwargs)

        # Combine: cls + target_attended + subs + sector = 4 * embed_dim
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True
        )
        subs_emb = self.subs_enc(sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask)
        sector_emb = self.sector_enc(sector_mask)

        state_combined = torch.cat([cls_pooled, target_pooled, subs_emb, sector_emb], dim=-1)
        state_emb = self.state_combine(state_combined)

        action_emb = self.action_enc(action_ibp_ops, action_deltas)
        logits = self.scorer(state_emb, action_emb, expr_terms, expr_mask, action_mask)

        return logits, F.softmax(logits, dim=-1)

    def predict(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
        logits, _ = self.forward(
            expr_integrals, expr_coeffs, expr_mask,
            sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
            action_ibp_ops, action_deltas, action_mask,
            sector_mask, target_integral
        )
        return logits.argmax(dim=-1)


if __name__ == '__main__':
    print("Testing IBP Action Classifier v5 (batched subs)...")

    # Quick test with dummy data
    b, max_subs, max_repl = 2, 3, 5
    model = IBPActionClassifierV5(embed_dim=128, n_heads=2, n_expr_layers=1, n_cross_layers=1, n_subs_layers=1, prime=1009)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    logits, _ = model(
        torch.zeros(b, 5, 7, dtype=torch.long),  # expr_integrals
        torch.zeros(b, 5, dtype=torch.long),      # expr_coeffs
        torch.ones(b, 5, dtype=torch.bool),       # expr_mask
        torch.zeros(b, max_subs, 7, dtype=torch.long),           # sub_keys
        torch.zeros(b, max_subs, max_repl, 7, dtype=torch.long), # sub_repl_ints
        torch.zeros(b, max_subs, max_repl, dtype=torch.long),    # sub_repl_coeffs
        torch.zeros(b, max_subs, max_repl, dtype=torch.bool),    # sub_repl_mask
        torch.zeros(b, max_subs, dtype=torch.bool),              # sub_mask (no subs)
        torch.zeros(b, 10, dtype=torch.long),     # action_ibp_ops
        torch.zeros(b, 10, 7, dtype=torch.long),  # action_deltas
        torch.ones(b, 10, dtype=torch.bool),      # action_mask
        torch.ones(b, 6, dtype=torch.long),       # sector_mask
        torch.zeros(b, 7, dtype=torch.long)       # target_integral
    )
    print(f"Output shape: {logits.shape}")
    print("OK")
