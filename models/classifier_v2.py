"""
IBP Action Classifier v2

Improvements over v1:
1. Better coefficient encoding - encode actual magnitude, not just log-normalized
2. Cross-attention between state and actions - actions attend to expression terms
3. Weight-aware features - explicit encoding of integral weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IntegralEncoder(nn.Module):
    """
    Encode a 7-tuple integral index into a vector.
    Also computes and encodes the weight of the integral.
    """
    def __init__(self, embed_dim=64, max_index=20, min_index=-10):
        super().__init__()
        self.embed_dim = embed_dim
        self.min_index = min_index
        self.num_values = max_index - min_index + 1

        # Separate embedding for each of 7 positions
        self.position_embeds = nn.ModuleList([
            nn.Embedding(self.num_values, embed_dim // 2) for _ in range(7)
        ])

        # Weight features encoder (sum_pos, sum_neg)
        self.weight_enc = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )

        # Combine the 7 position embeddings + weight
        self.combine = nn.Sequential(
            nn.Linear(7 * (embed_dim // 2) + embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, integral):
        """
        Args:
            integral: (batch, 7) or (batch, seq_len, 7) tensor of indices
        Returns:
            (batch, embed_dim) or (batch, seq_len, embed_dim)
        """
        # Shift indices to be non-negative
        shifted = integral - self.min_index
        shifted = shifted.clamp(0, self.num_values - 1)

        orig_shape = shifted.shape[:-1]
        shifted_flat = shifted.reshape(-1, 7)
        integral_flat = integral.reshape(-1, 7).float()

        # Embed each position
        embeds = []
        for i in range(7):
            embeds.append(self.position_embeds[i](shifted_flat[:, i]))

        # Compute weight features
        sum_pos = integral_flat.clamp(min=0).sum(dim=-1, keepdim=True)  # (N, 1)
        sum_neg = (-integral_flat).clamp(min=0).sum(dim=-1, keepdim=True)  # (N, 1)
        # Normalize weights
        weight_features = torch.cat([sum_pos / 10.0, sum_neg / 10.0], dim=-1)  # (N, 2)
        weight_emb = self.weight_enc(weight_features)  # (N, embed_dim//2)

        # Concatenate and combine
        combined = torch.cat(embeds + [weight_emb], dim=-1)
        output = self.combine(combined)

        return output.reshape(*orig_shape, self.embed_dim)


class CoefficientEncoder(nn.Module):
    """
    Improved coefficient encoding with multiple representations:
    1. Log-magnitude + sign (for large values)
    2. Direct value embedding (for small common values)
    3. Modular features (for detecting patterns like (d-n))
    """
    def __init__(self, embed_dim=64, prime=2147483647):
        super().__init__()
        self.prime = prime
        self.half_prime = prime // 2

        # For small coefficients (direct embedding)
        # Common small values: -10 to 10, and their negatives mod prime
        self.small_embed = nn.Embedding(64, embed_dim // 2)  # -31 to 31

        # For large coefficients (learned projection from features)
        self.large_embed = nn.Sequential(
            nn.Linear(4, embed_dim // 2),  # (log_mag, sign, mod_small, is_small)
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )

        # Combine small and large representations
        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, coeff):
        """
        Args:
            coeff: (batch,) or (batch, seq_len) tensor of coefficients
        Returns:
            (batch, embed_dim) or (batch, seq_len, embed_dim)
        """
        orig_shape = coeff.shape
        coeff_flat = coeff.reshape(-1).float()

        # Convert to signed representation
        signed = torch.where(
            coeff_flat > self.half_prime,
            coeff_flat - self.prime,
            coeff_flat
        )

        # Small coefficient embedding (clamp to embedding range)
        small_idx = (signed.long() + 32).clamp(0, 63)
        small_emb = self.small_embed(small_idx)

        # Large coefficient features
        abs_val = torch.abs(signed)
        log_mag = torch.log1p(abs_val) / 20.0  # Normalize
        sign = (signed >= 0).float()
        mod_small = (abs_val % 100).float() / 100.0  # Modular pattern
        is_small = (abs_val < 100).float()

        large_features = torch.stack([log_mag, sign, mod_small, is_small], dim=-1)
        large_emb = self.large_embed(large_features)

        # Combine
        combined = torch.cat([small_emb, large_emb], dim=-1)
        output = self.combine(combined)

        return output.reshape(*orig_shape, -1)


class ExpressionEncoder(nn.Module):
    """
    Encode expression with per-term embeddings for cross-attention.
    Returns both pooled embedding AND per-term embeddings.
    """
    def __init__(self, embed_dim=128, max_index=20, min_index=-10, prime=2147483647):
        super().__init__()
        self.integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)
        self.coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)

        # Per-element transform
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Post-pooling transform
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, integrals, coeffs, mask=None, return_per_term=False):
        """
        Args:
            integrals: (batch, max_terms, 7) integral indices
            coeffs: (batch, max_terms) coefficients
            mask: (batch, max_terms) bool mask, True for valid terms
            return_per_term: if True, also return per-term embeddings
        Returns:
            pooled: (batch, embed_dim) expression embedding
            per_term: (batch, max_terms, embed_dim) if return_per_term
        """
        # Encode integrals and coefficients
        int_emb = self.integral_enc(integrals)
        coeff_emb = self.coeff_enc(coeffs)

        # Combine
        combined = torch.cat([int_emb, coeff_emb], dim=-1)

        # Apply phi
        per_term = self.phi(combined)

        # Mask out padding
        if mask is not None:
            per_term_masked = per_term * mask.unsqueeze(-1).float()
        else:
            per_term_masked = per_term

        # Sum pooling
        pooled = per_term_masked.sum(dim=1)

        # Apply rho
        pooled = self.rho(pooled)

        if return_per_term:
            return pooled, per_term
        return pooled


class SubstitutionEncoder(nn.Module):
    """
    Encode substitution history using Transformer.
    """
    def __init__(self, embed_dim=128, max_index=20, min_index=-10,
                 n_heads=4, n_layers=2, max_subs=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.integral_enc = IntegralEncoder(embed_dim, max_index, min_index)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_subs, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, sub_integrals, mask=None):
        """
        Args:
            sub_integrals: (batch, max_subs, 7)
            mask: (batch, max_subs) bool mask
        Returns:
            (batch, embed_dim)
        """
        batch_size, max_subs, _ = sub_integrals.shape

        # Handle empty substitution case
        if mask is not None:
            has_subs = mask.any(dim=1)
            if not has_subs.all():
                result = torch.zeros(batch_size, self.embed_dim, device=sub_integrals.device)
                if has_subs.any():
                    valid_idx = has_subs.nonzero(as_tuple=True)[0]
                    valid_result = self.forward(
                        sub_integrals[valid_idx],
                        mask[valid_idx] if mask is not None else None
                    )
                    result[valid_idx] = valid_result
                return result

        # Encode integrals
        emb = self.integral_enc(sub_integrals)

        # Add positional encoding
        emb = emb + self.pos_encoding[:max_subs].unsqueeze(0)

        # Create attention mask
        attn_mask = ~mask if mask is not None else None

        # Apply transformer
        encoded = self.transformer(emb, src_key_padding_mask=attn_mask)

        # Pool using learned query
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded, key_padding_mask=attn_mask)

        return pooled.squeeze(1)


class ActionEncoder(nn.Module):
    """
    Encode action (ibp_op, delta) with weight-aware features.
    """
    def __init__(self, embed_dim=128, n_ibp_ops=9, max_index=20, min_index=-10):
        super().__init__()

        # Embed ibp_op
        self.ibp_embed = nn.Embedding(n_ibp_ops, embed_dim // 2)

        # Embed delta (use weight-aware IntegralEncoder)
        self.delta_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, ibp_op, delta):
        """
        Args:
            ibp_op: (batch,) or (batch, n_actions)
            delta: (batch, 7) or (batch, n_actions, 7)
        Returns:
            (batch, embed_dim) or (batch, n_actions, embed_dim)
        """
        ibp_emb = self.ibp_embed(ibp_op)
        delta_emb = self.delta_enc(delta)

        combined = torch.cat([ibp_emb, delta_emb], dim=-1)
        return self.combine(combined)


class CrossAttentionScorer(nn.Module):
    """
    Score actions by attending to expression terms.

    Each action queries the expression terms to compute relevance-weighted scores.
    """
    def __init__(self, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.embed_dim = embed_dim

        # Project state components to common dimension
        self.state_proj = nn.Linear(embed_dim, embed_dim)
        self.action_proj = nn.Linear(embed_dim, embed_dim)
        self.expr_term_proj = nn.Linear(embed_dim, embed_dim)

        # Cross-attention: actions attend to expression terms
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        self.cross_attn_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(0.1)
            ) for _ in range(n_layers)
        ])
        self.cross_attn_ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])

        # Final scorer
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, state_emb, action_emb, expr_terms, expr_mask, action_mask):
        """
        Args:
            state_emb: (batch, embed_dim) pooled state
            action_emb: (batch, n_actions, embed_dim) action embeddings
            expr_terms: (batch, n_terms, embed_dim) per-term expression embeddings
            expr_mask: (batch, n_terms) True for valid terms
            action_mask: (batch, n_actions) True for valid actions
        Returns:
            logits: (batch, n_actions)
        """
        batch_size, n_actions, _ = action_emb.shape

        # Project
        state_proj = self.state_proj(state_emb)  # (batch, embed_dim)
        action_proj = self.action_proj(action_emb)  # (batch, n_actions, embed_dim)
        expr_proj = self.expr_term_proj(expr_terms)  # (batch, n_terms, embed_dim)

        # Cross-attention: actions query expression terms
        # key_padding_mask needs True for positions to ignore
        key_pad_mask = ~expr_mask if expr_mask is not None else None

        attended = action_proj
        for i in range(len(self.cross_attn_layers)):
            # Cross-attention
            attn_out, _ = self.cross_attn_layers[i](
                attended, expr_proj, expr_proj,
                key_padding_mask=key_pad_mask
            )
            attended = self.cross_attn_norms[i](attended + attn_out)

            # FFN
            ffn_out = self.cross_attn_ffn[i](attended)
            attended = self.cross_attn_ffn_norms[i](attended + ffn_out)

        # Combine with global state
        state_expanded = state_proj.unsqueeze(1).expand(-1, n_actions, -1)
        combined = torch.cat([state_expanded, attended], dim=-1)

        # Score
        logits = self.scorer(combined).squeeze(-1)

        # Mask invalid actions
        logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits


class IBPActionClassifierV2(nn.Module):
    """
    Improved classifier with cross-attention and better coefficient encoding.
    """
    def __init__(self, embed_dim=256, n_heads=4, n_cross_layers=2, prime=2147483647, **kwargs):
        super().__init__()
        self.prime = prime

        # Encoders - only ExpressionEncoder needs prime
        self.expr_enc = ExpressionEncoder(embed_dim, prime=prime, **kwargs)
        self.subs_enc = SubstitutionEncoder(embed_dim, **kwargs)
        self.action_enc = ActionEncoder(embed_dim, **kwargs)

        # Combine expression and substitution history
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Cross-attention scorer
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_integrals, sub_mask,
                action_ibp_ops, action_deltas, action_mask):
        """
        Args:
            expr_integrals: (batch, max_terms, 7)
            expr_coeffs: (batch, max_terms)
            expr_mask: (batch, max_terms)
            sub_integrals: (batch, max_subs, 7)
            sub_mask: (batch, max_subs)
            action_ibp_ops: (batch, max_actions)
            action_deltas: (batch, max_actions, 7)
            action_mask: (batch, max_actions)

        Returns:
            logits: (batch, max_actions)
            probs: (batch, max_actions)
        """
        # Encode expression (get both pooled and per-term)
        expr_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, expr_mask, return_per_term=True
        )

        # Encode substitution history
        subs_emb = self.subs_enc(sub_integrals, sub_mask)

        # Combine into state embedding
        state_combined = torch.cat([expr_pooled, subs_emb], dim=-1)
        state_emb = self.state_combine(state_combined)

        # Encode actions
        action_emb = self.action_enc(action_ibp_ops, action_deltas)

        # Score using cross-attention
        logits = self.scorer(state_emb, action_emb, expr_terms, expr_mask, action_mask)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict(self, expr_integrals, expr_coeffs, expr_mask,
                sub_integrals, sub_mask,
                action_ibp_ops, action_deltas, action_mask):
        """Predict the best action index."""
        logits, _ = self.forward(
            expr_integrals, expr_coeffs, expr_mask,
            sub_integrals, sub_mask,
            action_ibp_ops, action_deltas, action_mask
        )
        return logits.argmax(dim=-1)


# Reuse collate_samples from v1
from classifier_v1 import collate_samples


if __name__ == '__main__':
    import json

    print("Loading classifier training data...")
    with open('/home/shih/work/IBPreduction/data/classifier_training_data.jsonl') as f:
        samples = [json.loads(line) for line in f][:8]

    print(f"Loaded {len(samples)} samples")

    batch = collate_samples(samples)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # Create model
    model = IBPActionClassifierV2(embed_dim=256, n_heads=4, n_cross_layers=2)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    logits, probs = model(
        batch['expr_integrals'],
        batch['expr_coeffs'],
        batch['expr_mask'],
        batch['sub_integrals'],
        batch['sub_mask'],
        batch['action_ibp_ops'],
        batch['action_deltas'],
        batch['action_mask']
    )

    print(f"\nLogits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")

    # Loss
    loss = F.cross_entropy(logits, batch['labels'])
    print(f"\nCross-entropy loss: {loss.item():.4f}")
