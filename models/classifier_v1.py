"""
IBP Action Classifier v1

Architecture:
- State encoder: combines expression (set) + substitution history (sequence)
- Action encoder: embeds (ibp_op, delta) pairs
- Scorer: computes compatibility between state and each action
- Output: softmax over valid actions

The model is permutation equivariant w.r.t. valid actions by design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IntegralEncoder(nn.Module):
    """
    Encode a 7-tuple integral index into a vector.

    Each position gets its own embedding table (since positions have different semantics),
    then we combine them.
    """
    def __init__(self, embed_dim=64, max_index=20, min_index=-10):
        super().__init__()
        self.embed_dim = embed_dim
        self.min_index = min_index
        self.num_values = max_index - min_index + 1

        # Separate embedding for each of 7 positions
        self.position_embeds = nn.ModuleList([
            nn.Embedding(self.num_values, embed_dim) for _ in range(7)
        ])

        # Combine the 7 position embeddings
        self.combine = nn.Sequential(
            nn.Linear(7 * embed_dim, embed_dim),
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

        # Embed each position
        embeds = []
        for i in range(7):
            embeds.append(self.position_embeds[i](shifted_flat[:, i]))

        # Concatenate and combine
        combined = torch.cat(embeds, dim=-1)
        output = self.combine(combined)

        return output.reshape(*orig_shape, self.embed_dim)


class CoefficientEncoder(nn.Module):
    """
    Encode coefficient (integer mod PRIME) into a vector.

    Since coefficients are in a huge space, we use a learned projection
    from a normalized representation.
    """
    def __init__(self, embed_dim=64, prime=2147483647):
        super().__init__()
        self.prime = prime
        self.embed = nn.Sequential(
            nn.Linear(2, embed_dim),  # (normalized_value, sign)
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

        # Normalize to [-1, 1] range and extract sign
        half_prime = self.prime / 2
        centered = coeff_flat - half_prime
        sign = (centered >= 0).float() * 2 - 1
        normalized = torch.log1p(torch.abs(centered)) / math.log(half_prime)

        features = torch.stack([normalized, sign], dim=-1)
        output = self.embed(features)

        return output.reshape(*orig_shape, -1)


class ExpressionEncoder(nn.Module):
    """
    Encode expression (set of integral-coefficient pairs) using DeepSets.

    Expression is permutation invariant, so we use sum pooling.
    """
    def __init__(self, embed_dim=128, max_index=20, min_index=-10):
        super().__init__()
        self.integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)
        self.coeff_enc = CoefficientEncoder(embed_dim // 2)

        # Per-element transform (phi in DeepSets)
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        # Post-pooling transform (rho in DeepSets)
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, integrals, coeffs, mask=None):
        """
        Args:
            integrals: (batch, max_terms, 7) integral indices
            coeffs: (batch, max_terms) coefficients
            mask: (batch, max_terms) bool mask, True for valid terms
        Returns:
            (batch, embed_dim) expression embedding
        """
        # Encode integrals and coefficients
        int_emb = self.integral_enc(integrals)  # (batch, max_terms, embed_dim/2)
        coeff_emb = self.coeff_enc(coeffs)       # (batch, max_terms, embed_dim/2)

        # Combine
        combined = torch.cat([int_emb, coeff_emb], dim=-1)  # (batch, max_terms, embed_dim)

        # Apply phi
        transformed = self.phi(combined)

        # Mask out padding
        if mask is not None:
            transformed = transformed * mask.unsqueeze(-1).float()

        # Sum pooling
        pooled = transformed.sum(dim=1)  # (batch, embed_dim)

        # Apply rho
        return self.rho(pooled)


class SubstitutionEncoder(nn.Module):
    """
    Encode substitution history (ordered sequence) using Transformer.

    Each substitution is (integral -> solution), where solution is itself
    a set of (integral, coeff) pairs. We simplify by encoding just the
    substituted integral (the key), since the solution can be derived.
    """
    def __init__(self, embed_dim=128, max_index=20, min_index=-10,
                 n_heads=4, n_layers=2, max_subs=50):
        super().__init__()
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
            sub_integrals: (batch, max_subs, 7) substituted integral indices (in order)
            mask: (batch, max_subs) bool mask, True for valid subs
        Returns:
            (batch, embed_dim) substitution history embedding
        """
        batch_size, max_subs, _ = sub_integrals.shape
        embed_dim = self.pos_encoding.shape[1]

        # Handle empty substitution case
        if mask is not None:
            # Check for samples with no substitutions
            has_subs = mask.any(dim=1)  # (batch,)
            if not has_subs.all():
                # Process samples with subs, return zeros for those without
                result = torch.zeros(batch_size, embed_dim, device=sub_integrals.device)
                if has_subs.any():
                    # Recursively process non-empty samples
                    valid_idx = has_subs.nonzero(as_tuple=True)[0]
                    valid_result = self.forward(
                        sub_integrals[valid_idx],
                        mask[valid_idx] if mask is not None else None
                    )
                    result[valid_idx] = valid_result
                return result

        # Encode integrals
        emb = self.integral_enc(sub_integrals)  # (batch, max_subs, embed_dim)

        # Add positional encoding
        emb = emb + self.pos_encoding[:max_subs].unsqueeze(0)

        # Create attention mask for transformer (True = ignore)
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        # Apply transformer
        encoded = self.transformer(emb, src_key_padding_mask=attn_mask)

        # Pool using learned query
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded, key_padding_mask=attn_mask)

        return pooled.squeeze(1)  # (batch, embed_dim)


class ActionEncoder(nn.Module):
    """
    Encode action (ibp_op, delta) into a vector.
    """
    def __init__(self, embed_dim=128, n_ibp_ops=9, max_index=20, min_index=-10):
        super().__init__()

        # Embed ibp_op (categorical)
        self.ibp_embed = nn.Embedding(n_ibp_ops, embed_dim // 2)

        # Embed delta (7-tuple, similar to integral)
        self.delta_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, ibp_op, delta):
        """
        Args:
            ibp_op: (batch,) or (batch, n_actions) ibp operator indices
            delta: (batch, 7) or (batch, n_actions, 7) delta values
        Returns:
            (batch, embed_dim) or (batch, n_actions, embed_dim)
        """
        ibp_emb = self.ibp_embed(ibp_op)
        delta_emb = self.delta_enc(delta)

        combined = torch.cat([ibp_emb, delta_emb], dim=-1)
        return self.combine(combined)


class StateEncoder(nn.Module):
    """
    Combine expression and substitution history into a single state embedding.
    """
    def __init__(self, embed_dim=256, **kwargs):
        super().__init__()
        self.expr_enc = ExpressionEncoder(embed_dim // 2, **kwargs)
        self.subs_enc = SubstitutionEncoder(embed_dim // 2, **kwargs)

        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_integrals, sub_mask):
        """
        Returns:
            (batch, embed_dim) state embedding
        """
        expr_emb = self.expr_enc(expr_integrals, expr_coeffs, expr_mask)
        subs_emb = self.subs_enc(sub_integrals, sub_mask)

        combined = torch.cat([expr_emb, subs_emb], dim=-1)
        return self.combine(combined)


class IBPActionClassifier(nn.Module):
    """
    Full classifier: predicts which action to take from valid action set.

    Uses score function approach for permutation equivariance.
    """
    def __init__(self, embed_dim=256, **kwargs):
        super().__init__()
        self.state_enc = StateEncoder(embed_dim, **kwargs)
        self.action_enc = ActionEncoder(embed_dim, **kwargs)

        # Scorer: maps (state_emb, action_emb) -> scalar
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_integrals, sub_mask,
                action_ibp_ops, action_deltas, action_mask):
        """
        Args:
            expr_integrals: (batch, max_terms, 7)
            expr_coeffs: (batch, max_terms)
            expr_mask: (batch, max_terms) True for valid terms
            sub_integrals: (batch, max_subs, 7)
            sub_mask: (batch, max_subs) True for valid subs
            action_ibp_ops: (batch, max_actions) ibp operator indices
            action_deltas: (batch, max_actions, 7) delta values
            action_mask: (batch, max_actions) True for valid actions

        Returns:
            logits: (batch, max_actions) scores for each action
            probs: (batch, max_actions) probabilities (softmax with masking)
        """
        batch_size, max_actions = action_ibp_ops.shape

        # Encode state (computed once)
        state_emb = self.state_enc(
            expr_integrals, expr_coeffs, expr_mask,
            sub_integrals, sub_mask
        )  # (batch, embed_dim)

        # Encode all actions
        action_emb = self.action_enc(action_ibp_ops, action_deltas)  # (batch, max_actions, embed_dim)

        # Expand state to match actions
        state_expanded = state_emb.unsqueeze(1).expand(-1, max_actions, -1)

        # Compute scores
        combined = torch.cat([state_expanded, action_emb], dim=-1)
        logits = self.scorer(combined).squeeze(-1)  # (batch, max_actions)

        # Mask invalid actions (set to -inf before softmax)
        logits = logits.masked_fill(~action_mask, float('-inf'))

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict(self, expr_integrals, expr_coeffs, expr_mask,
                sub_integrals, sub_mask,
                action_ibp_ops, action_deltas, action_mask):
        """
        Predict the best action index.

        Returns:
            (batch,) indices of predicted actions
        """
        logits, _ = self.forward(
            expr_integrals, expr_coeffs, expr_mask,
            sub_integrals, sub_mask,
            action_ibp_ops, action_deltas, action_mask
        )
        return logits.argmax(dim=-1)


# ============================================================================
# Dataset and collation utilities
# ============================================================================

def collate_samples(samples, max_terms=200, max_subs=50, max_actions=900):
    """
    Collate a batch of classifier samples into tensors.

    Args:
        samples: list of dicts from classifier_training_data.jsonl

    Returns:
        dict of batched tensors
    """
    batch_size = len(samples)

    # Initialize tensors
    expr_integrals = torch.zeros(batch_size, max_terms, 7, dtype=torch.long)
    expr_coeffs = torch.zeros(batch_size, max_terms, dtype=torch.long)
    expr_mask = torch.zeros(batch_size, max_terms, dtype=torch.bool)

    sub_integrals = torch.zeros(batch_size, max_subs, 7, dtype=torch.long)
    sub_mask = torch.zeros(batch_size, max_subs, dtype=torch.bool)

    action_ibp_ops = torch.zeros(batch_size, max_actions, dtype=torch.long)
    action_deltas = torch.zeros(batch_size, max_actions, 7, dtype=torch.long)
    action_mask = torch.zeros(batch_size, max_actions, dtype=torch.bool)

    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(samples):
        # Expression
        expr = sample['expr']
        n_terms = min(len(expr), max_terms)
        for j in range(n_terms):
            expr_integrals[i, j] = torch.tensor(expr[j][0])
            expr_coeffs[i, j] = expr[j][1]
            expr_mask[i, j] = True

        # Substitutions (just the keys, in order)
        subs = sample['subs']
        n_subs = min(len(subs), max_subs)
        for j in range(n_subs):
            sub_integrals[i, j] = torch.tensor(subs[j][0])
            sub_mask[i, j] = True

        # Valid actions
        actions = sample['valid_actions']
        n_actions = min(len(actions), max_actions)
        for j in range(n_actions):
            action_ibp_ops[i, j] = actions[j][0]
            action_deltas[i, j] = torch.tensor(actions[j][1])
            action_mask[i, j] = True

        # Label (index of chosen action)
        # Need to handle if chosen_action_idx >= max_actions
        labels[i] = min(sample['chosen_action_idx'], max_actions - 1)

    return {
        'expr_integrals': expr_integrals,
        'expr_coeffs': expr_coeffs,
        'expr_mask': expr_mask,
        'sub_integrals': sub_integrals,
        'sub_mask': sub_mask,
        'action_ibp_ops': action_ibp_ops,
        'action_deltas': action_deltas,
        'action_mask': action_mask,
        'labels': labels
    }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    import json

    # Load some samples
    print("Loading classifier training data...")
    with open('/home/shih/work/IBPreduction/data/classifier_training_data.jsonl') as f:
        samples = [json.loads(line) for line in f][:8]  # Small batch for testing

    print(f"Loaded {len(samples)} samples")

    # Collate into batch
    batch = collate_samples(samples)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # Create model
    model = IBPActionClassifier(embed_dim=128)
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

    # Predictions
    preds = model.predict(
        batch['expr_integrals'],
        batch['expr_coeffs'],
        batch['expr_mask'],
        batch['sub_integrals'],
        batch['sub_mask'],
        batch['action_ibp_ops'],
        batch['action_deltas'],
        batch['action_mask']
    )

    print(f"\nPredictions: {preds.tolist()}")
    print(f"Labels:      {batch['labels'].tolist()}")

    # Loss
    loss = F.cross_entropy(logits, batch['labels'])
    print(f"\nCross-entropy loss: {loss.item():.4f}")
