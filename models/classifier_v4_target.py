"""
IBP Action Classifier v4 with Target Conditioning

Same as v3 but also takes the TARGET INTEGRAL as explicit input.
This allows the model to learn: given expression, subs, sector, AND target,
what action should be taken to reduce that specific target.

Key improvement: Target is included as a token in the transformer sequence,
allowing it to attend to expression terms and vice versa.

This supports training data where targets are randomly selected (not always highest weight).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse components from v3 and v2
from classifier_v3 import CrossAttentionScorer
from classifier_v2 import IntegralEncoder, CoefficientEncoder, ActionEncoder, SubstitutionEncoder


class TransformerExpressionEncoderWithTarget(nn.Module):
    """
    Encode expression AND target using Transformer self-attention.

    The target integral is added as a special [TARGET] token that can attend
    to all expression terms, allowing the model to learn target-expression relationships.

    Sequence: [CLS] [TARGET] term1 term2 ... termN
    """
    def __init__(self, embed_dim=256, max_index=20, min_index=-10, prime=2147483647,
                 n_heads=4, n_layers=2, max_terms=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Per-term encoders (for expression terms)
        self.integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)
        self.coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)

        # Target integral encoder (no coefficient, just the integral indices)
        self.target_integral_enc = IntegralEncoder(embed_dim, max_index, min_index)

        # Project combined (integral, coeff) to embed_dim for expression terms
        self.term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Project target embedding
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Learnable [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projections
        self.cls_output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.target_output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, integrals, coeffs, target_integral, mask=None, return_per_term=False):
        """
        Args:
            integrals: (batch, max_terms, 7) integral indices
            coeffs: (batch, max_terms) coefficients
            target_integral: (batch, 7) target integral indices
            mask: (batch, max_terms) bool mask, True for valid terms
        Returns:
            cls_pooled: (batch, embed_dim) expression embedding (from [CLS] token)
            target_pooled: (batch, embed_dim) target embedding (from [TARGET] token)
            per_term: (batch, max_terms, embed_dim) if return_per_term
        """
        batch_size, max_terms, _ = integrals.shape

        # Encode expression terms (integrals and coefficients)
        int_emb = self.integral_enc(integrals)  # (batch, max_terms, embed_dim//2)
        coeff_emb = self.coeff_enc(coeffs)      # (batch, max_terms, embed_dim//2)
        combined = torch.cat([int_emb, coeff_emb], dim=-1)
        term_emb = self.term_proj(combined)  # (batch, max_terms, embed_dim)

        # Encode target integral
        target_emb = self.target_integral_enc(target_integral)  # (batch, embed_dim)
        target_emb = self.target_proj(target_emb)  # (batch, embed_dim)
        target_emb = target_emb.unsqueeze(1)  # (batch, 1, embed_dim)

        # Prepend [CLS] and [TARGET] tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        # Sequence: [CLS] [TARGET] term1 term2 ... termN
        seq = torch.cat([cls_tokens, target_emb, term_emb], dim=1)  # (batch, 2 + max_terms, embed_dim)

        # Create attention mask: [CLS] and [TARGET] are always valid, then use term mask
        if mask is not None:
            special_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=mask.device)  # [CLS] and [TARGET]
            full_mask = torch.cat([special_mask, mask], dim=1)
            # TransformerEncoder needs True for positions to IGNORE
            attn_mask = ~full_mask
        else:
            attn_mask = None

        # Apply transformer
        encoded = self.transformer(seq, src_key_padding_mask=attn_mask)

        # Extract [CLS] token (position 0) and [TARGET] token (position 1)
        cls_pooled = self.cls_output_proj(encoded[:, 0])  # (batch, embed_dim)
        target_pooled = self.target_output_proj(encoded[:, 1])  # (batch, embed_dim)

        # Per-term embeddings (excluding [CLS] and [TARGET])
        per_term = encoded[:, 2:]  # (batch, max_terms, embed_dim)

        if return_per_term:
            return cls_pooled, target_pooled, per_term
        return cls_pooled, target_pooled


class SectorEncoder(nn.Module):
    """
    Encode sector as a 6-bit mask into an embedding.

    The sector mask is [b0, b1, b2, b3, b4, b5] where each bi is 0 or 1,
    indicating whether propagator i is present in the sector.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable embeddings for each position being 0 or 1
        # 6 positions, 2 values each
        self.position_embeddings = nn.ModuleList([
            nn.Embedding(2, embed_dim // 6 + 1)  # +1 to handle rounding
            for _ in range(6)
        ])

        # Project concatenated position embeddings to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(6 * (embed_dim // 6 + 1), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, sector_mask):
        """
        Args:
            sector_mask: (batch, 6) tensor of 0/1 values
        Returns:
            sector_emb: (batch, embed_dim)
        """
        batch_size = sector_mask.size(0)

        # Embed each position
        pos_embs = []
        for i in range(6):
            pos_emb = self.position_embeddings[i](sector_mask[:, i].long())
            pos_embs.append(pos_emb)

        # Concatenate and project
        combined = torch.cat(pos_embs, dim=-1)
        sector_emb = self.proj(combined)

        return sector_emb


class IBPActionClassifierV4Target(nn.Module):
    """
    Classifier with transformer expression encoder, sector conditioning, and target input.

    Key improvement: Target is included as a [TARGET] token in the transformer sequence,
    allowing bidirectional attention between target and expression terms.
    """
    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 prime=2147483647, **kwargs):
        super().__init__()
        self.prime = prime
        self.embed_dim = embed_dim

        # Transformer expression encoder WITH target attention
        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers, **kwargs
        )

        # Substitution history encoder
        self.subs_enc = SubstitutionEncoder(embed_dim, n_heads=n_heads, n_layers=2, **kwargs)

        # Sector encoder
        self.sector_enc = SectorEncoder(embed_dim)

        # Action encoder
        self.action_enc = ActionEncoder(embed_dim, **kwargs)

        # Combine expression, substitution history, sector, and target-attended embedding
        # Input: cls (embed_dim) + target_attended (embed_dim) + subs (embed_dim) + sector (embed_dim) = 4 * embed_dim
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Cross-attention scorer
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_integrals, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
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
            sector_mask: (batch, 6) - sector conditioning
            target_integral: (batch, 7) - which integral we're reducing

        Returns:
            logits: (batch, max_actions)
            probs: (batch, max_actions)
        """
        # Encode expression with transformer, including target as a token
        # Returns: cls_pooled (global expr), target_pooled (target attended to expr), per-term embeddings
        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True
        )

        # Encode substitution history
        subs_emb = self.subs_enc(sub_integrals, sub_mask)

        # Encode sector
        sector_emb = self.sector_enc(sector_mask)

        # Combine into state embedding (cls + target_attended + subs + sector)
        # target_pooled now has context from attending to expression terms
        state_combined = torch.cat([cls_pooled, target_pooled, subs_emb, sector_emb], dim=-1)
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
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
        """Predict the best action index."""
        logits, _ = self.forward(
            expr_integrals, expr_coeffs, expr_mask,
            sub_integrals, sub_mask,
            action_ibp_ops, action_deltas, action_mask,
            sector_mask, target_integral
        )
        return logits.argmax(dim=-1)


def collate_samples_v4(samples, max_terms=200, max_subs=50, max_actions=900):
    """
    Collate a batch of classifier samples into tensors.
    Includes sector_mask and target_integral.

    Args:
        samples: list of dicts from training data jsonl

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

    sector_masks = torch.zeros(batch_size, 6, dtype=torch.long)
    target_integrals = torch.zeros(batch_size, 7, dtype=torch.long)  # NEW: target integral

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

        # Sector mask
        sector_masks[i] = torch.tensor(sample['sector_mask'])

        # Target integral (NEW)
        target_integrals[i] = torch.tensor(sample['target'])

        # Label (index of chosen action)
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
        'sector_mask': sector_masks,
        'target_integral': target_integrals,  # NEW
        'labels': labels
    }


if __name__ == '__main__':
    import json

    print("Testing IBP Action Classifier v3 with Sector Conditioning...")
    print("=" * 70)

    # Load a few samples from multisector data
    print("\nLoading multisector training data...")
    with open('/home/shih/work/IBPreduction/data/multisector_training_data.jsonl') as f:
        samples = [json.loads(line) for line in f][:8]

    print(f"Loaded {len(samples)} samples")
    print(f"Sample sector_masks: {[s['sector_mask'] for s in samples]}")

    batch = collate_samples_sector(samples)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # Create model
    model = IBPActionClassifierV3Sector(embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2, prime=1009)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Forward pass
    print("\nRunning forward pass...")
    logits, probs = model(
        batch['expr_integrals'],
        batch['expr_coeffs'],
        batch['expr_mask'],
        batch['sub_integrals'],
        batch['sub_mask'],
        batch['action_ibp_ops'],
        batch['action_deltas'],
        batch['action_mask'],
        batch['sector_mask']
    )

    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")

    # Loss
    loss = F.cross_entropy(logits, batch['labels'])
    print(f"\nCross-entropy loss: {loss.item():.4f}")

    # Check predictions
    preds = logits.argmax(dim=-1)
    acc = (preds == batch['labels']).float().mean()
    print(f"Batch accuracy: {acc.item():.4f}")

    print("\nv3_sector model test complete!")
