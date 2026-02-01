"""
IBP Action Classifier v3

Improvements over v2:
1. Transformer encoder for expressions - terms attend to each other
2. Better context for action selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Reuse encoders from v2
from classifier_v2 import IntegralEncoder, CoefficientEncoder, ActionEncoder, SubstitutionEncoder


class TransformerExpressionEncoder(nn.Module):
    """
    Encode expression using Transformer self-attention.
    Terms can attend to each other to learn relationships.
    """
    def __init__(self, embed_dim=256, max_index=20, min_index=-10, prime=2147483647,
                 n_heads=4, n_layers=2, max_terms=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Per-term encoders
        self.integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index)
        self.coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)

        # Project combined (integral, coeff) to embed_dim
        self.term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Learnable [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Learnable positional encoding (optional - can be disabled for pure permutation invariance)
        # Using no positional encoding makes the model permutation invariant
        # self.pos_encoding = nn.Parameter(torch.randn(max_terms + 1, embed_dim) * 0.02)

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

        # Output projection
        self.output_proj = nn.Sequential(
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
        Returns:
            pooled: (batch, embed_dim) expression embedding (from [CLS] token)
            per_term: (batch, max_terms, embed_dim) if return_per_term
        """
        batch_size, max_terms, _ = integrals.shape

        # Encode integrals and coefficients
        int_emb = self.integral_enc(integrals)  # (batch, max_terms, embed_dim//2)
        coeff_emb = self.coeff_enc(coeffs)      # (batch, max_terms, embed_dim//2)

        # Combine and project
        combined = torch.cat([int_emb, coeff_emb], dim=-1)
        term_emb = self.term_proj(combined)  # (batch, max_terms, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, term_emb], dim=1)  # (batch, 1 + max_terms, embed_dim)

        # Create attention mask: [CLS] is always valid, then use term mask
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            # TransformerEncoder needs True for positions to IGNORE
            attn_mask = ~full_mask
        else:
            attn_mask = None

        # Apply transformer
        encoded = self.transformer(seq, src_key_padding_mask=attn_mask)

        # Extract [CLS] token as pooled representation
        pooled = self.output_proj(encoded[:, 0])  # (batch, embed_dim)

        # Per-term embeddings (excluding [CLS])
        per_term = encoded[:, 1:]  # (batch, max_terms, embed_dim)

        if return_per_term:
            return pooled, per_term
        return pooled


class CrossAttentionScorer(nn.Module):
    """
    Score actions by attending to expression terms.
    Same as v2 but can use the transformer-encoded terms.
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
                nn.GELU(),
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
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
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
        state_proj = self.state_proj(state_emb)
        action_proj = self.action_proj(action_emb)
        expr_proj = self.expr_term_proj(expr_terms)

        # Cross-attention: actions query expression terms
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


class IBPActionClassifierV3(nn.Module):
    """
    Classifier with transformer expression encoder.
    """
    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 prime=2147483647, **kwargs):
        super().__init__()
        self.prime = prime

        # Transformer expression encoder
        self.expr_enc = TransformerExpressionEncoder(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers, **kwargs
        )

        # Substitution history encoder
        self.subs_enc = SubstitutionEncoder(embed_dim, n_heads=n_heads, n_layers=2, **kwargs)

        # Action encoder
        self.action_enc = ActionEncoder(embed_dim, **kwargs)

        # Combine expression and substitution history
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
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
        # Encode expression with transformer (get both pooled and per-term)
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

    print("Testing IBP Action Classifier v3...")
    print("=" * 70)

    # Load a few samples
    print("\nLoading classifier training data...")
    with open('/home/shih/work/IBPreduction/data/classifier_training_data_p1009.jsonl') as f:
        samples = [json.loads(line) for line in f][:8]

    print(f"Loaded {len(samples)} samples")

    batch = collate_samples(samples)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # Create model
    model = IBPActionClassifierV3(embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2, prime=1009)
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
        batch['action_mask']
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

    print("\nv3 model test complete!")
