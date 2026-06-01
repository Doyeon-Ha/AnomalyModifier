"""Transformer module.

Combines the causal and modifier-candidate adapter outputs into a
length-2 sequence, augments each token with a segment embedding
(causal = 0, modifier-candidate = 1), applies a multi-layer Transformer
encoder, and mean-pools the result into a single vector h.
"""

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Transformer over the causal and modifier-candidate tokens.

    The two adapter outputs are stacked into a length-2 sequence
    (batch, 2, hidden_dim). A learnable segment embedding marks the
    causal token (id 0) and the modifier-candidate token (id 1). A
    pre-norm Transformer encoder mixes the two tokens; the output is
    mean-pooled over the sequence and projected to produce h.

    Args:
        hidden_dim: Model dimensionality. Must be divisible by num_heads.
        num_heads: Number of attention heads per encoder layer.
        num_layers: Number of stacked Transformer encoder layers.
        dropout: Dropout probability inside the encoder layers.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.segment_embedding = nn.Embedding(2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, causal: torch.Tensor, modifier_candidate: torch.Tensor
    ) -> torch.Tensor:
        """Mix the two tokens with self-attention and mean-pool.

        Args:
            causal: Causal adapter output of shape (batch, hidden_dim).
            modifier_candidate: Modifier-candidate adapter output of shape
                (batch, hidden_dim).

        Returns:
            Transformer output h of shape (batch, hidden_dim).
        """
        combined = torch.stack([causal, modifier_candidate], dim=1)
        batch_size = causal.size(0)
        seg_ids = (
            torch.tensor([0, 1], device=causal.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        combined = combined + self.segment_embedding(seg_ids)
        attended = self.encoder(combined)
        pooled = attended.mean(dim=1)
        return self.output_proj(pooled)
