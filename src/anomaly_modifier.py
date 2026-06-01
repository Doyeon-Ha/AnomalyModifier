"""AnomalyModifier model.

A (causal, modifier-candidate) variant embedding pair, fused with eight
disease-aware gene features on the modifier-candidate side, is mapped to a
32-d latent z. The anomaly score is the squared latent distance from the
SVDD center mu.

Forward path:

    causal embedding             (B, 768) -> Causal Adapter             ---+
                                                                           +-> Transformer -> Latent Head -> z (B, 32)
    modifier-candidate embedding (B, 768) -> Modifier-candidate Adapter ---+
                                             + Gene-feature Fusion          |
    gene features                (B, 8)   -> Gene-feature Adapter ----------+

    anomaly score  s(c, m) = ||z - mu||^2

Input embeddings are produced by a frozen NTv3 100M_post DNA language
model (see nt3_encoder.py).
"""

import torch
import torch.nn as nn

from transformer import Transformer


class AnomalyModifier(nn.Module):
    """AnomalyModifier model.

    Args:
        embedding_dim: Dimensionality of the input DLM embedding.
        hidden_dim: Dimensionality of the adapters and the Transformer.
        latent_dim: Dimensionality of the latent z.
        num_heads: Attention heads in the Transformer.
        num_layers: Number of Transformer encoder layers.
        dropout: Global dropout probability.
        gene_feature_dim: Number of disease-aware gene features.
        gf_dropout: Dropout inside the Gene-feature Adapter.
        objective: SVDD objective, "soft-boundary" or "one-class".
        nu: Soft-boundary trade-off parameter.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        latent_dim: int = 32,
        num_heads: int = 12,
        num_layers: int = 10,
        dropout: float = 0.2,
        gene_feature_dim: int = 8,
        gf_dropout: float = 0.5,
        objective: str = "soft-boundary",
        nu: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gene_feature_dim = gene_feature_dim
        self.objective = objective
        self.nu = nu

        # Per-stream adapters: Linear -> LN -> GELU -> Dropout.
        self.causal_adapter = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.modifier_candidate_adapter = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gene-feature Adapter projects the eight features to the variant
        # embedding dimension.
        self.gene_feature_adapter = nn.Sequential(
            nn.Linear(gene_feature_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(gf_dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        # Modifier-candidate and Gene-feature Fusion: the projected gene
        # features are concatenated with the Modifier-candidate Adapter
        # output and linearly mixed.
        self.gene_feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.transformer = Transformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Latent Head maps the Transformer output h to the latent z.
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

        # Decoder reconstructs the Transformer output h from z for the
        # reconstruction term of the hypersphere-regularized AE loss.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.transformer.output_dim, bias=False),
        )

        # SVDD center mu and radius R are non-learnable buffers updated by
        # deterministic rules during training. The radius is a calibrated
        # threshold and does not enter the anomaly score.
        self.register_buffer("center", torch.zeros(latent_dim))
        self.register_buffer("center_initialized", torch.tensor(False))
        if objective == "soft-boundary":
            self.register_buffer("R", torch.tensor(0.0))

    def encode(
        self,
        causal_emb: torch.Tensor,
        modifier_candidate_emb: torch.Tensor,
        gene_features: torch.Tensor | None = None,
        return_transformer_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Map a variant pair to the latent z.

        Args:
            causal_emb: Causal variant embeddings of shape
                (batch, embedding_dim).
            modifier_candidate_emb: Modifier-candidate variant embeddings
                of shape (batch, embedding_dim).
            gene_features: Modifier-candidate gene features of shape
                (batch, gene_feature_dim). May be omitted.
            return_transformer_output: If True, also return the Transformer
                output h.

        Returns:
            The latent z of shape (batch, latent_dim), or a (z, h) tuple
            when return_transformer_output is True.
        """
        c = self.causal_adapter(causal_emb)
        m = self.modifier_candidate_adapter(modifier_candidate_emb)

        if gene_features is not None:
            g = self.gene_feature_adapter(gene_features)
            m = self.gene_feature_fusion(torch.cat([m, g], dim=-1))

        h = self.transformer(c, m)
        z = self.latent_head(h)
        return (z, h) if return_transformer_output else z

    def compute_anomaly_score(
        self,
        causal_emb: torch.Tensor,
        modifier_candidate_emb: torch.Tensor,
        gene_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the anomaly score s(c, m) = ||z - mu||^2.

        Higher scores indicate stronger suppressor-modifier signal.

        Args:
            causal_emb: Causal variant embeddings (batch, embedding_dim).
            modifier_candidate_emb: Modifier-candidate variant embeddings
                (batch, embedding_dim).
            gene_features: Modifier-candidate gene features
                (batch, gene_feature_dim).

        Returns:
            Score tensor of shape (batch,).
        """
        z = self.encode(causal_emb, modifier_candidate_emb, gene_features)
        return torch.sum((z - self.center) ** 2, dim=1)

    def forward(
        self,
        causal_emb: torch.Tensor,
        modifier_candidate_emb: torch.Tensor,
        gene_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for compute_anomaly_score."""
        return self.compute_anomaly_score(
            causal_emb, modifier_candidate_emb, gene_features
        )
