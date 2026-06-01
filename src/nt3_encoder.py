"""DNA language model encoder (NTv3 100M_post).

Produces the 768-d variant embeddings consumed by AnomalyModifier. The
DNA language model (DLM) is frozen and used only as an embedding
extractor.

Each variant's input string is the sequence in a 1,000 bp window on each
side of the CPRA (chromosome-position-reference-alternate) coordinate,
with the SNV/indel applied.
"""

from typing import Any

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

_DIM_MAP: dict[str, int] = {
    "InstaDeepAI/NTv3_8M_pre": 256,
    "InstaDeepAI/NTv3_100M_post": 768,
    "InstaDeepAI/NTv3_650M_post": 1536,
}


class NTv3Encoder:
    """Wrapper around InstaDeep NTv3 checkpoints for DNA embedding.

    The _post checkpoints register under AutoModel (encoder-only) and
    require a species_ids argument; the _pre checkpoints register under
    AutoModelForMaskedLM and need neither. Both cases are handled.

    Args:
        model_name: HuggingFace identifier for an NTv3 checkpoint.
        device: Target device; auto-detected when None.
        batch_size: Number of sequences per forward pass.
        mean_pooling: Mean-pool token embeddings to one vector per sequence.
    """

    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_100M_post",
        device: str | None = None,
        batch_size: int = 512,
        mean_pooling: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.mean_pooling = mean_pooling

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        try:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
        except ValueError:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)

        cfg = self.model.config
        self.output_dim = (
            _DIM_MAP.get(model_name)
            or getattr(cfg, "hidden_size", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "d_model", None)
        )
        self._uses_species = callable(
            getattr(self.model, "encode_species", None)
        )

    def freeze(self) -> None:
        """Set eval mode and disable gradients."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _inference(self, sequences: list[str]) -> torch.Tensor:
        """Run one forward pass and return per-token embeddings."""
        batch = self.tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        forward_kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "output_hidden_states": True,
        }
        if self._uses_species:
            forward_kwargs["species_ids"] = self.model.encode_species(
                ["human"] * len(sequences)
            ).to(self.device)
        out = self.model(**forward_kwargs)
        emb = getattr(out, "embedding", None)
        if emb is None:
            hidden_states = getattr(out, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError(
                    "NTv3 model returned neither `embedding` nor "
                    "`hidden_states`."
                )
            emb = hidden_states[-1]
        return emb

    @torch.no_grad()
    def encode(self, sequences: list[str]) -> torch.Tensor:
        """Encode DNA strings into pooled embeddings.

        Args:
            sequences: List of nucleotide strings.

        Returns:
            Embedding tensor of shape (len(sequences), output_dim) when
            mean_pooling is True.
        """
        all_embs: list[torch.Tensor] = []
        for i in range(0, len(sequences), self.batch_size):
            embs = self._inference(sequences[i : i + self.batch_size])
            if self.mean_pooling:
                embs = embs.mean(dim=1)
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)
