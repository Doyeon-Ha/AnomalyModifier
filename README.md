# AnomalyModifier Model Architecture

Reference implementation of the model architecture from the paper:

> **AnomalyModifier: Suppressor Modifier Discovery in Familial Hypercholesterolemia via One-Class Anomaly Detection**

AnomalyModifier is a one-class anomaly detection model for suppressor
modifier discovery. This folder contains the network definition needed to
reconstruct and run the model.

## Files

```
src/
├── anomaly_modifier.py    # the AnomalyModifier network
├── transformer.py         # the Transformer over the two variant tokens
├── nt3_encoder.py         # the DNA language model encoder (NTv3 100M_post)
└── config.yaml            # the model hyper-parameters
```

| File | Purpose |
|------|---------|
| `src/anomaly_modifier.py` | The AnomalyModifier network: adapters, gene-feature fusion, Transformer, latent head, decoder, SVDD buffers, and the anomaly-score readout. |
| `src/transformer.py` | The Transformer over the causal and modifier-candidate tokens. |
| `src/nt3_encoder.py` | The DNA language model encoder (NTv3 100M_post) that maps a variant sequence to a 768-d embedding. |
| `src/config.yaml` | The model hyper-parameters. |

## Architecture

```
variant sequence (1,000 bp on each side of CPRA, SNV/indel applied)
   |   frozen NTv3 100M_post, mean-pooled        (src/nt3_encoder.py)
   v
causal embedding             (B, 768) -> Causal Adapter             -----------+
                                         Linear -> LN -> GELU -> Drop(0.2)      |
                                                                                v
modifier-candidate embedding (B, 768) -> Modifier-candidate Adapter -+     Transformer
                                         Linear -> LN -> GELU -> Drop |     10 layers, 12 heads
gene features                (B, 8)   -> Gene-feature Adapter --------+     FFN 3072, pre-norm
                                         Linear -> LN -> GELU ->      |          |
                                         Drop(0.5) -> Linear          |          v
                                           +- concat + fusion -> Latent Head
                                              Linear 1536->768     768->768->32
                                                                         |
                                                                         v
                                                                    z  (B, 32)

anomaly score  s(c, m) = ||z - mu||^2
```

### Dimensions

- DLM embedding: 768 (NTv3 100M_post, frozen, mean-pooled)
- Adapter / Transformer dim: 768
- Transformer: 10 layers, 12 heads, FFN 3072, dropout 0.2, pre-norm
- Gene features: 8 disease-aware features, concat fusion, gene-feature dropout 0.5
- Latent dim: 32
- SVDD: soft-boundary, nu = 0.1, center_eps = 0.1

### Training objective and buffers

The model is trained by a hypersphere-regularized autoencoder loss over
patient variant pairs:

    L_AE = L_recon + lambda_c * L_cluster + lambda_a * L_norm

The `decoder` reconstructs the Transformer output h from z
(L_recon = ||Dec(z) - sg(h)||^2, where sg is stop-gradient) and is active
throughout training.

`center` (mu) and `R` are buffers, not learned parameters. The center is
fixed to the patient mean after a short warmup, and `R` is the (1 - nu)
quantile of patient distances, refreshed each epoch. The radius is a
calibrated threshold and does not appear in the anomaly score, which uses
only the latent distance from the center.

## Usage

The snippets below import the modules directly, so run them from the
`src/` directory.

```python
import torch
from anomaly_modifier import AnomalyModifier

model = AnomalyModifier()
model.eval()

B = 4
causal = torch.randn(B, 768)
modifier_candidate = torch.randn(B, 768)
gene_feat = torch.randn(B, 8)

scores = model(causal, modifier_candidate, gene_feat)   # (B,), higher = more anomalous
```

### From a variant sequence

```python
from nt3_encoder import NTv3Encoder

enc = NTv3Encoder("InstaDeepAI/NTv3_100M_post")
enc.freeze()
causal = enc.encode([causal_seq])
modifier_candidate = enc.encode([modifier_candidate_seq])
score = model(causal, modifier_candidate, gene_feat)
```
