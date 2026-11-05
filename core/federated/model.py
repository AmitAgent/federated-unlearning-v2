"""Model architecture for the federated learning engine (Phase 05).

A small Vision Transformer (ViT), parameterized from a dataset's manifest
metadata (channels, num_classes) so the same code trains on MNIST,
CIFAR-10, or CIFAR-100 without per-dataset branching.

Inputs are resized to a fixed 32x32 inside the model, so the 28x28 (MNIST)
vs 32x32 (CIFAR) difference never reaches the patch embedding or the
positional embeddings, and `build_model`'s signature stays unchanged for
every caller (FedAvg, gradient ascent, distillation, evaluation, MIA).

The model uses LayerNorm only (no BatchNorm), so there are no running
statistics to aggregate: FedAvg over `state_dict()` is exact, and
gradient-ascent unlearning does not corrupt any batch statistics.

`model_id` is `"transformer"`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MODEL_IDS = ("transformer",)

_IMAGE_SIZE = 32  # inputs are resized to this internally
_PATCH_SIZE = 4   # 32 / 4 = 8 -> 8x8 = 64 patches


class SmallViT(nn.Module):
    """Patch embedding + CLS token + a few pre-norm encoder blocks + linear head."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        num_patches = (_IMAGE_SIZE // _PATCH_SIZE) ** 2

        # Conv with kernel == stride == patch size is the standard (and
        # fastest) way to cut an image into patches and linearly embed them.
        self.patch_embed = nn.Conv2d(
            channels, embed_dim, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
            x = F.interpolate(x, size=(_IMAGE_SIZE, _IMAGE_SIZE), mode="bilinear", align_corners=False)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def build_model(model_id: str, channels: int, num_classes: int) -> nn.Module:
    """Factory: look up a model implementation by experiment-config id."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model id '{model_id}' (expected 'transformer')")
    return SmallViT(channels=channels, num_classes=num_classes)
