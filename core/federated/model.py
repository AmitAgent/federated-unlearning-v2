"""Model architectures for the federated learning engine (Phase 05).

Two small transformers, selected by ``model_id``:

* ``"transformer"`` -- a Vision Transformer (ViT) for image datasets
  (MNIST, CIFAR-10, CIFAR-100). Parameterized from the dataset manifest
  (channels, num_classes).
* ``"text_transformer"`` -- a text-classification transformer for text
  datasets (AG News): token + position embeddings, padding-masked encoder
  layers, a linear head. Vocabulary size and sequence length are fixed
  constants shared with the tokenizer (``core.datasets.text``), so the
  ``channels`` argument is unused for text.

Every other stage (FedAvg, gradient ascent, distillation, evaluation, MIA)
only calls ``build_model`` and feeds batches through the result, so it is
agnostic to which of the two is used.

Inputs are resized to a fixed 32x32 inside the model, so the 28x28 (MNIST)
vs 32x32 (CIFAR) difference never reaches the patch embedding or the
positional embeddings, and `build_model`'s signature stays unchanged for
every caller (FedAvg, gradient ascent, distillation, evaluation, MIA).

The model uses LayerNorm only (no BatchNorm), so there are no running
statistics to aggregate: FedAvg over `state_dict()` is exact, and
gradient-ascent unlearning does not corrupt any batch statistics.

Use ``check_model_dataset_compatible`` to reject a model/dataset pairing
of the wrong kind (e.g. the image ViT on a text dataset) with a clear error.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from core.datasets.text import PAD_ID, SEQ_LEN, VOCAB_SIZE

#: model id -> the kind of data it consumes ("image" or "text").
MODEL_MODALITY = {"transformer": "image", "text_transformer": "text"}
MODEL_IDS = tuple(MODEL_MODALITY)

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


class TextTransformer(nn.Module):
    """Token embedding + learned positions + pre-norm encoder + linear head.

    Input: LongTensor (B, SEQ_LEN) of token ids, 0 = padding. Position 0 is
    always the [CLS] token, so every row has at least one real token and
    attention never sees an all-masked sequence. The sentence vector is the
    mean of the encoder outputs over non-padding positions.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

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
        x = x.long()
        pad_mask = x == PAD_ID  # (B, L) True where padding
        h = self.token_embed(x) + self.pos_embed[:, : x.size(1)]
        h = self.encoder(self.embed_dropout(h), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)  # (B, L, 1)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(self.norm(pooled))


def build_model(model_id: str, channels: int | None, num_classes: int) -> nn.Module:
    """Factory: look up a model implementation by experiment-config id.

    ``channels`` is only used by the image model; the text model ignores it
    (its input size comes from ``core.datasets.text``).
    """
    if model_id == "transformer":
        return SmallViT(channels=channels, num_classes=num_classes)
    if model_id == "text_transformer":
        return TextTransformer(num_classes=num_classes)
    raise ValueError(f"Unknown model id '{model_id}' (expected one of {', '.join(MODEL_IDS)})")


#: Default local-training learning rate per model. The image ViT trains
#: with SGD (momentum 0.9); the text transformer uses AdamW, which reaches
#: ~80% on AG News after one epoch where SGD reaches only ~55-68%.
DEFAULT_LR = {"transformer": 0.01, "text_transformer": 0.003}


def make_optimizer(model_id: str, parameters, lr: float) -> torch.optim.Optimizer:
    """Local-training optimizer for ``model_id`` (used by every FL client)."""
    if model_id == "text_transformer":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9)


def check_model_dataset_compatible(model_id: str, dataset_meta: dict) -> None:
    """Raise ValueError if ``model_id`` can't consume this dataset's data kind."""
    if model_id not in MODEL_MODALITY:
        raise ValueError(f"Unknown model id '{model_id}' (expected one of {', '.join(MODEL_IDS)})")
    needed = MODEL_MODALITY[model_id]
    actual = dataset_meta.get("modality", "image")
    if needed != actual:
        usable = sorted(m for m, kind in MODEL_MODALITY.items() if kind == actual)
        raise ValueError(
            f"Model '{model_id}' is a {needed} model but dataset "
            f"'{dataset_meta.get('id', '?')}' is {actual} data. "
            f"Use one of: {', '.join(usable)}."
        )
