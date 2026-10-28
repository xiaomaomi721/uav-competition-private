"""B3 + Mask2Former-style head v0 -- independent experiment module.

Scope (frozen by 正式路线与预注册规则_20260915.md):
  * This is NOT a full Mask2Former reproduction. The pixel decoder is a simple
    FPN; the mask-classification paradigm is kept intact:
      learned mask queries / class+mask prediction / masked cross-attention
      refinement / Hungarian set prediction with class+mask losses.
  * Training interface and inference interface are separate. The set-prediction
    trainer consumes `class_logits`/`mask_logits`/`aux_outputs`; anything that
    needs a pixel-argmax consumes `semantic_scores()`. We deliberately do not
    force the frozen SegFormer CE contract onto this head.
  * Ignore indexing is excluded from matching, point sampling, BCE and Dice.
  * A query whose predicted mask is empty at every level must not produce NaNs.

CPU-only in this module: no CUDA calls, no device placement beyond cpu.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["B3M2FStyleHead", "SimplePixelDecoder", "QueryDecoderLayer",
           "SinePositionEmbedding", "build_head"]


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 32
    while groups > 1 and channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
            _group_norm(cout),
            nn.ReLU(inplace=True),
        )


class SinePositionEmbedding(nn.Module):
    """Fixed 2D sine embedding, shape (B, dim, H, W)."""

    def __init__(self, dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError(f"dim must be divisible by 4, got {dim}")
        self.dim = dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        dtype, device = x.dtype, x.device
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype), indexing="ij")
        ys = ys / max(h - 1, 1)
        xs = xs / max(w - 1, 1)
        freq = self.dim // 4
        dim_t = self.temperature ** (torch.arange(freq, device=device, dtype=dtype) / freq)
        px = xs.reshape(-1)[:, None] / dim_t
        py = ys.reshape(-1)[:, None] / dim_t
        pe = torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], dim=-1)
        return pe.transpose(0, 1).reshape(b, self.dim, h, w)


class SimplePixelDecoder(nn.Module):
    """FPN over the four SegFormer stages.

    Produces `mask_features` at the stride-4 resolution of stage 1 and three
    multi-scale levels at strides 8/16/32. No deformable attention (see
    接口核查结论.md: per-level grid_sample is too expensive here).
    """

    def __init__(self, in_channels=(64, 128, 320, 512), hidden_dim: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"expected 4 encoder stages, got {len(in_channels)}")
        self.in_channels = tuple(in_channels)
        self.hidden_dim = hidden_dim
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, 1) for c in in_channels])
        self.smooth = nn.ModuleList(
            [ConvNormAct(hidden_dim, hidden_dim, 3) for _ in range(3)])

    def forward(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise TypeError("SimplePixelDecoder expects the 4-tuple of encoder stages")
        for f in features:
            if f.dim() != 4:
                raise ValueError(f"encoder stage must be BCHW, got {tuple(f.shape)}")
        p1, p2, p3, p4 = (lat(f) for lat, f in zip(self.lateral, features))
        up3 = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth[2](p3 + up3)
        up2 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth[1](p2 + up2)
        up1 = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        mask_features = self.smooth[0](p1 + up1)
        return {"levels": [p2, p3, p4], "mask_features": mask_features}


class QueryDecoderLayer(nn.Module):
    """Self-attention + masked cross-attention + FFN over the query set."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, level: torch.Tensor,
                level_pe: torch.Tensor, attn_mask: torch.Tensor | None):
        # queries (Q,B,D); level/level_pe (B,D,H,W); attn_mask (B*nhead,Q,HW), True = blocked
        kv = (level + level_pe).flatten(2).permute(2, 0, 1).contiguous()  # (HW,B,D)
        q = self.norm1(queries)
        q = q + self.self_attn(q, q, q, need_weights=False)[0]
        q2 = self.norm2(q)
        q2 = q2 + self.cross_attn(q2, kv, kv, attn_mask=attn_mask,
                                  need_weights=False)[0]
        q3 = self.norm3(q2)
        return q3 + self.linear2(self.dropout(F.relu(self.linear1(q3))))


class B3M2FStyleHead(nn.Module):
    """Mask-classification head consuming the four SegFormer encoder stages.

    forward(features) -> {"class_logits": (B,Q,C+1), "mask_logits": (B,Q,h,w),
                          "aux_outputs": [same keys, ...]}
    The class channel C is no-object. Semantic output is derived separately by
    `semantic_scores()`, never by argmax over queries.
    """

    def __init__(self, in_channels=(64, 128, 320, 512), num_classes: int = 8,
                 hidden_dim: int = 256, num_queries: int = 100, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024,
                 dropout: float = 0.0, ignore_index: int = 255,
                 prior_prob: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.num_queries = num_queries
        self.nhead = nhead
        self.pixel_decoder = SimplePixelDecoder(in_channels, hidden_dim)
        self.position = SinePositionEmbedding(hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList(
            [QueryDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
             for _ in range(num_layers)])
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))
        # Set False only in tests that demonstrate the guard is load-bearing.
        self.nan_guard = True
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        if prior_prob > 0:
            bias = -float(torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
            nn.init.constant_(self.class_embed.bias, bias)
        with torch.no_grad():
            for layer in self.mask_embed:
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(
                        F.normalize(layer.weight, dim=1) * (layer.weight.shape[1] ** -0.5))

    def decode(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise TypeError("expected the 4-tuple of encoder stages (CWFA contract)")
        decoded = self.pixel_decoder(features)
        levels, mask_features = decoded["levels"], decoded["mask_features"]
        level_pe = [self.position(level) for level in levels]
        batch = mask_features.shape[0]
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)
        mask_logits = None
        outputs = []
        for index, layer in enumerate(self.layers):
            level_index = index % len(levels)
            level = levels[level_index]
            attn_mask = (None if mask_logits is None else
                         self.build_attention_mask(mask_logits, level.shape[-2:]))
            queries = layer(queries, level, level_pe[level_index], attn_mask)
            class_logits = self.class_embed(queries).permute(1, 0, 2)
            mask_logits = self.predict_masks(queries, mask_features)
            outputs.append({"class_logits": class_logits, "mask_logits": mask_logits,
                            "layer": index})
        return outputs

    def predict_masks(self, queries: torch.Tensor, mask_features: torch.Tensor):
        embedding = self.mask_embed(queries)                 # (Q,B,D)
        return torch.einsum("qbd,bdhw->bqhw", embedding, mask_features)

    def build_attention_mask(self, mask_logits: torch.Tensor, size) -> torch.Tensor:
        """Blocked positions per query at one level.

        A query whose predicted mask is empty everywhere would give an all-blocked
        softmax row -> NaN that survives into the backward pass. When nan_guard is
        on, such a row opens the position with the strongest mask response.
        """
        resized = F.interpolate(mask_logits, size=size, mode="bilinear",
                                align_corners=False)
        blocked = (resized.sigmoid() < 0.5).flatten(2)       # (B,Q,HW)
        if self.nan_guard:
            empty = blocked.all(-1)
            if bool(empty.any()):
                best = resized.flatten(2).argmax(-1)
                blocked = blocked.clone()
                b_idx, q_idx = empty.nonzero(as_tuple=True)
                blocked[b_idx, q_idx, best[b_idx, q_idx]] = False
        return blocked.unsqueeze(1).repeat(1, self.nhead, 1, 1).flatten(0, 1)

    def forward(self, features):
        outputs = self.decode(features)
        final = outputs[-1]
        return {"class_logits": final["class_logits"],
                "mask_logits": final["mask_logits"],
                "aux_outputs": outputs[:-1]}

    def semantic_scores(self, class_logits: torch.Tensor,
                        mask_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        from .semantic import semantic_scores
        return semantic_scores(class_logits, mask_logits, **kwargs)


def build_head(**kwargs) -> B3M2FStyleHead:
    return B3M2FStyleHead(**kwargs)
