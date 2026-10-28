"""Semantic synthesis for B3 + Mask2Former-style head v0.

Mask classification produces per-query class scores and per-query masks. The
official pipeline turns them into a segmentation by

    class_prob = softmax(class_logits)[..., :C]      # drop no-object
    mask_prob  = sigmoid(mask_logits)
    score      = einsum("bqc,bqhw->bchw", class_prob, mask_prob)

which is a *product of probabilities*, not a softmax over classes. It lives in a
different score space from our SegFormer champion logits, so the historical
CHAMPION bias must NOT be transplanted onto it unchanged; calibration for this
family has to be defined and measured on its own (see 正式路线与预注册规则).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["semantic_scores", "semantic_log_scores", "predict_masks_official_style"]


def semantic_scores(class_logits: torch.Tensor, mask_logits: torch.Tensor,
                    upsample_to=None) -> torch.Tensor:
    """(B,Q,C+1),(B,Q,h,w) -> (B,C,H,W) probability mixture."""
    if class_logits.dim() != 3 or mask_logits.dim() != 4:
        raise ValueError("expected (B,Q,C+1) and (B,Q,h,w)")
    if class_logits.shape[:2] != mask_logits.shape[:2]:
        raise ValueError("class/mask query dimensions disagree")
    class_prob = class_logits.softmax(-1)[..., :-1]
    mask_prob = mask_logits.sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    if upsample_to is not None and tuple(scores.shape[-2:]) != tuple(upsample_to):
        scores = F.interpolate(scores, size=tuple(upsample_to), mode="bilinear",
                               align_corners=False)
    return scores


def semantic_log_scores(class_logits: torch.Tensor, mask_logits: torch.Tensor,
                        eps: float = 1e-6, upsample_to=None) -> torch.Tensor:
    """log(mixture + eps): keeps per-pixel argmax but is CE-consistent.

    NOTE: log(score + eps) is *not* affine-equivalent to log(score) for the eps
    term, but it is monotone in `score`, so argmax is preserved exactly for any
    eps > 0. Use this when the surrounding engine requires logits; report the eps.
    """
    scores = semantic_scores(class_logits, mask_logits, upsample_to)
    return torch.log(scores + eps)


def predict_masks_official_style(class_logits: torch.Tensor,
                                 mask_logits: torch.Tensor) -> torch.Tensor:
    """argmax of the mixture, returned as official ids 1..C (no no-object class)."""
    return semantic_scores(class_logits, mask_logits).argmax(1) + 1
