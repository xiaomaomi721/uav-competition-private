"""Criterion v2 -- restores the reference point-mean mask BCE.

Why this file exists
--------------------
`loss.py` (v1) computes the training mask BCE as a SUM over every sampled point
of every matched mask, divided only by the number of masks:

    loss_mask = sum_over(points, masks) BCE / num_masks      -> ~8410 for P=12544

The installed reference criterion does not do that. Its real call path is

    point_logits = sample_point(...).squeeze(1)     # (M, P)   -- NOT (M, 1, P)
    point_labels = sample_point(...).squeeze(1)
    cross_entropy_loss.mean(1).sum() / num_masks    # mean over POINTS, sum over masks

so the reference normalises by the point count and the per-mask term is O(1).

v1's parity report did not catch this because the test fed `(M, 1, P)` straight
into a BCE helper, which is not the shape the reference's own `loss_masks()`
produces. That parity item is invalidated by its input shape; the rest of the
report stands. Measured evidence: v1 loss_mask 8410.451172 / 12544 = 0.670476,
matching the per-point BCE mean of ~0.68.

What v2 changes
---------------
Only the mask-BCE reduction. Dice already uses the documented per-pair
semantics (`dice_sum` is normalised per mask, so it is already O(1)); the
matcher, class loss, sampling and matching are all inherited unchanged.

Two reduction modes are provided so the reference contract can be proven
numerically rather than asserted:

  "reference"  per-mask BCE = mean over ALL points, weights ignored.
               This reproduces the installed reference exactly.
  "points"     per-mask BCE = sum(bce * w) / sum(w), ignore-aware.
               Degenerates to the same mean when every point is fully valid,
               and divides by the valid mass so ignored samples neither
               depress the loss nor shrink the gradient.

Both then divide by num_masks exactly once, as the reference does.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .loss import (  # noqa: F401  -- re-exported for callers switching modules
    EPS,
    SAMPLING,
    SetCriterion,
    HungarianMatcher,
    build_set_targets,
    valid_weight_map_from_target,
    sample_map,
    sample_point,
    sample_target_masks,
    random_points,
    weight_at_points,
    dice_sum,
    sigmoid_ce_sum,
)

__all__ = ["SetCriterionV2", "criterion_v2", "CriterionV2Fixture"]

BCE_REDUCTIONS = ("reference", "points")


class SetCriterionV2(SetCriterion):
    """SetCriterion with the mask-BCE reduction corrected to per-mask mean.

    Everything not overridden is inherited from v1: the matcher, the class loss,
    the auxiliary handling and the Dice term are untouched.
    """

    def __init__(self, *args, bce_reduction: str = "reference", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if bce_reduction not in BCE_REDUCTIONS:
            raise ValueError(f"bce_reduction must be one of {BCE_REDUCTIONS}, "
                             f"got {bce_reduction!r}")
        self.bce_reduction = bce_reduction

    @staticmethod
    def mask_bce_per_mask(logits: torch.Tensor, labels: torch.Tensor,
                          weight: torch.Tensor | None = None,
                          reduction: str = "reference") -> torch.Tensor:
        """BCE reduced over points ONLY -> one value per matched mask, shape (M,).

        `logits`/`labels` are sampled maps with a leading mask axis and a single
        channel: (M, 1, P) or already-flattened (M, P).
        """
        if reduction not in BCE_REDUCTIONS:
            raise ValueError(f"unknown reduction {reduction!r}")
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if bce.dim() < 2:
            raise ValueError(f"expected at least (M, P), got {tuple(bce.shape)}")
        bce = bce.flatten(1)                                  # (M, P)
        if reduction == "reference" or weight is None:
            return bce.mean(1)                                # point mean, as installed
        w = weight.flatten(1)                                 # (M, P)
        if w.shape != bce.shape:
            raise ValueError(f"weight {tuple(w.shape)} does not match "
                             f"bce {tuple(bce.shape)}")
        return (bce * w).sum(1) / w.sum(1).clamp_min(EPS)

    def loss_masks(self, mask_logits, targets, indices, points, point_weight,
                   valid_map=None, keep_ratio=0.25):
        b, q = mask_logits.shape[:2]
        num_masks = self._num_masks(targets)
        if valid_map is None:
            sampled_points, sampled_weight = points, point_weight
        else:
            sampled_points, sampled_weight = self._sample_points(
                mask_logits, points, point_weight, keep_ratio)
        loss_mask, loss_dice = 0.0, 0.0
        for i, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            tgt_masks = targets[i]["mask_labels"][tgt]
            src_masks = mask_logits[i][src]
            n_pairs = src_masks.shape[0]
            pts = sampled_points[i:i + 1].expand(n_pairs, -1, -1)
            if valid_map is None:
                sampled_tgt = sample_point(tgt_masks[:, None], pts)     # (M,1,P)
                sampled_out = sample_point(src_masks[:, None], pts)
                # Reference path: one DICE point-pair per mask, no reweighting.
                loss_mask = loss_mask + self.mask_bce_per_mask(
                    sampled_out, sampled_tgt, None, self.bce_reduction).sum()
                loss_dice = loss_dice + dice_sum(sampled_out[..., 0, :],
                                                 sampled_tgt[..., 0, :])
            else:
                sampled_tgt, w = sample_target_masks(
                    tgt_masks[:, None],
                    valid_map[i:i + 1].expand(n_pairs, -1, -1), pts)
                sampled_out = sample_point(src_masks[:, None], pts)
                loss_mask = loss_mask + self.mask_bce_per_mask(
                    sampled_out, sampled_tgt, w, self.bce_reduction).sum()
                loss_dice = loss_dice + dice_sum(sampled_out, sampled_tgt, w)
        if isinstance(loss_mask, float):
            zero = mask_logits.sum() * 0.0
            loss_mask, loss_dice = zero, zero
        return {"mask": loss_mask / num_masks, "dice": loss_dice / num_masks}


def criterion_v2(num_classes: int = 8, bce_reduction: str = "reference",
                 weight_dict=None, **kwargs) -> SetCriterionV2:
    """Default training criterion for M2F v2 (reference point-mean BCE)."""
    return SetCriterionV2(num_classes=num_classes, bce_reduction=bce_reduction,
                          weight_dict=weight_dict, **kwargs)


class CriterionV2Fixture:
    """Helper for pairing v2 against the installed reference in tests."""

    def __init__(self, reference_loss, num_masks: int) -> None:
        self.reference_loss = reference_loss
        self.num_masks = int(num_masks)
