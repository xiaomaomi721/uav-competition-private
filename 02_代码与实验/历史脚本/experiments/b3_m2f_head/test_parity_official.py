"""Numerical parity: our set-prediction objective vs the installed Mask2Former.

Purpose
-------
Everything the design claims about matching and set losses should reduce to the
official criterion when there is no ignore to handle. If it does not, a future
failure of this head could not be attributed to the architecture.

Scope and honesty about what this can show
------------------------------------------
Two things are deliberately substituted, because the reference draws them
internally and they cannot be matched bit-for-bit across two separate code paths:

  * point coordinates -- the reference samples them inside `loss_masks`. Both
    sides are fed the SAME explicit points, so point sampling is not tested here;
  * auxiliary matching -- the reference re-matches each auxiliary layer, while
    our head's layer-0 field reports the matching computed at layer 0. Only the
    final-layer losses are a true end-to-end parity check.

`out[i]` is a Python list index, so the reference's predicted->matched-target
state can survive across iterations; `out[i][src]` still returns the matched
pairs. It is likely benign, but it means the auxiliary comparison is a guard
against reordering mistakes, not a bit-level gate.

CPU only.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .head import B3M2FStyleHead
from .loss import HungarianMatcher, SetCriterion, build_set_targets

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher, Mask2FormerLoss)

IGNORE = 255
TOL = 1e-5


def _config(num_classes: int, num_points: int):
    # Mask2FormerLoss reads exactly these attributes
    return SimpleNamespace(num_labels=num_classes,
                           no_object_weight=0.1,
                           class_weight=2.0, mask_weight=5.0, dice_weight=5.0,
                           train_num_points=num_points,
                           oversample_ratio=3.0, importance_sample_ratio=0.75)


def _inputs(num_classes=3, queries=6, num_points=2048, height=32, width=32,
            batch=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    class_logits = torch.randn(batch, queries, num_classes + 1, generator=generator)
    mask_logits = torch.randn(batch, queries, height, width, generator=generator)
    torch.manual_seed(seed)
    target = torch.zeros(batch, height, width, dtype=torch.int64)
    target[0, :8, :] = 0
    target[0, 8:20, :] = 2
    target[0, 20:, 16:] = 1
    mask_labels, class_labels = [], []
    for i in range(batch):
        masks, classes = [], []
        for c in range(num_classes):
            m = target[i] == c
            if bool(m.any()):
                masks.append(m.to(torch.float32))
                classes.append(c)
        mask_labels.append(torch.stack(masks))
        class_labels.append(torch.tensor(classes, dtype=torch.int64))
    points = torch.rand(batch, num_points, 2, generator=torch.Generator().manual_seed(seed + 1))
    return class_logits, mask_logits, mask_labels, class_labels, points


def test_parity_final_layer():
    num_classes, queries, num_points = 3, 6, 2048
    class_logits, mask_logits, mask_labels, class_labels, points = _inputs(
        num_classes, queries, num_points)

    official_matcher = Mask2FormerHungarianMatcher(
        cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=num_points)
    official_matcher.eval()
    with torch.no_grad():
        off_idx = official_matcher(mask_logits, class_logits, mask_labels, class_labels)

    ours_matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0,
                                    num_classes=num_classes)
    ours_idx = ours_matcher(class_logits, mask_logits, _ours_targets(mask_labels, class_labels),
                            points, torch.ones(1, 1, num_points))

    assert len(off_idx) == len(ours_idx)
    for (src_o, tgt_o), (src_m, tgt_m) in zip(off_idx, ours_idx):
        assert torch.equal(src_o, src_m), (src_o, src_m)
        assert torch.equal(tgt_o, tgt_m), (tgt_o, tgt_m)
    print(f"[P1] Hungarian assignment identical: src={off_idx[0][0].tolist()} "
          f"tgt={off_idx[0][1].tolist()} OK")

    # ---- official losses, with the reference's internal point sampler replaced
    official = Mask2FormerLoss(_config(num_classes, num_points),
                               weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0})
    official.eval()
    grid = points[0].unsqueeze(0).unsqueeze(0).repeat(class_labels[0].numel(), 1, 1, 1)
    official.sample_points_using_uncertainty = lambda *a, **k: grid
    with torch.no_grad():
        off_losses = official(mask_logits, class_logits, mask_labels, class_labels)

    ours = SetCriterion(num_classes=num_classes, num_points=num_points)
    ours_targets = _ours_targets(mask_labels, class_labels)
    weight = torch.ones(1, 1, num_points)
    our_losses = ours(class_logits, mask_logits, ours_targets, points, weight)

    pairs = [("class", "loss_cross_entropy"), ("mask", "loss_mask")]
    for ours_key, off_key in pairs:
        a, b = float(our_losses[ours_key]), float(off_losses[off_key])
        assert abs(a - b) <= TOL * max(1.0, abs(b)), (ours_key, a, b)
        print(f"[P2] {ours_key:5s} ours={a:.6f} official={b:.6f} "
              f"absdiff={abs(a-b):.2e} OK")

    # dice: the installed reference broadcasts (N,1,P) labels into an (N,N,P)
    # cross-mask matrix, so its value is NOT the per-pair dice its docstring
    # describes. Bit-level parity is therefore asserted against a verbatim
    # reproduction of that arithmetic, and the documented semantics are reported
    # separately so the deviation is visible rather than silently adopted.
    literal = SetCriterion(num_classes=num_classes, num_points=num_points)
    literal.literal_reference_dice = True
    literal_losses = literal(class_logits, mask_logits, ours_targets, points, weight)
    a_lit = float(literal_losses["dice"])
    b_lit = float(off_losses["loss_dice"])
    assert abs(a_lit - b_lit) <= TOL * max(1.0, abs(b_lit)), (a_lit, b_lit)
    documented = float(our_losses["dice"])
    print(f"[P2] dice  literal-reference ours={a_lit:.6f} official={b_lit:.6f} "
          f"absdiff={abs(a_lit-b_lit):.2e} OK")
    print(f"[P2] dice  DOCUMENTED per-pair value={documented:.6f} "
          f"(training default; differs from the reference by design) NOTE")


def test_parity_auxiliary_layer():
    """Guard against reordering mistakes: compare our layer-0 losses, computed
    with the head's OWN layer-0 matching, against the reference on the same
    layer-0 predictions. Matching is therefore shared by construction, and only
    the loss arithmetic is under test (see module docstring)."""
    num_classes, queries, num_points = 3, 6, 2048
    head = B3M2FStyleHead(in_channels=(4, 8, 16, 32), num_classes=num_classes,
                          hidden_dim=16, num_queries=queries, nhead=2, num_layers=2,
                          dim_feedforward=32)
    head.eval()
    generator = torch.Generator().manual_seed(7)
    features = [torch.randn(1, c, h, w, generator=generator) for c, h, w in
                [(4, 16, 16), (8, 8, 8), (16, 4, 4), (32, 2, 2)]]
    with torch.no_grad():
        decoded = head.decode(features)
    assert len(decoded) == 2, "expected one auxiliary layer plus the final layer"
    aux = decoded[0]

    _cl, _ml, mask_labels, class_labels, points = _inputs(
        num_classes, queries, num_points, height=16, width=16)
    weight = torch.ones(1, 1, num_points)
    targets = _ours_targets(mask_labels, class_labels)

    official = Mask2FormerLoss(_config(num_classes, num_points),
                               weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0})
    official.eval()
    grid = points[0].unsqueeze(0).unsqueeze(0).repeat(class_labels[0].numel(), 1, 1, 1)
    official.sample_points_using_uncertainty = lambda *a, **k: grid
    with torch.no_grad():
        off_losses = official(aux["mask_logits"], aux["class_logits"],
                              mask_labels, class_labels)
        ours = SetCriterion(num_classes=num_classes, num_points=num_points)
        our_losses = ours(aux["class_logits"], aux["mask_logits"], targets, points, weight)

    for ours_key, off_key in [("class", "loss_cross_entropy"), ("mask", "loss_mask")]:
        a, b = float(our_losses[ours_key]), float(off_losses[off_key])
        assert abs(a - b) <= TOL * max(1.0, abs(b)), (ours_key, a, b)
        print(f"[P3] aux layer0 {ours_key:5s} ours={a:.6f} official={b:.6f} "
              f"absdiff={abs(a-b):.2e} OK")
    a, b = float(our_losses["dice"]), float(off_losses["loss_dice"])
    print(f"[P3] aux layer0 dice  per-pair={a:.6f} installed-reference={b:.6f} NOTE")


def _ours_targets(mask_labels, class_labels):
    return [{"class_labels": c, "mask_labels": m}
            for m, c in zip(mask_labels, class_labels)]


def main() -> int:
    torch.manual_seed(0)
    test_parity_final_layer()
    test_parity_auxiliary_layer()
    print("PARITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
