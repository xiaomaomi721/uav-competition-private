"""CPU invariants for the B3 + Mask2Former-style head v0.

Six staged tests, each one asserting a property the design claims:

  1. perfect prediction      -- loss of an exact prediction is the floor
  2. ignore invariance       -- changing predictions on ignored pixels changes nothing
  3. permutation invariance  -- GT order cannot matter after Hungarian matching
  4. all-masked attention    -- an empty predicted mask cannot produce NaN
  5. semantic synthesis      -- hand-computed mixture matches the implementation
  6. empty targets           -- absent classes / all-ignore images stay finite

Only stdlib asserts + torch on CPU.
"""
from __future__ import annotations

import torch

from .head import B3M2FStyleHead
from .loss import (HungarianMatcher, SetCriterion, build_set_targets, dice_loss,
                   sample_map, sample_target_masks, sigmoid_ce,
                   valid_weight_map_from_target, weight_at_points)
from .semantic import predict_masks_official_style, semantic_scores

IGNORE = 255


def _tiny_head(queries=4, classes=3, layers=2):
    torch.manual_seed(0)
    return B3M2FStyleHead(in_channels=(4, 8, 16, 32), num_classes=classes,
                          hidden_dim=16, num_queries=queries, nhead=2,
                          num_layers=layers, dim_feedforward=32)


def _features(b=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, c, h, w, generator=g) for c, h, w in
            [(4, 16, 16), (8, 8, 8), (16, 4, 4), (32, 2, 2)]]


def _targets_from(class_masks, h=16, w=16, ignore_regions=()):
    target = torch.full((1, h, w), IGNORE, dtype=torch.int64)
    for index, mask in enumerate(class_masks):
        target[0][mask] = index
    for region in ignore_regions:
        target[0][region] = IGNORE
    return target


# ---------------------------------------------------------------- 1. perfect
def test_perfect_prediction_is_floor():
    h = w = 8
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[2:6, 2:6] = True
    gt = mask.float().unsqueeze(0)
    # an ideal classifier/mask pair: high logit on the true class, saturated mask
    good = torch.logit(torch.tensor(1 - 1e-4)).item()
    bad = torch.logit(torch.tensor(1e-4)).item()
    logits = torch.full((1, 2, h, w), bad)
    logits[0, 0][mask] = good
    logits[0, 1][~mask] = good
    labels = torch.cat([gt, 1 - gt]).unsqueeze(0)             # (1,2,h,w)
    assert sigmoid_ce(logits, labels).item() < 1e-3
    assert dice_loss(logits, labels).item() < 1e-3
    noisy = logits + 2.0 * torch.randn_like(logits)
    assert sigmoid_ce(noisy, labels).item() > sigmoid_ce(logits, labels).item()
    assert dice_loss(noisy, labels).item() > dice_loss(logits, labels).item()
    print("[1] perfect-prediction floor: OK")


# ----------------------------------------------------------------- 2. ignore
def test_ignore_invariance():
    h = w = 8
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[1:5, 1:5] = True
    ignore = torch.zeros(h, w, dtype=torch.bool)
    ignore[0, :] = True
    target = torch.zeros(1, h, w, dtype=torch.int64)
    target[0][~mask] = 1
    target[0][ignore] = IGNORE
    wmap = valid_weight_map_from_target(target, IGNORE)       # (1,8,8) full space
    assert wmap.shape == target.shape
    points = torch.rand((1, 128, 2), generator=torch.Generator().manual_seed(3))
    point_weight = weight_at_points(wmap, points)
    assert (point_weight >= 0).all() and (point_weight <= 1 + 1e-5).all()

    logits = torch.randn(1, 2, h, w, generator=torch.Generator().manual_seed(4))
    labels = torch.stack([mask.float(), (~mask).float()]).unsqueeze(0)

    # (i) full-resolution path: weights must live in full-image space, where the
    # support is 7x7, not 8x8 -- the loss broadcasts across the 8x8 maps.
    full_w = wmap
    base_ce = sigmoid_ce(logits, labels, full_w).item()
    base_dice = dice_loss(logits, labels, full_w).item()

    # mutate ONLY ignored positions, including points whose 2x2 footprint touches
    # an ignored pixel -- exactly the leak a plain zero-mask would allow
    mutated = logits.clone()
    mutated[:, :, ignore] = 50.0
    mutated[:, :, 0, :] = -50.0
    assert sigmoid_ce(mutated, labels, full_w).item() == base_ce
    assert dice_loss(mutated, labels, full_w).item() == base_dice

    # (ii) sampled-point path used by matching: mutate the sampled values only at
    # points whose weight is zero and confirm the cost is unchanged.
    sampled_logits = torch.randn(1, 2, 128, generator=torch.Generator().manual_seed(9))
    sampled_labels = (torch.rand(1, 2, 128, generator=torch.Generator().manual_seed(10)) > 0.5)
    sampled_labels = sampled_labels.float()
    zero_w = (point_weight < 1e-6).float()
    ce_a = sigmoid_ce(sampled_logits, sampled_labels, point_weight).item()
    ce_b = sigmoid_ce(sampled_logits * (1 - zero_w) + 25.0 * zero_w,
                      sampled_labels, point_weight).item()
    assert abs(ce_a - ce_b) < 1e-6
    dice_a = dice_loss(sampled_logits, sampled_labels, point_weight).item()
    dice_b = dice_loss(sampled_logits * (1 - zero_w) + 25.0 * zero_w,
                       sampled_labels, point_weight).item()
    assert abs(dice_a - dice_b) < 1e-6

    # a point inside the ignore band must carry zero (or near-zero) weight
    inside = torch.tensor([[[0.5, 0.01]]])
    assert weight_at_points(wmap, inside).item() < 0.2

    # (iii) target renormalisation: a target sample whose bilinear footprint is
    # partly ignore is dragged toward 0 by a naive sample. Renormalising by the
    # valid fraction must recover the mean label among valid neighbours.
    h2 = w2 = 4
    tgt2 = torch.zeros(1, h2, w2, dtype=torch.int64)           # valid rows are class 0
    tgt2[0, 0, :] = IGNORE                                     # whole first row ignored
    one_class = (tgt2 == 0).float().unsqueeze(0)               # (1,1,h,w)
    vmap2 = valid_weight_map_from_target(tgt2, IGNORE)
    # (x, y) with align_corners=False: y=0.30 lands between rows 0 and 1, so the
    # footprint is 30% ignored row and 70% valid row
    boundary_pt = torch.tensor([[[0.375, 0.30]]])
    naive = sample_map(one_class, boundary_pt)[0, 0, 0].item()
    renorm, wval = sample_target_masks(one_class, vmap2, boundary_pt)
    assert abs(naive - 0.7) < 1e-5, naive                      # biased low
    assert abs(renorm[0, 0, 0].item() - 1.0) < 1e-6, renorm.item()
    assert abs(wval[0, 0, 0].item() - 0.7) < 1e-5, wval.item()
    print("[2] ignore invariance: OK")


# ------------------------------------------------------------ 3. permutation
def test_permutation_invariance():
    matcher = HungarianMatcher(num_classes=3)
    criterion = SetCriterion(num_classes=3, num_points=256)
    target = _targets_from([torch.zeros(8, 8, dtype=torch.bool) |
                            (torch.arange(8)[:, None] >= 0) &
                            (torch.arange(8)[None, :] >= 0)], h=8, w=8)
    target[0][3:, :] = IGNORE
    targets, points, weight, valid_map = build_set_targets(target, 3, IGNORE,
                                                num_points=256, sample_mode="full")
    best = targets[0]
    shuffled = {"class_labels": best["class_labels"].flip(0),
                "mask_labels": best["mask_labels"].flip(0)}
    # two classes present: make it a genuine 2-object case
    target = torch.zeros(1, 8, 8, dtype=torch.int64)
    target[0, :4] = IGNORE
    target[0, 4:, :4] = 1
    target[0, 4:, 4:] = 2
    targets, points, weight, valid_map = build_set_targets(target, 3, IGNORE,
                                                num_points=256, sample_mode="full")
    shuffled = {"class_labels": targets[0]["class_labels"].flip(0),
                "mask_labels": targets[0]["mask_labels"].flip(0)}

    gl = torch.Generator().manual_seed(11)
    class_logits = torch.randn(1, 4, 4, generator=gl)
    mask_logits = torch.randn(1, 4, 8, 8, generator=gl)
    loss_a = criterion.loss_labels(class_logits, targets,
                                   matcher(class_logits, mask_logits, targets,
                                           points, weight, valid_map))
    loss_b = criterion.loss_labels(class_logits, [shuffled],
                                   matcher(class_logits, mask_logits, [shuffled],
                                           points, weight, valid_map))
    assert torch.allclose(loss_a["class"], loss_b["class"], atol=1e-6)
    print("[3] permutation invariance: OK")


# ------------------------------------------------------------ 4. all-masked
def test_all_masked_attention_is_finite():
    """An all-blocked attention row must not produce NaN, and the guard must be
    what prevents it.

    MEASURED on torch 2.13.0: `nn.MultiheadAttention` returns finite values even
    for a fully blocked row, so `guard_effect` is False there and the guard is a
    cross-version defence, not a demonstrated necessity. The assertion is written
    both ways: finite is required always; if a future stack does produce NaN
    without the guard, `guard_effect` becomes True and the guard is proven to be
    load-bearing. Do not read this test as proof the NaN failure was reproduced.
    """
    def _blocked_prediction(self, queries, mask_features):
        return torch.full((mask_features.shape[0], self.num_queries,
                           mask_features.shape[-2], mask_features.shape[-1]), -30.0)

    head = _tiny_head()
    head.train()
    head.predict_masks = _blocked_prediction.__get__(head, type(head))
    head.nan_guard = False
    out_raw = head(_features())
    raw_finite = bool(torch.isfinite(out_raw["mask_logits"]).all()
                      and torch.isfinite(out_raw["class_logits"]).all())
    head.nan_guard = True
    out = head(_features())
    assert torch.isfinite(out["class_logits"]).all()
    assert torch.isfinite(out["mask_logits"]).all()

    target = torch.zeros(1, 16, 16, dtype=torch.int64)
    targets, points, weight, valid_map = build_set_targets(target, 3, IGNORE, num_points=64)
    criterion = SetCriterion(num_classes=3, num_points=64)
    losses = criterion(out["class_logits"], out["mask_logits"], targets, points,
                       weight, out["aux_outputs"], valid_map)
    total = criterion.total(losses)
    assert torch.isfinite(total).all()
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters()
               if p.grad is not None)
    print(f"[4] all-masked attention: OK (finite always; NaN without guard="
          f"{not raw_finite}, guard_effect={not raw_finite}, loss={float(total.detach()):.4f})")


# --------------------------------------------------------------- 5. semantic
def test_semantic_synthesis_matches_hand_computation():
    # 2 queries x 3 classes x 4x4 masks, computed by hand below
    class_logits = torch.tensor([[[2.0, 0.0, 0.0, -1.0],
                                  [0.0, 0.0, 2.0, -1.0]]])       # (1,2,4)
    mask_logits = torch.zeros(1, 2, 4, 4)
    mask_logits[0, 0] = 4.0
    mask_logits[0, 1] = -4.0
    scores = semantic_scores(class_logits, mask_logits)
    prob = class_logits.softmax(-1)[..., :-1]
    sig = mask_logits.sigmoid()
    manual = torch.zeros(1, 3, 4, 4)
    for q in range(2):
        for c in range(3):
            manual[0, c] += prob[0, q, c] * sig[0, q]
    assert torch.allclose(scores, manual, atol=1e-6)
    assert (scores >= 0).all() and (scores <= 1 + 1e-6).all()
    # the mixture is NOT class-normalised: it is a sum of Q independent products,
    # which is precisely why the historical CHAMPION bias cannot be reused here
    assert not torch.allclose(scores.sum(1), torch.ones_like(scores[:, 0]), atol=1e-3)
    pred = predict_masks_official_style(class_logits, mask_logits)
    assert pred.min() >= 1 and pred.max() <= 3
    assert int(pred[0, 0, 0]) == 1                      # query 0 wins, class 0
    assert int(pred[0, 0, 0]) != 4                      # no-object never emitted
    # query permutation must not change the mixture
    perm_scores = semantic_scores(class_logits.flip(1), mask_logits.flip(1))
    assert torch.allclose(scores, perm_scores, atol=1e-6)
    print("[5] semantic synthesis (hand-checked): OK")


# ------------------------------------------------------------------ 6. empty
def test_empty_targets_and_all_ignore_images():
    criterion = SetCriterion(num_classes=3, num_points=64)
    matcher = HungarianMatcher(num_classes=3)
    gl = torch.Generator().manual_seed(5)
    class_logits = torch.randn(1, 4, 4, generator=gl)
    mask_logits = torch.randn(1, 4, 8, 8, generator=gl)
    # (a) no class present at all
    target = torch.full((1, 8, 8), IGNORE, dtype=torch.int64)
    try:
        build_set_targets(target, 3, IGNORE, num_points=64)
        raise AssertionError("all-ignore image must be rejected explicitly")
    except ValueError as exc:
        assert "valid pixels" in str(exc)
    # (b) classes absent, but some valid background pixels
    target = torch.full((1, 8, 8), IGNORE, dtype=torch.int64)
    target[0, :4] = 0
    targets, points, weight, valid_map = build_set_targets(target, 3, IGNORE, num_points=64)
    assert targets[0]["class_labels"].numel() == 1
    indices = matcher(class_logits, mask_logits, targets, points, weight, valid_map)
    losses = criterion(class_logits, mask_logits, targets, points, weight, (), valid_map)
    for key, value in losses.items():
        assert torch.isfinite(value).all(), key
    print("[6] empty-class / all-ignore handling: OK")


def main() -> int:
    torch.manual_seed(0)
    test_perfect_prediction_is_floor()
    test_ignore_invariance()
    test_permutation_invariance()
    test_all_masked_attention_is_finite()
    test_semantic_synthesis_matches_hand_computation()
    test_empty_targets_and_all_ignore_images()
    print("ALL_CPU_INVARIANTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
