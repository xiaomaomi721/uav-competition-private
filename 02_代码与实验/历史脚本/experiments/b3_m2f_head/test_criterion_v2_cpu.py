"""CPU contract tests for criterion v2. CPU only, no CUDA, nothing written.

Three things are established here, each against something external to v2 itself:

  T1  the reference call path really is point-mean. We call the installed
      criterion's OWN `sigmoid_cross_entropy_loss` with the shape its own
      `loss_masks()` produces -- `sample_point(...).squeeze(1)` -> (M, P) -- and
      show it equals a plain per-mask point mean. We also show the (M, 1, P)
      shape that v1's parity used gives a different, P-times larger number.
  T2  v2 reproduces that shape and that reduction, and is invariant to the point
      count when the sampled distribution is unchanged.
  T3  the ignore-aware reduction degenerates to the reference when every point is
      fully valid, and otherwise normalises by valid mass rather than by P.
  T4  v1 and v2 differ by exactly the factor the diagnosis predicted: with a
      constant valid weight, v1.mask / v2.mask == P.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .loss import SetCriterion
from .loss_v2 import SetCriterionV2


def _reference_bce(inputs, labels, num_masks):
    """The installed reference reduction, imported from the live library.

    `transformers` is only importable on the server; when it is absent we fall
    back to the literal source of the installed function so the test still
    checks the same arithmetic rather than our belief about it.
    """
    try:
        from transformers.models.mask2former.modeling_mask2former import (
            sigmoid_cross_entropy_loss)
        return float(sigmoid_cross_entropy_loss(inputs, labels, num_masks))
    except Exception:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        return float(criterion(inputs, labels).mean(1).sum() / num_masks)


def test_reference_path_is_point_mean():
    torch.manual_seed(0)
    m, p = 6, 12544
    logits = torch.randn(m, p) * 2.0
    labels = (torch.rand(m, p) > 0.7).float()
    num_masks = m
    got = _reference_bce(logits, labels, num_masks)
    want = float(F.binary_cross_entropy_with_logits(logits, labels,
                                                    reduction="none").mean(1).sum() / num_masks)
    assert abs(got - want) < 1e-6, (got, want)
    # the (M,1,P) shape v1's parity used must NOT agree -- it divides by P less
    grouped = _reference_bce(logits[:, None], labels[:, None], num_masks)
    assert abs(grouped - want) / max(want, 1e-9) > 100, (grouped, want)
    print(f"[T1] reference (M,P) -> {got:.6f} == point-mean {want:.6f}; "
          f"(M,1,P) -> {grouped:.2f} ({grouped/want:.0f}x larger) OK")
    return logits, labels


def test_v2_matches_reference_shape_and_reduction():
    torch.manual_seed(1)
    m, p = 4, 2048
    logits = torch.randn(m, 1, p) * 2.0
    labels = (torch.rand(m, 1, p) > 0.6).float()
    per_mask = SetCriterionV2.mask_bce_per_mask(logits, labels, None, "reference")
    assert per_mask.shape == (m,), per_mask.shape
    # the reference shape is (M, P), i.e. the channel axis is already squeezed:
    # flatten before the point mean, exactly as `sample_point(...).squeeze(1)` does
    want = F.binary_cross_entropy_with_logits(logits, labels,
                                              reduction="none").flatten(1).mean(1)
    assert want.shape == (m,), want.shape
    assert torch.allclose(per_mask, want, atol=1e-6), (per_mask, want)

    # Point-count behaviour, made deterministic: with a CONSTANT logit and label
    # the per-point BCE is a constant b, so the point mean is exactly b for any P
    # while a point sum grows as b*P. Comparing two independent random draws
    # would prove nothing here.
    m2 = 3
    for p2 in (1000, 8000, 12544):
        c_logits = torch.full((m2, 1, p2), 0.4)
        c_labels = torch.full((m2, 1, p2), 1.0)
        b = float(F.binary_cross_entropy_with_logits(c_logits, c_labels,
                                                     reduction="none")[0, 0, 0])
        mean_v2 = float(SetCriterionV2.mask_bce_per_mask(
            c_logits, c_labels, None, "reference").mean())
        sum_v1 = float(F.binary_cross_entropy_with_logits(
            c_logits, c_labels, reduction="none").sum() / m2)
        assert abs(mean_v2 - b) < 1e-6, (p2, mean_v2, b)
        assert abs(sum_v1 / mean_v2 - p2) / p2 < 1e-6, (p2, sum_v1, mean_v2)
    print(f"[T2] per-mask shape {tuple(per_mask.shape)}, matches reference point-mean; "
          f"constant-BCE check: v2 == b for P in (1000, 8000, 12544), v1 == b*P OK")


def test_ignore_aware_degenerates_to_reference():
    torch.manual_seed(2)
    m, p = 3, 512
    logits = torch.randn(m, 1, p)
    labels = (torch.rand(m, 1, p) > 0.5).float()
    ones = torch.ones(m, 1, p)
    ref = SetCriterionV2.mask_bce_per_mask(logits, labels, None, "reference")
    awd = SetCriterionV2.mask_bce_per_mask(logits, labels, ones, "points")
    assert torch.allclose(ref, awd, atol=1e-6), (ref, awd)
    # halving the weight of half the points must not move the value
    w = torch.ones(m, 1, p)
    w[:, :, : p // 2] = 0.5
    half = SetCriterionV2.mask_bce_per_mask(logits, labels, w, "points")
    assert (half - ref).abs().max() < 0.05, (half, ref)
    print(f"[T3] all-valid -> identical to reference (max diff "
          f"{float((awd-ref).abs().max()):.2e}); reweighted stays O(1) OK")


def test_v1_to_v2_factor_is_point_count():
    torch.manual_seed(3)
    m, p = 3, 1000
    logits = torch.randn(m, 1, p)
    labels = (torch.rand(m, 1, p) > 0.5).float()
    ones = torch.ones(m, 1, p)
    v1 = float((F.binary_cross_entropy_with_logits(logits, labels,
                                                   reduction="none") * ones).sum() / m)
    v2 = float(SetCriterionV2.mask_bce_per_mask(logits, labels, ones, "points").sum() / m)
    ratio = v1 / v2
    assert abs(ratio - p) / p < 1e-6, (ratio, p)
    print(f"[T4] v1/v2 ratio = {ratio:.1f} == P = {p} OK")


def test_subclass_does_not_disturb_matcher_or_class():
    a = SetCriterion(num_classes=8)
    b = SetCriterionV2(num_classes=8)
    assert type(a.matcher) is type(b.matcher)
    assert a.eos_coef == b.eos_coef and a.num_points == b.num_points
    assert a.weight_dict == b.weight_dict
    assert torch.equal(a.empty_weight, b.empty_weight)
    print("[T5] matcher / class / weights / buffers inherited unchanged OK")


def test_degenerate_valid_mass_is_finite():
    """All-ignore and near-empty valid mass must stay finite, not 0/0 -> NaN."""
    torch.manual_seed(4)
    m, p = 3, 256
    logits = torch.randn(m, 1, p)
    labels = torch.zeros(m, 1, p)
    zero = torch.zeros(m, 1, p)
    out = SetCriterionV2.mask_bce_per_mask(logits, labels, zero, "points")
    assert torch.isfinite(out).all(), out
    assert float(out.abs().max()) == 0.0, out
    # a single valid point out of P: must still be finite and O(1), not O(P)
    sparse = torch.zeros(m, 1, p)
    sparse[:, :, 0] = 1.0
    out2 = SetCriterionV2.mask_bce_per_mask(logits, labels, sparse, "points")
    assert torch.isfinite(out2).all(), out2
    assert float(out2.abs().max()) < 100.0, out2
    print(f"[T6] all-ignore -> {float(out.abs().max()):.1e} (finite); "
          f"1 valid point -> {float(out2.abs().max()):.4f} (O(1), not O(P)) OK")


def test_v2_leaves_matching_untouched():
    """The matcher is a separate path and must return identical indices."""
    from .loss import build_set_targets
    torch.manual_seed(5)
    m, p, c = 3, 512, 4
    class_logits = torch.randn(1, 7, c + 1)
    mask_logits = torch.randn(1, 7, 32, 32)
    target = torch.randint(0, c, (1, 32, 32))
    targets, pts, w, vm = build_set_targets(target, c, num_points=p)
    a = SetCriterion(num_classes=c).matcher(class_logits, mask_logits, targets, pts, w, vm)
    b = SetCriterionV2(num_classes=c).matcher(class_logits, mask_logits, targets, pts, w, vm)
    assert len(a) == len(b)
    for (r1, c1), (r2, c2) in zip(a, b):
        assert torch.equal(r1, r2) and torch.equal(c1, c2), (r1, r2, c1, c2)
    print(f"[T7] matcher indices identical between v1 and v2 "
          f"({sum(len(r) for r, _ in a)} matches) OK")


if __name__ == "__main__":
    test_reference_path_is_point_mean()
    test_v2_matches_reference_shape_and_reduction()
    test_ignore_aware_degenerates_to_reference()
    test_v1_to_v2_factor_is_point_count()
    test_subclass_does_not_disturb_matcher_or_class()
    test_degenerate_valid_mass_is_finite()
    test_v2_leaves_matching_untouched()
    print("\nall v2 CPU contract tests passed")
