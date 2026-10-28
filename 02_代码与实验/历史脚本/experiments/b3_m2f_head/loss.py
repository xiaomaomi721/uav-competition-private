"""Ignore-aware set-prediction losses for B3 + Mask2Former-style head v0.

Ignores (`ignore_index`, i.e. official id 0) are excluded from four places and
none of them is optional:

  1. set targets        -- no object is invented from ignore pixels;
  2. matching cost      -- point-sampled mask cost sees only valid points;
  3. mask BCE / Dice    -- numerators *and* denominators drop ignored points,
                           including points whose interpolation footprint
                           touches an ignored pixel (valid weight < 1);
  4. class target       -- recorded only for objects that are not entirely ignore.

Everything is plain torch on CPU. `valid_weight_map` is assumed precomputed by
the caller from the target mask via `valid_weight_map_from_target`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EPS = 1e-8

# Single source of truth for every point-sampling call. The ignore-aware
# renormalisation interp(mask*valid)/interp(valid) is only exact while the
# numerator and the denominator go through THIS kernel: same mode, same padding,
# same align_corners. Nothing in this module may call grid_sample directly.
SAMPLING = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}


def valid_weight_map_from_target(target: torch.Tensor,
                                 ignore_index: int = 255) -> torch.Tensor:
    """(B,h,w) per-position validity map (1 = valid, 0 = ignored), same resolution.

    Two uses, with different precision:

      * full-map BCE/Dice: multiplied into the loss and its denominator, so
        positions flagged ignored contribute nothing;
      * point sampling: `weight_at_points` bilinearly samples this map, so a
        point whose interpolation footprint touches an ignored pixel gets a
        weight below 1 and only its non-ignored share survives. That is an
        approximation of the exact `grid_sample` weights, which is why every
        mask term is normalised by retained mass rather than a pixel count.
    """
    return (target != ignore_index).to(torch.float32)


def sample_map(maps: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear sample (B,C,h,w) at normalized (B,P,2) points -> (B,P,C)."""
    b, c, _, _ = maps.shape
    grid = points[:, :, None, :] * 2 - 1                      # (B,P,1,2)
    out = F.grid_sample(maps, grid, **SAMPLING)
    return out[:, :, :, 0].permute(0, 2, 1)


def sample_points_using_uncertainty(logits: torch.Tensor, points: torch.Tensor,
                                    oversample: int = 3,
                                    importance: float = 0.0) -> torch.Tensor:
    """PointRend-style importance sampling over the valid points."""
    b, p = points.shape[0], points.shape[1]
    sampled = sample_map(logits, points)
    probs = sampled.sigmoid().flatten(1)
    uncertainty = -torch.where(probs > 0.5, 1 - probs, probs)
    n = min(int(importance * p), p)
    if n > 0:
        chosen = uncertainty.topk(n, dim=1).indices
        points = torch.gather(points, 1, chosen[:, :, None].repeat(1, 1, 2))
    return points


def sample_point(maps: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Same as sample_map but keeps (B,C,P) like the reference implementation."""
    return sample_map(maps, points).permute(0, 2, 1)


def weight_at_points(weight_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """(B,h,w) -> (B,1,P): how much non-ignored mass each sampled point carries."""
    return sample_map(weight_map.unsqueeze(1), points).permute(0, 2, 1)


def random_points(batch: int, count: int, mode: str = "full", ratio: float = 0.25,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """(B,P,2) in [0,1]. `mode` mirrors `Mask2FormerLoss.point_sample`: full,
    full+random, full+half, or none (pure random)."""
    full = int((1 - ratio) * count)
    if mode == "none":
        return torch.rand((batch, count, 2), generator=generator)
    grid = int(count ** 0.5)
    y, x = torch.meshgrid(torch.linspace(0, 1, grid), torch.linspace(0, 1, grid),
                          indexing="ij")
    pts = torch.stack([x, y], dim=-1).reshape(1, -1, 2).repeat(batch, 1, 1)
    if mode == "full":
        return pts
    if mode == "full+random":
        extra = torch.rand((batch, count - full, 2), generator=generator)
        return torch.cat([pts, extra], dim=1)
    if mode == "full+half":
        extra = torch.rand((batch, count - full, 2), generator=generator) * 0.5 + 0.25
        return torch.cat([pts, extra], dim=1)
    raise ValueError(f"unknown point_sample mode: {mode}")


def build_set_targets(target: torch.Tensor, num_classes: int,
                      ignore_index: int = 255, sample_mode: str = "full",
                      num_points: int = 12544, oversample: int = 3,
                      importance: float = 0.0,
                      generator: torch.Generator | None = None):
    """Semantic target (B,H,W) with ignore -> per-image class labels and masks.

    One target per semantic class present in the image (not per connected
    component), matching the official semantic-segmentation formulation.

    Returns `targets` (list of dicts with valid class_labels/mask_labels only),
    `point_coordinates` (B,P,2) in [0,1], `point_weight` (B,1,P) and
    `valid_map` (B,h,w) for ignore-aware target renormalisation.
    """
    if target.dim() == 4:
        target = target.squeeze(1)
    b, h, w = target.shape
    if (target != ignore_index).sum() == 0:
        raise ValueError("image has no valid pixels")
    valid_map = valid_weight_map_from_target(target, ignore_index)
    points = random_points(b, num_points, sample_mode, generator=generator)
    weight = weight_at_points(valid_map, points)
    targets = []
    for i in range(b):
        img = target[i]
        classes, labels, masks = [], [], []
        for c in range(num_classes):
            m = img == c
            if not bool(m.any()):
                continue
            classes.append(c)
            masks.append(m.to(torch.float32))
        targets.append({
            "class_labels": torch.tensor(classes, dtype=torch.int64),
            "mask_labels": torch.stack(masks) if masks else torch.zeros((0, h, w)),
        })
    return targets, points, weight, valid_map


def sigmoid_ce_sum(logits: torch.Tensor, labels: torch.Tensor,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
    """Summed BCE over all elements, with ignored mass dropped.

    The reference criterion sums BCE over every point of every matched mask and
    divides only by the number of target masks, so a mean here would differ by a
    factor of the point count. Weighted points simply contribute their weight;
    dropping a zero-weight point therefore cannot change the sum.
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if weight is None:
        return loss.sum()
    w = weight
    while w.dim() < loss.dim():
        w = w.unsqueeze(0) if w.dim() == 1 else w.unsqueeze(1)
    if w.shape[-2:] != loss.shape[-2:]:
        raise ValueError(f"weight {tuple(w.shape[-2:])} does not match "
                         f"loss {tuple(loss.shape[-2:])}")
    return (loss * w).sum()


def sigmoid_ce(logits: torch.Tensor, labels: torch.Tensor,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE with ignored mass removed from numerator and denominator.

    `logits`/`labels` are either full-resolution maps (Nq,h,w)/(M,h,w) or
    already-sampled (Nq,P)/(M,P). The weight carries only B plus the trailing
    spatial/point axis and is broadcast over the leading dimension.
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if weight is None:
        return loss.mean()
    w = weight
    while w.dim() < loss.dim():
        w = w.unsqueeze(1)                            # (B,S) -> (B,1,S); (B,h,w) -> (B,1,h,w)
    w = w.reshape(w.shape[0], 1, *w.shape[2:])
    if w.shape[0] != loss.shape[0]:
        raise ValueError("weight/loss batch mismatch")
    if w.shape[2:] != loss.shape[2:]:
        raise ValueError(f"weight {tuple(w.shape[2:])} does not match "
                         f"loss {tuple(loss.shape[2:])}")
    return (loss * w).sum() / w.expand_as(loss).sum().clamp_min(EPS)


def dice_sum(inputs: torch.Tensor, labels: torch.Tensor,
             weight: torch.Tensor | None = None) -> torch.Tensor:
    """Summed dice over matched masks, with ignored mass removed.

    Weighted dice stops being a ratio of sums once the weight varies per point,
    so this uses normalised weighted means: sum(w*p*t)/sum(w) and sum(w*p)/sum(w).
    That is exact, reduces to the reference expression when w is constant, and
    keeps each mask's term bounded in [0, 1] independently of the point count.
    """
    prob = inputs.sigmoid()
    tgt = labels
    if weight is None:
        prob = prob.flatten(1)
        tgt = tgt.flatten(1)
        num = 2 * (prob * tgt).sum(-1)
        den = prob.sum(-1) + tgt.sum(-1)
        return (1 - (num + 1.0) / (den + 1.0)).sum()
    w = weight
    while w.dim() < 3:
        w = w.unsqueeze(0)
    if prob.dim() == 4:
        prob = prob.flatten(2)
        tgt = tgt.flatten(2)
        w = w.flatten(2)
    if w.shape[-1] != prob.shape[-1]:
        raise ValueError("weight point count does not match dice inputs")
    wsum = w.sum(-1).clamp_min(EPS)
    num = 2 * ((prob * tgt * w).sum(-1) / wsum)
    den = ((prob * w).sum(-1) + (tgt * w).sum(-1)) / wsum
    return (1 - (num + 1.0) / (den + 1.0)).sum()


def dice_sum_reference(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Verbatim reproduction of the installed version's `dice_loss` arithmetic.

    Transcribed without correction on purpose. Two defects compound when the
    reference feeds its `(N,1,P)` labels into a function documented for `(N,P)`:

      * `probs * labels` broadcasts to an (N,N,P) cross-mask tensor, so pairs
        other than the diagonal enter the sum;
      * `probs.sum(-1) + labels.sum(-1)` broadcasts (N,) with (N,1), giving
        `denominator[i, j] = probs_sum[j] + labels_sum[i]` -- the two indices come
        from different masks.

    The diagonal entry (i,i) is still the true per-pair dice, which is what the
    documented semantics compute; everything off-diagonal is spurious. Kept only
    so parity can be asserted bit-for-bit against the installed code. NOT used in
    training (see SetCriterion.literal_reference_dice).
    """
    probs = inputs.sigmoid().flatten(1)                  # (N,P)
    numerator = 2 * (probs * labels).sum(-1)             # broadcast -> (N,N)
    denominator = probs.sum(-1) + labels.sum(-1)         # broadcast -> (N,N)
    return (1 - (numerator + 1) / (denominator + 1)).sum()

def dice_loss(inputs: torch.Tensor, labels: torch.Tensor,
              weight: torch.Tensor | None = None) -> torch.Tensor:
    """Dice with ignored mass removed from numerator and denominator.

    `weight=None` reduces exactly to the reference `dice_loss` up to its
    /num_masks normalisation, which the criterion applies separately: the
    reference's loss.sum()/num_masks equals the mean over matched masks when each
    is weighted 1/num_masks.
    """
    prob = inputs.sigmoid()
    tgt = labels
    if weight is not None:
        w = weight
        if prob.dim() == 4:                                   # map space (Nq,h,w)
            while w.dim() < 4:
                w = w.unsqueeze(1)
            if w.shape[-2:] != prob.shape[-2:]:
                raise ValueError("weight map does not match dice input map")
        else:                                                 # sampled (Nq,P)
            while w.dim() < 3:
                w = w.unsqueeze(0)
            if w.shape[-1] != prob.shape[-1]:
                raise ValueError("weight point count does not match dice inputs")
        weight = w
    if prob.dim() == 4:
        prob = prob.flatten(2)
        tgt = tgt.flatten(2)
        if weight is not None:
            weight = weight.flatten(2)
    if weight is None:
        num = 2 * (prob * tgt).sum(-1)
        den = prob.sum(-1) + tgt.sum(-1)
    else:
        num = 2 * (prob * tgt * weight).sum(-1)
        den = (prob * weight).sum(-1) + (tgt * weight).sum(-1)
    return (1 - (num + 1.0) / (den + 1.0)).mean()


def pair_wise_dice(inputs: torch.Tensor, labels: torch.Tensor,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
    """(Nq,M) dice cost used by the matcher; 1 - dice.

    `inputs` (Nq,P) and `labels` (M,P) are already-sampled point values.
    """
    prob = inputs.sigmoid()
    tgt = labels
    if weight is None:
        num = 2 * torch.einsum("np,mp->nm", prob, tgt)
        den = prob.sum(-1)[:, None] + tgt.sum(-1)[None, :]
        return 1 - (num + 1.0) / (den + 1.0)
    w = weight.flatten(1)                                    # (1,P)
    num = 2 * torch.einsum("np,mp,p->nm", prob, tgt, w[0])
    den = (prob * w).sum(-1)[:, None] + (tgt * w).sum(-1)[None, :]
    return 1 - (num + 1.0) / (den + 1.0)


def pair_wise_sigmoid_ce(inputs: torch.Tensor, labels: torch.Tensor,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
    """(Nq,M) mask BCE cost used by the matcher.

    `inputs` (Nq,P) and `labels` (M,P) are already-sampled point values.
    """
    nq, p = inputs.shape
    prob = inputs
    tgt = labels
    loss = F.binary_cross_entropy_with_logits(
        prob[:, None].expand(-1, tgt.shape[0], -1),
        tgt[None].expand(nq, -1, -1), reduction="none")      # (Nq,M,P)
    if weight is None:
        return loss.mean(-1)
    w = weight.flatten(1)                                    # (1,P)
    return (loss * w).sum(-1) / w.sum(-1).clamp_min(EPS)


def sample_target_masks(mask_maps: torch.Tensor, valid_map: torch.Tensor,
                        points: torch.Tensor, eps: float = EPS):
    """Ignore-aware sampling of target masks.

    A bilinear sample of a target mask next to an ignored pixel mixes real class
    labels with the ignored (label-0) neighbour, biasing targets downward. Since
    ignore pixels are 0 in every class mask but not necessarily in the valid map,
    dividing by the sampled valid fraction recovers the mean label *among valid
    neighbours*; where nothing valid is in the footprint the weight is 0 and the
    point drops out of every normalised term.

    Returns (sampled_targets (B,M,P), weight (B,1,P)).
    """
    raw = sample_map(mask_maps, points)                        # (B,P,M)
    w = sample_map(valid_map.unsqueeze(1), points)             # (B,P,1)
    renorm = torch.where(w > eps, raw / w.clamp_min(eps), torch.zeros_like(raw))
    return (renorm.clamp(0.0, 1.0).permute(0, 2, 1),
            w.permute(0, 2, 1))


class HungarianMatcher(nn.Module):
    """Cost = class CE + mask BCE + mask Dice, computed in sampled point space.

    The matcher never sees `ignore_index` directly: it receives already-sampled,
    already-weighted points, so ignored mass cannot create or destroy a match.
    """

    def __init__(self, cost_class: float = 2.0, cost_mask: float = 5.0,
                 cost_dice: float = 5.0, num_classes: int = 8) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, class_logits, mask_logits, targets, points, point_weight,
                valid_map=None):
        b, q = class_logits.shape[:2]
        indices = []
        for i in range(b):
            out_prob = class_logits[i].softmax(-1)                  # over C+1, as the reference does
            classes = targets[i]["class_labels"]
            if classes.numel() == 0:
                indices.append((torch.zeros(0, dtype=torch.int64),
                                torch.zeros(0, dtype=torch.int64)))
                continue
            tgt_masks = targets[i]["mask_labels"].unsqueeze(0)      # (1,M,h,w)
            out_masks = mask_logits[i].unsqueeze(0)                 # (1,Q,h,w)
            if valid_map is None:
                sampled_tgt = sample_point(tgt_masks, points[i:i + 1])[0]
                w = point_weight[i:i + 1]
            else:
                sampled_tgt, w = sample_target_masks(
                    tgt_masks, valid_map[i:i + 1], points[i:i + 1])
                sampled_tgt = sampled_tgt[0]
            sampled_out = sample_point(out_masks, points[i:i + 1])[0]  # (Q,P)
            cost_class = -out_prob[:, classes]
            cost_mask = pair_wise_sigmoid_ce(sampled_out, sampled_tgt, w)
            cost_dice = pair_wise_dice(sampled_out, sampled_tgt, w)
            cost = (self.cost_class * cost_class + self.cost_mask * cost_mask
                    + self.cost_dice * cost_dice)
            cost = torch.nan_to_num(cost, nan=1e9, posinf=1e9, neginf=1e9)
            row, col = linear_sum_assignment(cost.cpu())
            indices.append((torch.as_tensor(row, dtype=torch.int64),
                            torch.as_tensor(col, dtype=torch.int64)))
        return indices


class SetCriterion(nn.Module):
    """Class CE (with no-object) + mask BCE + mask Dice over sampled points."""

    def __init__(self, num_classes: int = 8, weight_dict=None,
                 matcher: HungarianMatcher | None = None,
                 eos_coef: float = 0.1, num_points: int = 12544,
                 oversample: int = 3, importance: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {"class": 2.0, "mask": 5.0, "dice": 5.0}
        self.matcher = matcher or HungarianMatcher(num_classes=num_classes)
        self.eos_coef = eos_coef
        self.num_points = num_points
        self.oversample = oversample
        self.importance = importance
        # False (default): per-pair dice, as the reference documents. True: the
        # broadcast cross-mask arithmetic the installed reference actually runs.
        # Parity tests exercise both; training uses per-pair dice.
        self.literal_reference_dice = False
        empty = torch.ones(num_classes + 1)
        empty[-1] = eos_coef
        self.register_buffer("empty_weight", empty)

    @torch.no_grad()
    def _sample_points(self, mask_logits, points, point_weight, keep_ratio=0.25):
        """Keep points that carry valid mass; top up by resampling the kept pool."""
        b, _, p = point_weight.shape
        keep = point_weight[:, 0] > 0
        out_pts, out_w = [], []
        for i in range(b):
            pts, w = points[i][keep[i]], point_weight[i, 0][keep[i]]
            if pts.shape[0] == 0:
                # Entire image is ignore-free mass absent; fall back to raw points
                # with zero weight so every masked term degenerates to 0/0 -> 0.
                out_pts.append(points[i])
                out_w.append(torch.zeros(p, dtype=points.dtype))
                continue
            n_keep = min(int(keep_ratio * p), pts.shape[0])
            idx = torch.randperm(pts.shape[0])[:n_keep]
            extra = idx[torch.randint(n_keep, (p - n_keep,))] if p > n_keep else idx[:0]
            sel = torch.cat([torch.arange(n_keep), extra]) if p > n_keep else idx
            out_pts.append(pts[sel])
            out_w.append(w[sel])
        return torch.stack(out_pts), torch.stack(out_w)[:, None]

    def forward(self, class_logits, mask_logits, targets, points, point_weight,
                aux_outputs=(), valid_map=None):
        """Reference-compatible path: on valid-only inputs (valid_map=None) this
        reproduces the installed Mask2Former criterion, matching included. The
        ignore-aware path is entered only when valid_map is supplied."""
        indices = self.matcher(class_logits, mask_logits, targets, points,
                               point_weight, valid_map)
        num_masks = self._num_masks(targets)
        losses = self.loss_labels(class_logits, targets, indices)
        losses.update(self.loss_masks(mask_logits, targets, indices, points,
                                      point_weight, valid_map))
        for i, aux in enumerate(aux_outputs):
            # the reference re-matches the auxiliary layer's own predictions
            aux_indices = self.matcher(aux["class_logits"], aux["mask_logits"],
                                       targets, points, point_weight, valid_map)
            aux_losses = self.loss_labels(aux["class_logits"], targets, aux_indices)
            aux_losses.update(self.loss_masks(aux["mask_logits"], targets, aux_indices,
                                              points, point_weight, valid_map))
            for key, value in aux_losses.items():
                losses[f"{key}_{i}"] = value
        return losses

    @staticmethod
    def _num_masks(targets) -> float:
        return float(max(sum(len(t["class_labels"]) for t in targets), 1))

    def loss_labels(self, class_logits, targets, indices):
        b, q = class_logits.shape[:2]
        target_classes = torch.full((b, q), self.num_classes, dtype=torch.int64,
                                    device=class_logits.device)
        for i, (src, tgt) in enumerate(indices):
            if src.numel():
                target_classes[i, src] = targets[i]["class_labels"][tgt]
        loss = F.cross_entropy(class_logits.transpose(1, 2), target_classes,
                               weight=self.empty_weight)
        return {"class": loss}

    def loss_masks(self, mask_logits, targets, indices, points, point_weight,
                   valid_map=None, keep_ratio=0.25):
        b, q = mask_logits.shape[:2]
        num_masks = self._num_masks(targets)
        if valid_map is None:
            # reference path: no point subsampling, no reweighting
            sampled_points, sampled_weight = points, point_weight
        else:
            sampled_points, sampled_weight = self._sample_points(
                mask_logits, points, point_weight, keep_ratio)
        loss_mask, loss_dice = 0.0, 0.0
        for i, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            # (M,h,w) for both sides; the reference stacks matched pairs as
            # (N,1,h,w) and repeats the SAME point set for every mask, so the
            # sampled tensors come out as (M,1,P).
            tgt_masks = targets[i]["mask_labels"][tgt]
            src_masks = mask_logits[i][src]
            n_pairs = src_masks.shape[0]
            pts = sampled_points[i:i + 1].expand(n_pairs, -1, -1)
            if valid_map is None:
                sampled_tgt = sample_point(tgt_masks[:, None], pts)   # (M,1,P)
                sampled_out = sample_point(src_masks[:, None], pts)
                # Matched pairs are SUMMED and the whole term divided by num_masks
                # once, exactly as the reference does; a per-pair mean would differ
                # from it by the point count.
                loss_mask = loss_mask + sigmoid_ce_sum(sampled_out, sampled_tgt)
                if self.literal_reference_dice:
                    loss_dice = loss_dice + dice_sum_reference(sampled_out, sampled_tgt)
                else:                                    # documented per-pair dice
                    loss_dice = loss_dice + dice_sum(sampled_out[..., 0, :],
                                                     sampled_tgt[..., 0, :])
            else:
                sampled_tgt, w = sample_target_masks(tgt_masks[:, None],
                                                     valid_map[i:i + 1].expand(n_pairs, -1, -1),
                                                     pts)
                sampled_out = sample_point(src_masks[:, None], pts)
                loss_mask = loss_mask + sigmoid_ce_sum(sampled_out, sampled_tgt, w)
                loss_dice = loss_dice + dice_sum(sampled_out, sampled_tgt, w)
        if isinstance(loss_mask, float):
            zero = mask_logits.sum() * 0.0
            loss_mask, loss_dice = zero, zero
        return {"mask": loss_mask / num_masks, "dice": loss_dice / num_masks}

    def total(self, losses) -> torch.Tensor:
        total = None
        for key, value in losses.items():
            name = key.split("_aux")[0]
            weight = self.weight_dict.get(name)
            if weight is None:
                continue
            term = value * weight
            total = term if total is None else total + term
        if total is None:
            raise ValueError("no weighted loss terms were produced")
        return total

    def precomputed_semantic_loss(self, semantic_logits, target,
                                  ignore_index: int = 255) -> torch.Tensor:
        """Optional pixel-CE on `log(semantic_scores)`, for engine compatibility.

        This is the escape hatch for reusing the frozen trainer. It is a *pixel*
        objective, not the set-prediction objective, and must be reported as
        such; `weight_dict` still governs the auxiliary set losses.
        """
        return F.cross_entropy(semantic_logits, target.long(), ignore_index=ignore_index)
