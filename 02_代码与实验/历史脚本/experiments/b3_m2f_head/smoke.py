"""CPU smoke test: forward / loss / backward on a tiny synthetic batch."""
from __future__ import annotations

import torch

from .head import B3M2FStyleHead
from .loss import SetCriterion, build_set_targets


def main() -> int:
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    head = B3M2FStyleHead(num_queries=8, hidden_dim=32, nhead=4, num_layers=2,
                          dim_feedforward=64)
    head.train()
    features = [torch.randn(1, c, h, w) for c, h, w in
                [(64, 16, 16), (128, 8, 8), (320, 4, 4), (512, 2, 2)]]
    out = head(features)
    assert out["class_logits"].shape == (1, 8, 9), out["class_logits"].shape
    assert out["mask_logits"].shape == (1, 8, 16, 16), out["mask_logits"].shape
    assert len(out["aux_outputs"]) == 1
    for key in ("class_logits", "mask_logits"):
        assert torch.isfinite(out[key]).all(), key

    target = torch.randint(0, 8, (1, 16, 16), dtype=torch.int64)
    targets, points, weight, valid_map = build_set_targets(target, num_classes=8,
                                                num_points=64, sample_mode="full")
    criterion = SetCriterion(num_classes=8, num_points=64)
    losses = criterion(out["class_logits"], out["mask_logits"], targets, points,
                       weight, out["aux_outputs"], valid_map)
    for key, value in losses.items():
        assert torch.isfinite(value).all(), key
    total = criterion.total(losses)
    total.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    print(f"SMOKE_OK total={float(total):.4f} "
          f"losses={ {k: round(float(v), 4) for k, v in losses.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
