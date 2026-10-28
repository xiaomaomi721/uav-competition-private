"""CPU integration test: real MiT-B3 encoder -> our query head.

Verifies the assumptions the head hard-codes against the actual installed
checkpoint rather than against documentation:

  * four stage tensors, BCHW, channels 64/128/320/512, strides 4/8/16/32;
  * at 1024x1024 the spatial sizes are 256/128/64/32;
  * the head's first positional argument is exactly that 4-tuple, which is the
    contract the bundle's CWFA pre-hook asserts;
  * forward + matching + backward produce finite values end to end.

CPU only, no CUDA, no training, no checkpoint writes.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSemanticSegmentation

from .head import B3M2FStyleHead
from .loss import SetCriterion, build_set_targets

MODEL_DIR = "/home/jhy/models/segformer-b3-ade512"
IGNORE = 255


def _stages(model, size: int):
    # no_grad, not inference_mode: inference tensors cannot enter an autograd
    # graph, and E2 backprops through the head built on top of these stages.
    with torch.no_grad():
        outputs = model.segformer(pixel_values=torch.zeros(1, 3, size, size),
                                  output_hidden_states=True)
    stages = outputs.hidden_states
    assert stages is not None, "hidden_states requires output_hidden_states=True"
    return stages


def test_encoder_shapes():
    model = AutoModelForSemanticSegmentation.from_pretrained(
        MODEL_DIR, num_labels=8, ignore_mismatched_sizes=True, local_files_only=True)
    model.eval()
    size = 1024
    stages = _stages(model, size)
    assert len(stages) == 4, len(stages)
    channels = tuple(s.shape[1] for s in stages)
    spatial = tuple(s.shape[-1] for s in stages)
    assert channels == (64, 128, 320, 512), channels
    assert spatial == (256, 128, 64, 32), spatial
    for index, stage in enumerate(stages):
        assert stage.shape[:2] == (1, channels[index]), (index, stage.shape)
        assert stage.is_contiguous()
    print(f"[E1] real encoder at {size}: channels={channels} spatial={spatial} OK")
    return model


def test_head_on_real_stages(model):
    size = 256                                     # small crop keeps CPU time sane
    stages = _stages(model, size)
    head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                          hidden_dim=64, num_queries=16, nhead=4, num_layers=2,
                          dim_feedforward=128)
    head.train()
    out = head(stages)
    expected_hw = (size // 4, size // 4)
    assert out["class_logits"].shape == (1, 16, 9), out["class_logits"].shape
    assert out["mask_logits"].shape == (1, 16, *expected_hw), out["mask_logits"].shape
    assert torch.isfinite(out["class_logits"]).all()
    assert torch.isfinite(out["mask_logits"]).all()

    target = torch.randint(0, 8, (1, size, size), dtype=torch.int64)
    target[0, :8, :] = IGNORE
    targets, points, weight, valid_map = build_set_targets(
        target, num_classes=8, num_points=512, sample_mode="full")
    criterion = SetCriterion(num_classes=8, num_points=512)
    losses = criterion(out["class_logits"], out["mask_logits"], targets, points,
                       weight, out["aux_outputs"], valid_map)
    total = criterion.total(losses)
    assert torch.isfinite(total).all()
    total.backward()
    encoder_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert not encoder_grads, "encoder must not receive gradient in this detached test"
    head_grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert head_grads and all(torch.isfinite(g).all() for g in head_grads)
    sem = head.semantic_scores(out["class_logits"], out["mask_logits"],
                               upsample_to=(size, size))
    assert sem.shape == (1, 8, size, size)
    assert torch.isfinite(sem).all() and (sem >= 0).all()
    print(f"[E2] head on real stages at {size}: logits {tuple(out['mask_logits'].shape)}, "
          f"semantic {tuple(sem.shape)}, total={float(total.detach()):.4f} OK")


def test_cwfa_arg_contract(model):
    """The bundle's CWFA hook calls decode_head(*args) with args[0] a 4-tuple."""
    captured = {}

    def hook(module, args, kwargs):
        captured["n"] = len(args)
        captured["tuple"] = isinstance(args[0], (tuple, list))
        captured["len"] = len(args[0]) if captured["tuple"] else None
        captured["dims"] = [t.dim() for t in args[0]] if captured["tuple"] else None
        return None

    size = 128
    stages = _stages(model, size)
    head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                          hidden_dim=32, num_queries=4, nhead=2, num_layers=1,
                          dim_feedforward=32)
    handle = head.register_forward_pre_hook(hook, with_kwargs=True)
    head(stages)
    handle.remove()
    assert captured["n"] == 1, captured
    assert captured["tuple"] and captured["len"] == 4, captured
    assert captured["dims"] == [4, 4, 4, 4], captured
    print(f"[E3] CWFA arg contract: args[0] is a 4-tuple of 4-D tensors OK {captured}")


def main() -> int:
    torch.manual_seed(0)
    model = test_encoder_shapes()
    test_head_on_real_stages(model)
    test_cwfa_arg_contract(model)
    print("ENCODER_INTEGRATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
