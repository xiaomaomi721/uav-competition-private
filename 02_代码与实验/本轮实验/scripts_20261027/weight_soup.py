#!/usr/bin/env python3
"""权重平均（合规：训练后同架构 checkpoint 参数平均成单一权重）。
候选来源：H75 起点系的所有 source-only 续训收敛 checkpoint。
"""
import argparse, json, sys
from pathlib import Path
import torch

R = Path("/home/jhy/aic2026_h76_20261026")
TRAIN_PREFIX = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."


def is_avg(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)


def load(p):
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    return d.get("model", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base", required=True, help="非平均参数从这里复制（通常是第一个/H75）")
    a = ap.parse_args()

    states = [load(p) for p in a.ckpts]
    ws = a.weights or [1.0] * len(states)
    assert len(ws) == len(states)
    wsum = sum(ws)

    ref = states[0]
    keys = set(ref.keys())
    for s in states:
        assert set(s.keys()) == keys, "key set mismatch"

    out = {}
    for k in keys:
        if is_avg(k):
            acc = None
            for s, w in zip(states, ws):
                t = s[k].float() * (w / wsum)
                acc = t if acc is None else acc + t
            out[k] = acc.to(ref[k].dtype)
        else:
            out[k] = ref[k].clone()

    outp = Path(a.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out, "soup": {
        "ckpts": a.ckpts, "weights": ws,
        "avg_keys": sum(1 for k in keys if is_avg(k)),
        "total_keys": len(keys),
    }}, outp)
    print(f"saved {outp}")
    print(f"averaged {sum(1 for k in keys if is_avg(k))} / {len(keys)} tensors")


if __name__ == "__main__":
    main()
