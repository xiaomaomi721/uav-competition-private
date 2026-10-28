#!/usr/bin/env python3
"""Priority schedule's genuine fixed-batch1 precision gate BEFORE any benchmark.

CPU: python -B tta_priority_gates.py selftest
GPU, only after old Phase B exits and its integrity marker passes:
  CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID python tta_priority_gates.py \
    precision --config CFG --checkpoint CKPT --val-list VAL \
    --cache-dir OLD/mean_logits_fp16 --anatomy OLD/TTA_ANATOMY.json \
    --out-dir NEW --sample FIXED16_JSON --integrity PHASE_B_OUTPUT_COMPLETE.json

This NEW adapter leaves tta_v2_gates.py and all existing reports/cache untouched.
The schedule is integrity -> THIS precision -> benchmark -> correctness -> priority
F69997/F65000 -> later C/D. This report does NOT claim benchmark/correctness PASS.
A benchmark is neither executed nor fabricated here. The batch size is explicitly
pre-registered as 1, not selected for speed. A FAIL stops this priority schedule;
calibration on the old fp16 cache is blocked pending an explicit precise strategy.

Output: immutable NEW FP16_LOGIT_CACHE_PRECISION_AUDIT.json. The full original
precision comparison schema is preserved, with protocol=prebenchmark_fixed_batch1
and benchmark_required_for_this_precision=false. D must use its explicit priority
adapter, not pretend the original benchmark-dependent acceptance rule was met.
Existing output paths are rejected; select a fresh output directory to retry.

Each artifact's payload_sha256 hashes canonical compact sorted-key UTF-8 JSON,
ensure_ascii=False/allow_nan=False, excluding only payload_sha256 itself. Referenced
input/source artifacts have full-file SHA256 bindings. The inherited writer records
gates_source_sha256 for the immutable base; priority_gates_source_sha256 identifies
THIS entrypoint. No digest is represented as a cryptographic signature.
Exit: 0 measured PASS; 2 measured FAIL; 3 BLOCKED prerequisite; 4 unexpected error.
No SSH is used. selftest neither imports torch nor initializes CUDA.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tta_v2_gates as gates

PROTOCOL = "prebenchmark_fixed_batch1"
BATCH = 1
GROUPS = ("old_cache_combined_drift", "v2_self_quantization")
OUTPUT = "FP16_LOGIT_CACHE_PRECISION_AUDIT.json"


def preflight(args):
    """Reuse base identity/device/model checks without a benchmark prerequisite.

    The base's 'benchmark' preflight branch means 'no benchmark required yet'; it
    only validates prerequisites and builds the model. We call gpu_preflight, NOT
    benchmark, with a copied namespace, leave args unchanged, and never monkeypatch
    module globals. Its final selected-batch value must be None. Our actual batch
    is the pre-registered constant BATCH=1, never a fabricated benchmark selection.
    """
    copied = SimpleNamespace(**vars(args))
    copied.command = "benchmark"
    copied.expected_checkpoint_sha256 = gates.EXPECTED_CHECKPOINT
    copied.expected_config_sha256 = gates.EXPECTED_CONFIG
    engine, model, ident, sample, marker, bindings, selected = gates.gpu_preflight(copied)
    gates.require(selected is None, "Prebenchmark preflight unexpectedly selected a batch")
    gates.require(ident["checkpoint_sha256"] == gates.EXPECTED_CHECKPOINT and
                  ident["config_sha256"] == gates.EXPECTED_CONFIG, "Priority precision must be Golden")
    gates.require([sum(s["domain"] == d for s in sample["samples"]) for d in ("D0", "D1", "D2")] == [6, 5, 5],
                  "Fixed16 domain quotas must be D0/D1/D2=6/5/5")
    inventory = marker.get("cache_inventory", {})
    gates.require(len(inventory) == gates.N_VAL and marker.get("cache_inventory_sha256") == gates.canonical_hash(inventory),
                  "Integrity inventory is not full1399 SHA-bound")
    bindings = dict(bindings)
    bindings[str(Path(gates.__file__).resolve())] = gates.sha256_file(gates.__file__)
    bindings[str(Path(__file__).resolve())] = gates.sha256_file(__file__)
    return engine, model, ident, sample, marker, bindings


def compare_image(precise, cached, gt, stem, thresholds):
    """CPU comparison, but caller supplies genuine fresh GPU float32 TTA mean."""
    gates.require(precise.dtype == np.float32 and cached.dtype == np.float32, "Comparison input dtype must be float32")
    quantized = precise.astype(np.float16).astype(np.float32)
    out = {kind: {} for kind in GROUPS}
    for kind, candidate in ((GROUPS[0], cached), (GROUPS[1], quantized)):
        for name, vec in gates.BIASES.items():
            bias = np.asarray(vec, dtype=np.float32).reshape(1, 1, 8)
            report = gates.compare_logits(precise + bias, candidate + bias, gt,
                                           thresholds["disagreement"], float("inf"))
            report["stem"] = stem
            out[kind][name] = report
    return out


def aggregate(reports, thresholds):
    out = {}
    passed = True
    for kind in GROUPS:
        out[kind] = {}
        for name in gates.BIASES:
            rows = [r[kind][name] for r in reports]
            a = sum((np.asarray(r["confusion_reference"], dtype=np.int64) for r in rows), np.zeros((8, 8), np.int64))
            b = sum((np.asarray(r["confusion_candidate"], dtype=np.int64) for r in rows), np.zeros((8, 8), np.int64))
            metrics = gates.metric_delta(a, b)
            ok = (max(r["disagreement"] for r in rows) < thresholds["disagreement"] and
                  metrics["max_iou_delta"] < thresholds["aggregate_iou_delta"] and
                  metrics["miou_delta"] is not None and metrics["miou_delta"] < thresholds["aggregate_iou_delta"] and
                  all(r["meaningful_disagreements"] == 0 for r in rows))
            passed = passed and ok
            total = sum(r["pixels"] for r in rows)
            out[kind][name] = {"status": "PASS" if ok else "FAIL", **metrics,
                "max_per_image_disagreement": max(r["disagreement"] for r in rows),
                "aggregate_pixel_disagreement": sum(r["disagreement_pixels"] for r in rows) / total,
                "mean_abs_logit_error": sum(r["mean_abs_logit_error"] * r["pixels"] for r in rows) / total,
                "max_abs_logit_error": max(r["max_abs_logit_error"] for r in rows),
                "confusion_reference": a.tolist(), "confusion_candidate": b.tolist(), "images": rows}
    return out, passed


def check_label_mapping(item, gt):
    """Attest the base official1..8->train0..7 mapping against production source."""
    from PIL import Image
    from src.label_mapping import official_to_train
    with Image.open(item["mask"]) as image:
        actual = official_to_train(np.asarray(image).copy())
    gates.require(np.array_equal(gt, actual), "Base GT conversion differs from production official_to_train")


def precision(args):
    engine, model, ident, sample, marker, bindings = preflight(args)
    import torch
    import src.label_mapping as mapping
    import src.transforms as transforms
    for module in (mapping, transforms):
        bindings[str(Path(module.__file__).resolve())] = gates.sha256_file(module.__file__)
    reports = []
    thresholds = sample["thresholds"]
    for item in sample["samples"]:
        gates.old_process_terminal(args.old_pid, args.old_start_ticks)
        gates.gpu_exclusive()
        stem = item["stem"]
        cache = args.cache_dir / f"{stem}.npy"
        expected = marker["cache_inventory"][stem]["sha256"]
        gates.require(gates.sha256_file(cache) == expected, f"Old cache changed: {stem}")
        cached16 = np.load(cache, allow_pickle=False)
        gates.validate_cache_array(cached16, stem)
        gates.require(gates.sha256_file(cache) == expected, f"Old cache changed while loading: {stem}")
        cached = cached16.astype(np.float32)
        del cached16
        px, gt = gates.image_tensor(item)
        check_label_mapping(item, gt)
        with torch.inference_mode():
            # Fixed batch1 is intentional: this audit precedes speed selection.
            views = engine.forward_views(model, px, BATCH)
            gates.require(tuple(views.shape) == (8, 8, 1024, 1024) and
                          views.dtype == torch.float32 and views.is_cuda, "V2 forward contract failed")
            precise = views.mean(dim=0, dtype=torch.float32).permute(1, 2, 0).cpu().numpy()
        reports.append(compare_image(precise, cached, gt, stem, thresholds))
        del views, px, precise, cached
        print(f"Priority fixed-batch1 precision {len(reports)}/16", flush=True)
    comparisons, passed = aggregate(reports, thresholds)
    gates.require(len(reports) == 16, "Priority precision must evaluate exactly frozen16")
    gates.gpu_exclusive()
    # Artifacts/sources are not allowed to drift during the audit either.
    for path, digest in bindings.items():
        gates.require(gates.sha256_file(path) == digest, f"Prerequisite/source changed during audit: {path}")
    report = dict(ident,
        status="PASS" if passed else "FAIL", gate="precision", protocol=PROTOCOL, precision_execution=PROTOCOL,
        priority_gates_source_sha256=gates.sha256_file(__file__), artifact_sha256=bindings,
        precision_status="FP16_LOGIT_CACHE_PRECISION_PASS" if passed else "FP16_LOGIT_CACHE_PRECISION_FAILED",
        calibration_allowed=passed, requires_precise_strategy=not passed,
        priority_schedule_may_advance_to_benchmark=passed,
        phase_c_allowed_by_this_gate=False, later_benchmark_and_correctness_required=True,
        failure_action=None if passed else "BLOCKED_PRECISE_CACHE_REQUIRED: stop priority schedule; do not calibrate old fp16 cache; approve a precise strategy",
        n=16, sample_stems=[s["stem"] for s in sample["samples"]], view_batch_size=BATCH,
        batch_selection="fixed_preregistered_not_benchmarked", benchmark_required_for_this_precision=False,
        benchmark_evaluated=False, correctness_evaluated=False, backend=gates.backend_snapshot(),
        thresholds=thresholds, bias_vectors=gates.BIASES, comparisons=comparisons,
        reference="Fresh GPU float32 D4 TTA8 mean BEFORE any quantization; fixed view_batch_size=1",
        interpretation={GROUPS[0]: "Fresh fixedbatch1 v2 vs existing old fp16 cache; reduction drift AND quantization",
                        GROUPS[1]: "Fresh v2 float32 vs its own float16 round trip; isolates quantization"},
        old_cache_modified=False, dataset_recomputation=False)
    return gates.write_artifact(args.out_dir / OUTPUT, report)


def selftest():
    import tempfile
    from PIL import Image
    # Near-tie uses a pre-quantization reference; fp16 changes exactly one pixel.
    thresholds = {"disagreement": 1e-4, "aggregate_iou_delta": 1e-4, "logit_error": 1e-3}
    ref = np.zeros((1, 20000, 8), dtype=np.float32)
    ref[..., 0] = 1
    ref[0, 0, 1] = np.nextafter(np.float32(1), np.float32(2))
    gt = np.zeros((1, 20000), dtype=np.uint8)
    near = compare_image(ref, ref.astype(np.float16).astype(np.float32), gt, "tie", thresholds)
    row = near[GROUPS[0]]["no_bias"]
    assert row["disagreement_pixels"] == 1 and row["near_tie_disagreements"] == 1
    assert row["max_abs_logit_error"] > 0, "Reference must not already be fp16"
    # All three bias variants must exist; decisive changes must fail closed.
    strong = np.zeros((2, 2, 8), np.float32)
    strong[..., 0] = 10
    gt_small = np.zeros((2, 2), np.uint8)
    same = compare_image(strong, strong.copy(), gt_small, "same", thresholds)
    _, passed = aggregate([same] * 16, thresholds)
    assert passed and all(set(same[k]) == set(gates.BIASES) for k in GROUPS)
    changed = strong.copy()
    changed[..., 7] = 100
    bad = compare_image(strong, changed, gt_small, "bad", thresholds)
    _, passed = aggregate([bad] * 16, thresholds)
    assert not passed
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)
        labels = np.zeros((1024, 1024), np.uint8)
        labels[0, :10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 255]
        Image.fromarray(labels).save(p / "mask.png")
        mapped = gates.read_gt(p / "mask.png")
        assert mapped[0, :10].tolist() == [255, 0, 1, 2, 3, 4, 5, 6, 7, 255]
        marker = p / "receipt.json"
        gates.write_artifact(marker, {"status": "PASS", "protocol": PROTOCOL, "view_batch_size": 1})
        assert gates.load_artifact(marker)["protocol"] == PROTOCOL
        before = gates.sha256_file(marker)
        content = json.loads(marker.read_text())
        content["view_batch_size"] = 8
        marker.write_text(json.dumps(content))
        assert gates.sha256_file(marker) != before
        try:
            gates.load_artifact(marker)
        except gates.Blocked:
            pass
        else:
            raise AssertionError("Tampered identity accepted")
    args = parser().parse_args(["precision", *sum(([f"--{f}", "fixture"] for f in
        ("config", "checkpoint", "val-list", "cache-dir", "anatomy", "out-dir", "sample", "integrity")), [])])
    assert args.command == "precision" and not hasattr(args, "benchmark")
    assert "torch" not in sys.modules, "CPU selftest must not import torch"
    print("SELF_TEST_OK: fixedbatch1 prebenchmark, genuine fp32 near tie, three biases, hard drift, label mapping, SHA tamper rejection; CPU only")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="CPU-only numerical, mapping and identity tests")
    p = sub.add_parser("precision", help="Genuine fixedbatch1 audit before any benchmark")
    for name in ("config", "checkpoint", "val-list", "cache-dir", "anatomy", "out-dir", "sample", "integrity"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--old-pid", type=int, default=3803206)
    p.add_argument("--old-start-ticks", type=int, default=1542884730)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return selftest()
        gates.require(not (args.out_dir / OUTPUT).exists(), "Immutable precision artifact already exists")
        target = args.out_dir.resolve()
        gates.require(args.anatomy.parent.resolve() not in (target, *target.parents) and
                      args.cache_dir.resolve() not in (target, *target.parents),
                      "New artifacts cannot be written within old anatomy/cache directories")
        return precision(args)
    except gates.Blocked as exc:
        print(json.dumps({"status": "BLOCKED", "gate": args.command, "protocol": PROTOCOL, "reason": str(exc)}), file=sys.stderr)
        return 3
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "gate": args.command, "protocol": PROTOCOL,
                          "error": type(exc).__name__, "reason": str(exc)}), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
