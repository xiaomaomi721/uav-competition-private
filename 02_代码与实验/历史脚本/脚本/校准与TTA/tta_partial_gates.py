#!/usr/bin/env python3
"""Authorized PARTIAL Phase-B reference gates; never assert full-B completion.

Commands: reference/select/precision/correctness/benchmark/selftest. Common inputs:
 --config CFG --checkpoint CKPT --val-list VAL --cache-dir CACHE --out-dir NEW
 --stop-receipt PHASE_B_EARLY_STOPPED_FOR_PERFORMANCE.json
select additionally needs --reference JSON --assignments CSV --image-dir DIR --mask-dir DIR.
GPU commands additionally need --reference JSON --sample TTA_V2_FIXED16.json.
Run precision -> correctness -> benchmark in the same NEW output directory. No
benchmark prerequisite is fabricated. Benchmark checks all11 schemes for each
candidate; a selected-batch correctness rerun is not required by this protocol.

Only physical GPU2, CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID. All original
cache/report/script files are read-only. Invalid interrupted .npy files are recorded
and excluded, NEVER deleted/repaired. Reference PASS means >=1000 complete readable
finite fp16 HWC1024x1024x8 val files, NOT all1399 complete. Finite allzero arrays are
valid; zero-byte files are invalid. All valid files receive full SHA256 once in
reference audit; selected files are SHA-rechecked for GPU gates. Golden full-val D
calibration is deferred: calibration_allowed is ALWAYS false for this partial audit.

Outputs are immutable JSON. payload_sha256 = sorted compact UTF8 JSON hash excluding
only itself (ensure_ascii=False/allow_nan=False). artifact_sha256 binds source/input
file bytes. Exit0 PASS;2 measured FAIL;3 BLOCKED;4 unexpected ERROR. CPU selftest
uses only temporary files, no CUDA/SSH. This entrypoint never sends any process signal.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tta_v2_gates as base
import tta_priority_gates as priority

PROTOCOL = "authorized_partial_reference_v1"
OUTPUTS = {"reference": "PHASE_B_REFERENCE_SUFFICIENT.json", "select": "TTA_V2_FIXED16.json",
           "precision": "FP16_LOGIT_CACHE_PRECISION_AUDIT.json", "correctness": "TTA_V2_CORRECTNESS.json",
           "benchmark": "TTA_VIEW_BATCH_BENCHMARK.json"}
MIN_REFERENCE = 1000
STOP_RECEIPT_SHA256 = "45d5f0ea8cb3211cb637fb5214671f7918b8c9cac2acfabb8eb55a29faa4f8ff"


def stop_receipt(args):
    base.require(base.sha256_file(args.stop_receipt) == STOP_RECEIPT_SHA256, "Authorized stop receipt SHA mismatch")
    receipt = json.loads(args.stop_receipt.read_text(encoding="utf-8"))
    # Exact authorized receipt fields are checked below; no unsigned runtime
    # assertion such as --assume-stopped is accepted as a replacement.
    base.require(receipt.get("status") == "PHASE_B_EARLY_STOPPED_FOR_PERFORMANCE" and
                 receipt.get("reason") == "OLD_TTA_PATH_SUSTAINED_THROUGHPUT_DEGRADATION" and
                 receipt.get("directory_classification") == "PARTIAL_OLD_ENGINE_REFERENCE", "Wrong authorized stop receipt")
    base.require(receipt.get("pid") == args.old_pid, "Stop receipt PID mismatch")
    identity = receipt.get("terminal_identity", {})
    base.require(identity.get("pid") == args.old_pid and identity.get("starttime") == args.old_start_ticks and
                 identity.get("state") in ("Z", "X", "x"), "Stop receipt terminal identity mismatch")
    base.require(receipt.get("signal") == "SIGTERM_pidfd", "Receipt must document pidfd SIGTERM")
    base.require(receipt.get("artifacts_deleted") is False and receipt.get("old_cache_modified_by_auditor") is False,
                 "Stop receipt does not preserve old artifacts")
    base.require(Path(receipt.get("cache_dir", "")).resolve() == args.cache_dir.resolve(), "Stop receipt cache path mismatch")
    base.require(type(receipt.get("cache_count_after")) is int and receipt["cache_count_after"] >= MIN_REFERENCE,
                 "Stop receipt has insufficient reference inventory")
    return receipt


def terminal(args):
    base.require(sys.platform.startswith("linux"), "Partial live process check requires Linux; no SSH/GPU here")
    p = Path(f"/proc/{args.old_pid}/stat")
    try:
        text = p.read_text()
    except FileNotFoundError:
        return {"pid": args.old_pid, "state": "absent", "authorized_stop_receipt_preserved": True}
    fields = text[text.rfind(")") + 2:].split()
    actual_start = int(fields[19])
    if actual_start != args.old_start_ticks:
        return {"pid": args.old_pid, "state": "pid_reused", "actual_start_ticks": actual_start}
    base.require(fields[0] in ("Z", "X", "x"), f"Old Phase B still live: {fields[0]}")
    code = int(fields[49])
    base.require(code in (0, 15, 128 + 15, (128 + 15) << 8), f"Unexpected terminal exit code {code}")
    return {"pid": args.old_pid, "state": fields[0], "start_ticks": actual_start,
            "exit_code": code, "interpretation": "Authorized SIGTERM is terminal, not a quality failure"}


def identity(args):
    ident, stems = base.identity(args)
    # Parent helper's n_images1399 means validation universe only, NOT reference coverage.
    ident.pop("n_images")
    ident["validation_universe_n"] = len(stems)
    return ident, stems


def source_bindings():
    return {str(Path(m.__file__).resolve()): base.sha256_file(m.__file__)
            for m in (base, priority)} | {str(Path(__file__).resolve()): base.sha256_file(__file__)}


def emit(args, obj):
    return base.write_artifact(args.out_dir / OUTPUTS[args.command], dict(obj, protocol=PROTOCOL,
        partial_gates_source_sha256=base.sha256_file(__file__), calibration_allowed=False,
        full_phase_b_complete=False))


def scan_cache(cache_dir, stems, minimum=MIN_REFERENCE, shape=base.SHAPE):
    expected = set(stems)
    inventory, excluded = {}, []
    for p in sorted(cache_dir.glob("*.npy")):
        if p.stem not in expected:
            excluded.append({"file": p.name, "reason": "not_in_frozen_val_list", "preserved": True})
            continue
        try:
            before = p.stat()
            base.require(before.st_size > 0, "zero_byte_file")
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            base.validate_cache_array(arr, p.stem, shape)
            del arr
            digest = base.sha256_file(p)
            after = p.stat()
            base.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file_changed_during_audit")
            inventory[p.stem] = {"sha256": digest, "size_bytes": after.st_size}
        except (base.Blocked, ValueError, OSError, EOFError) as exc:
            excluded.append({"file": p.name, "reason": str(exc), "preserved": True})
    base.require(len(inventory) >= minimum, f"Only {len(inventory)} complete finite caches; require >= {minimum}")
    return inventory, excluded


def reference(args):
    receipt = stop_receipt(args)
    process = terminal(args)
    ident, stems = identity(args)
    inventory, excluded = scan_cache(args.cache_dir, stems)
    terminal(args)
    missing = [s for s in stems if s not in inventory]
    return emit(args, dict(ident, status="PASS", gate="reference", classification="PARTIAL_REFERENCE_NOT_COMPLETE",
        valid_count=len(inventory), missing_count=len(missing), missing_stems=missing, excluded=excluded,
        cache_inventory=inventory, cache_inventory_sha256=base.canonical_hash(inventory), minimum_required=MIN_REFERENCE,
        finite_validation="all readable candidates, not sampled", stop_receipt=receipt, process=process,
        artifact_sha256=source_bindings() | {str(args.stop_receipt.resolve()): base.sha256_file(args.stop_receipt)}))


def reference_inputs(args):
    stop_receipt(args)
    terminal(args)
    ident, stems = identity(args)
    ref = base.load_artifact(args.reference)
    base.require(ref.get("protocol") == PROTOCOL and ref.get("gate") == "reference" and
                 ref.get("classification") == "PARTIAL_REFERENCE_NOT_COMPLETE", "Not an authorized partial reference")
    for k, v in ident.items():
        base.require(ref.get(k) == v, f"Reference identity mismatch: {k}")
    inventory = ref["cache_inventory"]
    base.require(ref.get("valid_count") == len(inventory) >= MIN_REFERENCE and set(inventory) <= set(stems) and
                 ref["cache_inventory_sha256"] == base.canonical_hash(inventory), "Invalid partial cache inventory")
    base.require(ref["artifact_sha256"].get(str(args.stop_receipt.resolve())) == base.sha256_file(args.stop_receipt),
                 "Stop receipt changed")
    bindings = source_bindings() | {str(args.reference.resolve()): base.sha256_file(args.reference),
                                   str(args.stop_receipt.resolve()): base.sha256_file(args.stop_receipt)}
    return ident, ref, bindings


def select(args):
    ident, ref, bindings = reference_inputs(args)
    base.require(0 < args.disagreement_threshold <= 1e-4 and 0 < args.iou_threshold <= 1e-4, "Invalid frozen thresholds")
    with args.assignments.open(encoding="utf-8-sig", newline="") as stream:
        domains = {}
        for row in csv.DictReader(stream):
            stem = Path(row["image_name"]).stem
            base.require(stem not in domains, f"Duplicate domain assignment: {stem}")
            domains[stem] = int(str(row["domain_id"]).removeprefix("D"))
    pools = {d: [] for d in range(3)}
    for stem in sorted(ref["cache_inventory"]):
        base.require(domains.get(stem) in pools, f"No D0/1/2 assignment: {stem}")
        gt = base.read_gt(args.mask_dir / f"{stem}.png")
        barren, agri = int((gt == 4).sum()), int((gt == 6).sum())
        pools[domains[stem]].append({"stem": stem, "barren_gt_pixels": barren, "agri_gt_pixels": agri})
    rng = random.Random(42)
    picked = []
    for domain, quota in enumerate((6, 5, 5)):
        pool = pools[domain]
        base.require(len(pool) >= quota, f"D{domain} insufficient complete-cache images")
        rng.shuffle(pool)
        pool.sort(key=lambda r: -(r["barren_gt_pixels"] + r["agri_gt_pixels"]))
        chosen = []
        # Guarantee each rare class when it exists within the domain, then seeded
        # sample from the favored half. Union across domains must contain both.
        for key in ("barren_gt_pixels", "agri_gt_pixels"):
            candidates = [r for r in pool if r[key] > 0 and r not in chosen]
            if candidates:
                chosen.append(max(candidates, key=lambda r: r[key]))
        remaining = [r for r in pool[:max(quota, (len(pool)+1)//2)] if r not in chosen]
        chosen += rng.sample(remaining, quota-len(chosen))
        for row in chosen:
            s = row["stem"]
            ip, mp = args.image_dir / f"{s}.png", args.mask_dir / f"{s}.png"
            picked.append(dict(row, domain=f"D{domain}", image=str(ip.resolve()), mask=str(mp.resolve()),
                               image_sha256=base.sha256_file(ip), mask_sha256=base.sha256_file(mp)))
    picked = [picked[i] for i in (0, 6, 11, 1, 7, 12, 2, 8, 13, 3, 9, 14, 4, 10, 15, 5)]
    base.require(sum(r["barren_gt_pixels"] for r in picked) > 0 and sum(r["agri_gt_pixels"] for r in picked) > 0,
                 "Frozen16 must include both Barren and Agricultural GT")
    return emit(args, dict(ident, status="PASS", gate="select", seed=42, n=16, samples=picked,
        sample_stems_sha256=base.sha256_lines([r["stem"] for r in picked]), frozen_before_gpu=True,
        assignments_sha256=base.sha256_file(args.assignments), artifact_sha256=bindings, bias_vectors=base.BIASES,
        thresholds={"disagreement": args.disagreement_threshold, "aggregate_iou_delta": args.iou_threshold, "logit_error": 1e-3},
        correctness_n=args.correctness_n, selection_policy="Complete-cache-only D0/1/2 6/5/5; include Barren+Agricultural; seed42 favored-half",
        benchmark_policy={"batches": [1, 2, 4, 8], "warmup": 2, "repeats": 5, "max_relative_range": 0.20,
                          "speed_tie_fraction": 0.10, "minimum_margin_bytes": base.MIN_MARGIN}))


def preflight(args):
    ident, ref, bindings = reference_inputs(args)
    base.gpu_exclusive()  # before torch/model
    sample = base.load_artifact(args.sample)
    base.require(sample.get("protocol") == PROTOCOL and sample.get("gate") == "select" and sample.get("seed") == 42 and
                 sample.get("n") == 16 and sample.get("frozen_before_gpu") is True, "Invalid frozen16")
    for k, v in ident.items():
        base.require(sample.get(k) == v, f"Sample identity mismatch {k}")
    base.require(sample["artifact_sha256"].get(str(args.reference.resolve())) == base.sha256_file(args.reference), "Sample reference changed")
    rows = sample["samples"]
    base.require(len(rows) == len({r["stem"] for r in rows}) == 16 and all(r["stem"] in ref["cache_inventory"] for r in rows), "Invalid sample membership")
    base.require(sample["sample_stems_sha256"] == base.sha256_lines([r["stem"] for r in rows]), "Frozen stem hash mismatch")
    base.require([sum(r["domain"] == d for r in rows) for d in ("D0", "D1", "D2")] == [6, 5, 5], "Frozen domain quotas differ")
    base.require(sample["bias_vectors"] == base.BIASES and 0 < sample["thresholds"]["disagreement"] <= 1e-4 and
                 0 < sample["thresholds"]["aggregate_iou_delta"] <= 1e-4 and sample["thresholds"]["logit_error"] == 1e-3,
                 "Frozen bias/threshold drift")
    for row in rows:
        for kind in ("image", "mask"):
            base.require(base.sha256_file(row[kind]) == row[kind+"_sha256"], "Frozen image/mask changed")
    bindings[str(args.sample.resolve())] = base.sha256_file(args.sample)
    # Enforce required numerical gate order; reference/sample alone never means
    # precision/correctness acceptance. No fake benchmark prerequisite exists.
    required = [] if args.command == "precision" else ["precision"]
    if args.command == "benchmark":
        required.append("correctness")
    for name in required:
        path = args.out_dir / OUTPUTS[name]
        prior = base.load_artifact(path)
        base.require(prior.get("protocol") == PROTOCOL and prior.get("gate") == name, f"Wrong prior {name} gate")
        for k, v in ident.items():
            base.require(prior.get(k) == v, f"Prior {name} identity drift {k}")
        base.require(prior["artifact_sha256"].get(str(args.sample.resolve())) == bindings[str(args.sample.resolve())], "Prior sample drift")
        bindings[str(path.resolve())] = base.sha256_file(path)
    import torch
    import tta_anatomy as old
    import tta_anatomy_v2 as engine
    base.require(base.sha256_file(old.__file__) == base.EXPECTED_OLD_SOURCE and old.SCHEMES == engine.SCHEMES and old.VIEWS == engine.VIEWS,
                 "Audited reference/engine contract drift")
    base.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected GPU2 only")
    for module in (engine, old):
        bindings[str(Path(module.__file__).resolve())] = base.sha256_file(module.__file__)
    for name in required:
        prior = base.load_artifact(args.out_dir / OUTPUTS[name])
        base.require(prior.get("backend") == base.backend_snapshot(), "Backend changed since prior gate")
        for module in (engine, old, base, priority):
            path = str(Path(module.__file__).resolve())
            base.require(prior["artifact_sha256"].get(path) == bindings[path], "Source changed since prior gate")
    model = engine.build_model(args.config, args.checkpoint, torch.device("cuda:0"))
    base.require(all(p.device.type == "cuda" and p.device.index == 0 for p in model.parameters()), "Model not wholly GPU2")
    return engine, model, ident, ref, sample, bindings


def finish(args, ident, bindings, **kwargs):
    base.gpu_exclusive()
    terminal(args)
    for p, digest in bindings.items():
        base.require(base.sha256_file(p) == digest, f"Source/artifact changed during gate {p}")
    return emit(args, dict(ident, gate=args.command, artifact_sha256=bindings, backend=base.backend_snapshot(), **kwargs))


def precision(args):
    engine, model, ident, ref, sample, bindings = preflight(args)
    import torch
    reports = []
    for item in sample["samples"]:
        base.gpu_exclusive()
        stem = item["stem"]
        cache = args.cache_dir / f"{stem}.npy"
        base.require(base.sha256_file(cache) == ref["cache_inventory"][stem]["sha256"], f"Cache changed {stem}")
        cached = np.load(cache, allow_pickle=False)
        base.validate_cache_array(cached, stem)
        cached = cached.astype(np.float32)
        px, gt = base.image_tensor(item)
        priority.check_label_mapping(item, gt)
        with torch.inference_mode():
            views = engine.forward_views(model, px, 1)
            base.require(tuple(views.shape) == (8, 8, 1024, 1024) and views.dtype == torch.float32 and views.is_cuda, "Forward shape/dtype/device mismatch")
            precise = views.mean(dim=0, dtype=torch.float32).permute(1, 2, 0).cpu().numpy()
        reports.append(priority.compare_image(precise, cached, gt, stem, sample["thresholds"]))
        del views, px, precise, cached
        print(f"Partial precision {len(reports)}/16", flush=True)
    comparisons, passed = priority.aggregate(reports, sample["thresholds"])
    return finish(args, ident, bindings, status="PASS" if passed else "FAIL", n=16,
        sample_stems=[s["stem"] for s in sample["samples"]], thresholds=sample["thresholds"], bias_vectors=base.BIASES,
        precision_execution="partial_reference_fixed_batch1", view_batch_size=1,
        batch_selection="fixed_preregistered_not_benchmarked", benchmark_evaluated=False,
        precision_status="FP16_LOGIT_CACHE_PRECISION_PASS" if passed else "FP16_LOGIT_CACHE_PRECISION_FAILED",
        comparisons=comparisons, requires_precise_strategy=not passed,
        reference="Fresh GPU float32 D4 TTA8 mean BEFORE quantization; partial old cache, fixed view_batch_size=1",
        failure_action=None if passed else "STOP: require explicit precise strategy; no threshold relaxation",
        calibration_note="Original Golden full-val D deferred; partial audit never authorizes calibration")


def all_schemes(engine, views, reference, gt, thresholds):
    import torch
    import tta_anatomy as old
    base.require(tuple(views.shape) == (8, 8, 1024, 1024) and views.dtype == torch.float32 and views.is_cuda, "Forward contract mismatch")
    predictions = engine.scheme_predictions(views)
    base.require(set(predictions) == set(old.SCHEMES), "Missing prediction schemes")
    errors = [float(np.abs(reference[i] - views[i].permute(1, 2, 0).cpu().numpy()).max()) for i in range(8)]
    reports = {}
    for name, scores in engine.scheme_scores(views):
        base.require(name in old.SCHEMES and name not in reports, "Unexpected score scheme")
        members, mode = old.SCHEMES[name]
        expected = old.combine(reference, members, mode)
        got = scores.permute(1, 2, 0).float().cpu().numpy()
        r = base.compare_logits(expected, got, gt, thresholds["disagreement"], thresholds["logit_error"])
        pred = predictions[name].cpu().numpy()
        base.require(pred.dtype == np.uint8 and np.array_equal(pred, got.argmax(-1)), "Prediction/score mismatch")
        actual_cm = engine.confusion_tensor(predictions[name], torch.as_tensor(gt, device=views.device)).cpu().numpy()
        base.require(np.array_equal(actual_cm, np.asarray(r["confusion_candidate"])), "GPU confusion mismatch")
        r["gpu_confusion_matches_cpu"] = True
        r["score_kind"] = "probability" if mode == "prob" else "logit"
        reports[name] = r
    base.require(set(reports) == set(old.SCHEMES), "Missing score schemes")
    return {"status": "PASS" if max(errors) <= thresholds["logit_error"] and all(r["status"] == "PASS" for r in reports.values()) else "FAIL",
            "view_max_abs_errors": errors, "schemes": reports}


def correctness(args):
    engine, model, ident, ref, sample, bindings = preflight(args)
    import torch
    n = sample["correctness_n"]
    base.require(n in (2, 3, 4), "Correctness requires frozen2-4")
    reports = []
    for item in sample["samples"][:n]:
        base.gpu_exclusive()
        px, gt = base.image_tensor(item)
        priority.check_label_mapping(item, gt)
        reference = base.old_reference(model, px)
        with torch.inference_mode():
            views = engine.forward_views(model, px, 1)
            r = all_schemes(engine, views, reference, gt, sample["thresholds"])
        reports.append(dict(r, stem=item["stem"]))
        del views, reference, px
    return finish(args, ident, bindings, status="PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL",
        n=n, view_batch_size=1, images=reports, thresholds=sample["thresholds"], benchmark_evaluated=False,
        reference="Literal audited old per-view CPU NumPy reference; all11 scores/predictions/GPU confusion")


def benchmark(args):
    import gc
    engine, model, ident, ref, sample, bindings = preflight(args)
    import torch
    import tta_anatomy as old
    item = sample["samples"][0]
    px, gt = base.image_tensor(item)
    priority.check_label_mapping(item, gt)
    reference = base.old_reference(model, px)
    policy = sample["benchmark_policy"]
    records = []
    def run(batch):
        if batch == "old_batch1":
            oldviews = base.old_reference(model, px)
            for members, mode in old.SCHEMES.values():
                old.combine(oldviews, members, mode).argmax(-1)
        else:
            with torch.inference_mode():
                v = engine.forward_views(model, px, batch)
                engine.scheme_predictions(v)
    with base.UtilSampler() as sampler:
        for batch in ("old_batch1", 1, 2, 4, 8):
            start = time.time()
            try:
                base.gpu_exclusive()
                gc.collect()
                torch.cuda.empty_cache()
                for _ in range(policy["warmup"]):
                    run(batch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(policy["repeats"]):
                    base.gpu_exclusive()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    run(batch)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter()-t)
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                free, total = torch.cuda.mem_get_info()
                margin = max(0, int(free - max(0, reserved-torch.cuda.memory_reserved())))
                equivalent = {"status": "PASS", "reference": "literal old baseline"}
                if batch != "old_batch1":
                    with torch.inference_mode():
                        v = engine.forward_views(model, px, batch)
                        equivalent = all_schemes(engine, v, reference, gt, sample["thresholds"])
                    del v
                median = statistics.median(timings)
                records.append({"view_batch_size": batch, "status": equivalent["status"], "seconds": timings,
                    "median_seconds": median, "stable": (max(timings)-min(timings))/median <= policy["max_relative_range"],
                    "margin_bytes": margin, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                    "total_bytes": total, "all11_equivalence_vs_old_batch1": equivalent})
            except torch.cuda.OutOfMemoryError:
                records.append({"view_batch_size": batch, "status": "OOM", "stable": False,
                                "fallback": "Only registered batch changes; no model/crop/precision change"})
                gc.collect()
                torch.cuda.empty_cache()
            records[-1]["sample_window_unix"] = [start, time.time()]
    utilization = sampler.report()
    for row in records:
        start, end = row["sample_window_unix"]
        sampled = [s for s in utilization["samples"] if start <= s["unix"] <= end]
        row["utilization"] = {"samples": sampled,
                              "mean_percent": statistics.mean(s["utilization_percent"] for s in sampled) if sampled else None,
                              "max_percent": max((s["utilization_percent"] for s in sampled), default=None)}
        if row["status"] == "PASS" and records[0]["status"] == "PASS":
            row["speedup_vs_old_batch1"] = records[0]["median_seconds"] / row["median_seconds"]
    selected, reason = None, None
    try:
        base.require(records[0]["status"] == "PASS" and records[0]["stable"], "Old baseline unavailable/unstable")
        selected = base.choose_batch(records[1:])
    except base.Blocked as exc:
        reason = str(exc)
    return finish(args, ident, bindings, status="PASS" if selected is not None else "FAIL",
        selected_view_batch_size=selected, reason=reason, old_batch1_reference=records[0], batches=records[1:],
        policy=policy, utilization=utilization, all11_checked_each_candidate=True,
        image_stem=item["stem"], selected_batch_additional_correctness_required=False,
        timing_scope="Same image/model; all8 forwards/inverses and11 scheme predictions; excludes data read/equivalence",
        utilization_scope="Includes warmup/repeats/untimed all11 equivalence; nvidia-smi -i2 only")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        np.save(root / "valid.npy", np.zeros((2, 2, 8), np.float16))
        (root / "zero.npy").write_bytes(b"")
        (root / "partial.npy").write_bytes(b"\x93NUMPY\x01")
        np.save(root / "nan.npy", np.full((2, 2, 8), np.nan, np.float16))
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        inventory, excluded = scan_cache(root, ["valid", "zero", "partial", "nan", "missing"], minimum=1, shape=(2, 2, 8))
        assert set(inventory) == {"valid"} and len(excluded) == 3
        assert {p.name: p.read_bytes() for p in root.iterdir()} == before
        try:
            scan_cache(root, ["valid"], minimum=2, shape=(2, 2, 8))
        except base.Blocked:
            pass
        else:
            raise AssertionError("Insufficient reference accepted")
    priority.selftest()
    print("PARTIAL_SELF_TEST_OK: invalid/truncated/zero-byte preserved, finitezero accepted, minimum reference enforced; CPU only")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    for command in OUTPUTS:
        p = sub.add_parser(command)
        for name in ("config", "checkpoint", "val-list", "cache-dir", "out-dir", "stop-receipt"):
            p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--old-pid", type=int, default=3803206)
        p.add_argument("--old-start-ticks", type=int, default=1542884730)
        if command != "reference":
            p.add_argument("--reference", type=Path, required=True)
        if command == "select":
            for name in ("assignments", "image-dir", "mask-dir"):
                p.add_argument("--"+name, type=Path, required=True)
            p.add_argument("--disagreement-threshold", type=float, default=1e-4)
            p.add_argument("--iou-threshold", type=float, default=1e-4)
            p.add_argument("--correctness-n", type=int, choices=(2, 3, 4), default=4)
        elif command != "reference":
            p.add_argument("--sample", type=Path, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return selftest()
        target = args.out_dir.resolve()
        base.require(args.cache_dir.resolve() not in (target, *target.parents), "Do not write inside original cache")
        base.require(target != args.cache_dir.parent.resolve(), "New gate directory must not be original anatomy directory")
        base.require(not (args.out_dir / OUTPUTS[args.command]).exists(), "Immutable output already exists")
        return globals()[args.command](args)
    except base.Blocked as exc:
        print(json.dumps({"status": "BLOCKED", "gate": args.command, "reason": str(exc)}), file=sys.stderr)
        return 3
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "gate": args.command, "error": type(exc).__name__, "reason": str(exc)}), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
