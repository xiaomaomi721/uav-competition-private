#!/usr/bin/env python3
"""Fail-closed, read-only-old-output gates for the isolated TTA v2 engine.

CPU commands (never initialize CUDA):
  python tta_v2_gates.py selftest
  python tta_v2_gates.py integrity --config CFG --checkpoint CKPT --val-list VAL \
      --cache-dir OLD/mean_logits_fp16 --anatomy OLD/TTA_ANATOMY.json --out-dir NEW
  python tta_v2_gates.py select --val-list VAL --assignments CSV \
      --image-dir IMAGES --mask-dir MASKS --out-dir NEW
GPU commands, ONLY after the old child exits and GPU2 is exclusive:
  CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID python tta_v2_gates.py \
      benchmark|correctness|precision --config CFG --checkpoint CKPT --val-list VAL \
      --cache-dir OLD/mean_logits_fp16 --anatomy OLD/TTA_ANATOMY.json --out-dir NEW \
      --sample NEW/TTA_V2_FIXED16.json --integrity NEW/PHASE_B_OUTPUT_COMPLETE.json
  correctness and precision additionally require --benchmark BENCHMARK_JSON.
  benchmark/correctness optionally accept --expected-checkpoint-sha256 HASH and
  --expected-config-sha256 HASH for another frozen model (e.g. Stage66); the oldB
  integrity prerequisite remains bound to Golden, and precision is Golden-only.

All outputs are NEW immutable JSON, never old reports/cache. Existing output paths
are rejected; use a new output directory for a new attempt. payload_sha256 is SHA256
of UTF-8 sorted-key compact JSON (ensure_ascii=False, allow_nan=False) with ONLY the
payload_sha256 key removed. artifact_sha256 maps input paths to full file hashes.
Raw val-list SHA and canonical ordered stem-list SHA are deliberately separate.
Exit codes: 0 PASS; 2 measured gate FAIL; 3 BLOCKED precondition; 4 unexpected error
(including malformed input). An integrity marker is emitted ONLY on PASS. Other
completed numerical gates emit PASS/FAIL reports; precondition failures emit no
misleading acceptance report. Precision FAIL blocks calibration on OLD cache, not C.

Correctness requires per-image/per-scheme disagreement < frozen threshold and
max absolute score/logit residual <= 1e-3. Disagreements are allowed ONLY when the
old-reference top-two margin <= 2*that pixel's measured max score residual plus
8*float32 epsilon*max(1,score magnitude); this explicitly excludes meaningful
class changes. Probability schemes compare probabilities, not mislabeled logits.
Precision audits BOTH actual old-cache combined numerical drift and v2 float32 ->
float16 self-quantization separately. No already-quantized reference is accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

EXPECTED_CHECKPOINT = "d3740b4ef73fbb3ac57ddeedc3547ce4ce1ba31442dc7c70e1fe96b1fab3e228"
EXPECTED_CONFIG = "68cf3a98b7bb17dc564fbbb31dced103543e86bd9e01b967c161c37f1120995c"
EXPECTED_VAL = "2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74"
EXPECTED_OLD_SOURCE = "3aaa1dc600875c6835aef8a742ad42429fc524cd1716f191b9b1ad8e77465209"
SHAPE = (1024, 1024, 8)
N_VAL = 1399
MIN_MARGIN = 2 * 1024**3
BIASES = {
    "no_bias": [0.0] * 8,
    "FIXED8D_EXP41_STYLE": [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5],
    "LOCKED8D_69_276229": [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.1932, 3.3215, -0.5],
}
OUTPUTS = {"integrity": "PHASE_B_OUTPUT_COMPLETE.json", "select": "TTA_V2_FIXED16.json",
           "benchmark": "TTA_VIEW_BATCH_BENCHMARK.json", "correctness": "TTA_V2_CORRECTNESS.json",
           "precision": "FP16_LOGIT_CACHE_PRECISION_AUDIT.json"}


class Blocked(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Blocked(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_lines(lines):
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def load_artifact(path, status="PASS"):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = obj.get("payload_sha256")
    require(digest == canonical_hash({k: v for k, v in obj.items() if k != "payload_sha256"}),
            f"Artifact payload SHA mismatch: {path}")
    if status is not None:
        require(obj.get("status") == status, f"Artifact is not {status}: {path}")
    return obj


def write_artifact(path, obj):
    path = Path(path)
    require(not path.exists(), f"Refuse to overwrite existing artifact: {path}")
    obj = dict(obj, schema_version=1, created_unix=time.time(),
               gates_source_sha256=sha256_file(__file__))
    obj["payload_sha256"] = canonical_hash(obj)
    raw = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: even concurrent invocations cannot clobber old output.
    with path.open("x", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"artifact": str(path.resolve()), "sha256": sha256_file(path),
                      "status": obj["status"]}), flush=True)
    return 0 if obj["status"] == "PASS" else 2


def stems_from_list(path, expected=True):
    if expected:
        require(sha256_file(path) == EXPECTED_VAL, "Raw val-list SHA mismatch")
    lines = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    stems = [Path(x).stem for x in lines]
    require(len(stems) == N_VAL and len(set(stems)) == N_VAL, "Expected 1399 unique val stems")
    require(all(s and Path(s).name == s and "/" not in s and "\\" not in s for s in stems),
            "Invalid val stem")
    return stems


def old_process_terminal(pid=3803206, start_ticks=1542884730):
    require(sys.platform.startswith("linux"), "Live /proc identity check requires Linux; no GPU/SSH here")
    p = Path(f"/proc/{pid}/stat")
    try:
        text = p.read_text()
    except FileNotFoundError:
        return {"pid": pid, "expected_start_ticks": start_ticks, "state": "absent",
                "terminal": True, "natural_exit_evidence": "complete output checked separately; no signal sent"}
    # comm can contain spaces and parentheses; split after its LAST closing paren.
    fields = text[text.rfind(")") + 2:].split()
    actual_start = int(fields[19])  # field22, relative to state(field3)
    state = fields[0]
    if actual_start != start_ticks:
        return {"pid": pid, "expected_start_ticks": start_ticks, "actual_start_ticks": actual_start,
                "state": "pid_reused", "terminal": True}
    require(state in ("Z", "X", "x"), f"Old Phase B still live: PID {pid}, state {state}")
    require(len(fields) > 49 and int(fields[49]) == 0, "Old Phase B terminal exit_code is not zero")
    return {"pid": pid, "expected_start_ticks": start_ticks, "state": state,
            "exit_code": int(fields[49]), "terminal": True}


def identity(args):
    stems = stems_from_list(args.val_list)
    ck, cfg = sha256_file(args.checkpoint), sha256_file(args.config)
    expected_ck = getattr(args, "expected_checkpoint_sha256", EXPECTED_CHECKPOINT)
    expected_cfg = getattr(args, "expected_config_sha256", EXPECTED_CONFIG)
    require(ck == expected_ck, "Checkpoint SHA mismatch")
    require(cfg == expected_cfg, "Config SHA mismatch")
    return {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": ck,
            "config": str(args.config.resolve()), "config_sha256": cfg,
            "split_sha256": sha256_file(args.val_list), "image_list_sha256": sha256_lines(stems),
            "cache_dir": str(args.cache_dir.resolve()), "n_images": N_VAL}, stems


def validate_cache_array(a, stem, shape=SHAPE):
    require(a.shape == shape, f"{stem}: cache shape {a.shape} != {shape}")
    require(a.dtype == np.float16, f"{stem}: cache dtype is not float16")
    require(bool(np.isfinite(a).all()), f"{stem}: nonfinite cache values")
    # Finite all-zero logits can be a valid tensor; empty/truncated files are
    # rejected by the file-size/np.load integrity checks, not pixel values.


def validate_anatomy(r, ident, stems):
    import tta_anatomy as old
    for key in ("checkpoint_sha256", "config_sha256", "n_images", "image_list_sha256"):
        require(r.get(key) == ident[key], f"Incomplete/mismatched anatomy field {key}")
    require(r.get("split") == "val", "Anatomy is not val")
    require(r.get("views") == old.VIEW_NAMES, "Anatomy D4 views mismatch")
    require(set(r.get("schemes", {})) == set(old.SCHEMES), "Anatomy must contain exactly all 11 schemes")
    valid_totals = set()
    for name, m in r["schemes"].items():
        for key in ("mIoU", "pixel_accuracy"):
            v = m.get(key)
            require(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1,
                    f"Invalid anatomy {name}.{key}")
        for key in ("per_class_iou", "per_class_recall"):
            v = m.get(key)
            require(isinstance(v, list) and len(v) == 8 and all(x is None or
                    (isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1) for x in v),
                    f"Invalid anatomy {name}.{key}")
        for key in ("gt_pixels", "pred_pixels"):
            v = m.get(key)
            require(isinstance(v, list) and len(v) == 8 and all(type(x) is int and x >= 0 for x in v),
                    f"Invalid anatomy {name}.{key}")
        require(sum(m["gt_pixels"]) == sum(m["pred_pixels"]) > 0, "Anatomy pixel totals mismatch")
        require(sum(m["gt_pixels"]) <= N_VAL * SHAPE[0] * SHAPE[1], "Anatomy impossible pixel coverage")
        finite_iou = [x for x in m["per_class_iou"] if x is not None]
        require(bool(finite_iou) and abs(statistics.mean(finite_iou) - m["mIoU"]) <= 1e-10,
                f"Anatomy {name} mIoU inconsistent with per-class values")
        valid_totals.add(tuple(m["gt_pixels"]))
    require(len(valid_totals) == 1, "Anatomy scheme GT coverage differs")
    names = list(old.SCHEMES)
    pairs = {f"{a}_vs_{b}" for i, a in enumerate(names) for b in names[i+1:]}
    p = r.get("pairwise_disagreement", {})
    require(set(p) == pairs and all(isinstance(x, (int, float)) and math.isfinite(x) and
                                   0 <= x <= 1 for x in p.values()), "Incomplete pairwise anatomy")
    cache = r.get("mean_logits_cache", {})
    require(cache.get("dtype") == "float16" and cache.get("shape") == list(SHAPE), "Anatomy cache spec mismatch")
    require(Path(cache.get("dir") or "").resolve() == Path(ident["cache_dir"]), "Anatomy cache path mismatch")
    require(isinstance(r.get("elapsed_seconds"), (int, float)) and
            math.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0, "Anatomy elapsed invalid")


def integrity(args):
    process = old_process_terminal(args.old_pid, args.old_start_ticks)
    ident, stems = identity(args)
    r = json.loads(args.anatomy.read_text(encoding="utf-8"))
    validate_anatomy(r, ident, stems)
    md = args.anatomy.with_suffix(".md")
    require(md.is_file() and md.stat().st_size > 0, "Old anatomy markdown report missing/empty")
    require({p.name for p in args.cache_dir.glob("*.npy")} == {f"{s}.npy" for s in stems},
            "Cache inventory must match EXACTLY 1399 val stems")
    inventory = {}
    for i, stem in enumerate(stems):
        p = args.cache_dir / f"{stem}.npy"
        before = p.stat()
        require(before.st_size > 0, f"Empty cache file: {stem}")
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        validate_cache_array(a, stem)
        del a
        digest = sha256_file(p)
        after = p.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Cache changed during audit: {stem}")
        inventory[stem] = {"sha256": digest, "size_bytes": after.st_size}
        if (i + 1) % 100 == 0:
            print(f"Integrity {i+1}/{len(stems)}", flush=True)
    process_final = old_process_terminal(args.old_pid, args.old_start_ticks)
    result = dict(ident, status="PASS", gate="integrity", process=process, process_final=process_final,
                  cache_inventory=inventory, cache_inventory_sha256=canonical_hash(inventory),
                  artifact_sha256={str(args.anatomy.resolve()): sha256_file(args.anatomy), str(md.resolve()): sha256_file(md)},
                  old_fp16_gate_is_not_evidence=True,
                  exit_evidence_note="Terminal PID plus complete reports/cache; absent PID cannot retrospectively prove OS exit status.")
    return write_artifact(args.out_dir / OUTPUTS["integrity"], result)


def read_gt(path):
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im).copy()
    require(a.ndim == 2 and a.shape == SHAPE[:2], f"GT shape mismatch: {path}")
    require(bool(np.isin(a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 255]).all()), f"Unexpected official GT IDs: {path}")
    gt = np.full(a.shape, 255, dtype=np.uint8)
    valid = (a >= 1) & (a <= 8)
    gt[valid] = a[valid] - 1
    return gt


def select(args):
    stems = stems_from_list(args.val_list)
    require(0 < args.disagreement_threshold <= 1e-4 and 0 < args.iou_threshold <= 1e-4,
            "Frozen thresholds must be positive and cannot relax requested 1e-4 limits")
    with args.assignments.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    domains = {}
    for row in rows:
        s = Path(row["image_name"]).stem
        require(s not in domains, f"Duplicate domain assignment: {s}")
        domains[s] = int(str(row["domain_id"]).removeprefix("D"))
    require(all(domains.get(s) in (0, 1, 2) for s in stems), "Every val image needs D0/D1/D2 assignment")
    rng = random.Random(42)
    pools = {d: [] for d in range(3)}
    for s in stems:
        gt = read_gt(args.mask_dir / f"{s}.png")
        rare = int(((gt == 4) | (gt == 6)).sum())
        pools[domains[s]].append((s, rare))
    selected = []
    # Stratified fixed quotas 6/5/5. Shuffle before stable rare-pixel sort means
    # seed42 resolves ties; select from the top half to favor Barren/agriculture
    # without reducing the audit to only the largest rare-class masks.
    for d, quota in enumerate((6, 5, 5)):
        pool = pools[d]
        require(len(pool) >= quota, f"D{d} has too few images for fixed16")
        rng.shuffle(pool)
        pool.sort(key=lambda x: -x[1])
        favored = pool[:max(quota, (len(pool) + 1)//2)]
        for s, rare in rng.sample(favored, quota):
            ip, mp = args.image_dir / f"{s}.png", args.mask_dir / f"{s}.png"
            require(ip.is_file(), f"Image missing: {ip}")
            selected.append({"stem": s, "domain": f"D{d}", "barren_agri_gt_pixels": rare,
                             "image": str(ip.resolve()), "mask": str(mp.resolve()),
                             "image_sha256": sha256_file(ip), "mask_sha256": sha256_file(mp)})
    # Interleave domains so first 3/4 correctness images cover all domains.
    selected = [selected[i] for i in (0, 6, 11, 1, 7, 12, 2, 8, 13, 3, 9, 14, 4, 10, 15, 5)]
    return write_artifact(args.out_dir / OUTPUTS["select"], {
        "status": "PASS", "gate": "select", "seed": 42, "n": 16, "samples": selected,
        "split_sha256": sha256_file(args.val_list), "image_list_sha256": sha256_lines(stems),
        "assignments_sha256": sha256_file(args.assignments),
        "selection_policy": "D0/D1/D2=6/5/5; seed42 sample from top-half Barren+Agricultural GT pixel counts",
        "thresholds": {"disagreement": args.disagreement_threshold, "aggregate_iou_delta": args.iou_threshold,
                       "logit_error": 1e-3}, "bias_vectors": BIASES,
        "correctness_n": args.correctness_n,
        "benchmark_policy": {"batches": [1, 2, 4, 8], "warmup": 2, "repeats": 5,
                             "max_relative_range": 0.20, "speed_tie_fraction": 0.10,
                             "minimum_margin_bytes": MIN_MARGIN},
        "frozen_before_gpu": True})


def confusion(pred, gt):
    valid = (gt >= 0) & (gt < 8)
    return np.bincount((gt[valid].astype(np.int64)*8 + pred[valid].astype(np.int64)), minlength=64).reshape(8, 8)


def ious(cm):
    diag = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1) - diag
    return np.divide(diag, den, out=np.full(8, np.nan), where=den > 0)


def metric_delta(cm_ref, cm_new):
    a, b = ious(cm_ref), ious(cm_new)
    # A present class becoming absent (or vice versa) must not disappear via nanmean.
    both_missing = np.isnan(a) & np.isnan(b)
    delta = np.where(both_missing, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    ma = float(np.nanmean(a)) if np.isfinite(a).any() else None
    mb = float(np.nanmean(b)) if np.isfinite(b).any() else None
    return {"per_class_iou_reference": [None if np.isnan(x) else float(x) for x in a],
            "per_class_iou_candidate": [None if np.isnan(x) else float(x) for x in b],
            "iou_delta": delta.tolist(), "max_iou_delta": float(delta.max()),
            "miou_reference": ma, "miou_candidate": mb,
            "miou_delta": abs(ma-mb) if ma is not None and mb is not None else None}


def compare_logits(reference, candidate, gt, disagreement_threshold=1e-4, logit_tolerance=1e-3):
    require(reference.shape == candidate.shape and reference.ndim == 3 and reference.shape[-1] == 8,
            "Comparison expects matching HWC8 scores")
    require(gt.shape == reference.shape[:2], "Comparison GT shape mismatch")
    require(reference.dtype == np.float32 and candidate.dtype == np.float32,
            "Comparison requires genuine float32 arrays (caller must not use a quantized reference)")
    require(bool(np.isfinite(reference).all() and np.isfinite(candidate).all()), "Nonfinite comparison scores")
    error = np.abs(reference - candidate)
    max_pixel_error = error.max(-1)
    a, b = reference.argmax(-1), candidate.argmax(-1)
    changed = a != b
    top = np.partition(reference, -2, axis=-1)[..., -2:]
    margin = top.max(-1) - top.min(-1)
    slack = 8 * np.finfo(np.float32).eps * np.maximum(1, np.abs(reference).max(-1))
    meaningful = changed & (margin > 2 * max_pixel_error + slack)
    cm_a, cm_b = confusion(a, gt), confusion(b, gt)
    frac = float(changed.mean())
    passed = frac < disagreement_threshold and float(error.max()) <= logit_tolerance and not meaningful.any()
    perclass = []
    for c in range(8):
        mask = gt == c
        perclass.append(float(changed[mask].mean()) if mask.any() else None)
    return {"status": "PASS" if passed else "FAIL", "mean_abs_logit_error": float(error.mean(dtype=np.float64)),
            "max_abs_logit_error": float(error.max()), "disagreement": frac,
            "disagreement_pixels": int(changed.sum()), "pixels": int(changed.size),
            "per_class_disagreement": perclass, "meaningful_disagreements": int(meaningful.sum()),
            "near_tie_disagreements": int((changed & ~meaningful).sum()),
            "confusion_reference": cm_a.tolist(), "confusion_candidate": cm_b.tolist(),
            "confusion_l1_difference": int(np.abs(cm_a-cm_b).sum()), **metric_delta(cm_a, cm_b)}


def choose_batch(records):
    eligible = [r for r in records if r.get("status") == "PASS" and r.get("stable") and
                r.get("margin_bytes", 0) >= MIN_MARGIN and math.isfinite(r.get("median_seconds", float("inf"))) and
                r["median_seconds"] > 0]
    require(bool(eligible), "No stable batch has >=2 GiB free-memory margin")
    fastest = min(r["median_seconds"] for r in eligible)
    return max(r["view_batch_size"] for r in eligible if r["median_seconds"] <= fastest * 1.10)


def smi(query, kind="gpu"):
    cmd = ["nvidia-smi", "-i", "2", f"--query-{kind}={query}", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    return [x.strip() for x in result.stdout.splitlines() if x.strip()]


def gpu_exclusive():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "CUDA_VISIBLE_DEVICES must be exactly 2")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    rows = smi("pid", "compute-apps")
    foreign = [int(x) for x in rows if x.strip().isdigit() and int(x) != os.getpid()]
    require(not foreign, f"GPU2 is not exclusive; foreign compute PIDs: {foreign}")
    require(all(x.isdigit() for x in rows), "Unparseable GPU2 process query")


class UtilSampler:
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                rows = smi("utilization.gpu,memory.used,memory.total")
                require(len(rows) == 1, "GPU2 sampler expected one GPU")
                u, used, total = [float(x.strip()) for x in rows[0].split(",")]
                gpu_exclusive()
                self.samples.append({"unix": time.time(), "utilization_percent": u,
                                     "used_mib": used, "total_mib": total})
            except Exception as e:
                self.errors.append(str(e))
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=25)
        require(not self.thread.is_alive(), "GPU utilization sampler did not stop")

    def report(self):
        require(not self.errors and bool(self.samples), f"GPU2 sampling failed: {self.errors}")
        u = [s["utilization_percent"] for s in self.samples]
        return {"command_scope": "nvidia-smi -i 2 only", "sample_interval_seconds": 0.5,
                "samples": self.samples, "mean_percent": statistics.mean(u), "max_percent": max(u),
                "note": "Sampled utilization, not profiler kernel occupancy; short bursts can be missed."}


def backend_snapshot():
    import torch
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "torch_version": torch.__version__}


def gpu_preflight(args):
    old_process_terminal(args.old_pid, args.old_start_ticks)
    gpu_exclusive()  # checked BEFORE torch import/model construction
    ident, stems = identity(args)
    marker = load_artifact(args.integrity)
    require(marker.get("gate") == "integrity", "Wrong integrity artifact type")
    require(marker.get("checkpoint_sha256") == EXPECTED_CHECKPOINT and
            marker.get("config_sha256") == EXPECTED_CONFIG, "Integrity prerequisite is not Golden")
    for key in ("split_sha256", "image_list_sha256", "cache_dir", "n_images"):
        require(marker.get(key) == ident[key], f"Integrity binding mismatch: {key}")
    require(marker["artifact_sha256"].get(str(args.anatomy.resolve())) == sha256_file(args.anatomy),
            "Old anatomy changed since integrity")
    sample = load_artifact(args.sample)
    require(sample.get("gate") == "select" and sample.get("frozen_before_gpu") is True and
            sample.get("seed") == 42 and sample.get("n") == 16, "Invalid frozen fixed16 sample")
    require(sample["split_sha256"] == ident["split_sha256"] and sample["bias_vectors"] == BIASES,
            "Sample split/bias binding mismatch")
    samples = sample["samples"]
    require(len(samples) == 16 and len({s["stem"] for s in samples}) == 16 and
            all(s["stem"] in stems for s in samples), "Invalid fixed16 membership")
    require({s["domain"] for s in samples} == {"D0", "D1", "D2"}, "Fixed16 misses a domain")
    for s in samples:
        for kind in ("image", "mask"):
            require(sha256_file(s[kind]) == s[f"{kind}_sha256"], f"Frozen sample {kind} changed: {s['stem']}")
    thresholds = sample["thresholds"]
    require(0 < thresholds["disagreement"] <= 1e-4 and 0 < thresholds["aggregate_iou_delta"] <= 1e-4 and
            thresholds["logit_error"] == 1e-3, "Invalid frozen thresholds")
    import tta_anatomy as old
    import tta_anatomy_v2 as engine
    require(sha256_file(old.__file__) == EXPECTED_OLD_SOURCE, "Old reference source SHA differs from GPU audit")
    require(engine.SCHEMES == old.SCHEMES and engine.VIEWS == old.VIEWS, "Engine scheme/view contract mismatch")
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected exactly one visible CUDA GPU")
    # Preserve historical/production backend defaults; gates must not silently
    # validate a different TF32/cuDNN protocol from the engine actually deployed.
    bindings = {str(args.integrity.resolve()): sha256_file(args.integrity),
                str(args.sample.resolve()): sha256_file(args.sample),
                str(Path(engine.__file__).resolve()): sha256_file(engine.__file__),
                str(Path(old.__file__).resolve()): sha256_file(old.__file__)}
    batch = None
    if args.command in ("correctness", "precision"):
        bench = load_artifact(args.benchmark)
        require(bench.get("gate") == "benchmark", "Wrong benchmark artifact")
        require(bench.get("backend") == backend_snapshot(), "Backend protocol changed since benchmark")
        for key, val in ident.items():
            require(bench.get(key) == val, f"Benchmark binding mismatch: {key}")
        require(all(bench["artifact_sha256"].get(k) == v for k, v in bindings.items()), "Benchmark source/sample drift")
        batch = bench["selected_view_batch_size"]
        require(batch in (1, 2, 4, 8) and choose_batch(bench["batches"]) == batch, "Invalid benchmark selection")
        bindings[str(args.benchmark.resolve())] = sha256_file(args.benchmark)
    model = engine.build_model(args.config, args.checkpoint, torch.device("cuda:0"))
    require(all(p.device.type == "cuda" and p.device.index == 0 for p in model.parameters()), "Model not wholly GPU2")
    return engine, model, ident, sample, marker, bindings, batch


def image_tensor(sample):
    from PIL import Image
    from src.transforms import normalize_to_tensor
    with Image.open(sample["image"]) as im:
        a = np.asarray(im.convert("RGB")).copy()
    require(a.shape == (1024, 1024, 3), "No crop/model-size changes permitted")
    return normalize_to_tensor(a).unsqueeze(0).to("cuda:0"), read_gt(sample["mask"])


def old_reference(model, px):
    """Independent literal historical forward: per-view GPU inverse/resize -> CPU NumPy."""
    import torch
    import tta_anatomy as old
    views = []
    with torch.inference_mode():
        for k, hf in old.VIEWS:
            v = px
            if k:
                v = torch.rot90(v, k=k, dims=[-2, -1])
            if hf:
                v = v.flip(dims=[-1])
            lg = model(pixel_values=v).logits
            if hf:
                lg = lg.flip(dims=[-1])
            if k:
                lg = torch.rot90(lg, k=-k, dims=[-2, -1])
            if tuple(lg.shape[-2:]) != (1024, 1024):
                lg = torch.nn.functional.interpolate(lg, size=(1024, 1024), mode="bilinear", align_corners=False)
            views.append(lg[0].permute(1, 2, 0).float().cpu().numpy())
    return views


def benchmark(args):
    import gc
    engine, model, ident, sample, marker, bindings, _ = gpu_preflight(args)
    import torch
    import tta_anatomy as old
    px, gt = image_tensor(sample["samples"][0])
    policy = sample["benchmark_policy"]
    thresholds = sample["thresholds"]
    records = []
    reference_views = old_reference(model, px)
    reference_mean = old.combine(reference_views, tuple(range(8)), "logit")
    del reference_views
    backend = backend_snapshot()

    def run_one(batch):
        if batch == "old_batch1":
            views = old_reference(model, px)
            for members, mode in old.SCHEMES.values():
                old.combine(views, members, mode).argmax(-1)
        else:
            with torch.inference_mode():
                views = engine.forward_views(model, px, batch)
                engine.scheme_predictions(views)

    with UtilSampler() as sampler:
        for batch in ("old_batch1", 1, 2, 4, 8):
            batch_started = time.time()
            try:
                gpu_exclusive()
                gc.collect()
                torch.cuda.empty_cache()
                for _ in range(policy["warmup"]):
                    run_one(batch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(policy["repeats"]):
                    gpu_exclusive()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    run_one(batch)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                free, total = torch.cuda.mem_get_info()
                reserved = torch.cuda.memory_reserved()
                peak_reserved = torch.cuda.max_memory_reserved()
                # Subtract allocations not owned by this allocator/context too.
                margin = max(0, int(free - max(0, peak_reserved - reserved)))
                median = statistics.median(times)
                # Numerical validation is OUTSIDE the timing scope. Every batch,
                # not just the winner, is compared to the same old float32 mean.
                if batch == "old_batch1":
                    eq = compare_logits(reference_mean, reference_mean, gt,
                                        thresholds["disagreement"], thresholds["logit_error"])
                else:
                    with torch.inference_mode():
                        eq_views = engine.forward_views(model, px, batch)
                        candidate_mean = eq_views.mean(0).permute(1, 2, 0).float().cpu().numpy()
                    eq = compare_logits(reference_mean, candidate_mean, gt,
                                        thresholds["disagreement"], thresholds["logit_error"])
                    del eq_views, candidate_mean
                records.append({"view_batch_size": batch, "status": eq["status"], "seconds": times,
                                "equivalence_vs_old_batch1": eq,
                                "median_seconds": median, "stable": (max(times)-min(times))/median <= policy["max_relative_range"],
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_reserved_bytes": peak_reserved, "total_bytes": total, "margin_bytes": margin})
            except torch.cuda.OutOfMemoryError:
                records.append({"view_batch_size": batch, "status": "OOM", "stable": False,
                                "fallback": "Try other registered view batch only; no crop/model/precision change"})
                gc.collect()
                torch.cuda.empty_cache()
            records[-1]["sample_window_unix"] = [batch_started, time.time()]
    utilization = sampler.report()
    for row in records:
        start, end = row["sample_window_unix"]
        samples = [s for s in utilization["samples"] if start <= s["unix"] <= end]
        row["utilization"] = {"samples": samples,
                              "mean_percent": statistics.mean(s["utilization_percent"] for s in samples) if samples else None,
                              "max_percent": max((s["utilization_percent"] for s in samples), default=None),
                              "peak_used_mib": max((s["used_mib"] for s in samples), default=None)}
    gpu_exclusive()
    candidates = records[1:]
    selected = None
    reason = None
    try:
        require(records[0]["status"] == "PASS" and records[0]["stable"], "Old batch1 reference unavailable/unstable")
        selected = choose_batch(candidates)
    except Blocked as e:
        reason = str(e)
    for row in candidates:
        if row["status"] == "PASS" and records[0]["status"] == "PASS":
            row["speedup_vs_old_batch1"] = records[0]["median_seconds"] / row["median_seconds"]
    return write_artifact(args.out_dir / OUTPUTS["benchmark"], dict(ident,
        status="PASS" if selected is not None else "FAIL", gate="benchmark", artifact_sha256=bindings,
        selected_view_batch_size=selected, reason=reason, old_batch1_reference=records[0], batches=candidates,
        image_stem=sample["samples"][0]["stem"], policy=policy, utilization=utilization,
        timing_scope="same model/image; 8-view forward+inverse+all11 scheme aggregation/prediction; excludes disk and image decode",
        precision="float32, no autocast; backend defaults unchanged", backend=backend,
        utilization_scope="Per-batch sample windows include warmup, repeats and untimed equivalence validation; timing excludes equivalence",
        fallback="Only registered view batch changes"))


def correctness(args):
    engine, model, ident, sample, marker, bindings, batch = gpu_preflight(args)
    import torch
    import tta_anatomy as old
    thresholds = sample["thresholds"]
    reports = []
    n = sample["correctness_n"]
    require(n in (2, 3, 4), "Correctness must use frozen 2-4 images")
    for item in sample["samples"][:n]:
        gpu_exclusive()
        px, gt = image_tensor(item)
        reference = old_reference(model, px)
        with torch.inference_mode():
            views = engine.forward_views(model, px, batch)
            require(tuple(views.shape) == (8, 8, 1024, 1024) and views.dtype == torch.float32 and views.is_cuda,
                    "V2 forward output contract failed")
            predictions = engine.scheme_predictions(views)
            require(set(predictions) == set(old.SCHEMES), "V2 prediction schemes incomplete")
            view_errors = [float(np.abs(reference[i] - views[i].permute(1, 2, 0).cpu().numpy()).max()) for i in range(8)]
            schemes = {}
            for name, score in engine.scheme_scores(views):
                require(name in old.SCHEMES and name not in schemes, "Unexpected/duplicate scheme score")
                members, mode = old.SCHEMES[name]
                ref = old.combine(reference, members, mode)
                got = score.permute(1, 2, 0).float().cpu().numpy()
                report = compare_logits(ref, got, gt, thresholds["disagreement"], thresholds["logit_error"])
                pred = predictions[name].cpu().numpy()
                require(pred.dtype == np.uint8 and np.array_equal(pred, got.argmax(-1)), "scheme_predictions differs from scores")
                if hasattr(engine, "confusion_tensor"):
                    actual_cm = engine.confusion_tensor(predictions[name], torch.as_tensor(gt, device=views.device)).cpu().numpy()
                    require(np.array_equal(actual_cm, np.asarray(report["confusion_candidate"])),
                            "V2 GPU confusion differs from independent CPU confusion")
                    report["gpu_confusion_matches_cpu"] = True
                report["score_kind"] = "probability" if mode == "prob" else "logit"
                schemes[name] = report
            require(set(schemes) == set(old.SCHEMES), "V2 score schemes incomplete")
        reports.append({"stem": item["stem"], "view_max_abs_errors": view_errors, "schemes": schemes,
                        "status": "PASS" if max(view_errors) <= thresholds["logit_error"] and
                        all(r["status"] == "PASS" for r in schemes.values()) else "FAIL"})
        del reference, views, predictions, px
    gpu_exclusive()
    return write_artifact(args.out_dir / OUTPUTS["correctness"], dict(ident,
        status="PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL", gate="correctness",
        artifact_sha256=bindings, view_batch_size=batch, n=len(reports), images=reports, thresholds=thresholds,
        reference="Literal audited old batch1 forward, inverse then resize, per-view CPU NumPy; independent float32",
        tolerance_rule="disagreement < threshold AND max residual <=1e-3; disagree only at measured near ties",
        meaningful_failure_policy="Fail closed; never silently relax threshold, crop, precision or architecture"))


def precision(args):
    engine, model, ident, sample, marker, bindings, batch = gpu_preflight(args)
    import torch
    thresholds = sample["thresholds"]
    report_groups = {kind: {name: [] for name in BIASES} for kind in ("old_cache_combined_drift", "v2_self_quantization")}
    for item in sample["samples"]:
        gpu_exclusive()
        stem = item["stem"]
        cache = args.cache_dir / f"{stem}.npy"
        require(sha256_file(cache) == marker["cache_inventory"][stem]["sha256"], f"Old cache changed: {stem}")
        cached = np.load(cache, allow_pickle=False)
        validate_cache_array(cached, stem)
        cached = cached.astype(np.float32)
        px, gt = image_tensor(item)
        with torch.inference_mode():
            views = engine.forward_views(model, px, batch)
            require(tuple(views.shape) == (8, 8, 1024, 1024) and views.dtype == torch.float32 and views.is_cuda,
                    "V2 forward contract failed")
            precise = views.mean(dim=0, dtype=torch.float32).permute(1, 2, 0).cpu().numpy()
        quantized = precise.astype(np.float16).astype(np.float32)
        for kind, candidate in (("old_cache_combined_drift", cached), ("v2_self_quantization", quantized)):
            for name, vec in BIASES.items():
                bias = np.asarray(vec, dtype=np.float32).reshape(1, 1, 8)
                r = compare_logits(precise + bias, candidate + bias, gt,
                                   thresholds["disagreement"], float("inf"))
                r["stem"] = stem
                # Precision pass is decision/IoU based; raw residual is reported,
                # not forced to correctness's unrelated 1e-3 score threshold.
                report_groups[kind][name].append(r)
        del views, px, precise, cached, quantized
    aggregates = {}
    passed = True
    for kind, bias_reports in report_groups.items():
        aggregates[kind] = {}
        for name, rows in bias_reports.items():
            cm_a = sum((np.asarray(r["confusion_reference"], dtype=np.int64) for r in rows), np.zeros((8, 8), np.int64))
            cm_b = sum((np.asarray(r["confusion_candidate"], dtype=np.int64) for r in rows), np.zeros((8, 8), np.int64))
            metrics = metric_delta(cm_a, cm_b)
            ok = (max(r["disagreement"] for r in rows) < thresholds["disagreement"] and
                  metrics["max_iou_delta"] < thresholds["aggregate_iou_delta"] and
                  metrics["miou_delta"] is not None and metrics["miou_delta"] < thresholds["aggregate_iou_delta"] and
                  all(r["meaningful_disagreements"] == 0 for r in rows))
            passed &= ok
            aggregates[kind][name] = {"status": "PASS" if ok else "FAIL", **metrics,
                "max_per_image_disagreement": max(r["disagreement"] for r in rows),
                "aggregate_pixel_disagreement": sum(r["disagreement_pixels"] for r in rows)/sum(r["pixels"] for r in rows),
                "mean_abs_logit_error": statistics.mean(r["mean_abs_logit_error"] for r in rows),
                "max_abs_logit_error": max(r["max_abs_logit_error"] for r in rows),
                "confusion_reference": cm_a.tolist(), "confusion_candidate": cm_b.tolist(), "images": rows}
    gpu_exclusive()
    return write_artifact(args.out_dir / OUTPUTS["precision"], dict(ident,
        status="PASS" if passed else "FAIL", gate="precision", artifact_sha256=bindings,
        precision_status="FP16_LOGIT_CACHE_PRECISION_PASS" if passed else "FP16_LOGIT_CACHE_PRECISION_FAILED",
        calibration_allowed=passed, phase_c_allowed_by_this_gate=True, requires_precise_strategy=not passed,
        failure_action=None if passed else "BLOCKED_PRECISE_CACHE_REQUIRED: do not calibrate old fp16 cache; explicitly approve a precise strategy",
        n=16, sample_stems=[s["stem"] for s in sample["samples"]], view_batch_size=batch,
        thresholds=thresholds, bias_vectors=BIASES, comparisons=aggregates,
        reference="Fresh GPU float32 TTA8 mean BEFORE any quantization",
        interpretation={"old_cache_combined_drift": "Fresh v2 vs existing old fp16 cache; includes batch/reduction drift AND quantization",
                        "v2_self_quantization": "Fresh v2 float32 vs its own float16 round trip; isolates quantization"},
        old_cache_modified=False, dataset_recomputation=False))


def selftest():
    import tempfile
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(16, 16, 8)).astype(np.float32)
    gt = rng.integers(0, 8, size=(16, 16), dtype=np.uint8)
    gt[0, 0] = 255
    same = compare_logits(ref, ref.copy(), gt)
    assert same["status"] == "PASS" and sum(map(sum, same["confusion_reference"])) == 255
    bad = ref.copy()
    bad[..., 7] += 100
    assert compare_logits(ref, bad, gt)["status"] == "FAIL"
    tie = np.zeros((1, 20000, 8), np.float32)
    tie[..., 0] = 1
    tie[..., 1] = 0.1
    tie[0, 0, 1] = np.nextafter(np.float32(1), np.float32(2))
    q = tie.astype(np.float16).astype(np.float32)
    near = compare_logits(tie, q, np.zeros((1, 20000), np.uint8))
    assert near["status"] == "PASS" and near["disagreement_pixels"] == 1
    assert near["near_tie_disagreements"] == 1 and near["meaningful_disagreements"] == 0
    dense = np.repeat(tie[:, :1], 2, axis=1)
    assert compare_logits(dense, dense.astype(np.float16).astype(np.float32), np.zeros((1, 2), np.uint8))["status"] == "FAIL"
    for broken in (np.full((2, 2, 8), np.nan, np.float16),
                   np.ones((2, 2, 8), np.float32)):
        try:
            validate_cache_array(broken, "fixture", (2, 2, 8))
        except Blocked:
            pass
        else:
            raise AssertionError("Bad cache accepted")
    validate_cache_array(np.zeros((2, 2, 8), np.float16), "zero-valued-valid", (2, 2, 8))
    validate_cache_array(np.ones((2, 2, 8), np.float16), "fixture", (2, 2, 8))
    records = [{"view_batch_size": b, "status": "PASS", "stable": True,
                "median_seconds": t, "margin_bytes": m} for b, t, m in
               ((1, 2.0, MIN_MARGIN), (2, 1.0, MIN_MARGIN), (4, 1.09, MIN_MARGIN), (8, 1.0, MIN_MARGIN-1))]
    assert choose_batch(records) == 4
    records[2]["stable"] = False
    assert choose_batch(records) == 2
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "artifact.json"
        assert write_artifact(p, {"status": "PASS", "x": 1}) == 0
        assert load_artifact(p)["x"] == 1
        try:
            write_artifact(p, {"status": "PASS"})
        except Blocked:
            pass
        else:
            raise AssertionError("Artifact overwrite accepted")
        obj = json.loads(p.read_text())
        obj["x"] = 2
        p.write_text(json.dumps(obj))
        try:
            load_artifact(p)
        except Blocked:
            pass
        else:
            raise AssertionError("Tampered artifact accepted")
    print("SELF_TEST_OK: independent fp32, near ties, hard drift, strict thresholds, cache validation, batch tie/margin, immutable SHA artifacts; CPU only")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = ap.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest", help="CPU numpy tests; no CUDA/SSH")
    for command in OUTPUTS:
        p = commands.add_parser(command)
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--val-list", type=Path, required=True)
        if command == "select":
            for flag in ("assignments", "image-dir", "mask-dir"):
                p.add_argument(f"--{flag}", type=Path, required=True)
            p.add_argument("--disagreement-threshold", type=float, default=1e-4)
            p.add_argument("--iou-threshold", type=float, default=1e-4)
            p.add_argument("--correctness-n", type=int, choices=(2, 3, 4), default=4)
            continue
        for flag in ("config", "checkpoint", "cache-dir", "anatomy"):
            p.add_argument(f"--{flag}", type=Path, required=True)
        p.add_argument("--old-pid", type=int, default=3803206)
        p.add_argument("--old-start-ticks", type=int, default=1542884730)
        if command != "integrity":
            for flag in ("sample", "integrity"):
                p.add_argument(f"--{flag}", type=Path, required=True)
        if command in ("benchmark", "correctness"):
            p.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT,
                           help="Frozen model checkpoint identity; explicit Stage66 hash supported")
            p.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG,
                           help="Frozen model config identity; oldB integrity stays Golden-bound")
        if command in ("correctness", "precision"):
            p.add_argument("--benchmark", type=Path, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return selftest()
        target = args.out_dir / OUTPUTS[args.command]
        require(not target.exists(), f"Output already exists; immutable artifact: {target}")
        if hasattr(args, "cache_dir"):
            require(args.anatomy.parent.resolve() not in (args.out_dir.resolve(), *args.out_dir.resolve().parents) and
                    args.cache_dir.resolve() not in (args.out_dir.resolve(), *args.out_dir.resolve().parents),
                    "New gate artifacts must not be written inside old anatomy/cache directory")
        return globals()[args.command](args)
    except Blocked as e:
        print(json.dumps({"status": "BLOCKED", "gate": args.command, "reason": str(e)}), file=sys.stderr)
        return 3
    except Exception as e:
        print(json.dumps({"status": "ERROR", "gate": args.command, "error": type(e).__name__, "reason": str(e)}), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
