#!/usr/bin/env python3
"""Fixed MIC step8000 inference.

This is an additive replacement for the defective historical `_mic_infer.py`.
It performs inference only and contains no training or submission functionality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
COUNT = 1300
OUTPUT_SIZE = 1024
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
VIEWS = [(rotation, horizontal_flip) for rotation in range(4) for horizontal_flip in (False, True)]
EXPECTED_CHECKPOINT_SHA256 = "e17fa56fbd809e9b8362f9d6ea3cf8723b9da3e1024ec3d8c1580825d1d455d0"
EXPECTED_GPU_UUID = "GPU-cacb7500-f2e9-c482-d85c-53c7f8d1def9"
EXPECTED_GPU_PCI = "00000000:B1:00.0"
ZIP_NAME = "MIC_TREAT_S20260927_step8000_D4_TTA8_Bstar_FIXED_test2.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def exact_names(count: int = COUNT) -> list[str]:
    return [f"test2_{index}.png" for index in range(1, count + 1)]


def validate_input_names(root: Path, count: int = COUNT) -> list[str]:
    expected = exact_names(count)
    actual = [path.name for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".png"]
    if len(actual) != len(set(actual)):
        raise RuntimeError("duplicate input basenames")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra or len(actual) != count:
        raise RuntimeError(f"test2 identity mismatch: count={len(actual)} missing={missing[:5]} extra={extra[:5]}")
    for name in expected:
        if not (root / name).is_file():
            raise RuntimeError(f"missing exact-case input: {name}")
    return expected


def d4_forward(tensor, index: int):
    rotation, horizontal_flip = VIEWS[index]
    if rotation:
        tensor = tensor.rot90(rotation, (-2, -1))
    if horizontal_flip:
        tensor = tensor.flip((-1,))
    return tensor


def d4_inverse(tensor, index: int):
    rotation, horizontal_flip = VIEWS[index]
    if horizontal_flip:
        tensor = tensor.flip((-1,))
    if rotation:
        tensor = tensor.rot90(-rotation, (-2, -1))
    return tensor.contiguous()


def gpu_query() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": parts[0], "uuid": parts[1], "pci_bus_id": parts[2], "name": ",".join(parts[3:]).strip()})
    return {"command": command, "rows": rows}


def verify_gpu_environment(device: str) -> dict:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 2")
    if device != "cuda:0":
        raise RuntimeError("logical device must be cuda:0")
    query = gpu_query()
    row = next((item for item in query["rows"] if item["index"] == "2"), None)
    if row is None:
        raise RuntimeError("physical GPU2 missing from nvidia-smi")
    if row["uuid"] != EXPECTED_GPU_UUID or row["pci_bus_id"].upper() != EXPECTED_GPU_PCI.upper():
        raise RuntimeError(f"GPU2 identity mismatch: {row}")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    logical_name = torch.cuda.get_device_name(0)
    logical_uuid = None
    if hasattr(torch.cuda, "get_device_properties"):
        properties = torch.cuda.get_device_properties(0)
        logical_uuid = getattr(properties, "uuid", None)
    if logical_uuid is not None:
        normalized_logical_uuid = str(logical_uuid)
        if not normalized_logical_uuid.startswith("GPU-"):
            normalized_logical_uuid = "GPU-" + normalized_logical_uuid
        if normalized_logical_uuid != EXPECTED_GPU_UUID:
            raise RuntimeError(f"PyTorch UUID mismatch: {logical_uuid}")
    return {"environment": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2", "logical_device": device}, "nvidia_smi": query, "logical_name": logical_name, "logical_uuid": str(logical_uuid) if logical_uuid is not None else None}


def load_state(path: Path):
    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        state = obj.get("model", obj)
    else:
        state = obj
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint has no model state")
    return state


def validate_checkpoint(path: Path, expected_sha: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: {actual}")
    return actual


def audit_existing_predictions(
    predictions: Path,
    selected_names: list[str],
    selected_indices: list[int],
    input_root: Path,
) -> list[dict]:
    from PIL import Image

    actual = [path.name for path in predictions.iterdir() if path.is_file() and path.suffix.lower() == ".png"]
    if set(actual) != set(selected_names) or len(actual) != len(selected_names):
        raise RuntimeError("existing prediction basename set does not match selected inputs")
    rows = []
    for position, (dataset_index, name) in enumerate(zip(selected_indices, selected_names), 1):
        input_path = input_root / name
        output_path = predictions / name
        with Image.open(output_path) as image:
            image.load()
            array = np.asarray(image)
            if image.mode != "L" or image.size != (OUTPUT_SIZE, OUTPUT_SIZE) or array.dtype != np.uint8:
                raise RuntimeError(f"bad existing PNG contract: {name}")
            label_min = int(array.min())
            label_max = int(array.max())
            if label_min < 1 or label_max > NUM_CLASSES:
                raise RuntimeError(f"existing label range violation: {name} {label_min}..{label_max}")
        rows.append(
            {
                "position": position,
                "dataset_index": dataset_index,
                "input_absolute_path": str(input_path),
                "input_basename": name,
                "input_sha256": sha256_file(input_path),
                "output_basename": name,
                "output_sha256": sha256_file(output_path),
                "mode": "L",
                "shape": [OUTPUT_SIZE, OUTPUT_SIZE],
                "label_min": label_min,
                "label_max": label_max,
                "basename_identity": True,
            }
        )
    return rows


def finalize_package(
    output_dir: Path,
    selected_names: list[str],
    identity_rows: list[dict],
    zip_name: str,
    package_metadata: dict,
) -> dict:
    predictions = output_dir / "predictions"
    if not all(row["basename_identity"] for row in identity_rows):
        raise RuntimeError("not all output basenames match inputs")
    write_json(output_dir / "INPUT_OUTPUT_IDENTITY.json", {"count": len(identity_rows), "all_basename_identity": True, "rows": identity_rows})
    zip_path = output_dir / zip_name
    if zip_path.exists():
        raise RuntimeError(f"ZIP already exists: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in selected_names:
            archive.write(predictions / name, name)
    manifest = {
        **package_metadata,
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "training_performed": False,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    write_json(output_dir / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--test2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--zip-name", default=ZIP_NAME)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()

    # CPU-only recovery finalization never loads CUDA, but still pins every input identity.
    if args.finalize_existing:
        checkpoint_sha = validate_checkpoint(args.checkpoint, args.expected_checkpoint_sha256)
        config_sha = sha256_file(args.config)
        if config_sha != args.config_sha256:
            raise RuntimeError(f"config SHA mismatch: {config_sha}")
        names = validate_input_names(args.test2_root)
        selected = args.indices if args.indices else list(range(1, COUNT + 1))
        if not selected or len(selected) != len(set(selected)) or any(index < 1 or index > COUNT for index in selected):
            raise RuntimeError("indices must be unique values in 1..1300")
        selected_names = [names[index - 1] for index in selected]
        predictions = args.output_dir / "predictions"
        if not predictions.is_dir():
            raise RuntimeError(f"existing predictions directory missing: {predictions}")
        identity_rows = audit_existing_predictions(predictions, selected_names, selected, args.test2_root)
        manifest = finalize_package(
            args.output_dir,
            selected_names,
            identity_rows,
            args.zip_name,
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "config": str(args.config),
                "config_sha256": config_sha,
                "test2_root": str(args.test2_root),
                "selected_count": len(selected_names),
                "selected_names": selected_names,
                "protocol": "cpu_identity_recovery_of_existing_predictions",
                "views": None,
                "elapsed_sec": 0.0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "recovery_finalization": True,
            },
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return

    # All GPU policy and identity checks happen before output creation.
    gpu_identity = verify_gpu_environment(args.device)
    checkpoint_sha = validate_checkpoint(args.checkpoint, args.expected_checkpoint_sha256)
    config_sha = sha256_file(args.config)
    if config_sha != args.config_sha256:
        raise RuntimeError(f"config SHA mismatch: {config_sha}")
    names = validate_input_names(args.test2_root)
    selected = args.indices if args.indices else list(range(1, COUNT + 1))
    if not selected or len(selected) != len(set(selected)) or any(index < 1 or index > COUNT for index in selected):
        raise RuntimeError("indices must be unique values in 1..1300")
    selected_names = [names[index - 1] for index in selected]
    if args.output_dir.exists():
        raise RuntimeError(f"output directory already exists: {args.output_dir}")

    sys.path.insert(0, str(args.snapshot_root))
    sys.path.insert(0, "/home/jhy/gpu2_day_queue")
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.models import build_model
    from src.transforms import normalize_to_tensor
    from src.utils import load_config

    cfg = load_config(str(args.config))
    cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0, "norm_eps": 1e-12, "start_step": 1}
    cfg.pop("mixstyle", None)
    model = build_model(cfg)
    model.load_state_dict(load_state(args.checkpoint), strict=True)
    model = model.to(args.device).eval()
    bias = torch.from_numpy(BSTAR).to(args.device)[:, None, None]

    args.output_dir.mkdir(parents=False, exist_ok=False)
    predictions = args.output_dir / "predictions"
    predictions.mkdir()
    write_json(args.output_dir / "RUNTIME_IDENTITY.json", {"gpu": gpu_identity, "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha, "pid": os.getpid(), "cwd": os.getcwd(), "argv": sys.argv})

    identity_rows = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for position, name in enumerate(selected_names, 1):
            input_path = args.test2_root / name
            with Image.open(input_path) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            pixels = normalize_to_tensor(rgb).unsqueeze(0).to(args.device)
            accumulator = None
            for view_index in range(len(VIEWS)):
                view = d4_forward(pixels, view_index)
                with torch.amp.autocast("cuda", enabled=False):
                    logits = model(pixel_values=view).logits.float()
                canonical = d4_inverse(logits, view_index)[0].contiguous()
                accumulator = canonical if accumulator is None else accumulator + canonical
            native_mean = accumulator / float(len(VIEWS))
            deployed = F.interpolate((native_mean + bias)[None], size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)[0]
            prediction = deployed.argmax(0).byte().cpu().numpy() + 1
            output_path = predictions / input_path.name
            Image.fromarray(prediction, mode="L").save(output_path)
            if output_path.name != input_path.name:
                raise RuntimeError("input/output basename identity violation")
            identity_rows.append(
                {
                    "position": position,
                    "dataset_index": selected[position - 1],
                    "input_absolute_path": str(input_path),
                    "input_basename": input_path.name,
                    "input_sha256": sha256_file(input_path),
                    "output_basename": output_path.name,
                    "output_sha256": sha256_file(output_path),
                    "mode": "L",
                    "shape": [OUTPUT_SIZE, OUTPUT_SIZE],
                    "label_min": int(prediction.min()),
                    "label_max": int(prediction.max()),
                    "basename_identity": output_path.name == input_path.name,
                }
            )
            if position % 100 == 0 or position == len(selected_names):
                print(json.dumps({"completed": position, "total": len(selected_names), "name": name, "elapsed_sec": round(time.time() - started, 1)}), flush=True)

    package_manifest = finalize_package(
        args.output_dir,
        selected_names,
        identity_rows,
        args.zip_name,
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "config": str(args.config),
            "config_sha256": config_sha,
            "test2_root": str(args.test2_root),
            "selected_count": len(selected_names),
            "selected_names": selected_names,
            "protocol": "single_checkpoint_D4_TTA8_FP32_native_mean_Bstar_once_bilinear1024_argmax_plus1",
            "views": [{"rotation": rotation, "horizontal_flip": horizontal_flip} for rotation, horizontal_flip in VIEWS],
            "elapsed_sec": round(time.time() - started, 1),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "recovery_finalization": False,
        },
    )
    print(json.dumps(package_manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
