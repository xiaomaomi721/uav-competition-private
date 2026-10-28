#!/usr/bin/env python3
"""Independent audit for MIC FIXED submission ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_zip(path: Path, expected_count: int = 1300) -> dict:
    expected = [f"test2_{index}.png" for index in range(1, expected_count + 1)]
    rows = []
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        bad_member = archive.testzip()
        if members != expected:
            missing = sorted(set(expected) - set(members))
            extra = sorted(set(members) - set(expected))
            raise RuntimeError(f"ZIP member order/identity mismatch: missing={missing[:5]} extra={extra[:5]}")
        if len(members) != len(set(members)):
            raise RuntimeError("duplicate ZIP members")
        for name in members:
            if "/" in name or "\\" in name:
                raise RuntimeError(f"nested ZIP member: {name}")
            payload = archive.read(name)
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                array = np.asarray(image)
                if image.mode != "L" or image.size != (1024, 1024) or array.dtype != np.uint8:
                    raise RuntimeError(f"bad PNG contract: {name} mode={image.mode} size={image.size} dtype={array.dtype}")
                minimum = int(array.min())
                maximum = int(array.max())
                if minimum < 1 or maximum > 8:
                    raise RuntimeError(f"label range violation: {name} {minimum}..{maximum}")
                rows.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "mode": image.mode, "shape": [1024, 1024], "label_min": minimum, "label_max": maximum})
    return {
        "zip_path": str(path),
        "zip_sha256": sha256_file(path),
        "zip_size": path.stat().st_size,
        "zip_test_bad_member": bad_member,
        "count": len(rows),
        "exact_order": True,
        "flat_root": True,
        "unique_names": True,
        "all_png_contracts_pass": True,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-count", type=int, default=1300)
    args = parser.parse_args()
    result = audit_zip(args.zip, args.expected_count)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
