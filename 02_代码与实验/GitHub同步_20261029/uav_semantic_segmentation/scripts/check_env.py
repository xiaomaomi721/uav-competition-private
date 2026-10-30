#!/usr/bin/env python3
"""Report the active immutable execution environment and CUDA readiness."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import environment_info, save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    info = environment_info()
    info["dependency_presence"] = {
        name: bool(importlib.util.find_spec(name))
        for name in ("torch", "transformers", "numpy", "PIL", "yaml")
    }
    print(info)
    if args.output:
        save_json(args.output.resolve(), info)
    missing = [name for name, present in info["dependency_presence"].items() if not present]
    if missing:
        raise RuntimeError(f"missing dependencies: {missing}")
    if args.require_cuda and not info["cuda_available"]:
        raise RuntimeError("CUDA is required but unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
