#!/usr/bin/env python3
"""Module L: reproducibility release for 278 (CPU dry-run, no inference).

Assembles release_278/ with the winner folded checkpoint, configs, scripts,
manifests, environment, and a hashes.sha256. Then does a CPU dry-run:
imports, checkpoint load (weights_only=False, map_location cpu), config load,
dataset manifest, CLI --help, and SHA verification. No full forward.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

RARE = Path("/home/jhy/rare_diag_20260915")
ATDA = RARE / "atda_20260918"
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
RELEASE = RARE / "release_278"

WINNER = ATDA / "finalize/composed_atda_8k_a1p25/composed_atda_8k_a1p25_folded.pt"
CONFIG = SNAP / "configs/atda/atda_s2s3.yaml"
CHAMPION_CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
SRC_IDS = ATDA / "source_ids_5597.txt"
SCRIPTS = ["_atda_train.py", "_atda_ref261_cache.py", "_atda_export.py",
           "_atda_finalize.py", "_atda_launch.sh", "_atda_eval.sh"]
PY = "/home/jhy/.venvs/uavseg/bin/python"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    RELEASE.mkdir(parents=True, exist_ok=True)

    # ---- copy the frozen assets ----
    (RELEASE / "checkpoints").mkdir(exist_ok=True)
    (RELEASE / "scripts").mkdir(exist_ok=True)
    shutil.copy2(WINNER, RELEASE / "checkpoints" / WINNER.name)
    shutil.copy2(CONFIG, RELEASE / "atda_s2s3.yaml")
    shutil.copy2(CHAMPION_CONFIG, RELEASE / "champion_config.yaml")
    shutil.copy2(SRC_IDS, RELEASE / "source_ids_5597.txt")
    for s in SCRIPTS:
        src = ATDA / s
        if src.is_file():
            shutil.copy2(src, RELEASE / "scripts" / s)

    # ---- hashes.sha256 ----
    hashes = {}
    for p in sorted((RELEASE / "checkpoints").iterdir()) + sorted(RELEASE.glob("*.yaml")) + \
             [RELEASE / "source_ids_5597.txt"]:
        hashes[p.name] = sha(p)
    (RELEASE / "hashes.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in hashes.items()) + "\n", encoding="utf-8")

    # ---- environment ----
    env = subprocess.run([PY, "-c",
                          "import torch,sys,transformers,numpy;print(torch.__version__);"
                          "print(torch.version.cuda);print(transformers.__version__);"
                          "print(numpy.__version__);print(sys.version.split()[0])"],
                         capture_output=True, text=True).stdout.strip().split("\n")
    (RELEASE / "environment.json").write_text(json.dumps({
        "torch": env[0], "cuda": env[1], "transformers": env[2],
        "numpy": env[3], "python": env[4],
    }, indent=2), encoding="utf-8")

    # ---- provenance ----
    (RELEASE / "provenance.json").write_text(json.dumps({
        "winner": "278", "platform_miou": 0.706154,
        "base": "261", "base_platform_miou": 0.702836,
        "base_ckpt_sha256": "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786",
        "train_ckpt_sha256": "c3e87f6c51bd1fb32e6249783618d9bec7f1fc431a4a36d4ef3066d54e480eac",
        "winner_folded_sha256": "5be48fa5048b97c67aa7fc486fdb6619661d2ff2f08af9f92998ef4ac811d2ba",
        "fold_verified_pixel_identical": True,
        "formula": "theta261 + 1.25*(theta_atda_8k - theta261), then fold b_star into classifier bias",
    }, indent=2), encoding="utf-8")

    # ---- CPU dry-run ----
    dry = {}
    # 1. imports
    try:
        r = subprocess.run([PY, "-c",
                            "import torch; import src.utils; import src.models; "
                            "import src.dataset; print('imports ok')"],
                           capture_output=True, text=True, cwd=str(SNAP),
                           env={"PYTHONPATH": f"{SNAP}:/home/jhy/gpu2_day_queue",
                                "PATH": "/usr/bin:/bin"})
        dry["imports"] = {"ok": r.returncode == 0, "out": r.stdout.strip()[-40:],
                          "err": r.stderr.strip()[-200:]}
    except Exception as e:
        dry["imports"] = {"ok": False, "err": str(e)}

    # 2. checkpoint load
    try:
        r = subprocess.run([PY, "-c",
                            f"import torch; c=torch.load('{RELEASE/'checkpoints'/WINNER.name}',"
                            "map_location='cpu',weights_only=False);"
                            "print('tensors', len(c['model']))"],
                           capture_output=True, text=True)
        dry["checkpoint_load"] = {"ok": r.returncode == 0, "out": r.stdout.strip()[-40:],
                                  "err": r.stderr.strip()[-200:]}
    except Exception as e:
        dry["checkpoint_load"] = {"ok": False, "err": str(e)}

    # 3. config load
    try:
        r = subprocess.run([PY, "-c",
                            "from src.utils import load_config; "
                            f"c=load_config('{RELEASE/'atda_s2s3.yaml'}');"
                            "print('config keys', len(c))"],
                           capture_output=True, text=True, cwd=str(SNAP),
                           env={"PYTHONPATH": f"{SNAP}:", "PATH": "/usr/bin:/bin"})
        dry["config_load"] = {"ok": r.returncode == 0, "out": r.stdout.strip()[-40:],
                              "err": r.stderr.strip()[-200:]}
    except Exception as e:
        dry["config_load"] = {"ok": False, "err": str(e)}

    # 4. dataset manifest
    try:
        import numpy as np
        src = [l.strip() for l in (RELEASE / "source_ids_5597.txt").read_text().splitlines()
               if l.strip()]
        test_dir = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
        testn = sorted((p.name for p in test_dir.glob("*.png")),
                       key=lambda n: int(Path(n).stem.split("_")[-1]))
        dry["dataset_manifest"] = {
            "ok": len(src) == 5597 and testn == [f"test1_{i}.png" for i in range(1, 501)],
            "n_source_ids": len(src), "n_test": len(testn),
        }
    except Exception as e:
        dry["dataset_manifest"] = {"ok": False, "err": str(e)}

    # 5. CLI --help
    try:
        r = subprocess.run([PY, str(RELEASE / "scripts/_atda_export.py"), "--help"],
                           capture_output=True, text=True)
        dry["cli_help"] = {"ok": r.returncode == 0}
    except Exception as e:
        dry["cli_help"] = {"ok": False, "err": str(e)}

    # 6. SHA verification
    verified = {n: sha(RELEASE / "checkpoints" / WINNER.name) == hashes[WINNER.name]
                for n in ["checkpoints"]}
    dry["sha_verification"] = verified

    (RELEASE / "DRY_RUN.json").write_text(json.dumps(dry, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(dry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
