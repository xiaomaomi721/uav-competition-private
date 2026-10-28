#!/usr/bin/env python3
"""Build a RETROSPECTIVE manifest for submissions 54-63.

These ten packages were scored on the platform, but no contemporaneous
SHA-bound PLATFORM_RECEIPT was written at the time. Scores exist only in
线上成绩记录表.md. This script reconstructs the mapping so future provenance
checks do not depend on inference -- and it labels itself as a reconstruction,
NOT as the original receipt, to avoid corrupting provenance further.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REC = Path("提交包/清单与记录")
SOURCE_MANIFEST = REC / "提交清单_54_63.json"
SCORE_TABLE = Path("文档归档/04_实验决策与成绩/线上成绩记录表.md")

# scores transcribed from 线上成绩记录表.md, "2026-09-15 回填：54–63" section
SCORES = {
    54: 0.687420, 55: 0.685654, 56: 0.686940, 57: 0.686217,
    58: 0.686001, 59: 0.686893, 60: 0.686346, 61: 0.686748,
    62: 0.674232, 63: 0.689630,
}
BASE_S62_CHAMPION = 0.686610


def main() -> int:
    src = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    packages = {p["index"]: p for p in src["packages"]}

    missing = [i for i in SCORES if i not in packages]
    if missing:
        raise SystemExit(f"index missing from source manifest: {missing}")

    rows = []
    for idx in sorted(SCORES):
        p = packages[idx]
        score = SCORES[idx]
        rows.append({
            "index": idx,
            "file": p["file"],
            "source": p.get("source"),
            "content": p.get("content"),
            "why": p.get("why"),
            "sha256": p.get("sha256"),
            "bytes": p.get("bytes"),
            "platform_mIoU": score,
            "vs_s62_champion_pp": round((score - BASE_S62_CHAMPION) * 100, 4),
        })

    out = {
        "schema": "RETROSPECTIVE_SUBMISSION_MANIFEST_V1",
        "record_type": "RETROSPECTIVE_RECONSTRUCTION",
        "warning": (
            "这不是原始平台回执。这十个包在评分时没有写 contemporaneous "
            "SHA-bound PLATFORM_RECEIPT，分数仅存于线上成绩记录表.md。"
            "本文件是在 2026-09-15 事后把『现有包 SHA256』与『历史成绩』重建的映射，"
            "用于 provenance 核对。请勿当作当时就存在的原始回执引用。"
        ),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "score_source": str(SCORE_TABLE).replace("\\", "/"),
        "score_source_section": "## 2026-09-15 回填：54–63（s62 模型专属单轴探针 + Paramwise01）",
        "sha_source": str(SOURCE_MANIFEST).replace("\\", "/"),
        "binding_basis": "用户按 54→63 顺序粘贴回执；包 SHA256 来自提交前记录的清单",
        "confidence": "medium-high: 顺序由用户给出，SHA 在提交前记录，但两者未在评分当时交叉绑定",
        "baseline": {"s62_champion": BASE_S62_CHAMPION},
        "rows": rows,
    }
    dst = REC / "54-63_retrospective_manifest.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {dst}")
    print(f"{'idx':>4} {'file':<34} {'score':>9} {'vs s62+champ':>13}")
    for r in rows:
        print(f"{r['index']:>4} {str(r['file']):<34} {r['platform_mIoU']:>9.6f} "
              f"{r['vs_s62_champion_pp']:>+13.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
