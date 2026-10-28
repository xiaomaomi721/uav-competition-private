#!/usr/bin/env python3
"""Record the platform receipt for KD Run1 (indices 64-73) and print the Z/F/C/D table."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REC = Path("提交包/清单与记录")
MANIFEST = REC / "提交清单_KD_Run1.json"

# user-reported scores, in the order pasted from the platform UI
REPORTED = [0.681538, 0.691925, 0.677788, 0.690002, 0.680602,
            0.692202, 0.678168, 0.690212, 0.688171, 0.688084]

# frozen anchors
S62_ZERO, S62_CHAMP = 0.680973, 0.686610
STAGE23B_ZERO, STAGE23B_CHAMP = 0.678023, 0.692812
RECORD_COMPLIANT = 0.693096      # submission 13, Stage23 Agricultural -0.4
RECORD_ALLTIME = 0.695637        # submission 43, ensemble B+s62+s58 + champion
TARGET = 0.700000
TEACHER_ENSEMBLE_ZERO = 0.683237


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pkgs = manifest["packages"]
    if len(pkgs) != len(REPORTED):
        raise SystemExit(f"count mismatch: {len(pkgs)} packages vs {len(REPORTED)} scores")

    rows = []
    for pkg, score in zip(pkgs, REPORTED):
        pkg["platform_mIoU"] = score
        pkg["status"] = "已提交并评分"
        rows.append({
            "index": pkg["index"], "file": pkg["file"],
            "checkpoint": pkg["checkpoint"], "role": pkg["role"],
            "sha256": pkg["sha256"], "platform_mIoU": score,
            "Z": score if pkg["role"] == "零偏置" else None,
            "F": score if pkg["role"] == "CHAMPION向量" else None,
        })

    # pair zero/champion by checkpoint for the decision table
    table = []
    by_ckpt: dict[str, dict] = {}
    for pkg in pkgs:
        if pkg["role"] in ("零偏置", "CHAMPION向量"):
            by_ckpt.setdefault(pkg["checkpoint"], {})[pkg["role"]] = pkg
    for ckpt, pair in by_ckpt.items():
        Z = pair.get("零偏置", {}).get("platform_mIoU")
        F = pair.get("CHAMPION向量", {}).get("platform_mIoU")
        table.append({
            "checkpoint": ckpt, "Z": Z, "F": F,
            "delta_Z_pp": round((Z - S62_ZERO) * 100, 4),
            "C_pp": round((F - Z) * 100, 4) if (Z and F) else None,
            "D_pp": round((TARGET - F) * 100, 4) if F else None,
        })
    table.sort(key=lambda r: -(r["F"] or 0))

    # the two calibration-amplitude probes scored against s62 champion
    probes = [{"index": p["index"], "file": p["file"], "role": p["role"],
               "deltas": p.get("deltas"), "platform_mIoU": p["platform_mIoU"],
               "vs_s62_champion_pp": round((p["platform_mIoU"] - S62_CHAMP) * 100, 4)}
              for p in pkgs if p["role"] == "校准幅度探测"]

    receipt = {
        "schema": "PLATFORM_RECEIPT_KD_RUN1_20260915",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": "用户报告的 10 个平台分数",
        "binding": {
            "method": "positional, submission order 64-73",
            "confidence": "high",
            "evidence": "前 8 个分数呈 低-高-低-高 交替 "
                        "(0.6815/0.6919/0.6778/0.6900/0.6806/0.6922/0.6782/0.6902)，"
                        "与 64-73 的 {zero, champion} 交替结构一致；最后两个 0.6882/0.6881 "
                        "是 s62 校准幅度探针，量级与 s62 冠军 0.686610 自洽",
            "caveat": "用户未逐个点名包名。若粘贴顺序与提交顺序不同，本回执需重录",
        },
        "anchors": {
            "s62_zero": S62_ZERO, "s62_champion": S62_CHAMP,
            "stage23B_zero": STAGE23B_ZERO, "stage23B_champion": STAGE23B_CHAMP,
            "record_compliant_sub13": RECORD_COMPLIANT,
            "record_alltime_sub43_noncompliant": RECORD_ALLTIME,
            "teacher_ensemble_zero": TEACHER_ENSEMBLE_ZERO,
            "target": TARGET,
        },
        "rows": rows,
        "decision_table": table,
        "calibration_probes": probes,
    }
    out = REC / "PLATFORM_RECEIPT_KD_Run1_20260915.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    w = 34
    print(f"{'checkpoint':<{w}} {'Z':>9} {'F':>9} {'ΔZ(pp)':>8} {'C(pp)':>8} {'D(pp)':>8}")
    print("-" * 80)
    for r in table:
        print(f"{r['checkpoint']:<{w}} {r['Z']:>9.6f} {r['F']:>9.6f} "
              f"{r['delta_Z_pp']:>+8.4f} {r['C_pp']:>+8.4f} {r['D_pp']:>8.4f}")
    print("-" * 80)
    print(f"{'[ref] s62':<{w}} {S62_ZERO:>9.6f} {S62_CHAMP:>9.6f} "
          f"{(S62_ZERO-S62_ZERO)*100:>+8.4f} {(S62_CHAMP-S62_ZERO)*100:>+8.4f} "
          f"{(TARGET-S62_CHAMP)*100:>8.4f}")
    print(f"{'[ref] Stage23B':<{w}} {STAGE23B_ZERO:>9.6f} {STAGE23B_CHAMP:>9.6f} "
          f"{(STAGE23B_ZERO-S62_ZERO)*100:>+8.4f} {(STAGE23B_CHAMP-STAGE23B_ZERO)*100:>+8.4f} "
          f"{(TARGET-STAGE23B_CHAMP)*100:>8.4f}")
    print(f"\n合规历史最高 (13号)   {RECORD_COMPLIANT}   D={(TARGET-RECORD_COMPLIANT)*100:+.4f}pp")
    print(f"全时历史最高 (43号)   {RECORD_ALLTIME}   D={(TARGET-RECORD_ALLTIME)*100:+.4f}pp")
    print(f"教师集成零偏置        {TEACHER_ENSEMBLE_ZERO}")
    print(f"\n校准幅度探针 vs s62 冠军 {S62_CHAMP}:")
    for p in probes:
        print(f"  {p['index']}  {p['platform_mIoU']:.6f}  {p['vs_s62_champion_pp']:+.4f}pp  {p['deltas']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
