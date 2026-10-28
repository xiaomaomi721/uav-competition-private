#!/usr/bin/env python3
"""Write the terminal SHBR decision. This module never uploads a submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import shbr_common as common


def maybe_load(path: Path) -> dict | None:
    return common.load_json(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    output = args.run_root / "FINAL_DECISION.json"
    manual = args.run_root / "给队友看的最终结论.md"
    if output.exists() or manual.exists():
        raise SystemExit("final decision already exists")

    preflight = maybe_load(args.run_root / "00_preflight" / "ARTIFACT_RECEIPT.json")
    source = maybe_load(args.run_root / "02_gate" / "GATE_DECISION.json")
    target = maybe_load(args.run_root / "03_target_support" / "TARGET_SUPPORT.json")
    beta = maybe_load(args.run_root / "03_target_support" / "FIXED_BETA_RECEIPT.json")
    health = maybe_load(args.run_root / "04_training" / "SOURCE_HEALTH.json")
    package = maybe_load(args.run_root / "05_package" / "PACKAGE_MANIFEST.json")

    if args.status == "READY_FOR_MANUAL_PAIRED_SUBMISSION":
        expected = {
            "preflight": preflight and preflight.get("status") == "PASS",
            "source_gate": source and source.get("status") == "PROCEED_SOURCE_GATE",
            "target_support": target and target.get("status") == "PROCEED_TARGET_SUPPORT",
            "beta": beta and beta.get("status") == "PROCEED_TREATMENT_SCALE",
            "source_health": health and health.get("status") == "PASS_SOURCE_HEALTH",
            "package": package and package.get("status") == "READY_FOR_FINALIZATION",
        }
        failed = [key for key, value in expected.items() if not value]
        if failed:
            raise RuntimeError(f"cannot emit READY; prerequisite failures: {failed}")
        submission = {
            "manual_only": True,
            "slot_1": {
                "candidate": "RAS_KD_CONTROL_xlogy",
                "zip": preflight["artifacts"]["control_zip"]["path"],
                "zip_sha256": common.EXPECTED_CONTROL_ZIP_SHA,
            },
            "slot_2": {
                "candidate": "SHBR_TREAT_step2000",
                "zip": package["zip"],
                "zip_sha256": package["zip_sha256"],
            },
            "do_not_submit_one_without_the_other": True,
        }
    else:
        submission = {"manual_only": True, "submit_nothing": True}

    existing_receipts = []
    for path in sorted(args.run_root.rglob("*.json")):
        if path != output:
            existing_receipts.append(
                {"path": str(path.relative_to(args.run_root)), "sha256": common.sha256_file(path)}
            )
    decision = {
        "schema_version": 1,
        "status": args.status,
        "reason": args.reason,
        "warning": "Artifact correctness and source donor evidence do not establish the sign of test2 accuracy.",
        "submission": submission,
        "receipts": existing_receipts,
        "automatic_submission_performed": False,
        "platform_submission_performed": False,
    }
    common.write_json(output, decision)

    if args.status == "READY_FOR_MANUAL_PAIRED_SUBMISSION":
        text = f"""# SHBR 最终结论\n\n状态：**READY_FOR_MANUAL_PAIRED_SUBMISSION**\n\n这表示工程、source donor gate、test2 支持域和固定剂量门均已通过；**不表示已证明线上会涨分**。\n\n人工提交顺序（脚本未上传）：\n\n1. `RAS_KD_CONTROL_xlogy`\n   - ZIP：`{submission['slot_1']['zip']}`\n   - SHA256：`{submission['slot_1']['zip_sha256']}`\n2. `SHBR_TREAT_step2000`\n   - ZIP：`{submission['slot_2']['zip']}`\n   - SHA256：`{submission['slot_2']['zip_sha256']}`\n\n若当天不足两个槽，两个包都不交，保留到下一次可成对提交的日期。不得单交 TREAT。\n\n回分后只计算：`SHBR causal effect = TREAT - CONTROL`；若不为正，关闭本固定配方，不扫 beta/阈值。\n"""
    else:
        text = f"""# SHBR 最终结论\n\n状态：**{args.status}**\n\n原因：{args.reason or '见 FINAL_DECISION.json 和对应阶段 receipt。'}\n\n本链条已按预注册条件停止：**不训练后续阶段、不打提交包、不占平台名额**。不得用旧包或临时参数变体替代。\n"""
    manual.write_text(text, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
