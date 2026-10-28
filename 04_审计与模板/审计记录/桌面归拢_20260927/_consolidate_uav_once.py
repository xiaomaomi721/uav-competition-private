from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

DOC_DESKTOP = Path(r"C:\Users\Lenovo\OneDrive\Documents\桌面")
ALT_DESKTOP = Path(r"C:\Users\Lenovo\OneDrive\Desktop")
ROOT = DOC_DESKTOP / "无人机"
AUDIT = ROOT / "审计记录" / "桌面归拢_20260927"
DOCS = ROOT / "文档归档" / "04_实验决策与成绩" / "20260926_20260927_实验报告与执行手册"
ORIG = ROOT / "scripts_20260927" / "desktop_originals"
B4AUD = ROOT / "审计记录" / "B4追溯审计_20260927" / "desktop_originals"
RAWDUP = ROOT / "归档_整理记录" / "桌面归拢_20260927" / "原始副本" / "Documents桌面"
B4PKG = ROOT / "提交包" / "批次_20260927_B4"
MICINV = ROOT / "提交包" / "对照与诊断_不提交" / "MIC_INVALID_20260927"
RULES = ROOT / "文档归档" / "01_赛题与规则"
A05AUD = ROOT / "审计记录" / "A05_WEIGHT_20260926"

for parent in [ROOT, ROOT / "审计记录", ROOT / "文档归档", ROOT / "提交包", ROOT / "归档_整理记录"]:
    if not parent.is_dir():
        raise SystemExit(f"expected parent missing: {parent}")
for directory in [AUDIT, DOCS, ORIG, B4AUD, RAWDUP, B4PKG, MICINV, RULES, A05AUD]:
    directory.mkdir(parents=True, exist_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(source, destination, kind, reason, canonical=None):
    return {
        "source": str(source),
        "destination": str(destination),
        "kind": kind,
        "reason": reason,
        "canonical_path": str(canonical) if canonical else None,
    }


moves = [
    entry(
        DOC_DESKTOP / "A05_WEIGHT_alpha05_test2.zip",
        RAWDUP / "A05_WEIGHT_alpha05_test2.zip",
        "duplicate_provenance",
        "Byte-identical loose copy; retain without deleting",
        ROOT / "提交包" / "批次_20260926_A05_D1Barren" / "A05_WEIGHT_alpha05_test2.zip",
    ),
    entry(
        DOC_DESKTOP / "A05_WEIGHT_执行报告_20260926.md",
        RAWDUP / "A05_WEIGHT_执行报告_20260926.md",
        "duplicate_provenance",
        "Byte-identical loose report; retain without deleting",
        ROOT / "A05_WEIGHT_执行报告_20260926.md",
    ),
    entry(
        DOC_DESKTOP / "B4_Capacity_实验报告_20260927.md",
        RAWDUP / "B4_Capacity_实验报告_20260927.md",
        "duplicate_provenance",
        "Byte-identical loose report; retain without deleting",
        ROOT / "B4_Capacity_实验报告_20260927.md",
    ),
]

for name in [
    "AIC2026_B3_A05残余D1成对正式实验执行手册_20260927.md",
    "AIC2026_B3_新方法研究与冻结执行手册_20260927.md",
    "AIC2026_B3新杠杆_TASTE执行手册_20260926.md",
    "TAST-E_PhaseC诊断无效_纠偏与CPU重跑指令_20260926.md",
    "无人机_当前最优方案全面诊断与突破执行手册_20260926.md",
    "无人机_当前最优方案诊断与突破执行手册_V2纠偏版_20260926.md",
    "给执行AI_B3补证后连续完成Formal成对实验_20260927.md",
    "给执行AI_B4追溯审计与ZERO单包_20260927.md",
    "给执行AI_B4_PhaseA审计补正清单_20260927.md",
    "给执行AI_B4_PhaseA最终一致性补正_20260927.md",
]:
    moves.append(entry(DOC_DESKTOP / name, DOCS / name, "historical_document", "UAV execution/research document archived as historical snapshot"))

moves.extend(
    [
        entry(
            DOC_DESKTOP / "MIC成对实验执行报告_20260927.md",
            DOCS / "MIC成对实验执行报告_20260927_原始版.md",
            "historical_report",
            "Original pre-forensic MIC report retained unchanged",
        ),
        entry(
            DOC_DESKTOP / "B4_STAGE56_6097_BEST53200_D4_ZERO_test2.zip",
            B4PKG / "B4_STAGE56_6097_BEST53200_D4_ZERO_test2.zip",
            "scored_package",
            "B4 scored package 68.194181",
        ),
        entry(
            DOC_DESKTOP / "MIC_TREAT_S20260927_step8000_D4_TTA8_Bstar_test2.zip",
            MICINV / "MIC_TREAT_S20260927_step8000_D4_TTA8_Bstar_test2_INVALID.zip",
            "invalid_scored_package",
            "Confirmed filename identity corruption and D4 defect; never resubmit",
        ),
    ]
)

for name in [
    "_a05_pair.py",
    "_b4_infer.py",
    "_check_circular.py",
    "_enum_pseudo.py",
    "_final_fix.py",
    "_fix_circular.py",
    "_gen_index.py",
    "_gen_manifests.py",
    "_mic_infer.py",
    "_mic_train.py",
    "_verify_ret.py",
]:
    moves.append(entry(ALT_DESKTOP / name, ORIG / name, "historical_script", "Desktop execution/audit script preserved byte-identically"))

for name in [
    "B4_B3_COMPARABILITY.json",
    "CHECKPOINT_AUDIT.json",
    "RETROSPECTIVE_EXECUTION_REPORT.md",
    "b4_audit_ARTIFACT_HASHES.json",
    "b4_audit_B4_B3_COMPARABILITY.json",
    "b4_audit_CHECKPOINT_AUDIT.json",
    "b4_audit_RETENTION_MANIFEST.json",
    "b4_audit_RETROSPECTIVE_EXECUTION_REPORT.md",
]:
    moves.append(entry(ALT_DESKTOP / "无人机" / name, B4AUD / name, "historical_audit", "B4 desktop audit artifact preserved"))

for name in ["A05_WEIGHT_执行报告_20260926.md", "B4_Capacity_实验报告_20260927.md"]:
    moves.append(entry(ROOT / name, DOCS / name, "project_report", "Move root report into dated experiment archive"))
for name in ["eval_proto_v2_1000.json", "eval_theta278_baseline.json"]:
    moves.append(entry(ROOT / name, A05AUD / name, "audit_record", "Move root evaluation audit into dated audit directory"))
for name in [
    "关于举办算法挑战赛道复赛的通知.pdf",
    "附件1：第八届AIC算法大赛算法挑战赛道复赛赛事日程表.pdf",
    "附件2：第八届AIC算法大赛算法挑战赛道复赛作品提交要求.pdf",
]:
    moves.append(entry(ROOT / name, RULES / name, "official_rule", "Move official competition document into rules archive"))

allow_sources = {Path(item["source"]) for item in moves}
untouched = []
for desktop in [DOC_DESKTOP, ALT_DESKTOP]:
    for path in desktop.iterdir():
        if path in allow_sources or path == ROOT or path == ALT_DESKTOP / "无人机":
            continue
        stat = path.stat()
        untouched.append(
            {
                "path": str(path),
                "is_dir": path.is_dir(),
                "size": None if path.is_dir() else stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

pre = []
errors = []
for index, move in enumerate(moves, 1):
    source = Path(move["source"])
    destination = Path(move["destination"])
    if not source.is_file():
        errors.append(f"missing source: {source}")
        continue
    if destination.exists():
        errors.append(f"destination exists: {destination}")
        continue
    stat = source.stat()
    pre.append(
        {
            **move,
            "index": index,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha(source),
            "status": "PLANNED",
        }
    )
if errors:
    raise SystemExit("\n".join(errors))

created = datetime.now(timezone.utc).isoformat()
meta = {
    "created_at_utc": created,
    "policy": {"delete": False, "overwrite": False, "auto_rename": False, "allowlist_only": True},
    "count": len(pre),
    "total_bytes": sum(item["size"] for item in pre),
}
(AUDIT / "MOVE_PLAN.json").write_text(json.dumps({**meta, "moves": pre}, ensure_ascii=False, indent=2), encoding="utf-8")
(AUDIT / "PRE_MOVE_HASHES.json").write_text(
    json.dumps(
        {**meta, "files": [{"source": item["source"], "size": item["size"], "sha256": item["sha256"]} for item in pre]},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
(AUDIT / "UNTOUCHED_ITEMS.json").write_text(json.dumps({"captured_at_utc": created, "items": untouched}, ensure_ascii=False, indent=2), encoding="utf-8")

receipts = []
for record in pre:
    source = Path(record["source"])
    destination = Path(record["destination"])
    if sha(source) != record["sha256"] or source.stat().st_size != record["size"]:
        raise SystemExit(f"source changed before move: {source}")
    if destination.exists():
        raise SystemExit(f"late destination collision: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    post_sha = sha(destination)
    okay = post_sha == record["sha256"] and destination.stat().st_size == record["size"] and not source.exists()
    receipts.append(
        {
            **record,
            "destination_sha256": post_sha,
            "destination_size": destination.stat().st_size,
            "source_absent_after_move": not source.exists(),
            "status": "MOVED_VERIFIED" if okay else "VERIFY_FAILED",
        }
    )
    if not okay:
        raise SystemExit(f"post-move verify failed: {source} -> {destination}")

untouched_bad = []
for item in untouched:
    path = Path(item["path"])
    if not path.exists():
        untouched_bad.append({"path": str(path), "error": "missing"})
        continue
    stat = path.stat()
    current = {"is_dir": path.is_dir(), "size": None if path.is_dir() else stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if any(current[key] != item[key] for key in current):
        untouched_bad.append({"path": str(path), "before": item, "after": current})

finished = datetime.now(timezone.utc).isoformat()
(AUDIT / "MOVE_RECEIPT.json").write_text(
    json.dumps(
        {
            "completed_at_utc": finished,
            "count": len(receipts),
            "total_bytes": sum(item["size"] for item in receipts),
            "all_verified": all(item["status"] == "MOVED_VERIFIED" for item in receipts),
            "untouched_check_pass": not untouched_bad,
            "untouched_violations": untouched_bad,
            "moves": receipts,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
(AUDIT / "POST_MOVE_HASHES.json").write_text(
    json.dumps(
        {
            "completed_at_utc": finished,
            "files": [
                {"destination": item["destination"], "size": item["destination_size"], "sha256": item["destination_sha256"]}
                for item in receipts
            ],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "moved_count": len(receipts),
            "total_bytes": sum(item["size"] for item in receipts),
            "all_verified": True,
            "untouched_check_pass": not untouched_bad,
            "audit_dir": str(AUDIT),
        },
        ensure_ascii=False,
        indent=2,
    )
)
)
