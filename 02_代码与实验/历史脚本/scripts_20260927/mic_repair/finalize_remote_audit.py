from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image

ROOT = Path("/home/jhy/rare_diag_20260915/a05_micmask_20260927/09_mic_infer_repair_20260927")
OUT = Path("/home/jhy/rare_diag_20260915/a05_micmask_20260927/10_infer_treat_fixed_20260927")
FIXED = OUT / "MIC_TREAT_S20260927_step8000_D4_TTA8_Bstar_FIXED_test2.zip"
A05 = Path("/home/jhy/rare_diag_20260915/breakthrough_v2_20260926/09_candidates/A05_WEIGHT_alpha05/A05_WEIGHT_alpha05_test2.zip")
OLD = Path("/home/jhy/rare_diag_20260915/a05_micmask_20260927/08_infer_treat/MIC_TREAT_S20260927_step8000_D4_TTA8_Bstar_test2.zip")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array(payload):
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image, dtype=np.uint8)


names = [f"test2_{index}.png" for index in range(1, 1301)]
class_a05 = np.zeros(9, np.int64)
class_fixed = np.zeros(9, np.int64)
flips = []
old_virtual = []
with zipfile.ZipFile(A05) as a05_zip, zipfile.ZipFile(FIXED) as fixed_zip, zipfile.ZipFile(OLD) as old_zip:
    lexical = sorted(names)
    old_map = {lexical[index]: names[index] for index in range(1300)}
    for name in names:
        a05_value = array(a05_zip.read(name))
        fixed_value = array(fixed_zip.read(name))
        old_value = array(old_zip.read(old_map[name]))
        class_a05 += np.bincount(a05_value.ravel(), minlength=9)[:9]
        class_fixed += np.bincount(fixed_value.ravel(), minlength=9)[:9]
        flips.append(float((a05_value != fixed_value).mean()))
        old_virtual.append(float((old_value != fixed_value).mean()))

behavior = {
    "no_gt_descriptive_only": True,
    "fixed_zip_sha256": sha(FIXED),
    "a05_zip_sha256": sha(A05),
    "fixed_vs_a05_flip_mean_pct": 100 * float(np.mean(flips)),
    "fixed_vs_a05_flip_median_pct": 100 * float(np.median(flips)),
    "fixed_vs_a05_flip_max_pct": 100 * float(np.max(flips)),
    "fixed_class_share_pct": (100 * class_fixed[1:9] / class_fixed[1:9].sum()).tolist(),
    "a05_class_share_pct": (100 * class_a05[1:9] / class_a05[1:9].sum()).tolist(),
    "fixed_vs_virtual_reindexed_old_flip_mean_pct": 100 * float(np.mean(old_virtual)),
}
(OUT / "BEHAVIOR_VS_A05.json").write_text(json.dumps(behavior, indent=2), encoding="utf-8")
identity = json.loads((OUT / "INPUT_OUTPUT_IDENTITY.json").read_text())
audit = json.loads((OUT / "PACKAGE_AUDIT.json").read_text())
contract_sha = sha(ROOT / "ENTRY_CONTRACT.json")
lines = [
    "# MIC corrected inference execution report",
    "",
    "- training_performed: false",
    "- physical GPU: 2 / GPU-cacb7500-f2e9-c482-d85c-53c7f8d1def9 / 00000000:B1:00.0",
    "- checkpoint SHA256: e17fa56fbd809e9b8362f9d6ea3cf8723b9da3e1024ec3d8c1580825d1d455d0",
    "- entrypoint SHA256: 199b884468bef4259043939b72461761137c53532647969ac7f80626d8a33740",
    f"- entry contract SHA256: {contract_sha}",
    "- A05 anchor replay: 8/8 byte-exact",
    f"- input/output basename identity: {identity['count']}/1300, all={identity['all_basename_identity']}",
    f"- ZIP SHA256: {behavior['fixed_zip_sha256']}",
    f"- ZIP audit: count={audit['count']}, exact_order={audit['exact_order']}, flat={audit['flat_root']}, PNG PASS={audit['all_png_contracts_pass']}",
    "- test2_ground_truth_used: false",
    "- platform_submission_performed: false",
    "",
    "Behavior statistics are descriptive only, not accuracy evidence.",
    "",
]
(OUT / "EXECUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
artifacts = []
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name not in {"ARTIFACT_HASHES.json", "RETENTION_MANIFEST.json"}:
        artifacts.append({"path": str(path), "sha256": sha(path), "size": path.stat().st_size})
(OUT / "ARTIFACT_HASHES.json").write_text(json.dumps({"self_excluded": True, "files": artifacts}, indent=2), encoding="utf-8")
retained = [{"path": item["path"], "sha256": item["sha256"], "locked": True, "may_auto_delete": False} for item in artifacts]
retained.append({"path": str(OUT / "ARTIFACT_HASHES.json"), "sha256": sha(OUT / "ARTIFACT_HASHES.json"), "locked": True, "may_auto_delete": False})
(OUT / "RETENTION_MANIFEST.json").write_text(json.dumps({"self_excluded": True, "items": retained}, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "package_audit": {"count": audit["count"], "exact_order": audit["exact_order"], "png_pass": audit["all_png_contracts_pass"]},
            "identity": {"count": identity["count"], "all": identity["all_basename_identity"]},
            "behavior": behavior,
            "report_sha": sha(OUT / "EXECUTION_REPORT.md"),
            "artifact_hashes_sha": sha(OUT / "ARTIFACT_HASHES.json"),
            "retention_sha": sha(OUT / "RETENTION_MANIFEST.json"),
        },
        indent=2,
    )
)
