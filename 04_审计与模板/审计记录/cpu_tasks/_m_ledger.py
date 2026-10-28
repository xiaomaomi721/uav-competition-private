#!/usr/bin/env python3
"""Module M: automatic score ledger (single source of truth).

Every row is pinned by its zip SHA256 so future lookups never rely on the
filename. Covers the main lineage soupC -> 209 -> 237 -> 261 -> 278.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path("/home/jhy/rare_diag_20260915/SCORE_LEDGER")

ROWS = [
    {
        "submission_id": "150",
        "name": "soupC_agri2.1215",
        "score": 0.696865,
        "zip_sha256": "55ba2c6c9e623becf571d2806a5f78694fd3d587cd39cd251145c79a836d7a25",
        "checkpoint_sha256": None,
        "parent": "soupC (stage58+64 weighted)",
        "formula": "soupC + agri bias 2.1215",
        "layer_scope": "soup",
        "inference_operator": "Bstar (agri 2.1215)",
        "date": "2026-09-16",
    },
    {
        "submission_id": "209",
        "name": "N9_deep-0.1",
        "score": None,
        "score_note": "platform score not found in local/remote receipts; DO NOT guess. "
                      "checkpoint sha is authoritative, score needs the platform receipt",
        "zip_sha256": "c9d176d55296891b016f53f61be2f442df5d5f46c633a85ee14fb4e03e2dc4f2",
        "checkpoint_sha256": "39e8b59847957c26758067db7a044e2c09d15a929ce29986badeed5eec6d2af3",
        "parent": "soupC",
        "formula": "layerwise donor delta (N9_e050_dm010) applied to stage2",
        "layer_scope": "stage2",
        "inference_operator": "Bstar",
        "date": "2026-09-17",
    },
    {
        "submission_id": "237",
        "name": "S_semeda_M",
        "score": 0.702574,
        "zip_sha256": "7dd1ee920a42f725aae5a3bdfeaea97467bf940e515bc28f3c94018f55cccf7f",
        "checkpoint_sha256": "1485a66a14342932a17ae3afb4180449b1b8ef0679db5bc8af8aea42f37c1cea",
        "parent": "209",
        "formula": "SEMEDA matched (S_semeda_M) donor gate",
        "layer_scope": "full",
        "inference_operator": "Bstar",
        "date": "2026-09-17",
    },
    {
        "submission_id": "261",
        "name": "A_cleanSEM_M_k100_th209",
        "score": 0.702836,
        "zip_sha256": "b0b9e0b031b6b15866f3202df0fafc54809f037c22601e7f9fe929d841255e1f",
        "checkpoint_sha256": "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786",
        "parent": "209",
        "formula": "theta209 + 1.0*R_old-normalized clean SEMEDA stage2 delta",
        "layer_scope": "stage2",
        "inference_operator": "Bstar",
        "date": "2026-09-18",
    },
    {
        "submission_id": "278",
        "name": "atda_8k_a1p25",
        "score": 0.706154,
        "zip_sha256": "3daa0b4a64898ff6ac0e779344027321c2d719dde6c4ce6743874b04df5f591c",
        "checkpoint_sha256": "c3e87f6c51bd1fb32e6249783618d9bec7f1fc431a4a36d4ef3066d54e480eac",
        "parent": "261",
        "formula": "theta261 + 1.25*(theta_atda_8k - theta261)",
        "layer_scope": "stage2+stage3",
        "inference_operator": "Bstar",
        "date": "2026-09-18",
    },
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SCORE_LEDGER.json").write_text(
        json.dumps({"rows": ROWS}, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["submission_id", "name", "score", "zip_sha256", "checkpoint_sha256",
              "parent", "formula", "layer_scope", "inference_operator", "date"]
    with open(OUT / "SCORE_LEDGER.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in ROWS:
            w.writerow(r)

    print(json.dumps(ROWS, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
