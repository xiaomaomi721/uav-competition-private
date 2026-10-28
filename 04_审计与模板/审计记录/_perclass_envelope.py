#!/usr/bin/env python3
"""Cross-model per-class envelope from existing val metrics. CPU only, no GPU.

The open question for the 0.71 plan is whether the remaining gap is a
CONCENTRATED CLASS DEFICIT or a GLOBAL REPRESENTATION DEFICIT. That can be
partly answered from data already on disk: every formal run wrote per-class IoU
on the same 1399-image val split.

For each class c:
    I_c^max  = max over models of IoU_c          (proven achievable in-project)
    I_c^champ= the soup member's value            (as a stand-in for the champion)
    G_c      = I_c^max - I_c^champ                (proven class headroom)

Note the caveat this script prints explicitly: these are single-view ZERO-bias
val numbers, while the leaderboard operator is champion-bias on test1. The
envelope is indicative of what the architecture can represent, not a prediction
of the submission score.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

B = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
CLASSES = ["Background", "Building", "Road", "Water",
           "Barren", "Vegetation", "Agricultural", "Vehicle"]

# the five soup members (and their close relatives) plus key historical models
WANTED = {
    "s58": "stage58_w1_exp58_b3_cwfa_ce_regionC_seed42",
    "s64": "stage64_w1_exp64_b3_cwfa_wd0005_seed42",
    "s55b": "stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42",
    "stage23b": "stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42",
    "s62": "stage62_w1_resume55b_pseudo500_seed42",
    "stage65": "stage65_w1_p1_pseudolabel_round2_60ep_seed42",
    "stage66": "stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42",
    "regionB": "stage57_w1_exp57_b3_cwfa_ce_regionB_seed42",
    "barrenAgri": "stage59_w1_exp59_b3_cwfa_ce_regionC_barrenAgri_seed42",
    "wd005": "stage63_w1_exp63_b3_cwfa_wd005_seed42",
}


def read_iou(run_dir: Path):
    m = run_dir / "metrics.json"
    if m.is_file():
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            v = d.get("per_class_iou")
            if isinstance(v, list) and len(v) == 8:
                return [float(x) for x in v], d.get("mIoU")
        except Exception:
            pass
    c = run_dir / "per_class_iou.csv"
    if c.is_file():
        try:
            rows = list(csv.reader(c.open(encoding="utf-8")))
            vals = []
            for r in rows:
                for cell in r:
                    try:
                        vals.append(float(cell))
                    except ValueError:
                        pass
            if len(vals) >= 8:
                return vals[-8:], None
        except Exception:
            pass
    return None, None


def main() -> int:
    data = {}
    for name, run in WANTED.items():
        d = B / f"{'stage' + ''.join(ch for ch in run.split('_')[0] if ch.isdigit())}_runs" / run
        if not d.is_dir():
            d = B / run
        iou, miou = read_iou(d)
        if iou:
            data[name] = iou
    if not data:
        raise SystemExit("no per-class data found")

    print("=== per-class IoU across models (val, single-view, ZERO bias) ===")
    hdr = "%-12s" % "model" + "".join("%9s" % c[:8] for c in CLASSES)
    print(hdr)
    for name, iou in data.items():
        print("%-12s" % name + "".join("%9.4f" % v for v in iou))
    print()

    env = [max(data[n][i] for n in data) for i in range(8)]
    env_model = [max(data, key=lambda n: data[n][i]) for i in range(8)]
    print("=== envelope ===")
    print("%-12s" % "MAX" + "".join("%9.4f" % v for v in env))
    print("%-12s" % "held by" + "".join("%9s" % m[:8] for m in env_model))
    print()

    for champ in ("s58", "s64", "stage23b"):
        if champ not in data:
            continue
        print(f"=== proven class headroom G_c = MAX - {champ} ===")
        tot = 0.0
        for i, c in enumerate(CLASSES):
            g = env[i] - data[champ][i]
            tot += g
            print("  %-12s %8.4f -> %8.4f   G=%+.4f  (held by %s)"
                  % (c, data[champ][i], env[i], g, env_model[i]))
        print("  sum G = %+.4f class-IoU points  =>  %+.4fpp mIoU if all captured"
              % (tot, 100 * tot / 8))
        print()

    print("CAVEAT: val / single-view / ZERO bias. The submission operator is")
    print("champion-bias on test1. This is a representability envelope, not a")
    print("prediction of the leaderboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
