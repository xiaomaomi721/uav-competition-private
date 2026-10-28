"""Module K: ATDA training-log analysis (parse the 8k log, no GPU).

Extracts per-25-step loss/LR/ramp/agreement/EMA-delta/grad-norm/AMP-scale traces
and analyses whether 4k->8k was still converging. Also locates the two AMP
overflow events (scale halving) and the surrounding steps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
LOG = ATDA / "run_s2s3_base261" / "atda.log"
OUT = ATDA / "cpu_log_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# step line pattern:
# [HH:MM:SS] step   NNNN src X hard Y cons Z loss W lr L ramp R w W emad E agree A% scale S |g| G
STEP_RE = re.compile(
    r"step\s+(\d+)\s+src\s+([\d.eE+-]+)\s+hard\s+([\d.eE+-]+)\s+cons\s+([\d.eE+-]+)"
    r"\s+loss\s+([\d.eE+-]+)\s+lr\s+([\d.eE+-]+)\s+ramp\s+([\d.]+)\s+w\s+([\d.]+)"
    r"\s+emad\s+([\d.eE+-]+)\s+agree\s+([\d.]+)%\s+scale\s+(\d+)\s+\|g\|\s+([\d.eE+-]+)"
)
SKIP_RE = re.compile(r"step (\d+): optimizer did NOT step -- (.*)")
NONFINITE_RE = re.compile(r"step (\d+): NONFINITE")


def main():
    rows = []
    skips = []
    scale_events = []
    prev_scale = None
    for line in LOG.read_text(encoding="utf-8").splitlines():
        m = STEP_RE.search(line)
        if m:
            g = m.groups()
            step = int(g[0])
            scale = int(g[10])
            if prev_scale is not None and scale < prev_scale:
                scale_events.append({"step": step, "from": prev_scale, "to": scale})
            prev_scale = scale
            rows.append({
                "step": step, "source": float(g[1]), "hard": float(g[2]),
                "cons": float(g[3]), "loss": float(g[4]), "lr": float(g[5]),
                "ramp": float(g[6]), "w": float(g[7]), "ema_delta": float(g[8]),
                "agree_pct": float(g[9]), "amp_scale": scale, "grad_norm": float(g[11]),
            })
        m = SKIP_RE.search(line)
        if m:
            skips.append({"step": int(m.group(1)), "reason": m.group(2)})
        m = NONFINITE_RE.search(line)
        if m:
            skips.append({"step": int(m.group(1)), "reason": "nonfinite"})

    # ---- convergence: 4k vs 8k windows ----
    def win(lo, hi):
        return [r for r in rows if lo <= r["step"] < hi]

    def stats(sub, key):
        v = [r[key] for r in sub]
        return {"mean": round(float(np.mean(v)), 6),
                "std": round(float(np.std(v)), 6),
                "last": round(float(v[-1]), 6) if v else None}

    windows = {
        "0-1k": win(0, 1000), "1k-2k": win(1000, 2000), "2k-4k": win(2000, 4000),
        "4k-6k": win(4000, 6000), "6k-8k": win(6000, 8000),
    }
    trace = {}
    for name, sub in windows.items():
        if not sub:
            continue
        trace[name] = {k: stats(sub, k) for k in
                       ("source", "hard", "cons", "loss", "agree_pct", "ema_delta", "grad_norm")}

    report = {
        "scope": "atda_training_log_analysis",
        "n_logged_steps": len(rows),
        "scale_halving_events": scale_events,
        "skip_events": skips,
        "window_stats": trace,
        "last_step": rows[-1] if rows else None,
        "convergence_4k_to_8k": {
            "source_loss_4k6k": trace.get("4k-6k", {}).get("source"),
            "source_loss_6k8k": trace.get("6k-8k", {}).get("source"),
            "hard_loss_4k6k": trace.get("4k-6k", {}).get("hard"),
            "hard_loss_6k8k": trace.get("6k-8k", {}).get("hard"),
            "still_converging": (
                trace.get("6k-8k", {}).get("loss", {}).get("mean", 1)
                < trace.get("4k-6k", {}).get("loss", {}).get("mean", 1)
            ),
        },
    }
    (OUT / "log_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    # CSV of every 25-step row
    with open(OUT / "trace.csv", "w") as f:
        f.write("step,source,hard,cons,loss,lr,ramp,w,ema_delta,agree_pct,amp_scale,grad_norm\n")
        for r in rows:
            f.write(f"{r['step']},{r['source']},{r['hard']},{r['cons']},{r['loss']},"
                    f"{r['lr']},{r['ramp']},{r['w']},{r['ema_delta']},{r['agree_pct']},"
                    f"{r['amp_scale']},{r['grad_norm']}\n")
    print(json.dumps({
        "n_logged_steps": len(rows),
        "scale_halving_events": scale_events,
        "skip_events": skips,
        "window_loss_mean": {k: v["loss"]["mean"] for k, v in trace.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
