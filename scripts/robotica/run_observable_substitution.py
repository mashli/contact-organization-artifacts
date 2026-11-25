#!/usr/bin/env python3
"""Cylinder observable substitution using the same dwell definition as
scripts/analyze_guard_threshold_sensitivity.py.

Each scalar is cut at its own 60th percentile of absolute value so occupancy
is comparable. This is not an OPCG rerun.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _paths import REPO, load_json  # noqa: E402
OUT = REPO / "results" / "robotica" / "observable_substitution_cylinder.csv"

KEYS = [
    ("|f_x|", "resultant_force_f_x", "fx"),
    ("|f_y|", "resultant_force_f_y", None),
    ("|f_z|", "resultant_force_f_z", None),
    ("|tau_z|", "resultant_torque_tau_z", None),
    ("||f_xy||", "f_xy", None),
]


def values(ep, feat_key, fallback):
    xs = []
    for fr in ep["frames"]:
        feat = fr.get("features") or {}
        if feat_key == "f_xy":
            try:
                xs.append(float(np.hypot(feat["resultant_force_f_x"], feat["resultant_force_f_y"])))
                continue
            except (TypeError, ValueError, KeyError):
                xs.append(np.nan)
                continue
        v = feat.get(feat_key)
        if v is None and fallback:
            v = fr.get(fallback)
        try:
            xs.append(abs(float(v)))
        except (TypeError, ValueError):
            xs.append(np.nan)
    return np.asarray(xs, dtype=float)


def mean_dwell(mask: np.ndarray) -> tuple[float, float]:
    if len(mask) == 0:
        return 0.0, 0.0
    runs_hi, runs_lo = [], []
    cur = bool(mask[0])
    n = 1
    for b in mask[1:]:
        b = bool(b)
        if b == cur:
            n += 1
        else:
            (runs_hi if cur else runs_lo).append(n)
            cur, n = b, 1
    (runs_hi if cur else runs_lo).append(n)
    hi = float(np.mean(runs_hi)) if runs_hi else 0.0
    lo = float(np.mean(runs_lo)) if runs_lo else 0.0
    return hi, lo


def audit(episodes, feat_key, fallback, thr):
    highs, lows = [], []
    n = n_hi = n_sw = 0
    for ep in episodes:
        x = values(ep, feat_key, fallback)
        if not np.isfinite(x).all() or len(x) < 2:
            continue
        mask = x >= thr
        hi, lo = mean_dwell(mask)
        highs.append(hi)
        lows.append(lo)
        n += len(mask)
        n_hi += int(mask.sum())
        n_sw += int(np.sum(mask[1:] != mask[:-1]))
    mean_hi = float(np.mean(highs)) if highs else float("nan")
    mean_lo = float(np.mean(lows)) if lows else float("nan")
    return {
        "threshold": thr,
        "n_episodes": len(highs),
        "frac_high": n_hi / max(n, 1),
        "mean_high_dwell": mean_hi,
        "mean_low_dwell": mean_lo,
        "dwell_ratio": mean_hi / max(mean_lo, 0.01),
        "switch_rate": n_sw / max(n, 1),
        "P1_like": bool(mean_hi > mean_lo),
    }


def main():
    episodes = load_json("data/robotica/cylinder.json.gz")["episodes"]
    rows = []
    for name, key, fb in KEYS:
        all_x = np.concatenate([values(ep, key, fb) for ep in episodes])
        all_x = all_x[np.isfinite(all_x)]
        for q, rule in ((0.60, "own_p60"), (0.45, "own_p45")):
            thr = float(np.quantile(all_x, q))
            row = {"observable": name, "rule": rule}
            row.update(audit(episodes, key, fb, thr))
            rows.append(row)
    # reference: declared |f_x| guard at 8 N
    ref = {"observable": "|f_x|", "rule": "declared_8N"}
    ref.update(audit(episodes, "resultant_force_f_x", "fx", 8.0))
    rows.append(ref)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['observable']:8} {r['rule']:12} thr={r['threshold']:.3g} "
            f"frac={r['frac_high']:.3f} dwell={r['dwell_ratio']:.3f} "
            f"P1like={r['P1_like']} sw={r['switch_rate']:.4f} n={r['n_episodes']}"
        )


if __name__ == "__main__":
    main()
