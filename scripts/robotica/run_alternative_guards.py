#!/usr/bin/env python3
"""Alternative-guard boundary audit.

Two questions, kept separate:
1. Self-aligned: each scalar defines its own cell, threshold, near-guard
   margin, and boundary event. Score (cell, signed distance) on that event.
2. Cross: the same (cell, signed distance) predicting the declared f_x
   boundary. A win for f_x on (2) is alignment with an f_x-defined label.

Thresholds for non-f_x channels match the train-set high-cell fraction of
|f_x| >= 8 N. Margins match the train-set near-guard fraction of |f_x-8|<=2 N.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_predictive_readout import (  # noqa: E402
    LinearReadout,
    episode_escape,
    load_yaml,
    metric_row,
    split_episode_indices,
)

REPO = Path(__file__).resolve().parents[4]
RAS = ROOT.parent
MAP = RAS / "configs" / "existing_trajectory_map.yaml"
SCORER = RAS / "configs" / "predictive_scorer.yaml"
OUT = RAS / "results" / "alternative_guard_boundary.csv"

CHANNELS = ("fx", "fy", "fz", "tau_z", "fxy")


def channel_value(frame: dict, name: str) -> float:
    feat = frame.get("features") or {}
    if name == "fx":
        if frame.get("fx") is not None:
            return abs(float(frame["fx"]))
        return abs(float(feat.get("resultant_force_f_x", np.nan)))
    key = {
        "fy": "resultant_force_f_y",
        "fz": "resultant_force_f_z",
        "tau_z": "resultant_torque_tau_z",
    }.get(name)
    if name == "fxy":
        return float(np.hypot(feat.get("resultant_force_f_x", np.nan), feat.get("resultant_force_f_y", np.nan)))
    return abs(float(feat.get(key, np.nan)))


def episode_series(ep: dict, name: str) -> np.ndarray:
    return np.asarray([channel_value(fr, name) for fr in ep["frames"]], dtype=float)


def match_margin(values: np.ndarray, thr: float, target_frac: float) -> float:
    d = np.abs(values - thr)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    return float(np.quantile(d, np.clip(target_frac, 0.0, 1.0)))


def main() -> None:
    data_map = load_yaml(MAP)
    scorer = load_yaml(SCORER)
    fx_thr = float(scorer["fx_threshold"])
    fx_margin = float(scorer["boundary_margin"])
    seed = int(scorer["seed"])
    rows = []
    for geo in ("cylinder", "ellipsoid", "block"):
        spec = data_map["geometries"][geo]
        episodes = json.loads((REPO / spec["path"]).read_text())["episodes"]
        frame0 = episodes[0]["frames"][0]
        if not isinstance(frame0, dict) or not np.isfinite(channel_value(frame0, "fy")):
            print(f"[{geo}] missing fy/torque features; skip")
            continue
        y_esc = [episode_escape(ep) for ep in episodes]
        tr, _, te = split_episode_indices(
            y_esc,
            seed=seed,
            train_frac=float(scorer["split"]["train"]),
            val_frac=float(scorer["split"]["val"]),
            test_frac=float(scorer["split"]["test"]),
        )
        train_fx = np.concatenate([episode_series(episodes[i], "fx") for i in tr])
        train_fx = train_fx[np.isfinite(train_fx)]
        occ = float(np.mean(train_fx >= fx_thr))
        near = float(np.mean(np.abs(train_fx - fx_thr) <= fx_margin))
        series = {}
        thr = {}
        margin = {}
        for name in CHANNELS:
            vals = [episode_series(ep, name) for ep in episodes]
            if not all(np.isfinite(v).all() for v in vals):
                print(f"[{geo}] {name} non-finite; skip channel")
                continue
            series[name] = vals
            train_v = np.concatenate([vals[i] for i in tr])
            if name == "fx":
                thr[name] = fx_thr
                margin[name] = fx_margin
            else:
                thr[name] = float(np.quantile(train_v, 1.0 - occ))
                margin[name] = match_margin(train_v, thr[name], near)

        def pairs(idx, src, tgt):
            xs, ys = [], []
            for ei in idx:
                s = series[src][ei]
                g = series[tgt][ei]
                cell = s >= thr[src]
                for t in range(len(s) - 1):
                    dist = abs(float(s[t]) - thr[src])
                    # one-sided distance, same convention as the f_x organization
                    signed = dist
                    y = int(cell[t] != cell[t + 1] and abs(float(g[t]) - thr[tgt]) <= margin[tgt])
                    # boundary of the TARGET channel uses the target's own cell
                    if src != tgt:
                        pass
                    xs.append([1.0 if cell[t] else 0.0, signed])
                    ys.append(y)
            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)

        def target_boundary(idx, tgt):
            """Boundary event of the target channel: its own cell switch near its guard."""
            ys_idx = []
            for ei in idx:
                g = series[tgt][ei]
                cell = g >= thr[tgt]
                for t in range(len(g) - 1):
                    ys_idx.append((ei, t, int(cell[t] != cell[t + 1] and abs(float(g[t]) - thr[tgt]) <= margin[tgt])))
            return ys_idx

        for src in series:
            for mode, tgt in (("self", src), ("fx_boundary", "fx")):
                if mode == "fx_boundary" and src == "fx":
                    mode = "self"
                    # f_x self is recorded once
                    if any(r["geometry"] == geo and r["source"] == "fx" and r["mode"] == "self" for r in rows):
                        continue
                y_index = target_boundary(range(len(episodes)), tgt if mode != "self" else src)
                # rebuild with source features aligned to those (ei,t)
                def xy(idx_set):
                    xs, ys = [], []
                    want = set(int(i) for i in idx_set)
                    for ei, t, y in y_index:
                        if ei not in want:
                            continue
                        s = series[src][ei]
                        cell = float(s[t]) >= thr[src]
                        signed = abs(float(s[t]) - thr[src])
                        xs.append([1.0 if cell else 0.0, signed])
                        ys.append(y)
                    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)

                Xtr, ytr = xy(tr)
                Xte, yte = xy(te)
                readout = LinearReadout(scorer["logistic"]).fit(Xtr, ytr)
                if not readout.fitted or len(yte) < 2 or len(set(yte.tolist())) < 2:
                    auc = float("nan")
                    n_pos = int(np.sum(yte == 1)) if len(yte) else 0
                else:
                    p = readout.predict_proba(Xte)
                    auc = metric_row(y=yte, p=p, ece_bins=int(scorer["ece_bins"]))["auc"]
                    n_pos = int(np.sum(yte == 1))
                row = {
                    "geometry": geo,
                    "source": src,
                    "mode": "self" if src == (tgt if mode != "self" else src) and mode != "fx_boundary" else mode,
                    "threshold": thr[src],
                    "margin": margin[src],
                    "test_auc": auc,
                    "n_test": int(len(yte)),
                    "n_positive": n_pos,
                }
                # fix mode label
                if mode == "self" or (src == "fx" and mode == "self"):
                    row["mode"] = "self"
                else:
                    row["mode"] = "predict_fx_boundary"
                rows.append(row)
                print(
                    f"{geo:10} {src:6} {row['mode']:22} auc={auc if np.isfinite(auc) else float('nan'):.3f} "
                    f"pos={n_pos} thr={thr[src]:.3g} m={margin[src]:.3g}",
                    flush=True,
                )

    # de-duplicate fx self if the loop double-wrote
    uniq = []
    seen = set()
    for r in rows:
        key = (r["geometry"], r["source"], r["mode"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)
    print(OUT)


if __name__ == "__main__":
    main()
