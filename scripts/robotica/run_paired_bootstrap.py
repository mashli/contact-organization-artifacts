#!/usr/bin/env python3
"""Paired episode bootstrap of organization minus ablation baselines.

Same train/test split as the predictive readout. Fit once; resample test
episodes with one shared index so the difference CI is paired.
Primary metric: balanced accuracy on next_basin, AUC on escape and boundary.
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
    build_targets,
    collect_frame_task,
    episode_escape,
    fit_representations,
    load_yaml,
    metric_row,
    representation_matrix,
    split_episode_indices,
)

REPO = Path(__file__).resolve().parents[4]
RAS = ROOT.parent
MAP = RAS / "configs" / "existing_trajectory_map.yaml"
PREREG = RAS / "configs" / "predictive_prereg.yaml"
SCORER = RAS / "configs" / "predictive_scorer.yaml"
OUT = RAS / "results" / "predictive_paired_delta_ci.csv"

B = 500
SEED = 0
REPS = [
    "qualified_organization",
    "scalar_fx",
    "scalar_fx_quadratic",
    "organization_cell_only",
]
BASELINES = ["scalar_fx", "scalar_fx_quadratic", "organization_cell_only"]


def primary(task: str, row: dict) -> float:
    if task == "next_basin":
        return float(row["balanced_accuracy"])
    return float(row["auc"])


def main() -> None:
    data_map = load_yaml(MAP)
    prereg = load_yaml(PREREG)
    scorer = load_yaml(SCORER)
    thr = float(scorer["fx_threshold"])
    margin = float(scorer["boundary_margin"])
    horizon = int(scorer["next_horizon"])
    seed = int(scorer["seed"])
    ece_bins = int(scorer["ece_bins"])
    tasks = [t["name"] for t in prereg["prediction_tasks"]]
    rng = np.random.default_rng(SEED)
    rows = []
    for g in prereg["datasets"]["geometries"]:
        episodes = json.loads((REPO / data_map["geometries"][g]["path"]).read_text())["episodes"]
        y_esc = [episode_escape(ep) for ep in episodes]
        tr, _, te = split_episode_indices(
            y_esc,
            seed=seed,
            train_frac=float(scorer["split"]["train"]),
            val_frac=float(scorer["split"]["val"]),
            test_frac=float(scorer["split"]["test"]),
        )
        fitted = fit_representations(episodes, tr, scorer_cfg=scorer, threshold=thr)
        te = np.asarray(te, dtype=int)
        print(f"[{g}] n_test={len(te)}", flush=True)
        for task in tasks:
            pairs_tr = collect_frame_task(
                episodes, tr, threshold=thr, margin=margin, horizon=horizon, task=task
            )
            y_tr = build_targets(
                episodes, pairs_tr, task=task, threshold=thr, margin=margin, horizon=horizon
            )
            blocks = {rep: [] for rep in REPS}
            readouts = {}
            ok = True
            for rep in REPS:
                X_tr = representation_matrix(episodes, pairs_tr, rep, fitted, threshold=thr)
                readouts[rep] = LinearReadout(scorer["logistic"]).fit(X_tr, y_tr)
                if not readouts[rep].fitted:
                    ok = False
            if not ok:
                print(f"  {task}: readout not fit")
                continue
            for ei in te:
                pairs = collect_frame_task(
                    episodes, np.asarray([ei]), threshold=thr, margin=margin, horizon=horizon, task=task
                )
                if not pairs:
                    for rep in REPS:
                        blocks[rep].append((np.asarray([], dtype=int), np.asarray([], dtype=float)))
                    continue
                y = build_targets(
                    episodes, pairs, task=task, threshold=thr, margin=margin, horizon=horizon
                )
                for rep in REPS:
                    X = representation_matrix(episodes, pairs, rep, fitted, threshold=thr)
                    blocks[rep].append((y, readouts[rep].predict_proba(X)))

            def score(rep, indices):
                ys, ps = [], []
                for i in indices:
                    y, p = blocks[rep][i]
                    if len(y):
                        ys.append(y)
                        ps.append(p)
                if not ys:
                    return float("nan")
                m = metric_row(y=np.concatenate(ys), p=np.concatenate(ps), ece_bins=ece_bins)
                return primary(task, m)

            n = len(te)
            point = {rep: score(rep, range(n)) for rep in REPS}
            deltas = {b: [] for b in BASELINES}
            for _ in range(B):
                samp = rng.integers(0, n, size=n)
                s_org = score("qualified_organization", samp)
                if not np.isfinite(s_org):
                    continue
                for b in BASELINES:
                    s_b = score(b, samp)
                    if np.isfinite(s_b):
                        deltas[b].append(s_org - s_b)
            for b in BASELINES:
                arr = np.asarray(deltas[b], dtype=float)
                if len(arr) < 10:
                    lo = hi = float("nan")
                else:
                    lo, hi = float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
                d = point["qualified_organization"] - point[b]
                rows.append(
                    {
                        "geometry": g,
                        "task": task,
                        "baseline": b,
                        "org": point["qualified_organization"],
                        "baseline_score": point[b],
                        "delta": d,
                        "delta_ci_lo": lo,
                        "delta_ci_hi": hi,
                        "ci_includes_zero": bool(np.isfinite(lo) and lo <= 0 <= hi),
                        "B": B,
                    }
                )
                print(
                    f"  {task:16} org-{b:24} d={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] zero={lo<=0<=hi}",
                    flush=True,
                )
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
