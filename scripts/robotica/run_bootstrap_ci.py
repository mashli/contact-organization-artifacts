#!/usr/bin/env python3
"""Episode-level bootstrap CIs for RAS Section 6 (B1), optimized.

Fit once on train; score every test episode once; bootstrap by resampling
episode-level (y, p) blocks (B=400, seed=0).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_SCRIPT = Path(__file__).resolve().parent
RAS = ROOT_SCRIPT.parent
sys.path.insert(0, str(ROOT_SCRIPT))

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
MAP = RAS / "configs" / "existing_trajectory_map.yaml"
PREREG = RAS / "configs" / "predictive_prereg.yaml"
SCORER = RAS / "configs" / "predictive_scorer.yaml"
OUT = RAS / "results" / "predictive_bootstrap_ci.csv"

B = 400
BOOT_SEED = 0
PRIMARY_REPS = [
    "scalar_fx",
    "qualified_organization",
    "two_state_hmm",
    "force_aware_clustering",
    "full_observation",
]


def episode_blocks(
    episodes, te_idx, task, thr, margin, horizon, rep, fitted, readout
):
    """Return list of (y_ep, p_ep) arrays, one per test episode."""
    blocks = []
    for ei in te_idx:
        pairs = collect_frame_task(
            episodes,
            np.asarray([ei], dtype=int),
            threshold=thr,
            margin=margin,
            horizon=horizon,
            task=task,
        )
        if not pairs:
            continue
        y = build_targets(
            episodes, pairs, task=task, threshold=thr, margin=margin, horizon=horizon
        )
        X = representation_matrix(episodes, pairs, rep, fitted, threshold=thr)
        p = readout.predict_proba(X)
        blocks.append((y, p))
    return blocks


def concat_blocks(blocks, indices):
    ys, ps = [], []
    for i in indices:
        y, p = blocks[i]
        ys.append(y)
        ps.append(p)
    if not ys:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(ys), np.concatenate(ps)


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
    geoms = list(prereg["datasets"]["geometries"])
    rng = np.random.default_rng(BOOT_SEED)

    rows = []
    for g in geoms:
        spec = data_map["geometries"][g]
        episodes = json.loads((REPO / spec["path"]).read_text())["episodes"]
        y_esc = [episode_escape(ep) for ep in episodes]
        tr, va, te = split_episode_indices(
            y_esc,
            seed=seed,
            train_frac=float(scorer["split"]["train"]),
            val_frac=float(scorer["split"]["val"]),
            test_frac=float(scorer["split"]["test"]),
        )
        fitted = fit_representations(episodes, tr, scorer_cfg=scorer, threshold=thr)
        te = np.asarray(te, dtype=int)
        n_te = len(te)
        print(f"[{g}] n_test_episodes={n_te}", flush=True)

        for task in tasks:
            pairs_tr = collect_frame_task(
                episodes, tr, threshold=thr, margin=margin, horizon=horizon, task=task
            )
            y_tr = build_targets(
                episodes, pairs_tr, task=task, threshold=thr, margin=margin, horizon=horizon
            )
            for rep in PRIMARY_REPS:
                X_tr = representation_matrix(episodes, pairs_tr, rep, fitted, threshold=thr)
                readout = LinearReadout(scorer["logistic"]).fit(X_tr, y_tr)
                if not readout.fitted:
                    continue
                blocks = episode_blocks(
                    episodes, te, task, thr, margin, horizon, rep, fitted, readout
                )
                if len(blocks) < 2:
                    continue
                y_te, p_te = concat_blocks(blocks, range(len(blocks)))
                point = metric_row(y=y_te, p=p_te, ece_bins=ece_bins)

                ba_boot, auc_boot = [], []
                n_blocks = len(blocks)
                for _ in range(B):
                    samp = rng.integers(0, n_blocks, size=n_blocks)
                    y_b, p_b = concat_blocks(blocks, samp)
                    if len(y_b) < 2 or len(set(y_b.tolist())) < 2:
                        continue
                    m = metric_row(y=y_b, p=p_b, ece_bins=ece_bins)
                    if np.isfinite(m["balanced_accuracy"]):
                        ba_boot.append(m["balanced_accuracy"])
                    if np.isfinite(m["auc"]):
                        auc_boot.append(m["auc"])

                def ci(xs):
                    if len(xs) < 10:
                        return (float("nan"), float("nan"), 0)
                    a = np.asarray(xs, dtype=float)
                    return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)), len(a))

                ba_lo, ba_hi, n_ba = ci(ba_boot)
                auc_lo, auc_hi, n_auc = ci(auc_boot)
                rows.append(
                    {
                        "geometry": g,
                        "task": task,
                        "representation": rep,
                        "n_test_episodes": int(n_te),
                        "n_samples": point["n_samples"],
                        "n_positive": point["n_positive"],
                        "positive_rate": point["n_positive"] / max(point["n_samples"], 1),
                        "balanced_accuracy": point["balanced_accuracy"],
                        "ba_ci_lo": ba_lo,
                        "ba_ci_hi": ba_hi,
                        "auc": point["auc"],
                        "auc_ci_lo": auc_lo,
                        "auc_ci_hi": auc_hi,
                        "n_boot_ba": n_ba,
                        "n_boot_auc": n_auc,
                        "B": B,
                        "resample_unit": "episode",
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"n_rows": len(rows), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
