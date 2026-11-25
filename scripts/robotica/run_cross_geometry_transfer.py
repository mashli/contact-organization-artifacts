#!/usr/bin/env python3
"""Cross-geometry transfer audit for RAS Section 6 (A3).

Train readout on geometry A train split; evaluate on geometry B test split.
Uses the same frozen 60/20/20 episode splits and fx threshold as the main scorer.
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
    fit_representations,
    load_yaml,
    metric_row,
    representation_matrix,
    split_episode_indices,
    episode_escape,
)

REPO = Path(__file__).resolve().parents[4]
MAP = RAS / "configs" / "existing_trajectory_map.yaml"
PREREG = RAS / "configs" / "predictive_prereg.yaml"
SCORER = RAS / "configs" / "predictive_scorer.yaml"
OUT = RAS / "results" / "cross_geometry_transfer.csv"


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
    reps = [
        "scalar_fx",
        "scalar_fx_quadratic",
        "organization_cell_only",
        "qualified_organization",
    ]
    geoms = list(prereg["datasets"]["geometries"])

    loaded: dict = {}
    for g in geoms:
        spec = data_map["geometries"][g]
        path = REPO / spec["path"]
        episodes = json.loads(path.read_text())["episodes"]
        y_esc = [episode_escape(ep) for ep in episodes]
        tr, va, te = split_episode_indices(
            y_esc,
            seed=seed,
            train_frac=float(scorer["split"]["train"]),
            val_frac=float(scorer["split"]["val"]),
            test_frac=float(scorer["split"]["test"]),
        )
        loaded[g] = {"episodes": episodes, "train": tr, "val": va, "test": te, "role": spec["role"]}

    rows = []
    for src in geoms:
        fitted = fit_representations(
            loaded[src]["episodes"],
            loaded[src]["train"],
            scorer_cfg=scorer,
            threshold=thr,
        )
        for task in tasks:
            for rep in reps:
                pairs_tr = collect_frame_task(
                    loaded[src]["episodes"],
                    loaded[src]["train"],
                    threshold=thr,
                    margin=margin,
                    horizon=horizon,
                    task=task,
                )
                y_tr = build_targets(
                    loaded[src]["episodes"],
                    pairs_tr,
                    task=task,
                    threshold=thr,
                    margin=margin,
                    horizon=horizon,
                )
                X_tr = representation_matrix(
                    loaded[src]["episodes"], pairs_tr, rep, fitted, threshold=thr
                )
                readout = LinearReadout(scorer["logistic"]).fit(X_tr, y_tr)
                if readout.failure or not readout.fitted:
                    continue
                for dst in geoms:
                    pairs_te = collect_frame_task(
                        loaded[dst]["episodes"],
                        loaded[dst]["test"],
                        threshold=thr,
                        margin=margin,
                        horizon=horizon,
                        task=task,
                    )
                    y_te = build_targets(
                        loaded[dst]["episodes"],
                        pairs_te,
                        task=task,
                        threshold=thr,
                        margin=margin,
                        horizon=horizon,
                    )
                    X_te = representation_matrix(
                        loaded[dst]["episodes"], pairs_te, rep, fitted, threshold=thr
                    )
                    if X_te.shape[1] != X_tr.shape[1] or len(y_te) < 2:
                        continue
                    pred = readout.predict_proba(X_te)
                    metrics = metric_row(y=y_te, p=pred, ece_bins=ece_bins)
                    rows.append(
                        {
                            "train_geometry": src,
                            "test_geometry": dst,
                            "train_role": loaded[src]["role"],
                            "test_role": loaded[dst]["role"],
                            "task": task,
                            "representation": rep,
                            "n_test": int(len(y_te)),
                            "n_positive": int(y_te.sum()),
                            "positive_rate": float(y_te.mean()) if len(y_te) else float("nan"),
                            "balanced_accuracy": metrics.get("balanced_accuracy"),
                            "auc": metrics.get("auc"),
                            "same_geometry": src == dst,
                        }
                    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({"n_rows": len(rows), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
