#!/usr/bin/env python3
"""Run Paper I MSM diagnostics A–C on cylinder + ellipsoid (and optional paths).

Estimator: non-reversible row-stochastic MLE on MODE/basin sequences.
Shell rule R2: reference shell t*=8 N for cylinder/ellipsoid/block; sphere uses
declared D4 calibration family (see docs/CROSS_GEOMETRY_SPHERE_THRESHOLD_AMENDMENT.md).

Usage:
  PYTHONPATH=src python scripts/analyze_msm_diagnostics.py
  PYTHONPATH=src python scripts/analyze_msm_diagnostics.py --n-bootstrap 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cge.emergence.measure import load_trajectory_episodes  # noqa: E402
from cge.emergence.msm_diagnostics import run_msm_diagnostics  # noqa: E402

DEFAULT_COHORTS = {
    "cylinder": {
        "path": REPO / "results/phase1/landscape_analysis/landscape_trajectories.json",
        "state_order": ["grazing", "loaded"],
        "fx_threshold_N": 8.0,
    },
    "ellipsoid": {
        "path": REPO
        / "results/phase1/characterization_ellipsoid/full/landscape_trajectories.json",
        "state_order": ["dwell", "loaded"],
        "fx_threshold_N": 8.0,
    },
    "block": {
        "path": REPO
        / "results/phase1/characterization_block/full/landscape_trajectories.json",
        "state_order": ["grazing", "loaded"],
        "fx_threshold_N": 8.0,
    },
    "sphere": {
        "path": REPO
        / "results/phase1/cross_geometry_recovery/sphere/full/landscape_trajectories.json",
        "state_order": ["grazing", "loaded"],
        "fx_threshold_N": 1.0,
    },
}

OUT_DIR = REPO / "results/phase1/msm_diagnostics"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lag-ref", type=int, default=1)
    ap.add_argument(
        "--cohorts",
        nargs="+",
        default=["cylinder", "ellipsoid", "block", "sphere"],
        choices=list(DEFAULT_COHORTS.keys()),
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined: dict = {
        "shell_rule": "R2",
        "shell_rule_note": (
            "Reference shell S_ref: t*=8 N for cylinder/ellipsoid/block. "
            "Sphere uses declared D4 family (geometry-calibrated t* via preregistered "
            "select_threshold; see CROSS_GEOMETRY_SPHERE_THRESHOLD_AMENDMENT.md). "
            "Signature-P4 is primary; boundary-proximity P4 is diagnostic only."
        ),
        "estimator": {
            "name": "row_stochastic_MLE",
            "reversible": False,
            "lag_unit": "frame",
            "detailed_balance": False,
        },
        "cohorts": {},
    }

    for name in args.cohorts:
        cfg = DEFAULT_COHORTS[name]
        path: Path = cfg["path"]
        if not path.is_file():
            print(f"[skip] missing {path}", file=sys.stderr)
            continue
        _, episodes = load_trajectory_episodes(path)
        report = run_msm_diagnostics(
            episodes,
            cohort_name=name,
            source_path=str(path),
            state_order=cfg["state_order"],
            lag_ref=args.lag_ref,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.seed,
            fx_threshold_N=cfg["fx_threshold_N"],
            shell_rule="R2",
        )
        out_path = args.out_dir / f"{name}_msm_diagnostics.json"
        payload = report.to_dict()
        # Drop large sample lists if ever added; keep summary only
        out_path.write_text(json.dumps(payload, indent=2))
        combined["cohorts"][name] = payload
        print(report.summary)
        print(f"  wrote {out_path}")

    summary_path = args.out_dir / "msm_diagnostics_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
