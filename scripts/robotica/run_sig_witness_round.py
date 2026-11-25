#!/usr/bin/env python3
"""Same-function signature status for the witness round.

Sig is P1--P4 from cge.emergence.organization_signature at the frozen
t*=8 N lock. Residence uses stored basin labels. This does not retune
tau_ref, and it does not score OPCG. OPCG remains the separate quantitative
witness (PyBullet C1/C4 fail at the locked tuple).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from cge.emergence.organization_signature import audit_from_path  # noqa: E402

RAS = Path(__file__).resolve().parents[1]
OUT = RAS / "results" / "sig_witness_round.csv"

COHORTS = [
    ("cylinder_mujoco", "results/phase1/landscape_analysis/landscape_trajectories.json"),
    ("pybullet_cylinder", "results/phase1/cross_engine/pybullet_probe/landscape_trajectories.json"),
    ("ellipsoid", "results/phase1/cross_geometry_recovery/ellipsoid/full/landscape_trajectories.json"),
    ("box_pivot_wall", "results/stage1/box_pivot_wall_pilot/landscape_tau_ref30_thr5.0.json"),
    ("cylinder_horizontal", "results/stage1/cylinder_horizontal_pilot/landscape_trajectories.json"),
]


def main() -> None:
    rows = []
    for name, rel in COHORTS:
        path = REPO / rel
        rep = audit_from_path(path, cohort_name=name, fx_threshold=8.0)
        flags = {p.property_id: bool(p.pass_) for p in rep.properties}
        p1 = next(p for p in rep.properties if p.property_id == "P1").metrics
        row = {
            "cohort": name,
            "n_episodes": rep.n_episodes,
            "n_frames": rep.n_frames,
            "P1": flags["P1"],
            "P2": flags["P2"],
            "P3": flags["P3"],
            "P4": flags["P4"],
            "Sig": rep.all_properties_pass,
            "loaded_mean_tau": p1.get("loaded_mean_tau"),
            "grazing_mean_tau": p1.get("grazing_mean_tau"),
            "tail_separation_at_t4": p1.get("tail_separation_at_t4"),
            "lambda2": rep.lambda2_mode,
        }
        rows.append(row)
        print(
            f"{name:22} n={rep.n_episodes:4} Sig={rep.all_properties_pass} "
            f"P={flags} load={row['loaded_mean_tau']} graz={row['grazing_mean_tau']}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
