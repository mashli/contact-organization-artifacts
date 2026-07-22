#!/usr/bin/env python3
"""Residence time distribution (RTD) analysis on landscape trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CGE_SRC = REPO / "src"
if str(CGE_SRC) not in sys.path:
    sys.path.insert(0, str(CGE_SRC))

from cge.emergence.residence import estimate_residence_times, load_trajectory_episodes  # noqa: E402


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Metastable residence time analysis")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/emergence_measure_baseline_PREREG.yaml",
    )
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--name", type=str, default="custom")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    if args.trajectory is not None:
        cohorts = [{"name": args.name, "path": str(args.trajectory)}]
    else:
        cohorts = cfg.get("cohorts", [])

    reports: dict[str, dict] = {}
    for c in cohorts:
        name = str(c["name"])
        path = resolve_path(c["path"])
        if not path.is_file():
            print(f"SKIP {name}: missing {path}", flush=True)
            continue
        _meta, eps = load_trajectory_episodes(path)
        rep = estimate_residence_times(
            eps,
            source_path=str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
        )
        reports[name] = rep.to_dict()
        loaded = rep.by_basin.get("loaded")
        grazing = rep.by_basin.get("grazing")
        l_mean = loaded.mean_run_length if loaded else float("nan")
        g_mean = grazing.mean_run_length if grazing else float("nan")
        p_l4 = (loaded.survival_tail.get("4") if loaded else None) or float("nan")
        p_g4 = (grazing.survival_tail.get("4") if grazing else None) or float("nan")
        print(
            f"{name}: τ_loaded={l_mean:.2f} τ_grazing={g_mean:.2f} "
            f"P(τ>4) L/G={p_l4:.2f}/{p_g4:.2f} tail_sep={rep.tail_separation_at_t4:.2f} "
            f"switches={rep.n_regime_switches} ts={rep.timescale_separation_verdict}",
            flush=True,
        )

    out_path = REPO / "results/phase1/emergence_theory/residence_time_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "interpretation": (
            "Timescale separation: loaded residence tails heavier than grazing (P(τ>t)). "
            "Organization = metastable decomposition; mechanism = dwell + sparse switches."
        ),
        "theorem0_candidate": (
            "Theorem 0 (Coarse-Graining Necessity): μ_ℱ need not concentrate; "
            "under organization-preserving Π, μ_𝒪 shows non-uniform occupancy and finite transition topology."
        ),
        "cohorts": reports,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
