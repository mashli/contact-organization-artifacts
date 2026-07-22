#!/usr/bin/env python3
"""Estimate μ_ℱ (feature) and μ_𝒪 (organization) from landscape_trajectories.json."""

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

from cge.emergence.measure import estimate_measures, load_trajectory_episodes  # noqa: E402


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Organization + feature measure baseline (ToCO Ch 4)")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/emergence_measure_baseline_PREREG.yaml",
    )
    parser.add_argument("--trajectory", type=Path, default=None, help="Single JSON override")
    parser.add_argument("--name", type=str, default="custom")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    n_bins = int(cfg.get("analysis", {}).get("histogram_bins", 16))

    reports: dict[str, dict] = {}

    if args.trajectory is not None:
        cohorts = [{"name": args.name, "path": str(args.trajectory)}]
    else:
        cohorts = cfg.get("cohorts", [])

    for c in cohorts:
        name = str(c["name"])
        path = resolve_path(c["path"])
        if not path.is_file():
            print(f"SKIP {name}: missing {path}", flush=True)
            continue
        meta, eps = load_trajectory_episodes(path)
        fx_t = meta.get("fx_threshold")
        rep = estimate_measures(
            eps,
            source_path=str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
            fx_threshold=float(fx_t) if fx_t is not None else None,
            n_bins=n_bins,
        )
        reports[name] = rep
        f = rep["feature_mu_F"]
        o = rep["organization_mu_O"]
        print(
            f"{name}: "
            f"μ_ℱ ratio={f['entropy_ratio']:.3f} ({f['concentration_verdict']}) | "
            f"μ_𝒪 joint={o['joint_entropy_ratio']:.3f} ({o['concentration_verdict']}) "
            f"pi_rho={o['pi_u_spearman']:.3f} "
            f"M1_ℱ_falsified={rep['m1_on_feature_falsified']}",
            flush=True,
        )

    out_rel = cfg.get("output_json", "results/phase1/emergence_theory/measure_baseline_report.json")
    out_path = resolve_path(out_rel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prereg": cfg.get("prereg"),
        "config": str(cfg_path.relative_to(REPO) if cfg_path.is_relative_to(REPO) else cfg_path),
        "theorem0_revision": (
            "Feature occupation need not concentrate; organization occupation (μ_𝒪 on basins) does."
        ),
        "cohorts": reports,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
