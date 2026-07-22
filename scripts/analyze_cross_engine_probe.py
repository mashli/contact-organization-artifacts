#!/usr/bin/env python3
"""Analyze PyBullet cross-engine probe vs frozen MuJoCo reference (force channel)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO.parent
SCRIPTS = Path(__file__).resolve().parent
CGE_SRC = REPO / "src"
for p in (str(REPO_ROOT), str(SCRIPTS), str(CGE_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_recovery_matrices import (  # noqa: E402
    PI_PARTIAL,
    PI_PASS,
    TRANS_PARTIAL,
    TRANS_PASS,
    _classify_pi,
    _classify_trans,
)
from landscape_audit_common import (  # noqa: E402
    FX_THRESH,
    load_trajectories,
    pi_loaded_by_u,
    spearman,
    transition_counts,
)


def cge_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def resolve_reference(paths: list[str]) -> Path | None:
    for rel in paths:
        p = cge_path(rel)
        if p.is_file():
            return p
    return None


def peak_rows(episodes: list[dict]) -> list[dict]:
    rows = []
    for ep in episodes:
        peak = max(ep["frames"], key=lambda fr: abs(float(fr["fx"])))
        rows.append(
            {
                "episode_id": int(ep["episode_id"]),
                "push_eccentricity_m": float(ep["push_eccentricity_m"]),
                "resultant_force_f_x": float(peak["fx"]),
                "basin_label": peak["basin"],
            }
        )
    return rows


def pi_loaded_by_u_episodes(episodes: list[dict]) -> dict[float, float]:
    """Occupancy π(u): mean fraction of loaded frames per episode (force channel)."""
    from collections import defaultdict

    by_u: dict[float, list[float]] = defaultdict(list)
    for ep in episodes:
        u = round(float(ep["push_eccentricity_m"]), 4)
        frames = ep.get("frames") or []
        if not frames:
            continue
        frac = sum(fr["basin"] == "loaded" for fr in frames) / len(frames)
        by_u[u].append(frac)
    return {u: float(sum(v) / len(v)) for u, v in sorted(by_u.items())}


def pi_spearman_from_episodes(episodes: list[dict]) -> float:
    by_u = pi_loaded_by_u_episodes(episodes)
    if len(by_u) < 3:
        return float("nan")
    us = np.array(sorted(by_u.keys()), dtype=np.float64)
    pis = np.array([by_u[u] for u in us], dtype=np.float64)
    return spearman(us, pis)


def transition_spearman(ref_counts: dict[str, int], test_counts: dict[str, int]) -> float:
    keys = sorted(set(ref_counts) | set(test_counts))
    if len(keys) < 2:
        return float("nan")
    a = np.array([ref_counts.get(k, 0) for k in keys], dtype=np.float64)
    b = np.array([test_counts.get(k, 0) for k in keys], dtype=np.float64)
    return spearman(a, b)


def force_layer_pattern(episodes: list[dict]) -> dict[str, Any]:
    pi_sp = pi_spearman_from_episodes(episodes)
    trans = transition_counts(episodes)
    has_l2g = int(trans.get("loaded->grazing", 0) > 0)
    occ = _classify_pi(pi_sp)
    n_edges = sum(1 for v in trans.values() if v > 0)
    trans_class = "pass" if has_l2g and n_edges >= 2 else ("partial" if has_l2g else "fail")
    return {
        "pi_spearman": pi_sp,
        "occupancy": occ,
        "transition": trans_class,
        "transition_counts": trans,
        "has_loaded_to_grazing": bool(has_l2g),
        "n_transition_edges": n_edges,
    }


def pattern_match_score(ref_layers: dict, test_layers: dict) -> tuple[int, int]:
    """Coarse 2-layer force pattern match (occ + trans) → out of 4 nominal cells."""
    score = 0
    total = 2
    for layer in ("occupancy", "transition"):
        if ref_layers[layer] == test_layers[layer]:
            score += 1
        elif ref_layers[layer] == "pass" and test_layers[layer] == "partial":
            score += 0.5
        elif ref_layers[layer] == "partial" and test_layers[layer] == "pass":
            score += 0.5
    return int(round(score * 2)), total * 2  # scale to /16 style


def verdict(
    pi_internal: float,
    pi_cross: float,
    trans_sp_vs_ref: float,
    pattern_4: int,
    test_layers: dict,
) -> str:
    trans_ok = (
        trans_sp_vs_ref >= 0.70
        and test_layers.get("has_loaded_to_grazing")
        and test_layers.get("n_transition_edges", 0) >= 3
    )
    pi_cross_ok = math.isfinite(pi_cross) and pi_cross >= 0.50
    pi_internal_ok = math.isfinite(pi_internal) and abs(pi_internal) >= PI_PARTIAL

    if trans_ok and pi_cross_ok and pattern_4 >= 4:
        return "pass"
    if trans_ok and (pi_cross_ok or pi_internal_ok) and pattern_4 >= 3:
        return "partial"
    return "fail"


def maybe_plot(report: dict, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ref_pi = report["mujoco_reference"]["pi_by_u"]
    test_pi = report["pybullet"]["pi_by_u"]
    us = sorted(set(map(float, ref_pi.keys())) | set(map(float, test_pi.keys())))
    ref_v = [ref_pi.get(str(u), ref_pi.get(u, 0)) for u in us]
    test_v = [test_pi.get(str(u), test_pi.get(u, 0)) for u in us]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(us, ref_v, "o-", label="MuJoCo ref", color="#1f77b4")
    ax.plot(us, test_v, "s--", label="PyBullet", color="#d62728")
    ax.set_xlabel("push eccentricity u (m)")
    ax.set_ylabel("P(loaded) at peak")
    ax.set_title("Cross-engine occupancy probe")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/cross_engine_pybullet_probe_PREREG.yaml",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO / args.config
    yaml_cfg = yaml.safe_load(cfg_path.read_text())

    pb_path = cge_path(yaml_cfg["output_dir"]) / "landscape_trajectories.json"
    if not pb_path.is_file():
        raise SystemExit(f"Missing PyBullet trajectories: {pb_path}")

    ref_path = resolve_reference(yaml_cfg.get("mujoco_reference_trajectories", []))
    if ref_path is None:
        raise SystemExit("No MuJoCo reference trajectories found")

    pb_eps = load_trajectories(pb_path)
    ref_eps = load_trajectories(ref_path)

    ref_layers = force_layer_pattern(ref_eps)
    test_layers = force_layer_pattern(pb_eps)

    ref_trans = ref_layers["transition_counts"]
    test_trans = test_layers["transition_counts"]
    trans_sp = transition_spearman(ref_trans, test_trans)

    ref_pi = pi_loaded_by_u_episodes(ref_eps)
    test_pi = pi_loaded_by_u_episodes(pb_eps)

    shared_u = sorted(set(ref_pi.keys()) & set(test_pi.keys()))
    if len(shared_u) >= 3:
        ref_v = np.array([ref_pi[u] for u in shared_u], dtype=np.float64)
        test_v = np.array([test_pi[u] for u in shared_u], dtype=np.float64)
        pi_cross_sp = spearman(ref_v, test_v)
    else:
        pi_cross_sp = float("nan")
    # JSON keys as strings
    ref_pi_json = {str(k): v for k, v in ref_pi.items()}
    test_pi_json = {str(k): v for k, v in test_pi.items()}

    pat_score, pat_total = pattern_match_score(
        {"occupancy": ref_layers["occupancy"], "transition": ref_layers["transition"]},
        {"occupancy": test_layers["occupancy"], "transition": test_layers["transition"]},
    )

    v = verdict(
        test_layers["pi_spearman"],
        pi_cross_sp,
        trans_sp,
        pat_score,
        test_layers,
    )

    report = {
        "prereg": "docs/CROSS_ENGINE_PROBE_PREREG.md",
        "fx_threshold": FX_THRESH,
        "mujoco_reference": {
            "path": str(ref_path.relative_to(REPO) if ref_path.is_relative_to(REPO) else ref_path),
            "n_episodes": len(ref_eps),
            "layers": ref_layers,
            "pi_by_u": ref_pi_json,
        },
        "pybullet": {
            "path": str(pb_path.relative_to(REPO) if pb_path.is_relative_to(REPO) else pb_path),
            "n_episodes": len(pb_eps),
            "layers": test_layers,
            "pi_by_u": test_pi_json,
        },
        "cross_engine": {
            "occupancy_pattern_spearman": pi_cross_sp,
            "transition_spearman": trans_sp,
            "layer_pattern_match": f"{pat_score}/{pat_total}",
            "verdict": v,
            "success_bands": {
                "pass": "trans>=0.70, pi_cross>=0.50, pattern>=4/4",
                "partial": "trans>=0.70, (pi_cross>=0.50 or |pi_internal|>=0.50), pattern>=3/4",
            },
        },
        "interpretation": (
            "Transition hierarchy replicates across engines; occupancy pattern aligned."
            if v == "pass"
            else (
                "Transition topology replicates; occupancy partially aligned — not solver-only artifact."
                if v == "partial"
                else "Hierarchy does not replicate; recoverability is engine-conditional."
            )
        ),
    }

    out_dir = pb_path.parent
    report_path = out_dir / "cross_engine_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    maybe_plot(report, out_dir / "cross_engine_probe_summary.png")

    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
