#!/usr/bin/env python3
"""Evaluate OPCG witness (S0-3) at locked thresholds for one trajectory cohort.

Locked tuple (COT_OPCG_THRESHOLD_PROTOCOL.md v1.1):
  delta_1=0.25, theta=0.95, S_max=0.20, sigma=0.90
  tau_ref=6.0 (cylinder reference slow timescale)

Usage:
  python scripts/analyze_opcg_witness.py \\
    --trajectory results/phase1/cross_engine/pybullet_probe/landscape_trajectories.json \\
    --name pybullet_probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CGE_SRC = REPO / "src"
if str(CGE_SRC) not in sys.path:
    sys.path.insert(0, str(CGE_SRC))

from cge.emergence.guard_partition import estimate_guard_partitions  # noqa: E402
from cge.emergence.measure import estimate_measures, load_trajectory_episodes  # noqa: E402
from cge.emergence.residence import estimate_residence_times  # noqa: E402

# Locked — do not tune per cohort (COT_OPCG_THRESHOLD_PROTOCOL.md v1.1)
DELTA_1 = 0.25
THETA = 0.95
S_MAX = 0.20
SIGMA = 0.90
TAU_REF = 6.0
GUARD_ID = "G_fx_mode"
# C2 alternate (OPCG Cond 2): occupancy differs across control u
PI_U_RANGE_MIN = 0.15
PI_U_SPEARMAN_MIN = 0.25


def _json_safe(obj: object) -> object:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def evaluate_opcg(
    episodes: list[dict],
    *,
    cohort_name: str,
    source_path: str,
    fx_threshold: float = 8.0,
    ref_lambda2: float | None = None,
    ref_tau2: float | None = None,
) -> dict:
    guard_rep = estimate_guard_partitions(
        episodes, source_path=source_path, fx_threshold=fx_threshold
    )
    measures = estimate_measures(episodes, source_path=source_path, fx_threshold=fx_threshold)
    mu_o = measures["organization_mu_O"]
    residence = estimate_residence_times(episodes, source_path=source_path)

    g_fx = next((g for g in guard_rep.guards if g.guard_id == GUARD_ID), None)
    if g_fx is None:
        raise RuntimeError(f"Guard {GUARD_ID} not found")

    tau2 = float(g_fx.tau2_implied)
    rel_tau_err = abs(tau2 - TAU_REF) / TAU_REF if math.isfinite(tau2) and TAU_REF > 0 else float("inf")
    c1_pass = math.isfinite(tau2) and rel_tau_err <= DELTA_1

    h2_h0_joint = float(mu_o["joint_entropy_ratio"])
    h2_h0_basin = float(mu_o["basin_entropy_ratio"])
    pi_by_u = mu_o.get("pi_by_u") or {}
    pi_vals = [float(v) for v in pi_by_u.values()]
    pi_range = (max(pi_vals) - min(pi_vals)) if pi_vals else 0.0
    pi_sp = float(mu_o.get("pi_u_spearman", float("nan")))
    n_edges = int(mu_o.get("n_transition_edges", 0))

    c2_primary = math.isfinite(h2_h0_joint) and h2_h0_joint < THETA
    c2_alternate = (
        n_edges >= 2
        and (
            pi_range >= PI_U_RANGE_MIN
            or (math.isfinite(pi_sp) and abs(pi_sp) >= PI_U_SPEARMAN_MIN)
        )
    )
    c2_pass = c2_primary or c2_alternate

    n_frames = sum(len(ep.get("frames") or []) for ep in episodes)
    switch_rate = residence.n_regime_switches / max(n_frames, 1)
    c3_pass = switch_rate <= S_MAX

    # C4: internal spectrum match (guard vs MODE within cohort)
    sigma_internal = float(g_fx.spectrum_match_score)
    c4_internal_pass = math.isfinite(sigma_internal) and sigma_internal >= SIGMA

    # C4 cross-ref: compare λ₂ to cylinder reference if provided
    lambda2 = float(g_fx.lambda2)
    c4_cross_pass = None
    if ref_lambda2 is not None and math.isfinite(ref_lambda2) and math.isfinite(lambda2):
        dl2 = abs(lambda2 - ref_lambda2)
        # Map Δλ₂ to match score (same formula as guard_partition)
        dt2 = abs(tau2 - (ref_tau2 or TAU_REF)) if math.isfinite(tau2) else float("inf")
        sigma_cross = 1.0 / (1.0 + dl2 + 0.1 * dt2) if math.isfinite(dl2) else 0.0
        c4_cross_pass = sigma_cross >= SIGMA
    else:
        sigma_cross = float("nan")

    c4_pass = c4_internal_pass if c4_cross_pass is None else (c4_internal_pass and c4_cross_pass)
    psi = c1_pass and c2_pass and c3_pass and c4_pass

    loaded = residence.by_basin.get("loaded")
    grazing = residence.by_basin.get("grazing")

    return {
        "cohort_name": cohort_name,
        "source_path": source_path,
        "fx_threshold": fx_threshold,
        "guard_witness": GUARD_ID,
        "thresholds": {
            "delta_1": DELTA_1,
            "theta": THETA,
            "S_max": S_MAX,
            "sigma": SIGMA,
            "tau_ref": TAU_REF,
        },
        "conditions": {
            "C1": {
                "test": "rel_tau2_error",
                "threshold": f"<= {DELTA_1}",
                "tau2": tau2,
                "tau_ref": TAU_REF,
                "relative_error": rel_tau_err,
                "pass": c1_pass,
            },
            "C2": {
                "test": "joint_entropy_ratio H2/H0 < theta OR u-dependent occupancy",
                "threshold": f"primary: < {THETA}; alternate: pi_range>={PI_U_RANGE_MIN} or |pi_spearman|>={PI_U_SPEARMAN_MIN}",
                "H2_H0_joint": h2_h0_joint,
                "H2_H0_basin": h2_h0_basin,
                "pi_u_range": pi_range,
                "pi_u_spearman": pi_sp,
                "n_transition_edges": n_edges,
                "pass_primary": c2_primary,
                "pass_alternate": c2_alternate,
                "pass": c2_pass,
            },
            "C3": {
                "test": "switch_rate N_sw/T",
                "threshold": f"<= {S_MAX}",
                "N_sw": residence.n_regime_switches,
                "T_frames": n_frames,
                "switch_rate": switch_rate,
                "pass": c3_pass,
            },
            "C4": {
                "test": "spectrum_match",
                "threshold": f">= {SIGMA}",
                "sigma_internal": sigma_internal,
                "sigma_cross_ref": sigma_cross,
                "lambda2": lambda2,
                "ref_lambda2": ref_lambda2,
                "pass_internal": c4_internal_pass,
                "pass_cross_ref": c4_cross_pass,
                "pass": c4_pass,
            },
        },
        "Psi_OPCG": psi,
        "guard_partition_summary": {
            "n_states": g_fx.n_states,
            "lambda2": lambda2,
            "tau2_implied": tau2,
            "ari_vs_mode": g_fx.ari_vs_mode,
            "n_regime_switches_guard": g_fx.n_regime_switches,
        },
        "S0-2": {
            "Sigma_slow_proxy": {
                "loaded_mean_tau": loaded.mean_run_length if loaded else float("nan"),
                "grazing_mean_tau": grazing.mean_run_length if grazing else float("nan"),
                "tail_separation_at_t4": residence.tail_separation_at_t4,
                "pass_proxy": (
                    loaded is not None
                    and grazing is not None
                    and loaded.mean_run_length > grazing.mean_run_length
                    and residence.tail_separation_at_t4 >= 1.0
                ),
            },
        },
        "verdict": "PASS" if psi else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OPCG witness evaluation (S0-3)")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--name", type=str, default="custom")
    parser.add_argument(
        "--ref-trajectory",
        type=Path,
        default=None,
        help="MuJoCo reference for C4 cross-spectrum (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/phase1/emergence_theory/opcg_witness_report.json",
    )
    args = parser.parse_args()

    traj = args.trajectory if args.trajectory.is_absolute() else REPO / args.trajectory
    if not traj.is_file():
        raise SystemExit(f"Missing trajectory: {traj}")

    meta, eps = load_trajectory_episodes(traj)
    fx_t = float(meta.get("fx_threshold", 8.0))

    ref_l2 = ref_t2 = None
    if args.ref_trajectory is not None:
        ref_path = args.ref_trajectory if args.ref_trajectory.is_absolute() else REPO / args.ref_trajectory
        if ref_path.is_file():
            _, ref_eps = load_trajectory_episodes(ref_path)
            ref_rep = estimate_guard_partitions(ref_eps, fx_threshold=fx_t)
            ref_g = next(g for g in ref_rep.guards if g.guard_id == GUARD_ID)
            ref_l2 = float(ref_g.lambda2)
            ref_t2 = float(ref_g.tau2_implied)

    report = evaluate_opcg(
        eps,
        cohort_name=args.name,
        source_path=_rel(traj),
        fx_threshold=fx_t,
        ref_lambda2=ref_l2,
        ref_tau2=ref_t2,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "COT_OPCG_THRESHOLD_PROTOCOL.md v1.1",
        "cohorts": {args.name: report},
    }
    # Merge if exists
    if args.out.is_file():
        existing = json.loads(args.out.read_text())
        existing.setdefault("cohorts", {}).update(payload["cohorts"])
        payload = existing

    args.out.write_text(json.dumps(_json_safe(payload), indent=2))

    c = report["conditions"]
    print(f"\n=== OPCG {args.name}: Psi={report['Psi_OPCG']} ({report['verdict']}) ===")
    print(f"  C1 tau2={c['C1']['tau2']:.3f} rel_err={c['C1']['relative_error']:.3f} pass={c['C1']['pass']}")
    print(f"  C2 joint={c['C2']['H2_H0_joint']:.3f} primary={c['C2']['pass_primary']} alt={c['C2']['pass_alternate']} pass={c['C2']['pass']}")
    print(f"  C3 switch_rate={c['C3']['switch_rate']:.3f} pass={c['C3']['pass']}")
    print(f"  C4 sigma_int={c['C4']['sigma_internal']:.3f} pass={c['C4']['pass']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
