#!/usr/bin/env python3
"""Multi-level compositional OPCG audit (Route D): L0 → L1 → L2 → …

Each level: restrict to parent label, apply preregistered refinement guard.
Level 0 uses flat evaluate_opcg (G_fx_mode, r=1 Emergence base case).

Usage:
  PYTHONPATH=src python scripts/analyze_hierarchical_opcg.py \\
    --trajectory results/phase1/landscape_analysis/landscape_trajectories.json \\
    --name cylinder_ref --max-level 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
CGE_SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for p in (str(CGE_SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cge.emergence.guard_partition import (  # noqa: E402
    GUARD_CATALOG,
    guard_g_fx_mode,
    label_episodes,
    transfer_spectrum_from_sequences,
)
from cge.emergence.measure import estimate_measures, load_trajectory_episodes  # noqa: E402

from analyze_opcg_witness import (  # noqa: E402
    DELTA_1,
    PI_U_RANGE_MIN,
    PI_U_SPEARMAN_MIN,
    S_MAX,
    THETA,
    evaluate_opcg,
)

FrameLabelFn = Callable[[dict, float], str]

LEVEL0_GUARD = "G_fx_mode"
MIN_SEGMENT_LEN = 2
FIBER_GAP_MIN = 0.15
# L2+ may have smaller gaps on thin fibers; structural tier uses 0.10
FIBER_GAP_MIN_L2 = 0.10
# Preregistered: thin L2 fibers (loaded_ax) may exceed global S_max slightly
S_MAX_BY_LEVEL: dict[int, float] = {1: S_MAX, 2: 0.30}

# Cylinder preregistered refinement tower (Route D v1.1)
CYLINDER_REFINEMENT_TOWER: list[dict[str, str]] = [
  # L0 parents (G_fx_mode labels) → L1 guards
    {
        "grazing": "G_spread",
        "loaded": "G_coupling",
    },
    # L1 parents → L2 guards (guard selection audit 2026-07-08)
    {
        "grazing_compact": "G_multiplicity",
        "grazing_dispersed": "G_coupling",
        "loaded_ax": "G_multiplicity",
        "loaded_lat": "G_multiplicity",
    },
]


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


def _label_fn(guard_id: str) -> FrameLabelFn:
    if guard_id not in GUARD_CATALOG:
        raise ValueError(f"Unknown guard {guard_id}")
    return GUARD_CATALOG[guard_id][1]


def restrict_to_parent_label(
    episodes: list[dict],
    *,
    parent_label: str,
    parent_label_fn: FrameLabelFn,
    fx_threshold: float,
    min_len: int = MIN_SEGMENT_LEN,
) -> list[dict]:
    """Maximal contiguous segments where parent_label_fn(frame) == parent_label."""
    restricted: list[dict] = []
    for ep in episodes:
        segment: list[dict] = []
        for frame in ep.get("frames") or []:
            if parent_label_fn(frame, fx_threshold) == parent_label:
                segment.append(frame)
            else:
                if len(segment) >= min_len:
                    restricted.append(
                        {
                            "frames": segment,
                            "episode_id": ep.get("episode_id"),
                            "push_eccentricity_m": ep.get("push_eccentricity_m"),
                            "parent_cell": parent_label,
                        }
                    )
                segment = []
        if len(segment) >= min_len:
            restricted.append(
                {
                    "frames": segment,
                    "episode_id": ep.get("episode_id"),
                    "push_eccentricity_m": ep.get("push_eccentricity_m"),
                    "parent_cell": parent_label,
                }
            )
    return restricted


def _mean_run_lengths(restricted: list[dict], guard_id: str, fx_threshold: float) -> dict[str, float]:
    label_fn = _label_fn(guard_id)
    runs: dict[str, list[int]] = {}
    for ep in restricted:
        labels = [label_fn(fr, fx_threshold) for fr in ep.get("frames") or []]
        if not labels:
            continue
        cur, n = labels[0], 1
        for lab in labels[1:]:
            if lab == cur:
                n += 1
            else:
                runs.setdefault(cur, []).append(n)
                cur, n = lab, 1
        runs.setdefault(cur, []).append(n)
    return {k: sum(v) / len(v) for k, v in runs.items() if v}


def hierarchical_tau_ref(
    restricted: list[dict], guard_id: str, fx_threshold: float
) -> float:
    dwell = _mean_run_lengths(restricted, guard_id, fx_threshold)
    if len(dwell) < 2:
        return float("nan")
    return max(max(dwell.values()), 1.0)


def _composed_cell_name(parent_cell: str, refinement_state: str) -> str:
    """Unique cell id: parent__refinement (always prefixed at L2+)."""
    if refinement_state == parent_cell:
        return parent_cell
    if refinement_state.startswith(f"{parent_cell}__"):
        return refinement_state
    if refinement_state.startswith(parent_cell + "_"):
        return refinement_state
    return f"{parent_cell}__{refinement_state}"


def evaluate_refinement_opcg(
    restricted: list[dict],
    *,
    level: int,
    parent_cell: str,
    refinement_guard: str,
    cohort_name: str,
    source_path: str,
    fx_threshold: float,
    tau_ref: float | None = None,
    gap_min: float | None = None,
) -> dict:
    if gap_min is None:
        gap_min = FIBER_GAP_MIN_L2 if level >= 2 else FIBER_GAP_MIN

    label_fn = _label_fn(refinement_guard)
    seqs, _ = label_episodes(restricted, label_fn, fx_threshold=fx_threshold)
    spec = transfer_spectrum_from_sequences(seqs)
    state_names = list(spec.get("state_names") or [])
    n_states = int(spec.get("n_states") or len(state_names))

    if tau_ref is None or not math.isfinite(tau_ref):
        tau_ref = hierarchical_tau_ref(restricted, refinement_guard, fx_threshold)

    lambda2 = float(spec.get("lambda2", float("nan")))
    tau2 = float(spec.get("tau2_implied", float("nan")))
    rel_tau_err = (
        abs(tau2 - tau_ref) / tau_ref if math.isfinite(tau2) and tau_ref > 0 else float("inf")
    )
    gap = 1.0 - lambda2 if math.isfinite(lambda2) else float("nan")
    c1_gap_pass = math.isfinite(gap) and gap >= gap_min
    c1_tau_pass = math.isfinite(tau2) and rel_tau_err <= DELTA_1
    c1_pass = c1_gap_pass or c1_tau_pass

    measures = estimate_measures(restricted, source_path=source_path, fx_threshold=fx_threshold)
    mu_o = measures["organization_mu_O"]
    h2_h0_joint = float(mu_o["joint_entropy_ratio"])
    pi_by_u = mu_o.get("pi_by_u") or {}
    pi_vals = [float(v) for v in pi_by_u.values()]
    pi_range = (max(pi_vals) - min(pi_vals)) if pi_vals else 0.0
    pi_sp = float(mu_o.get("pi_u_spearman", float("nan")))
    n_edges = int(mu_o.get("n_transition_edges", 0))

    dwell = _mean_run_lengths(restricted, refinement_guard, fx_threshold)
    dwell_vals = list(dwell.values())
    kappa = max(dwell_vals) / max(min(dwell_vals), 1e-6) if len(dwell_vals) >= 2 else float("nan")
    s0_2_pass = len(dwell_vals) >= 2 and kappa > 1.0

    c2_primary = math.isfinite(h2_h0_joint) and h2_h0_joint < THETA
    c2_alternate = n_edges >= 2 and (
        pi_range >= PI_U_RANGE_MIN
        or (math.isfinite(pi_sp) and abs(pi_sp) >= PI_U_SPEARMAN_MIN)
    )
    c2_pass = c2_primary or c2_alternate or s0_2_pass

    n_frames = sum(len(ep.get("frames") or []) for ep in restricted)
    n_sw = sum(1 for seq in seqs for a, b in zip(seq, seq[1:]) if a != b)
    switch_rate = n_sw / max(n_frames, 1)
    s_max = S_MAX_BY_LEVEL.get(level, S_MAX)
    c3_pass = switch_rate <= s_max

    composed_states = [_composed_cell_name(parent_cell, s) for s in state_names]
    binary_ok = n_states == 2
    c4_pass = binary_ok and math.isfinite(gap) and gap > 0 and lambda2 < 1.0 - 1e-6

    psi = c1_pass and c2_pass and c3_pass and c4_pass
    structural_pass = c3_pass and c4_pass and s0_2_pass

    return {
        "level": level,
        "parent_cell": parent_cell,
        "refinement_guard": refinement_guard,
        "cohort_name": cohort_name,
        "source_path": source_path,
        "n_segments": len(restricted),
        "n_frames": n_frames,
        "state_names": state_names,
        "composed_state_names": composed_states,
        "n_states": n_states,
        "r_slow_proxy": 1 if n_states == 2 and c4_pass else None,
        "tau_ref_level": tau_ref,
        "gap_min_used": gap_min,
        "conditions": {
            "C1": {
                "tau2": tau2,
                "tau_ref": tau_ref,
                "relative_error": rel_tau_err,
                "spectral_gap": gap,
                "pass_gap": c1_gap_pass,
                "pass_tau": c1_tau_pass,
                "pass": c1_pass,
            },
            "C2": {
                "H2_H0_joint": h2_h0_joint,
                "pass_primary": c2_primary,
                "pass_alternate": c2_alternate,
                "pass_s0_2_proxy": s0_2_pass,
                "pass": c2_pass,
            },
            "C3": {"switch_rate": switch_rate, "S_max_level": s_max, "pass": c3_pass},
            "C4": {
                "binary_fiber": binary_ok,
                "lambda2": lambda2,
                "spectral_gap": gap,
                "pass": c4_pass,
            },
        },
        "S0-2_proxy": {
            "mean_dwell_by_state": dwell,
            "kappa_ratio": kappa,
            "pass": s0_2_pass,
        },
        "Psi_OPCG": psi,
        "structural_pass": structural_pass,
        "verdict": "PASS" if psi else ("STRUCTURAL" if structural_pass else "FAIL"),
    }


def analyze_compositional_tower(
    episodes: list[dict],
    *,
    cohort_name: str,
    source_path: str,
    fx_threshold: float,
    refinement_tower: list[dict[str, str]] | None = None,
    max_level: int | None = None,
) -> dict:
    tower = refinement_tower or CYLINDER_REFINEMENT_TOWER
    if max_level is not None:
        tower = tower[:max_level]

    level0 = evaluate_opcg(
        episodes,
        cohort_name=f"{cohort_name}_L0",
        source_path=source_path,
        fx_threshold=fx_threshold,
    )

    levels: list[dict] = []
    # Current leaf cells: (label, label_fn_guard_id, episodes_restricted source)
    # Level 1 parents use L0 guard labels
    current_parents: list[tuple[str, str]] = [
        ("grazing", LEVEL0_GUARD),
        ("loaded", LEVEL0_GUARD),
    ]

    all_fibers: dict[str, dict] = {}
    composed_cells: list[str] = []

    for lvl_idx, refinement_map in enumerate(tower):
        level_num = lvl_idx + 1
        level_fibers: dict[str, dict] = {}
        next_parents: list[tuple[str, str]] = []
        level_composed: list[str] = []

        for parent_cell, parent_guard_id in current_parents:
            if parent_cell not in refinement_map:
                continue
            refinement_guard = refinement_map[parent_cell]
            parent_label_fn = _label_fn(parent_guard_id)
            restricted = restrict_to_parent_label(
                episodes,
                parent_label=parent_cell,
                parent_label_fn=parent_label_fn,
                fx_threshold=fx_threshold,
            )
            tau_ref = hierarchical_tau_ref(restricted, refinement_guard, fx_threshold)
            fiber_key = f"L{level_num}_{parent_cell}"
            report = evaluate_refinement_opcg(
                restricted,
                level=level_num,
                parent_cell=parent_cell,
                refinement_guard=refinement_guard,
                cohort_name=f"{cohort_name}_{fiber_key}",
                source_path=source_path,
                fx_threshold=fx_threshold,
                tau_ref=tau_ref,
            )
            level_fibers[parent_cell] = report
            all_fibers[fiber_key] = report

        for parent_cell, report in level_fibers.items():
            refinement_guard = refinement_map[parent_cell]
            for state, composed in zip(
                report.get("state_names") or [],
                report.get("composed_state_names") or [],
            ):
                next_parents.append((state, refinement_guard))
                if composed not in level_composed:
                    level_composed.append(composed)

        composed_cells = level_composed

        levels.append(
            {
                "level": level_num,
                "n_parents": len(level_fibers),
                "fibers": level_fibers,
                "K_after_level": len(composed_cells),
                "target_K": 2 ** (level_num + 1),
            }
        )
        current_parents = next_parents

    target_k = 2 ** (len(tower) + 1)
    level0_pass = level0["verdict"] == "PASS"
    all_pass = all(f["verdict"] == "PASS" for f in all_fibers.values())
    all_structural = all(
        f.get("structural_pass") or f["verdict"] == "PASS" for f in all_fibers.values()
    )

    return {
        "protocol": "COT_HIERARCHICAL_CYLINDER_2TO4_PROTOCOL.md v1.1 + L2",
        "route": "D — compositional emergence (avoid flat r≥2)",
        "cohort_name": cohort_name,
        "source_path": source_path,
        "max_level": len(tower),
        "level0": {
            "guard": LEVEL0_GUARD,
            "K": 2,
            "r_slow": 1,
            "report": level0,
        },
        "levels": levels,
        "all_fibers": all_fibers,
        "composed": {
            "cells": composed_cells,
            "K": len(composed_cells),
            "target_K": target_k,
            # This is a witness-level statement: all audited fibers pass OPCG.
            # It is NOT a theorem-level proof that the full K-state object already
            # satisfies Sig(O)=top as one global chain.
            "fiber_level_opcg_pass": level0_pass and all_pass and len(composed_cells) == target_k,
            "structural_composition_hold": level0_pass
            and all_structural
            and len(composed_cells) == target_k,
        },
        "verdict": (
            "PASS"
            if (level0_pass and all_pass)
            else ("STRUCTURAL" if (level0_pass and all_structural) else "PARTIAL")
        ),
    }


# Backward-compatible alias
analyze_hierarchical = analyze_compositional_tower
CYLINDER_REFINEMENT_MAP = CYLINDER_REFINEMENT_TOWER[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compositional OPCG tower (Route D)")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--name", type=str, default="cylinder_ref")
    parser.add_argument(
        "--max-level",
        type=int,
        default=2,
        help="Refinement depth: 1 => K=4, 2 => K=8 (default 2)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/phase1/emergence_theory/hierarchical_opcg_report.json",
    )
    args = parser.parse_args()

    traj = args.trajectory if args.trajectory.is_absolute() else REPO / args.trajectory
    if not traj.is_file():
        raise SystemExit(f"Missing trajectory: {traj}")

    meta, eps = load_trajectory_episodes(traj)
    fx_t = float(meta.get("fx_threshold", 8.0))

    report = analyze_compositional_tower(
        eps,
        cohort_name=args.name,
        source_path=_rel(traj),
        fx_threshold=fx_t,
        max_level=args.max_level,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(report), indent=2))

    l0 = report["level0"]["report"]
    print(f"\n=== Compositional OPCG: {args.name} → {report['verdict']} ===")
    print(f"  L0 {LEVEL0_GUARD}: Psi={l0['Psi_OPCG']} K=2")
    for key, fib in report["all_fibers"].items():
        c4 = fib["conditions"]["C4"]
        print(
            f"  {key} {fib['refinement_guard']}: {fib['verdict']} "
            f"states={fib['composed_state_names']} λ2={c4['lambda2']}"
        )
    comp = report["composed"]
    print(
        f"  Composed K={comp['K']} (target {comp['target_K']}) "
        f"fiber_level_opcg_pass={comp['fiber_level_opcg_pass']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
