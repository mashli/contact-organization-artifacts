#!/usr/bin/env python3
"""CE-2 (G_multiplicity) λ₂ bootstrap CI on cylinder reference cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cge.emergence.guard_partition import guard_g_multiplicity  # noqa: E402
from cge.emergence.measure import load_trajectory_episodes  # noqa: E402
from cge.emergence.msm_diagnostics import (  # noqa: E402
    bootstrap_lambda2,
    chapman_kolmogorov_test,
    estimate_transfer_at_lag,
    its_vs_lag,
)


def multiplicity_sequences(episodes: list[dict], *, fx_threshold: float = 8.0):
    raw: list[list[str]] = []
    labels: set[str] = set()
    for ep in episodes:
        labs = [guard_g_multiplicity(fr, fx_threshold) for fr in (ep.get("frames") or [])]
        raw.append(labs)
        labels.update(labs)
    states = sorted(labels)
    idx = {s: i for i, s in enumerate(states)}
    seqs = []
    for labs in raw:
        seq = [idx[b] for b in labs if b in idx]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs, states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trajectory",
        type=Path,
        default=REPO / "results/phase1/landscape_analysis/landscape_trajectories.json",
    )
    ap.add_argument("--n-bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/phase1/msm_diagnostics/ce2_multiplicity_msm_diagnostics.json",
    )
    args = ap.parse_args()

    _, episodes = load_trajectory_episodes(args.trajectory)
    seqs, states = multiplicity_sequences(episodes)
    n_states = len(states)
    point = estimate_transfer_at_lag(seqs, n_states, lag=1)
    boot = bootstrap_lambda2(
        seqs, n_states, lag=1, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    its = its_vs_lag(seqs, n_states)
    ck = chapman_kolmogorov_test(seqs, n_states, lag=1)

    payload = {
        "arm": "CE-2_G_multiplicity",
        "source_path": str(args.trajectory),
        "n_episodes": len(episodes),
        "states": states,
        "estimator": "row_stochastic_MLE",
        "reversible": False,
        "point_lag1": point,
        "bootstrap": boot,
        "its": its,
        "ck": ck,
        "note": (
            "CE-2 destruction arm: multiplicity partition on cylinder reference. "
            "Point λ₂≈0.372 historically; CI from episode bootstrap."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    ci = boot["lambda2_ci"]
    print(
        f"CE-2: λ₂={boot['lambda2_point']:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] "
        f"states={states} N_trans={point.get('n_transitions')} "
        f"CK_max={ck.get('max_abs_deviation_worst'):.3f}"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
