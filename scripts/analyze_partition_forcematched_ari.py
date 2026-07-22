#!/usr/bin/env python3
"""Force-matched partition ARI — discriminating test T1 (trajectory drift) vs T2 (boundary).

Implements docs/PARTITION_STABILITY_ARI_PREREG.md §8.

Exp-A aligned on the PROTOCOL coordinate (episode, phase, phase_step) and found the realized
partition is protocol-conditioned. But guards are a fixed pure function of features, so that
could be either:
  T1  a longer ramp reaches a DIFFERENT physical state at the same phase-step (boundary
      stable, only the trajectory moved), or
  T2  even at a MATCHED physical state the label still differs (partition not stable in
      physical coordinates).

This test aligns on the PHYSICAL load coordinate f_x (fixed 1.0 N bins, edges on integers so
8.0 N is a bin edge) and asks whether the cell label agrees at matched (episode, f_x-bin).
No new replay: reads the Exp-A ramp{3,6,12}/hold8 caches.

Verdict (Steinley 2004 bands, frozen in prereg §8):
  ari_L1_full >= 0.80 -> T1 (trajectory drift); Paper I object definition stands.
  < 0.65              -> T2 (coordinate-dependent partition); genuine turning point.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# These imports do NOT pull robosuite (that import is lazy in the Exp-A script).
from analyze_partition_stability_ari import (  # noqa: E402
    MIN_F_CMP,
    STEINLEY_FAIL,
    STEINLEY_PASS,
    adjusted_rand_index,
    drift_matrix,
    episode_bootstrap_ci,
    l1_label,
    steinley_label,
)

BIN_WIDTH = 1.0  # N; edges on integers so 8.0 N (load threshold) is a bin edge
FX_THRESHOLD = 8.0


def fx_bin(fx: float) -> int:
    return int(math.floor(fx / BIN_WIDTH))


def representative_by_bin(episode: dict, fx: float) -> dict[int, dict]:
    """(f_x bin) -> chosen frame nearest bin center (tie-break earliest step)."""
    best: dict[int, tuple[float, int, dict]] = {}
    for fr in episode.get("frames") or []:
        lb = l1_label(fr, fx)
        if lb is None:
            continue
        v = float(fr.get("fx", 0.0))
        b = fx_bin(v)
        center = (b + 0.5) * BIN_WIDTH
        dist = abs(v - center)
        step = int(fr.get("step", 0))
        key = (dist, step)
        if b not in best or key < (best[b][0], best[b][1]):
            best[b] = (dist, step, {"label": lb, "fx": v, "bin": b})
    return {b: t[2] for b, t in best.items()}


def cohort_reps(episodes: list[dict], fx: float) -> dict[int, dict[int, dict]]:
    return {int(ep["episode_id"]): representative_by_bin(ep, fx) for ep in episodes}


def matched(
    reps_a: dict, reps_b: dict, *, regime: str | None
) -> tuple[list[str], list[str], list[int], dict]:
    """Units = (episode, f_x bin) present in both. regime: None|'loaded'|'grazing'."""
    la: list[str] = []
    lb: list[str] = []
    ep_u: list[int] = []
    n_both = n_either = 0
    for e in sorted(set(reps_a) & set(reps_b)):
        ba, bb = reps_a[e], reps_b[e]
        for b in set(ba) | set(bb):
            in_regime = True
            if regime == "loaded":
                in_regime = b >= int(FX_THRESHOLD)
            elif regime == "grazing":
                in_regime = b < int(FX_THRESHOLD)
            if not in_regime:
                continue
            has = (b in ba) and (b in bb)
            n_either += 1 if (b in ba or b in bb) else 0
            if has:
                n_both += 1
                la.append(ba[b]["label"])
                lb.append(bb[b]["label"])
                ep_u.append(e)
    f_cmp = float(n_both / n_either) if n_either else float("nan")
    cov = {
        "regime": regime or "all",
        "n_units_matched": n_both,
        "n_units_union": n_either,
        "comparable_sample_fraction": f_cmp,
        "representative": bool(np.isfinite(f_cmp) and f_cmp >= MIN_F_CMP),
    }
    return la, lb, ep_u, cov


def sub_ari(reps_a, reps_b, *, regime, n_boot, seed) -> dict:
    la, lb, ep_u, cov = matched(reps_a, reps_b, regime=regime)
    ari = adjusted_rand_index(la, lb)
    ci = episode_bootstrap_ci(la, lb, ep_u, n_boot=n_boot, seed=seed)
    return {
        "ari": ari,
        "steinley_band": steinley_label(ari),
        "ci95_episode_bootstrap": ci["ci95"],
        "coverage": cov,
        "label_counts_a": {c: la.count(c) for c in sorted(set(la))},
    }


def purity_control(all_eps: list[dict], fx: float, *, prec_fx=0.5, prec_fy=0.5, prec_sp=0.0005) -> dict:
    """Upper-bound control: label must be a pure function of (f_x,f_y,spread).

    Pool all frames (both protocols), bin finely on the full guard-relevant vector, measure
    label purity within each bin. Expect ~1.0 (confirms label=f(features); pipeline sane)."""
    from collections import defaultdict

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for ep in all_eps:
        for fr in ep.get("frames") or []:
            lb = l1_label(fr, fx)
            if lb is None:
                continue
            f = fr.get("features") or {}
            key = (
                round(float(fr.get("fx", 0.0)) / prec_fx),
                round(float(f.get("resultant_force_f_y", 0.0)) / prec_fy),
                round(float(f.get("patch_spread_s_xy", 0.0)) / prec_sp),
            )
            buckets[key].append(lb)
    pure = sum(1 for v in buckets.values() if len(set(v)) == 1)
    total = len(buckets)
    weighted_pure = sum(max((v.count(c) for c in set(v)), default=0) for v in buckets.values())
    weighted_total = sum(len(v) for v in buckets.values())
    return {
        "n_bins": total,
        "frac_bins_pure": float(pure / max(total, 1)),
        "frame_weighted_purity": float(weighted_pure / max(weighted_total, 1)),
        "precision": {"f_x": prec_fx, "f_y": prec_fy, "spread": prec_sp},
        "note": "Expect ~1.0: label is a fixed function of (f_x,f_y,spread). Confirms pipeline.",
    }


def classify(ari_full: float, cov_rep: bool) -> str:
    if not cov_rep or not np.isfinite(ari_full):
        return "NOT_REPRESENTATIVE"
    if ari_full >= STEINLEY_PASS:
        return "T1_TRAJECTORY_DRIFT"
    if ari_full >= STEINLEY_FAIL:
        return "INCONCLUSIVE"
    return "T2_COORDINATE_DEPENDENT_PARTITION"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", type=int, default=8)
    p.add_argument("--ramps", type=str, default="3,6,12")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument(
        "--cache-dir", type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/ari_replay_cache",
    )
    p.add_argument(
        "--out", type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/partition_forcematched_ari.json",
    )
    args = p.parse_args()
    ramps = [int(x) for x in args.ramps.split(",") if x.strip()]
    fx = FX_THRESHOLD

    cohorts: dict[int, list[dict]] = {}
    for r in ramps:
        cache = args.cache_dir / f"ramp{r}_hold{args.hold}.json"
        if not cache.exists():
            raise SystemExit(f"missing cache {cache} — run analyze_partition_stability_ari.py first")
        cohorts[r] = json.loads(cache.read_text(encoding="utf-8"))["episodes"]
        print(f"loaded ramp={r}: {len(cohorts[r])} episodes")

    reps = {r: cohort_reps(eps, fx) for r, eps in cohorts.items()}

    control = purity_control([ep for eps in cohorts.values() for ep in eps], fx)
    print(f"[control] frame-weighted label purity = {control['frame_weighted_purity']:.4f} "
          f"(expect ~1.0)")

    from itertools import combinations

    pairs: dict[str, dict] = {}
    for i, (ra, rb) in enumerate(combinations(sorted(cohorts), 2)):
        key = f"ramp{ra}_vs_ramp{rb}"
        la, lb, ep_u, cov = matched(reps[ra], reps[rb], regime=None)
        ari_full = adjusted_rand_index(la, lb)
        ci = episode_bootstrap_ci(la, lb, ep_u, n_boot=args.n_boot, seed=300 + i)
        dm = drift_matrix(la, lb)
        entry = {
            "ari_L1_full": ari_full,
            "steinley_band": steinley_label(ari_full),
            "ci95_episode_bootstrap": ci["ci95"],
            "coverage": cov,
            "boundary_drift_matrix": dm,
            "ari_within_loaded": sub_ari(reps[ra], reps[rb], regime="loaded", n_boot=args.n_boot, seed=400 + i),
            "ari_within_grazing": sub_ari(reps[ra], reps[rb], regime="grazing", n_boot=args.n_boot, seed=500 + i),
            "mechanism": classify(ari_full, cov["representative"]),
        }
        pairs[key] = entry
        wl = entry["ari_within_loaded"]; wg = entry["ari_within_grazing"]
        print(
            f"[{key}] force-matched ARI_full={ari_full:.3f} ({entry['steinley_band']})  "
            f"within_loaded(ax/lat)={wl['ari']:.3f}  within_grazing(cmp/dsp)={wg['ari']:.3f}  "
            f"f_cmp={cov['comparable_sample_fraction']:.2f} -> {entry['mechanism']}"
        )

    reps_ok = [v for v in pairs.values() if v["coverage"]["representative"]]
    if not reps_ok:
        agg = "NOT_REPRESENTATIVE"
    elif all(v["mechanism"] == "T1_TRAJECTORY_DRIFT" for v in reps_ok):
        agg = "T1_TRAJECTORY_DRIFT"
    elif any(v["mechanism"] == "T2_COORDINATE_DEPENDENT_PARTITION" for v in reps_ok):
        agg = "T2_COORDINATE_DEPENDENT_PARTITION"
    else:
        agg = "INCONCLUSIVE"

    report = {
        "experiment": "Force-matched partition ARI (T1 vs T2 discriminating test)",
        "prereg": "docs/PARTITION_STABILITY_ARI_PREREG.md#8",
        "design": {
            "hold": args.hold,
            "ramps": ramps,
            "alignment_coordinate": "resultant_force_f_x (normal load)",
            "bin_width_N": BIN_WIDTH,
            "bin_edges": "integer N (8.0 is a bin edge; no bin straddles L0)",
            "matching_unit": "(episode_id, f_x_bin)",
            "representative": "frame nearest bin center in f_x (tie: earliest step)",
            "bands_steinley2004": {"T1>=": STEINLEY_PASS, "T2<": STEINLEY_FAIL},
            "bootstrap_unit": "episode",
        },
        "purity_control": control,
        "pairwise": pairs,
        "aggregate_mechanism": agg,
        "comparison_to_expA": {
            "expA_phase_matched_ari": {"ramp3_vs_ramp6": 0.517, "ramp6_vs_ramp12": 0.412, "ramp3_vs_ramp12": 0.134},
            "note": (
                "Exp-A aligned on protocol phase-step. This test aligns on physical load f_x. "
                "If force-matched ARI recovers to >=0.80 while phase-matched was poor, the "
                "Exp-A drift was trajectory (T1), not boundary; Paper I object stands with the "
                "clarification that protocol changes the trajectory->organization map. If it "
                "stays <0.65, the partition is coordinate-dependent (T2)."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", args.out)
    print("AGGREGATE MECHANISM:", agg)


if __name__ == "__main__":
    main()
