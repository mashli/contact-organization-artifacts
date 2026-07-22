#!/usr/bin/env python3
"""L1 boundary-proximity analysis (Paper I Scheme B test).

For force-matched (episode, f_x_bin) loaded units, tests whether ax/lat
disagreements concentrate near the fixed coupling threshold |f_y|/|f_x|=0.15.

Primary pair: ramp3 vs ramp6 (within-loaded ARI ~0.475).
Design aligned with analyze_partition_forcematched_ari.py.

Output:
  results/phase1/mechanism_discrimination/l1_boundary_proximity_{pair}.json
  optional PNG histogram
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_partition_forcematched_ari import (  # noqa: E402
    BIN_WIDTH,
    FX_THRESHOLD,
    cohort_reps,
    fx_bin,
    matched,
    sub_ari,
)
from analyze_partition_stability_ari import (  # noqa: E402
    COUPLING,
    FX,
    adjusted_rand_index,
    l1_label,
)

THRESHOLD_COUP = 0.15
THRESHOLD_FC = 0.15
FZ_MIN = 0.5
NEAR_EPS = 0.01
NEAR_BAND = (0.14, 0.16)


def coupling_ratio(fr: dict) -> float | None:
    fx = abs(float(fr.get("fx", 0.0)))
    if fx < 1e-6:
        return None
    fy = abs(float((fr.get("features") or {}).get("resultant_force_f_y", 0.0)))
    return fy / fx


def friction_cone_ratio(fr: dict) -> float | None:
    fz = abs(float((fr.get("features") or {}).get("resultant_force_f_z", 0.0)))
    if fz < FZ_MIN:
        return None
    fy = abs(float((fr.get("features") or {}).get("resultant_force_f_y", 0.0)))
    return fy / fz


def representative_rich(episode: dict, fx: float) -> dict[int, dict]:
    """(f_x bin) -> frame dict with labels and ratios."""
    best: dict[int, tuple[float, int, dict]] = {}
    for fr in episode.get("frames") or []:
        lb = l1_label(fr, fx)
        if lb is None:
            continue
        v = float(fr.get("fx", 0.0))
        b = fx_bin(v)
        if b < int(FX_THRESHOLD):
            continue  # loaded only
        rc = coupling_ratio(fr)
        if rc is None:
            continue
        center = (b + 0.5) * BIN_WIDTH
        dist = abs(v - center)
        step = int(fr.get("step", 0))
        key = (dist, step)
        payload = {
            "label": lb,
            "fx": v,
            "bin": b,
            "ratio_coup": rc,
            "ratio_fc": friction_cone_ratio(fr),
            "l1_axlat": "loaded_lat" if rc > THRESHOLD_COUP else "loaded_ax",
        }
        if b not in best or key < (best[b][0], best[b][1]):
            best[b] = (dist, step, payload)
    return {b: t[2] for b, t in best.items()}


def cohort_reps_rich(episodes: list[dict], fx: float) -> dict[int, dict[int, dict]]:
    return {int(ep["episode_id"]): representative_rich(ep, fx) for ep in episodes}


def collect_matched_units(reps_a: dict, reps_b: dict) -> list[dict]:
    rows: list[dict] = []
    for e in sorted(set(reps_a) & set(reps_b)):
        ba, bb = reps_a[e], reps_b[e]
        for b in set(ba) & set(bb):
            if b < int(FX_THRESHOLD):
                continue
            ra, rb = ba[b], bb[b]
            agree = ra["label"] == rb["label"]
            agree_axlat = ra["l1_axlat"] == rb["l1_axlat"]
            da = abs(ra["ratio_coup"] - THRESHOLD_COUP)
            db = abs(rb["ratio_coup"] - THRESHOLD_COUP)
            min_d = min(da, db)
            in_band_a = NEAR_BAND[0] <= ra["ratio_coup"] <= NEAR_BAND[1]
            in_band_b = NEAR_BAND[0] <= rb["ratio_coup"] <= NEAR_BAND[1]
            rows.append(
                {
                    "episode_id": e,
                    "fx_bin": b,
                    "fx_a": ra["fx"],
                    "fx_b": rb["fx"],
                    "label_a": ra["label"],
                    "label_b": rb["label"],
                    "l1_axlat_a": ra["l1_axlat"],
                    "l1_axlat_b": rb["l1_axlat"],
                    "agree_full": agree,
                    "agree_axlat": agree_axlat,
                    "ratio_coup_a": ra["ratio_coup"],
                    "ratio_coup_b": rb["ratio_coup"],
                    "dist_thresh_a": da,
                    "dist_thresh_b": db,
                    "min_dist_thresh": min_d,
                    "near_eps_a": da <= NEAR_EPS,
                    "near_eps_b": db <= NEAR_EPS,
                    "near_eps_either": (da <= NEAR_EPS) or (db <= NEAR_EPS),
                    "in_near_band_either": in_band_a or in_band_b,
                    "opposite_sides": (ra["ratio_coup"] - THRESHOLD_COUP) * (
                        rb["ratio_coup"] - THRESHOLD_COUP
                    )
                    < 0,
                }
            )
    return rows


def summarize_proximity(rows: list[dict]) -> dict:
    n = len(rows)
    disagree = [r for r in rows if not r["agree_axlat"]]
    agree = [r for r in rows if r["agree_axlat"]]
    nd = len(disagree)
    na = len(agree)

    def frac(sub, pred):
        return float(sum(1 for r in sub if pred(r)) / max(len(sub), 1))

    # Histogram bins for distance to threshold (min of both sides)
    bins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 10.0]
    hist_dis = np.histogram([r["min_dist_thresh"] for r in disagree], bins=bins)[0].tolist()
    hist_all = np.histogram([r["min_dist_thresh"] for r in rows], bins=bins)[0].tolist()
    frac_dis_le_002 = frac(disagree, lambda r: r["min_dist_thresh"] <= 0.02)

    # Confusion among disagreements
    conf = defaultdict(int)
    for r in disagree:
        conf[f"{r['l1_axlat_a']}->{r['l1_axlat_b']}"] += 1

    ratios_a = [r["ratio_coup_a"] for r in rows]
    ratios_b = [r["ratio_coup_b"] for r in rows]

    return {
        "n_matched_loaded_units": n,
        "n_agree_axlat": na,
        "n_disagree_axlat": nd,
        "ari_axlat": adjusted_rand_index(
            [r["l1_axlat_a"] for r in rows], [r["l1_axlat_b"] for r in rows]
        ),
        "frac_disagree_near_eps_0.01": frac(disagree, lambda r: r["near_eps_either"]),
        "frac_disagree_in_band_0.14_0.16": frac(disagree, lambda r: r["in_near_band_either"]),
        "frac_disagree_min_dist_le_0.02": frac_dis_le_002,
        "label_agreement_rate_axlat": float(na / max(n, 1)),
        "frac_disagree_opposite_sides": frac(disagree, lambda r: r["opposite_sides"]),
        "frac_agree_near_eps_0.01": frac(agree, lambda r: r["near_eps_either"]),
        "median_min_dist_disagree": float(np.median([r["min_dist_thresh"] for r in disagree]))
        if disagree
        else None,
        "median_min_dist_agree": float(np.median([r["min_dist_thresh"] for r in agree]))
        if agree
        else None,
        "ratio_coup_a_median": float(np.median(ratios_a)) if ratios_a else None,
        "ratio_coup_b_median": float(np.median(ratios_b)) if ratios_b else None,
        "ratio_coup_a_iqr": [
            float(np.percentile(ratios_a, 25)),
            float(np.percentile(ratios_a, 75)),
        ]
        if ratios_a
        else None,
        "ratio_coup_b_iqr": [
            float(np.percentile(ratios_b, 25)),
            float(np.percentile(ratios_b, 75)),
        ]
        if ratios_b else None,
        "hist_min_dist_disagree": {"bins": bins, "counts": hist_dis},
        "hist_min_dist_all": {"bins": bins, "counts": hist_all},
        "disagree_confusion_axlat": dict(conf),
    }


def recommend_scheme(stats: dict) -> dict:
    f_eps = stats["frac_disagree_near_eps_0.01"]
    f_band = stats["frac_disagree_in_band_0.14_0.16"]
    f_002 = stats.get("frac_disagree_min_dist_le_0.02", 0.0)
    agree_rate = stats["n_agree_axlat"] / max(stats["n_matched_loaded_units"], 1)
    med_d = stats.get("median_min_dist_disagree")
    if f_eps >= 0.70 or f_band >= 0.60:
        scheme = "B_offensive_threshold_proximity"
        read = (
            "Disagreements concentrate near fixed coupling threshold 0.15; "
            "supports epistemic ambiguity / fixed guard geometry (Scheme B)."
        )
    elif f_eps < 0.50 and f_band < 0.50 and (med_d is None or med_d > 0.05):
        scheme = "A_defensive_protocol_adapted_L1"
        read = (
            "Disagreements not threshold-localized; supports protocol-dependent "
            "coupling distribution at matched f_x (Scheme A)."
        )
    else:
        scheme = "B_qualified_mixed"
        read = (
            f"Mixed: label agreement {agree_rate:.1%} but ARI {stats['ari_axlat']:.3f}; "
            f"disagreements median {med_d:.3f} from threshold "
            f"({f_002:.0%} within 0.02). Use hierarchical framing: L0 invariant; "
            "L1 fixed guard with threshold-band ambiguity plus protocol-shifted "
            "coupling at matched f_x (Scheme B qualified + A)."
        )
    return {
        "scheme_recommendation": scheme,
        "gates": {
            "frac_disagree_near_eps_0.01": f_eps,
            "frac_disagree_in_band_0.14_0.16": f_band,
            "thresholds": {"eps": 0.70, "band": 0.60},
        },
        "read": read,
    }


def maybe_plot(rows: list[dict], out_png: Path, pair_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    disagree = [r for r in rows if not r["agree_axlat"]]
    agree = [r for r in rows if r["agree_axlat"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in [
        (axes[0], disagree, "Disagree (ax/lat)"),
        (axes[1], agree, "Agree (ax/lat)"),
    ]:
        if not data:
            ax.set_title(f"{title} (n=0)")
            continue
        ax.scatter(
            [r["ratio_coup_a"] for r in data],
            [r["ratio_coup_b"] for r in data],
            s=12,
            alpha=0.6,
        )
        ax.axvline(THRESHOLD_COUP, color="gray", ls="--", lw=0.8)
        ax.axhline(THRESHOLD_COUP, color="gray", ls="--", lw=0.8)
        ax.plot([0, 0.5], [0, 0.5], "k:", lw=0.6)
        ax.set_xlabel("ramp A $|f_y|/|f_x|$")
        ax.set_ylabel("ramp B $|f_y|/|f_x|$")
        ax.set_title(f"{title} (n={len(data)})")
        ax.set_xlim(0, max(0.35, max(r["ratio_coup_a"] for r in data) * 1.05))
        ax.set_ylim(0, max(0.35, max(r["ratio_coup_b"] for r in data) * 1.05))
    fig.suptitle(f"L1 boundary proximity — {pair_name}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", type=int, default=8)
    p.add_argument("--ramp-a", type=int, default=3)
    p.add_argument("--ramp-b", type=int, default=6)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/ari_replay_cache",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    pair_name = f"ramp{args.ramp_a}_vs_ramp{args.ramp_b}"
    out = args.out or (
        REPO / f"results/phase1/mechanism_discrimination/l1_boundary_proximity_{pair_name}.json"
    )

    def load(ramp: int) -> list[dict]:
        cache = args.cache_dir / f"ramp{ramp}_hold{args.hold}.json"
        if not cache.exists():
            raise SystemExit(f"Missing {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))["episodes"]

    eps_a, eps_b = load(args.ramp_a), load(args.ramp_b)
    reps_a = cohort_reps_rich(eps_a, FX_THRESHOLD)
    reps_b = cohort_reps_rich(eps_b, FX_THRESHOLD)
    rows = collect_matched_units(reps_a, reps_b)
    stats = summarize_proximity(rows)
    verdict = recommend_scheme(stats)

    # Cross-check with force-matched sub_ari on full 4-cell labels
    reps_std_a = cohort_reps(eps_a, FX_THRESHOLD)
    reps_std_b = cohort_reps(eps_b, FX_THRESHOLD)
    wl = sub_ari(reps_std_a, reps_std_b, regime="loaded", n_boot=2000, seed=42)

    report = {
        "experiment": "L1_boundary_proximity",
        "pair": pair_name,
        "design": {
            "matching": "(episode_id, f_x_bin) force-matched, loaded only",
            "guard": "G_coupling |f_y|/|f_x| threshold 0.15",
            "near_eps": NEAR_EPS,
            "near_band": list(NEAR_BAND),
            "decision_gates": "docs/PAPER1_L1_BOUNDARY_RESOLUTION_PLAN.md §5",
        },
        "cross_check_forcematched": {
            "ari_within_loaded_4cell": wl["ari"],
            "coverage": wl["coverage"],
        },
        "proximity": stats,
        "verdict": verdict,
        "units": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  n_loaded={stats['n_matched_loaded_units']} ari_axlat={stats['ari_axlat']:.3f}")
    print(f"  disagree={stats['n_disagree_axlat']} near_eps={stats['frac_disagree_near_eps_0.01']:.2%}")
    print(f"  near_band={stats['frac_disagree_in_band_0.14_0.16']:.2%}")
    print(f"  -> {verdict['scheme_recommendation']}")

    if args.plot:
        png = out.with_suffix(".png")
        maybe_plot(rows, png, pair_name)
        print(f"  plot: {png}")


if __name__ == "__main__":
    main()
