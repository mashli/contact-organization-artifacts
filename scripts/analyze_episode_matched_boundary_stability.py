#!/usr/bin/env python3
"""
Episode-matched boundary stability experiment.

Problem with force-matched (f_cmp < 0.5):
  ramp3 vs ramp12 explore very different f_x ranges, so only a tiny fraction
  of (episode, f_x_bin) units overlap.

New approach: episode-MATCHED comparison
  1. Find episodes that have the SAME initial conditions under different protocols
     (same episode_id, same seed)
  2. For each matched episode-pair, compare labels at the SAME f_x value
     (rather than same f_x_bin)
  3. This gives MUCH higher coverage since we're comparing the same physical
     episode at the same physical state

This directly tests: at the same initial condition and same physical load,
does the cell label agree?

Design: frozen per docs/PARTITION_STABILITY_ARI_PREREG.md
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

FX_THRESHOLD = 8.0
FZ_MIN = 0.5  # for friction cone


def l0_label(fx: float) -> str:
    return "loaded" if fx >= FX_THRESHOLD else "grazing"


def l1_friction_cone(fx: float, fy: float, fz: float) -> str | None:
    """L1 using friction cone proxy |f_y|/|f_z|."""
    if abs(fz) < FZ_MIN:
        return None
    ratio = abs(fy) / abs(fz)
    base = l0_label(fx)
    if base == "grazing":
        return None
    return "lat" if ratio > 0.15 else "ax"


def l1_coupling(fx: float, fy: float) -> str | None:
    """L1 using coupling ratio |f_y|/|f_x| (original G_coupling)."""
    if abs(fx) < 1e-6:
        return None
    ratio = abs(fy) / abs(fx)
    base = l0_label(fx)
    if base == "grazing":
        return None
    return "lat" if ratio > 0.15 else "ax"


def adjusted_rand_index(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return float("nan")
    a_ids = {c: i for i, c in enumerate(sorted(set(labels_a)))}
    b_ids = {c: i for i, c in enumerate(sorted(set(labels_b)))}
    cont = np.zeros((len(a_ids), len(b_ids)), dtype=float)
    for la, lb in zip(labels_a, labels_b):
        cont[a_ids[la], b_ids[lb]] += 1.0

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = float(np.sum(comb2(cont)))
    a_sums = cont.sum(axis=1)
    b_sums = cont.sum(axis=0)
    sum_a = float(np.sum(comb2(a_sums)))
    sum_b = float(np.sum(comb2(b_sums)))
    total = comb2(float(n))
    expected = (sum_a * sum_b) / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_ij - expected) / denom)


def steinley_label(ari: float) -> str:
    if not np.isfinite(ari):
        return "undefined"
    if ari >= 0.90:
        return "excellent"
    if ari >= 0.80:
        return "good"
    if ari >= 0.65:
        return "moderate"
    return "poor"


def episode_bootstrap_ci(labels_a: list[str], labels_b: list[str],
                         episode_ids: list[int], n_boot: int = 2000, seed: int = 42) -> dict:
    """Episode-level bootstrap CI for ARI."""
    np.random.seed(seed)
    unique_eps = sorted(set(episode_ids))
    if len(unique_eps) < 3:
        return {"ci95": [float("nan"), float("nan")], "note": f"too few episodes ({len(unique_eps)})"}

    ep_to_indices = defaultdict(list)
    for idx, ep in enumerate(episode_ids):
        ep_to_indices[ep].append(idx)

    boot_aris = []
    for _ in range(n_boot):
        sampled_eps = list(np.random.choice(unique_eps, size=len(unique_eps), replace=True))
        boot_idx = []
        for ep in sampled_eps:
            boot_idx.extend(ep_to_indices[ep])
        la_boot = [labels_a[i] for i in boot_idx]
        lb_boot = [labels_b[i] for i in boot_idx]
        boot_aris.append(adjusted_rand_index(la_boot, lb_boot))

    boot_aris = [a for a in boot_aris if np.isfinite(a)]
    if not boot_aris:
        return {"ci95": [float("nan"), float("nan")]}
    boot_aris = sorted(boot_aris)
    lo = boot_aris[int(0.025 * len(boot_aris))]
    hi = boot_aris[int(0.975 * len(boot_aris))]
    return {"ci95": [lo, hi], "mean": float(np.mean(boot_aris))}


def drift_matrix(labels_a: list[str], labels_b: list[str]) -> dict:
    cells = sorted(set(labels_a + labels_b))
    a_ids = {c: i for i, c in enumerate(cells)}
    b_ids = {c: i for i, c in enumerate(cells)}
    counts = np.zeros((len(cells), len(cells)), dtype=int)
    for la, lb in zip(labels_a, labels_b):
        counts[a_ids[la], b_ids[lb]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    row_pct = counts / row_sums * 100
    return {
        "cells": cells,
        "counts": counts.tolist(),
        "row_pct": row_pct.tolist(),
        "per_cell_stability_pct": {
            cells[i]: float(row_pct[i, b_ids.get(cells[i], 0)])
            for i in range(len(cells))
        },
    }


def load_and_index_episodes(cache_path: Path):
    """Load episodes and index by episode_id for fast lookup."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    by_id = {}
    for ep in data["episodes"]:
        by_id[int(ep["episode_id"])] = ep
    return by_id, len(data["episodes"])


def build_episode_frames(ep: dict) -> list[dict]:
    """Extract per-frame data with labels for an episode."""
    frames = []
    for fr in ep.get("frames", []):
        features = fr.get("features") or {}
        fx = float(fr.get("fx", 0.0))
        fy = float(features.get("resultant_force_f_y", 0.0))
        fz = float(features.get("resultant_force_f_z", 0.0))
        frames.append({
            "step": int(fr.get("step", 0)),
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "l0": l0_label(fx),
            "l1_fc": l1_friction_cone(fx, fy, fz),
            "l1_coup": l1_coupling(fx, fy),
        })
    return frames


def episode_matched_comparison(ep_a: dict, ep_b: dict, bin_width_fx: float = 0.5):
    """
    For two episodes with SAME initial conditions (same episode_id cohort),
    find frames at the same physical load (f_x) and compare their labels.
    """
    frames_a = build_episode_frames(ep_a)
    frames_b = build_episode_frames(ep_b)

    # Build f_x -> closest frames for each episode (fine binning)
    fxa_bins = defaultdict(list)
    fxb_bins = defaultdict(list)

    for fr in frames_a:
        b = int(round(fr["fx"] / bin_width_fx))
        fxa_bins[b].append(fr)
    for fr in frames_b:
        b = int(round(fr["fx"] / bin_width_fx))
        fxb_bins[b].append(fr)

    l0_a, l0_b, l1fc_a, l1fc_b, l1coup_a, l1coup_b, eps = [], [], [], [], [], [], []
    n_union = 0

    for b in sorted(set(fxa_bins) | set(fxb_bins)):
        has_a = b in fxa_bins
        has_b = b in fxb_bins
        n_union += 1
        if has_a and has_b:
            center = b * bin_width_fx
            best_a = min(fxa_bins[b], key=lambda f: abs(f["fx"] - center))
            best_b = min(fxb_bins[b], key=lambda f: abs(f["fx"] - center))
            ep_id = int(ep_a["episode_id"])

            l0_a.append(best_a["l0"])
            l0_b.append(best_b["l0"])
            eps.append(ep_id)

            if best_a["l1_fc"] is not None and best_b["l1_fc"] is not None:
                l1fc_a.append(best_a["l1_fc"])
                l1fc_b.append(best_b["l1_fc"])

            if best_a["l1_coup"] is not None and best_b["l1_coup"] is not None:
                l1coup_a.append(best_a["l1_coup"])
                l1coup_b.append(best_b["l1_coup"])

    f_cmp = len(l0_a) / n_union if n_union > 0 else 0.0
    return {
        "l0_a": l0_a, "l0_b": l0_b, "eps_l0": eps,
        "l1fc_a": l1fc_a, "l1fc_b": l1fc_b, "eps_fc": eps[:len(l1fc_a)],
        "l1coup_a": l1coup_a, "l1coup_b": l1coup_b, "eps_coup": eps[:len(l1coup_a)],
        "n_union": n_union, "f_cmp": f_cmp,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", type=int, default=8)
    p.add_argument("--ramps", type=str, default="3,6,12")
    p.add_argument("--bin-width-fx", type=float, default=0.5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument(
        "--cache-dir", type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/ari_replay_cache",
    )
    p.add_argument(
        "--out", type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/episode_matched_boundary.json",
    )
    args = p.parse_args()
    ramps = [int(x) for x in args.ramps.split(",") if x.strip()]

    # Load all cohorts
    cohorts: dict[int, dict] = {}
    for r in ramps:
        cache = args.cache_dir / f"ramp{r}_hold{args.hold}.json"
        if not cache.exists():
            raise SystemExit(f"Missing cache: {cache}")
        cohorts[r], n_eps = load_and_index_episodes(cache)
        print(f"Loaded ramp={r}: {n_eps} episodes")

    # Episode-matched pairwise comparison
    from itertools import combinations
    pairwise_results = {}

    print(f"\nEpisode-matched comparison (bin_width_fx={args.bin_width_fx} N):")
    for i, (ra, rb) in enumerate(combinations(sorted(cohorts), 2)):
        pair_name = f"ramp{ra}_vs_ramp{rb}"
        print(f"\n  [{pair_name}]")

        common_ids = sorted(set(cohorts[ra]) & set(cohorts[rb]))
        print(f"  Common episode_ids: {len(common_ids)}")

        # Aggregate across all matched episodes
        all_l0_a, all_l0_b, all_eps = [], [], []
        all_l1fc_a, all_l1fc_b, all_eps_fc = [], [], []
        all_l1coup_a, all_l1coup_b, all_eps_coup = [], [], []

        for ep_id in common_ids:
            ep_a = cohorts[ra][ep_id]
            ep_b = cohorts[rb][ep_id]
            res = episode_matched_comparison(ep_a, ep_b, bin_width_fx=args.bin_width_fx)

            all_l0_a.extend(res["l0_a"])
            all_l0_b.extend(res["l0_b"])
            all_eps.extend(res["eps_l0"])

            all_l1fc_a.extend(res["l1fc_a"])
            all_l1fc_b.extend(res["l1fc_b"])
            all_eps_fc.extend(res["eps_fc"])

            all_l1coup_a.extend(res["l1coup_a"])
            all_l1coup_b.extend(res["l1coup_b"])
            all_eps_coup.extend(res["eps_coup"])

        n_union = res["n_union"]
        results = {}

        # L0
        if all_l0_a:
            ari = adjusted_rand_index(all_l0_a, all_l0_b)
            ci = episode_bootstrap_ci(all_l0_a, all_l0_b, all_eps, n_boot=args.n_boot, seed=300+i)
            dm = drift_matrix(all_l0_a, all_l0_b)
            cov = {"regime": "L0", "n_matched": len(all_l0_a), "n_union": n_union,
                   "f_cmp": len(all_l0_a) / n_union if n_union > 0 else 0.0}
            rep = cov["f_cmp"] >= 0.50
            results["L0"] = {
                "ari": ari, "steinley": steinley_label(ari),
                "ci95": ci.get("ci95"), "coverage": cov,
                "drift_matrix": dm, "representative": rep,
            }
            print(f"  L0:  ARI={ari:.3f} ({steinley_label(ari)}) f_cmp={cov['f_cmp']:.3f} "
                  f"{'✓' if rep else '✗'} n_matched={cov['n_matched']}")

        # L1 FC
        if all_l1fc_a:
            ari_fc = adjusted_rand_index(all_l1fc_a, all_l1fc_b)
            ci_fc = episode_bootstrap_ci(all_l1fc_a, all_l1fc_b, all_eps_fc, n_boot=args.n_boot, seed=400+i)
            dm_fc = drift_matrix(all_l1fc_a, all_l1fc_b)
            cov_fc = {"regime": "L1_fc", "n_matched": len(all_l1fc_a), "n_union": n_union,
                      "f_cmp": len(all_l1fc_a) / n_union if n_union > 0 else 0.0}
            rep_fc = cov_fc["f_cmp"] >= 0.50
            results["L1_friction_cone"] = {
                "ari": ari_fc, "steinley": steinley_label(ari_fc),
                "ci95": ci_fc.get("ci95"), "coverage": cov_fc,
                "drift_matrix": dm_fc, "representative": rep_fc,
            }
            print(f"  L1(FC): ARI={ari_fc:.3f} ({steinley_label(ari_fc)}) f_cmp={cov_fc['f_cmp']:.3f} "
                  f"{'✓' if rep_fc else '✗'} n_matched={cov_fc['n_matched']}")

        # L1 Coup
        if all_l1coup_a:
            ari_coup = adjusted_rand_index(all_l1coup_a, all_l1coup_b)
            ci_coup = episode_bootstrap_ci(all_l1coup_a, all_l1coup_b, all_eps_coup, n_boot=args.n_boot, seed=500+i)
            dm_coup = drift_matrix(all_l1coup_a, all_l1coup_b)
            cov_coup = {"regime": "L1_coup", "n_matched": len(all_l1coup_a), "n_union": n_union,
                        "f_cmp": len(all_l1coup_a) / n_union if n_union > 0 else 0.0}
            rep_coup = cov_coup["f_cmp"] >= 0.50
            results["L1_coupling"] = {
                "ari": ari_coup, "steinley": steinley_label(ari_coup),
                "ci95": ci_coup.get("ci95"), "coverage": cov_coup,
                "drift_matrix": dm_coup, "representative": rep_coup,
            }
            print(f"  L1(Coup): ARI={ari_coup:.3f} ({steinley_label(ari_coup)}) f_cmp={cov_coup['f_cmp']:.3f} "
                  f"{'✓' if rep_coup else '✗'} n_matched={cov_coup['n_matched']}")

        # Delta
        if "L1_friction_cone" in results and "L1_coupling" in results:
            delta = results["L1_friction_cone"]["ari"] - results["L1_coupling"]["ari"]
            print(f"  Δ(FC-Coup): {delta:+.3f} ({'FC better' if delta > 0 else 'Coup better'})")

        pairwise_results[pair_name] = results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for pair_name, pr in pairwise_results.items():
        print(f"\n{pair_name}:")
        for k, v in pr.items():
            print(f"  {k}: ARI={v['ari']:.3f} ({v['steinley']}) "
                  f"f_cmp={v['coverage']['f_cmp']:.3f} "
                  f"{'✓' if v['representative'] else '✗'}")

    report = {
        "experiment": "Episode-matched boundary stability",
        "design": {
            "matching": "same episode_id, same f_x (±bin), compare labels",
            "bin_width_fx_N": args.bin_width_fx,
            "L1_fc_threshold": "|f_y|/|f_z| > 0.15",
            "L1_coup_threshold": "|f_y|/|f_x| > 0.15",
            "bootstrap_unit": "episode",
            "f_cmp_threshold": 0.50,
        },
        "cohorts": {f"ramp{r}": len(cohorts[r]) for r in ramps},
        "pairwise": pairwise_results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
