#!/usr/bin/env python3
"""Exp-A: Partition (boundary-identity) stability under fixed hold, varying ramp.

Implements docs/PARTITION_STABILITY_ARI_PREREG.md (Priority 1 experiment).

Question: is the guard-induced *partition* (which physical configuration maps to which
cell) a property of the system geometry, or of the protocol timing? Ramp changes the
ENTRY trajectory into the hold, not the dwell length, so it is the clean test that hold
(which natively changes exposure/occupation) cannot give.

Frozen design decisions (all from the preregistration, not chosen post-hoc):
  - Guard thresholds are the SAME fixed shared constants across every ramp (§2).
  - Frames aligned by (episode_id, protocol_phase, phase_step_from_start), NEVER by
    absolute frame index (§3).
  - Headline comparison = HOLD phase (protocols genuinely differ there).
  - Missing units (contact drop / early escape) are EXCLUDED from ARI but the
    comparable-sample fraction f_cmp and the missingness pattern are reported (§3b).
  - Determinism / self-replay floor: replaying one ramp twice must give ARI=1.0 (§3a).
  - Composite pass: ARI >= 0.80 AND no off-diagonal boundary confusion > 10% (§4a/§4b).
  - Bootstrap resampling unit = EPISODE, never frame (§5a).
  - Steinley (2004) ARI bands (§4).

This script computes numbers; it does NOT decide theory. Verdicts follow the frozen bands.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cge.emergence.guard_partition import GUARD_CATALOG  # noqa: E402

# NOTE: `collect_landscape_trajectories` pulls in the full robosuite/gym sim stack.
# It is imported lazily inside replay_cohort() so that cache-only re-analysis, the ARI
# logic, drift matrix, and bootstrap can run without the simulator installed.

FX = GUARD_CATALOG["G_fx_mode"][1]
SPREAD = GUARD_CATALOG["G_spread"][1]
COUPLING = GUARD_CATALOG["G_coupling"][1]

L1_CELLS = ["grazing_compact", "grazing_dispersed", "loaded_ax", "loaded_lat"]

# Steinley (2004) recovery bands, frozen by preregistration §4.
STEINLEY_PASS = 0.80  # >= good/excellent -> partition stable
STEINLEY_FAIL = 0.65  # < poor -> partition drifts
MAX_BOUNDARY_CONFUSION = 0.10  # §4a composite: no off-diagonal cell confusion > 10%
MIN_F_CMP = 0.50  # §3b: below this a comparison is NOT-REPRESENTATIVE
MIN_CELL_SUPPORT = 20  # §4b: a cell's confusion counts toward FAIL only if row support >= this


def l1_label(fr: dict, fx: float) -> str | None:
    """L1 cell label using the SAME fixed shared thresholds for every protocol."""
    base = FX(fr, fx)
    if base == "grazing":
        return SPREAD(fr, fx)
    if base == "loaded":
        return COUPLING(fr, fx)
    return None


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


def adjusted_rand_index(labels_a: list[str], labels_b: list[str]) -> float:
    """Hubert-Arabie ARI from the contingency table (self-contained, no sklearn).

    Label-permutation invariant: measures co-membership agreement, not label names.
    """
    n = len(labels_a)
    if n == 0:
        return float("nan")
    a_ids = {c: i for i, c in enumerate(sorted(set(labels_a)))}
    b_ids = {c: i for i, c in enumerate(sorted(set(labels_b)))}
    cont = np.zeros((len(a_ids), len(b_ids)), dtype=float)
    for la, lb in zip(labels_a, labels_b):
        cont[a_ids[la], b_ids[lb]] += 1.0

    def comb2(x: np.ndarray | float) -> np.ndarray | float:
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
        # Both partitions trivial (single cluster) -> perfect agreement by convention.
        return 1.0
    return float((sum_ij - expected) / denom)


def label_map(episode: dict, fx: float) -> dict[tuple[str, int], str]:
    """(protocol_phase, phase_step_from_start) -> L1 label for one episode replay."""
    out: dict[tuple[str, int], str] = {}
    for fr in episode.get("frames") or []:
        lb = l1_label(fr, fx)
        if lb is None:
            continue
        key = (str(fr.get("protocol_phase", "")), int(fr.get("phase_step_from_start", -1)))
        # First occurrence wins; phase_step is unique per step so this is 1:1.
        out.setdefault(key, lb)
    return out


def cohort_label_maps(episodes: list[dict], fx: float) -> dict[int, dict[tuple[str, int], str]]:
    return {int(ep["episode_id"]): label_map(ep, fx) for ep in episodes}


def matched_units(
    maps_a: dict[int, dict[tuple[str, int], str]],
    maps_b: dict[int, dict[tuple[str, int], str]],
    *,
    phase: str | None,
) -> tuple[list[str], list[str], list[int], dict]:
    """Return (labels_a, labels_b, episode_id_per_unit, coverage) over units present in
    BOTH protocols, restricted to `phase` if given. Missing units are excluded (§3b)."""
    la: list[str] = []
    lb: list[str] = []
    ep_of_unit: list[int] = []
    n_present_a = n_present_b = n_both = n_either = 0
    common_eps = sorted(set(maps_a) & set(maps_b))
    for e in common_eps:
        ka = {k for k in maps_a[e] if phase is None or k[0] == phase}
        kb = {k for k in maps_b[e] if phase is None or k[0] == phase}
        both = ka & kb
        either = ka | kb
        n_present_a += len(ka)
        n_present_b += len(kb)
        n_both += len(both)
        n_either += len(either)
        for k in sorted(both):
            la.append(maps_a[e][k])
            lb.append(maps_b[e][k])
            ep_of_unit.append(e)
    f_cmp = float(n_both / n_either) if n_either else float("nan")
    coverage = {
        "phase": phase or "all",
        "n_common_episodes": len(common_eps),
        "n_units_present_a": n_present_a,
        "n_units_present_b": n_present_b,
        "n_units_matched": n_both,
        "n_units_union": n_either,
        "comparable_sample_fraction": f_cmp,
        "representative": bool(np.isfinite(f_cmp) and f_cmp >= MIN_F_CMP),
    }
    return la, lb, ep_of_unit, coverage


def drift_matrix(labels_a: list[str], labels_b: list[str]) -> dict:
    """Row-normalized confusion matrix (§4b). Rows=A label, cols=B label."""
    cells = L1_CELLS
    idx = {c: i for i, c in enumerate(cells)}
    counts = np.zeros((len(cells), len(cells)), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1.0
    row = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0) * 100.0
    diag_stability = {
        cells[i]: (float(pct[i, i]) if row[i, 0] > 0 else None) for i in range(len(cells))
    }
    row_support = {cells[i]: int(row[i, 0]) for i in range(len(cells))}
    low_support = [c for c, s in row_support.items() if 0 < s < MIN_CELL_SUPPORT]
    # Largest off-diagonal per row = boundary confusion. Split into the value used for the
    # verdict (supported rows only, §4b) and the raw value (all rows) for transparency.
    max_conf_supported = 0.0
    where_supported = None
    max_conf_raw = 0.0
    where_raw = None
    for i in range(len(cells)):
        if row[i, 0] == 0:
            continue
        for j in range(len(cells)):
            if i == j:
                continue
            v = float(pct[i, j] / 100.0)
            if v > max_conf_raw:
                max_conf_raw = v
                where_raw = f"{cells[i]}->{cells[j]}"
            if row[i, 0] >= MIN_CELL_SUPPORT and v > max_conf_supported:
                max_conf_supported = v
                where_supported = f"{cells[i]}->{cells[j]}"
    return {
        "cells": cells,
        "counts": counts.astype(int).tolist(),
        "row_pct": [[round(float(x), 2) for x in r] for r in pct],
        "per_cell_stability_pct": diag_stability,
        "row_support": row_support,
        "low_support_cells": low_support,
        "min_cell_support": MIN_CELL_SUPPORT,
        # Verdict-driving confusion (supported rows only):
        "max_off_diagonal_confusion": max_conf_supported,
        "max_confusion_transition": where_supported,
        # Transparency (includes low-support rows):
        "max_off_diagonal_confusion_raw": max_conf_raw,
        "max_confusion_transition_raw": where_raw,
    }


def episode_bootstrap_ci(
    labels_a: list[str],
    labels_b: list[str],
    ep_of_unit: list[int],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap ARI CI resampling whole EPISODES with replacement (§5a)."""
    if not ep_of_unit:
        return {"n_boot": 0, "ci95": [None, None], "mean": None, "std": None}
    by_ep: dict[int, list[int]] = defaultdict(list)
    for u, e in enumerate(ep_of_unit):
        by_ep[e].append(u)
    eps = list(by_ep)
    rng = np.random.default_rng(seed)
    aris: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(len(eps), size=len(eps), replace=True)
        ua: list[str] = []
        ub: list[str] = []
        for di in drawn:
            for u in by_ep[eps[di]]:
                ua.append(labels_a[u])
                ub.append(labels_b[u])
        aris.append(adjusted_rand_index(ua, ub))
    arr = np.array([a for a in aris if np.isfinite(a)], dtype=float)
    if arr.size == 0:
        return {"n_boot": n_boot, "ci95": [None, None], "mean": None, "std": None}
    return {
        "n_boot": n_boot,
        "n_episodes_resampled": len(eps),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def compare_pair(
    maps_a: dict, maps_b: dict, *, phase: str | None, n_boot: int, seed: int
) -> dict:
    la, lb, ep_u, cov = matched_units(maps_a, maps_b, phase=phase)
    ari = adjusted_rand_index(la, lb)
    dm = drift_matrix(la, lb)
    ci = episode_bootstrap_ci(la, lb, ep_u, n_boot=n_boot, seed=seed)
    max_conf = dm["max_off_diagonal_confusion"]
    stable = bool(
        cov["representative"]
        and np.isfinite(ari)
        and ari >= STEINLEY_PASS
        and max_conf <= MAX_BOUNDARY_CONFUSION
    )
    if not cov["representative"]:
        verdict = "NOT_REPRESENTATIVE"
    elif not np.isfinite(ari):
        verdict = "UNDEFINED"
    elif stable:
        verdict = "PARTITION_STABLE"
    elif ari >= STEINLEY_PASS and max_conf > MAX_BOUNDARY_CONFUSION:
        verdict = "LOCALIZED_BOUNDARY_DRIFT"
    elif ari >= STEINLEY_FAIL:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PARTITION_DRIFTS"
    return {
        "ari": ari,
        "steinley_band": steinley_label(ari),
        "ari_ci95_episode_bootstrap": ci,
        "coverage": cov,
        "boundary_drift_matrix": dm,
        "composite_pass": stable,
        "verdict": verdict,
    }


def missingness_report(cohorts: dict[int, list[dict]], fx: float, hold: int) -> dict:
    """Log hold-phase coverage per ramp: how many episodes reach each hold step (§3b).

    Reason (a) contact-drop vs (b) early-termination cannot be perfectly separated from
    stored frames (sub-contact frames are not recorded), so we report the observable
    proxy: per-episode max hold step reached vs the configured hold length.
    """
    rep: dict[str, dict] = {}
    for ramp, eps in cohorts.items():
        reached_full = 0
        max_steps = []
        for ep in eps:
            hold_ks = [
                int(fr["phase_step_from_start"])
                for fr in ep.get("frames") or []
                if str(fr.get("protocol_phase")) == "hold" and l1_label(fr, fx) is not None
            ]
            mk = (max(hold_ks) + 1) if hold_ks else 0
            max_steps.append(mk)
            if mk >= hold:
                reached_full += 1
        rep[str(ramp)] = {
            "n_episodes": len(eps),
            "hold_len_config": hold,
            "frac_reaching_full_hold": float(reached_full / max(len(eps), 1)),
            "mean_hold_steps_covered": float(np.mean(max_steps)) if max_steps else 0.0,
        }
    return rep


def replay_cohort(
    char_cfg: dict, episode_rows: list[dict], *, ramp: int, hold: int, fx: float,
    max_episodes: int | None, pass_tag: str = "",
) -> list[dict]:
    from collect_landscape_trajectories import (  # lazy: needs robosuite/gym
        episode_cell_from_char,
        replay_trajectory,
    )

    c = dict(char_cfg)
    ctrl = dict(c.get("controller", {}))
    env = dict(c.get("env", {}))
    ctrl["ramp_steps"] = ramp
    ctrl["hold_steps"] = hold
    env["probe_budget"] = ramp + hold + 2
    c["controller"] = ctrl
    c["env"] = env
    out: list[dict] = []
    for r in episode_rows:
        if max_episodes is not None and len(out) >= max_episodes:
            break
        init = episode_cell_from_char(c, int(r["episode_id"]))
        if not init:
            continue
        traj = replay_trajectory(init, char_cfg=c, fx_threshold=fx)
        if traj:
            out.append(traj)
    return out


def get_cohort(
    char_cfg, episode_rows, *, ramp, hold, fx, max_episodes, cache_dir, reuse, tag=""
) -> list[dict]:
    cache = cache_dir / f"ramp{ramp}_hold{hold}{tag}.json"
    if reuse and cache.exists():
        print(f"reuse cache {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))["episodes"]
    print(f"replay ramp={ramp} hold={hold} {tag} ...")
    eps = replay_cohort(
        char_cfg, episode_rows, ramp=ramp, hold=hold, fx=fx, max_episodes=max_episodes
    )
    cache.write_text(
        json.dumps({"ramp_steps": ramp, "hold_steps": hold, "episodes": eps}, indent=2),
        encoding="utf-8",
    )
    print(f"  cached {len(eps)} episodes -> {cache}")
    return eps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", type=int, default=8, help="fixed hold (dwell) length")
    p.add_argument("--ramps", type=str, default="3,6,12", help="ramp lengths to compare")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--phase", type=str, default="hold", choices=["hold", "ramp", "all"])
    p.add_argument("--skip-determinism", action="store_true")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument(
        "--cache-dir", type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/ari_replay_cache",
    )
    p.add_argument(
        "--out", type=Path,
        default=REPO / "results/phase1/mechanism_discrimination/partition_stability_ari_expA.json",
    )
    args = p.parse_args()

    flow_cfg = yaml.safe_load((REPO / "configs/landscape_flow.yaml").read_text(encoding="utf-8"))
    char_cfg = yaml.safe_load((REPO / flow_cfg["char_config"]).read_text(encoding="utf-8"))
    episode_rows = list(csv.DictReader((REPO / flow_cfg["input_labeled"]).open(encoding="utf-8")))
    fx = float(flow_cfg.get("fx_threshold", 8.0))
    ramps = [int(x) for x in args.ramps.split(",") if x.strip()]
    phase = None if args.phase == "all" else args.phase
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Assert single shared threshold set (prereg §2): fx is one scalar for all ramps.
    print(f"[guard] fixed shared thresholds: fx={fx} (single value for every ramp)")

    cohorts: dict[int, list[dict]] = {}
    for ramp in ramps:
        cohorts[ramp] = get_cohort(
            char_cfg, episode_rows, ramp=ramp, hold=args.hold, fx=fx,
            max_episodes=args.max_episodes, cache_dir=args.cache_dir, reuse=args.reuse_cache,
        )

    maps = {ramp: cohort_label_maps(eps, fx) for ramp, eps in cohorts.items()}

    # §3a determinism / self-replay floor: replay first ramp a 2nd independent pass.
    determinism = None
    if not args.skip_determinism:
        ramp0 = ramps[0]
        eps2 = get_cohort(
            char_cfg, episode_rows, ramp=ramp0, hold=args.hold, fx=fx,
            max_episodes=args.max_episodes, cache_dir=args.cache_dir,
            reuse=args.reuse_cache, tag="_selfreplay",
        )
        m2 = cohort_label_maps(eps2, fx)
        la, lb, ep_u, cov = matched_units(maps[ramp0], m2, phase=phase)
        d_ari = adjusted_rand_index(la, lb)
        determinism = {
            "ramp": ramp0,
            "ari_self_replay": d_ari,
            "coverage": cov,
            "passes_floor": bool(np.isfinite(d_ari) and d_ari >= 0.999),
            "note": (
                "Frozen §3a floor: identical protocol replayed twice must give ARI=1.0. "
                "If < 1.0 the engine is non-deterministic and the pipeline is suspect; "
                "switch to the pre-declared distributional-matching fallback."
            ),
        }
        print(f"[determinism] self-replay ARI(ramp={ramp0}) = {d_ari:.4f}")

    # Pairwise partition-stability comparisons across ramps (headline = hold phase).
    pairs: dict[str, dict] = {}
    for i, (ra, rb) in enumerate(combinations(sorted(cohorts), 2)):
        key = f"ramp{ra}_vs_ramp{rb}"
        pairs[key] = compare_pair(
            maps[ra], maps[rb], phase=phase, n_boot=args.n_boot, seed=100 + i
        )
        # Ramp-phase (partial overlap) reported separately, never headline.
        pairs[key]["ramp_phase_secondary"] = compare_pair(
            maps[ra], maps[rb], phase="ramp", n_boot=min(args.n_boot, 200), seed=900 + i
        )
        pr = pairs[key]
        print(
            f"[{key}] hold-phase ARI={pr['ari']:.4f} ({pr['steinley_band']})  "
            f"maxConf={pr['boundary_drift_matrix']['max_off_diagonal_confusion']*100:.1f}%  "
            f"f_cmp={pr['coverage']['comparable_sample_fraction']:.2f}  -> {pr['verdict']}"
        )

    miss = missingness_report(cohorts, fx, args.hold)

    # Aggregate verdict across ramp pairs (headline phase only).
    headline = [v for v in pairs.values()]
    reps = [v for v in headline if v["coverage"]["representative"]]
    if not reps:
        agg = "NOT_REPRESENTATIVE"
    elif all(v["verdict"] == "PARTITION_STABLE" for v in reps):
        agg = "PARTITION_STABLE"
    elif any(v["verdict"] == "PARTITION_DRIFTS" for v in reps):
        agg = "PARTITION_DRIFTS"
    elif any(v["verdict"] == "LOCALIZED_BOUNDARY_DRIFT" for v in reps):
        agg = "LOCALIZED_BOUNDARY_DRIFT"
    else:
        agg = "INCONCLUSIVE"

    report = {
        "experiment": "Exp-A: partition stability, fixed hold, varying ramp",
        "prereg": "docs/PARTITION_STABILITY_ARI_PREREG.md",
        "design": {
            "fixed_hold": args.hold,
            "ramps": ramps,
            "phase_headline": args.phase,
            "guard_thresholds": "fixed shared (fx=%.1f, mult 0.5/1.5, coupling 0.15, spread 0.002)" % fx,
            "alignment_key": "(episode_id, protocol_phase, phase_step_from_start)",
            "bootstrap_unit": "episode",
            "ari_bands_steinley2004": {"pass>=": STEINLEY_PASS, "fail<": STEINLEY_FAIL},
            "composite_pass": "ARI>=0.80 AND max_off_diagonal_confusion<=10%",
            "min_comparable_fraction": MIN_F_CMP,
        },
        "cohort_sizes": {str(r): len(e) for r, e in cohorts.items()},
        "determinism_floor": determinism,
        "pairwise": pairs,
        "missingness": miss,
        "aggregate_verdict": agg,
        "interpretation": (
            "Ramp changes the entry trajectory, not dwell; a stable partition here means "
            "the cell boundary tracks system geometry, not protocol timing. This does NOT "
            "by itself license writing O=(G,M) into Paper I unless aggregate_verdict is "
            "PARTITION_STABLE with a passing determinism floor and representative coverage."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", args.out)
    print("AGGREGATE VERDICT:", agg)


if __name__ == "__main__":
    main()
