#!/usr/bin/env python3
"""Block recovery audit: compute RCI vs cylinder reference using available channels.
Block only has CSV states → contact_graph + energy channels (no vision/duration images).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ── Load cylinder reference (for comparison) ──────────────────────────────────
CYLINDER_PATH = REPO / "results/phase1/cohort_independence/cylinder_heldout/full/landscape_trajectories.json"
BLOCK_PATH = REPO / "results/phase1/characterization_block/full/landscape_trajectories.json"
FX = 8.0


def load_episodes(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("episodes", [])


def label_episode(ep: dict, fx_thresh: float = FX) -> list[str]:
    return [("loaded" if abs(float(fr.get("fx", 0))) >= fx_thresh else "grazing")
            for fr in ep.get("frames", [])]


def occupancy_pi(episodes: list[dict], fx_thresh: float = FX) -> float:
    """Proportion index: frac_loaded."""
    all_labels = []
    for ep in episodes:
        all_labels.extend(label_episode(ep, fx_thresh))
    n = len(all_labels)
    if n == 0:
        return float("nan")
    return sum(1 for l in all_labels if l == "loaded") / n


def transition_matrix(episodes: list[dict], fx_thresh: float = FX) -> tuple[np.ndarray, float]:
    """Build 2×2 loaded/grazing transition matrix, return (P, lambda2)."""
    C = np.zeros((2, 2))
    idx = {"grazing": 0, "loaded": 1}
    for ep in episodes:
        labs = label_episode(ep, fx_thresh)
        for a, b in zip(labs, labs[1:]):
            if a in idx and b in idx:
                C[idx[a], idx[b]] += 1
    rs = C.sum(axis=1, keepdims=True)
    P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
    ev = sorted([float(x.real) for x in np.linalg.eigvals(P)], reverse=True)
    l2 = ev[1] if len(ev) > 1 else 0.0
    return P, l2


def transition_metric(episodes: list[dict], fx_thresh: float = FX) -> float:
    """Transience metric: fraction of transitions near guard boundary."""
    n_trans, n_boundary = 0, 0
    for ep in episodes:
        labs = label_episode(ep, fx_thresh)
        frames = ep.get("frames", [])
        for i in range(1, len(labs)):
            if labs[i] != labs[i - 1]:
                n_trans += 1
                fx_prev = abs(float(frames[i - 1].get("fx", 0)))
                fx_curr = abs(float(frames[i].get("fx", 0)))
                if min(abs(fx_prev - fx_thresh), abs(fx_curr - fx_thresh)) < 2.0:
                    n_boundary += 1
    return n_boundary / max(n_trans, 1)


def _extract_features(fr: dict) -> list[float]:
    """Extract scalar features from frame dict (works for both block CSV-derived and cylinder)."""
    f = fr.get("features", {})
    if isinstance(f, dict):
        # Cylinder landscape_trajectories.json format
        return [
            float(f.get("multiplicity_M", 0)),
            float(f.get("patch_centroid_c_x", 0)),
            float(f.get("patch_centroid_c_y", 0)),
        ]
    elif isinstance(f, (list, tuple)):
        return list(f[:3])
    return [0.0, 0.0, 0.0]


def partition_ari(episodes: list[dict], fx_thresh: float = FX) -> float:
    """ARI vs true loaded/grazing labels using per-frame features."""
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return float("nan")

    true_labels, pred_labels = [], []
    for ep in episodes:
        labs = label_episode(ep, fx_thresh)
        if not labs:
            continue
        feats = np.array([_extract_features(fr) for fr in ep.get("frames", [])])
        if len(feats) < 2:
            continue
        from sklearn.cluster import KMeans
        try:
            pred = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(feats)
        except Exception:
            continue
        cluster_to_label: dict[int, list[str]] = {}
        for lb, cl in zip(labs, pred):
            cluster_to_label.setdefault(cl, []).append(lb)
        label_map = {cl: max(lbs, key=lbs.count) for cl, lbs in cluster_to_label.items()}
        mapped = [label_map[c] for c in pred]
        true_labels.extend(labs)
        pred_labels.extend(mapped)

    if len(set(true_labels)) < 2:
        return float("nan")
    return adjusted_rand_score(true_labels, pred_labels)


def dwell_ratio(episodes: list[dict], fx_thresh: float = FX) -> float:
    """Mean loaded dwell / mean grazing dwell."""
    loaded_dwells, graz_dwells = [], []
    for ep in episodes:
        labs = label_episode(ep, fx_thresh)
        if not labs:
            continue
        cur, dwell = labs[0], 1
        for lb in labs[1:]:
            if lb == cur:
                dwell += 1
            else:
                (loaded_dwells if cur == "loaded" else graz_dwells).append(dwell)
                cur, dwell = lb, 1
        (loaded_dwells if cur == "loaded" else graz_dwells).append(dwell)
    ml = float(np.mean(loaded_dwells)) if loaded_dwells else 0.0
    mg = float(np.mean(graz_dwells)) if graz_dwells else 0.0
    return ml / max(mg, 0.01)


def boundary_match(episodes: list[dict], fx_thresh: float = FX) -> bool:
    """Boundary: loaded episodes have higher mean fx than grazing."""
    loaded_fx, graz_fx = [], []
    for ep in episodes:
        labs = label_episode(ep, fx_thresh)
        fxs = [abs(float(fr.get("fx", 0))) for fr in ep.get("frames", [])]
        if not labs or not fxs:
            continue
        loaded_fx.extend([fxs[i] for i in range(len(labs)) if labs[i] == "loaded"])
        graz_fx.extend([fxs[i] for i in range(len(labs)) if labs[i] == "grazing"])
    if not loaded_fx or not graz_fx:
        return False
    return float(np.mean(loaded_fx)) > float(np.mean(graz_fx))


def audit_episodes(episodes: list[dict], name: str) -> dict[str, Any]:
    pi = occupancy_pi(episodes)
    _, l2 = transition_matrix(episodes)
    trans = transition_metric(episodes)
    ari = partition_ari(episodes)
    dr = dwell_ratio(episodes)
    bnd = boundary_match(episodes)

    print(f"\n{name}:")
    print(f"  PI (frac_loaded) = {pi:.3f}")
    print(f"  λ2 (spectral)    = {l2:.4f}")
    print(f"  Trans (boundary) = {trans:.3f}")
    print(f"  ARI (partition)  = {ari:.3f}" if ari == ari else f"  ARI = nan")
    print(f"  Dwell ratio      = {dr:.3f}")
    print(f"  Boundary match   = {bnd}")

    return {
        "occupancy_pi": round(pi, 4) if pi == pi else float("nan"),
        "lambda2": round(l2, 4),
        "transition_metric": round(trans, 4),
        "partition_ari": round(ari, 4) if ari == ari else float("nan"),
        "dwell_ratio": round(dr, 3),
        "boundary_match": bnd,
    }


def main() -> None:
    print("Loading cylinder reference ...")
    cyl_eps = load_episodes(CYLINDER_PATH)
    print(f"  {len(cyl_eps)} episodes")

    print("Loading block data ...")
    blk_eps = load_episodes(BLOCK_PATH)
    print(f"  {len(blk_eps)} episodes")

    cyl_audit = audit_episodes(cyl_eps, "Cylinder (reference)")
    blk_audit = audit_episodes(blk_eps, "Block")

    # RCI-like comparison: per-layer match vs cylinder reference
    print("\n=== Block vs Cylinder RCI-like comparison ===")

    # Occupancy: PI within 0.15 of cylinder
    pi_diff = abs(cyl_audit["occupancy_pi"] - blk_audit["occupancy_pi"])
    occ_match = pi_diff < 0.15

    # Partition: ARI available for both
    ari_match = (
        not (cyl_audit["partition_ari"] != cyl_audit["partition_ari"]) and
        not (blk_audit["partition_ari"] != blk_audit["partition_ari"]) and
        abs(cyl_audit["partition_ari"] - blk_audit["partition_ari"]) < 0.30
    )

    # Transition: λ2 within 0.15
    l2_match = abs(cyl_audit["lambda2"] - blk_audit["lambda2"]) < 0.15

    # Boundary: both True or both False
    bnd_match = cyl_audit["boundary_match"] == blk_audit["boundary_match"]

    n_match = sum(1 for m in [occ_match, ari_match, l2_match, bnd_match] if m)
    rci = n_match / 4.0

    print(f"\n  Occupancy (PI):   {'✓' if occ_match else '✗'} (Δ={pi_diff:.3f})")
    print(f"  Partition (ARI):  {'✓' if ari_match else '✗'}")
    print(f"  Transition (λ2):  {'✓' if l2_match else '✗'} (Δ={abs(cyl_audit['lambda2'] - blk_audit['lambda2']):.4f})")
    print(f"  Boundary:         {'✓' if bnd_match else '✗'}")
    print(f"\n  RCI (4-layer) = {rci:.3f}")

    out = REPO / "results/phase1/cross_geometry_recovery/block/recovery_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "cylinder_reference": cyl_audit,
        "block_target": blk_audit,
        "layer_matches": {
            "occupancy": occ_match,
            "partition": ari_match,
            "transition": l2_match,
            "boundary": bnd_match,
        },
        "RCI": round(rci, 4),
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
