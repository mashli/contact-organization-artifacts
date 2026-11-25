#!/usr/bin/env python3
"""Guard threshold sensitivity: sweep t* ∈ {4,6,8,10,12}N across 3 geometries.
Replays trajectories at each threshold and re-audits P1/P2/P3/P4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cge.emergence.organization_signature import audit_organization_signatures

THRESHOLDS = [4.0, 6.0, 8.0, 10.0, 12.0]

# Geometry → search paths (first found wins); block handled via CSV in load_episodes
GEOMETRY_PATHS: dict[str, list[str]] = {
    "cylinder": [
        "data/robotica/cylinder.json.gz",
        "results/phase1/landscape_analysis/landscape_trajectories.json",
        "results/phase1/cohort_independence/cylinder_heldout/full/landscape_trajectories.json",
        "results/phase1/cohort_independence/cylinder_heldout/pilot/landscape_trajectories.json",
    ],
    "sphere": [
        "data/robotica/sphere.json.gz",
        "results/phase1/cross_geometry_recovery/sphere/full/landscape_trajectories.json",
        "results/phase1/cross_geometry_recovery/sphere/pilot/landscape_trajectories.json",
    ],
    "block": [],  # special: loaded via CSV or data/robotica/block.json.gz
}


def _load_json_episodes(p: Path) -> list[dict]:
    if str(p).endswith(".gz"):
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return list(json.load(fh).get("episodes", []))
    return list(json.loads(p.read_text(encoding="utf-8")).get("episodes", []))


def load_episodes(geo: str) -> list[dict]:
    if geo == "block":
        for cand in (
            REPO / "data/robotica/block.json.gz",
            REPO / "data/robotica/block.json",
        ):
            if cand.is_file():
                print(f"  [block] loaded {cand}")
                return _load_json_episodes(cand)
        # Block uses states.csv; split into per-episode frame sequences
        csv_path = REPO / "results/phase1/characterization_block/full/states.csv"
        if not csv_path.is_file():
            csv_path = REPO / "results/phase1/characterization_block/pilot/states.csv"
        if csv_path.is_file():
            return _load_block_csv(csv_path)
        print(f"  [block] NOT FOUND")
        return []
    for path_str in GEOMETRY_PATHS.get(geo, []):
        p = REPO / path_str
        if p.is_file():
            print(f"  [{geo}] loaded {p} ({p.stat().st_size // 1024} KB)")
            return _load_json_episodes(p)
    print(f"  [{geo}] NOT FOUND")
    return []


def _load_block_csv(csv_path: Path) -> list[dict]:
    import csv
    rows = list(csv.DictReader(csv_path.open()))
    by_ep: dict[int, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(int(r["episode_id"]), []).append(r)
    episodes = []
    for ep_id, ep_rows in by_ep.items():
        frames = []
        for r in ep_rows:
            frames.append({
                "fx": float(r.get("tangential_force_norm", 0)),
                "sim_time": float(r.get("sim_time", 0)),
                "phase": r.get("phase", ""),
            })
        episodes.append({"episode_id": ep_id, "frames": frames})
    print(f"  [block] loaded {len(episodes)} episodes from {csv_path.name}")
    return episodes


def label_episode(ep: dict, fx_thresh: float):
    """Return list of 'loaded'/'grazing' labels per frame."""
    labels = []
    for fr in ep.get("frames", []):
        fx = float(fr.get("fx", 0))
        labels.append("loaded" if abs(fx) >= fx_thresh else "grazing")
    return labels


def dwell_stats(labels):
    if not labels:
        return 0.0, 0.0, 0, 0
    dwells_loaded, dwells_graz = [], []
    cur, dwell = labels[0], 1
    for lb in labels[1:]:
        if lb == cur:
            dwell += 1
        else:
            (dwells_loaded if cur == "loaded" else dwells_graz).append(dwell)
            cur, dwell = lb, 1
    (dwells_loaded if cur == "loaded" else dwells_graz).append(dwell)
    ml = float(np.mean(dwells_loaded)) if dwells_loaded else 0.0
    mg = float(np.mean(dwells_graz)) if dwells_graz else 0.0
    return ml, mg, len(dwells_loaded), len(dwells_graz)


def audit_at_threshold(episodes: list[dict], fx: float):
    """Compute P1/P2/P3/P4 at given threshold."""
    if not episodes:
        return None
    all_labels = []
    for ep in episodes:
        all_labels.extend(label_episode(ep, fx))
    n = len(all_labels)
    n_loaded = sum(1 for l in all_labels if l == "loaded")
    frac_loaded = n_loaded / max(n, 1)

    total_dwells_loaded, total_dwells_graz = [], []
    n_switch = 0
    n_at_boundary = 0
    for ep in episodes:
        labs = label_episode(ep, fx)
        ml, mg, _, _ = dwell_stats(labs)
        total_dwells_loaded.append(ml)
        total_dwells_graz.append(mg)
        for idx, (a, b) in enumerate(zip(labs, labs[1:]), start=1):
            if a != b:
                n_switch += 1
                # Check if near boundary.
                # NOTE: idx must be the actual switch position, not labs.index(a)
                # (the first occurrence of the label), otherwise the proximity
                # flag is computed at the wrong frame and collapses to 0.
                fx_prev = abs(float(ep["frames"][idx].get("fx", 0)))
                nxt = min(idx + 1, len(ep["frames"]) - 1)
                fx_curr = abs(float(ep["frames"][nxt].get("fx", 0)))
                if min(abs(fx_prev - fx), abs(fx_curr - fx)) < 2.0:
                    n_at_boundary += 1

    mean_loaded = float(np.mean(total_dwells_loaded))
    mean_graz = float(np.mean(total_dwells_graz))
    switch_rate = n_switch / max(n, 1)
    # P1: mean loaded dwell > mean grazing dwell
    p1 = mean_loaded > mean_graz
    # P2: switch rate < 0.20
    p2 = switch_rate < 0.20
    # P3: 0.15 < frac_loaded < 0.85
    p3 = 0.15 < frac_loaded < 0.85
    # P4: >50% of switches near boundary
    p4 = (n_at_boundary / max(n_switch, 1)) > 0.5 if n_switch > 0 else False
    return {
        "frac_loaded": round(frac_loaded, 4),
        "mean_loaded_dwell": round(mean_loaded, 2),
        "mean_grazing_dwell": round(mean_graz, 2),
        "dwell_ratio": round(mean_loaded / max(mean_graz, 0.01), 3),
        "switch_rate": round(switch_rate, 4),
        "P1": p1, "P2": p2, "P3": p3, "P4": p4,
        "n_switches": n_switch,
        "n_at_boundary": n_at_boundary,
        "n_episodes": len(episodes),
    }


def main() -> None:
    results = {"thresholds": THRESHOLDS, "geometries": []}
    for geo in GEOMETRY_PATHS:
        print(f"\nGeometry: {geo}")
        episodes = load_episodes(geo)
        geo_row = {"geometry": geo, "n_episodes_total": len(episodes), "thresholds": []}
        for t in THRESHOLDS:
            audit = audit_at_threshold(episodes, t)
            if audit:
                flag = "".join(f"{p}={'✓' if audit[p] else '✗'}" for p in ("P1", "P2", "P3", "P4"))
                print(f"  t*={t:4.1f}N: {flag} frac={audit['frac_loaded']:.1%} dwell_ratio={audit['dwell_ratio']:.2f}")
            else:
                print(f"  t*={t:4.1f}N: no data")
            geo_row["thresholds"].append(audit)
        results["geometries"].append(geo_row)

    out = REPO / "results/phase1/guard_threshold_sensitivity/guard_threshold_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
