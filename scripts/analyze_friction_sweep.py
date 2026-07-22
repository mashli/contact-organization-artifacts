#!/usr/bin/env python3
"""Analyze friction sweep: P1/P2/P3/P4 vs μ from landscape_trajectories.json.

Prefer tier=n50 under results/phase1/friction_sweep_n50/ when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CGE_SRC = REPO / "src"
if str(CGE_SRC) not in sys.path:
    sys.path.insert(0, str(CGE_SRC))

MUS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FX = 8.0

# Map mu → possible directory tags
_MU_TAGS = {
    0.0: ["0.00", "0.0"],
    0.2: ["0.20", "0.2"],
    0.3: ["0.30", "0.3"],
    0.4: ["0.40", "0.4"],
    0.5: ["0.50", "0.5"],
    0.6: ["0.60", "0.6"],
    0.7: ["0.70", "0.7"],
    0.8: ["0.80", "0.8"],
    0.9: ["0.90", "0.9"],
    1.0: ["1.00", "1.0"],
}


def load_trajectories(mu: float, *, prefer_tier: str = "n50") -> tuple[list[dict] | None, str]:
    tags = _MU_TAGS.get(mu, [f"{mu:.2f}"])
    # Also try alternate tags for each root
    candidates: list[tuple[str, str]] = []
    for tag in tags:
        candidates.append((f"friction_sweep_n50/friction_mu_{tag}/{prefer_tier}", prefer_tier))
    for tag in tags:
        candidates.append((f"friction_sweep/friction_mu_{tag}/pilot", "pilot"))
        candidates.append((f"emergence_ablation/friction_mu_{tag}/pilot", "pilot-legacy"))

    for subdir, src in candidates:
        path = REPO / f"results/phase1/{subdir}/landscape_trajectories.json"
        if path.is_file():
            return json.loads(path.read_text()).get("episodes", []), f"{subdir} ({src})"
    return None, ""


def label_episode(ep: dict, fx_thresh: float = FX) -> list[str]:
    basins = []
    for fr in ep.get("frames", []):
        fx = float(fr.get("fx", 0))
        basins.append("loaded" if fx >= fx_thresh else "grazing")
    return basins


def compute_metrics(episodes: list[dict]):
    if not episodes:
        return None

    all_basins = []
    for ep in episodes:
        basins = label_episode(ep)
        all_basins.extend(basins)

    n_total = len(all_basins)
    n_loaded = sum(1 for b in all_basins if b == "loaded")
    frac_loaded = n_loaded / max(n_total, 1)

    loaded_dwells, graz_dwells = [], []
    for ep in episodes:
        basins = label_episode(ep)
        if not basins:
            continue
        prev = basins[0]
        dwell = 1
        for b in basins[1:]:
            if b == prev:
                dwell += 1
            else:
                (loaded_dwells if prev == "loaded" else graz_dwells).append(dwell)
                prev = b
                dwell = 1
        (loaded_dwells if prev == "loaded" else graz_dwells).append(dwell)

    mean_loaded = float(np.mean(loaded_dwells)) if loaded_dwells else 0.0
    mean_graz = float(np.mean(graz_dwells)) if graz_dwells else 0.0
    dwell_ratio = mean_loaded / max(mean_graz, 0.01)

    n_switch = 0
    n_at_bdry = 0
    for ep in episodes:
        basins = label_episode(ep)
        frames = ep.get("frames", [])
        n_switch += sum(1 for a, b in zip(basins, basins[1:]) if a != b)
        for i in range(1, len(basins)):
            if basins[i] != basins[i - 1]:
                fx_prev = float(frames[i - 1].get("fx", 0))
                fx_curr = float(frames[i].get("fx", 0))
                if abs(fx_prev - FX) < 2.0 or abs(fx_curr - FX) < 2.0:
                    n_at_bdry += 1
    switch_rate = n_switch / max(n_total, 1)

    return {
        "n_episodes": len(episodes),
        "frac_loaded": round(frac_loaded, 4),
        "mean_loaded_dwell": round(mean_loaded, 2),
        "mean_grazing_dwell": round(mean_graz, 2),
        "dwell_ratio": round(dwell_ratio, 3),
        "switch_rate": round(switch_rate, 4),
        "P1": bool(mean_loaded > mean_graz),
        "P2": bool(switch_rate < 0.20),
        "P3": bool(0.15 < frac_loaded < 0.85),
        "P4": bool(n_at_bdry / max(n_switch, 1) > 0.5),
        "n_switches": n_switch,
        "n_at_boundary": n_at_bdry,
        "total_frames": n_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer-tier", default="n50")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/phase1/friction_sweep/friction_sweep_analysis.json",
    )
    args = ap.parse_args()

    rows = []
    for mu in MUS:
        eps, src = load_trajectories(mu, prefer_tier=args.prefer_tier)
        if eps is None:
            rows.append({"mu": mu, "status": "missing"})
            print(f"mu={mu}: MISSING")
            continue
        m = compute_metrics(eps)
        assert m is not None
        m["mu"] = mu
        m["source"] = src
        m["status"] = "ok" if m["n_episodes"] >= 50 else "undersampled"
        rows.append(m)
        flag = "".join(f"{p}={'Y' if m[p] else 'N'}" for p in ("P1", "P2", "P3", "P4"))
        print(
            f"mu={mu}: n={m['n_episodes']} {flag} frac={m['frac_loaded']:.1%} "
            f"ratio={m['dwell_ratio']:.2f} [{m['status']}] src={src}"
        )

    payload = {
        "prefer_tier": args.prefer_tier,
        "fx_threshold": FX,
        "target_n_per_arm": 50,
        "arms": rows,
        "n_arms_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_arms_undersampled": sum(1 for r in rows if r.get("status") == "undersampled"),
    }
    # Also write under n50 root when that data exists
    out_paths = [args.out]
    n50_out = REPO / "results/phase1/friction_sweep_n50/friction_sweep_analysis.json"
    out_paths.append(n50_out)
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
