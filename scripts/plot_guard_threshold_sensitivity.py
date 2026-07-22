#!/usr/bin/env python3
"""Guard threshold sensitivity heatmap: P1/P2/P3/P4 vs t* × geometry.
Output: results/phase1/guard_threshold_sensitivity/guard_threshold_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    data = json.loads((REPO / "results/phase1/guard_threshold_sensitivity/guard_threshold_sensitivity.json").read_text())
    thresholds = data["thresholds"]
    geos = [g["geometry"] for g in data["geometries"]]
    n_t = len(thresholds)
    n_g = len(geos)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("OPCG Audit vs Guard Threshold t* across Geometries", fontsize=12)

    props = ["P1", "P2", "P3", "P4"]
    titles = ["P1: Metastable dwell", "P2: Sparse switch", "P3: Non-uniform", "P4: Guard-aligned"]
    cmap = matplotlib.colors.ListedColormap([[0.85, 0.85, 0.85], [0.2, 0.7, 0.2]])

    for ax, prop, title in zip(axes, props, titles):
        mat = np.zeros((n_g, n_t))
        for gi, geo_row in enumerate(data["geometries"]):
            for ti, audit in enumerate(geo_row["thresholds"]):
                if audit:
                    mat[gi, ti] = 1 if audit[prop] else 0
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n_t))
        ax.set_xticklabels([f"{t:.0f}" for t in thresholds], fontsize=8)
        ax.set_yticks(range(n_g))
        ax.set_yticklabels(geos, fontsize=9)
        ax.set_xlabel("t* (N)")
        ax.set_title(title, fontsize=9)
        # Annotate
        for gi in range(n_g):
            for ti in range(n_t):
                audit = data["geometries"][gi]["thresholds"][ti]
                if audit:
                    dr = audit["dwell_ratio"]
                    ax.text(ti, gi, f"{dr:.2f}" if prop == "P1" else ("✓" if audit[prop] else "✗"),
                            ha="center", va="center", fontsize=7,
                            color="white" if mat[gi, ti] == 1 else "gray")
        # Reference line at t*=8
        ref_idx = thresholds.index(8.0) if 8.0 in thresholds else 2
        ax.axvline(ref_idx - 0.5, color="orange", ls="--", lw=1.5, alpha=0.7)

    plt.tight_layout()
    out = REPO / "results/phase1/guard_threshold_sensitivity/guard_threshold_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close()


if __name__ == "__main__":
    main()
