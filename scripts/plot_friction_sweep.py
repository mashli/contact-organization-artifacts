#!/usr/bin/env python3
"""Generate friction sweep figure: P1/P2/P3/P4 vs μ, frac_loaded, dwell_ratio.
Output: results/phase1/friction_sweep/friction_sweep_figure.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    # Prefer n50 data; fall back to legacy sweep
    n50_path = REPO / "results/phase1/friction_sweep_n50/friction_sweep_analysis.json"
    data_path = n50_path if n50_path.is_file() else REPO / "results/phase1/friction_sweep/friction_sweep_analysis.json"
    data = json.loads(data_path.read_text()).get("arms", [])
    rows = [r for r in data if r.get("status") == "ok"]
    if not rows:
        print("No data")
        sys.exit(1)

    mus = np.array([r["mu"] for r in rows])
    frac = np.array([r["frac_loaded"] for r in rows])
    dr = np.array([r["dwell_ratio"] for r in rows])
    p1 = np.array([int(r["P1"]) for r in rows])
    p2 = np.array([int(r["P2"]) for r in rows])
    p3 = np.array([int(r["P3"]) for r in rows])
    p4 = np.array([int(r["P4"]) for r in rows])
    sw = np.array([r["switch_rate"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Contact Organization vs Friction Coefficient μ", fontsize=13)

    # Panel A: frac_loaded
    ax = axes[0, 0]
    ax.plot(mus, frac * 100, "o-", color="steelblue", lw=2, ms=6)
    ax.axhline(50, color="gray", ls="--", lw=0.8, label="uniform (50%)")
    ax.axvline(0.6, color="orange", ls=":", lw=1.5, label="reference μ=0.6")
    ax.set_xlabel("Friction μ")
    ax.set_ylabel("frac_loaded (%)")
    ax.set_title("P3: Non-uniform occupancy")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # Panel B: dwell ratio (loaded/grazing)
    ax = axes[0, 1]
    colors = ["crimson" if p else "gray" for p in p1]
    ax.bar(mus, dr, width=0.06, color=colors, alpha=0.7)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="equal dwell")
    ax.axvline(0.6, color="orange", ls=":", lw=1.5, label="reference μ=0.6")
    ax.set_xlabel("Friction μ")
    ax.set_ylabel("mean_loaded_dwell / mean_grazing_dwell")
    ax.set_title("P1: Metastable dwell (red = pass)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3)
    ax.grid(alpha=0.3, axis="y")

    # Panel C: switch rate
    ax = axes[1, 0]
    ax.plot(mus, sw, "s-", color="seagreen", lw=2, ms=6)
    ax.axhline(0.20, color="red", ls="--", lw=1, label="threshold (0.20)")
    ax.axvline(0.6, color="orange", ls=":", lw=1.5, label="reference μ=0.6")
    ax.set_xlabel("Friction μ")
    ax.set_ylabel("Switch rate (switches/step)")
    ax.set_title("P2: Sparse transitions (all pass)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.25)
    ax.grid(alpha=0.3)

    # Panel D: P1/P2/P3/P4 heatmap
    ax = axes[1, 1]
    mat = np.column_stack([p1, p2, p3, p4]).T
    labels_col = [f"μ={m}" for m in mus]
    labels_row = ["P1 (dwell)", "P2 (switch)", "P3 (occpancy)", "P4 (boundary)"]
    cmap = matplotlib.colors.ListedColormap([[0.9, 0.9, 0.9], [0.2, 0.7, 0.3]])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(mus)))
    ax.set_xticklabels(labels_col, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels_row, fontsize=8)
    ax.set_title("P1–P4 Audit Results (green = pass)")
    for i in range(4):
        for j in range(len(mus)):
            ax.text(j, i, "✓" if mat[i, j] else "✗", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.7)

    plt.tight_layout()
    out = REPO / "results/phase1/friction_sweep/friction_sweep_figure.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close()


if __name__ == "__main__":
    main()
