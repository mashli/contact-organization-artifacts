#!/usr/bin/env python3
"""Qualification operating map: P1--P4 over (mu, t*) for the push--roll protocol.

Replays existing (never re-simulated) trajectories from the friction sweep
cohort at every declared load threshold and re-audits the four qualification
predicates. No trajectory is regenerated; no threshold is retuned after the
fact; every cell of the map is computed with the frozen predicate definitions
used for the manuscript's signature round.

Outputs
-------
results/robotica/qualification_map_mu_threshold.csv
results/robotica/qualification_map_mu_threshold.json

Self-check: the (mu, t*=8 N) slice must reproduce the frozen
results/phase1/friction_sweep_n50/friction_sweep_analysis.json numbers.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_friction_sweep import load_trajectories  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sens", REPO / "scripts/analyze_guard_threshold_sensitivity.py"
)
sens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sens)

MUS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
THRESHOLDS = [4.0, 6.0, 8.0, 10.0, 12.0]

# The cylinder cohort used throughout the manuscript (n=179): same file the
# frozen scripts/robotica/run_threshold_shell_n179.py replays.
CYLINDER_PATHS = [
    "results/phase1/landscape_analysis/landscape_trajectories.json",
]


def label_episode(ep: dict, fx_thresh: float) -> list[str]:
    return [
        "loaded" if float(fr.get("fx", 0)) >= fx_thresh else "grazing"
        for fr in ep.get("frames", [])
    ]


def compute_metrics(episodes: list[dict], fx_thresh: float) -> dict | None:
    """Frozen P1--P4 predicates, parameterised only by the declared shell t*."""
    if not episodes:
        return None

    all_basins: list[str] = []
    for ep in episodes:
        all_basins.extend(label_episode(ep, fx_thresh))

    n_total = len(all_basins)
    n_loaded = sum(1 for b in all_basins if b == "loaded")
    frac_loaded = n_loaded / max(n_total, 1)

    loaded_dwells: list[int] = []
    graz_dwells: list[int] = []
    for ep in episodes:
        basins = label_episode(ep, fx_thresh)
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
        basins = label_episode(ep, fx_thresh)
        frames = ep.get("frames", [])
        n_switch += sum(1 for a, b in zip(basins, basins[1:]) if a != b)
        for i in range(1, len(basins)):
            if basins[i] != basins[i - 1]:
                fx_prev = float(frames[i - 1].get("fx", 0))
                fx_curr = float(frames[i].get("fx", 0))
                if abs(fx_prev - fx_thresh) < 2.0 or abs(fx_curr - fx_thresh) < 2.0:
                    n_at_bdry += 1
    switch_rate = n_switch / max(n_total, 1)

    return {
        "n_episodes": len(episodes),
        "n_frames": n_total,
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
    }


def load_cylinder() -> tuple[list[dict] | None, str]:
    # Public artifacts first
    for rel in (
        "data/robotica/cylinder.json.gz",
        "data/robotica/cylinder.json",
        *CYLINDER_PATHS,
    ):
        p = REPO / rel
        if p.is_file():
            if str(p).endswith(".gz"):
                import gzip

                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    return list(json.load(fh).get("episodes", [])), rel
            return list(json.loads(p.read_text(encoding="utf-8")).get("episodes", [])), rel
    return None, ""


def cylinder_metrics(episodes: list[dict], fx: float) -> dict:
    """Author's own threshold-shell audit (audit_at_threshold) on the n=179 cohort.

    Reproduces the manuscript's Table tab_threshold_shell numbers exactly and
    additionally reports the switch-proximity flag with the corrected switch
    index. NOTE: this proximity flag is NOT signature P4 (which is an
    ARI-based guard-alignment score in cge.emergence.organization_signature).
    """
    a = sens.audit_at_threshold(episodes, fx)
    out = {
        "n_episodes": len(episodes),
        "n_frames": sum(len(ep.get("frames", [])) for ep in episodes),
        "frac_loaded": a["frac_loaded"],
        "mean_loaded_dwell": a["mean_loaded_dwell"],
        "mean_grazing_dwell": a["mean_grazing_dwell"],
        "dwell_ratio": a["dwell_ratio"],
        "switch_rate": a["switch_rate"],
        "P1": bool(a["P1"]),
        "P2": bool(a["P2"]),
        "P3": bool(a["P3"]),
        "P4_proximity": bool(a["P4"]),
        "n_switches": a.get("n_switches", ""),
        "n_at_boundary": a.get("n_at_boundary", ""),
    }
    return out


def main() -> int:
    out_dir = REPO / "results/robotica"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # --- friction sweep arms: mu x t* -------------------------------------
    for mu in MUS:
        eps, src = load_trajectories(mu, prefer_tier="n50")
        if eps is None:
            print(f"mu={mu}: MISSING")
            continue
        for t in THRESHOLDS:
            m = compute_metrics(eps, t)
            assert m is not None
            row = {
                "cohort": "friction_sweep",
                "mu": mu,
                "fx_threshold_N": t,
                "source": src,
                "status": "ok" if m["n_episodes"] >= 50 else "undersampled",
                **m,
            }
            row["P4_proximity"] = row.pop("P4")
            rows.append(row)
            print(
                f"mu={mu:<4} t*={t:<5} n={m['n_episodes']:<4} "
                f"P1={int(m['P1'])} P2={int(m['P2'])} P3={int(m['P3'])} "
                f"P4prox={int(row['P4_proximity'])} "
                f"ratio={m['dwell_ratio']:<6} frac={m['frac_loaded']:.3f}"
            )

    # --- cylinder reference cohort (the one qualified witness) ------------
    cyl, cyl_src = load_cylinder()
    if cyl:
        for t in THRESHOLDS:
            m = cylinder_metrics(cyl, t)
            row = {
                "cohort": "cylinder_reference_n179",
                "mu": "",
                "fx_threshold_N": t,
                "source": cyl_src,
                "status": "ok",
                **m,
            }
            rows.append(row)
            print(
                f"cylinder  t*={t:<5} n={m['n_episodes']:<4} "
                f"P1={int(m['P1'])} P2={int(m['P2'])} P3={int(m['P3'])} "
                f"P4prox={int(m['P4_proximity'])} "
                f"ratio={m['dwell_ratio']:<6} frac={m['frac_loaded']:.3f}"
            )
    else:
        print("cylinder reference cohort: MISSING")

    # --- write ------------------------------------------------------------
    csv_path = out_dir / "qualification_map_mu_threshold.csv"
    fields = [
        "cohort", "mu", "fx_threshold_N", "n_episodes", "n_frames", "frac_loaded",
        "mean_loaded_dwell", "mean_grazing_dwell", "dwell_ratio", "switch_rate",
        "P1", "P2", "P3", "P4_proximity", "qualified", "n_switches", "n_at_boundary",
        "status", "source",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["qualified"] = bool(r["P1"] and r["P2"] and r["P3"] and r["P4_proximity"])
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {csv_path}")

    # --- self-check against the frozen t*=8 N slice -----------------------
    frozen_path = REPO / "data/robotica/friction_sweep_analysis.json"
    if not frozen_path.is_file():
        frozen_path = REPO / "results/phase1/friction_sweep_n50/friction_sweep_analysis.json"
    check = {"checked": False}
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text())
        fmap = {round(float(a["mu"]), 3): a for a in frozen.get("arms", []) if a.get("status") != "missing"}
        diffs = []
        for r in rows:
            if r["cohort"] != "friction_sweep" or r["fx_threshold_N"] != 8.0:
                continue
            f = fmap.get(round(float(r["mu"]), 3))
            if not f:
                continue
            for k in ("n_episodes", "frac_loaded", "dwell_ratio", "switch_rate", "n_switches", "n_at_boundary"):
                if abs(float(f[k]) - float(r[k])) > 1e-6:
                    diffs.append({"mu": r["mu"], "field": k, "frozen": f[k], "recomputed": r[k]})
            for k in ("P1", "P2", "P3"):
                if bool(f[k]) != bool(r[k]):
                    diffs.append({"mu": r["mu"], "field": k, "frozen": f[k], "recomputed": r[k]})
        check = {"checked": True, "n_cells_compared": len([r for r in rows if r["cohort"] == "friction_sweep" and r["fx_threshold_N"] == 8.0]), "n_diffs": len(diffs), "diffs": diffs}
        print(f"self-check vs frozen t*=8N slice: {check['n_diffs']} diffs over {check['n_cells_compared']} arms")
    else:
        print("self-check skipped: frozen friction_sweep_analysis.json not found")

    payload = {
        "description": "P1--P4 qualification operating map over (mu, t*) for the push--roll protocol",
        "note": "existing trajectories only; no re-simulation; frozen predicate definitions",
        "thresholds_N": THRESHOLDS,
        "mu": MUS,
        "self_check": check,
        "rows": rows,
    }
    json_path = out_dir / "qualification_map_mu_threshold.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
