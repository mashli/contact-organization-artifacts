#!/usr/bin/env python3
"""Signature-vs-friction audit: the canonical signature (Sig = P1--P4, ARI-based
P4) replayed over the friction arms and the cylinder reference cohort.

Why this exists
---------------
The manuscript's qualification operating map (Figure 3) is rendered with the
switch-proximity proxy P4' computed by the pooled-dwell re-implementation.
The paper's own signature (Sig) is defined by cge.emergence.organization_signature,
which scores P4 as a guard-alignment ARI and estimates residence times with
censoring-aware machinery. This script computes BOTH on the same relabelled
trajectories so that (a) the map can be re-rendered with the canonical
definition if desired, and (b) every cell where the two definitions disagree
is recorded rather than hidden.

No re-simulation. No threshold retuning. Frozen predicate definitions.

Output: results/robotica/friction_signature_audit.csv
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from _paths import DATA, PKG, ensure_src_on_path, resolve_data  # noqa: E402

ensure_src_on_path()
from cge.emergence.organization_signature import audit_from_path  # noqa: E402

THRESHOLDS = [4.0, 6.0, 8.0, 10.0, 12.0]
MU_TAGS = ["0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00"]
FRICTION_REL = "data/robotica/friction/mu_{tag}.json.gz"
CYLINDER_REL = "results/phase1/landscape_analysis/landscape_trajectories.json"


def load_obj(rel: str) -> dict:
    path = resolve_data(rel)
    if str(path).endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def relabel(obj: dict, thr: float, dst: Path) -> Path:
    data = json.loads(json.dumps(obj))
    for ep in data.get("episodes", []):
        for fr in ep.get("frames", []):
            fx = abs(float(fr.get("fx", 0.0)))
            fr["basin"] = "loaded" if fx >= thr else "grazing"
    dst.write_text(json.dumps(data))
    return dst


def proxy_flags(obj: dict, thr: float) -> dict:
    """Pooled-dwell proxy used for the operating map (P4' = switch proximity)."""
    labels, switches, near = [], 0, 0
    loaded_d, graz_d = [], []
    for ep in obj.get("episodes", []):
        labs = [
            "loaded" if abs(float(fr.get("fx", 0.0))) >= thr else "grazing"
            for fr in ep.get("frames", [])
        ]
        frames = ep.get("frames", [])
        labels.extend(labs)
        for i in range(1, len(labs)):
            if labs[i] != labs[i - 1]:
                switches += 1
                f0 = abs(float(frames[i - 1].get("fx", 0.0)))
                f1 = abs(float(frames[i].get("fx", 0.0)))
                if min(abs(f0 - thr), abs(f1 - thr)) < 2.0:
                    near += 1
        prev, dwell = (labs[0], 1) if labs else (None, 0)
        for b in labs[1:]:
            if b == prev:
                dwell += 1
            else:
                (loaded_d if prev == "loaded" else graz_d).append(dwell)
                prev, dwell = b, 1
        if labs:
            (loaded_d if prev == "loaded" else graz_d).append(dwell)
    n = max(len(labels), 1)
    frac = sum(1 for l in labels if l == "loaded") / n
    ml = float(np.mean(loaded_d)) if loaded_d else 0.0
    mg = float(np.mean(graz_d)) if graz_d else 0.0
    sw_rate = switches / n
    return {
        "proxy_P1": bool(ml > mg),
        "proxy_P2": bool(sw_rate < 0.20),
        "proxy_P3": bool(0.15 < frac < 0.85),
        "proxy_P4prime": bool(near / max(switches, 1) > 0.5),
        "proxy_dwell_ratio": round(ml / max(mg, 0.01), 3),
        "proxy_frac_loaded": round(frac, 4),
        "proxy_switch_rate": round(sw_rate, 4),
    }


def main() -> None:
    out_dir = PKG / "results" / "robotica"
    alt = Path(__file__).resolve().parents[2] / "results" / "robotica"
    if alt.parent.is_dir():
        out_dir = alt
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    cohorts = [(f"friction_mu_{t}", FRICTION_REL.format(tag=t), t) for t in MU_TAGS]
    cohorts.append(("cylinder_reference_n179", CYLINDER_REL, ""))

    rows = []
    for name, rel, tag in cohorts:
        obj = load_obj(rel)
        for thr in THRESHOLDS:
            p = relabel(obj, thr, tmp / f"{name}_{thr}.json")
            rep = audit_from_path(p, cohort_name=name, fx_threshold=thr)
            flags = {x.property_id: bool(x.pass_) for x in rep.properties}
            p1m = next(x for x in rep.properties if x.property_id == "P1").metrics
            px = proxy_flags(obj, thr)
            row = {
                "cohort": name,
                "mu": tag,
                "fx_threshold_N": thr,
                "n_episodes": rep.n_episodes,
                "n_frames": rep.n_frames,
                "P1": flags["P1"],
                "P2": flags["P2"],
                "P3": flags["P3"],
                "P4": flags["P4"],
                "Sig": rep.all_properties_pass,
                "lambda2": rep.lambda2_mode,
                "loaded_mean_tau": p1m.get("loaded_mean_tau"),
                "grazing_mean_tau": p1m.get("grazing_mean_tau"),
                **px,
            }
            row["proxy_qualified"] = bool(
                px["proxy_P1"] and px["proxy_P2"] and px["proxy_P3"] and px["proxy_P4prime"]
            )
            row["definitions_agree"] = bool(row["Sig"] == row["proxy_qualified"])
            rows.append(row)
            print(
                f"{name:22} t*={thr:<5} n={rep.n_episodes:<4} Sig={str(row['Sig']):5} "
                f"P1={str(flags['P1']):5} P4={str(flags['P4']):5} "
                f"proxy={str(row['proxy_qualified']):5} agree={row['definitions_agree']} "
                f"lam2={rep.lambda2_mode}"
            )

    fields = list(rows[0].keys())
    csv_path = out_dir / "friction_signature_audit.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    dis = [r for r in rows if not r["definitions_agree"]]
    print(f"\ndisagreement cells: {len(dis)} / {len(rows)}")
    for r in dis:
        print(
            f"  {r['cohort']:22} t*={r['fx_threshold_N']:<5} Sig={r['Sig']} proxy={r['proxy_qualified']} n={r['n_episodes']}"
        )
    (out_dir / "friction_signature_audit.json").write_text(
        json.dumps(
            {
                "note": "canonical signature vs pooled-dwell proxy on identical relabelled trajectories",
                "thresholds_N": THRESHOLDS,
                "n_cells": len(rows),
                "n_disagreements": len(dis),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
