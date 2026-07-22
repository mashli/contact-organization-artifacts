#!/usr/bin/env python3
"""Compare two Recovery Matrix audit summaries → Recovery Consistency Index (RCI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LAYERS = ("occupancy", "partition", "transition", "boundary")
CHANNELS = ("vision", "duration", "contact_graph", "energy")

# Preregistered thresholds (frozen — must match audits)
PI_PASS = 0.70
PI_PARTIAL = 0.50
ARI_PASS = 0.30
ARI_PARTIAL = 0.15
TRANS_PASS = 0.70
TRANS_PARTIAL = 0.50


def _classify_pi(v: float) -> str:
    if not (v == v):  # nan
        return "fail"
    if v >= PI_PASS:
        return "pass"
    if v >= PI_PARTIAL:
        return "partial"
    return "fail"


def _classify_ari(v: float) -> str:
    if not (v == v):
        return "fail"
    if v >= ARI_PASS:
        return "pass"
    if v >= ARI_PARTIAL:
        return "partial"
    return "fail"


def _classify_trans(v: float) -> str:
    if not (v == v):
        return "fail"
    if v >= TRANS_PASS:
        return "pass"
    if v >= TRANS_PARTIAL:
        return "partial"
    return "fail"


def _classify_boundary(check: bool | None) -> str:
    if check is True:
        return "pass"
    return "fail"


def best_method_metrics(report: dict, channel: str) -> dict[str, Any]:
    """Extract metrics for preregistered best method per channel."""
    if channel == "vision":
        verdict = report.get("VLA_verdict") or {}
        best = verdict.get("best_method")
        by = {m["feature_method"]: m for m in report.get("by_method", [])}
        row = by.get(best) or report.get("by_method", [{}])[0]
        v = row.get("VLA_verdict") or verdict
        return {
            "method": best or row.get("feature_method"),
            "pi": float(v.get("pi_spearman_vs_force_lc", float("nan"))),
            "ari": float(v.get("k2_ari_vs_fx_basin", float("nan"))),
            "trans": float(v.get("transition_spearman_vs_force_lc", float("nan"))),
            "boundary": (v.get("checks") or {}).get("boundary_ratio_ge_5x"),
        }
    key = {"duration": "DZLA", "contact_graph": "CGLA", "energy": "ELA"}[channel]
    verdict_key = f"{key}_verdict"
    verdict = report.get(verdict_key) or {}
    best = verdict.get("best_method")
    by = {m["feature_method"]: m for m in report.get("by_method", [])}
    row = by.get(best) or (report.get("by_method") or [{}])[0]
    v = row.get(verdict_key) or verdict
    return {
        "method": best or row.get("feature_method"),
        "pi": float(v.get("pi_spearman_vs_force_lc", float("nan"))),
        "ari": float(v.get("k2_ari_vs_fx_basin", float("nan"))),
        "trans": float(v.get("transition_spearman_vs_force_lc", float("nan"))),
        "boundary": (v.get("checks") or {}).get("boundary_ratio_ge_5x"),
    }


def layer_classes(metrics: dict) -> dict[str, str]:
    return {
        "occupancy": _classify_pi(metrics["pi"]),
        "partition": _classify_ari(metrics["ari"]),
        "transition": _classify_trans(metrics["trans"]),
        "boundary": _classify_boundary(metrics.get("boundary")),
    }


def load_channel_reports(summary_or_dir: Path) -> dict[str, dict]:
    if summary_or_dir.is_file():
        payload = json.loads(summary_or_dir.read_text())
        if "channels" in payload:
            ch = payload["channels"]
            if ch and isinstance(next(iter(ch.values())), dict) and "report_path" in next(iter(ch.values())):
                out: dict[str, dict] = {}
                root = Path(__file__).resolve().parents[2]
                for name, meta in ch.items():
                    p = Path(meta["report_path"])
                    if not p.is_absolute():
                        p = root / "contact_grammar_emergence" / p
                    out[name] = json.loads(p.read_text())
                return out
        summary_or_dir = summary_or_dir.parent

    candidates = {
        "vision": [
            "vision_audit/full_v2/vla_report.json",
            "vision_audit/vla_report.json",
        ],
        "duration": [
            "duration_zeta_audit/full/dzla_report.json",
            "duration_zeta_audit/dzla_report.json",
        ],
        "contact_graph": [
            "contact_graph_audit/full/cgla_report.json",
            "contact_graph_audit/cgla_report.json",
        ],
        "energy": [
            "energy_audit/full/ela_report.json",
            "energy_audit/ela_report.json",
        ],
    }
    out: dict[str, dict] = {}
    for ch, rels in candidates.items():
        for rel in rels:
            p = summary_or_dir / rel
            if p.exists():
                out[ch] = json.loads(p.read_text())
                break
        if ch not in out:
            alt = summary_or_dir / f"{ch}_report.json"
            if alt.exists():
                out[ch] = json.loads(alt.read_text())
    return out


def build_matrix(reports: dict[str, dict]) -> dict[str, dict[str, str]]:
    matrix: dict[str, dict[str, str]] = {}
    for ch in CHANNELS:
        if ch not in reports:
            continue
        m = best_method_metrics(reports[ch], ch)
        matrix[ch] = layer_classes(m)
    return matrix


def compare_matrices(ref: dict[str, dict[str, str]], tgt: dict[str, dict[str, str]]) -> dict:
    cells: list[dict] = []
    match = 0
    total = 0
    for ch in CHANNELS:
        if ch not in ref or ch not in tgt:
            continue
        for layer in LAYERS:
            r, t = ref[ch].get(layer), tgt[ch].get(layer)
            if r is None or t is None:
                continue
            total += 1
            ok = r == t
            if ok:
                match += 1
            cells.append({"channel": ch, "layer": layer, "reference": r, "target": t, "match": ok})
    rci = match / total if total else float("nan")
    return {
        "recovery_consistency_index": rci,
        "n_cells": total,
        "n_match": match,
        "cells": cells,
        "reference_matrix": ref,
        "target_matrix": tgt,
    }


def write_cylinder_reference_summary(cfg_path: Path, out_path: Path) -> None:
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    cyl = cfg["geometries"]["cylinder"]
    reports = {}
    for ch, rel in cyl["audit_reports"].items():
        p = Path(rel)
        if not p.is_absolute():
            root = Path(__file__).resolve().parents[2]
            p = root / "contact_grammar_emergence" / p
        reports[ch] = json.loads(p.read_text())
    matrix = build_matrix(reports)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "geometry": "cylinder",
                "role": "reference",
                "matrix": matrix,
                "channels": {k: {"report_path": str(cyl["audit_reports"][k])} for k in reports},
            },
            indent=2,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Recovery Consistency Index (RCI)")
    ap.add_argument("--reference", type=Path, required=True, help="Reference summary.json or audit dir")
    ap.add_argument("--target", type=Path, help="Target summary.json or audit dir")
    ap.add_argument("--output", type=Path, help="Output JSON for compare mode")
    ap.add_argument("--write-cylinder-reference", action="store_true")
    ap.add_argument("--config", type=Path, default=Path("configs/cross_geometry_recovery_matrix_PREREG.yaml"))
    args = ap.parse_args()

    if args.write_cylinder_reference:
        write_cylinder_reference_summary(args.config, args.reference)
        print(f"Wrote cylinder reference: {args.reference}")
        return

    if args.target is None or args.output is None:
        ap.error("--target and --output required unless --write-cylinder-reference")

    ref_reports = load_channel_reports(args.reference)
    tgt_reports = load_channel_reports(args.target)
    ref_m = build_matrix(ref_reports)
    tgt_m = build_matrix(tgt_reports)
    result = compare_matrices(ref_m, tgt_m)
    result["reference_path"] = str(args.reference)
    result["target_path"] = str(args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"RCI = {result['recovery_consistency_index']:.3f} ({result['n_match']}/{result['n_cells']} cells)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
