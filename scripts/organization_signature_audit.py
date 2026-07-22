#!/usr/bin/env python3
"""Phase 0 — P1–P4 organization signature audit + mechanism evidence matrix (offline).

Reads frozen landscape_trajectories.json cohorts; writes:
  results/phase1/emergence_theory/organization_signature_audit.json
  results/phase1/emergence_theory/mechanism_evidence_matrix.json

No new simulations. Does NOT claim Emergence Theorem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CGE_SRC = REPO / "src"
if str(CGE_SRC) not in sys.path:
    sys.path.insert(0, str(CGE_SRC))

from cge.emergence.organization_signature import audit_from_path  # noqa: E402

THEORY_DIR = REPO / "results/phase1/emergence_theory"


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def load_json_if_exists(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text())
    return None


def build_k_stability_table(
    audits: dict[str, dict],
    guard_report: dict | None,
    transfer_report: dict | None,
) -> dict[str, dict]:
    """E5 — cross-cohort K, λ₂, G_fx match (offline)."""
    table: dict[str, dict] = {}
    for name, audit in audits.items():
        row: dict = {
            "K_mode": audit.get("k_mode"),
            "lambda2_mode": audit.get("lambda2_mode"),
            "G_fx_spectrum_match": audit.get("g_fx_spectrum_match"),
            "all_P_pass": audit.get("all_properties_pass"),
            "summary": audit.get("summary"),
            "source_path": audit.get("source_path"),
        }
        if guard_report and name in guard_report.get("cohorts", {}):
            g = guard_report["cohorts"][name]
            row["tau2_implied"] = g.get("mode_spectrum", {}).get("tau2_implied")
            row["G_fx_verdict"] = next(
                (x.get("verdict") for x in g.get("guards", []) if x.get("guard_id") == "G_fx_mode"),
                None,
            )
            g_mult = next(
                (x for x in g.get("guards", []) if x.get("guard_id") == "G_multiplicity"),
                None,
            )
            if g_mult:
                row["G_multiplicity_lambda2"] = g_mult.get("lambda2")
                row["G_multiplicity_spectrum_match"] = g_mult.get("spectrum_match_score")
        if transfer_report and name in transfer_report.get("cohorts", {}):
            t = transfer_report["cohorts"][name]
            row["msm_ari_vs_mode"] = t.get("msm_ari_vs_mode")
            row["natural_pi_verdict"] = t.get("natural_pi_verdict")
        table[name] = row
    return table


def build_mechanism_matrix(
    audits: dict[str, dict],
    k_table: dict[str, dict],
) -> dict:
    ref = audits.get("cylinder_reference", {})
    f2 = audits.get("ablation_friction_mu_0.2", {})
    f1 = audits.get("ablation_friction_mu_0.0", {})

    ref_pass = ref.get("all_properties_pass", False)
    f2_fail = not f2.get("all_properties_pass", True) if f2 else None
    f1_fail = not f1.get("all_properties_pass", True) if f1 else None

    def arrow_status(num: int, supported: bool, partial: bool, note: str) -> dict:
        if supported:
            st = "supported"
        elif partial:
            st = "partial"
        else:
            st = "open"
        return {"arrow": num, "status": st, "note": note}

    necessity = [
        {
            "mechanism": "Persistent contact + constraint",
            "physical_prediction": "Accessible set restricted; no org without contact",
            "observable": "Contact frame fraction; loaded org absent when contact destroyed",
            "arrow": 1,
            "status": "partial",
            "evidence": {
                "F1_mu0_destroys_contact": f1_fail,
                "F2_mu02_no_loaded": f2.get("counterexamples", {}).get("CE4_no_loaded_regime") if f2 else None,
                "reference_P_pass": ref_pass,
            },
        },
        {
            "mechanism": "Dissipation (friction)",
            "physical_prediction": "Energy drains; longer dwell when μ↑",
            "observable": "τ_loaded↑; switches↓; μ ablation",
            "arrow": 2,
            "status": "supported",
            "evidence": {
                "H-A_F2_no_loaded_at_mu02": f2.get("counterexamples", {}).get("CE4_no_loaded_regime") if f2 else None,
                "reference_metastable": ref_pass,
            },
        },
        {
            "mechanism": "Relaxation / concentration",
            "physical_prediction": "Non-uniform occupation on M_c",
            "observable": "π(u) structured; μ_O non-uniform; not μ_F uniform alone",
            "arrow": "2-3",
            "status": "supported",
            "evidence": {
                "CE1_m1_falsified_on_reference": ref.get("counterexamples", {}).get("CE1_feature_measure_uniform"),
                "reference_P3": ref.get("properties", {}).get("P3", {}).get("pass"),
            },
        },
        {
            "mechanism": "Hybrid / guards",
            "physical_prediction": "Finite sparse transitions",
            "observable": "Transition count; guard partition; λ₂ on MODE",
            "arrow": 5,
            "status": "supported",
            "evidence": {
                "CE2_multiplicity_wrong_spectrum": ref.get("counterexamples", {}).get("CE2_multiplicity_wrong_spectrum"),
                "reference_G_fx_match": ref.get("g_fx_spectrum_match"),
            },
        },
        {
            "mechanism": "Bounded excitation",
            "physical_prediction": "No uniform fill of M_c; phased revisit",
            "observable": "Protocol phases; τ capped at probe_budget",
            "arrow": 2,
            "status": "partial",
            "evidence": {"note": "Implicit in pulse-probe; E7–E9 prereg pending"},
        },
        {
            "mechanism": "Combined → P1–P4",
            "physical_prediction": "P1–P4 jointly when mechanisms present",
            "observable": "organization_signature_audit",
            "arrow": 6,
            "status": "supported" if ref_pass else "partial",
            "evidence": {
                "reference_all_P": ref_pass,
                "heldout_all_P": audits.get("cylinder_heldout_full", {}).get("all_properties_pass"),
                "F2_fails": f2_fail,
            },
        },
    ]

    destruction = [
        {
            "remove": "Friction μ→0",
            "predicted_loss": "Contact + loaded org",
            "arm": "F1 ablation_friction_mu_0.0",
            "observed": f1_fail,
            "supports": "Dissipation / loaded",
        },
        {
            "remove": "Low μ=0.2",
            "predicted_loss": "Loaded branch",
            "arm": "F2 ablation_friction_mu_0.2",
            "observed": f2_fail,
            "supports": "Dissipation threshold",
        },
        {
            "remove": "Wrong guard (multiplicity Π)",
            "predicted_loss": "P2 spectrum, not labels",
            "arm": "CE-2 offline",
            "observed": ref.get("counterexamples", {}).get("CE2_multiplicity_wrong_spectrum"),
            "supports": "Hybrid guard not patch count",
        },
        {
            "remove": "Raw ℱ measure",
            "predicted_loss": "P3 on wrong chart",
            "arm": "CE-1 offline",
            "observed": ref.get("counterexamples", {}).get("CE1_feature_measure_uniform"),
            "supports": "OPCG necessary",
        },
        {
            "remove": "ℱ MSM",
            "predicted_loss": "P1–P4",
            "arm": "CE-3 offline",
            "observed": ref.get("counterexamples", {}).get("CE3_feature_msm_misaligned"),
            "supports": "Representation",
        },
    ]

    return {
        "plan_version": "v0.1",
        "phase": 0,
        "wording": (
            "Mechanism evidence matrix — data supports/falsifies arrows; "
            "does NOT prove Emergence Theorem (Layer ④ open)."
        ),
        "parent_plan": "docs/EMERGENCE_MECHANISM_EVIDENCE_PLAN.md",
        "sources": [
            "results/phase1/emergence_theory/organization_signature_audit.json",
            "results/phase1/emergence_theory/guard_partition_report.json",
            "results/phase1/emergence_theory/transfer_operator_report.json",
            "results/phase1/emergence_theory/measure_baseline_report.json",
        ],
        "necessity_matrix": necessity,
        "arrow_status": {
            "1": arrow_status(1, False, True, "F1/F2 ablation; E1 contact fraction planned"),
            "2": arrow_status(2, True, False, "H-A friction ablation + reference RTD"),
            "3": arrow_status(3, True, False, "Frozen RTD timescale separation"),
            "4": arrow_status(4, True, False, "Frozen P(τ>6) L/G tails"),
            "5": arrow_status(5, True, False, "K=2 stable; CE-2 multiplicity falsified"),
            "6": arrow_status(6, ref_pass, not ref_pass, "P1–P4 audit on reference cohort"),
        },
        "k_stability_table": k_table,
        "destruction_table": destruction,
        "phase0_audit_summary": {
            "reference_passes_P1_P4": ref_pass,
            "heldout_passes_P1_P4": audits.get("cylinder_heldout_full", {}).get("all_properties_pass"),
            "F2_fails_appropriately": f2_fail,
            "F1_fails_appropriately": f1_fail,
            "cylinder_K2_stable_across_mu": all(
                k_table.get(n, {}).get("K_mode") == 2
                for n in (
                    "cylinder_reference",
                    "cylinder_heldout_full",
                    "ablation_friction_mu_0.6",
                    "ablation_friction_mu_1.0",
                )
                if n in k_table
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P1–P4 organization signature audit (Phase 0)")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/emergence_measure_baseline_PREREG.yaml",
    )
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--name", type=str, default="custom")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    if args.trajectory is not None:
        cohorts = [{"name": args.name, "path": str(args.trajectory)}]
    else:
        cohorts = cfg.get("cohorts", [])

    transfer_report = load_json_if_exists(THEORY_DIR / "transfer_operator_report.json")
    guard_report = load_json_if_exists(THEORY_DIR / "guard_partition_report.json")

    audits: dict[str, dict] = {}
    for c in cohorts:
        name = str(c["name"])
        path = resolve_path(c["path"])
        if not path.is_file():
            print(f"SKIP {name}: missing {path}", flush=True)
            continue
        tr = None
        if transfer_report:
            tr = transfer_report.get("cohorts", {}).get(name)
        rep = audit_from_path(path, cohort_name=name, transfer_report=tr)
        audits[name] = rep.to_dict()
        props = rep.properties
        flags = " ".join(f"{p.property_id}={'✓' if p.pass_ else '✗'}" for p in props)
        print(
            f"{name}: {flags} K={rep.k_mode} λ₂={rep.lambda2_mode:.3f} "
            f"summary={rep.summary}",
            flush=True,
        )

    audit_out = THEORY_DIR / "organization_signature_audit.json"
    audit_payload = {
        "project": "Phase0_organization_signature_audit",
        "plan": "docs/EMERGENCE_MECHANISM_EVIDENCE_PLAN.md",
        "wording": (
            "P1–P4 booleans on frozen cohorts — signatures consistent with Organization; "
            "NOT a proof of Organization or Emergence Theorem."
        ),
        "cohorts": audits,
    }
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit_payload, indent=2))
    print(f"\nWrote {audit_out}", flush=True)

    k_table = build_k_stability_table(audits, guard_report, transfer_report)
    matrix = build_mechanism_matrix(audits, k_table)
    matrix_out = THEORY_DIR / "mechanism_evidence_matrix.json"
    matrix_out.write_text(json.dumps(matrix, indent=2))
    print(f"Wrote {matrix_out}", flush=True)

    print("\n--- E5 K stability (cylinder family) ---", flush=True)
    for name in sorted(k_table):
        row = k_table[name]
        print(
            f"  {name}: K={row.get('K_mode')} λ₂={row.get('lambda2_mode', float('nan')):.3f} "
            f"G_fx={row.get('G_fx_spectrum_match', float('nan')):.3f} P_all={row.get('all_P_pass')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
