# Artifact index — batch 2026-07-22

All paths relative to the public repo root (scaffold).

## Paper I

| Manuscript ref | Frozen path | Script |
|----------------|-------------|--------|
| Tab canonical examples (cylinder / failure registry) | `results/frozen/paper1/canonical_ladder/organization_signature_audit.json` | `scripts/organization_signature_audit.py` |
| CE / destruction arms | `results/frozen/paper1/canonical_ladder/mechanism_evidence_matrix.json` | `scripts/organization_signature_audit.py` |
| Occupation / geometry baselines | `results/frozen/paper1/canonical_ladder/measure_baseline_report.json` | `scripts/estimate_occupation_measure.py` |
| Ellipsoid OPCG audit | `results/frozen/paper1/canonical_ladder/ellipsoid_opcg_audit.json` | `scripts/analyze_ellipsoid_recovery.py` |
| Tab MSM diagnostics | `results/frozen/paper1/msm_diagnostics/msm_diagnostics_summary.json` | `scripts/analyze_msm_diagnostics.py` |
| CE-2 multiplicity MSM | `results/frozen/paper1/msm_diagnostics/ce2_multiplicity_msm_diagnostics.json` | `scripts/analyze_ce2_msm_diagnostics.py` |
| Cross-geometry / RCI (sphere, ellipsoid, sliding, held-out, block, PyBullet) | `results/frozen/paper1/cross_geometry/` | `scripts/compare_recovery_matrices.py` |
| Friction sweep / CE-4 | `results/frozen/paper1/friction_sweep/friction_sweep_n50_analysis.json` | `scripts/analyze_friction_sweep.py` |
| Partition stability (Exp-A, force-matched, episode-matched, L1 zones, proximity) | `results/frozen/paper1/partition_stability/` | `scripts/analyze_*partition*` / `analyze_l1_boundary_proximity.py` |
| Guard threshold sensitivity | `results/frozen/paper1/guard_threshold/guard_threshold_sensitivity.json` | `scripts/analyze_guard_threshold_sensitivity.py` |

## Paper II

| Manuscript ref | Frozen path | Script |
|----------------|-------------|--------|
| OPCG MuJoCo cylinder / S0 registry | `results/frozen/paper2/s0_mujoco_cylinder/opcg_cylinder_reference.json` | `scripts/analyze_opcg_witness.py` |
| Cylinder RTD (S0-2) | `results/frozen/paper2/s0_mujoco_cylinder/residence_time_cylinder_full_comparison.json` | `scripts/analyze_residence_times.py` |
| Held-out / μ=0.6 replications | `results/frozen/paper2/s0_mujoco_cylinder/opcg_heldout.json`, `opcg_friction_mu06.json` | `scripts/analyze_opcg_witness.py` |
| Ellipsoid / PyBullet / sliding / μ=0.2 partials | `results/frozen/paper2/s0_partial/` | `scripts/analyze_opcg_witness.py`, `analyze_cross_engine_probe.py` |
| MD-1–MD-3 hold sweep | `results/frozen/paper2/mechanism_discrimination/balanced_operator_report.json` | `scripts/analyze_balanced_protocol_operator.py` |
| Predictive audit table | `results/frozen/paper2/predictive_audit/` | `scripts/predictive_audit.py` |

## Supplemental

| Item | Path |
|------|------|
| Independent verification report | `supplemental/INDEPENDENT_VERIFICATION_REPORT.pdf` |

## Prereg protocol texts

See `docs/prereg/`.
