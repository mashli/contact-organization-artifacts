# First curated sync — batch 2026-07-22

Populated under `docs/latex_ijrr/artifacts_repo_scaffold/` from the private
`contact_grammar_emergence` tree. **No manuscripts** included.

| Bundle | Count | Size (approx.) |
|--------|------:|----------------|
| `results/frozen/` | 42 | 584 KB |
| `configs/` | 23 | 96 KB |
| `scripts/` | 21 | 236 KB |
| `docs/prereg/` | 6 | 48 KB |
| `supplemental/` | 1 PDF | 160 KB |
| **Total curated payload** | | **~1.1 MB** |

Private `results/` remains ~637 MB and is **not** mirrored.

---

## Paper I → `results/frozen/paper1/`

| Public path | Private source |
|-------------|----------------|
| `canonical_ladder/organization_signature_audit.json` | `results/phase1/emergence_theory/organization_signature_audit.json` |
| `canonical_ladder/mechanism_evidence_matrix.json` | `results/phase1/emergence_theory/mechanism_evidence_matrix.json` |
| `canonical_ladder/measure_baseline_report.json` | `results/phase1/emergence_theory/measure_baseline_report.json` |
| `canonical_ladder/ellipsoid_opcg_audit.json` | `results/phase1/cross_geometry/ellipsoid_opcg_audit.json` |
| `cross_geometry/block_opcg_audit.json` | `results/phase1/cross_geometry/block_opcg_audit.json` |
| `cross_geometry/sphere_full_rci.json` | `results/phase1/cross_geometry_recovery/sphere_full_rci.json` |
| `cross_geometry/block_recovery_summary.json` | `results/phase1/cross_geometry_recovery/block/recovery_summary.json` |
| `cross_geometry/cylinder_reference_summary.json` | `results/phase1/cross_geometry_recovery/cylinder/summary.json` |
| `cross_geometry/ellipsoid_full_rci.json` | `results/phase1/cross_geometry_recovery/ellipsoid_full_rci.json` |
| `cross_geometry/sliding_full_rci.json` | `results/phase1/second_task_recovery/sliding_full_rci.json` |
| `cross_geometry/cylinder_heldout_full_rci.json` | `results/phase1/cohort_independence/cylinder_heldout_full_rci.json` |
| `cross_geometry/cross_engine_probe_report.json` | `results/phase1/cross_engine/pybullet_probe/cross_engine_probe_report.json` |
| `friction_sweep/friction_sweep_n50_analysis.json` | `results/phase1/friction_sweep_n50/friction_sweep_analysis.json` |
| `msm_diagnostics/msm_diagnostics_summary.json` | `results/phase1/msm_diagnostics/msm_diagnostics_summary.json` |
| `msm_diagnostics/ce2_multiplicity_msm_diagnostics.json` | `results/phase1/msm_diagnostics/ce2_multiplicity_msm_diagnostics.json` |
| `msm_diagnostics/*_msm_diagnostics.json` | per-cohort MSM files (if present) |
| `partition_stability/partition_stability_ari_expA.json` | `results/phase1/mechanism_discrimination/partition_stability_ari_expA.json` |
| `partition_stability/partition_forcematched_ari.json` | `results/phase1/mechanism_discrimination/partition_forcematched_ari.json` |
| `partition_stability/episode_matched_boundary.json` | `results/phase1/mechanism_discrimination/episode_matched_boundary.json` |
| `partition_stability/episode_matched_l1_zones_ramp3v6.json` | `results/phase1/mechanism_discrimination/episode_matched_l1_zones_ramp3v6.json` |
| `partition_stability/l1_boundary_proximity_ramp3_vs_ramp6.json` | `results/phase1/mechanism_discrimination/l1_boundary_proximity_ramp3_vs_ramp6.json` |
| `guard_threshold/guard_threshold_sensitivity.json` | `results/phase1/guard_threshold_sensitivity/guard_threshold_sensitivity.json` |

## Paper II → `results/frozen/paper2/`

| Public path | Private source |
|-------------|----------------|
| `s0_mujoco_cylinder/opcg_cylinder_reference.json` | `results/phase1/emergence_theory/opcg_witness_report.json` |
| `s0_mujoco_cylinder/residence_time_cylinder_full_comparison.json` | same name under `emergence_theory/` |
| `s0_mujoco_cylinder/opcg_heldout.json` | `opcg_replication_heldout.json` |
| `s0_mujoco_cylinder/opcg_friction_mu06.json` | `opcg_replication_mu06.json` |
| `s0_positive_replication/opcg_replication_mu10.json` | `opcg_replication_mu10.json` |
| `s0_partial/sliding_opcg_fail.json` | `opcg_witness_sliding_full.json` |
| `s0_partial/ellipsoid_opcg.json` | `cross_geometry/ellipsoid_opcg_audit.json` |
| `s0_partial/pybullet_probe.json` | `cross_engine/.../cross_engine_probe_report.json` |
| `s0_partial/measure_baseline_report.json` | `emergence_theory/measure_baseline_report.json` |
| `s0_partial/mu02_destruction.json` | `predictive_audit/mu0.2_audit.json` |
| `mechanism_discrimination/balanced_operator_report.json` | same under `mechanism_discrimination/` |
| `predictive_audit/{cylinder,ellipsoid,sphere,mu0.6}_audit.json` | `results/phase1/predictive_audit/` |

## Scripts (21)

`organization_signature_audit.py`, `analyze_msm_diagnostics.py`, `analyze_ce2_msm_diagnostics.py`,
`compare_recovery_matrices.py`, `analyze_partition_stability_ari.py`,
`analyze_partition_forcematched_ari.py`, `analyze_episode_matched_boundary_stability.py`,
`analyze_l1_boundary_proximity.py`, `analyze_friction_sweep.py`,
`analyze_guard_threshold_sensitivity.py`, `analyze_cross_engine_probe.py`,
`analyze_opcg_witness.py`, `analyze_balanced_protocol_operator.py`,
`analyze_ellipsoid_recovery.py`, `analyze_block_recovery.py`,
`analyze_residence_times.py`, `analyze_hierarchical_opcg.py`,
`predictive_audit.py`, `estimate_occupation_measure.py`,
`plot_friction_sweep.py`, `plot_guard_threshold_sensitivity.py`

> Note: scripts may still import private `src/` helpers. Batch 1 ships **frozen
> numbers + entry points**; making scripts fully standalone is a follow-up.

## Configs (23)

PREREG YAML for friction, cross-geometry recovery, PyBullet probe, held-out cylinder,
sliding, ellipsoid, sphere, emergence baseline, controlled perturbation, and
`contact_graph_landscape_convergence*_PREREG.yaml` shells.  
Skipped (missing): `partition_stability_ari_PREREG.yaml` — protocol lives in
`docs/prereg/PARTITION_STABILITY_ARI_PREREG.md`.

## Prereg docs (6)

`CROSS_GEOMETRY_RECOVERY_MATRIX_PREREG.md`, `CROSS_ENGINE_PROBE_PREREG.md`,
`PARTITION_STABILITY_ARI_PREREG.md`, `CYLINDER_HELDOUT_SEED_PREREG.md`,
`SECOND_TASK_SLIDING_PREREG.md`, `EMERGENCE_PHYSICS_ABLATION_PREREG.md`

## Supplemental

`supplemental/INDEPENDENT_VERIFICATION_REPORT.pdf`

---

## Push checklist

```bash
# Create empty GitHub repo, then:
PRIV=.../contact_grammar_emergence
PUB=/path/to/contact-organization-artifacts
rsync -a --delete \
  --exclude '.git' \
  "$PRIV/docs/latex_ijrr/artifacts_repo_scaffold/" "$PUB/"
cd "$PUB"
# edit README URL placeholder
git init
git add .
git commit -m "Initial IJRR Paper I+II frozen artifacts (batch 2026-07-22)"
git branch -M main
git remote add origin git@github.com:<USER>/contact-organization-artifacts.git
git push -u origin main
git tag ijrr-submission-2026-07
git push origin ijrr-submission-2026-07
```

Then update both manuscripts’ Data availability to the public URL.
