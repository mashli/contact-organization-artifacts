# Cross-Geometry Recovery Matrix — Confirmatory Preregistration

**Status:** **PREREGISTERED** · cylinder **complete** · sphere **closed** (pilot 0.56 · full **0.75** — hierarchy preserved)  
**Paper 2:** recovery mission **complete** — do not extend geometry (`PAPER2_CLOSURE.md`)  
**Theory framing:** `docs/THEORY_OF_MESOSCOPIC_CONTACT_ORGANIZATION.md` · `docs/RESEARCH_PRIORITY_2026.md`  
**IJRR note:** Cross-geometry (including block) = **same-task** evidence. For main-journal **principle** claims, **second task** outranks third geometry — see `docs/IJRR_STRATEGY.md`. Block, if run, is **supplement tier**, not headline.  
**Config:** `configs/cross_geometry_recovery_matrix_PREREG.yaml`  
**Compare:** `scripts/compare_recovery_matrices.py`

---

## Scientific question (revised emphasis)

> Is the **Recovery Hierarchy pattern** preserved under geometry deformation—and **why** do some layers remain observable on sphere while others do not?

Secondary (statistics): Recovery Consistency Index (RCI) vs cylinder reference.

This is **not** “re-prove Landscape on sphere.”  
It tests whether cross-modal **layer ranking** is geometry-invariant or **hierarchically deformed**.

```text
Cylinder Matrix  vs  Sphere Matrix  →  RCI + layer-order interpretation
```

---

## Full result (700 ep — complete 2026-07-03)

| Item | Value |
|------|-------|
| Episodes | 700 (100/u × 7 u) |
| D4 threshold | $t^\ast = 1.0$ N |
| RCI | **0.75** (12/16 cells) |
| Mismatches vs cylinder | vision/duration/energy **partition**; energy **transition** (sphere pass, cylinder fail) |
| Pattern | Occupancy ✓ (vision, duration); transition ✓ (vision, duration, graph); partition weak; boundary fail all |

**Interpretation:** Full cohort crosses into **pattern-invariance band** (RCI ≥ 0.70) but **partition** remains geometry-limited. Pilot→full shift = statistical sharpening, not task-wide generality. See `results/phase1/cross_geometry_recovery/sphere_full_rci.json`.

---

## Pilot result (already informative)

| Item | Value |
|------|-------|
| Episodes | 70 (10/u × 7 u) |
| D4 threshold | $t^\ast = 1.0$ N |
| RCI | **0.56** (9/16 cells) |
| Basin @ peak | 22 grazing / 2 loaded |
| Pattern | Occupancy ✓ (vision, duration); partition/transition weak; boundary fail all channels (as cylinder) |

**Interpretation (primary):** **Partial observability deformation** — organization **measurable** after calibration; higher layers geometry-limited.  
**Not:** sphere lacks contact organization.

---

## Hypothesis (confirmatory)

Under frozen channel protocols and declared D4 deformations:

| Deformation | Doc |
|-------------|-----|
| D4-impulse | `docs/CROSS_GEOMETRY_SPHERE_IMPULSE_AMENDMENT.md` — sphere **0.22**, cylinder **0.15** |
| D4-threshold | `docs/CROSS_GEOMETRY_SPHERE_THRESHOLD_AMENDMENT.md` — sphere $t^\ast$; cylinder **8 N** |

1. **Layer ranking preserved** (occupancy most recoverable; boundary force-specific).  
2. **Channel ranking** qualitatively similar on **core layers**; partition/transition may deform on sphere.  
3. **No third persistent basin** under C1–C2.

**Not required:** RCI ≥ 0.70 for scientific contribution if partial deformation is theoretically explained.

---

## Frozen discipline

| Rule | Requirement |
|------|-------------|
| F1–F3 | Same feature methods, clustering, seven checks (`RECOVERY_MATRIX_METHODS.md`) |
| F4 | No retuning after seeing sphere results |
| F5 | RCI ≥ 0.70 → **geometry-invariant pattern** (strong confirmatory) |
| | RCI ∈ [0.50, 0.70) → **partial deformation** (expected sphere pilot outcome) |
| D4-impulse / D4-threshold | As amended — not tuned on RCI |

---

## Four-phase flow

```text
Phase 1  Threshold calibration (seeds 5000)     →  lock t*     [DONE]
Phase 2  Pilot 70 ep (seeds 3000) + 4 modalities →  RCI=0.56    [DONE]
Phase 3  Theory + manuscript integration         →  PRIMARY NOW
Phase 4  Full 700 ep + 4 modalities + RCI       →  ENHANCEMENT (optional / revision / main journal)
```

| Phase | Script |
|-------|--------|
| 1 | `run_sphere_threshold_calibration.sh` |
| 2 | `run_sphere_cross_geometry_pilot.sh` |
| 4 | `run_sphere_cross_geometry_full.sh` |

Phase 4 **does not** replace Phase 3. It adds **statistical precision** if universality is a headline claim.

---

## Protocol

| Item | Cylinder | Sphere |
|------|----------|--------|
| u levels | 0–3 cm (7) | same |
| episodes/u (pilot) | — | 10 |
| episodes/u (full) | 100 | 100 |
| probe_budget | 8 | 8 |
| fx threshold | 8.0 N | $t^\ast = 1.0$ N |
| impulse | 0.15 | 0.22 |
| camera | frontview 320×240 | same |

---

## Pass / fail (interpretation)

| Outcome | Action |
|---------|--------|
| RCI ≥ 0.70 | Report geometry-invariant recovery pattern |
| RCI ∈ [0.50, 0.70) | Report **hierarchical observability deformation**; explain **why** (contact mechanics + sparse loaded) |
| RCI < 0.50 | Narrow to geometry-specific observability; do not claim universality |
| Measurement collapse @ 8 N without calibration | Report as **measurement failure** (Paper 2 theme) |

---

## Outputs

```text
results/phase1/cross_geometry_recovery/cylinder/     # reference
results/phase1/cross_geometry_recovery/sphere/pilot/   # primary cross-geometry write-up
results/phase1/cross_geometry_recovery/sphere/full/    # enhancement tier
results/phase1/cross_geometry_recovery/sphere_pilot_rci.json
results/phase1/cross_geometry_recovery/sphere_full_rci.json
```

---

## Relation to Paper 2

Upgrades from single-geometry matrix to **cross-geometry interpretive evidence**.  
Primary contribution remains **Recovery Hierarchy theory** + cylinder matrix; sphere adds **explanatory** geometry-deformation case study.
