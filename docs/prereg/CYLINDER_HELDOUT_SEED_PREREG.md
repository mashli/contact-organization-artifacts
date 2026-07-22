# Cylinder Held-Out Seed Cohort — Independent Dataset Preregistration

**Status:** PREREGISTERED · July 2026 · IJRR cohort-independence tier  
**Companion:** `docs/PROTOCOL_FREEZE.md` · `docs/RECOVERY_INDEPENDENCE.md`

---

## Scientific question

> Is the Recovery Hierarchy a **dataset artifact** of the original cylinder seed block (base_seed=2000), or does it replicate on a **disjoint held-out seed cohort** under the **frozen** audit protocol?

This tests **cohort independence** (tier 1 of Recovery Independence), distinct from:
- **Cross-geometry** (sphere) — same task, different shape
- **Cross-task** (sliding) — same shape, different manipulation

---

## Frozen design (zero tuning)

| Parameter | Reference cohort | Held-out pilot |
|-----------|------------------|----------------|
| Task | Cylinder push-roll | **Same** |
| Control u | 7 eccentricity levels | **Same** |
| friction_mu | 0.6 (fixed) | **Same** |
| impulse | 0.15 | **Same** |
| fx threshold | 8.0 N | **Same** (locked) |
| Audit battery | 7-check RIA mirror | **Same** |
| Feature methods | crop HSV, runlength, edge_binary, kinetic_only | **Same** |
| episodes_per_u | 100 (reference) | **10** (pilot) |
| base_seed | **2000** | **6000** (disjoint block) |
| Total episodes | 700 (reference) | **70** (pilot) |

**Seed formula (unchanged):** `base_seed + u_idx × 100_000 + rep`

**Disjoint seed blocks in use:** 2000 (cylinder ref), 3000 (sphere), 4000/4100/4200/5000 (sliding). Block **6000** is reserved for held-out.

---

## Success bands (mirror cross-geometry / sliding)

| RCI vs cylinder reference | Interpretation |
|---------------------------|----------------|
| ≥ 0.70 | Cohort-independent hierarchy **supported** |
| [0.50, 0.70) | Partial replication; report layers |
| < 0.50 | Protocol or seed sensitivity; narrow claims |

**Primary read:** layer-pattern match on vision + duration (occupancy, transition; boundary force-only fail), not numeric RCI equality.

---

## Explicit non-goals

- Retuning fx threshold, audit thresholds, or feature methods on held-out data
- Friction sweep on Task A (confounds with sliding μ-as-u design)
- Third geometry (block)
- Why / emergence mechanism

---

## Outputs

```text
results/phase1/cohort_independence/cylinder_heldout/pilot/
  characterization/formation_states_timeline.csv
  landscape_trajectories.json
  vision_audit/ … energy_audit/
results/phase1/cohort_independence/cylinder_heldout_pilot_rci.json
```

**Orchestration:** `scripts/run_cylinder_heldout_pilot.sh`

---

## Upgrade path

| Tier | Episodes | When |
|------|----------|------|
| Pilot | 70 | **Now** — minimum cohort-independence evidence |
| Full | 700 | Only if pilot RCI ≥ 0.70 and reviewers request CI |
