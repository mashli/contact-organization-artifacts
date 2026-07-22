# Second Task B — Sliding Push (PREREGISTERED)

**Status:** PREREGISTERED · IJRR Path B · locked 2026-07-03  
**Task ID:** T2a sliding  
**Venue:** `VENUE_PATHS.md` · `IJRR_STRATEGY.md`

---

## Scientific question

> Does the **Recovery Hierarchy pattern** hold under **sliding-dominated** contact dynamics — and is it **task-invariant** with Task A (push-roll)?

**IJRR target sentence (if supported):**

> The recovery hierarchy is task-invariant across distinct contact-rich manipulation regimes.

**Not claimed:** numeric RCI equality with Task A; full manipulation universality with n=1 second task.

---

## Task A vs Task B

| | Task A (push-roll) | Task B (sliding) |
|--|------------------|------------------|
| Geometry | Cylinder | **Same cylinder** |
| Post-contact dynamics | Impulse along X (roll) | Sustained tangential push (axis Y) + light normal |
| Control u | `push_eccentricity_m` (7 levels) | **`friction_mu_levels`** (7 levels) |
| Approach | Eccentricity sweep | Fixed `approach_eccentricity_m = 0.015` m |
| Controller | `PulseProbeController` | `SlidePushController` (no reverse pulse) |
| Audit protocol | **Frozen** — same RCI thresholds, same channels |

---

## Pipeline alias (frozen)

For compatibility with basin / landscape / RCI scripts:

- `push_eccentricity_m` column stores **friction μ** (control u for sliding)
- `approach_eccentricity_m` in sweep cells stores true approach offset (0.015 m)

Document in methods; do not reinterpret as eccentricity sweep on Task B.

---

## Frozen protocol (zero tuning)

Mirror `CROSS_GEOMETRY_RECOVERY_MATRIX_PREREG.md`:

- Channels: force 𝓛c, vision (`contact_crop_hsv`), duration (`runlength_stats`), graph (`edge_binary`), energy (`kinetic_only`)
- Seven-check audit battery unchanged
- RCI vs **cylinder Task A** reference (`results/phase1/cross_geometry_recovery/cylinder/summary.json`)
- Basin split: `resultant_force_f_x` @ **8.0 N** (cylinder-inherited; calibrate if smoke shows single basin — D4-sliding amendment)

---

## Sweep (pilot)

```yaml
task_mode: sliding
geometry: cylinder
friction_mu_levels: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
approach_eccentricity_m: 0.015
tangential_effort: 0.12
episodes_per_u: 10   # → 70 pilot
base_seed: 4000
```

**Smoke:** 2 μ × 3 ep = 6 episodes (`base_seed: 4100`).

---

## Phase 0 smoke result (2026-07-03)

Pipeline **end-to-end OK** (`run_sliding_task_smoke.sh`, exit 0).

| Item | Value |
|------|-------|
| Episodes with contact | 2 / 6 |
| Basins @ 1.0 N (smoke) | all grazing (single-basin) |
| RCI vs cylinder | 0.50 (smoke n=2; not interpretable) |

**Before pilot:** ~~run D4-sliding fx threshold calibration~~ **Done 2026-07-03** — locked **t\* = 8.0 N** (37 contact / 140 ep calibration; loaded frac 54% vs cylinder 45%).

**Contact yield:** ~26% episodes reach `peak_push` on calibration — monitor on pilot; tune controller if pilot underpowered.

**Controller tuning (v0):** `normal_magnitude=0.10`, `approach_eccentricity_m=0.020`, `tangential_effort=0.10`, `probe_budget=10`.

---

## Success bands (mirror sphere)

| RCI vs cylinder | Interpretation |
|-----------------|----------------|
| ≥ 0.70 | Task-invariant hierarchy **supported** |
| [0.50, 0.70) | Partial task deformation (report layers) |
| < 0.50 | Task-specific recovery (narrow claims) |

---

## Outputs

```text
results/phase1/second_task_recovery/sliding/
  smoke/ | pilot/ | full/
    landscape_trajectories.json
    vision_audit/ … energy_audit/
sliding_pilot_rci.json  (RCI vs cylinder)
```

---

## Explicit non-goals

- Block / third geometry
- Retuning audit thresholds post-hoc
- Why / emergence mechanism (Paper 3)

---

## Implementation

| Asset | Path |
|-------|------|
| Controller | `force_mass_experiments/controllers.py` — `SlidePushController` |
| Task runtime | `src/cge/task_runtime.py` |
| Char config | `configs/sliding_contact_configuration_*` |
| Orchestration | `scripts/run_sliding_task_*.sh` |
| Config gen | `scripts/generate_sliding_recovery_configs.py` |
