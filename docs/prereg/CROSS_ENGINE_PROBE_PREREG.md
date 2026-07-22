# Cross-Engine Probe (MuJoCo → PyBullet) — Preregistration

**Status:** PREREGISTERED · July 2026 · IJRR simulator-invariance tier  
**Companion:** `docs/PROTOCOL_FREEZE.md` · `configs/cross_engine_pybullet_probe_PREREG.yaml`

---

## Scientific question

> Is the **force-channel recovery hierarchy pattern** (occupancy, transition) a **MuJoCo contact-solver artifact**, or does it **replicate under an independent physics engine** with frozen audit logic?

This is **F6** in the claim–evidence map: *simulator-specificity hypothesis*.

---

## Frozen design (minimal probe — not a second paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Task | Cylinder push-roll | Same $u$ levels as reference |
| Engine | PyBullet DIRECT | Independent from MuJoCo/robosuite |
| Channel | **Force only** | No vision/duration/graph rebuild |
| Control $u$ | `[0.0 … 0.030]` m (7 levels) | Same as `cylinder_contact_configuration_heldout_pilot` |
| `friction_mu` | 0.6 | Frozen |
| `impulse_magnitude` | 0.15 | Maps to frozen `pusher_speed_m_s` in config (SI bridge, not outcome-tuned) |
| `fx_threshold` | **8.0 N** | Frozen — no re-calibration on PyBullet |
| `probe_budget` | 8 contact frames | Frozen |
| `episodes_per_u` | 10 | 70 episodes total |
| `base_seed` | **7000** | Held-out block (disjoint from ref 2000) |
| Layers tested | Occupancy, Transition | Boundary/partition **not** headline for this probe |
| MuJoCo reference | `results/phase1/cohort_independence/cylinder_heldout/full/landscape_trajectories.json` if present; else `results/phase1/landscape_analysis/landscape_trajectories.json` |

**Seed formula (unchanged):** `base_seed + u_idx × 100_000 + rep`

---

## Success criteria (pattern match, not numeric equality)

| Metric | Pass | Partial |
|--------|------|---------|
| Occupancy $\pi(u)$ Spearman (internal) | $\geq 0.70$ | $\geq 0.50$ |
| Transition topology vs MuJoCo ref | dominant edges match (`loaded→grazing` present; no rank inversion) | Spearman $\geq 0.50$ on edge counts |
| Layer pattern (force) | occ + trans **pass** | occ or trans **partial** |
| Coarse RCI-style | $\geq 10/16$ pattern match vs ref force layers | $\geq 8/16$ |

**Failure interpretation (also valuable):** hierarchy inverts or collapses → report *engine-conditional recoverability*; narrow main-text scope.

---

## Explicit non-goals

- Retuning `fx_threshold`, audit gates, or PyBullet friction on outcomes
- Full 4-channel Recovery Matrix on PyBullet
- Vision / tactile / hardware
- New geometry or task

---

## Outputs

```text
results/phase1/cross_engine/pybullet_probe/
  landscape_trajectories.json
  cross_engine_probe_report.json
  cross_engine_probe_summary.png   # optional plot
```

**Orchestration:** `scripts/run_cross_engine_probe.sh`

---

## Pilot results (July 2026)

| Metric | MuJoCo held-out ref | PyBullet probe | Cross-engine |
|--------|---------------------|----------------|--------------|
| Episodes | 192 | **70/70** | — |
| Occupancy π(u) trend | partial (ρ≈0.64 peak) | strong internal (|ρ|≈0.82) | pattern ρ≈0.0 |
| Transition topology | pass (ℓ→g present) | **pass** (ℓ→g present) | **Spearman=1.0** |
| Verdict | — | — | **partial** |

**Read:** transition layer replicates across engines; occupancy shape is engine-deformed but not collapsed. Supports *not solver-only* for transition; occupancy remains engine-sensitive (scope discipline).

---

## Paper sentence (if pass)

> Under frozen force-only protocol, occupancy and transition patterns replicate across MuJoCo and PyBullet, arguing against solver-specific artifact as the sole explanation of the observed hierarchy.
