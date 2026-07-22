# Emergence Physics Ablation — PREREGISTERED (Project A)

**Status:** PREREGISTERED · July 2026 · **Emergence Theory Stage 1**  
**Parent:** `docs/EMERGENCE_THEORY_RESEARCH_PLAN.md`  
**Tier:** Necessary-conditions for measure concentration — **not** Paper 2 recovery audit

---

## Scientific question

> **Which physical mechanisms are necessary for trajectory occupation measure $\mu_\Pi$ to concentrate** (non-uniform support, coherent residence, nontrivial transition topology)?

This is **not** “does Organization recover cross-modally?” (Paper 2).  
It is: **if we remove mechanism X, does layered measure structure disappear?**

---

## Hypotheses under test (Stage 1)

| ID | Mechanism | Ablation | Predicted effect if necessary |
|----|-----------|----------|-------------------------------|
| **H-A** | Sliding friction | $\mu \in \{0.0, 0.2, 0.6, 1.0\}$ | $\mu{=}0$: no sustained loaded residence; $H(\mu_\Pi)\uparrow$; transition $\ell{\to}g$ rare |
| **H-B** | Dissipation / contact damping | solref scale $\in \{0.5, 1.0, 2.0\}$ × reference | extreme: oscillatory / no metastable residence |
| **H-C** | Rolling / spin DOF | fixed object yaw; zero initial spin | partition collapses or $\pi(u)$ flattens |
| **H-D** | Compliance | solimp scale $\in \{0.5, 1.0, 2.0\}$ | boundary layer loses discriminative escape |

**Phase A1 (now):** H-A + H-B only. H-C/H-D deferred to Phase A2.

---

## Frozen task shell (identical across ablation arms)

| Parameter | Reference | Ablation arms |
|-----------|-----------|---------------|
| Task | Cylinder push-roll | **Same** |
| Control $u$ | 7 eccentricity levels | **Same** |
| `impulse_magnitude` | 0.15 | **Same** |
| `fx_threshold` (readout only) | 8.0 N | **Same** — **not** retuned per arm |
| `probe_budget` | 8 | **Same** |
| `episodes_per_u` | 10 (pilot) / 100 (full) | **10 pilot first** |
| `base_seed` | **8000** (disjoint from 2000/6000/7000) | **Same block per arm** |
| Controller | PulseProbe frozen | **Same** |

**Seed formula (unchanged):** `base_seed + u_idx × 100_000 + rep`

**One ablation axis per run** — never combine friction + solref in one collection.

---

## Ablation arms (Phase A1)

### Arm F0 — Reference physics (within ablation block)

- `friction_mu = 0.6`
- MuJoCo solref/solimp: robosuite defaults (documented in run metadata)

### Arm F1 — Zero friction (H-A critical)

- `friction_mu = 0.0` (object–table + object–pusher via `set_sliding_friction`)

### Arm F2 — Low friction

- `friction_mu = 0.2`

### Arm F3 — High friction

- `friction_mu = 1.0`

### Arm D1 — High contact damping (H-B)

- solref time constant × **0.5** (stiffer contacts) — implementation via env XML override or wrapper

### Arm D2 — Low contact damping (H-B)

- solref time constant × **2.0** (softer / bouncier contacts)

**Pilot only:** 7 u × 10 ep = **70 episodes per arm**.

---

## Readout metrics (organization-first — Project B0/B1)

**Primary object:** $\mu_\mathcal{O}(B_i) = T(B_i)/T$ on MODE basins (after $\Pi$).  
**Falsified object:** $\mu_\mathcal{F}$ histogram on raw ℱ (do **not** use as organization test).

All arms analyzed with **`scripts/estimate_occupation_measure.py`** (dual $\mu_\mathcal{F}$ + $\mu_\mathcal{O}$):

| Metric | Definition | Organization lost if… |
|--------|------------|------------------------|
| $\mu_\mathcal{O}$ joint ratio | $H_2(\mu_\mathcal{O}(u,B)) / H_0$ | → 1.0 |
| $\pi(u)$ slope | Spearman($u$, $P(\mathrm{loaded})$) | → 0 |
| loaded frame fraction | Mean loaded-frame share | → 0 (all grazing) |
| Transition edges | Count of $\ell{\to}g$, … | → single edge only |
| $\mu_\mathcal{F}$ ratio | (diagnostic only) | uniform **does not** imply no org |

**Primary necessary-condition test (H-A):**  
Arm F1 vs F0: loaded frames → 0 **and** contact rate ↓ → **friction necessary** for loaded organization.

---

## Phase A1 results — friction axis pilot (2026-07-05)

| Arm | peak_push | traj | loaded / grazing frames | H-A read |
|-----|-----------|------|-------------------------|----------|
| F1 `μ=0.0` | 4/70 | 2 | 1 / 12 | contact destroyed |
| F2 `μ=0.2` | 15/70 | 15 | **0 / 38** | loaded org destroyed |
| F0 `μ=0.6` | 18/70 | 17 | 68 / 42 | reference |
| F3 `μ=1.0` | 19/70 | 19 | 75 / 37 | reference-like |

**Verdict:** H-A **supported** — friction necessary for **loaded basin organization** (stronger than contact alone).

Dual-measure report includes all four arms in `measure_baseline_report.json`.

---

## Explicit non-goals

- Retuning $t^\ast$ or audit thresholds per arm
- Cross-modality (vision/duration/graph)
- New geometry or task
- Claiming universal physics law from cylinder pilot

---

## Outputs

```text
results/phase1/emergence_ablation/
  friction_mu_0.0/pilot/landscape_trajectories.json
  friction_mu_0.2/pilot/...
  friction_mu_0.6/pilot/...     # reference arm
  friction_mu_1.0/pilot/...
  solref_x0.5/pilot/...
  solref_x2.0/pilot/...
  emergence_ablation_report.json
```

**Orchestration (Phase A1):** `scripts/run_emergence_physics_ablation.py` (stub → implement after measure baseline)

**Configs:** `configs/emergence_ablation_friction_*_PREREG.yaml`

---

## Success bands (interpretive)

| Observation | Read |
|-------------|------|
| F1 destroys concentration + transitions | H-A **supported** (friction necessary) |
| F1 still concentrated | H-A **rejected** for cylinder push; revise mechanism list |
| D1/D2 shift magnitude but preserve topology | damping modulates **scale**, not **existence** |
| All arms concentrated | ablation range insufficient; expand axis |

---

## Epistemic ordering

1. **Offline measure baseline** on existing reference JSON (seed 2000) — **before** any ablation collection  
2. Preregister this document — **before** ablation pilot  
3. Collect ablation pilots — **frozen** arms above  
4. Post-hoc compare $H$, transitions — **no** threshold tuning on outcomes

---

## Upgrade path

| Tier | When |
|------|------|
| Pilot 70 ep/arm | **Now** (Phase A1) |
| Full 700 ep/arm | Only if pilot shows clear necessary-condition separation |
| H-C/H-D arms | Phase A2 (Q1 2027) |
