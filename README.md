# Contact Organization — Supplemental Artifacts

**Companion code and frozen results** for the *Robotica* manuscript:

> Auditing Contact Organization in Contact-Rich Manipulation:
> Qualification Criteria, Selective Failures, and a Null Predictive Result

| Path | Contents |
|------|----------|
| `results/robotica/` | Frozen CSVs / JSON / hashes cited by the Robotica tables |
| `configs/robotica/` | Prereg / scorer / trajectory-map YAML |
| `scripts/robotica/` | Readout and audit entry points |

> **This repository does not contain the manuscript** (no `main.tex` / journal PDF).
> Manuscripts are submitted via Cambridge ScholarOne for *Robotica*.

**Corresponding author:** Shengli Ma (`mashli@xauat.edu.cn`)  
**Institution:** Xi'an University of Architecture and Technology

---

## Relation to other work (disclosed)

Simulation tooling overlaps with a separate manuscript by the same author on
post-contact kinematics and explicit mass (ROBOT-D-26-01372R1, under review at
*Robotics and Autonomous Systems*). That paper does **not** reuse the Scheme D /
OPCG / $\mathfrak{S}_0$ claims, experiments, or data packaged here.

Earlier README text referred to IJRR Paper I/II stubs. Those stubs are
superseded for the Robotica submission; use `results/robotica/` as the source
of truth for numbers cited in the Robotica PDF.

---

## Quick start

```bash
# Inspect frozen Robotica readout
ls results/robotica/
cat results/robotica/prereg_hash.txt
head -20 results/robotica/sig_witness_round.csv
```

Python scripts under `scripts/robotica/` regenerate table/figure numbers from
existing trajectory maps (no new cohorts). See `results/robotica/README.md`
for hashes, split policy, and the B0n-null reading of the predictive audit.

---

## What reviewers should check first

1. `results/robotica/sig_witness_round.csv` — signature accepts/refuses.
2. `results/robotica/predictive_paired_delta_ci.csv` — Org vs B0n / fx / cell.
3. `results/robotica/*_msm_diagnostics.json` — block λ2 = 0.889 with P1 fail.
4. `results/robotica/README.md` — label note (prereg roles ≠ signature verdicts).

Raw multi-GB trajectory dumps are **not** stored here.
