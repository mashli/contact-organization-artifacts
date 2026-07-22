# Contact Organization — Supplemental Artifacts

**Companion code and frozen results** for two IJRR submissions:

| Manuscript (not in this repo) | Role |
|-------------------------------|------|
| Paper I — *Contact Organization: A Mesoscopic Object and Measurement Framework…* | Qualification, OPCG, evidence ladder |
| Paper II — *Emergence of Contact Organization: Base-Case Existence on a Declared Witness Class* | Conditional emergence on $\mathfrak{S}_0$ |

> **This repository does not contain the manuscripts** (no `main.tex` / journal PDFs).
> Manuscripts are submitted via Sage Track. For reviewer access during peer review,
> use the journal portal or contact the corresponding author.

**Corresponding author:** Shengli Ma (`mashli@xauat.edu.cn`)  
**Institution:** Xi'an University of Architecture and Technology

---

## What is included

| Path | Contents |
|------|----------|
| `configs/` | Frozen YAML protocols used in the papers (PREREG / reference shells) |
| `scripts/` | Analysis and audit entry points that regenerate table/figure numbers |
| `src/` | Shared Python helpers (if released) |
| `results/frozen/` | Compact JSON/CSV summaries cited by Paper I / II tables |
| `docs/` | Reproducibility notes, preregistered protocol summaries, FAQ pointers |
| `supplemental/` | Independent verification report PDF (optional mirror of Sage supplemental) |

Raw multi-GB trajectory dumps are **not** stored here. If released, they will be
linked from `docs/DATA_LOCATORS.md` (Zenodo / OSF DOI).

---

## Quick start

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # once populated

# Example: regenerate a frozen audit summary (paths TBD after first sync)
python scripts/analyze_mechanism_discrimination.py --help
```

Exact one-command recipes for each paper table will live in:

- `docs/REPRODUCE_PAPER1.md`
- `docs/REPRODUCE_PAPER2.md`

---

## Mapping to manuscripts

### Paper I (qualification evidence)

| Paper object (approx.) | Artifact |
|------------------------|----------|
| Cylinder / ellipsoid / sphere / block ladder | `results/frozen/paper1/canonical_ladder/` |
| Friction sweep / CE-4 | `results/frozen/paper1/friction_sweep/` |
| Partition stability (ARI) | `results/frozen/paper1/partition_stability/` |
| MSM / ITS / CK diagnostics | `results/frozen/paper1/msm_diagnostics/` |
| Cross-geometry deformation / RCI | `results/frozen/paper1/cross_geometry/` |

### Paper II ($\mathfrak{S}_0$ certificates)

| Paper object (approx.) | Artifact |
|------------------------|----------|
| MuJoCo cylinder $\in\mathfrak{S}_0$ | `results/frozen/paper2/s0_mujoco_cylinder/` |
| Ellipsoid / PyBullet partial OPCG | `results/frozen/paper2/s0_partial/` |
| Mechanism discrimination (MD-1–MD-3) | `results/frozen/paper2/mechanism_discrimination/` |
| Predictive audit | `results/frozen/paper2/predictive_audit/` |

Fill concrete filenames in `docs/ARTIFACT_INDEX.md` when syncing from the private
working tree (see `MANIFEST.md`).

---

## License

- **Code** (`scripts/`, `src/`): MIT (see `LICENSE`)
- **Frozen numerical summaries** (`results/frozen/`): CC-BY-4.0 unless a file says otherwise
- **Prereg / protocol docs**: CC-BY-4.0

Manuscripts remain under the journal / author copyright and are **not** licensed here.

---

## Citation

Please cite the published IJRR articles (DOI TBD). Until then, cite the Sage Track
submission titles above and this repository URL:

```text
https://github.com/<ORG_OR_USER>/contact-organization-artifacts
```

Replace `<ORG_OR_USER>` after the public repo is created, and update the Data
availability statement in both manuscripts to the same URL (and Zenodo DOI if used).

---

## Contact

Questions about reproducing a specific table or audit: `mashli@xauat.edu.cn`.
