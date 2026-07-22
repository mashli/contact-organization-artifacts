# MANIFEST — what to sync into the public artifacts repo

**Status:** Batch 1 curated copy is already populated in this scaffold directory
(`results/frozen/`, `configs/`, `scripts/`, `docs/prereg/`, `supplemental/`).
See `SYNC_LIST.md` for the exact file list and push commands.

Private working tree (this project):

```text
contact_grammar_emergence/
```

Public repo (create on GitHub, then copy this entire scaffold folder):

```text
contact-organization-artifacts/
```

Current private `results/` is ~637 MB. Do **not** mirror it wholesale.
Batch 1 frozen payload is ~584 KB.

---

## Include (recommended)

### From `configs/`

Copy PREREG / locked shells used in Paper I–II, for example:

- `contact_graph_landscape_convergence*_PREREG.yaml`
- `cross_geometry_PREREG.yaml`
- `cross_engine_pybullet_probe_PREREG.yaml`
- `controlled_perturbation_PREREG.yaml`
- friction / sphere / ellipsoid / block related PREREG YAMLs cited in the papers

Skip exploratory one-off configs not cited in the manuscripts.

### From `scripts/`

Copy scripts that regenerate frozen audits (examples; prune after dry-run):

- `analyze_mechanism_discrimination*.py`
- `analyze_friction_sweep.py`
- `analyze_guard_threshold_sensitivity.py`
- `analyze_episode_matched_boundary_stability.py`
- `analyze_l1_boundary_proximity.py`
- `analyze_cross_engine_probe.py`
- `analyze_ce2_msm_diagnostics.py`
- related `run_*.sh` entry points for the above

Do not copy `__pycache__/` or unpublished exploratory notebooks.

### From `results/` → `results/frozen/`

Curate **small JSON/CSV** that back manuscript tables, e.g.:

| Destination | Source hint (private) |
|-------------|------------------------|
| `results/frozen/paper1/cross_geometry/` | `results/phase1/cross_geometry/*.json` |
| `results/frozen/paper1/friction_sweep/` | `results/phase1/friction_sweep_n50/**` summaries only |
| `results/frozen/paper2/s0_*` | OPCG / certificate JSON cited in Paper II |
| `results/frozen/paper2/mechanism_discrimination/` | hold-sweep operator metrics |

Omit episode dumps, videos, and large binary caches.

### From `docs/` (selected)

- Short PREREG protocol summaries that reviewers need to audit thresholds
- Optional: `docs/latex_ijrr/supplementary/INDEPENDENT_VERIFICATION_REPORT.pdf`
  → `supplemental/` (also upload via Sage Track if required)

### From `src/`

Only modules required by the released scripts.

---

## Exclude (hard rules)

| Path / pattern | Why |
|----------------|-----|
| `docs/latex_ijrr/paper1_object/`, `paper2_theory/`, `paper3_*` | Main manuscripts |
| `docs/latex_cot/` full TeX tree | Manuscript source |
| `**/backup*/` | Stale drafts |
| `docs/latex_ijrr/IJRR_*LETTER*`, rebuttal/response drafts | Editorial private |
| `notes/` | Internal lab notes |
| Raw `results/stage*`, full `results/phase1` trees | Too large / regenerable |
| `.env`, credentials, absolute machine paths | Security |

---

## Suggested bootstrap commands

```bash
# 1) Create empty GitHub repo: contact-organization-artifacts
# 2) On your machine:
PRIV=/path/to/contact_grammar_emergence
PUB=/path/to/contact-organization-artifacts

cp -R "$PRIV/docs/latex_ijrr/artifacts_repo_scaffold/." "$PUB/"
mkdir -p "$PUB"/{configs,scripts,src,results/frozen/paper1,results/frozen/paper2,docs,supplemental}

# 3) Copy curated files (edit lists after checking citations)
# rsync -av --include='*_PREREG.yaml' --exclude='*' "$PRIV/configs/" "$PUB/configs/"
# ... then scripts and frozen JSON ...

# 4) Fill docs/ARTIFACT_INDEX.md with exact filenames ↔ paper tables
# 5) git init && git add && git commit && git push
```

---

## After the repo is public

1. Replace `https://github.com/<ORG_OR_USER>/contact-organization-artifacts` in `README.md`.
2. Update both papers’ Data availability paragraph to the same URL (+ Zenodo DOI if any).
3. Paste the URL into Sage Track (and keep an anonymous mirror only if the journal requests double-blind; IJRR is typically not double-blind for these submissions).
4. Tag a release, e.g. `ijrr-submission-2026-07`, matching the frozen file set cited in the PDFs.
