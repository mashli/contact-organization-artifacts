# Contact Organization — Supplemental Artifacts (Robotica)

Companion **code + frozen results + replayable trajectories** for:

> Auditing Contact Organization in Contact-Rich Manipulation:
> Qualification Criteria, Selective Failures, and a Null Predictive Result

| Path | Contents |
|------|----------|
| `data/robotica/` | Gzipped existing trajectories (~8.6 MB) for the audited cohorts + friction arms |
| `src/cge/` | Frozen scorer surface (`organization_signature`, guards, measures, residence) |
| `configs/robotica/` | Prereg / scorer / trajectory-map YAML |
| `scripts/robotica/` | Path-fixed audits + `run_all.py` one-click check |
| `results/robotica/` | Frozen CSVs/JSON cited by the manuscript tables |

**Corresponding author:** Shengli Ma (`mashli@xauat.edu.cn`)

This repository does **not** contain the manuscript PDF/LaTeX.

---

## One-click reproduce (no MuJoCo required)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/robotica/run_all.py
```

That runs:

1. cylinder threshold shell (`n=179`) — matches Table threshold-shell  
2. signature witness round (P1–P4 / Sig)  
3. $(\mu,t^\ast)$ qualification map with self-check vs frozen friction slice (0 diffs)

Also: `python3 scripts/robotica/run_friction_signature_audit.py` (canonical Sig vs P4′ proxy over the (μ,t*) grid).

Individual entry points:

```bash
python3 scripts/robotica/run_threshold_shell_n179.py
python3 scripts/robotica/run_sig_witness_round.py
python3 scripts/analyze_qualification_map_mu_threshold.py
```

Paths resolve through `scripts/robotica/_paths.py` against `data/robotica/*.json.gz`.
Legacy `results/phase1/...` strings in configs remain accepted as aliases.

---

## Relation to other work (disclosed)

Simulation tooling overlaps with ROBOT-D-26-01372R1 (*Robotics and Autonomous
Systems*, force–mass). That paper does **not** reuse Scheme~D / OPCG /
$\mathfrak{S}_0$ claims or the Robotica data packaged here.

---

## Reviewer checklist

1. `results/robotica/sig_witness_round.csv` — signature accepts/refuses  
2. `results/robotica/predictive_paired_delta_ci.csv` — Org vs B0n  
3. `results/robotica/qualification_map_mu_threshold.csv` — $(\mu,t^\ast)$ band  
4. `results/robotica/*_msm_diagnostics.json` — block $\lambda_2=0.889$ with P1 fail  
5. Re-run `python3 scripts/robotica/run_all.py` on a clean clone
