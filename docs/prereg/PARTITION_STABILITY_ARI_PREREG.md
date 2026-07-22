# Partition Stability (Matched-Frame ARI) — Preregistration

**Status:** preregistered · **Exp-A RUN 2026-07-09** → phase-matched PARTITION_DRIFTS (ARI
0.13–0.52; determinism 1.0; audit §9.8) · **Force-matched T1/T2 test RUN 2026-07-09** →
NOT_REPRESENTATIVE (`f_cmp<0.5`); point est. L0-axis leans T1, ax/lat leans T2; audit §9.9 ·
**Partition status = UNRESOLVED** · representative force-matched re-test + Exp-B/C pending ·
created 2026-07-09
**Decides:** whether the guard-induced *partition* (cell boundary identity) — not just the
cell *vocabulary* — is protocol-stable. This is the experiment that determines whether the
two-layer object $\mathcal O=(\mathcal G,\mathcal M)$ may be written into Paper I.

> **Freeze rule:** the alignment rule (§3) and the ARI decision bands (§4) are fixed by this
> document *before* any run. No post-hoc adjustment of alignment or thresholds is permitted.
> If a design flaw is found, it must be corrected here, dated, and the run re-executed — not
> patched after seeing results.

---

## 0. What this experiment is, and is NOT

| Proposition | Status entering this experiment | Established by |
|---|---|---|
| Fixed guard thresholds, no per-condition fit | ✅ established | code audit (`COT_PROTOCOL_INVARIANCE_AUDIT.md` §9.6) |
| **Vocabulary stability** — 4-symbol alphabet appears in all protocols | ✅ established | balanced operator audit §9.3 |
| **Partition stability** — same physical state → same cell across protocols | ⏳ **UNKNOWN — this experiment** | matched-frame ARI |
| Dynamics ($P,\pi,\eta,\kappa$) protocol-conditioned | ✅ established | balanced operator audit §9.3 |
| Two-layer object $\mathcal O=(\mathcal G,\mathcal M)$ | ⏳ interpretation, not written to paper | requires partition PASS |

**Why vocabulary ≠ partition.** The guards can *only* emit the four symbols by
construction, so an identical alphabet is nearly automatic once each cell is populated.
The non-trivial question is whether a given physical contact configuration keeps the same
label across protocols. Counterexample the test must be able to catch: a frame that is
`loaded_ax` under protocol A but `loaded_lat` under protocol B — identical vocabulary,
identical grammar edge-set, yet the **boundary has drifted**. ARI is label-permutation
invariant, so it measures partition agreement (co-membership structure), not label names.

---

## 1. Hypothesis

> Under the single fixed shared guard threshold set, physically comparable frames receive
> the **same** cell label across protocol conditions, i.e. the guard-induced partition is
> protocol-stable (ARI in the "good"/"excellent" band). The competing outcome is that the
> partition boundary drifts systematically with protocol (ARI in the "moderate"/"poor"
> band), in which case the Paper I object definition must be revised.

---

## 2. Hard prerequisite (non-negotiable)

Every protocol/geometry condition in this experiment MUST label frames with the **same
fixed hardcoded guard thresholds** already audited (`f_x ≥ 8.0`, multiplicity `0.5/1.5`,
coupling `0.15`, spread `0.002`). If any per-condition calibration is introduced, the
result is void: a "stable" ARI would then just be the vocabulary artifact repeated, not
independent evidence. The analysis script must assert a single `fx` value and single
guard-catalog instance shared across all conditions, and log it.

---

## 3. Frame alignment rule (FROZEN) — phase-relative, never absolute frame index

**Problem this prevents.** With different hold lengths, episodes have different total
durations. Comparing "frame 100 in A" to "frame 100 in B" is meaningless: frame 100 may be
mid-hold in A but still in ramp in B. A low ARI from that mismatch would be a trivial
time-axis misalignment artifact — the same class of error as the earlier "fixed transition
count" bug.

**Frozen alignment protocol:**

1. **Shared initial condition.** Compared protocols A and B are replayed from the **same
   sampled initial conditions** (same `episode_id` cohort, same seed). Determinism given
   `(init, action schedule)` is assumed; §3a checks it.
2. **Phase tagging (already in the data).** Each frame carries
   `protocol_phase ∈ {ramp, hold, settle, tail}` and `phase_step_from_start` (the
   `collect_landscape_trajectories` annotation), used **instead of** absolute frame index.
3. **Comparable unit = `(episode_id i, protocol_phase, phase_step_from_start k)`.** ARI is
   computed over the set of units present in **both** protocols. For the varying dimension,
   phase steps are matched only up to `k ≤ min(len_A, len_B)` of that phase; steps beyond
   the shorter phase are **excluded** (not zero-filled, not wrapped).
4. **Headline = hold-phase, phase-step-matched comparison** — where the protocols genuinely
   differ. (In Exp-A ramp differs by design, so ramp-phase units match only partially and
   are reported separately, never as the headline.)
5. **Two partitions compared.** For each matched unit we have label$_A$ and label$_B$; ARI
   is computed between the two label vectors over all matched units. ARI is
   label-permutation invariant, so it measures co-membership structure, not label names.

**§3a. Determinism / self-replay sanity floor (FROZEN).** Before the cross-protocol number
is read, replay one protocol **twice** from identical inits and compute ARI on identically
tagged units. This must give **ARI = 1.0**. If it does not, the engine is non-deterministic
and the whole pipeline is suspect → **stop** and switch to the pre-declared fallback below.
(For same-ramp comparisons the identical ramp prefix additionally gives a trivial ARI ≈ 1
control.)

**Fallback (pre-declared).** If deterministic replay from identical inits is not achievable,
fall back to **phase-binned distributional matching**: within each `(phase, k)` bin, compare
the cell assignment of nearest-neighbor frames in contact-feature space. This fallback is
declared here, before running, so its use is not a post-hoc choice.

**§3b. Structured missingness rule (FROZEN).** A unit `(i, phase, k)` can be **absent** in a
protocol for two reasons: (a) the frame was dropped because `multiplicity_M < 1` (no
established contact at that step), or (b) the episode terminated early (e.g. escape) so that
step never occurred. Such a unit is **excluded from the ARI computation** (it has no label
under one protocol), but the missingness is **not** silently discarded:

- The **comparable-sample fraction** `f_cmp = (#units present in both) / (#units present in
  at least one)` is computed **per (condition-pair, phase)** and reported as a **first-class
  metric alongside ARI** — a low `f_cmp` means the ARI number rests on a small, possibly
  unrepresentative overlap and must be read with that caveat.
- The **missingness pattern itself is logged** (per protocol: fraction of units dropped for
  reason (a) vs (b), by phase), because protocol-systematic early-escape is itself
  informative (e.g. short protocols escaping more), not mere nuisance.
- If `f_cmp < 0.50` for a headline comparison, the ARI verdict for that comparison is marked
  **NOT-REPRESENTATIVE** and cannot be used to declare PASS/FAIL (only reported).

---

## 4. Decision bands (FROZEN) — Steinley (2004) ARI recovery guidelines

Adjusted Rand index interpretation, per Steinley, D. (2004), *Properties of the
Hubert–Arabie adjusted Rand index*, Psychological Methods 9(3), 386–396 — the standard
cluster-validation benchmark:

| ARI range | Steinley label | Verdict for **partition stability** |
|---|---|---|
| ≥ 0.90 | excellent recovery | **PARTITION STABLE** — partition is protocol-invariant |
| 0.80 – 0.90 | good recovery | **PARTITION STABLE (qualified)** — report minor drift |
| 0.65 – 0.80 | moderate recovery | **INCONCLUSIVE** — boundary partially drifts; do not claim invariance |
| < 0.65 | poor recovery | **PARTITION DRIFTS** — Paper I object definition must be revised |

**Primary decision threshold: ARI ≥ 0.80** to declare partition stability (aligns with
"good or excellent" recovery). ARI < 0.65 falsifies partition invariance.

These bands are fixed now, before any number is seen. No renegotiation post-hoc. In
particular, an observed `0.78` is **INCONCLUSIVE**, not "also fine."

### 4a. Composite pass criterion (FROZEN)

A single scalar ARI cannot distinguish "every boundary drifts a little" from "one boundary
drifts a lot" — theoretically very different. So the pass criterion is **composite**:

```
PARTITION STABLE  ⟺  ARI ≥ 0.80  AND  no off-diagonal boundary confusion > 10%
```

where "boundary confusion" is read from the mandatory Boundary Drift Matrix (§4b). Both
conditions must hold; ARI ≥ 0.80 with a single 30%-drifting cell is **not** declared stable
(it is flagged as *localized boundary drift*).

### 4b. Boundary Drift Matrix (MANDATORY output — always report, not only on failure)

Every ARI comparison must be accompanied by a per-cell **confusion matrix** $M$ over the
matched units, rows = label under protocol A, columns = label under protocol B, row-normalized
to percentages:

```
            B: gc   gd   la   ll
   A: gc    [ 93    5    2    0 ]
      gd    [  4   91    3    2 ]
      la    [  1    2   96    1 ]
      ll    [  0    3    2   95 ]
```

- **Diagonal** = fraction of a cell's frames that keep their label across protocols
  (per-cell stability).
- **Off-diagonal** = boundary drift; the largest off-diagonal per row is the "confusion" in
  §4a. Report per-cell stability (diagonal) explicitly, e.g. `loaded_ax 96% stable,
  grazing_compact 71% → localized drift`.
- This localizes *which* boundary moves, turning an ambiguous ARI into a mechanistic
  statement about the object's structure.

**Minimum cell support (FROZEN, added 2026-07-09 before the full run).** A cell (row) with
very few matched units can show 100% "confusion" from a single frame, which is noise, not
boundary drift. Therefore the composite criterion (§4a) only counts a cell's off-diagonal
confusion toward FAIL if that cell has **≥ 20 matched units** in the comparison. Rows below
this support are still reported (with a `low_support` flag and their raw counts) but do
**not** drive the PASS/FAIL gate. Both the supported-only max confusion (used for the
verdict) and the raw max confusion (all rows) are reported for transparency.

---

## 5. Experiments — priority ordered by THEORETICAL RISK (not convenience)

All reuse the fixed guard thresholds (§2), the frozen alignment (§3), the composite
criterion (§4a), and the mandatory drift matrix (§4b).

### Priority 1 — Exp-A: fix hold, vary ramp (ARI) — IMPLEMENT FIRST

Highest information gain. The dominant confound in the hold sweep is that **hold changes
exposure/dwell**. Ramp does **not**: it changes the *entry trajectory* into the hold, not
the dwell length. So Exp-A is the first clean test of the real question:

> Is the cell boundary a property of the **system geometry**, or of the **protocol timing**?

If the partition truly represents Contact Organization, different ramps (different entry
paths) should land in the **same** cell boundary. ARI ≥ 0.80 with no >10% boundary
confusion here is worth more than ten more hold sweeps — it is the first genuine
**boundary-stability** result. If it fails, that is an equally valuable, concrete revision
signal for the Paper I object definition.

### Priority 2 — Exp-B: fix ramp, vary hold (ARI)

Uses data we largely already have. Less critical than Exp-A because hold **natively** changes
occupation, so a hold-swept ARI is intrinsically harder to interpret cleanly (drift could be
occupation reweighting rather than boundary motion). Report, but subordinate to Exp-A.

### Priority 3 — Exp-C: fix grammar, vary protocol; plot $P(h)$ — LAST

Not a validation, a **characterization**: the continuous operator-vs-protocol curve, chosen
*after* the ARI verdict tells us which object to characterize. Exposes the κ (1.31→3.53) vs
η (saturating) asymmetry explicitly rather than via a single "net effect".

### Later / separate axis — Exp-GEN: vary object geometry (generality, not causal origin)

Cross-geometry partition ARI (cylinder / sphere / block) tests *generality* of the boundary
under fixed thresholds. Deliberately separated from Exp-A/B so geometry-generality is not
confused with protocol-causal-origin. Deferred until Exp-A resolves.

## 5a. Bootstrap CI (FROZEN) — resampling unit = **episode**, never frame

Frames within an episode are highly autocorrelated (a trajectory is not a set of independent
draws). Frame-level bootstrap would systematically **understate** the CI and give false
confidence. Therefore:

```
bootstrap resampling unit = episode (init condition)
```

Each bootstrap replicate resamples **whole episodes with replacement**, treating each
episode's entire phase-matched label sequence as one indivisible analysis unit, then recomputes
ARI (and the drift matrix) on the resampled cohort. The reported CI (e.g. 95% percentile
interval over ≥1000 episode-bootstrap replicates) then answers the correct question: *how
much would this ARI move under a different draw of episodes?* Declare the minimum matched
episode count per comparison before running; hold=2 remains a reducible small-$n$ boundary
case (reported, not verdict-setting).

---

## 6. Status ladder (what the paper may say, keyed to outcome)

```
Alphabet   :  PASS         (vocabulary invariance — implementation audit)
Partition  :  UNKNOWN      (this experiment; PASS iff ARI ≥ 0.80 under §3/§4)
Dynamics   :  CONDITIONED  (protocol-conditioned; established)
```

- If **Partition = PASS** → the two-layer object $\mathcal O=(\mathcal G,\mathcal M)$ is
  admissible; Paper I definition chapter may be upgraded (Experiment D).
- If **Partition = INCONCLUSIVE/DRIFTS** → Paper I object definition is revised to make the
  partition explicitly protocol-conditioned; the "structural stability" claim is dropped.

---

## 7. Anti-patterns this preregistration exists to prevent

1. Aligning frames by absolute index across unequal-length episodes (time-axis artifact).
2. Choosing "high vs low ARI" cutoffs after seeing the numbers (the loose post-hoc
   criteria that discredited the earlier hypothesis-discrimination matrix).
3. Re-fitting guard thresholds per condition (would make partition stability a repeat of
   the vocabulary artifact).
4. Writing $\mathcal O=(\mathcal G,\mathcal M)$ into a paper before Partition = PASS.
5. Frame-level bootstrap (understates CI via intra-episode autocorrelation) — unit is the
   episode (§5a).
6. Reporting a bare ARI without the Boundary Drift Matrix (§4b) or the comparable-sample
   fraction (§3b) — a scalar ARI alone hides both *where* the boundary moves and *how
   representative* the overlap is.
7. Silently dropping protocol-systematic early-escape missingness instead of logging it
   (§3b).

---

## 8. Force-matched discriminating test (T1 vs T2) — FROZEN addendum 2026-07-09

**Why.** Exp-A (§9.8 of the audit) aligned on the *protocol* coordinate `(episode, phase,
phase_step)`. It proved the map **(protocol, phase-step) → cell** is protocol-conditioned,
but it **cannot** distinguish the two mechanisms both consistent with that result:

- **T1 — Trajectory drift.** Guards are fixed; a longer ramp simply reaches a *different
  physical state* at the same phase-step (e.g. more accumulated normal force), so the label
  differs. The organizational boundary itself is stable; only the trajectory moved.
- **T2 — Coordinate-dependent partition.** Even at a *matched physical state* the label
  still differs, i.e. the organization is not stable in physical coordinates either.

Literal "the guard drifts" is already excluded by the code audit (fixed pure function). So
this test asks the sharper question: **does the cell label depend on protocol *beyond* what
the physical alignment coordinate explains?**

**FROZEN design (no new replay — reuses the Exp-A ramp{3,6,12}/hold8 caches):**

1. **Alignment coordinate = normal load `resultant_force_f_x`** (the organization's primary
   axis; the coordinate the L0 boundary lives on). Binned in **fixed 1.0 N bins with edges on
   integers**, so `8.0 N` (the load threshold) is a bin *edge* — no bin straddles the L0
   boundary.
2. **Matching unit = `(episode_id, f_x_bin)`** present in **both** protocols. Within a unit,
   the representative label is the frame **closest to bin center in `f_x`** (deterministic
   tie-break: earliest step). Per-episode (same init ⇒ physically comparable at matched load).
3. **Metrics (all with episode-bootstrap CI, drift matrix, `f_cmp`):**
   - `ari_L1_full` — over all four cells at matched `f_x`.
   - `ari_within_loaded` — ax vs lat, restricted to loaded bins (`f_x ≥ 8`). This is the pure
     **coupling-boundary** test at matched load.
   - `ari_within_grazing` — compact vs dispersed, restricted to grazing bins. Pure
     **spread-boundary** test at matched load.
4. **Upper-bound control (analogue of the §3a determinism floor).** Also align on the *full*
   guard-relevant vector `(f_x, f_y, patch_spread)` with fine bins; ARI **must ≈ 1.0** by
   construction (label is a fixed function of these). If it is not, the pipeline is buggy.

**FROZEN verdict map (Steinley bands, decided before the numbers):**

| Force-matched `ari_L1_full` (and sub-ARIs) | Mechanism | Consequence |
|---|---|---|
| ≥ 0.80 | **T1 trajectory drift** | Boundary stable in physical coords; Exp-A drift = entry-force accumulation. Paper I object definition **stands**, with the clarification: *protocol changes the trajectory→organization map, not the boundary.* |
| 0.65 – 0.80 | inconclusive | Report; neither mechanism declared. |
| < 0.65 | **T2 coordinate-dependent partition** | Real turning point: partition drifts even at matched load. Revise Paper I object definition **and** re-examine whether the guard omits a physical variable that governs organization. |

**Epistemological framing (report, not decoration).** If phase-alignment gives ARI ≈ 0.13
while force-alignment gives ARI ≈ 0.95, the finding is not "protocol invariance" but:
*Contact Organization is coordinate-invariant only in physically matched coordinates* — i.e.
whether the organization looks stable depends first on comparing it in the right physical
frame. That is a representation-dependence result, of interest beyond this system (MSM /
dynamical-systems / representation-learning all hit it).

**Do NOT rewrite Paper I to "partition is protocol-conditioned" until this test runs.** Exp-A
only licenses "the phase-step-indexed realized partition is protocol-conditioned."
