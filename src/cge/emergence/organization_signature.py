"""P1–P4 organization signature audit (Emergence Theory — Phase 0, offline).

Checks necessary organizational properties on frozen landscape trajectories.
Does NOT claim Organization is proved — only signatures consistent / inconsistent.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cge.emergence.guard_partition import estimate_guard_partitions
from cge.emergence.measure import estimate_measures, load_trajectory_episodes
from cge.emergence.residence import estimate_residence_times


@dataclass
class PropertyCheck:
    property_id: str
    label: str
    pass_: bool
    verdict: str
    metrics: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = d.pop("pass_")
        return d


@dataclass
class CounterexampleFlags:
    """CE-1..4 observables on this cohort (reference documents counterexamples)."""

    CE1_feature_measure_uniform: bool = False
    CE2_multiplicity_wrong_spectrum: bool = False
    CE3_feature_msm_misaligned: bool = False
    CE4_no_loaded_regime: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrganizationSignatureReport:
    cohort_name: str
    source_path: str
    n_episodes: int
    n_frames: int
    fx_threshold: float
    properties: list[PropertyCheck] = field(default_factory=list)
    counterexamples: CounterexampleFlags = field(default_factory=CounterexampleFlags)
    all_properties_pass: bool = False
    organization_signatures_consistent: bool = False
    k_mode: int = 0
    lambda2_mode: float = float("nan")
    g_fx_spectrum_match: float = float("nan")
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_name": self.cohort_name,
            "source_path": self.source_path,
            "n_episodes": self.n_episodes,
            "n_frames": self.n_frames,
            "fx_threshold": self.fx_threshold,
            "properties": {p.property_id: p.to_dict() for p in self.properties},
            "counterexamples": self.counterexamples.to_dict(),
            "all_properties_pass": self.all_properties_pass,
            "organization_signatures_consistent": self.organization_signatures_consistent,
            "k_mode": self.k_mode,
            "lambda2_mode": self.lambda2_mode,
            "g_fx_spectrum_match": self.g_fx_spectrum_match,
            "summary": self.summary,
        }


def _fx_bimodal_at_guard(
    episodes: list[dict],
    *,
    fx_threshold: float,
) -> dict[str, Any]:
    """P4 helper: loaded vs grazing |f_x| separation at guard threshold."""
    fx_loaded: list[float] = []
    fx_grazing: list[float] = []
    for ep in episodes:
        for fr in ep.get("frames") or []:
            fx = abs(float(fr.get("fx", 0.0)))
            basin = str(fr.get("basin", ""))
            if basin == "loaded":
                fx_loaded.append(fx)
            elif basin == "grazing":
                fx_grazing.append(fx)

    if not fx_loaded or not fx_grazing:
        return {
            "has_both_basins": False,
            "median_fx_loaded": float("nan"),
            "median_fx_grazing": float("nan"),
            "guard_separates": False,
        }

    med_l = float(np.median(fx_loaded))
    med_g = float(np.median(fx_grazing))
    return {
        "has_both_basins": True,
        "median_fx_loaded": med_l,
        "median_fx_grazing": med_g,
        "fx_threshold": fx_threshold,
        "guard_separates": med_l >= fx_threshold and med_g < fx_threshold,
        "median_gap": med_l - med_g,
    }


def _guard_lookup(guard_rep, guard_id: str):
    for g in guard_rep.guards:
        if g.guard_id == guard_id:
            return g
    return None


def audit_organization_signatures(
    episodes: list[dict],
    *,
    cohort_name: str = "",
    source_path: str = "",
    fx_threshold: float = 8.0,
    transfer_report: dict[str, Any] | None = None,
) -> OrganizationSignatureReport:
    """Audit P1–P4 + CE flags on one trajectory cohort."""
    n_frames = sum(len(ep.get("frames") or []) for ep in episodes)
    n_eps = len(episodes)

    if n_frames == 0:
        empty = OrganizationSignatureReport(
            cohort_name=cohort_name,
            source_path=source_path,
            n_episodes=n_eps,
            n_frames=0,
            fx_threshold=fx_threshold,
            summary="no_frames",
        )
        return empty

    residence = estimate_residence_times(episodes, source_path=source_path)
    measures = estimate_measures(episodes, source_path=source_path, fx_threshold=fx_threshold)
    guard_rep = estimate_guard_partitions(episodes, source_path=source_path, fx_threshold=fx_threshold)
    mu_f = measures["feature_mu_F"]
    mu_o = measures["organization_mu_O"]

    loaded = residence.by_basin.get("loaded")
    grazing = residence.by_basin.get("grazing")
    g_fx = _guard_lookup(guard_rep, "G_fx_mode")
    g_mult = _guard_lookup(guard_rep, "G_multiplicity")
    fx_stats = _fx_bimodal_at_guard(episodes, fx_threshold=fx_threshold)

    mode_k = int(guard_rep.mode_spectrum.get("n_states", 0) or 0)
    mode_l2 = float(guard_rep.mode_spectrum.get("lambda2", float("nan")))
    g_fx_match = float(g_fx.spectrum_match_score) if g_fx else float("nan")

    # --- P1: multiple long-lived occupied regions ---
    p1_loaded_runs = loaded.n_runs if loaded else 0
    p1_loaded_mean = loaded.mean_run_length if loaded else float("nan")
    p1_grazing_mean = grazing.mean_run_length if grazing else float("nan")
    p1_p6_loaded = float(loaded.survival_tail.get("6", float("nan"))) if loaded else float("nan")
    p1_p6_grazing = float(grazing.survival_tail.get("6", float("nan"))) if grazing else float("nan")
    p1_tail_sep = residence.tail_separation_at_t4
    n_basins = len(residence.by_basin)
    p1_pass = (
        n_basins >= 2
        and p1_loaded_runs >= 3
        and math.isfinite(p1_loaded_mean)
        and p1_loaded_mean >= 2.0
        and math.isfinite(p1_grazing_mean)
        and p1_grazing_mean >= 1.0
        and (
            (math.isfinite(p1_p6_loaded) and p1_p6_loaded > 0.0)
            or p1_loaded_mean >= 4.0
        )
        and (
            math.isfinite(p1_tail_sep) and p1_tail_sep >= 1.0
            or (
                math.isfinite(p1_p6_loaded)
                and math.isfinite(p1_p6_grazing)
                and p1_p6_loaded > p1_p6_grazing
            )
        )
    )
    p1 = PropertyCheck(
        property_id="P1",
        label="Multiple long-lived occupied regions",
        pass_=p1_pass,
        verdict="pass" if p1_pass else "fail",
        metrics={
            "n_basins": n_basins,
            "loaded_n_runs": p1_loaded_runs,
            "loaded_mean_tau": p1_loaded_mean,
            "grazing_mean_tau": p1_grazing_mean,
            "P_tau_gt_6_loaded": p1_p6_loaded,
            "P_tau_gt_6_grazing": p1_p6_grazing,
            "tail_separation_at_t4": p1_tail_sep,
            "metastability_ratio": residence.metastability_ratio,
        },
        rationale=(
            "Requires ≥2 basins, loaded runs with τ̄≥2 and heavy tail "
            "(P(τ>6)>0 or τ̄≥4), heavier than grazing."
        ),
    )

    # --- P2: finite guard-mediated transitions ---
    switch_rate = residence.n_regime_switches / max(n_frames, 1)
    cross_edges = len(residence.transition_counts)
    p2_pass = (
        mode_k == 2
        and residence.n_regime_switches >= 1
        and switch_rate <= 0.20
        and cross_edges >= 2
        and math.isfinite(g_fx_match)
        and g_fx_match >= 0.95
    )
    p2 = PropertyCheck(
        property_id="P2",
        label="Finite guard-mediated transitions",
        pass_=p2_pass,
        verdict="pass" if p2_pass else "fail",
        metrics={
            "k_mode": mode_k,
            "n_regime_switches": residence.n_regime_switches,
            "switch_rate": switch_rate,
            "cross_basin_edge_types": cross_edges,
            "transition_counts": residence.transition_counts,
            "G_fx_spectrum_match": g_fx_match,
            "lambda2_mode": mode_l2,
        },
        rationale="K=2 MODE, sparse cross-basin switches, finite transition graph, G_fx≈MODE.",
    )

    # --- P3: persistent occupation imbalance ---
    mu_by_basin = mu_o.get("mu_by_basin") or {}
    pi_by_u = mu_o.get("pi_by_u") or {}
    pi_vals = [float(v) for v in pi_by_u.values()]
    pi_range = max(pi_vals) - min(pi_vals) if pi_vals else 0.0
    pi_sp = float(mu_o.get("pi_u_spearman", float("nan")))
    joint_ratio = float(mu_o.get("joint_entropy_ratio", float("nan")))
    frac_loaded = float(mu_f.get("mean_frac_loaded", 0.0))
    min_basin_frac = min(mu_by_basin.values()) if mu_by_basin else 0.0
    p3_pass = (
        len(mu_by_basin) >= 2
        and min_basin_frac >= 0.05
        and frac_loaded >= 0.05
        and frac_loaded <= 0.95
        and (
            (math.isfinite(pi_sp) and abs(pi_sp) >= 0.25)
            or pi_range >= 0.15
            or (math.isfinite(joint_ratio) and joint_ratio < 0.98)
        )
        and bool(mu_o.get("has_loaded_to_grazing") or cross_edges >= 2)
    )
    p3 = PropertyCheck(
        property_id="P3",
        label="Persistent occupation imbalance",
        pass_=p3_pass,
        verdict="pass" if p3_pass else "fail",
        metrics={
            "mu_by_basin": mu_by_basin,
            "mean_frac_loaded": frac_loaded,
            "pi_by_u": pi_by_u,
            "pi_u_range": pi_range,
            "pi_u_spearman": pi_sp,
            "joint_entropy_ratio": joint_ratio,
            "has_loaded_to_grazing": mu_o.get("has_loaded_to_grazing"),
        },
        rationale="Non-uniform basin + u occupation; structured π(u); cross-basin topology.",
    )

    # --- P4: mechanically meaningful boundaries ---
    g_fx_ari = float(g_fx.ari_vs_mode) if g_fx else float("nan")
    p4_pass = (
        math.isfinite(g_fx_ari)
        and g_fx_ari >= 0.99
        and math.isfinite(g_fx_match)
        and g_fx_match >= 0.99
        and fx_stats.get("guard_separates", False)
    )
    p4 = PropertyCheck(
        property_id="P4",
        label="Mechanically meaningful boundaries",
        pass_=p4_pass,
        verdict="pass" if p4_pass else "fail",
        metrics={
            "G_fx_ari_vs_mode": g_fx_ari,
            "G_fx_spectrum_match": g_fx_match,
            "fx_bimodal": fx_stats,
            "fx_threshold": fx_threshold,
        },
        rationale="Load guard G_fx ≡ MODE; |f_x| bimodal at threshold t*.",
    )

    properties = [p1, p2, p3, p4]
    all_pass = all(p.pass_ for p in properties)

    # CE flags (documented counterexample observables)
    msm_ari = float("nan")
    if transfer_report:
        msm_ari = float(transfer_report.get("msm_ari_vs_mode", float("nan")))

    ce1 = bool(measures.get("m1_on_feature_falsified"))
    ce2 = bool(
        g_mult is not None
        and math.isfinite(g_mult.spectrum_match_score)
        and g_mult.spectrum_match_score < 0.60
    )
    ce3 = math.isfinite(msm_ari) and msm_ari < 0.15
    ce4 = p1_loaded_runs == 0 or frac_loaded < 0.01

    counterexamples = CounterexampleFlags(
        CE1_feature_measure_uniform=ce1,
        CE2_multiplicity_wrong_spectrum=ce2,
        CE3_feature_msm_misaligned=ce3,
        CE4_no_loaded_regime=ce4,
        metrics={
            "feature_entropy_ratio": mu_f.get("entropy_ratio"),
            "G_multiplicity_spectrum_match": g_mult.spectrum_match_score if g_mult else None,
            "G_multiplicity_lambda2": g_mult.lambda2 if g_mult else None,
            "msm_ari_vs_mode": msm_ari,
            "mean_frac_loaded": frac_loaded,
        },
    )

    if all_pass:
        summary = "signatures_consistent_with_organization"
    elif ce4:
        summary = "organization_signatures_absent_no_loaded_regime"
    else:
        failed = [p.property_id for p in properties if not p.pass_]
        summary = f"signatures_incomplete_failed_{'_'.join(failed)}"

    return OrganizationSignatureReport(
        cohort_name=cohort_name,
        source_path=source_path,
        n_episodes=n_eps,
        n_frames=n_frames,
        fx_threshold=fx_threshold,
        properties=properties,
        counterexamples=counterexamples,
        all_properties_pass=all_pass,
        organization_signatures_consistent=all_pass,
        k_mode=mode_k,
        lambda2_mode=mode_l2,
        g_fx_spectrum_match=g_fx_match,
        summary=summary,
    )


def audit_from_path(
    path,
    *,
    cohort_name: str = "",
    fx_threshold: float | None = None,
    transfer_report: dict[str, Any] | None = None,
) -> OrganizationSignatureReport:
    meta, eps = load_trajectory_episodes(path)
    fx_t = float(fx_threshold if fx_threshold is not None else meta.get("fx_threshold", 8.0))
    rel = str(path)
    return audit_organization_signatures(
        eps,
        cohort_name=cohort_name or rel,
        source_path=rel,
        fx_threshold=fx_t,
        transfer_report=transfer_report,
    )
