"""Organization and feature occupation measures from landscape trajectories (ToCO Ch 4).

Two measures (not one):
  μ_ℱ  — feature-space histogram occupation (representation candidate; falsified as org object)
  μ_𝒪  — basin-level organization measure AFTER coarse-graining Π (MODE basins)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Continuous ℱ chart (mirror formation_features.FORMATION_SCALAR_FEATURES minus multiplicity_M).
CONTINUOUS = (
    "patch_centroid_c_x",
    "patch_centroid_c_y",
    "patch_spread_s_xy",
    "patch_covariance_Sigma_xx",
    "patch_covariance_Sigma_xy",
    "patch_covariance_Sigma_yy",
    "resultant_force_f_x",
    "resultant_force_f_y",
    "resultant_force_f_z",
    "resultant_torque_tau_x",
    "resultant_torque_tau_y",
    "resultant_torque_tau_z",
    "mean_normal_n_x",
    "mean_normal_n_y",
    "mean_normal_n_z",
)


def load_trajectory_episodes(path: Path | str) -> tuple[dict, list[dict]]:
    p = Path(path)
    if str(p).endswith(".gz"):
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
    return payload.get("metadata", {}), list(payload.get("episodes", []))


def _frame_feature_matrix(episodes: list[dict]) -> tuple[np.ndarray, list[str], list[dict]]:
    """Stack all contact frames; attach episode_id, u, basin, step."""
    rows_meta: list[dict] = []
    feat_rows: list[list[float]] = []
    for ep in episodes:
        u = float(ep.get("push_eccentricity_m", 0.0))
        eid = int(ep.get("episode_id", -1))
        for fr in ep.get("frames") or []:
            feats = fr.get("features") or {}
            feat_rows.append([float(feats.get(k, 0.0)) for k in CONTINUOUS])
            rows_meta.append(
                {
                    "episode_id": eid,
                    "u": u,
                    "basin": str(fr.get("basin", "")),
                    "step": int(fr.get("step", 0)),
                    "fx": float(fr.get("fx", 0.0)),
                }
            )
    if not feat_rows:
        return np.zeros((0, len(CONTINUOUS))), CONTINUOUS, []
    return np.asarray(feat_rows, dtype=np.float64), CONTINUOUS, rows_meta


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        z = np.zeros_like(x)
        return z, np.zeros(x.shape[1]), np.ones(x.shape[1])
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-12)
    return (x - mean) / std, mean, std


def renyi2_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return float("nan")
    return float(-math.log2(np.sum(p * p)))


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimension via covariance participation ratio."""
    if len(x) < 3:
        return float("nan")
    z, _, _ = _standardize(x)
    cov = np.cov(z, rowvar=False)
    tr = float(np.trace(cov))
    if tr < 1e-18:
        return float("nan")
    return float(tr * tr / np.sum(cov * cov))


def _histogram_probs(x: np.ndarray, *, n_bins: int = 16) -> np.ndarray:
    """Uniform binning per dimension → joint histogram (flattened)."""
    if len(x) == 0:
        return np.array([])
    z, _, _ = _standardize(x)
    # Per-dim quantile bins for robustness
    edges = []
    for j in range(z.shape[1]):
        qs = np.linspace(0, 1, n_bins + 1)
        edges.append(np.quantile(z[:, j], qs))
    # Single composite index via digitize
    idx = np.zeros(len(z), dtype=np.int64)
    mult = 1
    for j in range(z.shape[1]):
        # avoid duplicate edges
        e = edges[j]
        e = np.maximum.accumulate(e)
        e[-1] += 1e-9
        idx += mult * np.clip(np.digitize(z[:, j], e[1:-1]), 0, n_bins - 1)
        mult *= n_bins
    counts = Counter(idx.tolist())
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    return probs


def pi_loaded_by_u_frames(meta: list[dict]) -> dict[str, float]:
    by_u: dict[float, list[str]] = defaultdict(list)
    for m in meta:
        u = round(float(m["u"]), 4)
        by_u[u].append(str(m["basin"]))
    return {str(u): sum(b == "loaded" for b in labs) / len(labs) for u, labs in sorted(by_u.items())}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def transition_counts(episodes: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ep in episodes:
        frames = ep.get("frames") or []
        for i in range(len(frames) - 1):
            a = frames[i].get("basin", "?")
            b = frames[i + 1].get("basin", "?")
            counts[f"{a}->{b}"] += 1
    return dict(counts)


@dataclass
class FeatureMeasureReport:
    """μ_ℱ: histogram occupation on raw ℱ chart (NOT organization measure)."""

    source_path: str
    n_episodes: int
    n_frames: int
    fx_threshold: float | None = None
    renyi2_entropy: float = float("nan")
    renyi2_uniform_null: float = float("nan")
    entropy_ratio: float = float("nan")
    participation_ratio: float = float("nan")
    mean_frac_loaded: float = float("nan")
    pi_by_u: dict[str, float] = field(default_factory=dict)
    pi_u_spearman: float = float("nan")
    transition_counts: dict[str, int] = field(default_factory=dict)
    n_transition_edges: int = 0
    has_loaded_to_grazing: bool = False
    concentration_verdict: str = "unknown"
    measure_kind: str = "feature_mu_F"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backward-compatible alias
MeasureReport = FeatureMeasureReport


@dataclass
class OrganizationMeasureReport:
    """μ_𝒪: occupation on coarse basin cells B_i after organization-preserving Π."""

    source_path: str
    n_episodes: int
    n_frames: int
    fx_threshold: float | None = None
    mu_by_basin: dict[str, float] = field(default_factory=dict)
    renyi2_basin: float = float("nan")
    renyi2_basin_uniform_null: float = float("nan")
    basin_entropy_ratio: float = float("nan")
    renyi2_joint_u_basin: float = float("nan")
    renyi2_joint_uniform_null: float = float("nan")
    joint_entropy_ratio: float = float("nan")
    pi_by_u: dict[str, float] = field(default_factory=dict)
    pi_u_spearman: float = float("nan")
    transition_counts: dict[str, int] = field(default_factory=dict)
    n_transition_edges: int = 0
    has_loaded_to_grazing: bool = False
    concentration_verdict: str = "unknown"
    measure_kind: str = "organization_mu_O"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _basin_frame_meta(episodes: list[dict]) -> list[dict]:
    meta: list[dict] = []
    for ep in episodes:
        u = float(ep.get("push_eccentricity_m", 0.0))
        eid = int(ep.get("episode_id", -1))
        for fr in ep.get("frames") or []:
            meta.append(
                {
                    "episode_id": eid,
                    "u": u,
                    "basin": str(fr.get("basin", "")),
                    "step": int(fr.get("step", 0)),
                }
            )
    return meta


def _prob_vector(counts: Counter[str]) -> np.ndarray:
    total = sum(counts.values())
    if total == 0:
        return np.array([])
    return np.array([counts[k] / total for k in sorted(counts)], dtype=np.float64)


def estimate_organization_measure(
    episodes: list[dict],
    *,
    source_path: str = "",
    fx_threshold: float | None = None,
) -> OrganizationMeasureReport:
    """μ_𝒪(B_i) = T(B_i)/T on MODE basin labels (coarse-graining Π applied first)."""
    meta = _basin_frame_meta(episodes)
    n_frames = len(meta)
    n_eps = len(episodes)
    if n_frames == 0:
        return OrganizationMeasureReport(source_path=source_path, n_episodes=n_eps, n_frames=0)

    basin_counts: Counter[str] = Counter(m["basin"] for m in meta)
    joint_counts: Counter[str] = Counter(f"u={round(float(m['u']), 4)}|{m['basin']}" for m in meta)

    p_basin = _prob_vector(basin_counts)
    p_joint = _prob_vector(joint_counts)
    h_basin = renyi2_entropy(p_basin)
    h_joint = renyi2_entropy(p_joint)
    k_basins = max(len(basin_counts), 1)
    k_joint = max(len(joint_counts), 1)
    h_basin_null = renyi2_entropy(np.full(k_basins, 1.0 / k_basins))
    h_joint_null = renyi2_entropy(np.full(k_joint, 1.0 / k_joint))

    pi = pi_loaded_by_u_frames(meta)
    us = np.array([float(k) for k in sorted(pi.keys())])
    pis = np.array([pi[str(u)] for u in us])
    pi_sp = _spearman(us, pis)

    trans = transition_counts(episodes)
    n_edges = sum(1 for v in trans.values() if v > 0)
    has_l2g = trans.get("loaded->grazing", 0) > 0

    basin_ratio = h_basin / h_basin_null if h_basin_null > 0 else float("nan")
    joint_ratio = h_joint / h_joint_null if h_joint_null > 0 else float("nan")

    # Organization concentration: joint (u, basin) departs from uniform AND topology nontrivial
    concentrated = (
        math.isfinite(joint_ratio)
        and joint_ratio < 0.85
        and (has_l2g or n_edges >= 3)
        and abs(pi_sp) > 0.3
    )
    partial = math.isfinite(joint_ratio) and joint_ratio < 0.95 and n_edges >= 2
    if concentrated:
        verdict = "concentrated"
    elif partial:
        verdict = "partial"
    else:
        verdict = "weak_or_uniform"

    return OrganizationMeasureReport(
        source_path=source_path,
        n_episodes=n_eps,
        n_frames=n_frames,
        fx_threshold=fx_threshold,
        mu_by_basin={k: v / n_frames for k, v in sorted(basin_counts.items())},
        renyi2_basin=h_basin,
        renyi2_basin_uniform_null=h_basin_null,
        basin_entropy_ratio=basin_ratio,
        renyi2_joint_u_basin=h_joint,
        renyi2_joint_uniform_null=h_joint_null,
        joint_entropy_ratio=joint_ratio,
        pi_by_u=pi,
        pi_u_spearman=pi_sp,
        transition_counts=trans,
        n_transition_edges=n_edges,
        has_loaded_to_grazing=has_l2g,
        concentration_verdict=verdict,
    )


def estimate_feature_measure(
    episodes: list[dict],
    *,
    source_path: str = "",
    fx_threshold: float | None = None,
    n_bins: int = 16,
) -> FeatureMeasureReport:
    """μ_ℱ: feature histogram occupation (falsified as organization object — see Stage 0)."""
    x, _, meta = _frame_feature_matrix(episodes)
    n_frames = len(meta)
    n_eps = len(episodes)

    if n_frames == 0:
        return FeatureMeasureReport(source_path=source_path, n_episodes=n_eps, n_frames=0)

    probs = _histogram_probs(x, n_bins=n_bins)
    h2 = renyi2_entropy(probs)
    # Uniform over occupied bins vs full grid upper bound
    n_occupied = max(len(probs), 1)
    grid_size = n_bins ** min(4, x.shape[1])  # cap joint grid for null
    h2_uniform_occ = renyi2_entropy(np.full(n_occupied, 1.0 / n_occupied))
    h2_uniform_full = renyi2_entropy(np.full(grid_size, 1.0 / grid_size))
    h2_null = min(h2_uniform_occ, h2_uniform_full)

    pi = pi_loaded_by_u_frames(meta)
    us = np.array([float(k) for k in sorted(pi.keys())])
    pis = np.array([pi[str(u)] for u in us])
    pi_sp = _spearman(us, pis)

    trans = transition_counts(episodes)
    n_edges = sum(1 for v in trans.values() if v > 0)
    has_l2g = trans.get("loaded->grazing", 0) > 0

    frac_loaded = sum(m["basin"] == "loaded" for m in meta) / n_frames

    # Concentration heuristic (protocol-scoped, not a theorem)
    concentrated = (
        math.isfinite(h2)
        and math.isfinite(h2_null)
        and h2 < 0.85 * h2_null
        and (has_l2g or n_edges >= 3)
        and frac_loaded > 0.05
        and frac_loaded < 0.95
    )
    partial = (
        math.isfinite(h2)
        and h2 < h2_null
        and n_edges >= 2
    )
    if concentrated:
        verdict = "concentrated"
    elif partial:
        verdict = "partial"
    else:
        verdict = "weak_or_uniform"

    return FeatureMeasureReport(
        source_path=source_path,
        n_episodes=n_eps,
        n_frames=n_frames,
        fx_threshold=fx_threshold,
        renyi2_entropy=h2,
        renyi2_uniform_null=h2_null,
        entropy_ratio=h2 / h2_null if h2_null > 0 else float("nan"),
        participation_ratio=participation_ratio(x),
        mean_frac_loaded=frac_loaded,
        pi_by_u=pi,
        pi_u_spearman=pi_sp,
        transition_counts=trans,
        n_transition_edges=n_edges,
        has_loaded_to_grazing=has_l2g,
        concentration_verdict=verdict,
    )


def estimate_occupation_measure(
    episodes: list[dict],
    *,
    source_path: str = "",
    fx_threshold: float | None = None,
    n_bins: int = 16,
) -> FeatureMeasureReport:
    """Deprecated name — returns μ_ℱ (feature measure). Prefer estimate_feature_measure."""
    return estimate_feature_measure(
        episodes,
        source_path=source_path,
        fx_threshold=fx_threshold,
        n_bins=n_bins,
    )


def estimate_measures(
    episodes: list[dict],
    *,
    source_path: str = "",
    fx_threshold: float | None = None,
    n_bins: int = 16,
) -> dict[str, Any]:
    """Both μ_ℱ and μ_𝒪 for a trajectory cohort."""
    mu_f = estimate_feature_measure(
        episodes, source_path=source_path, fx_threshold=fx_threshold, n_bins=n_bins
    )
    mu_o = estimate_organization_measure(
        episodes, source_path=source_path, fx_threshold=fx_threshold
    )
    return {
        "feature_mu_F": mu_f.to_dict(),
        "organization_mu_O": mu_o.to_dict(),
        "m1_on_feature_falsified": (
            mu_f.concentration_verdict in ("weak_or_uniform", "partial")
            and mu_f.entropy_ratio > 0.95
            and (
                abs(mu_o.pi_u_spearman) > 0.3
                or mu_o.n_transition_edges >= 2
            )
        ),
        "interpretation": (
            "If m1_on_feature_falsified is true, the baseline falsifies "
            "Organization = feature occupation measure μ_ℱ. "
            "Organization requires coarse-graining Π before defining μ_𝒪."
        ),
    }
