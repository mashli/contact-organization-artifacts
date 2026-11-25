"""Transfer operator / MSM analysis for natural coarse-graining Π (ToCO Project B1)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cge.emergence.measure import CONTINUOUS, _frame_feature_matrix, _standardize


def _episode_micro_sequences(
    episodes: list[dict], micro_labels: np.ndarray, meta: list[dict]
) -> list[list[int]]:
    """Per-episode microstate id sequences (aligned with meta row order)."""
    by_ep: dict[int, list[int]] = {}
    for m, mic in zip(meta, micro_labels):
        by_ep.setdefault(int(m["episode_id"]), []).append(int(mic))
    return [by_ep.get(int(ep.get("episode_id", -1)), []) for ep in episodes]


def build_count_matrix(sequences: list[list[int]], n_states: int) -> np.ndarray:
    c = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            if 0 <= a < n_states and 0 <= b < n_states:
                c[a, b] += 1.0
    return c


def row_stochastic(counts: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = counts.copy()
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return p / row_sums


def spectral_analysis(p: np.ndarray) -> dict[str, Any]:
    """Eigenvalues of row-stochastic P (sorted by magnitude)."""
    if p.size == 0:
        return {"eigenvalues": [], "spectral_gap": float("nan"), "tau2": float("nan")}
    vals = np.linalg.eigvals(p)
    mags = np.sort(np.abs(vals))[::-1]
    lam1 = float(mags[0]) if len(mags) else float("nan")
    lam2 = float(mags[1]) if len(mags) > 1 else float("nan")
    gap = lam1 - lam2 if math.isfinite(lam1) and math.isfinite(lam2) else float("nan")
    tau2 = float(-1.0 / math.log(lam2)) if 0 < lam2 < 1 else float("nan")
    return {
        "eigenvalues": [float(v) for v in mags[: min(8, len(mags))]],
        "lambda2": lam2,
        "spectral_gap": gap,
        "tau2_implied": tau2,
    }


def pcca_bipartition(p: np.ndarray, n_macro: int = 2) -> np.ndarray:
    """Assign each microstate to macro cluster via dominant slow eigenvectors (PCCA-lite)."""
    n = p.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n <= n_macro:
        return np.arange(n, dtype=int)

    # Right eigenvectors of P (columns); slow modes for row-stochastic P
    vals, vecs = np.linalg.eig(p.T)
    order = np.argsort(-np.abs(vals))
    # Use eigenvectors 1..n_macro-1 (skip stationary mode ~1)
    slow = np.real(vecs[:, order[1:n_macro]])
    if slow.ndim == 1:
        slow = slow.reshape(-1, 1)
    # k-means on slow coordinates
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_macro, n_init=10, random_state=0)
    return km.fit_predict(slow)


def adjusted_rand_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score as sklearn_ari

    if len(y_true) == 0:
        return float("nan")
    return float(sklearn_ari(y_true, y_pred))


@dataclass
class TransferOperatorReport:
    source_path: str
    n_episodes: int
    n_frames: int
    n_microstates: int
    # MODE reference (2-state)
    mode_states: list[str] = field(default_factory=list)
    mode_transition_counts: dict[str, int] = field(default_factory=dict)
    mode_P_spectral: dict[str, Any] = field(default_factory=dict)
    # MSM from ℱ microstates
    micro_P_spectral: dict[str, Any] = field(default_factory=dict)
    msm_ari_vs_mode: float = float("nan")
    msm_nmi_vs_mode: float = float("nan")
    fx_msm_ari_vs_mode: float = float("nan")
    spectral_gap: float = float("nan")
    spectral_gap_null_mean: float = float("nan")
    spectral_gap_null_std: float = float("nan")
    natural_pi_verdict: str = "unknown"
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shuffle_basin_null(episodes: list[dict], n_shuffles: int = 50) -> list[float]:
    """Null: permute basin labels within each episode; return spectral gaps."""
    rng = np.random.default_rng(0)
    gaps: list[float] = []
    for _ in range(n_shuffles):
        shuf_eps = []
        for ep in episodes:
            frames = [dict(fr) for fr in (ep.get("frames") or [])]
            basins = [str(fr.get("basin", "")) for fr in frames]
            perm = basins.copy()
            rng.shuffle(perm)
            for fr, b in zip(frames, perm):
                fr["basin"] = b
            shuf_eps.append({**ep, "frames": frames})
        states = sorted({str(fr.get("basin", "")) for ep in shuf_eps for fr in ep.get("frames") or []})
        idx = {s: i for i, s in enumerate(states)}
        seqs = []
        for ep in shuf_eps:
            seq = [idx[str(fr.get("basin", ""))] for fr in ep.get("frames") or []]
            if len(seq) >= 2:
                seqs.append(seq)
        if not seqs:
            continue
        c = build_count_matrix(seqs, len(states))
        p = row_stochastic(c)
        gaps.append(spectral_analysis(p)["spectral_gap"])
    return gaps


def _fx_microstates(episodes: list[dict], n_bins: int = 10) -> tuple[np.ndarray, list[dict]]:
    """Discretize |f_x| as microstates (constraint-relevant coordinate)."""
    meta: list[dict] = []
    fx_vals: list[float] = []
    for ep in episodes:
        for fr in ep.get("frames") or []:
            fx_vals.append(abs(float(fr.get("fx", 0.0))))
            meta.append(
                {
                    "episode_id": int(ep.get("episode_id", -1)),
                    "basin": str(fr.get("basin", "")),
                }
            )
    if not fx_vals:
        return np.array([]), []
    edges = np.quantile(fx_vals, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    edges[-1] += 1e-9
    micro = np.clip(np.digitize(fx_vals, edges[1:-1]), 0, n_bins - 1)
    return micro.astype(int), meta


def estimate_transfer_operator(
    episodes: list[dict],
    *,
    source_path: str = "",
    n_micro: int = 20,
    n_shuffles: int = 50,
) -> TransferOperatorReport:
    """B1: transfer operator on MODE basins + MSM natural Π vs MODE."""
    n_frames = sum(len(ep.get("frames") or []) for ep in episodes)
    if n_frames == 0:
        return TransferOperatorReport(source_path=source_path, n_episodes=len(episodes), n_frames=0, n_microstates=0)

    # --- MODE coarse transfer operator (cross-basin only for counts) ---
    mode_states = sorted({str(fr.get("basin", "")) for ep in episodes for fr in ep.get("frames") or []})
    mode_idx = {s: i for i, s in enumerate(mode_states)}
    mode_seqs: list[list[int]] = []
    mode_cross: dict[str, int] = {}
    for ep in episodes:
        frames = ep.get("frames") or []
        labels = [str(fr.get("basin", "")) for fr in frames]
        mode_seqs.append([mode_idx[b] for b in labels])
        for a, b in zip(labels, labels[1:]):
            if a != b:
                mode_cross[f"{a}->{b}"] = mode_cross.get(f"{a}->{b}", 0) + 1

    mode_counts = build_count_matrix(mode_seqs, len(mode_states))
    mode_p = row_stochastic(mode_counts)
    mode_spec = spectral_analysis(mode_p)

    # --- Microstate MSM from ℱ ---
    x, _, meta = _frame_feature_matrix(episodes)
    z, _, _ = _standardize(x)
    n_comp = min(4, z.shape[1], max(2, z.shape[0] // 10))
    if z.shape[0] >= n_micro + 2:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_comp, random_state=0)
        z_pca = pca.fit_transform(z)
        km = KMeans(n_clusters=n_micro, n_init=10, random_state=0)
        micro = km.fit_predict(z_pca)
    else:
        micro = np.zeros(len(z), dtype=int)
        n_micro = 1

    micro_seqs = _episode_micro_sequences(episodes, micro, meta)
    micro_seqs = [s for s in micro_seqs if len(s) >= 2]
    micro_counts = build_count_matrix(micro_seqs, n_micro)
    micro_p = row_stochastic(micro_counts)
    micro_spec = spectral_analysis(micro_p)

    # PCCA: micro -> macro (2 states), compare to MODE at frame level
    macro_of_micro = pcca_bipartition(micro_p, n_macro=2)
    mode_true = np.array([mode_idx[str(m["basin"])] for m in meta])
    # Map macro 0/1 to best alignment with mode (loaded/grazing)
    macro_frame = macro_of_micro[micro]
    # Relabel macro clusters to maximize ARI (2! permutations)
    best_ari = -1.0
    best_pred = macro_frame
    for flip in (False, True):
        pred = (1 - macro_frame) if flip else macro_frame
        ari = adjusted_rand_score(mode_true, pred)
        if ari > best_ari:
            best_ari = ari
            best_pred = pred

    from sklearn.metrics import normalized_mutual_info_score

    nmi = float(normalized_mutual_info_score(mode_true, best_pred)) if len(mode_true) else float("nan")

    # fx-binned MSM (constraint coordinate — candidate natural Π)
    fx_micro, fx_meta = _fx_microstates(episodes, n_bins=min(10, n_micro))
    fx_ari = float("nan")
    if len(fx_micro) > 0 and len(mode_states) >= 2:
        fx_idx = {s: i for i, s in enumerate(mode_states)}
        fx_true = np.array([fx_idx[str(m["basin"])] for m in fx_meta])
        fx_seqs = []
        by_ep: dict[int, list[int]] = {}
        for m, mic in zip(fx_meta, fx_micro):
            by_ep.setdefault(int(m["episode_id"]), []).append(int(mic))
        fx_seqs = [s for s in by_ep.values() if len(s) >= 2]
        n_fx = int(fx_micro.max()) + 1 if len(fx_micro) else 1
        fx_counts = build_count_matrix(fx_seqs, max(n_fx, 1))
        fx_p = row_stochastic(fx_counts)
        fx_macro = pcca_bipartition(fx_p, n_macro=2)
        fx_frame = fx_macro[fx_micro]
        best_fx_ari = -1.0
        for flip in (False, True):
            pred = (1 - fx_frame) if flip else fx_frame
            ari = adjusted_rand_score(fx_true, pred)
            if ari > best_fx_ari:
                best_fx_ari = ari
        fx_ari = best_fx_ari

    # Null spectral gap on shuffled MODE labels
    null_gaps = _shuffle_basin_null(episodes, n_shuffles=n_shuffles)
    null_mean = float(np.mean(null_gaps)) if null_gaps else float("nan")
    null_std = float(np.std(null_gaps)) if null_gaps else float("nan")
    gap = float(mode_spec["spectral_gap"])  # MODE-level gap (primary)
    micro_gap = float(micro_spec["spectral_gap"])

    if fx_ari >= 0.90 and math.isfinite(gap) and gap > 0.05:
        verdict = "natural_pi_supported"
    elif fx_ari >= 0.30 and math.isfinite(gap) and math.isfinite(null_mean) and gap > null_mean:
        verdict = "mode_spectrum_consistent"
    elif math.isfinite(gap) and gap > 0.10:
        verdict = "metastable_mode_P"
    else:
        verdict = "inconclusive"

    return TransferOperatorReport(
        source_path=source_path,
        n_episodes=len(episodes),
        n_frames=n_frames,
        n_microstates=n_micro,
        mode_states=mode_states,
        mode_transition_counts=mode_cross,
        mode_P_spectral=mode_spec,
        micro_P_spectral={**micro_spec, "feature_msm_gap": micro_gap},
        msm_ari_vs_mode=best_ari,
        msm_nmi_vs_mode=nmi,
        fx_msm_ari_vs_mode=fx_ari,
        spectral_gap=gap,
        spectral_gap_null_mean=null_mean,
        spectral_gap_null_std=null_std,
        natural_pi_verdict=verdict,
        interpretation=(
            "MSM slow modes on ℱ microstates compared to MODE loaded/grazing. "
            "High ARI + spectral gap above shuffle null ⇒ Π consistent with dynamics spectrum."
        ),
    )
