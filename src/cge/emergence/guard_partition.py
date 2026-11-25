"""Guard-induced partitions Π and transfer-operator comparison (ToCO B1)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from cge.basin_labeling import basin_label_from_fx
from cge.emergence.transfer_operator import (
    build_count_matrix,
    row_stochastic,
    spectral_analysis,
    adjusted_rand_score,
)


FrameLabelFn = Callable[[dict[str, Any], float], str]


def _feat(frame: dict, key: str, default: float = 0.0) -> float:
    f = frame.get("features") or {}
    return float(f.get(key, default))


def guard_g_fx_mode(frame: dict, fx_threshold: float) -> str:
    """G1: complementarity / load guard — MODE rule on |f_x|."""
    return basin_label_from_fx(float(frame.get("fx", 0.0)), threshold=fx_threshold)


def guard_g_contact(frame: dict, fx_threshold: float) -> str:
    """G2: contact persistence — patch multiplicity."""
    m = _feat(frame, "multiplicity_M")
    if m < 0.5:
        return "no_contact"
    return basin_label_from_fx(float(frame.get("fx", 0.0)), threshold=fx_threshold)


def guard_g_multiplicity(frame: dict, _fx_threshold: float) -> str:
    """G3: constraint-rank proxy — single vs multi-point patch."""
    m = _feat(frame, "multiplicity_M")
    if m < 0.5:
        return "no_contact"
    if m < 1.5:
        return "single_patch"
    return "multi_patch"


def guard_g_coupling(frame: dict, fx_threshold: float) -> str:
    """G4: lateral coupling guard — |f_y| vs |f_x| (friction cone proxy)."""
    fx = abs(float(frame.get("fx", 0.0)))
    fy = abs(_feat(frame, "resultant_force_f_y"))
    if _feat(frame, "multiplicity_M") < 0.5:
        return "no_contact"
    if fx < 1e-6:
        return "lateral_dominant"
    ratio = fy / fx
    base = basin_label_from_fx(float(frame.get("fx", 0.0)), threshold=fx_threshold)
    return f"{base}_lat" if ratio > 0.15 else f"{base}_ax"


def guard_g_spread(frame: dict, fx_threshold: float) -> str:
    """G5: patch geometry guard — spread low vs high within contact."""
    if _feat(frame, "multiplicity_M") < 0.5:
        return "no_contact"
    spread = _feat(frame, "patch_spread_s_xy")
    base = basin_label_from_fx(float(frame.get("fx", 0.0)), threshold=fx_threshold)
    return f"{base}_compact" if spread < 0.002 else f"{base}_dispersed"


def guard_g_composite(frame: dict, fx_threshold: float) -> str:
    """G6: composite guard — contact AND load (MODE ∩ persistence)."""
    if _feat(frame, "multiplicity_M") < 1.0:
        return "grazing"
    return basin_label_from_fx(float(frame.get("fx", 0.0)), threshold=fx_threshold)


GUARD_CATALOG: dict[str, tuple[str, FrameLabelFn]] = {
    "G_fx_mode": ("Load guard f_x ≥ t* (MODE)", guard_g_fx_mode),
    "G_contact": ("Contact + load guard", guard_g_contact),
    "G_multiplicity": ("Multiplicity rank proxy", guard_g_multiplicity),
    "G_coupling": ("Lateral/axial coupling guard", guard_g_coupling),
    "G_spread": ("Patch spread guard", guard_g_spread),
    "G_composite": ("M≥1 ∧ f_x guard (composite)", guard_g_composite),
}


def label_episodes(
    episodes: list[dict],
    label_fn: FrameLabelFn,
    *,
    fx_threshold: float,
) -> tuple[list[list[str]], list[str]]:
    """Per-episode label sequences + flat MODE reference labels."""
    seqs: list[list[str]] = []
    mode_flat: list[str] = []
    for ep in episodes:
        labels = [label_fn(fr, fx_threshold) for fr in ep.get("frames") or []]
        if len(labels) >= 2:
            seqs.append(labels)
        for fr in ep.get("frames") or []:
            mode_flat.append(guard_g_fx_mode(fr, fx_threshold))
    return seqs, mode_flat


def labels_to_indices(seqs: list[list[str]]) -> tuple[list[list[int]], list[str]]:
    states = sorted({s for seq in seqs for s in seq})
    idx = {s: i for i, s in enumerate(states)}
    return [[idx[s] for s in seq] for seq in seqs], states


def transfer_spectrum_from_sequences(seqs: list[list[str]]) -> dict[str, Any]:
    if not seqs:
        return spectral_analysis(np.zeros((0, 0)))
    idx_seqs, states = labels_to_indices(seqs)
    c = build_count_matrix(idx_seqs, len(states))
    p = row_stochastic(c)
    spec = spectral_analysis(p)
    spec["n_states"] = len(states)
    spec["state_names"] = states
    return spec


def cross_regime_switches(seqs: list[list[str]]) -> int:
    n = 0
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            if a != b:
                n += 1
    return n


@dataclass
class GuardPartitionResult:
    guard_id: str
    description: str
    n_states: int
    state_names: list[str]
    ari_vs_mode: float = float("nan")
    lambda2: float = float("nan")
    spectral_gap: float = float("nan")
    tau2_implied: float = float("nan")
    n_regime_switches: int = 0
    delta_lambda2_vs_mode: float = float("nan")
    delta_tau2_vs_mode: float = float("nan")
    spectrum_match_score: float = float("nan")  # higher = closer to MODE
    verdict: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardPartitionReport:
    source_path: str
    n_episodes: int
    n_frames: int
    fx_threshold: float
    mode_spectrum: dict[str, Any] = field(default_factory=dict)
    guards: list[GuardPartitionResult] = field(default_factory=list)
    best_guard_id: str = ""
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "n_episodes": self.n_episodes,
            "n_frames": self.n_frames,
            "fx_threshold": self.fx_threshold,
            "mode_spectrum": self.mode_spectrum,
            "guards": [g.to_dict() for g in self.guards],
            "best_guard_id": self.best_guard_id,
            "interpretation": self.interpretation,
        }


def estimate_guard_partitions(
    episodes: list[dict],
    *,
    source_path: str = "",
    fx_threshold: float = 8.0,
) -> GuardPartitionReport:
    """Compare guard-induced Π transfer spectra to MODE reference."""
    n_frames = sum(len(ep.get("frames") or []) for ep in episodes)
    if n_frames == 0:
        return GuardPartitionReport(source_path=source_path, n_episodes=0, n_frames=0, fx_threshold=fx_threshold)

    # MODE reference sequences
    mode_seqs, mode_flat = label_episodes(episodes, guard_g_fx_mode, fx_threshold=fx_threshold)
    mode_spec = transfer_spectrum_from_sequences(mode_seqs)
    mode_l2 = float(mode_spec.get("lambda2", float("nan")))
    mode_t2 = float(mode_spec.get("tau2_implied", float("nan")))

    results: list[GuardPartitionResult] = []
    for gid, (desc, fn) in GUARD_CATALOG.items():
        seqs, flat = label_episodes(episodes, fn, fx_threshold=fx_threshold)
        spec = transfer_spectrum_from_sequences(seqs)
        guard_flat = [fn(fr, fx_threshold) for ep in episodes for fr in ep.get("frames") or []]
        # ARI: only compare frames where guard maps to binary loaded/grazing-like
        # For guards that refine MODE (more states), compare coarse collapse
        coarse_guard = [guard_g_fx_mode(fr, fx_threshold) for ep in episodes for fr in ep.get("frames") or []]
        if gid in ("G_fx_mode", "G_composite", "G_contact"):
            ari = adjusted_rand_score(
                np.array(mode_flat), np.array(guard_flat[: len(mode_flat)])
            )
        else:
            ari = adjusted_rand_score(np.array(mode_flat), np.array(coarse_guard))

        l2 = float(spec.get("lambda2", float("nan")))
        t2 = float(spec.get("tau2_implied", float("nan")))
        dl2 = abs(l2 - mode_l2) if math.isfinite(l2) and math.isfinite(mode_l2) else float("nan")
        dt2 = abs(t2 - mode_t2) if math.isfinite(t2) and math.isfinite(mode_t2) else float("nan")
        match = 1.0 / (1.0 + dl2 + 0.1 * dt2) if math.isfinite(dl2) else 0.0

        if gid == "G_fx_mode":
            verdict = "reference"
        elif ari >= 0.99 and match >= 0.8:
            verdict = "equivalent_to_mode"
        elif match >= 0.7 and spec.get("n_states", 0) == 2:
            verdict = "spectrum_matched"
        elif ari >= 0.5:
            verdict = "partial_alignment"
        else:
            verdict = "distinct_partition"

        results.append(
            GuardPartitionResult(
                guard_id=gid,
                description=desc,
                n_states=int(spec.get("n_states", 0)),
                state_names=list(spec.get("state_names", [])),
                ari_vs_mode=ari,
                lambda2=l2,
                spectral_gap=float(spec.get("spectral_gap", float("nan"))),
                tau2_implied=t2,
                n_regime_switches=cross_regime_switches(seqs),
                delta_lambda2_vs_mode=dl2,
                delta_tau2_vs_mode=dt2,
                spectrum_match_score=match,
                verdict=verdict,
            )
        )

    best = max(results, key=lambda r: r.spectrum_match_score if r.guard_id != "G_fx_mode" else -1)

    return GuardPartitionReport(
        source_path=source_path,
        n_episodes=len(episodes),
        n_frames=n_frames,
        fx_threshold=fx_threshold,
        mode_spectrum=mode_spec,
        guards=results,
        best_guard_id=best.guard_id,
        interpretation=(
            "Guard-induced Π: compare transfer operator spectrum of constraint guards vs MODE. "
            "G_fx_mode is the load/complementarity guard; close λ₂ and τ₂ ⇒ Π derivable from guard."
        ),
    )
