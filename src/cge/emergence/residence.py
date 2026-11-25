"""Residence time distributions on metastable basins (ToCO — metastability probe)."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cge.emergence.measure import load_trajectory_episodes, transition_counts


def _survival_tail(lengths: list[int], thresholds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)) -> dict[str, float]:
    """Empirical survival P(τ > t) for residence runs."""
    if not lengths:
        return {str(t): float("nan") for t in thresholds}
    n = len(lengths)
    return {str(t): sum(l > t for l in lengths) / n for t in thresholds}


def _exp_decay_rate(lengths: list[int]) -> float:
    """MLE λ for exponential RTD: P(τ>t) ≈ exp(-λt). Returns nan if empty."""
    if not lengths:
        return float("nan")
    mean_t = float(np.mean(lengths))
    return 1.0 / mean_t if mean_t > 0 else float("nan")


@dataclass
class BasinResidenceStats:
    basin: str
    n_runs: int = 0
    n_frames: int = 0
    mean_run_length: float = float("nan")
    median_run_length: float = float("nan")
    max_run_length: int = 0
    std_run_length: float = float("nan")
    p90_run_length: float = float("nan")
    p95_run_length: float = float("nan")
    exp_decay_rate: float = float("nan")  # λ: slower decay ⇒ longer tail
    survival_tail: dict[str, float] = field(default_factory=dict)  # P(τ>t)
    run_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if len(d["run_lengths"]) > 20:
            d["run_lengths"] = d["run_lengths"][:20]
            d["run_lengths_truncated"] = True
        return d


@dataclass
class ResidenceReport:
    source_path: str
    n_episodes: int
    n_frames: int
    n_regime_switches: int  # cross-basin only
    transition_counts: dict[str, int] = field(default_factory=dict)  # cross-basin
    within_basin_steps: dict[str, int] = field(default_factory=dict)
    by_basin: dict[str, BasinResidenceStats] = field(default_factory=dict)
    metastability_ratio: float = float("nan")  # mean(loaded) / mean(grazing)
    tail_separation_at_t4: float = float("nan")  # P_loaded(τ>4) / P_grazing(τ>4)
    timescale_separation_verdict: str = "unknown"
    metastability_verdict: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "n_episodes": self.n_episodes,
            "n_frames": self.n_frames,
            "n_regime_switches": self.n_regime_switches,
            "transition_counts": self.transition_counts,
            "within_basin_steps": self.within_basin_steps,
            "by_basin": {k: v.to_dict() for k, v in self.by_basin.items()},
            "metastability_ratio": self.metastability_ratio,
            "tail_separation_at_t4": self.tail_separation_at_t4,
            "timescale_separation_verdict": self.timescale_separation_verdict,
            "metastability_verdict": self.metastability_verdict,
        }


def _basin_sequences(episodes: list[dict]) -> list[tuple[str, int, float, int]]:
    """Yield (basin, episode_id, u, run_length) for each contiguous run."""
    runs: list[tuple[str, int, float, int]] = []
    for ep in episodes:
        frames = ep.get("frames") or []
        if not frames:
            continue
        eid = int(ep.get("episode_id", -1))
        u = float(ep.get("push_eccentricity_m", 0.0))
        basins = [str(fr.get("basin", "")) for fr in frames]
        start = 0
        for i in range(1, len(basins) + 1):
            if i == len(basins) or basins[i] != basins[start]:
                runs.append((basins[start], eid, u, i - start))
                start = i
    return runs


def cross_basin_transition_counts(episodes: list[dict]) -> dict[str, int]:
    """Count regime switches only (a→b where a≠b)."""
    counts: Counter[str] = Counter()
    for ep in episodes:
        frames = ep.get("frames") or []
        for i in range(len(frames) - 1):
            a = str(frames[i].get("basin", "?"))
            b = str(frames[i + 1].get("basin", "?"))
            if a != b:
                counts[f"{a}->{b}"] += 1
    return dict(counts)


def estimate_residence_times(
    episodes: list[dict],
    *,
    source_path: str = "",
) -> ResidenceReport:
    """Residence time distribution per metastable basin (RTD)."""
    runs = _basin_sequences(episodes)
    n_frames = sum(len(ep.get("frames") or []) for ep in episodes)
    trans_all = transition_counts(episodes)
    trans = cross_basin_transition_counts(episodes)
    n_trans = sum(trans.values())

    by_basin_lengths: dict[str, list[int]] = {}
    for basin, _eid, _u, length in runs:
        by_basin_lengths.setdefault(basin, []).append(length)

    by_basin: dict[str, BasinResidenceStats] = {}
    for basin, lengths in sorted(by_basin_lengths.items()):
        arr = np.asarray(lengths, dtype=np.float64)
        by_basin[basin] = BasinResidenceStats(
            basin=basin,
            n_runs=len(lengths),
            n_frames=int(arr.sum()),
            mean_run_length=float(arr.mean()) if len(arr) else float("nan"),
            median_run_length=float(np.median(arr)) if len(arr) else float("nan"),
            max_run_length=int(arr.max()) if len(arr) else 0,
            std_run_length=float(arr.std()) if len(arr) > 1 else 0.0,
            p90_run_length=float(np.percentile(arr, 90)) if len(arr) else float("nan"),
            p95_run_length=float(np.percentile(arr, 95)) if len(arr) else float("nan"),
            exp_decay_rate=_exp_decay_rate(lengths),
            survival_tail=_survival_tail(lengths),
            run_lengths=sorted(lengths, reverse=True),
        )

    loaded = by_basin.get("loaded", BasinResidenceStats("loaded"))
    grazing = by_basin.get("grazing", BasinResidenceStats("grazing"))
    loaded_mean = loaded.mean_run_length
    grazing_mean = grazing.mean_run_length
    ratio = (
        loaded_mean / grazing_mean
        if math.isfinite(loaded_mean) and math.isfinite(grazing_mean) and grazing_mean > 0
        else float("nan")
    )

    p_loaded_t4 = loaded.survival_tail.get("4", float("nan"))
    p_grazing_t4 = grazing.survival_tail.get("4", float("nan"))
    tail_sep = (
        p_loaded_t4 / p_grazing_t4
        if math.isfinite(p_loaded_t4) and math.isfinite(p_grazing_t4) and p_grazing_t4 > 0
        else float("nan")
    )

    if math.isfinite(tail_sep) and tail_sep >= 2.0 and math.isfinite(ratio) and ratio >= 1.5:
        ts_verdict = "timescale_separated"
    elif loaded.n_runs == 0:
        ts_verdict = "no_loaded_regime"
    elif math.isfinite(ratio) and ratio >= 1.2:
        ts_verdict = "weak_separation"
    else:
        ts_verdict = "no_clear_separation"

    within: dict[str, int] = {}
    for k, v in transition_counts(episodes).items():
        a, _, b = k.partition("->")
        if a == b:
            within[k] = v

    if math.isfinite(ratio) and ratio >= 1.5 and n_trans >= 1:
        verdict = "metastable"
    elif by_basin.get("loaded", BasinResidenceStats("loaded")).n_runs == 0:
        verdict = "no_loaded_regime"
    elif n_trans == 0 and n_frames > 0:
        verdict = "single_regime_or_frozen"
    elif math.isfinite(ratio) and ratio >= 1.2:
        verdict = "weak_metastability"
    else:
        verdict = "mixed_or_short_runs"

    return ResidenceReport(
        source_path=source_path,
        n_episodes=len(episodes),
        n_frames=n_frames,
        n_regime_switches=n_trans,
        transition_counts=trans,
        within_basin_steps=within,
        by_basin=by_basin,
        metastability_ratio=ratio,
        tail_separation_at_t4=tail_sep,
        timescale_separation_verdict=ts_verdict,
        metastability_verdict=verdict,
    )


def load_and_analyze(path: Path | str) -> ResidenceReport:
    meta, eps = load_trajectory_episodes(path)
    p = Path(path)
    rel = str(p)
    return estimate_residence_times(eps, source_path=rel)
