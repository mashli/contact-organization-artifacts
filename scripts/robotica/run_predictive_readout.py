#!/usr/bin/env python3
"""RAS Section 6 predictive readout.

Existing trajectories only. Does not modify frozen IJRR scripts or sources.
Implements docs/latex_ras/contact_organization_ras/configs/predictive_prereg.yaml.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from _paths import REPO, load_json, resolve_data  # noqa: E402
ROOT = REPO
RAS = REPO  # configs live under configs/robotica in the public repo
PREREG_PATH = REPO / "configs" / "robotica" / "predictive_prereg.yaml"
MAP_PATH = REPO / "configs" / "robotica" / "existing_trajectory_map.yaml"
SCORER_PATH = REPO / "configs" / "robotica" / "predictive_scorer.yaml"
OUT_DIR = REPO / "results" / "robotica"

FORCE_NAME_MARKERS = (
    "force",
    "fx",
    "fy",
    "fz",
    "tau",
    "torque",
    "push_force",
)
MODE_KEYS = (
    "multiplicity_M",
    "n_table_contacts",
    "in_table_contact",
    "in_contact_window",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def frame_fx(frame: dict) -> float:
    if frame.get("fx") is not None and is_finite_number(frame["fx"]):
        return float(frame["fx"])
    cs = frame.get("continuous_state")
    if isinstance(cs, dict):
        for key in ("tangential_force_norm", "push_force", "normal_force_sum"):
            if is_finite_number(cs.get(key)):
                return float(cs[key])
    feat = frame.get("features")
    if isinstance(feat, dict) and is_finite_number(feat.get("resultant_force_f_x")):
        return float(feat["resultant_force_f_x"])
    return float("nan")


def frame_scalars(frame: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    feat = frame.get("features")
    if isinstance(feat, dict):
        for k, v in feat.items():
            if is_finite_number(v):
                out[str(k)] = float(v)
    elif isinstance(feat, list):
        for i, v in enumerate(feat):
            if is_finite_number(v):
                out[f"feat_{i}"] = float(v)
    cs = frame.get("continuous_state")
    if isinstance(cs, dict):
        for k, v in cs.items():
            if is_finite_number(v):
                out[str(k)] = float(v)
    fx = frame_fx(frame)
    if math.isfinite(fx):
        out.setdefault("fx", fx)
    return out


def contact_mode(frame: dict) -> float | None:
    feat = frame.get("features")
    if isinstance(feat, dict):
        for k in MODE_KEYS:
            if is_finite_number(feat.get(k)):
                return float(feat[k])
    cs = frame.get("continuous_state")
    if isinstance(cs, dict):
        for k in MODE_KEYS:
            if is_finite_number(cs.get(k)):
                return float(cs[k])
    return None


def guard_cell(fx: float, threshold: float) -> str:
    return "loaded" if float(fx) >= threshold else "grazing"


def signed_distance(fx: float, cell: str, threshold: float) -> float:
    if cell == "loaded":
        return max(float(fx) - threshold, 0.0)
    return max(threshold - float(fx), 0.0)


def stored_basin(frame: dict) -> str:
    b = str(frame.get("basin") or "")
    if b == "dwell":
        return "grazing"
    return b


def episode_escape(ep: dict) -> int:
    basins = [stored_basin(fr) for fr in ep["frames"]]
    return int(any(basins[i] == "loaded" and basins[i + 1] == "grazing" for i in range(len(basins) - 1)))


def boundary_event(fx_t: float, basin_t: str, basin_next: str, threshold: float, margin: float) -> int:
    return int(basin_t != basin_next and abs(float(fx_t) - threshold) <= margin)


def qualification_status(role: str) -> str:
    if role.startswith("qualified"):
        return "qualified"
    if "boundary" in role:
        return "failure_or_boundary"
    return "failure"


def split_episode_indices(
    y: list[int],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=int)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError("split fractions leave no training episodes")

    def _stratified_take(pool: np.ndarray, n_take: int, rng_local: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if n_take >= len(pool):
            return pool, np.array([], dtype=int)
        labels = y_arr[pool]
        chosen: list[int] = []
        remain: list[int] = []
        for lab in sorted(set(labels.tolist())):
            members = pool[labels == lab]
            members = members.copy()
            rng_local.shuffle(members)
            k = int(round(n_take * (len(members) / len(pool))))
            k = min(max(k, 1 if n_take > 0 and len(members) else 0), len(members))
            chosen.extend(members[:k].tolist())
            remain.extend(members[k:].tolist())
        chosen_arr = np.array(chosen, dtype=int)
        remain_arr = np.array(remain, dtype=int)
        if len(chosen_arr) > n_take:
            extra = chosen_arr[n_take:]
            chosen_arr = chosen_arr[:n_take]
            remain_arr = np.concatenate([remain_arr, extra]) if len(remain_arr) else extra
        elif len(chosen_arr) < n_take and len(remain_arr):
            need = n_take - len(chosen_arr)
            chosen_arr = np.concatenate([chosen_arr, remain_arr[:need]])
            remain_arr = remain_arr[need:]
        return chosen_arr, remain_arr

    # If a class is too rare for stratification, fall back to a plain shuffle.
    counts = {int(v): int(np.sum(y_arr == v)) for v in np.unique(y_arr)}
    if min(counts.values()) < 3:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        test_idx = shuffled[:n_test]
        val_idx = shuffled[n_test : n_test + n_val]
        train_idx = shuffled[n_test + n_val :]
        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    test_idx, rest = _stratified_take(idx, n_test, rng)
    val_idx, train_idx = _stratified_take(rest, n_val, rng)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


class GaussianHMM2:
    """Unsupervised 2-state Gaussian HMM on a 1-D observation. No label use."""

    def __init__(self, n_iter: int = 15, seed: int = 0) -> None:
        self.n_iter = n_iter
        self.seed = seed
        self.start = np.array([0.5, 0.5])
        self.trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.mu = np.zeros(2)
        self.var = np.ones(2)

    def fit(self, sequences: list[np.ndarray]) -> "GaussianHMM2":
        seqs = [np.asarray(s, dtype=float).reshape(-1) for s in sequences if len(s)]
        pile = np.concatenate(seqs)
        km = KMeans(n_clusters=2, random_state=self.seed, n_init=10).fit(pile.reshape(-1, 1))
        self.mu = km.cluster_centers_.ravel().astype(float)
        self.var = np.array(
            [max(float(np.var(pile[km.labels_ == i])), 1.0) for i in range(2)],
            dtype=float,
        )
        self.start = np.array([0.5, 0.5], dtype=float)
        self.trans = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
        for _ in range(self.n_iter):
            gamma_sum = np.zeros(2)
            xi_sum = np.zeros((2, 2))
            start_sum = np.zeros(2)
            mu_num = np.zeros(2)
            var_num = np.zeros(2)
            for x in seqs:
                alpha, beta, scale, emit = self._forward_backward(x)
                T = len(x)
                gamma = alpha * beta
                gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
                start_sum += gamma[0]
                for t in range(T - 1):
                    xi = (
                        alpha[t][:, None]
                        * self.trans
                        * emit[t + 1][None, :]
                        * beta[t + 1][None, :]
                    )
                    xi = xi / max(float(xi.sum()), 1e-12)
                    xi_sum += xi
                gamma_sum += gamma.sum(axis=0)
                for i in range(2):
                    mu_num[i] += float(np.dot(gamma[:, i], x))
                    var_num[i] += float(np.dot(gamma[:, i], (x - self.mu[i]) ** 2))
            self.start = start_sum / max(float(start_sum.sum()), 1e-12)
            self.trans = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
            self.mu = mu_num / np.maximum(gamma_sum, 1e-12)
            self.var = np.maximum(var_num / np.maximum(gamma_sum, 1e-12), 0.5)
        return self

    def _emit(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2))
        for i in range(2):
            z = (x - self.mu[i]) / math.sqrt(self.var[i])
            out[:, i] = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi * self.var[i])
            out[:, i] = np.maximum(out[:, i], 1e-12)
        return out

    def _forward_backward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        emit = self._emit(x)
        T = len(x)
        alpha = np.zeros((T, 2))
        scale = np.zeros(T)
        alpha[0] = self.start * emit[0]
        scale[0] = max(float(alpha[0].sum()), 1e-12)
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.trans) * emit[t]
            scale[t] = max(float(alpha[t].sum()), 1e-12)
            alpha[t] /= scale[t]
        beta = np.zeros((T, 2))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.trans * emit[t + 1] * beta[t + 1]).sum(axis=1)
            beta[t] /= scale[t + 1]
        return alpha, beta, scale, emit

    def filtered_posterior(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if len(x) == 0:
            return np.zeros((0, 2))
        alpha, _, _, _ = self._forward_backward(x)
        return alpha


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        total += (np.sum(mask) / n) * abs(acc - conf)
    return float(total)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    try:
        return float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]))
    except ValueError:
        return float("nan")


def is_force_like_name(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in FORCE_NAME_MARKERS)


class LinearReadout:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=float(cfg["C"]),
            class_weight=cfg["class_weight"],
            max_iter=int(cfg["max_iter"]),
            solver="lbfgs",
        )
        self.fitted = False
        self.n_features = 0
        self.failure = ""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearReadout":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_features = X.shape[1] if X.ndim == 2 else 0
        if len(y) < 4 or len(np.unique(y)) < 2:
            self.failure = "train_labels_not_binary_or_too_few"
            return self
        if X.size == 0 or X.shape[1] == 0:
            self.failure = "empty_features"
            return self
        Xs = self.scaler.fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        self.clf.fit(Xs, y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full(len(X), 0.5, dtype=float)
        Xs = self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        proba = self.clf.predict_proba(Xs)
        classes = list(self.clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X), dtype=float)

    @property
    def n_parameters(self) -> int:
        if not self.fitted:
            return 0
        return int(np.size(self.clf.coef_) + np.size(self.clf.intercept_))


def metric_row(
    *,
    y: np.ndarray,
    p: np.ndarray,
    ece_bins: int,
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    yhat = (p >= 0.5).astype(int)
    out = {
        "n_samples": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "log_loss": float("nan"),
        "calibration_ece": float("nan"),
        "auc": float("nan"),
    }
    if len(y) == 0:
        return out
    try:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y, yhat))
    except ValueError:
        pass
    try:
        out["macro_f1"] = float(f1_score(y, yhat, average="macro", zero_division=0))
    except ValueError:
        pass
    out["log_loss"] = safe_log_loss(y, p)
    out["calibration_ece"] = ece_score(y, p, n_bins=ece_bins)
    out["auc"] = safe_auc(y, p)
    return out


def collect_frame_task(
    episodes: list[dict],
    indices: np.ndarray,
    *,
    threshold: float,
    margin: float,
    horizon: int,
    task: str,
) -> list[tuple[int, int]]:
    """Return (episode_index, frame_index t) pairs for a frame-level task."""
    pairs: list[tuple[int, int]] = []
    for ei in indices:
        frames = episodes[int(ei)]["frames"]
        if task == "escape_event":
            if frames:
                pairs.append((int(ei), 0))
            continue
        for t in range(0, max(0, len(frames) - horizon)):
            pairs.append((int(ei), t))
    return pairs


def build_targets(
    episodes: list[dict],
    pairs: list[tuple[int, int]],
    *,
    task: str,
    threshold: float,
    margin: float,
    horizon: int,
) -> np.ndarray:
    y = []
    for ei, t in pairs:
        frames = episodes[ei]["frames"]
        if task == "escape_event":
            y.append(episode_escape(episodes[ei]))
            continue
        fr = frames[t]
        nxt = frames[t + horizon]
        basin_t = stored_basin(fr)
        basin_n = stored_basin(nxt)
        if task == "next_basin":
            y.append(int(basin_n == "loaded"))
        elif task == "boundary_event":
            y.append(boundary_event(frame_fx(fr), basin_t, basin_n, threshold, margin))
        else:
            raise ValueError(task)
    return np.asarray(y, dtype=int)


class GeometryFit:
    def __init__(self) -> None:
        self.scalar_keys: list[str] = []
        self.cluster_keys: list[str] = []
        self.kmeans: KMeans | None = None
        self.cluster_scaler: RobustScaler | None = None
        self.kmeans_force: KMeans | None = None
        self.force_cluster_scaler: RobustScaler | None = None
        self.force_cluster_keys: list[str] = []
        self.hmm: GaussianHMM2 | None = None
        self.mode_available = False
        self.failures: list[str] = []
        self.fit_seconds: dict[str, float] = {}
        self.observation_dimension: dict[str, int] = {}
        self.n_parameters: dict[str, int] = {}
        self.readouts: dict[tuple[str, str], LinearReadout] = {}


def fit_representations(
    episodes: list[dict],
    train_idx: np.ndarray,
    *,
    scorer_cfg: dict[str, Any],
    threshold: float,
) -> GeometryFit:
    fitted = GeometryFit()
    train_eps = [episodes[i] for i in train_idx]
    key_counts: dict[str, int] = defaultdict(int)
    n_frames = 0
    mode_vals: list[float] = []
    for ep in train_eps:
        for fr in ep["frames"]:
            n_frames += 1
            sc = frame_scalars(fr)
            for k in sc:
                key_counts[k] += 1
            mv = contact_mode(fr)
            if mv is not None:
                mode_vals.append(mv)
    # Keep keys present on at least half of train frames.
    fitted.scalar_keys = sorted(k for k, c in key_counts.items() if c >= max(1, n_frames // 2))
    fx_train = []
    X_full = []
    for ep in train_eps:
        for fr in ep["frames"]:
            sc = frame_scalars(fr)
            X_full.append([sc.get(k, 0.0) for k in fitted.scalar_keys])
            fx_train.append(frame_fx(fr))
    X_full = np.asarray(X_full, dtype=float) if X_full else np.zeros((0, 0))
    fx_train_a = np.asarray(fx_train, dtype=float)

    drop = set()
    for i, name in enumerate(fitted.scalar_keys):
        if is_force_like_name(name):
            drop.add(name)
            continue
        if len(fx_train_a) and np.all(np.isfinite(fx_train_a)) and X_full.size:
            col = X_full[:, i]
            if np.std(col) > 1e-12 and np.std(fx_train_a) > 1e-12:
                corr = abs(float(np.corrcoef(col, fx_train_a)[0, 1]))
                if corr >= float(scorer_cfg["kmeans_fx_corr_drop"]):
                    drop.add(name)
    fitted.cluster_keys = [k for k in fitted.scalar_keys if k not in drop]
    fitted.mode_available = len(mode_vals) >= max(4, n_frames // 4) and len(set(round(v, 6) for v in mode_vals)) >= 2
    if not fitted.mode_available:
        fitted.failures.append("instantaneous_contact_mode_degenerate_or_missing")

    t0 = time.perf_counter()
    if X_full.size and fitted.cluster_keys:
        cols = [fitted.scalar_keys.index(k) for k in fitted.cluster_keys]
        Xc = np.clip(X_full[:, cols], -1e6, 1e6)
        fitted.cluster_scaler = RobustScaler().fit(Xc)
        Xs = np.clip(fitted.cluster_scaler.transform(Xc), -5, 5)
        fitted.kmeans = KMeans(
            n_clusters=int(scorer_cfg["kmeans_clusters"]),
            random_state=int(scorer_cfg["seed"]),
            n_init="auto",
        ).fit(Xs)
    else:
        fitted.failures.append("unguarded_clustering_no_features")
    fitted.fit_seconds["unguarded_clustering"] = time.perf_counter() - t0

    # Fairness baseline: same k-means recipe, but force-like coordinates retained.
    t0 = time.perf_counter()
    fitted.force_cluster_keys = list(fitted.scalar_keys)
    if X_full.size and fitted.force_cluster_keys:
        Xf = np.clip(X_full, -1e6, 1e6)
        fitted.force_cluster_scaler = RobustScaler().fit(Xf)
        Xs_f = np.clip(fitted.force_cluster_scaler.transform(Xf), -5, 5)
        fitted.kmeans_force = KMeans(
            n_clusters=int(scorer_cfg["kmeans_clusters"]),
            random_state=int(scorer_cfg["seed"]),
            n_init="auto",
        ).fit(Xs_f)
    else:
        fitted.failures.append("force_aware_clustering_no_features")
    fitted.fit_seconds["force_aware_clustering"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seqs = []
    for ep in train_eps:
        seq = np.array([frame_fx(fr) for fr in ep["frames"]], dtype=float)
        seq = np.nan_to_num(seq, nan=0.0)
        if len(seq):
            seqs.append(seq)
    fitted.hmm = GaussianHMM2(n_iter=int(scorer_cfg["hmm_em_iterations"]), seed=int(scorer_cfg["seed"]))
    if seqs:
        fitted.hmm.fit(seqs)
    else:
        fitted.failures.append("two_state_hmm_no_sequences")
    fitted.fit_seconds["two_state_hmm"] = time.perf_counter() - t0
    fitted.fit_seconds["full_observation"] = 0.0
    fitted.fit_seconds["instantaneous_contact_mode"] = 0.0
    fitted.fit_seconds["qualified_organization"] = 0.0

    fitted.observation_dimension = {
        "full_observation": len(fitted.scalar_keys),
        "instantaneous_contact_mode": 1 if fitted.mode_available else 0,
        "unguarded_clustering": int(scorer_cfg["kmeans_clusters"]),
        "force_aware_clustering": int(scorer_cfg["kmeans_clusters"]),
        "two_state_hmm": 2,
        "scalar_fx": 1,
        "scalar_fx_quadratic": 2,
        "qualified_organization": 2,
        "organization_cell_only": 1,
    }
    fitted.fit_seconds["scalar_fx"] = 0.0
    fitted.fit_seconds["scalar_fx_quadratic"] = 0.0
    return fitted


def representation_matrix(
    episodes: list[dict],
    pairs: list[tuple[int, int]],
    name: str,
    fitted: GeometryFit,
    *,
    threshold: float,
) -> np.ndarray:
    rows: list[list[float]] = []
    hmm_cache: dict[int, np.ndarray] = {}
    for ei, t in pairs:
        frames = episodes[ei]["frames"]
        fr = frames[t]
        if name == "full_observation":
            sc = frame_scalars(fr)
            rows.append([sc.get(k, 0.0) for k in fitted.scalar_keys])
        elif name == "instantaneous_contact_mode":
            mv = contact_mode(fr)
            rows.append([float(mv) if mv is not None else 0.0])
        elif name == "unguarded_clustering":
            if fitted.kmeans is None or fitted.cluster_scaler is None or not fitted.cluster_keys:
                rows.append([0.0, 0.0])
                continue
            sc = frame_scalars(fr)
            vec = np.array([[sc.get(k, 0.0) for k in fitted.cluster_keys]], dtype=float)
            xs = np.clip(fitted.cluster_scaler.transform(np.clip(vec, -1e6, 1e6)), -5, 5)
            lab = int(fitted.kmeans.predict(xs)[0])
            one = [0.0, 0.0]
            one[min(lab, 1)] = 1.0
            rows.append(one)
        elif name == "force_aware_clustering":
            if fitted.kmeans_force is None or fitted.force_cluster_scaler is None or not fitted.force_cluster_keys:
                rows.append([0.0, 0.0])
                continue
            sc = frame_scalars(fr)
            vec = np.array([[sc.get(k, 0.0) for k in fitted.force_cluster_keys]], dtype=float)
            xs = np.clip(fitted.force_cluster_scaler.transform(np.clip(vec, -1e6, 1e6)), -5, 5)
            lab = int(fitted.kmeans_force.predict(xs)[0])
            one = [0.0, 0.0]
            one[min(lab, 1)] = 1.0
            rows.append(one)
        elif name == "two_state_hmm":
            if ei not in hmm_cache:
                seq = np.nan_to_num(np.array([frame_fx(f) for f in frames], dtype=float), nan=0.0)
                hmm_cache[ei] = fitted.hmm.filtered_posterior(seq) if fitted.hmm is not None else np.zeros((len(frames), 2))
            post = hmm_cache[ei]
            if t < len(post):
                rows.append(post[t].tolist())
            else:
                rows.append([0.5, 0.5])
        elif name == "scalar_fx":
            rows.append([float(frame_fx(fr))])
        elif name == "scalar_fx_quadratic":
            fx = float(frame_fx(fr))
            rows.append([fx, (fx - threshold) ** 2])
        elif name == "qualified_organization":
            fx = frame_fx(fr)
            cell = guard_cell(fx, threshold)
            rows.append([1.0 if cell == "loaded" else 0.0, signed_distance(fx, cell, threshold)])
        elif name == "organization_cell_only":
            fx = frame_fx(fr)
            cell = guard_cell(fx, threshold)
            rows.append([1.0 if cell == "loaded" else 0.0])
        else:
            raise ValueError(name)
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 1))


def compactness_flags(name: str) -> dict[str, Any]:
    shared = name in {
        "qualified_organization",
        "organization_cell_only",
        "unguarded_clustering",
        "force_aware_clustering",
        "two_state_hmm",
        "scalar_fx",
        "scalar_fx_quadratic",
    }
    return {
        "uses_full_force_history": name == "two_state_hmm",
        "refit_per_task": name == "full_observation",
        "shared_cell_guard_transition": name == "qualified_organization",
        "n_tasks_sharing_state": 3 if shared else 1,
    }


def run() -> dict[str, Any]:
    prereg = load_yaml(PREREG_PATH)
    data_map = load_yaml(MAP_PATH)
    scorer_cfg = load_yaml(SCORER_PATH)
    threshold = float(scorer_cfg["fx_threshold"])
    margin = float(scorer_cfg["boundary_margin"])
    horizon = int(scorer_cfg["next_horizon"])
    seed = int(scorer_cfg["seed"])
    ece_bins = int(scorer_cfg["ece_bins"])
    tasks = [t["name"] for t in prereg["prediction_tasks"]]
    reps = [r["name"] for r in prereg["representations"]]
    # Fairness amendments: force-aware k-means; raw scalar fx.
    if "force_aware_clustering" not in reps:
        reps = list(reps)
        if "unguarded_clustering" in reps:
            i = reps.index("unguarded_clustering") + 1
            reps = reps[:i] + ["force_aware_clustering"] + reps[i:]
        else:
            reps.append("force_aware_clustering")
    if "scalar_fx" not in reps:
        reps = list(reps)
        if "two_state_hmm" in reps:
            i = reps.index("two_state_hmm") + 1
            reps = reps[:i] + ["scalar_fx"] + reps[i:]
        else:
            reps.append("scalar_fx")
    for extra in ("scalar_fx_quadratic", "organization_cell_only"):
        if extra not in reps:
            reps = list(reps)
            if "scalar_fx" in reps:
                i = reps.index("scalar_fx") + 1
                reps = reps[:i] + [extra] + reps[i:]
            else:
                reps.append(extra)
    ablation_reps = list(dict.fromkeys(reps + ["organization_cell_only"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    split_record: dict[str, Any] = {}
    agreement: dict[str, float] = {}

    for geom in prereg["datasets"]["geometries"]:
        spec = data_map["geometries"][geom]
        path = resolve_data(spec["path"])
        if not path.exists():
            failures.append({"geometry": geom, "reason": f"missing_trajectory_file:{spec['path']}"})
            continue
        episodes = load_json(spec["path"])["episodes"]
        ids = [ep.get("episode_id") for ep in episodes]
        if len(ids) != len(set(ids)) or any(i is None for i in ids):
            failures.append({"geometry": geom, "reason": "episode_id_not_unique"})
        y_esc = [episode_escape(ep) for ep in episodes]
        train_idx, val_idx, test_idx = split_episode_indices(
            y_esc,
            seed=seed,
            train_frac=float(scorer_cfg["split"]["train"]),
            val_frac=float(scorer_cfg["split"]["val"]),
            test_frac=float(scorer_cfg["split"]["test"]),
        )
        id_sets = {
            "train": [episodes[i]["episode_id"] for i in train_idx],
            "val": [episodes[i]["episode_id"] for i in val_idx],
            "test": [episodes[i]["episode_id"] for i in test_idx],
        }
        overlap = (
            set(id_sets["train"]) & set(id_sets["val"])
            or set(id_sets["train"]) & set(id_sets["test"])
            or set(id_sets["val"]) & set(id_sets["test"])
        )
        if overlap:
            raise RuntimeError(f"episode leakage in {geom}: {sorted(overlap)[:8]}")
        split_record[geom] = {
            "n": len(episodes),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "ids": id_sets,
        }
        n_agree = n_tot = 0
        for ep in episodes:
            for fr in ep["frames"]:
                n_tot += 1
                fx = frame_fx(fr)
                if stored_basin(fr) == guard_cell(fx, threshold):
                    n_agree += 1
        agreement[geom] = n_agree / max(n_tot, 1)
        qstat = qualification_status(spec["role"])

        fitted = fit_representations(episodes, train_idx, scorer_cfg=scorer_cfg, threshold=threshold)
        for msg in fitted.failures:
            failures.append({"geometry": geom, "reason": msg})

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        org_boundary_auc: dict[str, float] = {}

        for task in tasks:
            for split_name, idx in split_map.items():
                pairs = collect_frame_task(
                    episodes, idx, threshold=threshold, margin=margin, horizon=horizon, task=task
                )
                y = build_targets(episodes, pairs, task=task, threshold=threshold, margin=margin, horizon=horizon)
                if split_name == "train":
                    for rep in ablation_reps:
                        if rep == "instantaneous_contact_mode" and not fitted.mode_available:
                            failures.append(
                                {
                                    "geometry": geom,
                                    "representation": rep,
                                    "task": task,
                                    "reason": "mode_unavailable_or_degenerate",
                                }
                            )
                        X = representation_matrix(episodes, pairs, rep, fitted, threshold=threshold)
                        t0 = time.perf_counter()
                        model = LinearReadout(scorer_cfg["logistic"]).fit(X, y)
                        fitted.readouts[(rep, task)] = model
                        fitted.fit_seconds[f"readout:{rep}:{task}"] = time.perf_counter() - t0
                        fitted.n_parameters[f"{rep}:{task}"] = model.n_parameters
                        if model.failure:
                            failures.append(
                                {
                                    "geometry": geom,
                                    "representation": rep,
                                    "task": task,
                                    "reason": model.failure,
                                }
                            )
                for rep in ablation_reps:
                    model = fitted.readouts.get((rep, task))
                    X = representation_matrix(episodes, pairs, rep, fitted, threshold=threshold)
                    if model is None:
                        p = np.full(len(y), 0.5)
                        status = "missing_readout"
                    elif model.failure:
                        p = np.full(len(y), 0.5)
                        status = model.failure
                    elif rep == "instantaneous_contact_mode" and not fitted.mode_available:
                        p = np.full(len(y), 0.5)
                        status = "mode_unavailable_or_degenerate"
                    else:
                        p = model.predict_proba(X)
                        status = "ok"
                    m = metric_row(y=y, p=p, ece_bins=ece_bins)
                    if task == "boundary_event" and m["n_positive"] == 0:
                        status = "no_positive_boundary_events" if status == "ok" else status
                        failures.append(
                            {
                                "geometry": geom,
                                "representation": rep,
                                "task": task,
                                "split": split_name,
                                "reason": "no_positive_boundary_events",
                            }
                        )
                    row = {
                        "geometry": geom,
                        "qualification_status": qstat,
                        "representation": rep,
                        "task": task,
                        "split": split_name,
                        "status": status,
                        "observation_dimension": fitted.observation_dimension.get(rep, float("nan")),
                        **m,
                    }
                    if rep in reps:
                        metric_rows.append(row)
                    if task == "boundary_event" and split_name in {"val", "test"}:
                        guard_rows.append(
                            {
                                "geometry": geom,
                                "qualification_status": qstat,
                                "variant": rep,
                                "split": split_name,
                                "boundary_auc": m["auc"],
                                "balanced_accuracy": m["balanced_accuracy"],
                                "n_positive": m["n_positive"],
                                "n_samples": m["n_samples"],
                                "status": status,
                            }
                        )
                        if rep == "qualified_organization":
                            org_boundary_auc[split_name] = m["auc"]

        for rep in reps:
            flags = compactness_flags(rep)
            compact_rows.append(
                {
                    "geometry": geom,
                    "qualification_status": qstat,
                    "representation": rep,
                    "observation_dimension": fitted.observation_dimension.get(rep, float("nan")),
                    "uses_full_force_history": flags["uses_full_force_history"],
                    "refit_per_task": flags["refit_per_task"],
                    "shared_cell_guard_transition": flags["shared_cell_guard_transition"],
                    "n_tasks_sharing_state": flags["n_tasks_sharing_state"],
                    "n_parameters_next_basin": fitted.n_parameters.get(f"{rep}:next_basin", 0),
                    "n_parameters_escape_event": fitted.n_parameters.get(f"{rep}:escape_event", 0),
                    "n_parameters_boundary_event": fitted.n_parameters.get(f"{rep}:boundary_event", 0),
                    "fit_seconds_representation": fitted.fit_seconds.get(rep, 0.0),
                    "guard_label_agreement": agreement[geom],
                }
            )

        for split_name in ("val", "test"):
            org_auc = org_boundary_auc.get(split_name, float("nan"))
            for row in guard_rows:
                if row["geometry"] == geom and row["split"] == split_name:
                    row["delta_boundary_auc_vs_organization"] = (
                        float(row["boundary_auc"] - org_auc)
                        if math.isfinite(row["boundary_auc"]) and math.isfinite(org_auc)
                        else float("nan")
                    )

    # Pooled: unweighted mean over geometries on each split, plus micro-pool of test predictions already stored.
    pooled_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for task in tasks:
            for rep in reps:
                subset = [
                    r
                    for r in metric_rows
                    if r["representation"] == rep and r["task"] == task and r["split"] == split_name
                ]
                def _mean(key: str) -> float:
                    vals = [r[key] for r in subset if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
                    return float(np.mean(vals)) if vals else float("nan")

                def _std(key: str) -> float:
                    vals = [r[key] for r in subset if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
                    return float(np.std(vals)) if vals else float("nan")

                pooled_rows.append(
                    {
                        "pool": "unweighted_geometry_mean",
                        "representation": rep,
                        "task": task,
                        "split": split_name,
                        "n_geometries": len(subset),
                        "balanced_accuracy_mean": _mean("balanced_accuracy"),
                        "balanced_accuracy_std": _std("balanced_accuracy"),
                        "macro_f1_mean": _mean("macro_f1"),
                        "macro_f1_std": _std("macro_f1"),
                        "log_loss_mean": _mean("log_loss"),
                        "calibration_ece_mean": _mean("calibration_ece"),
                        "auc_mean": _mean("auc"),
                        "auc_std": _std("auc"),
                    }
                )

    align_rows: list[dict[str, Any]] = []
    for geom in prereg["datasets"]["geometries"]:
        spec = data_map["geometries"][geom]
        qstat = qualification_status(spec["role"])
        for rep in reps:
            def pick(task: str, key: str) -> float:
                hits = [
                    r
                    for r in metric_rows
                    if r["geometry"] == geom
                    and r["representation"] == rep
                    and r["task"] == task
                    and r["split"] == "test"
                ]
                return hits[0][key] if hits else float("nan")

            align_rows.append(
                {
                    "geometry": geom,
                    "qualification_status": qstat,
                    "representation": rep,
                    "next_basin_balanced_accuracy": pick("next_basin", "balanced_accuracy"),
                    "next_basin_macro_f1": pick("next_basin", "macro_f1"),
                    "next_basin_log_loss": pick("next_basin", "log_loss"),
                    "escape_auc": pick("escape_event", "auc"),
                    "boundary_auc": pick("boundary_event", "auc"),
                    "observation_dimension": pick("next_basin", "observation_dimension"),
                }
            )

    prereg_hash = sha256_file(PREREG_PATH)
    (OUT_DIR / "prereg_hash.txt").write_text(prereg_hash + "\n")
    write_csv(OUT_DIR / "predictive_metrics_by_geometry.csv", metric_rows)
    write_csv(OUT_DIR / "predictive_metrics_pooled.csv", pooled_rows)
    write_csv(OUT_DIR / "compactness_summary.csv", compact_rows)
    write_csv(OUT_DIR / "guard_ablation.csv", guard_rows)
    write_csv(OUT_DIR / "qualification_prediction_alignment.csv", align_rows)
    (OUT_DIR / "split_ids.json").write_text(json.dumps(split_record, indent=2, default=_json_default) + "\n")
    (OUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    excluded = [g for g in prereg["datasets"]["geometries"] if g not in split_record]
    readme = render_readme(
        prereg_hash=prereg_hash,
        scorer_hash=sha256_file(SCORER_PATH),
        map_hash=sha256_file(MAP_PATH),
        code_hash=sha256_file(Path(__file__)),
        git_head=git_head(),
        scorer_cfg=scorer_cfg,
        split_record=split_record,
        agreement=agreement,
        failures=failures,
        excluded=excluded,
        metric_rows=metric_rows,
        compact_rows=compact_rows,
        guard_rows=guard_rows,
        align_rows=align_rows,
        data_map=data_map,
    )
    (OUT_DIR / "README.md").write_text(readme)
    return {
        "n_metric_rows": len(metric_rows),
        "n_failures": len(failures),
        "excluded": excluded,
        "prereg_hash": prereg_hash,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in keys})


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.10g}"
    return v


def _fmt_num(v: Any, digits: int = 3) -> str:
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v):.{digits}f}"


def render_readme(
    *,
    prereg_hash: str,
    scorer_hash: str,
    map_hash: str,
    code_hash: str,
    git_head: str,
    scorer_cfg: dict[str, Any],
    split_record: dict[str, Any],
    agreement: dict[str, float],
    failures: list[dict[str, Any]],
    excluded: list[str],
    metric_rows: list[dict[str, Any]],
    compact_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    align_rows: list[dict[str, Any]],
    data_map: dict[str, Any],
) -> str:
    lines = [
        "# RAS Section 6 predictive readout",
        "",
        "Generated by `scripts/run_predictive_readout.py`. Existing trajectories only.",
        "No geometry was dropped after scoring.",
        "",
        "## Versions",
        "",
        f"- git HEAD: `{git_head}`",
        f"- prereg sha256: `{prereg_hash}`",
        f"- trajectory map sha256: `{map_hash}`",
        f"- scorer config sha256: `{scorer_hash}`",
        f"- scorer script sha256: `{code_hash}`",
        f"- data policy: existing_trajectories_only; new_cohort: false",
        "",
        "## Split",
        "",
        "- unit: episode",
        f"- fractions: train {scorer_cfg['split']['train']}, val {scorer_cfg['split']['val']}, test {scorer_cfg['split']['test']}",
        f"- seed: {scorer_cfg['seed']}",
        "- stratification: episode-level escape label, with a shuffle fallback if a class has <3 episodes",
        "- leakage control: asserted empty intersection of train/val/test episode IDs",
        "- fit_on_train_only: KMeans, HMM EM, scalers, and logistic readouts",
        "- fx_threshold: 8.0 (frozen; not tuned on val or test)",
        f"- next_horizon: {scorer_cfg['next_horizon']}",
        "",
        "## Data files",
        "",
    ]
    for geom, spec in data_map["geometries"].items():
        rec = split_record.get(geom, {})
        lines.append(
            f"- {geom}: `{spec['path']}` n={rec.get('n', 'MISSING')} "
            f"train/val/test={rec.get('n_train')}/{rec.get('n_val')}/{rec.get('n_test')} "
            f"guard_label_agreement={agreement.get(geom, float('nan')):.3f}"
        )
    lines += [
        "",
        "## Training parameters",
        "",
        f"- logistic: C={scorer_cfg['logistic']['C']}, class_weight={scorer_cfg['logistic']['class_weight']}, max_iter={scorer_cfg['logistic']['max_iter']}",
        f"- KMeans (unguarded): k={scorer_cfg['kmeans_clusters']}, drop force-like names and columns with |corr(fx)| >= {scorer_cfg['kmeans_fx_corr_drop']}",
        f"- KMeans (force-aware fairness baseline): same k, retains force-like coordinates including fx",
        f"- HMM: unsupervised 2-state Gaussian EM on fx, iterations={scorer_cfg['hmm_em_iterations']}; basin labels are not used",
        "- organization features: [I(fx >= 8), signed distance to the frozen guard]",
        "- next_basin / boundary_event: frame t predicts the event at t→t+1",
        "- escape_event: first-frame features predict whether a loaded→grazing transition occurs later in the episode",
        "",
        "## Held-out test metrics (primary)",
        "",
        "| geometry | status | representation | next_basin bal_acc | escape AUC | boundary AUC | dim |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in align_rows:
        lines.append(
            f"| {row['geometry']} | {row['qualification_status']} | {row['representation']} | "
            f"{_fmt_num(row['next_basin_balanced_accuracy'])} | {_fmt_num(row['escape_auc'])} | "
            f"{_fmt_num(row['boundary_auc'])} | {_fmt_num(row['observation_dimension'], 0)} |"
        )
    lines += [
        "",
        "## Guard ablation (test boundary AUC)",
        "",
        "| geometry | variant | boundary AUC | delta vs organization | n_positive |",
        "|---|---|---:|---:|---:|",
    ]
    for row in guard_rows:
        if row["split"] != "test":
            continue
        if row["variant"] not in {
            "qualified_organization",
            "organization_cell_only",
            "unguarded_clustering",
        }:
            continue
        lines.append(
            f"| {row['geometry']} | {row['variant']} | {_fmt_num(row['boundary_auc'])} | "
            f"{_fmt_num(row.get('delta_boundary_auc_vs_organization'))} | {row['n_positive']} |"
        )
    uniq_fail = []
    seen = set()
    for f in failures:
        key = json.dumps(f, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq_fail.append(f)
    lines += [
        "",
        "## Failed experiments (retained, not excluded)",
        "",
    ]
    if not uniq_fail:
        lines.append("- none")
    else:
        for f in uniq_fail:
            lines.append(f"- {f}")
    lines += [
        "",
        "## Exclusions after scoring",
        "",
        f"- geometries excluded: {excluded if excluded else 'none'}",
        "- failed geometries (sphere, block) were retained",
        "- no threshold was tuned on test",
        "- force_aware_clustering added as a pre-submission fairness baseline (sees fx); original unguarded clustering retained",
        "",
        "## Compactness (shared across tasks)",
        "",
        "| geometry | representation | dim | force history | shared cell/guard/transition |",
        "|---|---|---:|---|---|",
    ]
    for row in compact_rows:
        lines.append(
            f"| {row['geometry']} | {row['representation']} | {row['observation_dimension']} | "
            f"{row['uses_full_force_history']} | {row['shared_cell_guard_transition']} |"
        )
    lines += [
        "",
        "Primary numbers are held-out **test** metrics. Validation is reported in the CSVs",
        "and was not used to select representations, thresholds, or splits.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))
