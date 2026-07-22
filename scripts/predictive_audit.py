#!/usr/bin/env python3
"""Predictive audit for Paper II (next-mode and k-step-ahead forecasts).

Compares an organizational transition matrix P_pi (estimated from ground-truth
basin labels) against simple baselines and a KMeans+Markov baseline that learns
a 2-state partition from trajectory features without guard information.

Outputs JSON for the paper table pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


def load_basin_sequences(path: str) -> tuple[list[list[int]], list[str]]:
    """Return integer basin sequences and the original label order."""
    with open(path) as f:
        data = json.load(f)
    episodes = data["episodes"]
    sequences = []
    label_to_idx: dict[str, int] = {}
    labels: list[str] = []
    for ep in episodes:
        basins = [fr["basin"] for fr in ep.get("frames", [])]
        seq = []
        for b in basins:
            if b not in label_to_idx:
                label_to_idx[b] = len(labels)
                labels.append(b)
            seq.append(label_to_idx[b])
        sequences.append(seq)
    return sequences, labels


def load_feature_sequences(path: str, *, feature_keys: list[str] | None = None) -> list[np.ndarray]:
    """Return feature vectors per frame for each episode.

    If feature_keys is None, use all scalar features; otherwise use only those keys.
    """
    with open(path) as f:
        data = json.load(f)
    episodes = data["episodes"]
    feats = []
    for ep in episodes:
        vectors = []
        for fr in ep.get("frames", []):
            feat = fr.get("features", {})
            if not feat:
                vectors.append(np.zeros(1))
                continue
            keys = feature_keys if feature_keys is not None else sorted(feat.keys())
            vec = []
            for k in keys:
                v = feat.get(k)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    vec.append(float(v))
            if not vec:
                vec = [0.0]
            vectors.append(np.array(vec, dtype=float))
        feats.append(np.array(vectors, dtype=float))
    return feats


def estimate_p(sequences: list[list[int]], n_states: int, *, lag: int = 1) -> np.ndarray:
    C = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        for i in range(len(seq) - lag):
            C[seq[i], seq[i + lag]] += 1.0
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return C / row_sums


def kmeans_markov_model(
    train_feats: list[np.ndarray],
    test_feats: list[np.ndarray],
    n_states: int,
) -> tuple[np.ndarray, object, list[list[int]], list[list[int]]]:
    """Learn a 2-state partition from robust-scaled, clipped features with KMeans, then estimate a Markov chain.

    Returns (P, scaler, train_cluster_seqs, test_cluster_seqs).
    """
    X_train = np.vstack(train_feats)
    # Robust scaling + clipping to avoid overflow from extreme force/torque features.
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = np.clip(scaler.transform(X_train), -5, 5)
    kmeans = KMeans(n_clusters=n_states, random_state=0, n_init="auto").fit(X_train_scaled)
    train_cluster = [kmeans.predict(np.clip(scaler.transform(f), -5, 5)) for f in train_feats]
    test_cluster = [kmeans.predict(np.clip(scaler.transform(f), -5, 5)) for f in test_feats]
    P = estimate_p(train_cluster, n_states, lag=1)
    return P, scaler, train_cluster, test_cluster


def k_step_metrics(
    sequences: list[list[int]], P: np.ndarray, k: int
) -> dict[str, float]:
    """Predict k-step ahead state and return accuracy / log-likelihood / Brier."""
    n_states = P.shape[0]
    Pk = np.linalg.matrix_power(P, k)
    correct = 0
    total = 0
    log_likelihood = 0.0
    brier = 0.0
    for seq in sequences:
        for t in range(len(seq) - k):
            s = seq[t]
            next_s = seq[t + k]
            pred = int(Pk[s].argmax())
            correct += int(pred == next_s)
            total += 1
            p_next = Pk[s, next_s]
            log_likelihood += math.log(max(p_next, 1e-12))
            for j in range(n_states):
                target = 1.0 if j == next_s else 0.0
                brier += (Pk[s, j] - target) ** 2
    return {
        "accuracy": correct / max(total, 1),
        "log_likelihood": log_likelihood / max(total, 1),
        "brier_score": brier / max(total, 1),
        "n_predictions": total,
    }


def one_step_metrics(sequences: list[list[int]], P: np.ndarray) -> dict[str, float]:
    return k_step_metrics(sequences, P, 1)


def baseline_uniform(n_states: int) -> np.ndarray:
    return np.ones((n_states, n_states)) / n_states


def baseline_marginal(train_sequences: list[list[int]], n_states: int) -> np.ndarray:
    counts = np.zeros(n_states)
    for seq in train_sequences:
        for s in seq:
            counts[s] += 1.0
    counts += 1e-12
    p = counts / counts.sum()
    return np.tile(p, (n_states, 1))


def baseline_persistent(n_states: int) -> np.ndarray:
    return np.eye(n_states)


def score_all(
    test_sequences: list[list[int]],
    train_sequences: list[list[int]],
    P_org: np.ndarray,
    n_states: int,
    *,
    k_steps: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    """Compute next-mode and k-step metrics for all baselines."""
    P_uniform = baseline_uniform(n_states)
    P_marginal = baseline_marginal(train_sequences, n_states)
    P_persistent = baseline_persistent(n_states)
    result: dict[str, Any] = {}
    for k in k_steps:
        result[f"k{k}"] = {
            "organizational": k_step_metrics(test_sequences, P_org, k),
            "uniform": k_step_metrics(test_sequences, P_uniform, k),
            "marginal": k_step_metrics(test_sequences, P_marginal, k),
            "persistent": k_step_metrics(test_sequences, P_persistent, k),
        }
    return result


def run_audit(
    trajectory_path: str,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
    k_steps: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    sequences, labels = load_basin_sequences(trajectory_path)
    n_states = max(max(s) for s in sequences) + 1
    n_test = max(1, int(round(len(sequences) * test_fraction)))
    indices = list(range(len(sequences)))
    random.shuffle(indices)
    test_idx = set(indices[:n_test])
    train = [s for i, s in enumerate(sequences) if i not in test_idx]
    test = [s for i, s in enumerate(sequences) if i in test_idx]
    P_org = estimate_p(train, n_states, lag=1)
    metrics = score_all(test, train, P_org, n_states, k_steps=k_steps)

    # KMeans+Markov baselines: only when >=2 states and we have features.
    # Two variants: (a) all robust-scaled features; (b) force-f_x only (guard-like).
    kmeans_metrics: dict[str, Any] | None = None
    if n_states >= 2:
        try:
            kmeans_metrics = {}
            for variant_name, feature_keys in [
                ("all_features", None),
                ("force_fx_only", ["resultant_force_f_x"]),
            ]:
                feats = load_feature_sequences(trajectory_path, feature_keys=feature_keys)
                train_feats = [feats[i] for i in range(len(sequences)) if i not in test_idx]
                test_feats = [feats[i] for i in range(len(sequences)) if i in test_idx]
                P_km, _, train_km, test_km = kmeans_markov_model(
                    train_feats, test_feats, n_states
                )
                km_metrics = score_all(test_km, train_km, P_km, n_states, k_steps=k_steps)
                kmeans_metrics[variant_name] = {
                    "P": P_km.tolist(),
                    "feature_keys": feature_keys,
                    "metrics": km_metrics,
                }
        except Exception as e:
            kmeans_metrics = {"error": str(e)}

    return {
        "source_path": trajectory_path,
        "n_episodes": len(sequences),
        "n_train": len(train),
        "n_test": len(test),
        "n_states": n_states,
        "labels": labels,
        "P_organizational": P_org.tolist(),
        "metrics": metrics,
        "kmeans_markov": kmeans_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-steps", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    result = run_audit(
        args.trajectory,
        test_fraction=args.test_fraction,
        seed=args.seed,
        k_steps=tuple(args.k_steps),
    )
    result["name"] = args.name
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")
    # Concise summary.
    print(f"{args.name}: n={result['n_episodes']} train={result['n_train']} test={result['n_test']} states={result['n_states']}")
    for k in args.k_steps:
        m = result["metrics"][f"k{k}"]
        km = result.get("kmeans_markov") or {}
        km_all = (km.get("all_features") or {}).get("metrics", {}).get(f"k{k}", {}).get("organizational", {})
        km_fx = (km.get("force_fx_only") or {}).get("metrics", {}).get(f"k{k}", {}).get("organizational", {})
        print(f"  k={k}: org acc={m['organizational']['accuracy']:.3f} "
              f"kmeans_all={km_all.get('accuracy', float('nan')):.3f} "
              f"kmeans_fx={km_fx.get('accuracy', float('nan')):.3f} "
              f"marg={m['marginal']['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
