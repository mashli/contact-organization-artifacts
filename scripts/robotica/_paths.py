"""Resolve the artifacts-repo root and load Robotica trajectory JSON (gzip OK).

Works in both layouts:
  - public clone: <repo>/{data,scripts,results,configs}/...
  - private tree: .../contact_organization_robotica/scripts/ with
    trajectories under <contact_grammar_emergence>/results/phase1/...
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _find_repo() -> Path:
    # Prefer an artifacts-style root that already has data/robotica
    for p in [_HERE, *_HERE.parents]:
        if (p / "data" / "robotica").is_dir() and (p / "scripts").is_dir():
            return p
    # Private latex package: scripts/ under contact_organization_robotica,
    # repo root is contact_grammar_emergence (has results/phase1)
    for p in _HERE.parents:
        if (p / "results" / "phase1").is_dir() and (p / "src" / "cge").is_dir():
            return p
    # Fallback: scripts/robotica → parents[2]
    return _HERE.parents[2]


REPO = _find_repo()
PKG = _HERE
DATA = REPO / "data" / "robotica"
SRC = REPO / "src"

_LEGACY = {
    "results/phase1/landscape_analysis/landscape_trajectories.json": "cylinder.json.gz",
    "results/phase1/cross_geometry_recovery/ellipsoid/full/landscape_trajectories.json": "ellipsoid.json.gz",
    "results/phase1/cross_geometry_recovery/sphere/full/landscape_trajectories.json": "sphere.json.gz",
    "results/phase1/characterization_block/full/landscape_trajectories.json": "block.json.gz",
    "results/phase1/cross_engine/pybullet_probe/landscape_trajectories.json": "pybullet_cylinder.json.gz",
    "results/stage1/box_pivot_wall_pilot/landscape_tau_ref30_thr5.0.json": "wall_pivot.json.gz",
    "results/stage1/cylinder_horizontal_pilot/landscape_trajectories.json": "cylinder_horizontal.json.gz",
}


def ensure_src_on_path() -> Path:
    import sys

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return SRC


def resolve_data(rel: str | Path) -> Path:
    """Map a config-relative path to a file under data/robotica or repo root."""
    rel_s = str(rel).replace("\\", "/").lstrip("./")
    candidates: list[Path] = []
    if rel_s in _LEGACY:
        candidates.append(DATA / _LEGACY[rel_s])
    if rel_s.startswith("data/robotica/"):
        candidates.append(REPO / rel_s)
        candidates.append(REPO / (rel_s + ".gz"))
    else:
        candidates.append(DATA / rel_s)
        candidates.append(DATA / (rel_s + ".gz"))
        candidates.append(DATA / Path(rel_s).name)
        candidates.append(DATA / (Path(rel_s).name + ".gz"))
    candidates.append(REPO / rel_s)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Could not resolve trajectory {rel!r}. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def load_json(rel: str | Path) -> Any:
    path = resolve_data(rel)
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def load_episodes(rel: str | Path) -> list[dict]:
    obj = load_json(rel)
    if isinstance(obj, dict) and "episodes" in obj:
        return list(obj["episodes"])
    if isinstance(obj, list):
        return obj
    raise ValueError(f"No episodes in {rel}")
