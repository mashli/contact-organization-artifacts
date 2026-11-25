"""Operational basin labels for Configuration Landscape validation.

Basin B is defined by physics-proxy rule (not clustering). Pre-registered threshold.
"""

from __future__ import annotations

from typing import Any

BASIN_FX_THRESHOLD_N = 8.0
BASIN_LABELS = ("grazing", "loaded")


def basin_label_from_fx(force_f_x: float, *, threshold: float = BASIN_FX_THRESHOLD_N) -> str:
    return "grazing" if float(force_f_x) < threshold else "loaded"


def basin_label_from_row(row: dict[str, Any], *, threshold: float = BASIN_FX_THRESHOLD_N) -> str:
    return basin_label_from_fx(float(row["resultant_force_f_x"]), threshold=threshold)


def basin_labels_unique(rows: list[dict], *, threshold: float = BASIN_FX_THRESHOLD_N) -> set[str]:
    return {basin_label_from_row(r, threshold=threshold) for r in rows}
