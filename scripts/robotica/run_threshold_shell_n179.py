#!/usr/bin/env python3
"""P1 shell on the same cylinder cohort as the predictive audit (n=179)."""
from __future__ import annotations

import importlib.util

from _paths import REPO, load_episodes  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "sens",
    REPO / "scripts" / "analyze_guard_threshold_sensitivity.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

episodes = load_episodes("data/robotica/cylinder.json.gz")
print("n", len(episodes), "cylinder.json.gz")
for t in (4.0, 6.0, 8.0, 10.0, 12.0):
    a = mod.audit_at_threshold(episodes, t)
    print(
        f"t={t:.0f} frac={a['frac_loaded']:.3f} dwell={a['dwell_ratio']:.3f} "
        f"sw={a['switch_rate']:.4f} P1={a['P1']} n_at_bdry={a['n_at_boundary']}"
    )
