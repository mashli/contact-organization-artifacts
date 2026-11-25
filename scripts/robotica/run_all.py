#!/usr/bin/env python3
"""One-click Robotica artifact checks (existing trajectories only).

Usage (from the artifacts repo root):

    python3 scripts/robotica/run_all.py

Runs the cheap audits that a reviewer can re-execute without MuJoCo:
  1) cylinder threshold shell (n=179) — must match tab_threshold_shell
  2) signature witness round (P1--P4 / Sig)
  3) (mu, t*) qualification map — self-checks against frozen friction slice
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run(label: str, argv: list[str]) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    r = subprocess.run(argv, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {r.returncode})")


def main() -> int:
    py = sys.executable
    run(
        "1/3 threshold shell (cylinder n=179)",
        [py, str(HERE / "run_threshold_shell_n179.py")],
    )
    run(
        "2/3 signature witness round",
        [py, str(HERE / "run_sig_witness_round.py")],
    )
    run(
        "3/3 (mu, t*) qualification map + self-check",
        [py, str(REPO / "scripts" / "analyze_qualification_map_mu_threshold.py")],
    )
    print("\nAll Robotica one-click checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
