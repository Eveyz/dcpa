"""Unified runner executing all six DCPA quantitative acceptance gates (Gate 1 -> Gate 6)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DCPA Acceptance Gates 1..6 sequentially.")
    ap.add_argument("--fast", action="store_true", help="Use reduced steps for quick CI smoke verification")
    args = ap.parse_args()

    gates_dir = Path(__file__).resolve().parent
    py = sys.executable

    commands = [
        [py, str(gates_dir / "gate1_sgsc_causality_and_scan.py")],
        [
            py,
            str(gates_dir / "gate2_rope_spelling_duality.py"),
            "--steps",
            "1500" if args.fast else "5000",
            "--preset",
            "tiny" if args.fast else "base",
        ],
        [py, str(gates_dir / "gate3_bcg_pc_merge_and_o1_mem.py")],
        [
            py,
            str(gates_dir / "gate4_stack_long_range_brackets.py"),
            "--steps",
            "600" if args.fast else "800",
        ],
        [
            py,
            str(gates_dir / "gate5_hippocampus_oneshot_wug.py"),
            "--cal-steps",
            "1600" if args.fast else "2200",
            "--episodes",
            "40",
        ],
        [py, str(gates_dir / "gate6_cross_stack_alignment_and_cpu_bench.py")],
    ]

    for i, cmd in enumerate(commands, start=1):
        print(f"\n{'=' * 72}\n>>> LAUNCHING GATE {i}/6: {' '.join(cmd)}\n{'=' * 72}", flush=True)
        subprocess.run(cmd, check=True)

    print("\nALL 6 DCPA ACCEPTANCE GATES PASSED!")


if __name__ == "__main__":
    main()
