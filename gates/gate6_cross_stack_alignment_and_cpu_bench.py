"""Gate 6: Python <-> Rust Cross-Stack Alignment & Single-Core CPU Latency / Zero-Alloc Benchmark.

Pass/Fail Quantitative Thresholds (Engineering Blueprint Chapter 7):
  (1) Rust dcpa-core-rs loads exported flat-named model.safetensors:
      per-module cosine similarity >= 0.9999 & max abs error < 1e-5 (float32 mode) across
      SGSC, Hippocampus Read & Post-Write, BCG-PC, and PMD (Broca CQ + Byte Motor + 128B Stack).
  (2) On single-core laptop CPU (DCPA-Base 4096x768 + 4096x192 Column-Major 64B-aligned bf16):
      Hippocampal single-step sparse column Householder erase-write < 50 microseconds,
      AND streaming byte ingest + Efference-Copy generation performs STRICTLY 0 allocs/byte!
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch

from dcpa.config import DCPAConfig
from dcpa.export import export_model_and_golden
from dcpa.model import DCPAModel


def verify_gate6() -> None:
    print("=== Running Gate 6: Python <-> Rust Cross-Stack Alignment & CPU Latency / Zero-Alloc Benchmark ===")
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Build dcpa-core-rs in release mode (+ verify no_std build check!)
    print("[1/4] Verifying dcpa-core-rs `no_std` compilation & building release `dcpa-cli`...")
    subprocess.run(
        ["cargo", "check", "-p", "dcpa-core-rs", "--no-default-features"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["cargo", "build", "--release", "-p", "dcpa-core-rs", "--bin", "dcpa-cli"],
        cwd=repo_root,
        check=True,
    )
    cli_bin = repo_root / "target" / "release" / "dcpa-cli"

    with tempfile.TemporaryDirectory(prefix="dcpa_gate6_") as tmp:
        tmp_dir = Path(tmp)

        # 2. Export Tiny model + Golden Test Vectors for full-module exact float32 parity
        print("[2/4] Exporting DCPA-Tiny slow weights + deterministic Golden Test Vector Bundle...")
        torch.manual_seed(202604)
        tiny_dir = tmp_dir / "tiny"
        tiny_model = DCPAModel(DCPAConfig.tiny())
        export_model_and_golden(
            tiny_model,
            tiny_dir,
            sample_text="A zibber is a tiny frog! 你好世界🦀 fn main() { let x = 1; }",
            lang_id=1,
            export_dtype=torch.float32,
        )

        # 3. Run Rust verify-golden (checks ALL 4 modules: cos_sim >= 0.9999, max_abs_err < 1e-5!)
        print("[3/4] Executing Rust `dcpa-cli verify-golden` against Python golden bundle...")
        subprocess.run([str(cli_bin), "verify-golden", str(tiny_dir)], cwd=repo_root, check=True)

        # 4. Export full DCPA-Base (d_s=192, D=768, d_h=4096, s=32 -> 7.52 MB L3 Hippocampus!) for CPU bench
        print("[4/4] Exporting full DCPA-Base (7.52 MB L3 Hippocampus) & running Rust single-core CPU bench...")
        torch.manual_seed(202605)
        base_dir = tmp_dir / "base"
        base_model = DCPAModel(DCPAConfig.base())
        export_model_and_golden(
            base_model,
            base_dir,
            sample_text="Hello DCPA Base! 你好🦀",
            lang_id=1,
            export_dtype=torch.bfloat16,
        )
        subprocess.run([str(cli_bin), "bench", str(base_dir)], cwd=repo_root, check=True)

    print("=== GATE 6 PASSED ===")


if __name__ == "__main__":
    verify_gate6()
