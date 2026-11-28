"""Gate 6 (DCPA-Flat v3.3): Python <-> Rust Cross-Stack Alignment, Zero-Alloc Bench & Sleep Consolidation.

Pass/Fail Quantitative Thresholds (`docs/DCPA_FLAT_V3_3_SPEC.md` + WS-6):
  1. `dcpa-core-rs` compiles cleanly under `--no-default-features` (`no_std`).
  2. Rust `dcpa-cli verify-golden-flat` achieves `cos_sim >= 0.9999` and `max_abs_err < 1e-5`
     across ALL 17 module tensors (SGSC, L1, Carrier G1, L2-L3, Carrier G2, Hippocampus, L4-L8,
     CopyRing, PMD) for BOTH `share_stem_ring=False` (+32KB sem ring) and `share_stem_ring=True`
     (0KB delta) AND `use_local_window_layers=False` ablation.
  3. Hot-mounting a trained Additive Dead-Zone plugin pack in Rust achieves:
       - Target domain `plugin_byte_logits`: `cos_sim >= 0.9999`, `max_abs_err < 1e-5`
       - Non-target EN/ZH anchor drift: STRICTLY `0.0`
       - Unmount restoration: STRICTLY `0.0`
  4. `dcpa-cli bench-flat` verifies STRICTLY `0 allocs/byte` and sub-microsecond `BrainSnapshotV3` fork/restore.
  5. `dcpa-sleep-daemon` executes real awake fragment capture -> unified `dcpa.train_plugin --mode sleep`
     -> Rust read-only base immutability & zero-drift verification -> atomic `personal_sleep.dcpa_expert.safetensors`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.export import export_flat_v3_and_golden
from dcpa.interfaces import DCPAFlatConfig, LANG_DOMAINS_V3
from dcpa.model_flat import DCPAModelFlat
from dcpa.train_plugin import train_additive_plugin_pack
from gates.run_flat_killgate_5k import _make_synthetic_grid_batch


def verify_gate6_flat_v3() -> None:
    print("=== Running Gate 6 (DCPA-Flat v3.3): Cross-Stack Parity, Zero-Alloc Bench & Sleep Daemon ===")

    # 1. Verify no_std compilation & build release dcpa-cli + dcpa-sleep-daemon
    print("[1/5] Checking dcpa-core-rs `no_std` & building release `dcpa-cli` + `dcpa-sleep-daemon`...")
    subprocess.run(
        ["cargo", "check", "-p", "dcpa-core-rs", "--no-default-features"],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "dcpa-cli", "--bin", "dcpa-sleep-daemon"],
        cwd=_REPO_ROOT,
        check=True,
    )
    cli_bin = _REPO_ROOT / "target" / "release" / "dcpa-cli"
    sleep_bin = _REPO_ROOT / "target" / "release" / "dcpa-sleep-daemon"

    with tempfile.TemporaryDirectory(prefix="dcpa_flat_gate6_") as tmp:
        tmp_dir = Path(tmp)

        # 2. Export micro model WITH trained Additive Dead-Zone SQL plugin & EN/ZH anchor
        print("[2/5] Training SQL Additive Dead-Zone plugin & exporting Flat v3.3 Golden Bundle...")
        torch.manual_seed(202611)
        cfg = DCPAFlatConfig.flat_micro()
        model = DCPAModelFlat(cfg)
        dev = torch.device("cpu")
        sql_batches = [
            _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=41, device=dev),
            _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=42, device=dev),
        ]
        anchor_batches = [
            _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["en"], seed=51, device=dev),
            _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["zh"], seed=52, device=dev),
        ]
        pack, report = train_additive_plugin_pack(
            model,
            pack_id="gate6_sql_plugin",
            target_batches=sql_batches,
            anchor_batches=anchor_batches,
            steps=10,
            experts_per_layer=1,
            lr=5e-3,
        )
        assert report.passed_zero_drift and report.anchor_max_logit_drift == 0.0
        model.unmount_expert_pack(pack.pack_id)

        dir_main = tmp_dir / "flat_w128_semring"
        export_flat_v3_and_golden(
            model,
            dir_main,
            sample_text="\x1dS\nsql_plan\x1dE\nSELECT acct_id, SUM(amount) FROM ledger_entries WHERE status = 'POSTED';\n",
            lang_id=LANG_DOMAINS_V3["sql_struct"],
            style_id=4,
            hip_enabled=True,
            plugin_pack=pack,
            anchor_text="The analytical engine computes deterministic structured representations across nested clauses.\n\n",
            anchor_lang_id=LANG_DOMAINS_V3["en"],
            anchor_style_id=0,
        )

        # 3. Rust verify-golden-flat across all 17 tensors + plugin parity + strict 0.0 anchor drift
        print("[3/5] Running Rust `dcpa-cli verify-golden-flat` (W=128 + sem_ring + Hippocampus + Plugin)...")
        subprocess.run([str(cli_bin), "verify-golden-flat", str(dir_main)], cwd=_REPO_ROOT, check=True)

        # Also check share_stem_ring=True (0KB delta) and use_local_window_layers=False (pure K=8)
        for tag, use_win, share_stem in (
            ("flat_w128_0kb", True, True),
            ("flat_pure_k8", False, False),
        ):
            torch.manual_seed(202612)
            c_sub = DCPAFlatConfig.flat_micro()
            c_sub.use_local_window_layers = use_win
            c_sub.share_stem_ring = share_stem
            m_sub = DCPAModelFlat(c_sub)
            d_sub = tmp_dir / tag
            export_flat_v3_and_golden(m_sub, d_sub, hip_enabled=True)
            subprocess.run([str(cli_bin), "verify-golden-flat", str(d_sub)], cwd=_REPO_ROOT, check=True)

        # 4. Single-core zero-alloc CPU bench + BrainSnapshotV3 fork/restore bench
        print("[4/5] Running Rust `dcpa-cli bench-flat` (0 allocs/byte & BrainSnapshotV3)...")
        subprocess.run([str(cli_bin), "bench-flat", str(dir_main)], cwd=_REPO_ROOT, check=True)

        # 5. End-to-end dcpa-sleep-daemon real-fragment consolidation -> personal_sleep.dcpa_expert.safetensors
        print("[5/5] Running end-to-end `dcpa-sleep-daemon` real-fragment -> sleep plugin consolidation...")
        sleep_out = tmp_dir / "sleep_plugins" / "personal_sleep.dcpa_expert.safetensors"
        sleep_work = tmp_dir / "sleep_work"
        subprocess.run(
            [
                str(sleep_bin),
                str(dir_main),
                str(sleep_out),
                "--steps",
                "6",
                "--python",
                sys.executable,
                "--work-dir",
                str(sleep_work),
            ],
            cwd=_REPO_ROOT,
            check=True,
        )
        assert sleep_out.is_file() and sleep_out.stat().st_size > 0

    print("=== GATE 6 (DCPA-FLAT v3.3) PASSED ===")


if __name__ == "__main__":
    verify_gate6_flat_v3()
