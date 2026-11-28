"""WS-6 End-to-End Tests: Python Streaming Parity, Rust `dcpa-core-rs` Gate 6, & `dcpa-sleep-daemon` v3.3."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.export import export_flat_v3_and_golden
from dcpa.interfaces import DCPAFlatConfig, LANG_DOMAINS_V3, StaticGridBatch
from dcpa.model_flat import DCPAModelFlat
from dcpa.train_plugin import load_plugin_pack_safetensors, train_additive_plugin_pack
from gates.run_flat_killgate_5k import _make_synthetic_grid_batch


@pytest.mark.parametrize("hip_enabled", [False, True])
def test_python_flat_batched_vs_streaming_parity(hip_enabled: bool) -> None:
    """Verify `DCPAModelFlat.forward_streaming_sequence` matches batched `forward` `< 1e-5` in eval mode."""
    torch.manual_seed(2026)
    cfg = DCPAFlatConfig.flat_micro()
    model = DCPAModelFlat(cfg)
    model.eval()

    text = "\x1dS\nplan: sum\x1dE\ndef foo(xs):\n    return sum(xs)\nA zibber is a tiny frog! 你好世界"
    raw = torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
    T = cfg.grid_bytes
    L = int(raw.numel())

    x = torch.zeros((1, T), dtype=torch.uint8)
    x[0, :L] = raw
    doc_id = torch.full((1, T), -1, dtype=torch.int32)
    doc_id[0, :L] = 0
    doc_st = torch.zeros((1, T), dtype=torch.bool)
    doc_st[0, 0] = True
    doc_en = torch.zeros((1, T), dtype=torch.bool)
    doc_en[0, L - 1] = True
    lw = torch.zeros((1, T), dtype=torch.float32)
    lw[0, :L] = 1.0
    meta = torch.zeros((64, 3), dtype=torch.int32)
    meta[0] = torch.tensor([3, 4, 1], dtype=torch.int32)
    batch = StaticGridBatch(x, doc_id, doc_st, doc_en, lw, meta)

    with torch.no_grad():
        bo = model(batch, stage="full" if hip_enabled else "cortex", hip_enabled=hip_enabled)
        so = model.forward_streaming_sequence(raw, lang_id=3, style_id=4, hip_enabled=hip_enabled)

    M = int(so["f_stem"].shape[0])
    for name, b_t, s_t in (
        ("f_stem", bo.bridge.f_stem[0, :M], so["f_stem"]),
        ("x3", bo.x3[0, :M], so["x3"]),
        ("g2_prior_slots", bo.g2_scan.slots_seq[0, :M], so["g2_prior_slots"]),
        ("z_hat", bo.pred.z_hat[0, :M], so["z_hat"]),
        ("u_n", bo.pred.u_n[0, :M], so["u_n"]),
        ("f_out", bo.f_out_seg[0, :M], so["f_out"]),
        ("byte_logits", bo.pmd.byte_logits[0, :L], so["byte_logits"]),
    ):
        diff = float((b_t.float() - s_t.float()).abs().max().item())
        assert diff < 1e-5, f"Batch vs streaming mismatch on {name} (hip={hip_enabled}): {diff:.3e}"


def test_rust_no_std_workspace_and_gate6_flat_plus_sleep_daemon(tmp_path: Path) -> None:
    """Full WS-6 end-to-end verification:
      1. `dcpa-core-rs` compiles under `--no-default-features` (`no_std`) & workspace unit tests pass.
      2. Python -> Rust `dcpa-cli verify-golden-flat` passes `< 1e-5` across all 17 tensors,
         hot-mounted plugin target logits `< 1e-5`, non-target anchor drift `== 0.0`, `0 allocs/byte`.
      3. `dcpa-cli bench-flat` passes `0 allocs/byte` and `BrainSnapshotV3` fork/restore.
      4. `dcpa-sleep-daemon` consolidates real awake fragments into `personal_sleep.dcpa_expert.safetensors`
         while keeping base `model.safetensors` SHA-256 byte-for-byte identical.
    """
    subprocess.run(
        ["cargo", "check", "-p", "dcpa-core-rs", "--no-default-features"],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(["cargo", "test", "--workspace"], cwd=_REPO_ROOT, check=True)
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "dcpa-cli", "--bin", "dcpa-sleep-daemon"],
        cwd=_REPO_ROOT,
        check=True,
    )
    cli_bin = _REPO_ROOT / "target" / "release" / "dcpa-cli"
    sleep_bin = _REPO_ROOT / "target" / "release" / "dcpa-sleep-daemon"

    torch.manual_seed(202611)
    cfg = DCPAFlatConfig.flat_micro()
    model = DCPAModelFlat(cfg)
    dev = torch.device("cpu")

    sql_batches = [
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=61, device=dev),
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=62, device=dev),
    ]
    anchor_batches = [
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["en"], seed=71, device=dev),
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["zh"], seed=72, device=dev),
    ]
    pack, report = train_additive_plugin_pack(
        model,
        pack_id="ws6_sql_plugin",
        target_batches=sql_batches,
        anchor_batches=anchor_batches,
        steps=8,
        experts_per_layer=1,
        lr=5e-3,
    )
    assert report.passed_zero_drift and report.anchor_max_logit_drift == 0.0
    model.unmount_expert_pack(pack.pack_id)

    base_dir = tmp_path / "base_flat"
    export_flat_v3_and_golden(
        model,
        base_dir,
        sample_text="\x1dS\nsql_plan\x1dE\nSELECT acct_id, SUM(amount) FROM ledger_entries WHERE status = 'POSTED';\n",
        lang_id=LANG_DOMAINS_V3["sql_struct"],
        style_id=4,
        hip_enabled=True,
        plugin_pack=pack,
        anchor_text="The analytical engine computes deterministic structured representations across nested clauses.\n\n",
        anchor_lang_id=LANG_DOMAINS_V3["en"],
        anchor_style_id=0,
    )

    base_sha_before = hashlib.sha256((base_dir / "model.safetensors").read_bytes()).hexdigest()

    # Verify Gate 6 cross-stack numerical parity & zero-alloc bench
    subprocess.run([str(cli_bin), "verify-golden-flat", str(base_dir)], cwd=_REPO_ROOT, check=True)
    subprocess.run([str(cli_bin), "bench-flat", str(base_dir)], cwd=_REPO_ROOT, check=True)

    # Verify end-to-end dcpa-sleep-daemon real-fragment consolidation
    out_sleep_plugin = tmp_path / "plugins" / "personal_sleep.dcpa_expert.safetensors"
    work_dir = tmp_path / "sleep_work"
    res = subprocess.run(
        [
            str(sleep_bin),
            str(base_dir),
            str(out_sleep_plugin),
            "--steps",
            "6",
            "--python",
            sys.executable,
            "--work-dir",
            str(work_dir),
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "promoted=true" in res.stdout, f"Unexpected sleep daemon output: {res.stdout}"
    assert "base_untouched=true" in res.stdout
    assert out_sleep_plugin.is_file()

    # Verify base model.safetensors is physically untouched (read-only guarantee)
    base_sha_after = hashlib.sha256((base_dir / "model.safetensors").read_bytes()).hexdigest()
    assert base_sha_before == base_sha_after

    # Verify promoted sleep plugin loads cleanly in Python & validates against DCPAFlatConfig
    loaded_sleep = load_plugin_pack_safetensors(out_sleep_plugin)
    loaded_sleep.validate(cfg.n_layers, cfg.d_concept, tau_min=cfg.plugin_tau_min)
