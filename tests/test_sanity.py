"""Deterministic unit sanity suite for DCPA (Iron Law #1.1 & Colab Cell 4).

Executes on pure `torch.device("cpu")` + `float32` whenever CUDA is unavailable
(completely eliminating Apple MPS floating-point rounding nondeterminism), and on
CUDA `float32` in Google Colab for cloud operator verification.
"""

from __future__ import annotations

import ast
import contextlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.ops.scan import parallel_diag_scan, parallel_stack_scan, sequential_diag_scan_ref
from utils.device import (
    amp_autocast,
    capture_rng_state,
    get_deterministic_test_device,
    is_cuda_available,
    resolve_micro_batch_plan,
    restore_rng_state,
    select_hardware_tier,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_device_and_four_tier_hardware_contract():
    dev = get_deterministic_test_device()
    if is_cuda_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu", "Deterministic sanity tests must NEVER run on MPS"

    # Verify all 4 hardware auto-sharding tiers from Iron Law #1.4
    h100 = select_hardware_tier("cuda", vram_gb=80.0, device_name="NVIDIA H100 80GB HBM3")
    assert h100.micro_batch_bytes == 131_072
    assert h100.grad_accum_steps == 1
    assert h100.use_bf16 is True

    a100_80 = select_hardware_tier("cuda", vram_gb=79.2, device_name="NVIDIA A100-SXM4-80GB")
    assert a100_80.micro_batch_bytes == 131_072
    assert a100_80.grad_accum_steps == 1

    a100_40 = select_hardware_tier("cuda", vram_gb=40.0, device_name="NVIDIA A100-SXM4-40GB")
    assert a100_40.micro_batch_bytes == 65_536
    assert a100_40.grad_accum_steps == 2
    assert a100_40.use_bf16 is True

    l4 = select_hardware_tier("cuda", vram_gb=24.0, device_name="NVIDIA L4")
    assert l4.micro_batch_bytes == 32_768
    assert l4.grad_accum_steps == 4
    assert l4.use_bf16 is True

    mps_local = select_hardware_tier("mps", vram_gb=0.0, target_tokens_bytes_per_step=8192)
    assert mps_local.micro_batch_bytes == 8192
    assert mps_local.grad_accum_steps == 1
    assert mps_local.use_bf16 is False

    cpu_local = select_hardware_tier("cpu", vram_gb=0.0, target_tokens_bytes_per_step=8192)
    assert cpu_local.micro_batch_bytes == 8192
    assert cpu_local.grad_accum_steps == 1
    assert cpu_local.use_bf16 is False

    # Verify micro-batch planning for seq_len=2048
    mbs, acc, total = resolve_micro_batch_plan(a100_40, seq_len=2048)
    assert (mbs, acc, total) == (32, 2, 131_072)

    # Verify AMP autocast safely falls back to nullcontext on CPU & MPS
    assert isinstance(amp_autocast(torch.device("cpu")), contextlib.nullcontext)
    assert isinstance(amp_autocast(torch.device("mps")), contextlib.nullcontext)


def test_cross_platform_rng_capture_and_restore():
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)

    snap = capture_rng_state()
    expected_py = random.random()
    expected_np = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())

    # Simulate loading a checkpoint produced on a different platform (missing CUDA/MPS states)
    portable_snap = dict(snap)
    portable_snap["cuda_rng_state_all"] = None
    portable_snap["mps_rng_state"] = None
    restore_rng_state(portable_snap)

    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np
    assert float(torch.rand(1).item()) == expected_torch


def test_no_bare_cuda_calls_outside_device_module():
    """Iron Law #1: forbid bare `.cuda()`, `get_device_properties(0)`, or raw `autocast('cuda'` outside utils/device.py."""
    scan_dirs = [
        REPO_ROOT / "utils",
        REPO_ROOT / "data",
        REPO_ROOT / "train",
        REPO_ROOT / "tests",
        REPO_ROOT / "gates",
        REPO_ROOT / "dcpa-train-py" / "dcpa",
    ]
    files: list[Path] = [REPO_ROOT / "smoke_overfit_64.py"]
    for d in scan_dirs:
        if d.exists():
            files.extend(p for p in d.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts)

    allowed_device_file = (REPO_ROOT / "utils" / "device.py").resolve()
    self_file = Path(__file__).resolve()

    violations: list[str] = []
    for path in files:
        resolved = path.resolve()
        if resolved in (allowed_device_file, self_file):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr == "cuda":
                    violations.append(f"{path}:{node.lineno}: bare .cuda() call")
                elif attr == "get_device_properties":
                    violations.append(f"{path}:{node.lineno}: bare get_device_properties() call")
                elif attr == "autocast":
                    violations.append(f"{path}:{node.lineno}: bare autocast() call (use utils.device.amp_autocast)")

    assert not violations, "Iron Law #1 violations found:\n" + "\n".join(violations)


def test_sgsc_strict_causality_and_utf8_integrity_t256():
    """Verify exact zero-leakage causality and 100% intact UTF-8 segmentation at T=256 (float32)."""
    device = get_deterministic_test_device()
    torch.manual_seed(2026)

    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    model.eval()

    B, T = 2, 256
    x_orig = torch.randint(0, 256, (B, T), dtype=torch.uint8, device=device)
    with torch.no_grad():
        out_orig = model.sgsc(x_orig, tau_s=0.5)

    for tamper_t in (0, 1, 31, 128, 255):
        x_tamp = x_orig.clone()
        x_tamp[:, tamper_t:] = ((x_tamp[:, tamper_t:].int() + 41) % 256).to(torch.uint8)
        with torch.no_grad():
            out_tamp = model.sgsc(x_tamp, tau_s=0.5)
        max_diff = (
            out_orig.prior_probs[:, : tamper_t + 1, :] - out_tamp.prior_probs[:, : tamper_t + 1, :]
        ).abs().max().item()
        assert max_diff == 0.0, f"Causality leak detected at t={tamper_t}: diff={max_diff}"

    # Verify UTF-8 multi-byte character completion guard across mixed ZH/EN/JA/Emoji/Code
    mixed = (
        "那只狴犴正趴在屋脊上吞云吐雾！🦀🔥🚀 "
        "日本語のテキスト処理テスト。 "
        "fn main() -> Result<(), &str> { Ok(()) } "
        "def solve(x: int) -> int: return x * 2"
    )
    raw_py = mixed.encode("utf-8")
    x_utf8 = torch.tensor(list(raw_py), dtype=torch.uint8, device=device).unsqueeze(0)
    with torch.no_grad():
        out_utf8 = model.sgsc(x_utf8, tau_s=0.35)
    cu = out_utf8.cu_seqlens.detach().cpu().tolist()
    pieces = []
    for i in range(len(cu) - 1):
        seg = raw_py[cu[i] : cu[i + 1]]
        assert len(seg) > 0
        pieces.append(seg.decode("utf-8", errors="strict"))
    assert "".join(pieces) == mixed


def test_ste_surrogate_gradients_and_parallel_scans_t256():
    """Verify exact scan equivalence and finite non-zero STE gradients at T=256 (float32)."""
    device = get_deterministic_test_device()
    torch.manual_seed(777)

    # 1. Parallel diagonal scan vs sequential reference at T=256
    B, T, D = 2, 256, 64
    a = torch.sigmoid(torch.randn(B, T, D, device=device, dtype=torch.float32))
    b = torch.randn(B, T, D, device=device, dtype=torch.float32) * 0.2
    h0 = torch.randn(D, device=device, dtype=torch.float32) * 0.1

    a1, b1, h0_1 = a.clone().requires_grad_(True), b.clone().requires_grad_(True), h0.clone().requires_grad_(True)
    a2, b2, h0_2 = a.clone().requires_grad_(True), b.clone().requires_grad_(True), h0.clone().requires_grad_(True)
    h_par = parallel_diag_scan(a1, b1, h0=h0_1)
    h_seq = sequential_diag_scan_ref(a2, b2, h0=h0_2)
    assert (h_par - h_seq).abs().max().item() < 1e-5

    g_out = torch.randn_like(h_par)
    (h_par * g_out).sum().backward()
    (h_seq * g_out).sum().backward()
    assert (a1.grad - a2.grad).abs().max().item() < 1e-5
    assert (b1.grad - b2.grad).abs().max().item() < 1e-5

    # 2. Parallel 128B stack scan determinism at T=256
    probs = torch.softmax(torch.randn(2, T, 3, device=device, dtype=torch.float32), dim=-1)
    v_tag = torch.tanh(torch.randn(2, T, 4, device=device, dtype=torch.float32))
    xi1 = parallel_stack_scan(probs[..., 0], probs[..., 1], probs[..., 2], v_tag)
    xi2 = parallel_stack_scan(probs[..., 0], probs[..., 1], probs[..., 2], v_tag)
    assert torch.allclose(xi1, xi2, atol=1e-6)

    # 3. End-to-end STE gradient flow across all 4 stages on T=256
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    raw = torch.randint(32, 126, (2, T), dtype=torch.uint8, device=device)
    lang = torch.tensor([1, 4], dtype=torch.long, device=device)

    for stage, use_cq in (("A", False), ("B", False), ("C", True), ("D", True)):
        model.set_stage_freeze(stage)
        model.zero_grad(set_to_none=True)
        out = model(raw, lang, stage=stage, tau_s=0.5, tau_m=0.5, use_broca_cq=use_cq)
        assert torch.isfinite(out.loss_total), f"Non-finite loss in stage {stage}"
        out.loss_total.backward()
        if stage != "D":
            assert model.sgsc.theta_raw.grad is not None
            assert torch.isfinite(model.sgsc.theta_raw.grad).all()
            assert model.sgsc.theta_raw.grad.abs().item() > 0.0, f"Zero STE grad on theta_raw in stage {stage}"
        else:
            assert model.sgsc.embed.weight.grad is None
            assert model.sgsc.theta_raw.grad is None
            assert model.bcg_pc.w_omega.weight.grad is not None
            assert torch.isfinite(model.bcg_pc.w_omega.weight.grad).all()
