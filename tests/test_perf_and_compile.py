"""Performance audit & acceleration tests for DCPA:
  1. `test_pmd_stack_parallel_vs_sequential_equivalence`: C=256 chunked parallel associative
     scan vs byte-by-byte sequential reference at T=1024 with 4 cu_seqlens resets (< 1e-5)
     plus static AST verification of zero byte-level `for t in range(T)` training loops.
  2. `test_compile_checkpoint_portability`: Selective `torch.compile(..., dynamic=False)` on
     Hotspot A (BCG step + 16-doc bucketing) & Hotspot B (SGSC/PMD scans) for 5 steps, save
     clean checkpoint (zero `_orig_mod.` keys), reload into `compile_mode="none"` model with
     `strict=True`, and verify forward diff `< 1e-5`.
  3. `test_step_throughput_benchmark`: Stage-3 worst-case (`4096 B/doc`, `T_total=32768` on M3 /
     `131072` on CUDA) 3-step warmup + 10-step timed benchmark reporting `ms/step` & `KB/s`.
"""

from __future__ import annotations

import ast
import inspect
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.compile_utils import configure_model_compile, get_clean_state_dict, strip_orig_mod_prefix
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.modules.pmd import PMD
from dcpa.modules.sgsc import SGSC
from dcpa.ops.scan import parallel_stack_scan, sequential_stack_scan_ref
from train.curriculum_state_machine import create_dcpa_optimizer
from train.train_dcpa import run_training
from utils.device import (
    amp_autocast,
    configure_cuda_tf32,
    get_deterministic_test_device,
    get_device,
    is_bf16_enabled,
    seed_everything,
    synchronize,
)


def _assert_no_t_byte_loop(fn_obj) -> None:
    """AST-verify that a training forward function contains NO byte-level `for t in range(T)` loop."""
    src = inspect.getsource(fn_obj)
    # Dedent class methods before ast.parse
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Reject any loop whose iterator is range(T) or range(seq_len) or range(total_t)
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                for arg in node.iter.args:
                    if isinstance(arg, ast.Name) and arg.id in ("T", "seq_len", "total_t", "N"):
                        raise AssertionError(
                            f"Forbidden byte-level Python loop `for ... in range({arg.id})` found in {fn_obj.__qualname__}!"
                        )


# =============================================================================
# Test 1: PMD 128B Pushdown Stack Chunked Parallel Scan vs Sequential Reference
# =============================================================================


def test_pmd_stack_parallel_vs_sequential_equivalence():
    """Verify C=256 chunked parallel associative scan matches sequential reference < 1e-5 at T=1024."""
    device = get_deterministic_test_device()
    configure_cuda_tf32(device, allow=False)
    seed_everything(42, device=device)

    # 1. Static AST audit: zero byte-level `for t in range(T)` loops in SGSC/PMD training forward!
    for fn in (
        SGSC.forward,
        SGSC._byte_scan_impl,
        PMD.decode_teacher_forced,
        PMD._motor_scan_impl,
        parallel_stack_scan,
    ):
        _assert_no_t_byte_loop(fn)
    assert PMD.step_single_byte is PMD.step_byte_streaming

    # 2. Construct T=1024 with 4 cu_seqlens document reset boundaries crossing 256-byte chunks
    B, T, d_tag = 1, 1024, 4
    cu_seqlens = [0, 200, 513, 777, 1024]  # 4 varlen documents crossing chunk boundaries 256, 512, 768
    reset_mask = torch.zeros((B, T), device=device, dtype=torch.bool)
    for start in cu_seqlens[:-1]:
        reset_mask[0, start] = True

    logits = torch.randn((B, T, 3), device=device, dtype=torch.float32, requires_grad=True)
    v_tag = torch.tanh(torch.randn((B, T, d_tag), device=device, dtype=torch.float32)).detach().requires_grad_(True)

    probs = torch.softmax(logits, dim=-1)
    p_push, p_pop, p_hold = probs[..., 0], probs[..., 1], probs[..., 2]

    # Case A: xi_init = None (standard packed Ragged training)
    # Warmup 2 calls before timing so reported parallel vs sequential speedup reflects steady-state compute
    for _ in range(2):
        _ = parallel_stack_scan(
            p_push.detach(), p_pop.detach(), p_hold.detach(), v_tag.detach(),
            xi_init=None, reset_mask=reset_mask, chunk_size=256,
        )
    synchronize(device)

    t0 = time.perf_counter()
    xi_par = parallel_stack_scan(
        p_push, p_pop, p_hold, v_tag, xi_init=None, reset_mask=reset_mask, chunk_size=256
    )
    synchronize(device)
    t_par_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    xi_ref = sequential_stack_scan_ref(
        p_push.detach(), p_pop.detach(), p_hold.detach(), v_tag.detach(), xi_init=None, reset_mask=reset_mask
    )
    synchronize(device)
    t_seq_ms = (time.perf_counter() - t1) * 1000.0

    # Also time full forward + backward pass (the real training cost)
    t_b0 = time.perf_counter()
    grad_out = torch.randn_like(xi_par)
    g_logits_par, g_vtag_par = torch.autograd.grad(
        (xi_par * grad_out).sum(), (logits, v_tag), retain_graph=False
    )
    synchronize(device)
    t_par_fb_ms = t_par_ms + (time.perf_counter() - t_b0) * 1000.0

    max_diff_zero_init = float((xi_par.detach() - xi_ref).abs().max().item())
    assert max_diff_zero_init < 1e-5, f"Parallel vs sequential stack max_abs_diff={max_diff_zero_init:.3e} >= 1e-5"

    logits_r = logits.detach().clone().requires_grad_(True)
    v_tag_r = v_tag.detach().clone().requires_grad_(True)
    t_rb0 = time.perf_counter()
    probs_r = torch.softmax(logits_r, dim=-1)
    xi_ref_grad = sequential_stack_scan_ref(
        probs_r[..., 0], probs_r[..., 1], probs_r[..., 2], v_tag_r, xi_init=None, reset_mask=reset_mask
    )
    g_logits_ref, g_vtag_ref = torch.autograd.grad(
        (xi_ref_grad * grad_out).sum(), (logits_r, v_tag_r), retain_graph=False
    )
    synchronize(device)
    t_seq_fb_ms = (time.perf_counter() - t_rb0) * 1000.0
    grad_diff_logits = float((g_logits_par - g_logits_ref).abs().max().item())
    grad_diff_vtag = float((g_vtag_par - g_vtag_ref).abs().max().item())
    assert grad_diff_logits < 1e-5, f"Grad logits diff {grad_diff_logits:.3e} >= 1e-5"
    assert grad_diff_vtag < 1e-5, f"Grad v_tag diff {grad_diff_vtag:.3e} >= 1e-5"

    # Case B: non-zero xi_init with 4 doc resets
    xi_init = torch.randn((B, 16, d_tag), device=device, dtype=torch.float32) * 0.5
    xi_par_init = parallel_stack_scan(
        p_push.detach(), p_pop.detach(), p_hold.detach(), v_tag.detach(),
        xi_init=xi_init, reset_mask=reset_mask, chunk_size=256,
    )
    xi_ref_init = sequential_stack_scan_ref(
        p_push.detach(), p_pop.detach(), p_hold.detach(), v_tag.detach(),
        xi_init=xi_init, reset_mask=reset_mask,
    )
    max_diff_with_init = float((xi_par_init - xi_ref_init).abs().max().item())
    assert max_diff_with_init < 1e-5, f"With xi_init max_abs_diff={max_diff_with_init:.3e} >= 1e-5"

    print(
        f"\n[Test 1: PMD Chunked C=256 Stack Scan] T={T}, 4 docs={cu_seqlens} | "
        f"fwd max_abs_diff={max_diff_zero_init:.3e} (with xi_init={max_diff_with_init:.3e}) | "
        f"bwd grad_diff=(logits:{grad_diff_logits:.3e}, v_tag:{grad_diff_vtag:.3e}) | "
        f"fwd+bwd: parallel={t_par_fb_ms:.2f}ms vs sequential={t_seq_fb_ms:.2f}ms "
        f"({t_seq_fb_ms / max(t_par_fb_ms, 1e-6):.2f}x speedup)"
    )


# =============================================================================
# Test 2: Selective torch.compile Checkpoint Portability (Zero `_orig_mod.`)
# =============================================================================


def test_compile_checkpoint_portability(tmp_path: Path):
    """Run 5 steps with compiled Hotspot A & B, save Checkpoint, load into `compile_mode='none'`
    with `strict=True`, assert zero `_orig_mod.` keys and forward diff `< 1e-5`.
    """
    device = get_deterministic_test_device()
    configure_cuda_tf32(device, allow=False)

    # First verify auto-fallback on non-CUDA defaults to 'none' cleanly
    if device.type != "cuda":
        m_fb = DCPAModel(DCPAConfig.micro()).to(device)
        rep_fb = configure_model_compile(m_fb, compile_mode="auto", device=device, allow_cpu_compile=False)
        assert rep_fb.effective_mode == "none"
        assert not hasattr(m_fb.bcg_pc, "_orig_mod")

    # Now run 5 real steps through run_training with selective torch.compile genuinely ACTIVE!
    # (Inductor on CUDA; aot_eager on CPU so real OptimizedModule `_orig_mod.` wrapping is tested!)
    ckpt_dir = tmp_path / "compiled_ckpts"
    observed_compiled_state: dict[str, bool] = {}

    def _inspect_compiled(_step: int, model: DCPAModel, _metrics: dict[str, float]) -> None:
        if "checked" not in observed_compiled_state:
            observed_compiled_state["checked"] = True
            observed_compiled_state["has_orig_mod"] = hasattr(model.bcg_pc, "_orig_mod")
            observed_compiled_state["raw_has_orig_mod_key"] = any(
                "_orig_mod" in k for k in model.state_dict().keys()
            )
            observed_compiled_state["doc_bucketing"] = bool(model.enable_doc_bucketing)
            observed_compiled_state["clean_has_orig_mod_key"] = any(
                "_orig_mod" in k for k in get_clean_state_dict(model).keys()
            )

    summary = run_training(
        config_path=REPO_ROOT / "configs" / "dcpa_48m_train.yaml",
        preset="micro",
        total_steps=5,
        stage_boundaries=(0, 2, 5, 6, 7),  # crosses Stage 1 -> Stage 2 so Hotspot A (BCG-PC) executes!
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        ckpt_dir=ckpt_dir,
        logs_dir=tmp_path / "logs",
        export_dir=tmp_path / "export",
        local_cache_dir=tmp_path / "nvme_cache",
        save_every_steps=5,
        milestone_every_steps=0,
        probe_every_steps=0,
        console_every_steps=5,
        resume="none",
        device_str="cuda" if device.type == "cuda" else "cpu",
        compile_mode="auto",
        allow_cpu_compile=True,
        deterministic=True,
        seed=123,
        export_final=True,
        step_hook=_inspect_compiled,
    )
    assert summary.completed_steps == 5
    assert observed_compiled_state.get("has_orig_mod") is True, "Hotspot A (bcg_pc) was not wrapped by torch.compile!"
    assert observed_compiled_state.get("raw_has_orig_mod_key") is True, "Expected raw state_dict to contain _orig_mod."
    assert observed_compiled_state.get("clean_has_orig_mod_key") is False, "get_clean_state_dict leaked _orig_mod.!"
    assert observed_compiled_state.get("doc_bucketing") is True
    assert torch._dynamo.config.cache_size_limit >= 64

    ckpt_path = ckpt_dir / "last_healthy.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    orig_mod_keys = [k for k in sd.keys() if "_orig_mod" in k]
    assert not orig_mod_keys, f"Checkpoint polluted with _orig_mod keys: {orig_mod_keys[:5]}"

    # Instantiate a brand-new `compile_mode="none"` model and load with strict=True!
    cfg = DCPAConfig(**ckpt["config"])
    model_uncompiled = DCPAModel(cfg).to(device)
    load_res = model_uncompiled.load_state_dict(sd, strict=True)
    assert not load_res.missing_keys and not load_res.unexpected_keys

    # Also build a compiled copy loaded from the exact same clean state_dict and verify forward diff < 1e-5
    model_compiled = DCPAModel(cfg).to(device)
    model_compiled.load_state_dict(sd, strict=True)
    configure_model_compile(model_compiled, compile_mode="auto", device=device, allow_cpu_compile=True)

    # Verify both models produce matching eval outputs (< 1e-5) on packed multi-doc Ragged input
    # including testing V=5 docs bucketed -> 16 vs unbucketed V=5!
    model_uncompiled.eval()
    model_compiled.eval()
    seed_everything(99, device=device)
    # 9 docs > 8 so compiled model exercises the V_pad=16 bucketed Hotspot A path!
    doc_lens = [32, 48, 40, 36, 52, 44, 38, 42, 46]
    total_t = sum(doc_lens)
    cu = torch.tensor([0] + [sum(doc_lens[: i + 1]) for i in range(len(doc_lens))], dtype=torch.int32, device=device)
    x_raw = torch.randint(32, 126, (1, total_t), dtype=torch.uint8, device=device)
    lang_ids = torch.tensor([i % 5 for i in range(len(doc_lens))], dtype=torch.long, device=device)

    with torch.no_grad():
        out_un = model_uncompiled(
            raw_bytes=x_raw, lang_ids=lang_ids, stage="B", tau_s=0.35, tau_m=0.5, doc_cu_seqlens=cu
        )
        out_co = model_compiled(
            raw_bytes=x_raw, lang_ids=lang_ids, stage="B", tau_s=0.35, tau_m=0.5, doc_cu_seqlens=cu
        )

    logits_diff = float((out_un.pmd.byte_logits - out_co.pmd.byte_logits).abs().max().item())
    loss_diff = abs(float(out_un.loss_total.item()) - float(out_co.loss_total.item()))
    cslow_diff = float((out_un.sgsc.c_slow - out_co.sgsc.c_slow).abs().max().item())

    # Double-check strip_orig_mod_prefix idempotency
    synthetic_polluted = {"bcg_pc._orig_mod.w_e.weight": torch.zeros(1), "_orig_mod.sgsc.h0": torch.zeros(1)}
    assert set(strip_orig_mod_prefix(synthetic_polluted).keys()) == {"bcg_pc.w_e.weight", "sgsc.h0"}

    print(
        f"\n[Test 2: Compile Portability] strict=True load OK (0 _orig_mod keys, {len(sd)} tensors) | "
        f"bucketed(9->16) compiled vs uncompiled: logits_diff={logits_diff:.3e}, "
        f"loss_diff={loss_diff:.3e}, c_slow_diff={cslow_diff:.3e}"
    )
    assert logits_diff < 1e-5, f"Compiled vs uncompiled logits diff {logits_diff:.3e} >= 1e-5!"
    assert loss_diff < 1e-5, f"Compiled vs uncompiled loss diff {loss_diff:.3e} >= 1e-5!"
    assert cslow_diff < 1e-5, f"Compiled vs uncompiled c_slow diff {cslow_diff:.3e} >= 1e-5!"


# =============================================================================
# Test 3: Stage-3 Worst-Case Throughput Benchmark (4096 B/doc, 3 warmup + 10 timed)
# =============================================================================


def test_step_throughput_benchmark():
    """Stage-3 worst-case benchmark: T_total=32768 (M3 local) or 131072 (CUDA), 4096 B/doc,
    3 warmup steps + 10 timed steps, reporting ms/step and KB/s.
    """
    device = get_device("auto", deterministic=True)  # CUDA if available, else fast deterministic CPU on M3
    use_bf16 = is_bf16_enabled(device, enabled=True)
    configure_cuda_tf32(device, allow=True)

    is_cuda = device.type == "cuda"
    doc_len = 4096
    t_total = 131072 if is_cuda else 32768
    n_docs = t_total // doc_len  # 32 docs on CUDA, 8 docs on local M3

    # Full 48M prototype on CUDA; compact micro on local M3 CPU so pytest finishes swiftly
    cfg = DCPAConfig.proto_48m() if is_cuda else DCPAConfig.micro()
    seed_everything(42, device=device)
    model = DCPAModel(cfg).to(device)
    # Calibrate LIF threshold to Stage-3 operating point (rho ~ 9.5x after Stages 1-2 L_rate anchoring)
    with torch.no_grad():
        target_theta = 7.0
        import math
        raw_val = math.log(math.expm1(max(target_theta - cfg.lif_theta_min, 1e-3))) / cfg.lif_theta_scale
        model.sgsc.theta_raw.fill_(raw_val)
    optimizer = create_dcpa_optimizer(model)

    rep = configure_model_compile(
        model,
        compile_mode="auto" if is_cuda else "none",
        device=device,
    )

    # Construct fixed Stage-3 synthetic Rust/code packed batch: n_docs x 4096 B (exact T_total)
    pattern = b"fn compute(xs: &[i32]) -> i32 { let mut s = 0; for &x in xs { if x > 0 { s += x; } } s }\n"
    doc_bytes = (pattern * ((doc_len // len(pattern)) + 1))[:doc_len]
    full_bytes = doc_bytes * n_docs
    x_raw = torch.tensor(list(full_bytes), dtype=torch.uint8, device=device).unsqueeze(0)
    cu = torch.arange(0, t_total + 1, doc_len, dtype=torch.int32, device=device)
    lang_ids = torch.full((n_docs,), 4, dtype=torch.long, device=device)

    model.train()
    n_warmup = 3
    n_timed = 10
    step_times_ms: list[float] = []

    for step_i in range(n_warmup + n_timed):
        synchronize(device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                raw_bytes=x_raw,
                lang_ids=lang_ids,
                stage="C",
                tau_s=0.25,
                tau_m=0.25,
                use_broca_cq=(step_i % 2 == 1),
                doc_cu_seqlens=cu,
            )
        out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if step_i >= n_warmup:
            step_times_ms.append(dt_ms)

    avg_ms = sum(step_times_ms) / len(step_times_ms)
    kb_per_sec = (t_total / 1024.0) / (avg_ms / 1000.0)
    assert all(t > 0.0 for t in step_times_ms)
    assert torch.isfinite(out.loss_total)

    print(
        f"\n[Test 3: Stage-3 Throughput Benchmark] device={device.type} | compile={rep.effective_mode} | "
        f"model={'proto_48m' if is_cuda else 'micro'} | T_total={t_total:,}B ({n_docs}x{doc_len}B) | "
        f"rho={out.metrics['compression_ratio']:.2f}x | "
        f"avg={avg_ms:.2f} ms/step (min={min(step_times_ms):.2f}, max={max(step_times_ms):.2f}) | "
        f"throughput={kb_per_sec:.2f} KB/s | final_loss={float(out.loss_total.item()):.4f}"
    )
