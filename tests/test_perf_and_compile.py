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
    capture_rng_state,
    configure_cuda_tf32,
    get_deterministic_test_device,
    get_device,
    is_bf16_enabled,
    restore_rng_state,
    seed_everything,
    select_hardware_tier,
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
        config_path=REPO_ROOT / "configs" / "v1" / "dcpa_48m_train.yaml",
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
        f"avg={avg_ms:.2f} ms/step (min={min(step_times_ms):.2f}, max={min(step_times_ms) if not step_times_ms else max(step_times_ms):.2f}) | "
        f"throughput={kb_per_sec:.2f} KB/s | final_loss={float(out.loss_total.item()):.4f}"
    )


# =============================================================================
# Test 4: G4 96GB Tier + Watermark Checkpointing + Compacted MoE + Async Prefetch Equivalence
# =============================================================================


def test_g4_96g_tier_watermark_compacted_moe_and_prefetch_equivalence(tmp_path: Path):
    """Verify 96GB G4 hardware tier, lossless compacted 4-BMM Pondering/MoE forward+backward
    equivalence (< 1e-5), watermark-bypassed vs checkpointed gradient identity, and deterministic
    async prefetcher cursor/rollback equivalence.
    """
    import copy
    import numpy as np
    from data.ragged_dataloader import IDX_DTYPE
    from train.curriculum_state_machine import CurriculumController
    from train.train_dcpa import _DynamicRaggedCursorStream

    # 1. Verify 96GB G4 Blackwell tier contract vs 80GB H100/A100 contract
    g4 = select_hardware_tier("cuda", vram_gb=95.0, device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert g4.tier == "blackwell_or_h200_96g"
    assert (g4.micro_batch_bytes, g4.grad_accum_steps, g4.target_tokens_bytes_per_step, g4.use_bf16) == (
        262_144,
        1,
        262_144,
        True,
    )
    h100 = select_hardware_tier("cuda", vram_gb=80.0, device_name="NVIDIA H100 80GB HBM3")
    assert (h100.tier, h100.micro_batch_bytes, h100.target_tokens_bytes_per_step) == (
        "h100_or_a100_80g",
        131_072,
        131_072,
    )

    device = get_deterministic_test_device()
    configure_cuda_tf32(device, allow=False)
    seed_everything(202611, device=device)

    # 2. Verify compiled forward_fast_dispatch + compacted 8-expert Top-2 MoE (N=80 -> c_cap < N!)
    #    matches baseline BCGPC.forward in BOTH forward and backward (< 1e-5)!
    cfg8 = DCPAConfig(
        d_s=64,
        d_concept=128,
        k_slots=8,
        d_r=32,
        d_merge=64,
        d_h=512,
        s_active=16,
        d_l=64,
        num_experts=8,
        top_k_experts=2,
        expert_hidden=192,
        l_max=8,
    )
    bcg_ref = DCPAModel(cfg8).bcg_pc.to(device=device, dtype=torch.float32)
    bcg_fast = copy.deepcopy(bcg_ref).to(device=device, dtype=torch.float32)
    bcg_fast.enable_compiled_subkernels(dynamic=False, backend="aot_eager")

    B_big = 80  # > 48 so capacity-compacted [E=8, C_cap, D] path is genuinely exercised!
    st0 = bcg_ref.init_state(B_big, device=device, dtype=torch.float32)
    c_in_base = torch.randn((B_big, cfg8.d_concept), device=device, dtype=torch.float32) * 1.5
    prec_in = torch.ones((B_big, cfg8.d_concept), device=device, dtype=torch.float32)
    ach_in = torch.zeros((B_big,), device=device, dtype=torch.float32)
    tau_t = torch.tensor(0.35, device=device, dtype=torch.float32)
    mix_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    am_in = torch.arange(B_big, device=device) < 73  # partial active mask (73/80 active)
    trig_in = (torch.arange(B_big, device=device) == 5).float()
    gumbel_u = torch.rand((B_big, cfg8.num_merge_pairs), device=device, dtype=torch.float32)

    c_ref = c_in_base.detach().clone().requires_grad_(True)
    c_fast = c_in_base.detach().clone().requires_grad_(True)

    bcg_ref.train()
    bcg_fast.train()
    out_ref = bcg_ref(
        st0.slots, st0.edges, st0.salience, c_ref, prec_in, ach_in, tau_t, am_in, mix_t,
        st0.slot0_pinned, trig_in, True, False, None, None, None, None, gumbel_u,
    )
    out_fast = bcg_fast.forward_fast_dispatch(
        st0.slots, st0.edges, st0.salience, c_fast, prec_in, ach_in, tau_t, am_in, mix_t,
        st0.slot0_pinned, trig_in, use_gumbel=True, update_c0_ema=False, gumbel_u=gumbel_u,
    )

    for idx_t, name_t in ((0, "z_star"), (2, "c_hat"), (5, "ponder_steps"), (6, "s_new"), (8, "omega_new"), (16, "moe_bal")):
        diff_t = float((out_ref[idx_t].float() - out_fast[idx_t].float()).abs().max().item())
        assert diff_t < 1e-5, f"Compacted fast dispatch {name_t} forward diff {diff_t:.3e} >= 1e-5!"

    w_loss = torch.randn_like(out_ref[0])
    loss_r = (out_ref[0] * w_loss).sum() + out_ref[5].sum() * 0.1 + out_ref[6].sum() * 0.05 + out_ref[16]
    loss_f = (out_fast[0] * w_loss).sum() + out_fast[5].sum() * 0.1 + out_fast[6].sum() * 0.05 + out_fast[16]
    loss_r.backward()
    loss_f.backward()

    c_grad_diff = float((c_ref.grad - c_fast.grad).abs().max().item())
    assert c_grad_diff < 1e-5, f"Input c_fused grad diff {c_grad_diff:.3e} >= 1e-5!"
    max_param_grad_diff = 0.0
    for (pname, p_r), (_, p_f) in zip(bcg_ref.named_parameters(), bcg_fast.named_parameters()):
        if p_r.grad is not None and p_f.grad is not None:
            gd = float((p_r.grad - p_f.grad).abs().max().item())
            max_param_grad_diff = max(max_param_grad_diff, gd)
            assert gd < 1e-4, f"Param {pname} grad diff {gd:.3e} >= 1e-4!"

    # 3. Verify full DCPAModel watermark-uncheckpointed (watermark=52.0GB) vs forced-checkpointed (watermark=0.0GB)
    #    produce BIT-IDENTICAL forward losses and parameter gradients!
    model_ckpt = DCPAModel(DCPAConfig.micro()).to(device=device, dtype=torch.float32)
    model_nockpt = copy.deepcopy(model_ckpt).to(device=device, dtype=torch.float32)
    model_ckpt.bcg_checkpoint_vram_watermark_gb = 0.0     # forces torch_checkpoint on every m_step
    model_nockpt.bcg_checkpoint_vram_watermark_gb = 52.0  # bypasses torch_checkpoint below 52 GB

    raw_doc = b"def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n" * 2
    x_pack = torch.tensor(list(raw_doc), dtype=torch.uint8, device=device).unsqueeze(0)
    cu_pack = torch.tensor([0, len(raw_doc) // 2, len(raw_doc)], dtype=torch.int32, device=device)
    lids_pack = torch.tensor([3, 3], dtype=torch.long, device=device)

    seed_everything(777, device=device)
    rng_s = capture_rng_state()
    out_c = model_ckpt(raw_bytes=x_pack, lang_ids=lids_pack, stage="B", tau_s=0.35, tau_m=0.4, doc_cu_seqlens=cu_pack)
    out_c.loss_total.backward()
    rng_after_c = capture_rng_state()

    restore_rng_state(rng_s)
    out_nc = model_nockpt(raw_bytes=x_pack, lang_ids=lids_pack, stage="B", tau_s=0.35, tau_m=0.4, doc_cu_seqlens=cu_pack)
    out_nc.loss_total.backward()
    rng_after_nc = capture_rng_state()

    wm_loss_diff = abs(float(out_c.loss_total.item()) - float(out_nc.loss_total.item()))
    assert wm_loss_diff < 1e-6, f"Watermark uncheckpointed vs checkpointed loss diff {wm_loss_diff:.3e}"
    assert torch.equal(rng_after_c["torch_cpu_rng_state"], rng_after_nc["torch_cpu_rng_state"])
    for (pn, pc), (_, pnc) in zip(model_ckpt.named_parameters(), model_nockpt.named_parameters()):
        if pc.grad is not None and pnc.grad is not None:
            g_diff = float((pc.grad - pnc.grad).abs().max().item())
            assert g_diff < 1e-6, f"Watermark grad mismatch on {pn}: {g_diff:.3e}"

    # 4. Verify async background prefetcher thread produces bit-identical batches across stage change & rollback
    bin_p = tmp_path / "train.bin"
    idx_p = tmp_path / "train.idx"
    docs = [f"Doc {i}: The quick brown fox jumps over {i * 17} lazy dogs.\n".encode("utf-8") for i in range(32)]
    bin_p.write_bytes(b"".join(docs))
    recs = np.zeros(len(docs), dtype=IDX_DTYPE)
    off = 0
    for i, db in enumerate(docs):
        recs[i] = (off, len(db), i % 5, 0, 0, i % 3)
        off += len(db)
    recs.tofile(idx_p)

    ctrl = CurriculumController(stage_boundaries=(0, 2, 5, 7, 10))
    s_sync = _DynamicRaggedCursorStream(bin_p, idx_p, seed=99, prefetch_depth=0)
    s_async = _DynamicRaggedCursorStream(bin_p, idx_p, seed=99, prefetch_depth=0)
    # Force ThreadPoolExecutor activation on s_async to test threaded lookahead + rollback deterministically
    from concurrent.futures import ThreadPoolExecutor
    s_async._prefetch_depth = 2
    s_async._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_prefetch")
    try:
        saved_cur = None
        for step_k in range(8):
            st_c = ctrl.get(step_k)
            b_s = s_sync.next_batch(target_batch_bytes=256, state=st_c)
            b_a = s_async.next_batch(target_batch_bytes=256, state=st_c)
            assert torch.equal(b_s["x_bytes"], b_a["x_bytes"])
            assert s_sync.cursor == s_async.cursor
            if step_k == 3:
                saved_cur = dict(s_sync.cursor)
        assert saved_cur is not None
        s_sync.restore_cursor(saved_cur)
        s_async.restore_cursor(saved_cur)
        for step_k in range(4, 8):
            st_c = ctrl.get(step_k)
            b_s = s_sync.next_batch(target_batch_bytes=256, state=st_c)
            b_a = s_async.next_batch(target_batch_bytes=256, state=st_c)
            assert torch.equal(b_s["x_bytes"], b_a["x_bytes"])
            assert s_sync.cursor == s_async.cursor
    finally:
        s_sync.close()
        s_async.close()

    print(
        f"\n[Test 4: G4 96GB Suite] tier={g4.tier} ({g4.micro_batch_bytes:,}B) | "
        f"compacted_moe_grad_diff={max_param_grad_diff:.2e} | "
        f"watermark_loss_diff={wm_loss_diff:.2e} | async_prefetch_rollback=EXACT"
    )
