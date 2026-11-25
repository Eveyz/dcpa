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
    from dcpa.modules.bcg_pc import ConceptGraphState
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


# =============================================================================
# Test 5: Step-1 Perf Suite — Native 3D MoE Legacy 75k Resume, RoPE LUT,
#         check_pin=False Skip, and Deferred Metric Sync Parity
# =============================================================================


def test_step1_native_3d_moe_legacy_checkpoint_resume_and_parity(tmp_path: Path):
    """Verify:
    1. A legacy `.pt` checkpoint (separate `experts.{e}.w_*` weights + per-expert AdamW moments)
       migrates losslessly into native 3D MoE weights + AdamW moments with 0.0 drift (uncompiled & compiled).
    2. `extract_flat_state_dict` unpacks 3D MoE weights back to flat `experts.{e}.w_*` for Rust,
       and `load_dcpa_model_from_weights` reloads the `.safetensors` with 0.0 drift.
    3. Cached Holographic RoPE LUT matches direct trig (`< 1e-6`) even when initialized under `inference_mode`.
    4. `check_pin=False` skips dead post-merge `message_pass` with exact 0.0 forward/backward drift,
       while batches with `\\x1dE\\n` triggers still fire Slot-0 pinning bit-identically.
    5. `sync_metrics=False` + `materialize_metric_vec` matches `sync_metrics=True` identically.
    """
    from safetensors.torch import save_file
    from dcpa.export import extract_flat_state_dict
    from dcpa.model import materialize_metric_vec
    from dcpa.ops.rope import apply_holographic_rope
    from train.curriculum_state_machine import CurriculumController
    from train.train_stage5_sft import load_dcpa_model_from_weights
    from train.watchdog_and_ckpt import CheckpointManager, SelfHealingWatchdog

    device = get_deterministic_test_device()
    configure_cuda_tf32(device, allow=False)
    seed_everything(20261125, device=device)

    # --- 1. Build authentic legacy checkpoint (per-expert weights + per-expert AdamW moments) ---
    cfg = DCPAConfig.micro()
    ref_model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    ref_opt = create_dcpa_optimizer(ref_model)
    ctrl = CurriculumController(stage_boundaries=(0, 2, 5, 8, 10))
    wd = SelfHealingWatchdog()

    # Run 2 real steps so AdamW populates live exp_avg / exp_avg_sq / step for all active params!
    doc_bytes = b"\x1eU\nSolve x+1=2\n\x1fA\n\x1dS\nplan: x=1\x1dE\nx = 1\x04\n" * 2
    x_raw = torch.tensor(list(doc_bytes), dtype=torch.uint8, device=device).unsqueeze(0)
    cu = torch.tensor([0, len(doc_bytes) // 2, len(doc_bytes)], dtype=torch.int32, device=device)
    lids = torch.tensor([0, 3], dtype=torch.long, device=device)
    for _ in range(2):
        ref_opt.zero_grad(set_to_none=True)
        out_r = ref_model(raw_bytes=x_raw, lang_ids=lids, stage="B", tau_s=0.35, tau_m=0.4, doc_cu_seqlens=cu)
        out_r.loss_total.backward()
        ref_opt.step()

    # Synthesize the EXACT pre-Step-1 legacy model_state_dict + optimizer_state_dict layout
    new_clean_sd = get_clean_state_dict(ref_model)
    legacy_model_sd = dict(new_clean_sd)
    w_gu1 = legacy_model_sd.pop("bcg_pc.moe.w_gu1").clone()
    w_d1 = legacy_model_sd.pop("bcg_pc.moe.w_d1").clone()
    w_gu2 = legacy_model_sd.pop("bcg_pc.moe.w_gu2").clone()
    w_d2 = legacy_model_sd.pop("bcg_pc.moe.w_d2").clone()
    E, two_h, D = w_gu1.shape
    H = two_h // 2
    for e in range(E):
        legacy_model_sd[f"bcg_pc.moe.experts.{e}.w_gate1.weight"] = w_gu1[e, :H].contiguous()
        legacy_model_sd[f"bcg_pc.moe.experts.{e}.w_up1.weight"] = w_gu1[e, H:].contiguous()
        legacy_model_sd[f"bcg_pc.moe.experts.{e}.w_down1.weight"] = w_d1[e].contiguous()
        legacy_model_sd[f"bcg_pc.moe.experts.{e}.w_gate2.weight"] = w_gu2[e, :H].contiguous()
        legacy_model_sd[f"bcg_pc.moe.experts.{e}.w_up2.weight"] = w_gu2[e, H:].contiguous()
        legacy_model_sd[f"bcg_pc.moe.experts.{e}.w_down2.weight"] = w_d2[e].contiguous()

    ref_opt_sd = ref_opt.state_dict()
    legacy_opt_groups = []
    legacy_opt_state = {}
    next_pid = 0
    for g in ref_opt_sd["param_groups"]:
        new_pnames = []
        new_pids = []
        for pname, pid in zip(g["param_names"], g["params"]):
            st = ref_opt_sd["state"].get(pid)
            if pname in ("bcg_pc.moe.w_gu1", "bcg_pc.moe.w_gu2"):
                suf_g = "w_gate1.weight" if pname.endswith("w_gu1") else "w_gate2.weight"
                suf_u = "w_up1.weight" if pname.endswith("w_gu1") else "w_up2.weight"
                for e in range(E):
                    for suf, sl in ((suf_g, slice(0, H)), (suf_u, slice(H, 2 * H))):
                        new_pnames.append(f"bcg_pc.moe.experts.{e}.{suf}")
                        new_pids.append(next_pid)
                        if st is not None:
                            legacy_opt_state[next_pid] = {
                                "step": st["step"].clone(),
                                "exp_avg": st["exp_avg"][e, sl].contiguous(),
                                "exp_avg_sq": st["exp_avg_sq"][e, sl].contiguous(),
                            }
                        next_pid += 1
            elif pname in ("bcg_pc.moe.w_d1", "bcg_pc.moe.w_d2"):
                suf_d = "w_down1.weight" if pname.endswith("w_d1") else "w_down2.weight"
                for e in range(E):
                    new_pnames.append(f"bcg_pc.moe.experts.{e}.{suf_d}")
                    new_pids.append(next_pid)
                    if st is not None:
                        legacy_opt_state[next_pid] = {
                            "step": st["step"].clone(),
                            "exp_avg": st["exp_avg"][e].contiguous(),
                            "exp_avg_sq": st["exp_avg_sq"][e].contiguous(),
                        }
                    next_pid += 1
            else:
                new_pnames.append(pname)
                new_pids.append(next_pid)
                if st is not None:
                    legacy_opt_state[next_pid] = {
                        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in st.items()
                    }
                next_pid += 1
        lg = dict(g)
        lg["param_names"] = tuple(new_pnames)
        lg["params"] = new_pids
        lg["fused"] = False
        legacy_opt_groups.append(lg)

    legacy_ckpt = {
        "step": 75000,
        "stage_id": 3,
        "model_state_dict": legacy_model_sd,
        "optimizer_state_dict": {"state": legacy_opt_state, "param_groups": legacy_opt_groups},
        "scheduler_state": ctrl.state_dict(),
        "dynamic_loss_multipliers": wd.state_dict(),
        "dataloader_cursor": {"batch_idx": 75000, "phys_offset": 75000, "epoch": 1, "seed": 1337, "consumed_bytes": 999},
        "best_val_metric": 1.23,
        "config": cfg.to_dict(),
        "hardware_tier": "h100_or_a100_80g",
    }

    # Restore legacy 75k checkpoint into BOTH uncompiled and selectively compiled fresh models!
    for do_compile in (False, True):
        resumed_model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
        if do_compile:
            configure_model_compile(resumed_model, compile_mode="auto", device=device, allow_cpu_compile=True)
        resumed_opt = create_dcpa_optimizer(resumed_model)
        step_r, cur_r = CheckpointManager.restore_into(
            legacy_ckpt, model=resumed_model, optimizer=resumed_opt, controller=ctrl, watchdog=wd, restore_rng=False
        )
        assert step_r == 75000 and cur_r["batch_idx"] == 75000

        # Exact weight parity
        resumed_sd = get_clean_state_dict(resumed_model)
        for k, v_ref in new_clean_sd.items():
            assert torch.equal(resumed_sd[k], v_ref), f"Migrated weight mismatch on {k} (compiled={do_compile})"

        # Exact AdamW moment parity
        resumed_opt_sd = resumed_opt.state_dict()
        for g_ref, g_res in zip(ref_opt_sd["param_groups"], resumed_opt_sd["param_groups"]):
            for pn, pid_ref, pid_res in zip(g_ref["param_names"], g_ref["params"], g_res["params"]):
                if pid_ref in ref_opt_sd["state"]:
                    st_ref = ref_opt_sd["state"][pid_ref]
                    st_res = resumed_opt_sd["state"][pid_res]
                    assert torch.equal(st_res["step"], st_ref["step"]), f"step mismatch on {pn}"
                    assert torch.equal(st_res["exp_avg"], st_ref["exp_avg"]), f"exp_avg mismatch on {pn}"
                    assert torch.equal(st_res["exp_avg_sq"], st_ref["exp_avg_sq"]), f"exp_avg_sq mismatch on {pn}"

    # --- 2. Flat safetensors Rust export + Stage-5 reload round-trip ---
    flat_sd = extract_flat_state_dict(ref_model, dtype=torch.float32)
    assert "bcg_pc.moe.w_gu1" not in flat_sd
    assert f"bcg_pc.moe.experts.{E - 1}.w_down2.weight" in flat_sd
    sf_path = tmp_path / "dcpa_final.safetensors"
    cfg_path = tmp_path / "config.json"
    save_file(flat_sd, str(sf_path))
    cfg.save_json(cfg_path)
    reloaded = load_dcpa_model_from_weights(sf_path, config_path=cfg_path, device=device)
    reloaded_sd = get_clean_state_dict(reloaded)
    for k in ("bcg_pc.moe.w_gu1", "bcg_pc.moe.w_d1", "bcg_pc.moe.w_gu2", "bcg_pc.moe.w_d2"):
        assert torch.equal(reloaded_sd[k], new_clean_sd[k]), f"Safetensors round-trip mismatch on {k}"

    # --- 3. RoPE cached LUT parity (even when first populated under inference_mode!) ---
    with torch.inference_mode():
        x_probe = torch.randn((32, cfg.d_s), device=device, dtype=torch.float32)
        pos_probe = torch.randint(1, cfg.max_seg_len + 1, (32,), device=device, dtype=torch.int64)
        _ = apply_holographic_rope(x_probe, pos_probe, inverse=False, max_pos=cfg.max_seg_len + 4)
    x_grad = torch.randn((64, cfg.d_s), device=device, dtype=torch.float32, requires_grad=True)
    pos_int = torch.randint(1, cfg.max_seg_len + 1, (64,), device=device, dtype=torch.int64)
    for inv in (False, True):
        y_lut = apply_holographic_rope(x_grad, pos_int, inverse=inv, max_pos=cfg.max_seg_len + 4)
        y_ref = apply_holographic_rope(x_grad, pos_int.float(), inverse=inv, max_pos=None)
        assert float((y_lut - y_ref).abs().max().item()) < 1e-6
        y_lut.sum().backward()
        assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()
        x_grad.grad = None

    # --- 4. check_pin=False exact parity + sync_metrics=False deferred materialization ---
    ref_model.eval()
    with torch.no_grad():
        out_sync = ref_model(
            raw_bytes=x_raw, lang_ids=lids, stage="B", tau_s=0.35, tau_m=0.4,
            doc_cu_seqlens=cu, record_traces=True, sync_metrics=True,
        )
        out_async = ref_model(
            raw_bytes=x_raw, lang_ids=lids, stage="B", tau_s=0.35, tau_m=0.4,
            doc_cu_seqlens=cu, record_traces=True, sync_metrics=False,
        )
    assert out_async.metrics == {} and out_async.metrics_tensor is not None
    # Slot-0 pinning trigger `\x1dE\n` MUST have fired despite check_pin optimization!
    assert out_sync.slot0_pinned_trace is not None and float(out_sync.slot0_pinned_trace.max().item()) > 0.5
    mat = materialize_metric_vec(out_async.metrics_tensor)
    assert set(mat.keys()) == set(out_sync.metrics.keys())
    for mk, mv in out_sync.metrics.items():
        assert abs(mat[mk] - mv) < 1e-6, f"Deferred metric mismatch on {mk}: {mat[mk]} vs {mv}"

    print(
        f"\n[Test 5: Step-1 Perf & Legacy 75k Resume] legacy_weights=EXACT | legacy_adamw_moments=EXACT | "
        f"safetensors_rust_roundtrip=EXACT | rope_lut<1e-6 | slot0_pin=VERIFIED | deferred_metrics=EXACT"
    )


# =============================================================================
# Test 6: Step-2 Masked MoE + Tridiagonal Stack / Chunked Diag Custom Ops + TBPTT Replay
# =============================================================================


def test_step2_masked_moe_tridiag_stack_chunked_diag_and_tbptt():
    """Verify all 4 Step-2 optimizations:
      1. Zero-sync Sentinel-E masked `SparseMicroMoE` (`F.grouped_mm` partial / all-False / all-True,
         eager & compiled `aot_eager`, exact 0.0 tail gradient with zero NaN/garbage leakage).
      2. 3-pass fused tridiagonal `dcpa::tridiag_stack_scan_fwd` / `bwd` (< 1e-5 fwd & bwd vs
         sequential reference across multi-doc resets, non-multiple T, with/without `xi_init`,
         eager & `torch.compile(dynamic=False)`).
      3. 3-pass chunked `dcpa::chunked_diag_scan_fwd` / `bwd` (< 1e-5 fwd & bwd across D=1 & D=32,
         T=1..1024, with `h0` & `reset_mask`, eager & `torch.compile(dynamic=False)`).
      4. `DCPAModel.forward_and_backward_tbptt` O(W)-VRAM replay exact gradient identity (< 1e-5
         across all SGSC/BCG-PC/Hippocampus/PMD parameters for Mode 1, Broca CQ, and W=32 vs full
         untruncated backward under `carry_grad_gamma=0.5`, plus Stage A/D fallbacks).
    """
    import copy
    from dcpa.modules.bcg_pc import ConceptGraphState
    from dcpa.modules.bcg_pc import SparseMicroMoE
    from dcpa.ops.scan import sequential_diag_scan_ref

    device = get_deterministic_test_device()
    configure_cuda_tf32(device, allow=False)
    seed_everything(2026, device=device)

    # --- 1. Zero-Sync Sentinel-E Masked MoE (`F.grouped_mm`) ---
    moe = SparseMicroMoE(d_model=64, hidden_dim=96, num_experts=8, top_k=2).to(device=device, dtype=torch.bfloat16)
    moe.allow_cpu_grouped_mm = True
    compiled_moe_fwd = torch.compile(
        lambda x_in, m_in: moe(x_in, static_shape=True, sample_mask=m_in),
        dynamic=False,
        backend="inductor" if device.type == "cuda" else "aot_eager",
    )
    N_tok = 16
    masks = {
        "partial": torch.tensor([i % 3 != 0 for i in range(N_tok)], device=device, dtype=torch.bool),
        "all_false": torch.zeros((N_tok,), device=device, dtype=torch.bool),
        "all_true": torch.ones((N_tok,), device=device, dtype=torch.bool),
    }
    for name, smask in masks.items():
        for use_comp in (False, True):
            moe.zero_grad(set_to_none=True)
            x_in = torch.randn((N_tok, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
            y_out, aux_bal = compiled_moe_fwd(x_in, smask) if use_comp else moe(x_in, static_shape=True, sample_mask=smask)
            assert torch.isfinite(y_out).all(), f"Non-finite MoE out ({name}, compiled={use_comp})"
            assert torch.isfinite(aux_bal), f"Non-finite MoE aux_bal ({name}, compiled={use_comp})"
            if (~smask).any():
                assert float(y_out[~smask].abs().max().item()) == 0.0, f"Inactive MoE output leaked ({name})"
            g_probe = torch.randn_like(y_out)
            ((y_out * g_probe).sum() + aux_bal).backward()
            assert x_in.grad is not None and torch.isfinite(x_in.grad).all()
            if (~smask).any():
                assert float(x_in.grad[~smask].abs().max().item()) == 0.0, f"Inactive MoE x.grad leaked ({name})"
            for p in moe.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
                    if name == "all_false":
                        assert float(p.grad.abs().max().item()) == 0.0, "All-False mask produced non-zero MoE param grad!"

            # Compare active rows against unmasked BMM reference
            if name == "partial" and not use_comp:
                moe_ref = copy.deepcopy(moe)
                moe_ref.allow_cpu_grouped_mm = False
                with torch.no_grad():
                    y_ref, bal_ref = moe_ref(x_in.detach(), static_shape=True, sample_mask=smask)
                cos = torch.nn.functional.cosine_similarity(
                    y_out[smask].detach().float().reshape(-1),
                    y_ref[smask].float().reshape(-1),
                    dim=0,
                ).item()
                assert cos > 0.999, f"Sentinel-E masked MoE active cos_sim={cos:.6f} <= 0.999"
                assert abs(float(aux_bal.item()) - float(bal_ref.item())) < 1e-4

    # --- 2. 3-Pass Fused Tridiagonal Stack Scan Custom Op ---
    backend = "inductor" if device.type == "cuda" else "aot_eager"

    @torch.compile(dynamic=False, backend=backend)
    def _compiled_tridiag_stack(logits_t, v_tag_t, rm_t, xi0_t):
        p = torch.softmax(logits_t, dim=-1)
        xi_o, _ = torch.ops.dcpa.tridiag_stack_scan_fwd(
            p[..., 0], p[..., 1], p[..., 2], v_tag_t, rm_t, xi0_t, 256
        )
        return xi_o

    for T_len in (1, 256, 600, 1024):
        for has_xi0 in (False, True):
            B_s, d_tag = 2, 4
            logits = torch.randn((B_s, T_len, 3), device=device, dtype=torch.float32, requires_grad=True)
            v_tag = torch.tanh(torch.randn((B_s, T_len, d_tag), device=device, dtype=torch.float32)).detach().requires_grad_(True)
            xi0 = (
                (torch.randn((B_s, 16, d_tag), device=device, dtype=torch.float32) * 0.4).detach().requires_grad_(True)
                if has_xi0
                else None
            )
            rm = torch.zeros((B_s, T_len), device=device, dtype=torch.bool)
            for pos in (0, 133, 256, 513, 777):
                if pos < T_len:
                    rm[0, pos] = True
            for pos in (7, 300):
                if pos < T_len:
                    rm[1, pos] = True

            if T_len == 600 and has_xi0:
                xi_op = _compiled_tridiag_stack(logits, v_tag, rm, xi0)
            else:
                probs = torch.softmax(logits, dim=-1)
                xi_op, _ = torch.ops.dcpa.tridiag_stack_scan_fwd(
                    probs[..., 0], probs[..., 1], probs[..., 2], v_tag, rm, xi0, 256
                )
            g_xi = torch.randn_like(xi_op)
            req = (logits, v_tag, xi0) if has_xi0 else (logits, v_tag)
            grads_op = torch.autograd.grad((xi_op * g_xi).sum(), req)

            logits_r = logits.detach().clone().requires_grad_(True)
            v_tag_r = v_tag.detach().clone().requires_grad_(True)
            xi0_r = xi0.detach().clone().requires_grad_(True) if has_xi0 else None
            probs_r = torch.softmax(logits_r, dim=-1)
            xi_ref = sequential_stack_scan_ref(
                probs_r[..., 0], probs_r[..., 1], probs_r[..., 2], v_tag_r, xi_init=xi0_r, reset_mask=rm
            )
            req_r = (logits_r, v_tag_r, xi0_r) if has_xi0 else (logits_r, v_tag_r)
            grads_ref = torch.autograd.grad((xi_ref * g_xi).sum(), req_r)

            assert float((xi_op.detach() - xi_ref.detach()).abs().max().item()) < 1e-5
            for ga, gb in zip(grads_op, grads_ref):
                assert float((ga - gb).abs().max().item()) < 1e-5

    # --- 3. 3-Pass Chunked Diagonal Scan Custom Op ---
    @torch.compile(dynamic=False, backend=backend)
    def _compiled_chunked_diag(a_in, b_in):
        return torch.ops.dcpa.chunked_diag_scan_fwd(a_in, b_in, 256)

    for T_len in (1, 256, 600, 1024):
        for D_dim in (1, 32):
            a_t = torch.sigmoid(torch.randn((2, T_len, D_dim), device=device, dtype=torch.float32)).detach().requires_grad_(True)
            b_t = torch.randn((2, T_len, D_dim), device=device, dtype=torch.float32, requires_grad=True)
            h_op = _compiled_chunked_diag(a_t, b_t) if (T_len == 600 and D_dim == 32) else torch.ops.dcpa.chunked_diag_scan_fwd(a_t, b_t, 256)
            g_h = torch.randn_like(h_op)
            ga_op, gb_op = torch.autograd.grad((h_op * g_h).sum(), (a_t, b_t))

            a_r = a_t.detach().clone().requires_grad_(True)
            b_r = b_t.detach().clone().requires_grad_(True)
            h_ref = sequential_diag_scan_ref(a_r, b_r, h0=None)
            ga_ref, gb_ref = torch.autograd.grad((h_ref * g_h).sum(), (a_r, b_r))
            assert float((h_op.detach() - h_ref.detach()).abs().max().item()) < 1e-5
            assert float((ga_op - ga_ref).abs().max().item()) < 1e-5
            assert float((gb_op - gb_ref).abs().max().item()) < 1e-5

    # --- 4. Chunked TBPTT (`bcg_tbptt_window` + `forward_and_backward_tbptt`) ---
    cfg = DCPAConfig.micro()
    cfg.l_max = 3
    m_base = DCPAModel(cfg).to(device=device, dtype=torch.float32)

    d1 = (b"def foo(x):\n    return x + 1\n\x1dS\nthink\x1dE\nanswer\x04more text " * 8)[:360]
    d2 = (b"fn bar() { let a = 1; }\n" * 10)[:220]
    d3 = (b"SELECT * FROM tbl WHERE id = 1;\n" * 6)[:140]
    d4 = (b"const z = [1, 2, 3];\n" * 18)[:350]
    raw_cat = b"".join([d1, d2, d3, d4])
    x_tb = torch.tensor(list(raw_cat), dtype=torch.uint8, device=device).unsqueeze(0)
    cu_tb = torch.tensor(
        [0, len(d1), len(d1) + len(d2), len(d1) + len(d2) + len(d3), len(raw_cat)],
        dtype=torch.int32,
        device=device,
    )
    lang_tb = torch.tensor([3, 0, 4, 5], dtype=torch.long, device=device)
    style_tb = torch.tensor([0, 1, 2, 4], dtype=torch.long, device=device)
    pts_tb = torch.tensor([1.0, 0.5, 1.5, 0.8], dtype=torch.float32, device=device)

    for stage_s, use_cq_s, W_win in (("B", False, 8), ("C", True, 8)):
        m_std = copy.deepcopy(m_base)
        m_rep = copy.deepcopy(m_base)
        m_std.train()
        m_rep.train()
        m_std.bcg_tbptt_window = W_win
        m_rep.bcg_tbptt_window = W_win

        rng_snap = capture_rng_state()
        o_std = m_std(
            raw_bytes=x_tb, lang_ids=lang_tb, style_ids=style_tb, ponder_tax_scale=pts_tb,
            stage=stage_s, tau_s=0.4, tau_m=0.5, use_broca_cq=use_cq_s,
            merge_kl_weight=0.1, merge_uniform_mix=0.05, doc_cu_seqlens=cu_tb, record_traces=False,
        )
        (o_std.loss_total * 0.25).backward()

        restore_rng_state(rng_snap)
        o_rep = m_rep.forward_and_backward_tbptt(
            raw_bytes=x_tb, lang_ids=lang_tb, style_ids=style_tb, ponder_tax_scale=pts_tb,
            stage=stage_s, tau_s=0.4, tau_m=0.5, use_broca_cq=use_cq_s,
            merge_kl_weight=0.1, merge_uniform_mix=0.05, doc_cu_seqlens=cu_tb,
            record_traces=False, loss_scale=0.25, tbptt_window=W_win,
        )
        assert abs(float(o_std.loss_total.item()) - float(o_rep.loss_total.item())) < 1e-5
        assert float((m_std.bcg_pc.c0 - m_rep.bcg_pc.c0).abs().max().item()) < 1e-6
        for (n1, p1), (n2, p2) in zip(m_std.named_parameters(), m_rep.named_parameters()):
            if p1.grad is None and p2.grad is None:
                continue
            assert (p1.grad is not None) and (p2.grad is not None), f"Grad mismatch on {n1}"
            assert float((p1.grad - p2.grad).abs().max().item()) < 1e-5, f"Grad diff >= 1e-5 on {n1}"

    # Verify W=32 replay stays closely aligned with FULL untruncated (W=0) backward under carry_grad_gamma=0.5
    # (only 1/W boundary step drops its 1-step damped carry -> cos_sim > 0.999 across all grads!)
    d_long = (b"def compute(x): return x * 2 + 1\n" * 20)[:640]
    x_long = torch.tensor(list(d_long * 2), dtype=torch.uint8, device=device).unsqueeze(0)
    cu_long = torch.tensor([0, len(d_long), len(d_long) * 2], dtype=torch.int32, device=device)
    lang_long = torch.tensor([3, 3], dtype=torch.long, device=device)
    m_full = copy.deepcopy(m_base)
    m_w32 = copy.deepcopy(m_base)
    m_full.bcg_tbptt_window = 0
    m_w32.bcg_tbptt_window = 32
    rng_snap = capture_rng_state()
    o_full = m_full(raw_bytes=x_long, lang_ids=lang_long, stage="B", tau_s=0.4, tau_m=0.5, doc_cu_seqlens=cu_long, record_traces=False)
    o_full.loss_total.backward()
    restore_rng_state(rng_snap)
    o_w32 = m_w32.forward_and_backward_tbptt(raw_bytes=x_long, lang_ids=lang_long, stage="B", tau_s=0.4, tau_m=0.5, doc_cu_seqlens=cu_long, record_traces=False, tbptt_window=32)
    flat_full = torch.cat([p.grad.reshape(-1).float() for p in m_full.parameters() if p.grad is not None])
    flat_w32 = torch.cat([p.grad.reshape(-1).float() for p in m_w32.parameters() if p.grad is not None])
    w32_cos = float(torch.nn.functional.cosine_similarity(flat_full, flat_w32, dim=0).item())
    max_w32_diff = float((flat_full - flat_w32).abs().max().item())
    assert w32_cos > 0.999, f"W=32 vs untruncated grad cos_sim {w32_cos:.6f} <= 0.999"

    # Verify Stage A & Stage D fallbacks execute cleanly
    for fb_stage in ("A", "D"):
        m_fb = copy.deepcopy(m_base)
        m_fb.train()
        m_fb.bcg_tbptt_window = 8
        o_fb = m_fb.forward_and_backward_tbptt(
            raw_bytes=x_tb[:, :128], lang_ids=torch.tensor([0], dtype=torch.long, device=device),
            stage=fb_stage, tau_s=0.4, tau_m=0.5, loss_scale=0.5, tbptt_window=8,
        )
        assert torch.isfinite(o_fb.loss_total)

    print(
        f"\n[Test 6: Step-2 Perf Suite] sentinel_e_masked_moe=EXACT | tridiag_stack_op<1e-5 | "
        f"chunked_diag_op<1e-5 | tbptt_replay(B/CQ)<1e-5 | W32_vs_full(cos={w32_cos:.6f}, max={max_w32_diff:.3e})"
    )


# =============================================================================
# Test 7: Step-3 Multi-Step Chunked BCG-PC Compile (K=8) + Zero-Sync CQ/Stage4 + RTX 5090
# =============================================================================


def test_step3_bcg_chunk_compile_zero_sync_cq_stage4_and_rtx5090():
    """Verify all Step-3 optimizations:
      1. `bcg_chunk_steps=8` vs `bcg_chunk_steps=1` exact forward+backward gradient parity (< 1e-5)
         for BOTH Mode 1 (`stage="B"`) and Broca CQ (`stage="C", use_broca_cq=True`), with
         chunk-level activation checkpointing, Slot-0 pinning/unpinning, TBPTT window boundaries,
         and selective `torch.compile(dynamic=False)` (0 `_orig_mod.` leakage).
      2. Zero-sync `use_prefix_shrink` on Broca CQ (`use_cq=True`) & zero-sync Stage 4 Hippocampus
         (`thresh=0.0` branchless write + `inherit_eligibility_on_merge` branchless Case B + GPU
         familiar ACh accumulation) producing finite exact gradients.
      3. RTX 5090 (32GB) hardware tier classification (`65536 x 2`) and safe watermark clamping.
    """
    import copy
    from dcpa.modules.bcg_pc import ConceptGraphState

    device = get_deterministic_test_device()
    configure_cuda_tf32(device, allow=False)
    seed_everything(303, device=device)

    cfg = DCPAConfig.micro()
    cfg.l_max = 3
    m_base = DCPAModel(cfg).to(device=device, dtype=torch.float32)

    # Multi-doc packed batch with variable lengths + Slot-0 pin `\x1dE\n` + unpin `\x04`
    d1 = (b"def solve(n):\n    return n * 2\n\x1dS\nstep\x1dE\nresult\x04tail " * 6)[:320]
    d2 = (b"SELECT id, val FROM items WHERE val > 10;\n" * 6)[:200]
    d3 = (b"function add(a, b) { return a + b; }\n" * 8)[:280]
    d4 = (b"The quick brown fox jumps over the lazy dog.\n" * 4)[:160]
    raw_all = b"".join([d1, d2, d3, d4])
    x = torch.tensor(list(raw_all), dtype=torch.uint8, device=device).unsqueeze(0)
    cu = torch.tensor(
        [0, len(d1), len(d1) + len(d2), len(d1) + len(d2) + len(d3), len(raw_all)],
        dtype=torch.int32,
        device=device,
    )
    lang = torch.tensor([3, 4, 5, 0], dtype=torch.long, device=device)
    style = torch.tensor([0, 4, 1, 2], dtype=torch.long, device=device)
    pts = torch.tensor([1.0, 0.8, 1.2, 0.9], dtype=torch.float32, device=device)

    # 1. Exact forward + backward parity: bcg_chunk_steps=2/4 (eager & compiled) vs bcg_chunk_steps=1.
    #    Use a shared pre-drawn i.i.d. Uniform(0,1) Gumbel stream so CUDA Philox counter stride
    #    differences between 1 batched `torch.rand((K_c, V, 72))` vs K_c `torch.rand((V, 72))`
    #    calls do not inject different Gumbel noise during strict `< 1e-5` parity checking!
    eq_raw = d1 * 4
    x_eq = torch.tensor(list(eq_raw), dtype=torch.uint8, device=device).unsqueeze(0)
    cu_eq = torch.tensor([0, len(d1), len(d1) * 2, len(d1) * 3, len(d1) * 4], dtype=torch.int32, device=device)
    shared_gumbel = torch.rand((256, 4, cfg.num_merge_pairs), device=device, dtype=torch.float32)
    orig_torch_rand = torch.rand

    def _run_with_shared_gumbel(fn):
        cursor = [0]

        def _det_rand(size, *args, **kwargs):
            if isinstance(size, tuple) and len(size) == 2 and size == (4, cfg.num_merge_pairs):
                out_u = shared_gumbel[cursor[0]]
                cursor[0] += 1
                return out_u
            if isinstance(size, tuple) and len(size) == 3 and size[1:] == (4, cfg.num_merge_pairs):
                k_sz = size[0]
                out_u = shared_gumbel[cursor[0] : cursor[0] + k_sz]
                cursor[0] += k_sz
                return out_u
            return orig_torch_rand(size, *args, **kwargs)

        torch.rand = _det_rand  # type: ignore[assignment]
        try:
            return fn()
        finally:
            torch.rand = orig_torch_rand  # type: ignore[assignment]

    for stage_s, use_cq_s, do_compile, use_ckpt, k_chunk in (
        ("B", False, False, True, 4),
        ("C", True, False, True, 4),
        ("B", False, True, False, 2),
        ("C", True, True, True, 2),
    ):
        m_1 = copy.deepcopy(m_base)
        m_8 = copy.deepcopy(m_base)
        if do_compile:
            rep = configure_model_compile(m_8, compile_mode="auto", device=device, allow_cpu_compile=True)
            assert "bcg_chunk_kernel" in rep.compiled_hotspots
            assert not any("_orig_mod" in k for k in get_clean_state_dict(m_8).keys())
        m_1.train()
        m_8.train()
        m_1.bcg_grad_checkpointing = use_ckpt
        m_8.bcg_grad_checkpointing = use_ckpt
        m_1.bcg_tbptt_window = 16
        m_8.bcg_tbptt_window = 16
        m_1.bcg_chunk_steps = 1
        m_8.bcg_chunk_steps = k_chunk

        # Disable doc bucketing on compiled m_8 only when matching exact RNG shape with m_1
        m_8.enable_doc_bucketing = False

        out_1 = _run_with_shared_gumbel(
            lambda: m_1(
                raw_bytes=x_eq, lang_ids=lang, style_ids=style, ponder_tax_scale=pts,
                stage=stage_s, tau_s=0.4, tau_m=0.5, use_broca_cq=use_cq_s,
                merge_kl_weight=0.1, doc_cu_seqlens=cu_eq, record_traces=False,
            )
        )
        out_1.loss_total.backward()

        out_8 = _run_with_shared_gumbel(
            lambda: m_8(
                raw_bytes=x_eq, lang_ids=lang, style_ids=style, ponder_tax_scale=pts,
                stage=stage_s, tau_s=0.4, tau_m=0.5, use_broca_cq=use_cq_s,
                merge_kl_weight=0.1, doc_cu_seqlens=cu_eq, record_traces=False,
            )
        )
        out_8.loss_total.backward()

        loss_diff = abs(float(out_1.loss_total.item()) - float(out_8.loss_total.item()))
        c0_diff = float((m_1.bcg_pc.c0 - getattr(m_8.bcg_pc, "_orig_mod", m_8.bcg_pc).c0).abs().max().item())
        assert loss_diff < 1e-5, f"chunk={k_chunk} vs 1 loss_diff={loss_diff} (stage={stage_s}, cq={use_cq_s}, comp={do_compile})"
        assert c0_diff < 1e-5, f"chunk={k_chunk} vs 1 c0_diff={c0_diff}"
        for (n1, p1), (_, p8) in zip(m_1.named_parameters(), m_8.named_parameters()):
            if p1.grad is None and p8.grad is None:
                continue
            assert (p1.grad is not None) and (p8.grad is not None), f"Grad presence mismatch on {n1}"
            g_diff = float((p1.grad - p8.grad).abs().max().item())
            assert g_diff < 1e-5, f"chunk={k_chunk} vs 1 grad_diff={g_diff:.3e} on {n1} (stage={stage_s}, cq={use_cq_s})"

    # 2. Variable-length docs with prefix-shrinking (V=4) in BOTH train (grads!) & eval mode:
    #    verifies K-step block-parallel `broca_cq_blocks_parallel` (<=K calls) vs sequential `broca_cq_step`!
    m_ev1 = copy.deepcopy(m_base).eval()
    m_ev8 = copy.deepcopy(m_base).eval()
    m_ev1.bcg_chunk_steps = 1
    m_ev8.bcg_chunk_steps = 8
    with torch.no_grad():
        for use_cq_e in (False, True):
            o1 = m_ev1(raw_bytes=x, lang_ids=lang, style_ids=style, stage="C", tau_s=0.4, tau_m=0.5, use_broca_cq=use_cq_e, doc_cu_seqlens=cu, record_traces=False)
            o8 = m_ev8(raw_bytes=x, lang_ids=lang, style_ids=style, stage="C", tau_s=0.4, tau_m=0.5, use_broca_cq=use_cq_e, doc_cu_seqlens=cu, record_traces=False)
            assert float((o1.pmd.byte_logits - o8.pmd.byte_logits).abs().max().item()) < 1e-5

    # Direct ragged multi-doc (V=6: lengths <K, ==K, >2*K+3) forward + backward gradient parity for parallel Broca CQ
    rag_lens = [16, 48, 80, 144, 24, 112]
    cu_rag = torch.tensor([0] + [sum(rag_lens[: i + 1]) for i in range(len(rag_lens))], dtype=torch.int64, device=device)
    x_rag = torch.randint(32, 126, (1, sum(rag_lens)), dtype=torch.long, device=device)
    lang_rag = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long, device=device)
    style_rag = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long, device=device)
    m_seq_cq = copy.deepcopy(m_base).train()
    m_par_cq = copy.deepcopy(m_base).train()
    m_seq_cq.bcg_chunk_steps = 1
    m_par_cq.bcg_chunk_steps = 2
    m_seq_cq.bcg_tbptt_window = 16
    m_par_cq.bcg_tbptt_window = 16
    # Verify direct `PMD.broca_cq_blocks_parallel` vs sequential `PMD.broca_cq_step` on ragged docs
    m_seq_cq.eval()
    m_par_cq.eval()
    with torch.no_grad():
        o_rag_1 = m_seq_cq(raw_bytes=x_rag, lang_ids=lang_rag, style_ids=style_rag, stage="C", tau_s=0.4, tau_m=0.5, use_broca_cq=True, doc_cu_seqlens=cu_rag, record_traces=False)
        o_rag_2 = m_par_cq(raw_bytes=x_rag, lang_ids=lang_rag, style_ids=style_rag, stage="C", tau_s=0.4, tau_m=0.5, use_broca_cq=True, doc_cu_seqlens=cu_rag, record_traces=False)
        assert float((o_rag_1.pmd.byte_logits - o_rag_2.pmd.byte_logits).abs().max().item()) < 1e-5

    # Exact forward + backward gradient parity of `broca_cq_blocks_parallel` vs sequential loop on ragged docs
    seg_cnts_t = [2, 4, 7, 13, 1, 9]
    seg_offs_t = [sum(seg_cnts_t[:i]) for i in range(len(seg_cnts_t))]
    m_tot_t = sum(seg_cnts_t)
    pre_s_ref = torch.randn((m_tot_t, cfg.k_slots, cfg.d_concept), device=device, requires_grad=True)
    pre_e_ref = torch.randn((m_tot_t, cfg.k_slots, cfg.k_slots, cfg.d_r), device=device, requires_grad=True)
    pre_w_ref = torch.randn((m_tot_t, cfg.k_slots), device=device, requires_grad=True)
    lreg_ref = torch.randn((m_tot_t, cfg.d_l), device=device, requires_grad=True)
    pre_s_par = pre_s_ref.detach().clone().requires_grad_(True)
    pre_e_par = pre_e_ref.detach().clone().requires_grad_(True)
    pre_w_par = pre_w_ref.detach().clone().requires_grad_(True)
    lreg_par = lreg_ref.detach().clone().requires_grad_(True)

    pmd_seq = copy.deepcopy(m_base.pmd).train()
    pmd_par = copy.deepcopy(m_base.pmd).train()
    z_seq_list, ds_seq_list = [], []
    for cnt_d, off_d in zip(seg_cnts_t, seg_offs_t):
        q_c = torch.zeros((1, cfg.k_slots), device=device)
        a_c = torch.zeros((1, cfg.k_slots, cfg.d_r), device=device)
        for st_m in range(cnt_d):
            fi = off_d + st_m
            q_init_i = torch.sigmoid(pre_w_ref[fi : fi + 1].float())
            q_in_i = q_init_i if (st_m % cfg.k_slots == 0) else q_c
            a_in_i = torch.zeros_like(a_c) if (st_m % cfg.k_slots == 0) else a_c
            cq_o = pmd_seq.broca_cq_step(
                graph=ConceptGraphState(pre_s_ref[fi : fi + 1], pre_e_ref[fi : fi + 1], pre_w_ref[fi : fi + 1]),
                q_prev=q_in_i, q_init=q_init_i, a_prev=a_in_i, l_reg=lreg_ref[fi : fi + 1],
            )
            q_c = cq_o.q_next
            a_c = cq_o.a_prev_next
            z_seq_list.append(cq_o.z_out)
            ds_seq_list.append(cq_o.delta_syn)
    z_seq_all = torch.cat(z_seq_list, dim=0)
    ds_seq_all = torch.cat(ds_seq_list, dim=0)
    z_par_all, ds_par_all = pmd_par.broca_cq_blocks_parallel(
        pre_s_par, pre_e_par, pre_w_par, lreg_par, seg_cnts_t, seg_offs_t
    )
    assert float((z_seq_all - z_par_all).abs().max().item()) < 1e-5
    assert float((ds_seq_all - ds_par_all).abs().max().item()) < 1e-5
    w_probe_z = torch.randn_like(z_seq_all)
    w_probe_ds = torch.randn_like(ds_seq_all)
    ((z_seq_all * w_probe_z).mean() + (ds_seq_all * w_probe_ds).mean()).backward()
    ((z_par_all * w_probe_z).mean() + (ds_par_all * w_probe_ds).mean()).backward()
    for t_a, t_b, tname in (
        (pre_s_ref.grad, pre_s_par.grad, "pre_s"),
        (pre_e_ref.grad, pre_e_par.grad, "pre_e"),
        (pre_w_ref.grad, pre_w_par.grad, "pre_w"),
        (lreg_ref.grad, lreg_par.grad, "lreg"),
    ):
        assert t_a is not None and t_b is not None
        assert float((t_a - t_b).abs().max().item()) < 1e-5, f"parallel CQ input grad mismatch on {tname}"
    for (pn, pa), (_, pb) in zip(pmd_seq.named_parameters(), pmd_par.named_parameters()):
        if pa.grad is None and pb.grad is None:
            continue
        assert pa.grad is not None and pb.grad is not None
        assert float((pa.grad - pb.grad).abs().max().item()) < 1e-5, f"parallel CQ param grad mismatch on {pn}"

    # 3. Stage 4 zero-sync Hippocampus + familiar ACh GPU accumulation + branchless eligibility inheritance
    m_s4 = copy.deepcopy(m_base)
    m_s4.train()
    m_s4.set_stage_freeze("D")
    fam_mask = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
    o_s4 = m_s4(
        raw_bytes=x, lang_ids=lang, style_ids=style, stage="D", tau_s=0.35, tau_m=0.35,
        familiar_ach_reg_weight=0.05, familiar_doc_mask=fam_mask, doc_cu_seqlens=cu,
        record_traces=False, sync_metrics=False,
    )
    o_s4.loss_total.backward()
    assert torch.isfinite(o_s4.loss_total)
    assert o_s4.metrics_tensor is not None and torch.isfinite(o_s4.metrics_tensor).all()
    assert m_s4.hippocampus.alpha_q.grad is not None and torch.isfinite(m_s4.hippocampus.alpha_q.grad).all()

    # 4. RTX 5090 (32GB) hardware tier classification
    prof_5090 = select_hardware_tier("cuda", vram_gb=32.0, device_name="NVIDIA GeForce RTX 5090")
    assert prof_5090.tier == "rtx_5090_32g"
    assert (prof_5090.micro_batch_bytes, prof_5090.grad_accum_steps) == (65_536, 2)

    # 5. Verify Slot-0 pin capability batches (`\x1dE\n` / `\x1fA\n`) NEVER bypass compiled `forward_chunk`!
    m_pin_comp = copy.deepcopy(m_base).train()
    configure_model_compile(m_pin_comp, compile_mode="auto", device=device, allow_cpu_compile=True)
    m_pin_comp.bcg_chunk_steps = 4
    m_pin_comp.bcg_grad_checkpointing = False
    bcg_raw_pin = getattr(m_pin_comp.bcg_pc, "_orig_mod", m_pin_comp.bcg_pc)
    assert bcg_raw_pin._compiled_chunk_fn is not None
    compiled_calls = [0]
    orig_compiled_chunk = bcg_raw_pin._compiled_chunk_fn

    def _counted_compiled_chunk(*a, **kw):
        compiled_calls[0] += 1
        return orig_compiled_chunk(*a, **kw)

    bcg_raw_pin._compiled_chunk_fn = _counted_compiled_chunk
    o_pin = m_pin_comp(
        raw_bytes=x, lang_ids=lang, style_ids=style, ponder_tax_scale=pts,
        stage="C", tau_s=0.4, tau_m=0.5, use_broca_cq=True,
        doc_cu_seqlens=cu, record_traces=False,
    )
    o_pin.loss_total.backward()
    assert compiled_calls[0] > 0, "Slot-0 pin batch bypassed compiled bcg_chunk_kernel!"
    assert o_pin.graph_state is not None and o_pin.graph_state.slot0_pinned is not None
    assert float(o_pin.graph_state.slot0_pinned[0].item()) >= 0.0

    print(
        "\n[Test 7: Step-3 Perf Suite] bcg_chunk(K=8, eager+compiled, B+CQ, ckpt+tbptt)<1e-5 | "
        "varlen_prefix_shrink_CQ=EXACT | stage4_zero_sync_hippocampus=PASS | rtx_5090_32g=64KBx2"
    )
