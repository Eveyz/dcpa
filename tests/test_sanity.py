"""Chief Audit Sanity Suite for DCPA (4 Mandatory Architectural Contract Tests + Iron Law #1 Guard).

Strictly enforces:
  - Deterministic seed=42, TF32 disabled, float32 precision, CPU when CUDA unavailable.
  - Test 1: Future-byte zero leakage (1A SGSC, 1B BCG-PC + PMD) & Ragged cu_seqlens cross-doc isolation (1C).
  - Test 2: Forward hard discreteness (s_t in {0,1}, alpha_ij one-hot) & full STE/Gumbel gradient connectivity.
  - Test 3: Hippocampal fast-weight physical buffer isolation & zero-grad one-shot Gated Delta / Householder overwrite.
  - Test 4: Strict O(1) constant runtime working memory across T=512 vs T=8192 (16x span, 0 byte delta).
"""

from __future__ import annotations

import ast
import contextlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
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


def _setup_deterministic_env(seed: int = 42) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return get_deterministic_test_device()


def _collect_state_tensors(obj: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Recursively collect all tensors in carry_state and fail if any dynamic Python list grows."""
    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        found.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            found.extend(_collect_state_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif hasattr(obj, "__dataclass_fields__"):
        for k in obj.__dataclass_fields__:
            found.extend(_collect_state_tensors(getattr(obj, k), f"{prefix}.{k}" if prefix else k))
    elif hasattr(obj, "__dict__"):
        for k, v in sorted(obj.__dict__.items()):
            found.extend(_collect_state_tensors(v, f"{prefix}.{k}" if prefix else k))
    return found


def _assert_no_dynamic_lists(obj: Any, path: str = "root") -> None:
    """Assert that carry_state / model contains no non-empty dynamic list caches."""
    if isinstance(obj, list):
        assert len(obj) == 0, f"Dynamic Python list cache detected at {path}: len={len(obj)}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_dynamic_lists(v, f"{path}[{k!r}]")
    elif hasattr(obj, "__dataclass_fields__"):
        for k in obj.__dataclass_fields__:
            _assert_no_dynamic_lists(getattr(obj, k), f"{path}.{k}")
    elif hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        for k, v in obj.__dict__.items():
            _assert_no_dynamic_lists(v, f"{path}.{k}")


# =============================================================================
# Test 1: Future-byte Zero Leakage & Ragged Cross-Document Isolation
# =============================================================================
def test_causality_and_varlen_isolation():
    device = _setup_deterministic_env(42)
    tol = 0.0 if device.type == "cpu" else 1e-7

    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    model.eval()

    # -------------------------------------------------------------------------
    # Test 1A: Byte-by-byte strict causality in SGSC (T = 256, split at 128)
    # -------------------------------------------------------------------------
    T = 256
    x_a = torch.randint(32, 127, (1, T), dtype=torch.uint8, device=device)
    x_b = x_a.clone()
    x_b[:, 128:] = ((x_b[:, 128:].int() + 53) % 95 + 32).to(torch.uint8)
    assert torch.equal(x_a[:, :128], x_b[:, :128])
    assert not torch.equal(x_a[:, 128:], x_b[:, 128:])

    lang_1 = torch.tensor([1], dtype=torch.long, device=device)

    with torch.no_grad():
        out_a = model(x_a, lang_1, stage="B", tau_s=0.35, tau_m=0.25, use_broca_cq=False)
        out_b = model(x_b, lang_1, stage="B", tau_s=0.35, tau_m=0.25, use_broca_cq=False)

    log_p_a = torch.log_softmax(out_a.sgsc.prior_logits[:, :128, :], dim=-1)
    log_p_b = torch.log_softmax(out_b.sgsc.prior_logits[:, :128, :], dim=-1)

    diff_log_p = (log_p_a - log_p_b).abs().max().item()
    diff_info = (out_a.sgsc.info[:, :128] - out_b.sgsc.info[:, :128]).abs().max().item()
    diff_v = (out_a.sgsc.v_tilde[:, :128] - out_b.sgsc.v_tilde[:, :128]).abs().max().item()
    diff_s = (out_a.sgsc.spikes[:, :128].float() - out_b.sgsc.spikes[:, :128].float()).abs().max().item()
    sgsc_diff = max(diff_log_p, diff_info, diff_v, diff_s)

    print(
        f"[Test 1A] SGSC causality: diff_log_p={diff_log_p:.3e}, diff_I={diff_info:.3e}, "
        f"diff_v={diff_v:.3e}, diff_s={diff_s:.3e}"
    )
    assert diff_log_p <= tol, f"1A FAIL: log_p_t leaked future bytes (max_diff={diff_log_p})"
    assert diff_info <= tol, f"1A FAIL: I_t leaked future bytes (max_diff={diff_info})"
    assert diff_v <= tol, f"1A FAIL: v_t leaked future bytes (max_diff={diff_v})"
    assert diff_s == 0.0, f"1A FAIL: s_t leaked future bytes (max_diff={diff_s})"

    # -------------------------------------------------------------------------
    # Test 1B: Cross-segment BCG-PC concept graph & PMD causality up to t_cut < 128
    # -------------------------------------------------------------------------
    spike_pos_a = torch.nonzero(out_a.sgsc.spikes[0], as_tuple=False).squeeze(-1)
    prefix_spikes_a = spike_pos_a[spike_pos_a < 128]
    assert prefix_spikes_a.numel() >= 1, "Expected at least one spike before byte 128"
    t_cut = int(prefix_spikes_a[-1].item())
    m_cut = int(prefix_spikes_a.numel())

    assert torch.equal(out_a.sgsc.spikes[0, : t_cut + 1], out_b.sgsc.spikes[0, : t_cut + 1]), (
        "1B FAIL: spike positions in <= t_cut diverged"
    )
    assert out_a.s_trace is not None and out_b.s_trace is not None, "1B FAIL: model must expose s_trace"
    assert out_a.a_trace is not None and out_b.a_trace is not None, "1B FAIL: model must expose a_trace"

    diff_c = (out_a.sgsc.c_slow[:m_cut] - out_b.sgsc.c_slow[:m_cut]).abs().max().item()
    diff_f = (out_a.sgsc.f_stem[:m_cut] - out_b.sgsc.f_stem[:m_cut]).abs().max().item()
    diff_S = (out_a.s_trace[:m_cut] - out_b.s_trace[:m_cut]).abs().max().item()
    diff_A = (out_a.a_trace[:m_cut] - out_b.a_trace[:m_cut]).abs().max().item()
    diff_pmd_logits = (
        out_a.pmd.byte_logits[:, : t_cut + 1, :] - out_b.pmd.byte_logits[:, : t_cut + 1, :]
    ).abs().max().item()
    bcg_pmd_diff = max(diff_c, diff_f, diff_S, diff_A, diff_pmd_logits)

    print(
        f"[Test 1B] t_cut={t_cut}, m_cut={m_cut}: diff_c={diff_c:.3e}, diff_f={diff_f:.3e}, "
        f"diff_S={diff_S:.3e}, diff_A={diff_A:.3e}, diff_pmd_logits={diff_pmd_logits:.3e}"
    )
    assert bcg_pmd_diff <= 1e-7, f"1B FAIL: cross-segment/decoder causality diff={bcg_pmd_diff}"

    # -------------------------------------------------------------------------
    # Test 1C: Ragged cu_seqlens = [0, 128, 256] cross-document zero crosstalk
    # -------------------------------------------------------------------------
    doc1_orig = torch.randint(32, 127, (128,), dtype=torch.uint8, device=device)
    doc1_tamp = ((doc1_orig.int() + 37) % 95 + 32).to(torch.uint8)
    doc2_fixed = torch.randint(32, 127, (128,), dtype=torch.uint8, device=device)

    stream_orig = torch.cat([doc1_orig, doc2_fixed], dim=0).unsqueeze(0)  # [1, 256]
    stream_tamp = torch.cat([doc1_tamp, doc2_fixed], dim=0).unsqueeze(0)  # [1, 256]
    doc_cu = torch.tensor([0, 128, 256], dtype=torch.int32, device=device)
    doc_langs = torch.tensor([1, 4], dtype=torch.long, device=device)

    with torch.no_grad():
        v_orig = model(
            stream_orig, doc_langs, stage="B", tau_s=0.35, tau_m=0.25,
            use_broca_cq=False, doc_cu_seqlens=doc_cu,
        )
        v_tamp = model(
            stream_tamp, doc_langs, stage="B", tau_s=0.35, tau_m=0.25,
            use_broca_cq=False, doc_cu_seqlens=doc_cu,
        )

    # Doc 2 byte-level states [128:256]
    diff_d2_v = (v_orig.sgsc.v_tilde[:, 128:] - v_tamp.sgsc.v_tilde[:, 128:]).abs().max().item()
    diff_d2_s = (v_orig.sgsc.spikes[:, 128:].float() - v_tamp.sgsc.spikes[:, 128:].float()).abs().max().item()
    diff_d2_xi = (v_orig.pmd.stack_xi[:, 128:] - v_tamp.pmd.stack_xi[:, 128:]).abs().max().item()
    diff_d2_m = (v_orig.pmd.m_seq[:, 128:] - v_tamp.pmd.m_seq[:, 128:]).abs().max().item()
    diff_d2_charge = (v_orig.pmd.charge_e[:, 128:] - v_tamp.pmd.charge_e[:, 128:]).abs().max().item()
    diff_d2_logits = (v_orig.pmd.byte_logits[:, 128:] - v_tamp.pmd.byte_logits[:, 128:]).abs().max().item()

    # Doc 2 segment-level concept graph states (filter segments belonging to doc 1 vs doc 2)
    seg_starts_orig = v_orig.sgsc.cu_seqlens[:-1]
    seg_starts_tamp = v_tamp.sgsc.cu_seqlens[:-1]
    d2_mask_orig = seg_starts_orig >= 128
    d2_mask_tamp = seg_starts_tamp >= 128
    assert int(d2_mask_orig.sum().item()) == int(d2_mask_tamp.sum().item())
    assert int(d2_mask_orig.sum().item()) > 0
    assert v_orig.s_trace is not None and v_tamp.s_trace is not None
    assert v_orig.a_trace is not None and v_tamp.a_trace is not None

    diff_d2_S = (v_orig.s_trace[d2_mask_orig] - v_tamp.s_trace[d2_mask_tamp]).abs().max().item()
    diff_d2_A = (v_orig.a_trace[d2_mask_orig] - v_tamp.a_trace[d2_mask_tamp]).abs().max().item()
    varlen_diff = max(
        diff_d2_v, diff_d2_s, diff_d2_xi, diff_d2_m, diff_d2_charge, diff_d2_logits, diff_d2_S, diff_d2_A
    )

    print(
        f"[Test 1C] Varlen Doc2 isolation: diff_v={diff_d2_v:.3e}, diff_S={diff_d2_S:.3e}, "
        f"diff_A={diff_d2_A:.3e}, diff_Xi={diff_d2_xi:.3e}, diff_charge={diff_d2_charge:.3e}, "
        f"diff_logits={diff_d2_logits:.3e}"
    )
    assert varlen_diff <= 1e-7, f"1C FAIL: Doc1 bled into Doc2 across cu_seqlens boundary (max_diff={varlen_diff})"
    print(f"[PASS Test 1] SGSC_diff={sgsc_diff:.3e}, BCG_diff={bcg_pmd_diff:.3e}, Varlen_Doc2_diff={varlen_diff:.3e}")


# =============================================================================
# Test 2: Forward Hard Discreteness & Full STE / Gumbel Gradient Connectivity
# =============================================================================
def test_discrete_forward_and_ste_gradients():
    device = _setup_deterministic_env(42)

    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    model.train()
    model.set_stage_freeze("C")

    mixed_text = (
        "The quick brown fox jumps over the lazy dog. "
        "那只灵巧的小鸟飞过了石桥并落在树枝上。 "
        "青い鳥が静かな森の上を飛んでいる。 "
        "fn calc(xs: &[i32]) -> i32 { let mut s = 0; for v in xs { if (*v > 0) { s += *v; } } s } "
    )
    raw_py = mixed_text.encode("utf-8")
    while len(raw_py) < 256:
        raw_py += raw_py
    raw_py = raw_py[:256]
    # Guarantee clean UTF-8 cut at byte 256
    while raw_py and (raw_py[-1] & 0xC0) == 0x80:
        raw_py = raw_py[:-1] + b" "
    if len(raw_py) < 256:
        raw_py = raw_py + b" " * (256 - len(raw_py))

    raw = torch.tensor(list(raw_py), dtype=torch.uint8, device=device).unsqueeze(0)  # [1, 256]
    lang = torch.tensor([4], dtype=torch.long, device=device)

    model.zero_grad(set_to_none=True)
    out = model(raw, lang, stage="C", tau_s=0.5, tau_m=0.6, use_broca_cq=True)

    # 1. Forward strict discreteness checks (no continuous cheating!)
    s_t = out.sgsc.s_ste
    assert torch.all((s_t == 0.0) | (s_t == 1.0)), "Forward s_t is NOT strictly binary {0, 1}!"
    n_spikes = int(s_t.sum().item())
    assert 1 < n_spikes < 256, f"Degenerate spike count: {n_spikes}"

    alpha_ij = out.merge_alpha_trace
    assert alpha_ij is not None and alpha_ij.numel() > 0, "Model must expose forward merge_alpha_trace [M, 72]"
    alpha_max = alpha_ij.max(dim=-1).values
    alpha_sum = alpha_ij.sum(dim=-1)
    assert torch.all(alpha_max == 1.0), f"Forward merge alpha_ij is NOT hard one-hot! max={alpha_max}"
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=0.0), (
        f"Forward merge alpha_ij sum != 1.0! sum={alpha_sum}"
    )

    # 2. Full 5-term joint loss backward & gradient connectivity across all bottlenecks
    loss_total = out.loss_lang + out.loss_pc + out.loss_rate + out.loss_ponder + out.loss_halt
    assert torch.isfinite(loss_total), f"Non-finite L_total: {loss_total}"
    loss_total.backward()

    checked_params: list[tuple[str, torch.nn.Parameter]] = [
        # SGSC
        ("SGSC.theta_raw (vartheta)", model.sgsc.theta_raw),
        ("SGSC.w_u", model.sgsc.w_u.weight),
        ("SGSC.w_alpha", model.sgsc.w_alpha.weight),
        ("SGSC.w_q", model.sgsc.w_q.weight),
        ("SGSC.w_k", model.sgsc.w_k.weight),
        ("SGSC.w_f (W_v)", model.sgsc.w_f.weight),
        # BCG-PC
        ("BCG-PC.w_merge", model.bcg_pc.w_merge.weight),
        ("BCG-PC.w_l (W_L)", model.bcg_pc.w_l.weight),
        ("BCG-PC.w_r (W_R)", model.bcg_pc.w_r.weight),
        ("BCG-PC.w_edge_score (W_E)", model.bcg_pc.w_edge_score.weight),
        ("BCG-PC.w_comp", model.bcg_pc.w_comp.weight),
        ("BCG-PC.w_g", model.bcg_pc.w_g.weight),
        ("BCG-PC.f_pred_1", model.bcg_pc.f_pred_1.weight),
        ("BCG-PC.f_pred_2", model.bcg_pc.f_pred_2.weight),
        ("BCG-PC.f_prec_1", model.bcg_pc.f_prec_1.weight),
        ("BCG-PC.f_prec_2", model.bcg_pc.f_prec_2.weight),
        # PMD
        ("PMD.w_ord", model.pmd.w_ord.weight),
        ("PMD.w_syn", model.pmd.w_syn.weight),
        ("PMD.u_demod", model.pmd.u_demod.weight),
        ("PMD.w_act", model.pmd.w_act.weight),
        ("PMD.w_tag", model.pmd.w_tag.weight),
        ("PMD.w_e_init (w_E)", model.pmd.w_e_init.weight),
        ("PMD.w_drain", model.pmd.w_drain.weight),
    ]

    norms: dict[str, float] = {}
    for label, param in checked_params:
        assert param.grad is not None, f"Gradient disconnected (None) for {label}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {label}"
        gnorm = float(param.grad.norm().item())
        norms[label] = gnorm
        print(f"  [Test 2 grad] {label:28s}: norm={gnorm:.4e}")
        assert gnorm > 1e-8, f"Gradient vanished (norm={gnorm:.3e} <= 1e-8) for {label}"

    print(
        f"[PASS Test 2] spikes={n_spikes}/256, alpha_steps={alpha_ij.shape[0]} (strict 1-hot), "
        f"vartheta_grad_norm={norms['SGSC.theta_raw (vartheta)']:.4e}, "
        f"w_merge_grad_norm={norms['BCG-PC.w_merge']:.4e}, "
        f"W_ord_grad_norm={norms['PMD.w_ord']:.4e}"
    )


# =============================================================================
# Test 3: Hippocampus Physical Isolation & Zero-Grad One-Shot Delta Algebra
# =============================================================================
def test_hippocampus_isolation_and_oneshot_delta():
    device = _setup_deterministic_env(42)

    cfg = DCPAConfig.tiny()  # d_h=2048, s_active=24 (<= 32), D=512, d_s=128
    cfg.gamma_hip = 1.0
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)

    # 1. Optimizer & autograd physical isolation across all 6 hippocampal tensors
    optimizer = torch.optim.AdamW(model.parameter_groups())
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    hip_buffers = dict(model.hippocampus.named_buffers())

    required_six = ("m_c_from_l", "m_l_from_c", "u_lex", "u_sem", "r_lex", "r_sem")
    for bname in required_six:
        assert bname in hip_buffers, f"Hippocampal tensor {bname!r} MUST be registered as a buffer!"
        tensor = hip_buffers[bname]
        assert tensor.requires_grad is False, f"{bname} has requires_grad=True!"

    hip_ids = {id(hip_buffers[b]) for b in required_six}
    overlap = hip_ids & opt_param_ids
    assert len(overlap) == 0, f"Hippocampal buffers leaked into AdamW optimizer! overlap={len(overlap)}"

    # Verify optimizer.step() produces bit-level ZERO change on r_lex and r_sem
    r_lex_before = model.hippocampus.r_lex.clone()
    r_sem_before = model.hippocampus.r_sem.clone()

    raw = torch.randint(32, 126, (1, 64), dtype=torch.uint8, device=device)
    lang = torch.tensor([1], dtype=torch.long, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(raw, lang, stage="D", tau_s=0.35, tau_m=0.25, use_broca_cq=True)
    out.loss_total.backward()
    optimizer.step()

    assert torch.equal(model.hippocampus.r_lex, r_lex_before), "r_lex mutated after optimizer.step()!"
    assert torch.equal(model.hippocampus.r_sem, r_sem_before), "r_sem mutated after optimizer.step()!"

    # 2. Zero-gradient one-shot binding & Householder overwrite erase algebra
    with torch.no_grad():
        fast = model.hippocampus.init_fast_state(1, mode="column_major", storage_dtype=torch.float32, device=device)
        fast.gamma_hip = 1.0

        f_wug = torch.randn((1, cfg.d_s), device=device, dtype=torch.float32)
        s_bird = torch.randn((1, cfg.d_concept), device=device, dtype=torch.float32)
        s_lizard = torch.randn((1, cfg.d_concept), device=device, dtype=torch.float32)

        psi_idx, psi_val = model.hippocampus.dg_lex(f_wug)
        phi_idx, phi_val = model.hippocampus.dg_sem(s_bird)

        # Materialize full sparse vector psi_wug in R^{d_h}
        psi_wug = torch.zeros((cfg.d_h,), device=device, dtype=torch.float32)
        psi_wug.scatter_(0, psi_idx[0], psi_val[0])
        phi_bird = torch.zeros((cfg.d_h,), device=device, dtype=torch.float32)
        phi_bird.scatter_(0, phi_idx[0], phi_val[0])

        nnz = int((psi_wug != 0).sum().item())
        psi_norm_err = abs(float(psi_wug.norm().item()) - 1.0)
        print(f"[Test 3] DG psi_wug nnz={nnz}, |norm-1|={psi_norm_err:.3e}")
        assert nnz <= 32 and nnz > 0
        assert psi_norm_err < 1e-5

        # Construct strictly orthogonal address psi_orth (disjoint support => psi_wug @ psi_orth == 0)
        used_cols = set(psi_idx[0].tolist())
        free_cols = [c for c in range(cfg.d_h) if c not in used_cols][: cfg.s_active]
        orth_idx = torch.tensor([free_cols], device=device, dtype=torch.int64)
        orth_raw = torch.rand((1, cfg.s_active), device=device, dtype=torch.float32) + 0.1
        orth_val = orth_raw / orth_raw.norm(dim=-1, keepdim=True)
        psi_orth = torch.zeros((cfg.d_h,), device=device, dtype=torch.float32)
        psi_orth.scatter_(0, orth_idx[0], orth_val[0])
        assert float(torch.dot(psi_wug, psi_orth).item()) == 0.0

        s_orth = torch.randn((1, cfg.d_concept), device=device, dtype=torch.float32)
        phi_orth_idx, phi_orth_val = model.hippocampus.dg_sem(s_orth)
        f_orth = torch.randn((1, cfg.d_s), device=device, dtype=torch.float32)

        beta_one = torch.tensor([1.0], device=device, dtype=torch.float32)

        # Pre-write orthogonal memory
        fast.write_slot(beta_one, orth_idx, orth_val, s_orth, phi_orth_idx, phi_orth_val, f_orth, write_threshold=0.0)
        orth_before = (fast.m_c_from_l.T @ psi_orth).clone()

        # One-shot bidirectional binding: (psi_wug <-> S_bird) and (phi_bird <-> f_wug)
        fast.write_slot(beta_one, psi_idx, psi_val, s_bird, phi_idx, phi_val, f_wug, write_threshold=0.0)

        read_c_bird = fast.m_c_from_l.T @ psi_wug  # [D]
        read_l_wug = fast.m_l_from_c.T @ phi_bird  # [d_s]
        bind_c_err = float((read_c_bird - s_bird[0]).abs().max().item())
        bind_l_err = float((read_l_wug - f_wug[0]).abs().max().item())
        print(f"[Test 3] One-shot bind errs: C<-L={bind_c_err:.3e}, L<-C={bind_l_err:.3e}")
        assert bind_c_err < 1e-5, f"Comprehension bind error too high: {bind_c_err}"
        assert bind_l_err < 1e-5, f"Expression bind error too high: {bind_l_err}"

        # One-shot correction overwrite: same psi_wug -> S_lizard
        phi_liz_idx, phi_liz_val = model.hippocampus.dg_sem(s_lizard)
        fast.write_slot(beta_one, psi_idx, psi_val, s_lizard, phi_liz_idx, phi_liz_val, f_wug, write_threshold=0.0)

        read_c_lizard = fast.m_c_from_l.T @ psi_wug
        overwrite_err = float((read_c_lizard - s_lizard[0]).abs().max().item())
        orth_after = fast.m_c_from_l.T @ psi_orth
        orth_drift = float((orth_after - orth_before).abs().max().item())

        print(f"[Test 3] Overwrite err={overwrite_err:.3e}, Orth drift={orth_drift:.3e}")
        assert overwrite_err < 1e-5, f"Householder erase-write failed! err={overwrite_err}"
        assert orth_drift == 0.0, f"Orthogonal memory interfered! drift={orth_drift}"

    print(
        f"[PASS Test 3] Opt_Overlap=0, Bind_max_err={max(bind_c_err, bind_l_err):.3e}, "
        f"Overwrite_max_err={overwrite_err:.3e}, Orth_drift={orth_drift:.3e}"
    )


# =============================================================================
# Test 4: Strict O(1) Constant Runtime Working Memory & Shape Invariants
# =============================================================================
def test_constant_o1_working_memory():
    device = _setup_deterministic_env(42)

    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    model.eval()

    x_short = torch.randint(32, 127, (1, 512), dtype=torch.uint8, device=device)
    x_long = torch.randint(32, 127, (1, 8192), dtype=torch.uint8, device=device)
    lang = torch.tensor([1], dtype=torch.long, device=device)

    chunk_size = 256
    with torch.no_grad():
        state_short = model.init_carry_state(batch_size=1, device=device, dtype=torch.float32)
        for pos in range(0, x_short.shape[1], chunk_size):
            _, state_short = model.forward_streaming_chunk(
                x_short[:, pos : pos + chunk_size], lang, carry_state=state_short, stage="D"
            )

        state_long = model.init_carry_state(batch_size=1, device=device, dtype=torch.float32)
        for pos in range(0, x_long.shape[1], chunk_size):
            _, state_long = model.forward_streaming_chunk(
                x_long[:, pos : pos + chunk_size], lang, carry_state=state_long, stage="D"
            )

    # Verify locked shapes
    for tag, st in (("T=512", state_short), ("T=8192", state_long)):
        assert st.graph_state.slots.shape == (1, 8, cfg.d_concept), f"{tag}: S shape drifted: {st.graph_state.slots.shape}"
        assert st.graph_state.edges.shape == (1, 8, 8, cfg.d_r), f"{tag}: A shape drifted: {st.graph_state.edges.shape}"
        assert st.pmd_xi.shape == (1, 16, 4), f"{tag}: Xi shape drifted: {st.pmd_xi.shape}"
        _assert_no_dynamic_lists(st, path=f"carry_state({tag})")

    tensors_short = _collect_state_tensors(state_short)
    tensors_long = _collect_state_tensors(state_long)

    numel_short = sum(t.numel() for _, t in tensors_short)
    numel_long = sum(t.numel() for _, t in tensors_long)
    bytes_short = sum(t.numel() * t.element_size() for _, t in tensors_short)
    bytes_long = sum(t.numel() * t.element_size() for _, t in tensors_long)

    print(
        f"[Test 4] T=512: numel={numel_short}, bytes={bytes_short} | "
        f"T=8192: numel={numel_long}, bytes={bytes_long} | "
        f"diff_numel={numel_long - numel_short}, diff_bytes={bytes_long - bytes_short}"
    )
    assert numel_short == numel_long, f"O(1) numel violation: {numel_short} != {numel_long}"
    assert bytes_short == bytes_long, f"O(1) byte size violation: {bytes_short} != {bytes_long}"
    assert [n for n, _ in tensors_short] == [n for n, _ in tensors_long]

    # Also verify model attributes contain zero growing list caches
    for attr_name, val in model.__dict__.items():
        if isinstance(val, list):
            assert len(val) == 0, f"Model attribute {attr_name} grew to len={len(val)}"

    print(f"[PASS Test 4] State_Bytes(T=512)={bytes_short}, State_Bytes(T=8192)={bytes_long}, Diff=0")


# =============================================================================
# Iron Law #1 Cross-Platform Device & Static CUDA Hygiene Guards
# =============================================================================
def test_device_abstraction_and_four_tier_hardware_contract():
    dev = get_deterministic_test_device()
    if is_cuda_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu", "Deterministic sanity tests must NEVER run on MPS"

    h100 = select_hardware_tier("cuda", vram_gb=80.0, device_name="NVIDIA H100 80GB HBM3")
    assert (h100.micro_batch_bytes, h100.grad_accum_steps, h100.use_bf16) == (131_072, 1, True)

    a100_40 = select_hardware_tier("cuda", vram_gb=40.0, device_name="NVIDIA A100-SXM4-40GB")
    assert (a100_40.micro_batch_bytes, a100_40.grad_accum_steps, a100_40.use_bf16) == (65_536, 2, True)

    l4 = select_hardware_tier("cuda", vram_gb=24.0, device_name="NVIDIA L4")
    assert (l4.micro_batch_bytes, l4.grad_accum_steps, l4.use_bf16) == (32_768, 4, True)

    mps_local = select_hardware_tier("mps", vram_gb=0.0, target_tokens_bytes_per_step=8192)
    assert (mps_local.micro_batch_bytes, mps_local.grad_accum_steps, mps_local.use_bf16) == (8192, 1, False)

    assert resolve_micro_batch_plan(a100_40, seq_len=2048) == (32, 2, 131_072)
    assert isinstance(amp_autocast(torch.device("cpu")), contextlib.nullcontext)
    assert isinstance(amp_autocast(torch.device("mps")), contextlib.nullcontext)

    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    snap = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    portable = dict(snap, cuda_rng_state_all=None, mps_rng_state=None)
    restore_rng_state(portable)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1).item())) == expected


def test_no_bare_cuda_calls_outside_device_module():
    scan_dirs = [
        REPO_ROOT / "utils",
        REPO_ROOT / "data",
        REPO_ROOT / "train",
        REPO_ROOT / "tests",
        REPO_ROOT / "gates",
        REPO_ROOT / "dcpa-train-py" / "dcpa",
    ]
    files: list[Path] = []
    if (REPO_ROOT / "smoke_overfit_64.py").exists():
        files.append(REPO_ROOT / "smoke_overfit_64.py")
    for d in scan_dirs:
        if d.exists():
            files.extend(p for p in d.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts)

    allowed = {(REPO_ROOT / "utils" / "device.py").resolve(), Path(__file__).resolve()}
    violations: list[str] = []
    for path in files:
        if path.resolve() in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("cuda", "get_device_properties", "autocast"):
                    violations.append(f"{path}:{node.lineno}: bare .{node.func.attr}() call")
    assert not violations, "Iron Law #1 violations found:\n" + "\n".join(violations)
