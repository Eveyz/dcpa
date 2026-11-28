"""Unit, gradient-safety, compile, zero-drift plugin, and Gate 6 streaming parity tests for WS-3."""

from __future__ import annotations

import ast
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.interfaces import (
    CarrierGraphState,
    DCPAFlatConfig,
    PluginExpertPack,
)
from dcpa.modules.cortical_moe import (
    CorticalStackFlat,
    DecoupledPluggableMoE,
)
from dcpa.modules.graph_carrier import ContractiveGraphCarrier


def _micro_cfg(**overrides) -> DCPAFlatConfig:
    return replace(DCPAFlatConfig.flat_micro(), **overrides)


# ---------------------------------------------------------------------------
# 1. ContractiveGraphCarrier: 64-step & 512-step long-doc backward + norms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guard_mode", ["none", "router_sg", "soft_clip"])
def test_carrier_64_and_512_step_backward_and_closed_ball(guard_mode: str) -> None:
    torch.manual_seed(42)
    cfg = _micro_cfg(carrier_grad_guard=guard_mode, grid_segs=64, c_scan=16)
    carrier = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)
    carrier.train()

    P, D, K = 2, cfg.d_concept, cfg.k_slots

    # --- A. 64-step single long document: verify >= 0.05 gradient retention & unit-ball norms ---
    S64 = 64
    x64 = torch.randn((P, S64, D), requires_grad=True)
    reset64 = torch.zeros((P, S64), dtype=torch.bool)
    reset64[:, 0] = True
    valid64 = torch.ones((P, S64), dtype=torch.bool)

    out64 = carrier.forward_scan(x64, reset64, valid64, chunk_size=16)
    assert out64.slots_seq.shape == (P, S64, K, D)
    assert out64.edges_seq.shape == (P, S64, K, K, cfg.d_r)
    assert out64.occ_seq.shape == (P, S64, K) and out64.occ_seq.dtype == torch.float32
    assert out64.evict_weight is not None and out64.evict_weight.shape == (P, S64, K)
    assert out64.evict_target is not None and out64.evict_target.shape == (P, S64, K, D)
    assert out64.empty_write_rate is not None

    # Closed unit ball strictly <= 1.0001 everywhere, and active slots in [0.15, 1.00]
    norms64 = out64.slots_seq.detach().norm(dim=-1)
    final_norms64 = out64.final_state.slots.detach().norm(dim=-1)
    assert torch.isfinite(norms64).all()
    assert float(norms64.max()) <= 1.0001
    assert float(final_norms64.max()) <= 1.0001
    active64 = out64.final_state.occ.detach() > (cfg.occ_min + 0.05)
    assert bool(active64.any())
    assert float(final_norms64[active64].min()) >= 0.15
    assert float(out64.final_state.edges.detach().abs().max()) < 1.0
    assert float(out64.final_state.occ.detach().min()) >= cfg.occ_min - 1e-6
    assert float(out64.final_state.occ.detach().max()) <= cfg.occ_max + 1e-6
    # Gate C empty-slot direct write rate on fresh doc >= 85%
    assert float(out64.empty_write_rate.detach()) >= 0.85

    # Loss at final state to measure 64-step temporal gradient retention back to early steps
    carrier.zero_grad(set_to_none=True)
    loss64 = out64.final_state.slots.pow(2).sum() + 0.1 * out64.final_state.edges.pow(2).sum()
    loss64.backward()
    assert x64.grad is not None and torch.isfinite(x64.grad).all()
    for p in carrier.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    # Check 64-step slot gradient retention ratio >= 0.05 across state carry
    g_early = x64.grad[:, :8].norm()
    g_late = x64.grad[:, -8:].norm().clamp_min(1e-12)
    # Also directly probe d(final_slots)/d(init_state.slots) retention over 64 steps
    carrier.zero_grad(set_to_none=True)
    init_st = carrier.init_state(P, device=x64.device, dtype=x64.dtype)
    s0_probe = init_st.slots.detach().clone().requires_grad_(True)
    st_probe = CarrierGraphState(
        slots=s0_probe,
        edges=init_st.edges.detach(),
        occ=torch.full((P, K), 0.70, dtype=torch.float32),
        slot0_pinned=torch.zeros((P,), dtype=torch.bool),
    )
    reset_none = torch.zeros((P, S64), dtype=torch.bool)
    out_probe = carrier.forward_scan(x64.detach(), reset_none, valid64, init_state=st_probe, chunk_size=16)
    g_out = torch.randn_like(out_probe.final_state.slots)
    g_out = g_out / g_out.norm().clamp_min(1e-6)
    (out_probe.final_state.slots * g_out).sum().backward()
    assert s0_probe.grad is not None and torch.isfinite(s0_probe.grad).all()
    retention_64 = float(s0_probe.grad.norm() / g_out.norm())
    assert retention_64 >= 0.05, f"guard={guard_mode} 64-step retention too low: {retention_64:.4f} (early/late={float(g_early/g_late):.4f})"

    # --- B. 512-step continuous long document stress test (Gate B) ---
    S512 = 512
    x512 = torch.randn((1, S512, D), requires_grad=True)
    reset512 = torch.zeros((1, S512), dtype=torch.bool)
    reset512[:, 0] = True
    valid512 = torch.ones((1, S512), dtype=torch.bool)

    carrier.zero_grad(set_to_none=True)
    out512 = carrier.forward_scan(x512, reset512, valid512, chunk_size=16, use_checkpoint=True)
    norms512 = out512.slots_seq.detach().norm(dim=-1)
    assert torch.isfinite(norms512).all()
    assert float(norms512.max()) <= 1.0001
    final_norms512 = out512.final_state.slots.detach().norm(dim=-1)
    assert float(final_norms512.min()) >= 0.15, f"Slot norm collapsed over 512 steps: {final_norms512}"
    assert float(final_norms512.max()) <= 1.0001

    loss512 = out512.slots_seq.pow(2).mean() + out512.final_state.slots.pow(2).mean()
    loss512.backward()
    assert x512.grad is not None and torch.isfinite(x512.grad).all()
    total_param_gn = torch.sqrt(
        sum(p.grad.float().pow(2).sum() for p in carrier.parameters() if p.grad is not None)
    )
    assert torch.isfinite(total_param_gn)
    assert float(total_param_gn) < 500.0, f"guard={guard_mode} 512-step grad exploded: {float(total_param_gn):.2f}"
    assert float(x512.grad.norm()) < 500.0, f"guard={guard_mode} 512-step input grad exploded: {float(x512.grad.norm()):.2f}"


def test_carrier_g1_post_vs_g2_prior_causality_and_valid_freeze() -> None:
    torch.manual_seed(7)
    cfg = _micro_cfg(grid_segs=32, c_scan=16)
    g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False)
    g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)
    g1.eval()
    g2.eval()

    P, S, D = 2, 32, cfg.d_concept
    x_a = torch.randn((P, S, D))
    x_b = x_a.clone()
    x_b[:, 12:] = torch.randn((P, S - 12, D))  # diverge at step 12

    reset = torch.zeros((P, S), dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 20] = True  # doc boundary at step 20
    valid = torch.ones((P, S), dtype=torch.bool)
    valid[:, 26:] = False  # padded tail

    with torch.no_grad():
        out_g2_a = g2.forward_scan(x_a, reset, valid, chunk_size=16)
        out_g2_b = g2.forward_scan(x_b, reset, valid, chunk_size=16)
        out_g1_a = g1.forward_scan(x_a, reset, valid, chunk_size=16)

    # G2 returns strictly causal prior: step 12 output MUST NOT depend on x[:, 12]!
    assert torch.equal(out_g2_a.slots_seq[:, :13], out_g2_b.slots_seq[:, :13])
    assert torch.equal(out_g2_a.edges_seq[:, :13], out_g2_b.edges_seq[:, :13])
    assert torch.equal(out_g2_a.occ_seq[:, :13], out_g2_b.occ_seq[:, :13])

    # At doc reset step 20, G2 prior must equal G_init (zero cross-doc leak!)
    init_g2 = g2.init_state(P, device=x_a.device, dtype=x_a.dtype)
    assert torch.allclose(out_g2_a.slots_seq[:, 20], init_g2.slots, atol=1e-6)
    assert torch.allclose(out_g2_a.occ_seq[:, 20], init_g2.occ, atol=1e-6)

    # Padded tail (`seg_valid=False` for s >= 26) freezes state bit-exactly
    assert torch.equal(out_g1_a.slots_seq[:, 25], out_g1_a.slots_seq[:, 31])
    assert torch.equal(out_g1_a.final_state.slots, out_g1_a.slots_seq[:, 25])
    assert out_g1_a.evict_weight is None
    assert out_g2_a.evict_weight is not None
    assert float(out_g2_a.evict_weight[:, 26:].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 2. G2 Slot-0 Pinning (and G1 ignoring pin triggers)
# ---------------------------------------------------------------------------


def test_g2_slot0_pinning_bit_exact_and_g1_ignores_pin() -> None:
    torch.manual_seed(11)
    cfg = _micro_cfg(grid_segs=64, window_size=32, c_scan=16)
    g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False)
    g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)
    g1.eval()
    g2.eval()

    P, S, D = 2, 64, cfg.d_concept
    x = torch.randn((P, S, D))
    reset = torch.zeros((P, S), dtype=torch.bool)
    reset[:, 0] = True
    reset[1, 36] = True  # lane 1 resets at step 36 while still pinned
    valid = torch.ones((P, S), dtype=torch.bool)

    pin_trig = torch.zeros((P, S), dtype=torch.bool)
    pin_trig[:, 4] = True  # pin Slot-0 after step 4 on both lanes
    unpin_trig = torch.zeros((P, S), dtype=torch.bool)
    unpin_trig[0, 30] = True  # lane 0 explicitly unpins at step 30

    with torch.no_grad():
        out_g1 = g1.forward_scan(x, reset, valid, seg_pin_trig=pin_trig, seg_unpin_trig=unpin_trig, chunk_size=16)
        out_g2 = g2.forward_scan(x, reset, valid, seg_pin_trig=pin_trig, seg_unpin_trig=unpin_trig, chunk_size=16)

    # G1 MUST ignore pin triggers completely
    assert not bool(out_g1.final_state.slot0_pinned.any())

    # G2: step 5 prior holds the Slot-0 written at end of step 4
    # While pinned (steps 5..29 on lane 0, steps 5..35 on lane 1), Slot-0 is BIT-EXACTLY preserved
    # even under full-load merges (S >> K=8) and occ[:, 0] >= 0.85!
    pinned_vec_lane0 = out_g2.slots_seq[0, 5, 0]
    pinned_occ_lane0 = out_g2.occ_seq[0, 5, 0]
    assert float(pinned_occ_lane0) >= 0.85 - 1e-6
    for s in range(6, 30):
        assert torch.equal(out_g2.slots_seq[0, s, 0], pinned_vec_lane0), f"Lane 0 Slot-0 drifted at step {s}"
        assert torch.equal(out_g2.occ_seq[0, s, 0], pinned_occ_lane0), f"Lane 0 occ[0] drifted at step {s}"

    pinned_vec_lane1 = out_g2.slots_seq[1, 5, 0]
    for s in range(6, 36):
        assert torch.equal(out_g2.slots_seq[1, s, 0], pinned_vec_lane1), f"Lane 1 Slot-0 drifted at step {s}"
        assert float(out_g2.occ_seq[1, s, 0]) >= 0.85 - 1e-6

    # Verify Slot-0 was NEVER marked as eviction victim while pinned (steps 5..29)
    assert out_g2.evict_weight is not None
    assert float(out_g2.evict_weight[0, 5:30, 0].abs().max()) == 0.0
    assert float(out_g2.evict_weight[1, 5:36, 0].abs().max()) == 0.0

    # After unpin at step 30 on lane 0, Slot-0 resumes normal evolution and slot0_pinned is False
    assert not bool(out_g2.final_state.slot0_pinned[0])
    assert not torch.equal(out_g2.slots_seq[0, 45, 0], pinned_vec_lane0)

    # After reset at step 36 on lane 1, Slot-0 resets cleanly and slot0_pinned is False
    assert not bool(out_g2.final_state.slot0_pinned[1])
    init_g2 = g2.init_state(P, device=x.device, dtype=x.dtype)
    assert torch.allclose(out_g2.slots_seq[1, 36], init_g2.slots[1], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. torch.compile(backend="aot_eager", fullgraph=True, dynamic=False)
# ---------------------------------------------------------------------------


def test_compile_fullgraph_carrier_chunk_and_cortical_stack() -> None:
    torch.manual_seed(19)
    cfg = _micro_cfg(grid_lanes=2, grid_segs=64, window_size=32, c_scan=16, use_local_window_layers=True)
    P, S, D, d_s = cfg.grid_lanes, cfg.grid_segs, cfg.d_concept, cfg.d_s

    # A. Compile ContractiveGraphCarrier.forward_chunk (G2 with pinning & prior)
    g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)
    g2.train()
    compiled_chunk = torch.compile(g2.forward_chunk, backend="aot_eager", fullgraph=True, dynamic=False)

    st = g2.init_state(P)
    x_c = torch.randn((P, cfg.c_scan, D), requires_grad=True)
    r_c = torch.zeros((P, cfg.c_scan), dtype=torch.bool)
    r_c[:, 0] = True
    v_c = torch.ones((P, cfg.c_scan), dtype=torch.bool)
    p_c = torch.zeros((P, cfg.c_scan), dtype=torch.bool)
    p_c[:, 3] = True
    u_c = torch.zeros((P, cfg.c_scan), dtype=torch.bool)

    out_c = compiled_chunk(st, x_c, r_c, v_c, p_c, u_c)
    loss_c = out_c[0].pow(2).mean() + out_c[3].slots.pow(2).mean()
    loss_c.backward()
    assert x_c.grad is not None and torch.isfinite(x_c.grad).all()

    # B. Compile all 3 stages of CorticalStackFlat (train AND eval, plus with a mounted plugin pack!)
    stack = CorticalStackFlat(cfg)
    g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False)

    comp_pre = torch.compile(stack.forward_pre_g1, backend="aot_eager", fullgraph=True, dynamic=False)
    comp_post = torch.compile(stack.forward_post_g1, backend="aot_eager", fullgraph=True, dynamic=False)
    comp_pred = torch.compile(stack.forward_predict_and_refine, backend="aot_eager", fullgraph=True, dynamic=False)

    x0 = torch.randn((P, S, D), requires_grad=True)
    f_stem = torch.randn((P, S, d_s))
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_start_depth = torch.zeros((P, S), dtype=torch.int32)
    seg_end_depth = torch.zeros((P, S), dtype=torch.int32)
    seg_l_reg = torch.randn((P, S, cfg.d_l))
    seg_valid = torch.ones((P, S), dtype=torch.bool)
    seg_reset = torch.zeros((P, S), dtype=torch.bool)
    seg_reset[:, 0] = True

    stack.train()
    x1, _ = comp_pre(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)
    g1_out = g1.forward_scan(x1, seg_reset, seg_valid)
    x3, c_kv_sem, _ = comp_post(x1, g1_out, seg_l_reg, seg_valid)
    assert c_kv_sem is not None
    g2_out = g2.forward_scan(x3, seg_reset, seg_valid)
    x_seed = torch.cat([torch.zeros((P, 1, D)), x3[:, :-1]], dim=1)
    pred_out = comp_pred(x_seed, x_seed, g2_out, c_kv_sem, seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid)

    loss_s = pred_out.z_hat.pow(2).mean()
    loss_s.backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()
    assert stack.w_unc.weight.grad is not None and float(stack.w_unc.weight.grad.abs().sum()) > 0.0

    # Eval mode fullgraph compile as well (including with a mounted PluginExpertPack!)
    pack = PluginExpertPack(
        pack_id="compile_plug",
        proto_keys=F.normalize(torch.randn((cfg.n_layers, 2, D)), dim=-1),
        tau=torch.full((cfg.n_layers, 2), cfg.plugin_tau_min),
        alpha=torch.ones((cfg.n_layers, 2)),
        w_gu=torch.randn((cfg.n_layers, 2, 2 * 32, D)) * 0.05,
        w_d=torch.zeros((cfg.n_layers, 2, D, 32)),
    )
    stack.mount_expert_pack(pack)
    comp_pred_plug = torch.compile(stack.forward_predict_and_refine, backend="aot_eager", fullgraph=True, dynamic=False)
    stack.eval()
    with torch.no_grad():
        pred_eval = comp_pred_plug(
            x_seed.detach(), x_seed.detach(), g2_out, c_kv_sem.detach(),
            seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid,
        )
    assert torch.isfinite(pred_eval.z_hat).all()
    stack.unmount_expert_pack("compile_plug")


def test_bfloat16_fp32_islands_carrier_and_stack() -> None:
    torch.manual_seed(55)
    cfg = _micro_cfg(grid_lanes=2, grid_segs=64, window_size=32, c_scan=16)
    P, S, D, d_s = cfg.grid_lanes, cfg.grid_segs, cfg.d_concept, cfg.d_s
    g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False).to(dtype=torch.bfloat16)
    g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True).to(dtype=torch.bfloat16)
    stack = CorticalStackFlat(cfg).to(dtype=torch.bfloat16)
    stack.train()

    x0 = torch.randn((P, S, D), dtype=torch.bfloat16, requires_grad=True)
    f_stem = torch.randn((P, S, d_s), dtype=torch.bfloat16)
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_reset = torch.zeros((P, S), dtype=torch.bool)
    seg_reset[:, 0] = True
    seg_start_depth = torch.zeros((P, S), dtype=torch.int32)
    seg_end_depth = torch.zeros((P, S), dtype=torch.int32)
    seg_l_reg = torch.randn((P, S, cfg.d_l), dtype=torch.bfloat16)
    seg_valid = torch.ones((P, S), dtype=torch.bool)

    x1, _ = stack.forward_pre_g1(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)
    g1_out = g1.forward_scan(x1, seg_reset, seg_valid)
    assert g1_out.occ_seq.dtype == torch.float32
    x3, sem, _ = stack.forward_post_g1(x1, g1_out, seg_l_reg, seg_valid)
    g2_out = g2.forward_scan(x3, seg_reset, seg_valid)
    assert g2_out.occ_seq.dtype == torch.float32
    seed = torch.cat([torch.zeros((P, 1, D), dtype=torch.bfloat16), x3[:, :-1]], dim=1)
    pred = stack.forward_predict_and_refine(
        seed, seed, g2_out, sem, seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid
    )
    assert pred.z_hat.dtype == torch.bfloat16
    assert pred.u_n.dtype == torch.float32
    pred.z_hat.float().pow(2).mean().backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()


# ---------------------------------------------------------------------------
# 4. R2 DecoupledPluggableMoE & CorticalStackFlat Zero-Drift Plugin Hot-Mount
# ---------------------------------------------------------------------------


def test_pluggable_moe_zero_drift_hot_mount_and_balance_bias() -> None:
    torch.manual_seed(23)
    cfg = _micro_cfg(grid_lanes=2, grid_segs=64, window_size=32)
    D = cfg.d_concept

    # A. Isolated DecoupledPluggableMoE exact 0.0 drift test
    moe = DecoupledPluggableMoE(
        d_model=D,
        shared_hidden=cfg.shared_hidden,
        routed_hidden=cfg.routed_hidden,
        num_experts=cfg.num_experts,
        top_k=1,
        plugin_top_k=2,
        plugin_tau_min=0.35,
    )
    moe.eval()

    # Construct orthogonal unit basis directions u_target and u_non_target (<u_t, u_nt> == 0)
    u_target = torch.zeros(D)
    u_target[0] = 1.0
    u_non_target = torch.zeros(D)
    u_non_target[1] = 1.0

    # Inputs in RMSNorm space (moe.norm has weight == 1 initially, so norm(u) aligns with u)
    x_batch = torch.stack([u_non_target * 2.0, -u_target * 1.5, u_target * 2.0], dim=0)  # [3, D]
    with torch.no_grad():
        y_base, counts_base = moe(x_batch)

    # Mount plugin pack keyed on +u_target with tau = 0.50 (> 0.35) and non-zero w_d
    pack = PluginExpertPack(
        pack_id="sql_pack_v1",
        proto_keys=u_target.view(1, 1, D).expand(cfg.n_layers, 1, D).clone(),
        tau=torch.full((cfg.n_layers, 1), 0.50),
        alpha=torch.full((cfg.n_layers, 1), 2.0),
        w_gu=torch.randn((cfg.n_layers, 1, 2 * 32, D)) * 0.2,
        w_d=torch.randn((cfg.n_layers, 1, D, 32)) * 0.2,
    )
    moe.mount_expert_pack(pack, layer_idx=0)
    with torch.no_grad():
        y_mounted, _ = moe(x_batch)

    # Non-target inputs (index 0: orthogonal cos=0.0; index 1: opposite cos=-1.0) MUST have EXACTLY 0.0 drift!
    drift_orth = float((y_mounted[0] - y_base[0]).abs().max())
    drift_opp = float((y_mounted[1] - y_base[1]).abs().max())
    assert drift_orth == 0.0, f"Expected bit-exact 0.0 drift on orthogonal input, got {drift_orth}"
    assert drift_opp == 0.0, f"Expected bit-exact 0.0 drift on opposite input, got {drift_opp}"

    # Matching target input (index 2: cos=1.0 > tau=0.50) MUST get non-zero residual injection!
    delta_target = float((y_mounted[2] - y_base[2]).abs().max())
    assert delta_target > 1e-4, f"Expected non-zero injection on target input, got {delta_target}"

    # Pure non-target batch exercises the 0-FLOP early skip path in eval mode
    with torch.no_grad():
        y_nt_base, _ = moe(x_batch[:2])
    assert torch.equal(y_nt_base, y_base[:2])

    # Unmount restores bit-exact baseline on ALL inputs (including target!)
    assert moe.unmount_expert_pack("sql_pack_v1") is True
    with torch.no_grad():
        y_unmounted, _ = moe(x_batch)
    assert torch.equal(y_unmounted, y_base)

    # Verify update_balance_bias shifts b_bal for selection only without contaminating g_base
    b_before = moe.b_bal.clone()
    moe.update_balance_bias(torch.tensor([100.0, 0.0, 0.0, 0.0]))
    assert not torch.equal(moe.b_bal, b_before)
    assert float(moe.b_bal[0]) < float(moe.b_bal[1])
    assert abs(float(moe.b_bal.mean())) < 1e-6

    # B. Full 8-layer CorticalStackFlat hot-mount / unmount test
    stack = CorticalStackFlat(cfg)
    stack.eval()
    P, S, d_s = 1, cfg.grid_segs, cfg.d_s
    x0 = torch.randn((P, S, D))
    f_stem = torch.randn((P, S, d_s))
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_end_depth = torch.zeros((P, S), dtype=torch.int32)
    seg_valid = torch.ones((P, S), dtype=torch.bool)

    with torch.no_grad():
        x1_base, _ = stack.forward_pre_g1(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)

    # Choose a prototype key guaranteed to have max cos < 0.90 < tau = 0.94 across all tokens
    # (random vector with tau=0.95 -> strictly in dead-zone for random inputs!)
    pack_high_tau = PluginExpertPack(
        pack_id="domain_strict",
        proto_keys=F.normalize(torch.randn((cfg.n_layers, 2, D)), dim=-1),
        tau=torch.full((cfg.n_layers, 2), 0.94),
        alpha=torch.full((cfg.n_layers, 2), 2.0),
        w_gu=torch.randn((cfg.n_layers, 2, 2 * 32, D)) * 0.2,
        w_d=torch.randn((cfg.n_layers, 2, D, 32)) * 0.2,
    )
    stack.mount_expert_pack(pack_high_tau)
    with torch.no_grad():
        x1_with_pack, _ = stack.forward_pre_g1(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)
    assert float((x1_with_pack - x1_base).abs().max()) == 0.0

    # Mount a second pack aligned directly with layer 0's normalized activations -> non-zero injection!
    l0 = stack.layers[0]
    with torch.no_grad():
        # Capture exact pre-MoE normalized activations at layer 0
        # Use a low-cosine key on other tokens and exact token-0 key on token 0
        x0_norm_dir = F.normalize(x1_base[0, 0].float(), dim=-1)
    # Build matching key using exact pre-MoE input of layer 0 by setting enable_window=False
    cfg_nowin = _micro_cfg(grid_lanes=1, grid_segs=32, window_size=32, use_local_window_layers=False)
    stack_nw = CorticalStackFlat(cfg_nowin)
    stack_nw.eval()
    x_nw = torch.randn((1, 32, D))
    with torch.no_grad():
        x1_nw_base, _ = stack_nw.forward_pre_g1(
            x_nw, torch.zeros((1, 32, d_s)),
            torch.zeros((1, 32), dtype=torch.int32),
            torch.zeros((1, 32), dtype=torch.int32),
            torch.ones((1, 32), dtype=torch.bool),
        )
        h0_hat = F.normalize(stack_nw.layers[0].moe.norm(x_nw[0, 0:1]).float(), dim=-1)  # [1, D]
    pack_match = PluginExpertPack(
        pack_id="match_tok0",
        proto_keys=h0_hat.unsqueeze(0).expand(cfg_nowin.n_layers, 1, D).clone(),
        tau=torch.full((cfg_nowin.n_layers, 1), 0.85),
        alpha=torch.full((cfg_nowin.n_layers, 1), 3.0),
        w_gu=torch.randn((cfg_nowin.n_layers, 1, 2 * 32, D)) * 0.25,
        w_d=torch.randn((cfg_nowin.n_layers, 1, D, 32)) * 0.25,
    )
    stack_nw.mount_expert_pack(pack_match)
    with torch.no_grad():
        x1_nw_mounted, _ = stack_nw.forward_pre_g1(
            x_nw, torch.zeros((1, 32, d_s)),
            torch.zeros((1, 32), dtype=torch.int32),
            torch.zeros((1, 32), dtype=torch.int32),
            torch.ones((1, 32), dtype=torch.bool),
        )
    assert float((x1_nw_mounted[0, 0] - x1_nw_base[0, 0]).abs().max()) > 1e-4
    assert float((x1_nw_mounted[0, 1:] - x1_nw_base[0, 1:]).abs().max()) == 0.0
    assert stack_nw.unmount_expert_pack("match_tok0") is True
    with torch.no_grad():
        x1_nw_restored, _ = stack_nw.forward_pre_g1(
            x_nw, torch.zeros((1, 32, d_s)),
            torch.zeros((1, 32), dtype=torch.int32),
            torch.zeros((1, 32), dtype=torch.int32),
            torch.ones((1, 32), dtype=torch.bool),
        )
    assert torch.equal(x1_nw_restored, x1_nw_base)


# ---------------------------------------------------------------------------
# 5. R1 Switch (use_local_window_layers x share_stem_ring) + Strict Causality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_win", [True, False])
@pytest.mark.parametrize("share_stem", [False, True])
def test_r1_switches_forward_backward_and_strict_causality(use_win: bool, share_stem: bool) -> None:
    torch.manual_seed(31)
    cfg = _micro_cfg(
        grid_lanes=2,
        grid_segs=64,
        window_size=32,
        c_scan=16,
        use_local_window_layers=use_win,
        share_stem_ring=share_stem,
    )
    P, S, D, d_s = cfg.grid_lanes, cfg.grid_segs, cfg.d_concept, cfg.d_s
    g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False)
    g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)
    stack = CorticalStackFlat(cfg)
    stack.train()

    x0 = torch.randn((P, S, D), requires_grad=True)
    f_stem = torch.randn((P, S, d_s), requires_grad=True)
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_doc_id[:, 32:] = 1  # second doc at s=32
    seg_reset = torch.zeros((P, S), dtype=torch.bool)
    seg_reset[:, 0] = True
    seg_reset[:, 32] = True
    seg_start_depth = torch.randint(0, 6, (P, S), dtype=torch.int32)
    seg_end_depth = torch.randint(0, 6, (P, S), dtype=torch.int32)
    seg_l_reg = torch.randn((P, S, cfg.d_l))
    seg_valid = torch.ones((P, S), dtype=torch.bool)

    x1, aux1 = stack.forward_pre_g1(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)
    g1_out = g1.forward_scan(x1, seg_reset, seg_valid)
    x3, c_kv_sem, aux2 = stack.forward_post_g1(x1, g1_out, seg_l_reg, seg_valid)

    if share_stem:
        assert c_kv_sem is None
        ring_kv = f_stem if use_win else None
    else:
        assert c_kv_sem is not None and c_kv_sem.shape == (P, S, cfg.d_sem_ring)
        ring_kv = c_kv_sem if use_win else None

    g2_out = g2.forward_scan(x3, seg_reset, seg_valid)
    x_seed = torch.cat([torch.zeros((P, 1, D)), x3[:, :-1]], dim=1)
    x_seed = torch.where(seg_reset.unsqueeze(-1), torch.zeros_like(x_seed), x_seed)

    pred = stack.forward_predict_and_refine(
        x_seed=x_seed,
        x3_prev=x_seed,
        g2_prior=g2_out,
        ring_kv=ring_kv,
        seg_doc_id=seg_doc_id,
        seg_start_depth=seg_start_depth,
        seg_end_depth=seg_end_depth,
        seg_l_reg=seg_l_reg,
        seg_valid=seg_valid,
    )
    assert pred.z_hat.shape == (P, S, D)
    assert pred.u_n.shape == (P, S)
    assert pred.refine_mask.shape == (P, S)
    assert len(aux1["expert_counts"]) == 1
    assert len(aux2["expert_counts"]) == 2
    assert len(pred.expert_counts) == 5

    # Verify update_all_balance_biases runs cleanly across all 8 layers
    stack.update_all_balance_biases()

    loss = pred.z_hat.pow(2).mean()
    loss.backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()
    if use_win:
        assert f_stem.grad is not None and torch.isfinite(f_stem.grad).all()
    # R5: u_n MUST receive non-zero gradient from L7–L8 MoD gate!
    assert stack.w_unc.weight.grad is not None and float(stack.w_unc.weight.grad.abs().sum()) > 0.0
    assert stack.w_unc.bias.grad is not None and float(stack.w_unc.bias.grad.abs().sum()) > 0.0

    # Strict causality check for L4/L6 when window is enabled: mutating ring_kv at s >= 15
    # MUST NOT affect predictions at s <= 15 (strict past 1 <= Δs <= W)!
    if use_win and ring_kv is not None:
        stack.eval()
        with torch.no_grad():
            pred_a = stack.forward_predict_and_refine(
                x_seed.detach(), x_seed.detach(), g2_out, ring_kv.detach(),
                seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid,
            )
            ring_b = ring_kv.detach().clone()
            ring_b[:, 15:] = torch.randn_like(ring_b[:, 15:])
            pred_b = stack.forward_predict_and_refine(
                x_seed.detach(), x_seed.detach(), g2_out, ring_b,
                seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid,
            )
        assert float((pred_a.z_hat[:, :16] - pred_b.z_hat[:, :16]).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 6. R5 Check: u_n Gradient + Eval Threshold vs Single-Step Streaming Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_win", [True, False])
def test_r5_eval_threshold_matches_single_step_streaming_parity(use_win: bool) -> None:
    torch.manual_seed(77)
    cfg = _micro_cfg(
        grid_lanes=1,
        grid_segs=64,
        window_size=32,
        c_scan=16,
        use_local_window_layers=use_win,
        share_stem_ring=False,
    )
    P, S, W, D, d_s = 1, cfg.grid_segs, cfg.window_size, cfg.d_concept, cfg.d_s

    g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False)
    g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)
    stack = CorticalStackFlat(cfg)

    # First run 1 train step to verify tau_think EMA updates and u_n gets gradient
    stack.train()
    tau_before = float(stack.tau_think)
    x0_tr = torch.randn((P, S, D))
    f_stem_tr = torch.randn((P, S, d_s))
    doc_tr = torch.zeros((P, S), dtype=torch.int32)
    rst_tr = torch.zeros((P, S), dtype=torch.bool)
    rst_tr[:, 0] = True
    sd_tr = torch.randint(0, 5, (P, S), dtype=torch.int32)
    ed_tr = torch.randint(0, 5, (P, S), dtype=torch.int32)
    reg_tr = torch.randn((P, S, cfg.d_l))
    val_tr = torch.ones((P, S), dtype=torch.bool)

    x1_tr, _ = stack.forward_pre_g1(x0_tr, f_stem_tr, doc_tr, ed_tr, val_tr)
    g1_tr = g1.forward_scan(x1_tr, rst_tr, val_tr)
    x3_tr, sem_tr, _ = stack.forward_post_g1(x1_tr, g1_tr, reg_tr, val_tr)
    g2_tr = g2.forward_scan(x3_tr, rst_tr, val_tr)
    seed_tr = torch.cat([torch.zeros((P, 1, D)), x3_tr[:, :-1]], dim=1)
    pred_tr = stack.forward_predict_and_refine(
        seed_tr, seed_tr, g2_tr, sem_tr if use_win else None, doc_tr, sd_tr, ed_tr, reg_tr, val_tr
    )
    pred_tr.z_hat.pow(2).mean().backward()
    assert stack.w_unc.weight.grad is not None and float(stack.w_unc.weight.grad.abs().sum()) > 0.0
    assert float(stack.tau_think) != tau_before

    # Now switch to eval mode and set tau_think near median of u_n so BOTH active (> tau) and
    # inactive (<= tau) branches are exercised across the 64-step single-document stream!
    g1.eval()
    g2.eval()
    stack.eval()

    x0 = torch.randn((P, S, D))
    f_stem = torch.randn((P, S, d_s))
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_reset = torch.zeros((P, S), dtype=torch.bool)
    seg_reset[:, 0] = True
    seg_start_depth = torch.randint(0, 6, (P, S), dtype=torch.int32)
    seg_end_depth = torch.randint(0, 6, (P, S), dtype=torch.int32)
    seg_l_reg = torch.randn((P, S, cfg.d_l))
    seg_valid = torch.ones((P, S), dtype=torch.bool)

    with torch.no_grad():
        x1_batch, _ = stack.forward_pre_g1(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)
        g1_batch = g1.forward_scan(x1_batch, seg_reset, seg_valid)
        x3_batch, sem_batch, _ = stack.forward_post_g1(x1_batch, g1_batch, seg_l_reg, seg_valid)
        g2_batch = g2.forward_scan(x3_batch, seg_reset, seg_valid)
        x_seed_batch = torch.cat([torch.zeros((P, 1, D)), x3_batch[:, :-1]], dim=1)
        pred_probe = stack.forward_predict_and_refine(
            x_seed_batch, x_seed_batch, g2_batch, sem_batch if use_win else None,
            seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid,
        )
        # Set tau_think to exact median of u_n so ~50% refine and ~50% skip
        stack.tau_think.copy_(pred_probe.u_n.median())
        pred_batch = stack.forward_predict_and_refine(
            x_seed_batch, x_seed_batch, g2_batch, sem_batch if use_win else None,
            seg_doc_id, seg_start_depth, seg_end_depth, seg_l_reg, seg_valid,
        )

    assert bool(pred_batch.refine_mask.any()) and bool((~pred_batch.refine_mask).any())

    # Step-by-step O(1) streaming execution over all S=64 segments
    ring_stem = torch.zeros((P, W, d_s))
    ring_sem = torch.zeros((P, W, cfg.d_sem_ring))
    ring_pos_l1 = torch.zeros((P, W), dtype=torch.int64)
    ring_dep_l1 = torch.zeros((P, W), dtype=torch.int32)
    ring_val_l1 = torch.zeros((P, W), dtype=torch.bool)

    ring_pos_past = torch.zeros((P, W), dtype=torch.int64)
    ring_dep_past = torch.zeros((P, W), dtype=torch.int32)
    ring_val_past = torch.zeros((P, W), dtype=torch.bool)

    g1_st = g1.init_state(P)
    g2_st = g2.init_state(P)
    prev_x3 = torch.zeros((P, D))

    max_diff_x1 = 0.0
    max_diff_x3 = 0.0
    max_diff_z6 = 0.0
    max_diff_zhat = 0.0
    max_diff_u = 0.0

    with torch.no_grad():
        for s in range(S):
            cur_pos = torch.full((P,), s, dtype=torch.int64)
            # Push current f_stem into L1 ring (0 <= Δs < W includes current s!)
            ring_stem = torch.cat([ring_stem[:, 1:, :], f_stem[:, s : s + 1, :]], dim=1)
            ring_pos_l1 = torch.cat([ring_pos_l1[:, 1:], cur_pos.unsqueeze(1)], dim=1)
            ring_dep_l1 = torch.cat([ring_dep_l1[:, 1:], seg_end_depth[:, s : s + 1]], dim=1)
            ring_val_l1 = torch.cat([ring_val_l1[:, 1:], torch.ones((P, 1), dtype=torch.bool)], dim=1)

            x1_s = stack.step_streaming_pre_g1(
                x0[:, s], ring_stem, ring_pos_l1, ring_dep_l1, ring_val_l1, cur_pos, seg_end_depth[:, s]
            )
            max_diff_x1 = max(max_diff_x1, float((x1_s - x1_batch[:, s]).abs().max()))

            g1_st, g1_out_s, _, _, _, _ = g1.step_single(g1_st, x1_s, seg_reset[:, s], seg_valid[:, s])
            x3_s, sem_s = stack.step_streaming_post_g1(x1_s, g1_out_s, seg_l_reg[:, s])
            max_diff_x3 = max(max_diff_x3, float((x3_s - x3_batch[:, s]).abs().max()))

            g2_st, g2_prior_s, _, _, _, _ = g2.step_single(g2_st, x3_s, seg_reset[:, s], seg_valid[:, s])

            # Predict segment s strictly from past ring (< s) & G2_prior!
            pred_s = stack.step_streaming(
                x_seed_step=prev_x3,
                x3_prev_step=prev_x3,
                g2_prior_state=g2_prior_s,
                ring_kv=ring_sem if use_win else None,
                ring_pos=ring_pos_past,
                ring_depth=ring_dep_past,
                ring_valid=ring_val_past,
                cur_pos=cur_pos,
                cur_start_depth=seg_start_depth[:, s],
                l_reg_step=seg_l_reg[:, s],
            )
            max_diff_z6 = max(max_diff_z6, float((pred_s.z_hat_6 - pred_batch.z_hat_6[:, s]).abs().max()))
            max_diff_zhat = max(max_diff_zhat, float((pred_s.z_hat - pred_batch.z_hat[:, s]).abs().max()))
            max_diff_u = max(max_diff_u, float((pred_s.u_n - pred_batch.u_n[:, s]).abs().max()))
            assert torch.equal(pred_s.refine_mask, pred_batch.refine_mask[:, s])

            # Advance strict-past ring with completed segment s
            assert sem_s is not None
            ring_sem = torch.cat([ring_sem[:, 1:, :], sem_s.unsqueeze(1)], dim=1)
            ring_pos_past = torch.cat([ring_pos_past[:, 1:], cur_pos.unsqueeze(1)], dim=1)
            ring_dep_past = torch.cat([ring_dep_past[:, 1:], seg_end_depth[:, s : s + 1]], dim=1)
            ring_val_past = torch.cat([ring_val_past[:, 1:], torch.ones((P, 1), dtype=torch.bool)], dim=1)
            prev_x3 = x3_s

    assert max_diff_x1 < 1e-5, f"L1 streaming parity diff={max_diff_x1:.2e}"
    assert max_diff_x3 < 1e-5, f"L2-L3 streaming parity diff={max_diff_x3:.2e}"
    assert max_diff_z6 < 1e-5, f"L4-L6 streaming parity diff={max_diff_z6:.2e}"
    assert max_diff_u < 1e-5, f"u_n streaming parity diff={max_diff_u:.2e}"
    assert max_diff_zhat < 1e-5, f"L7-L8 streaming parity diff={max_diff_zhat:.2e}"


# ---------------------------------------------------------------------------
# 7. Static Red-Line AST Guard on WS-3 Deliverables
# ---------------------------------------------------------------------------


def test_ws3_static_red_lines_ast() -> None:
    forbidden_attrs = {"nonzero", "repeat_interleave", "item", "cpu", "tolist", "cuda", "get_device_properties", "autocast"}
    for rel in (
        "dcpa-train-py/dcpa/modules/graph_carrier.py",
        "dcpa-train-py/dcpa/modules/cortical_moe.py",
    ):
        path = _REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden_attrs:
                    hits.append(f"{rel}:{node.lineno}: forbidden .{node.func.attr}()")
        assert not hits, "Red-line violations:\n" + "\n".join(hits)
