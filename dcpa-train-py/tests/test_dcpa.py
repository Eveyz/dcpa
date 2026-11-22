"""Comprehensive pytest suite verifying DCPA architectural invariants and all 4 modules."""

from __future__ import annotations

import random

import pytest
import torch

from dcpa.config import DCPAConfig
from dcpa.data.synthetic import (
    generate_dyck_sequence,
    generate_fast_mapping_episode,
    generate_grid_physics_episode,
)
from dcpa.model import DCPAModel
from dcpa.modules.hippocampus import ColumnMajorFastWeights, CompactHouseholderState, EligibilityState
from dcpa.ops.ragged import build_cu_seqlens, ragged_local_offsets, ragged_segment_cumsum, ragged_segment_softmax
from dcpa.ops.rope import apply_holographic_rope
from dcpa.ops.scan import parallel_diag_scan, parallel_stack_scan, sequential_diag_scan_ref
from dcpa.train.curriculum import CurriculumSchedule
from dcpa.train.monitor import AntiCollapseMonitor


def test_commandment_invariants():
    # Commandment #1: V must strictly be 256
    with pytest.raises(ValueError):
        DCPAConfig(vocab_size=32000)
    # Commandment #2: K must be <= 12
    with pytest.raises(ValueError):
        DCPAConfig(k_slots=16)
    # Variance hinge weight must be strictly > 0
    with pytest.raises(ValueError):
        DCPAConfig(lambda_var=0.0)


def test_rope_exact_orthogonality_and_inverse():
    torch.manual_seed(0)
    x = torch.randn(4, 192)
    pos = torch.tensor([1.0, 5.0, 12.0, 16.0])
    y = apply_holographic_rope(x, pos, inverse=False)
    # Strict norm preservation (SO(d_s))
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    # Exact inverse R(-tau) R(tau) == I
    x_rec = apply_holographic_rope(y, pos, inverse=True)
    assert (x - x_rec).abs().max().item() < 1e-5


def test_parallel_stack_scan_matches_sequential():
    torch.manual_seed(1)
    B, T = 2, 64
    logits = torch.randn(B, T, 3)
    probs = torch.softmax(logits, dim=-1)
    v_tag = torch.tanh(torch.randn(B, T, 4))

    xi_par = parallel_stack_scan(probs[..., 0], probs[..., 1], probs[..., 2], v_tag)

    # Sequential reference
    xi_cur = torch.zeros(B, 16, 4)
    seq_list = []
    for t in range(T):
        p_push = probs[:, t, 0:1].unsqueeze(-1)
        p_pop = probs[:, t, 1:2].unsqueeze(-1)
        p_hold = probs[:, t, 2:3].unsqueeze(-1)
        pushed = torch.cat([v_tag[:, t : t + 1, :], xi_cur[:, :-1, :]], dim=1)
        popped = torch.cat([xi_cur[:, 1:, :], torch.zeros(B, 1, 4)], dim=1)
        xi_cur = p_hold * xi_cur + p_push * pushed + p_pop * popped
        seq_list.append(xi_cur)
    xi_seq = torch.stack(seq_list, dim=1)
    assert (xi_par - xi_seq).abs().max().item() < 1e-5


def test_compact_householder_matches_column_major_bit_exact():
    """Verify Engineering Doc 2.4: CompactHouseholderState (W_max=32) == dense ColumnMajorFastWeights."""
    torch.manual_seed(2)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)

    compact = model.hippocampus.init_fast_state(1, mode="compact")
    dense = model.hippocampus.init_fast_state(1, mode="column_major", storage_dtype=torch.float32)
    assert isinstance(compact, CompactHouseholderState)
    assert isinstance(dense, ColumnMajorFastWeights)

    # Perform 8 sequential writes including an overwrite on key 0!
    f_keys = [torch.randn(1, cfg.d_s) for _ in range(7)]
    c_vals = [torch.randn(1, cfg.d_concept) for _ in range(8)]
    write_seq = [0, 1, 2, 3, 4, 5, 6, 0]  # Last write overwrites key 0!

    for step_idx, k_i in enumerate(write_seq):
        compact.advance_step()
        dense.advance_step()
        e_idx, e_val = model.hippocampus.dg_lex(f_keys[k_i])
        phi_idx, phi_val = model.hippocampus.dg_sem(c_vals[step_idx])
        beta = torch.tensor([0.85 if step_idx < 7 else 1.0])
        compact.write_slot(beta, e_idx, e_val, c_vals[step_idx], phi_idx, phi_val, f_keys[k_i], write_threshold=0.0)
        dense.write_slot(beta, e_idx, e_val, c_vals[step_idx], phi_idx, phi_val, f_keys[k_i], write_threshold=0.0)

    # Query every single key and assert bit-level equivalence (< 1e-5)
    for k_i in range(7):
        e_idx, e_val = model.hippocampus.dg_lex(f_keys[k_i])
        c_comp = compact.read_c_from_l(e_idx, e_val)
        c_dens = dense.read_c_from_l(e_idx, e_val)
        assert (c_comp - c_dens).abs().max().item() < 1e-5

        g_comp = compact.read_familiarity_lex(e_idx, e_val)
        g_dens = dense.read_familiarity_lex(e_idx, e_val)
        assert (g_comp - g_dens).abs().max().item() < 1e-5


def test_asymmetric_eligibility_inheritance_preserves_orthogonality():
    """Commandment #8: NEVER average eligibility traces; strictly inherit argmax eta verbatim."""
    torch.manual_seed(3)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)
    elig = EligibilityState.zeros(1, cfg.k_slots, cfg.s_active, cfg.d_s, torch.device("cpu"), torch.float32)

    f_novel = torch.randn(1, cfg.d_s)
    psi_idx, psi_val = model.hippocampus.dg_lex(f_novel)
    elig.e_idx[0, 2] = psi_idx[0]
    elig.e_val[0, 2] = psi_val[0]
    elig.f_tilde[0, 2] = f_novel[0]
    elig.eta[0, 2] = 0.95

    f_fam = torch.randn(1, cfg.d_s)
    cand_idx, cand_val = model.hippocampus.dg_lex(f_fam)
    # Merge slot 2 (novel, eta=0.95) with candidate slot K (familiar, ACh=0.02)
    out = model.hippocampus.inherit_eligibility_on_merge(
        elig,
        cand_idx=cand_idx,
        cand_val=cand_val,
        cand_f=f_fam,
        cand_ach=torch.tensor([0.02]),
        win_i=torch.tensor([2]),
        win_j=torch.tensor([cfg.k_slots]),
    )
    assert torch.equal(out.e_idx[0, 2], psi_idx[0])
    assert torch.allclose(out.e_val[0, 2], psi_val[0])
    assert abs(out.e_val[0, 2].norm().item() - 1.0) < 1e-5
    assert abs(out.eta[0, 2].item() - cfg.lambda_eta * 0.95) < 1e-5


def test_fast_weights_never_registered_as_optimizer_parameters():
    """Commandment #7: Hippocampal fast weight matrices MUST NOT be in model.parameters()."""
    model = DCPAModel(DCPAConfig.micro())
    param_names = {n for n, _ in model.named_parameters()}
    assert "hippocampus.r_lex" not in param_names
    assert "hippocampus.r_sem" not in param_names
    for bad in ("m_c_from_l", "m_l_from_c", "u_lex", "u_sem"):
        assert not any(bad in n for n in param_names)


def test_all_four_developmental_stages_forward_backward():
    torch.manual_seed(4)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)
    raw = torch.randint(32, 126, (2, 48), dtype=torch.uint8)
    lang = torch.tensor([1, 4], dtype=torch.long)

    for stage, use_cq in [("A", False), ("B", False), ("C", True), ("D", True)]:
        model.set_stage_freeze(stage)
        model.zero_grad(set_to_none=True)
        out = model(raw, lang, stage=stage, tau_s=0.5, tau_m=0.5, use_broca_cq=use_cq)
        assert torch.isfinite(out.loss_total), f"Non-finite loss in Stage {stage}"
        out.loss_total.backward()
        if stage == "D":
            assert model.sgsc.embed.weight.grad is None
            assert model.sgsc.theta_raw.grad is None


def test_curriculum_and_monitor_and_synthetic_generators():
    sched = CurriculumSchedule(total_steps=100_000)
    assert sched.get(0).stage == "A"
    assert sched.get(15_000).stage == "B"
    assert sched.get(55_000).stage == "C"
    assert sched.get(85_000).stage == "D"

    mon = AntiCollapseMonitor()
    d = mon.observe({"c_slow_std_mean": 0.2, "merge_entropy_nats": 0.05}, stage="B", step_in_stage=10)
    assert d.lambda_var_override == 5.0
    assert d.apply_wsem_spectral_norm is True
    assert d.merge_kl_weight == 0.02

    rng = random.Random(123)
    phys = generate_grid_physics_episode(rng)
    assert "[WORLD_INIT]" in phys
    dyck, closes = generate_dyck_sequence(rng, max_depth=12, min_bytes=512, rust_flavored=True)
    assert len(closes) >= 12 and len(dyck.encode("utf-8")) >= 512
    ep = generate_fast_mapping_episode(rng, lang="en")
    assert ep.pseudoword in ep.text and ep.plural_form in ep.text
