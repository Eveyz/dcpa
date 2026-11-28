"""Unit & exact-parity tests for WS-4 (`dcpa/modules/hippocampus_flat.py`)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.interfaces import CarrierGraphState, DCPAFlatConfig
from dcpa.modules.hippocampus_flat import (
    DualMorphemeWhitenedDG,
    HippocampusFlat,
    IntraChunkWriteBuffer,
)


def _init_mock_g2_state(cfg: DCPAFlatConfig, P: int, device: torch.device) -> CarrierGraphState:
    return CarrierGraphState(
        slots=torch.zeros((P, cfg.k_slots, cfg.d_concept), device=device, dtype=torch.float32),
        edges=torch.zeros((P, cfg.k_slots, cfg.k_slots, cfg.d_r), device=device, dtype=torch.float32),
        occ=torch.full((P, cfg.k_slots), cfg.occ_min, device=device, dtype=torch.float32),
        slot0_pinned=torch.zeros((P,), device=device, dtype=torch.bool),
    )


def _make_mock_carrier_step_fn(cfg: DCPAFlatConfig):
    """Deterministic lane-independent mock Carrier G2 step with realistic eviction & slot assignment."""
    K = cfg.k_slots
    d_r = cfg.d_r

    def carrier_step_fn(
        state: CarrierGraphState,
        x_in: torch.Tensor,
        reset: torch.Tensor,
        valid: torch.Tensor,
        pin_trig: torch.Tensor,
        unpin_trig: torch.Tensor,
    ):
        P, D = x_in.shape
        device = x_in.device
        rst_1 = reset.unsqueeze(-1)
        rst_2 = reset.view(P, 1, 1)
        rst_3 = reset.view(P, 1, 1, 1)

        # Reset state on doc boundary BEFORE producing prior_state
        prior_slots = torch.where(rst_2, torch.zeros_like(state.slots), state.slots)
        prior_edges = torch.where(rst_3, torch.zeros_like(state.edges), state.edges)
        prior_occ = torch.where(rst_1, torch.full_like(state.occ, cfg.occ_min), state.occ)
        prior_pin = torch.where(reset, torch.zeros_like(state.slot0_pinned), state.slot0_pinned)

        prior_state = CarrierGraphState(
            slots=prior_slots,
            edges=prior_edges,
            occ=prior_occ,
            slot0_pinned=prior_pin,
        )

        # Choose target slot deterministically from lowest occupancy (respecting Slot-0 pin)
        eff_occ = prior_occ.clone()
        slot_ids = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0)
        eff_occ = torch.where(prior_pin.unsqueeze(1) & (slot_ids == 0), torch.full_like(eff_occ, 1e9), eff_occ)
        target_slot = eff_occ.argmin(dim=-1)  # [P]
        slot_assign = F.one_hot(target_slot, num_classes=K).to(torch.float32)  # [P, K]

        # Evict intensity is high whenever the target slot already holds meaningful occupancy (> 0.20)
        # (triggers Write-Before-Merge on full/occupied slots while remaining 0 on brand-new empty slots)
        evict_gate = torch.sigmoid((prior_occ - 0.18) / 0.08) * slot_assign
        evict_weight = evict_gate * valid.to(torch.float32).unsqueeze(-1)
        # Pre-eviction concept snapshot
        evict_target = prior_slots

        # Update target slot onto closed unit ball ||S_k|| <= 1
        x_unit = F.normalize(x_in.float(), dim=-1, eps=1e-6) * 0.9
        is_tgt = slot_ids == target_slot.unsqueeze(1)
        upd_mask = valid.unsqueeze(1) & is_tgt
        next_slots = torch.where(
            upd_mask.unsqueeze(-1),
            0.3 * prior_slots + 0.7 * x_unit.unsqueeze(1),
            prior_slots * 0.98,
        )
        next_slots = torch.where(valid.view(P, 1, 1), next_slots, prior_slots)

        # Simple contractive edge update from slot inner products
        slot_dots = torch.bmm(next_slots, next_slots.transpose(1, 2)).unsqueeze(-1)  # [P, K, K, 1]
        edge_drive = torch.tanh(slot_dots.expand(P, K, K, d_r) * 0.5)
        next_edges = cfg.edge_decay * prior_edges + (1.0 - cfg.edge_decay) * edge_drive
        next_edges = torch.where(valid.view(P, 1, 1, 1), next_edges, prior_edges)

        # Occupancy increases on written slot and gently relaxes elsewhere
        next_occ = torch.where(
            is_tgt,
            (prior_occ * 0.5 + 0.45).clamp(cfg.occ_min, cfg.occ_max),
            (prior_occ * 0.99).clamp(cfg.occ_min, cfg.occ_max),
        )
        next_occ = torch.where(valid.unsqueeze(-1), next_occ, prior_occ)

        next_pin = (prior_pin | (pin_trig & valid)) & (~unpin_trig)
        next_state = CarrierGraphState(
            slots=next_slots,
            edges=next_edges,
            occ=next_occ,
            slot0_pinned=next_pin,
        )
        return next_state, prior_state, evict_weight, evict_target, slot_assign

    return carrier_step_fn


def _run_textbook_dense_reference_single_doc(
    hip: HippocampusFlat,
    cfg: DCPAFlatConfig,
    x3_doc: torch.Tensor,          # [L, D]
    f_stem_doc: torch.Tensor,      # [L, d_s]
    q_concept_doc: torch.Tensor,   # [L, D]
    sem_norm_doc: torch.Tensor,    # [L]
    mean_info_doc: torch.Tensor,   # [L]
    pin_doc: torch.Tensor,         # [L] bool
    unpin_doc: torch.Tensor,       # [L] bool
    carrier_step_fn,
) -> dict[str, torch.Tensor]:
    """Independent textbook dense `[d_h, D]` & `[d_h, d_s]` sequential Gated-Delta reference for 1 doc."""
    L = x3_doc.shape[0]
    device = x3_doc.device
    d_h = cfg.d_h
    D = cfg.d_concept
    d_s = cfg.d_s
    K = cfg.k_slots
    s_act = cfg.s_active
    gamma = float(cfg.gamma_hip)
    kappa = float(getattr(cfg, "kappa_quench", 0.95))
    w_thresh = float(cfg.hip_write_threshold)

    M_c = torch.zeros((d_h, D), device=device, dtype=torch.float32)
    M_l = torch.zeros((d_h, d_s), device=device, dtype=torch.float32)
    u_lex = torch.zeros((d_h,), device=device, dtype=torch.float32)
    u_sem = torch.zeros((d_h,), device=device, dtype=torch.float32)

    elig_idx = torch.zeros((1, K, s_act), device=device, dtype=torch.int64)
    elig_val = torch.zeros((1, K, s_act), device=device, dtype=torch.float32)
    elig_f = torch.zeros((1, K, d_s), device=device, dtype=torch.float32)
    elig_eta = torch.zeros((1, K), device=device, dtype=torch.float32)

    g2_state = _init_mock_g2_state(cfg, 1, device)
    prev_stem: torch.Tensor | None = None

    out_c_hip = []
    out_c_fused = []
    out_prec_bu = []
    out_ach = []
    out_gamma_hip = []
    out_f_hip = []
    out_rho_out = []
    out_slots_prior = []
    out_edges_prior = []
    out_occ_prior = []
    out_max_beta = []

    for t in range(L):
        reset_t = torch.tensor([t == 0], device=device, dtype=torch.bool)
        valid_t = torch.tensor([True], device=device, dtype=torch.bool)

        # 1. Strictly causal expression readout from M_l, u_sem BEFORE step t writes
        q_idx, q_val = hip.dg.dg_sem(q_concept_doc[t : t + 1])  # [1, s]
        q_vec = torch.zeros((d_h,), device=device, dtype=torch.float32)
        q_vec[q_idx[0]] = q_val[0]
        f_hip_t = q_vec @ M_l  # [d_s]
        g_sem_t = (q_vec.pow(2) * u_sem).sum().clamp(0.0, 1.0)
        rho_out_t = g_sem_t * torch.sigmoid(hip.alpha_out * (g_sem_t - hip.theta_out))
        out_f_hip.append(f_hip_t)
        out_rho_out.append(rho_out_t)

        # 2. Comprehension readout from M_c, u_lex BEFORE step t writes
        psi_idx, psi_val = hip.dg.dg_lex(f_stem_doc[t : t + 1], f_prev=prev_stem, seg_reset=reset_t)
        psi_vec = torch.zeros((d_h,), device=device, dtype=torch.float32)
        psi_vec[psi_idx[0]] = psi_val[0]

        c_hip_t = psi_vec @ M_c  # [D]
        g_hip_t = (psi_vec.pow(2) * u_lex).sum().clamp(0.0, 1.0)
        g_slow_t = torch.sigmoid(
            hip.alpha_1 * (hip.theta_i - mean_info_doc[t].float())
            + hip.alpha_2 * (sem_norm_doc[t].float() - hip.theta_n)
        )
        ach_t = (1.0 - g_slow_t) * (1.0 - g_hip_t)
        mu_t = g_hip_t / (g_slow_t + g_hip_t + cfg.eps)
        c_fused_t = (1.0 - mu_t) * x3_doc[t].float() + mu_t * c_hip_t
        pi_cort_t = torch.ones_like(x3_doc[t].float())
        prec_bu_t = (1.0 - ach_t) * pi_cort_t

        out_c_hip.append(c_hip_t)
        out_c_fused.append(c_fused_t)
        out_prec_bu.append(prec_bu_t)
        out_ach.append(ach_t)
        out_gamma_hip.append(g_hip_t)

        # 3. Carrier G2 step
        g2_state, prior_g2, ev_w, ev_t, slot_info = carrier_step_fn(
            g2_state,
            c_fused_t.unsqueeze(0),
            reset_t,
            valid_t,
            pin_doc[t : t + 1],
            unpin_doc[t : t + 1],
        )
        out_slots_prior.append(prior_g2.slots[0])
        out_edges_prior.append(prior_g2.edges[0])
        out_occ_prior.append(prior_g2.occ[0])

        # 4. Write-Before-Merge sequential dense Gated-Delta across k = 0..K-1
        _, beta = hip._compute_three_factor_beta(
            elig_eta=elig_eta,
            evict_weight=ev_w,
            evict_target=ev_t,
            edges=prior_g2.edges,
            valid=valid_t,
            write_threshold=w_thresh,
        )
        out_max_beta.append(beta[0].amax())

        phi_idx, phi_val = hip.dg.dg_sem(ev_t)  # [1, K, s]
        for k in range(K):
            bk = beta[0, k]
            # Comprehension dense rank-1 update: M_c <- M_c + bk * outer(e_k, target_k - e_k^T M_c)
            e_vec = torch.zeros((d_h,), device=device, dtype=torch.float32)
            e_vec.scatter_add_(0, elig_idx[0, k], elig_val[0, k])
            delta_c = bk * (ev_t[0, k].float() - (e_vec @ M_c))
            M_c = M_c + torch.outer(e_vec, delta_c)
            active_e = (e_vec > 0.0).float()
            u_lex = (u_lex + bk * (1.0 - u_lex) * active_e).clamp(0.0, 1.0)

            # Expression dense rank-1 update: M_l <- M_l + bk * outer(phi_k, f_k - phi_k^T M_l)
            p_vec = torch.zeros((d_h,), device=device, dtype=torch.float32)
            p_vec.scatter_add_(0, phi_idx[0, k], phi_val[0, k])
            delta_l = bk * (elig_f[0, k].float() - (p_vec @ M_l))
            M_l = M_l + torch.outer(p_vec, delta_l)
            active_p = (p_vec > 0.0).float()
            u_sem = (u_sem + bk * (1.0 - u_sem) * active_p).clamp(0.0, 1.0)

        # End-of-step decay
        M_c = gamma * M_c
        M_l = gamma * M_l
        u_lex = gamma * u_lex
        u_sem = gamma * u_sem

        # Update eligibility traces
        elig_idx, elig_val, elig_f, elig_eta = hip._update_eligibility_after_write(
            elig_idx=elig_idx,
            elig_val=elig_val,
            elig_f=elig_f,
            elig_eta=elig_eta,
            beta=beta,
            evict_weight=ev_w,
            psi_idx=psi_idx,
            psi_val=psi_val,
            f_cur=f_stem_doc[t : t + 1],
            ach=ach_t.unsqueeze(0),
            valid=valid_t,
            next_slots=g2_state.slots,
            x_in=c_fused_t.unsqueeze(0),
            slot_assign_or_merge_info=slot_info,
        )
        prev_stem = f_stem_doc[t : t + 1]
        _ = kappa

    return {
        "c_hip": torch.stack(out_c_hip, dim=0),
        "c_fused": torch.stack(out_c_fused, dim=0),
        "prec_bu": torch.stack(out_prec_bu, dim=0),
        "ach": torch.stack(out_ach, dim=0),
        "gamma_hip": torch.stack(out_gamma_hip, dim=0),
        "f_hip": torch.stack(out_f_hip, dim=0),
        "rho_out": torch.stack(out_rho_out, dim=0),
        "slots_prior": torch.stack(out_slots_prior, dim=0),
        "edges_prior": torch.stack(out_edges_prior, dim=0),
        "occ_prior": torch.stack(out_occ_prior, dim=0),
        "max_beta": torch.stack(out_max_beta, dim=0),
    }


def test_dual_morpheme_non_commutative_and_seg_reset() -> None:
    torch.manual_seed(123)
    cfg = DCPAFlatConfig.flat_micro()
    dg = DualMorphemeWhitenedDG(cfg)

    # Check frozen orthogonal buffers are never trainable parameters
    param_ids = {id(p) for p in dg.parameters()}
    assert id(dg.q_rot) not in param_ids
    assert id(dg.r_lex) not in param_ids
    assert id(dg.r_sem) not in param_ids
    assert id(dg.b_homeo) not in param_ids
    assert id(dg.b_homeo_sem) not in param_ids

    # Q_rot is strictly orthogonal (Q^T Q == I) and non-identity
    eye = torch.eye(cfg.d_s)
    assert torch.allclose(dg.q_rot.T @ dg.q_rot, eye, atol=1e-5)
    assert (dg.q_rot - eye).abs().max().item() > 0.1

    a = torch.randn(4, cfg.d_s)
    b = torch.randn(4, cfg.d_s)
    c = torch.randn(4, cfg.d_s)

    # Non-commutativity: bind(a, b) != bind(b, a)
    ab = dg.bind_dual_morpheme(a, b)
    ba = dg.bind_dual_morpheme(b, a)
    assert (ab - ba).norm(dim=-1).min().item() > 1.0

    # DG sparse keys reflect non-commutativity
    idx_ab, val_ab = dg.dg_lex(a, b)
    idx_ba, val_ba = dg.dg_lex(b, a)
    assert not torch.equal(idx_ab, idx_ba)
    assert torch.allclose(val_ab.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(val_ba.norm(dim=-1), torch.ones(4), atol=1e-5)

    # seg_reset=True zeroes previous morpheme cleanly -> independent of previous doc's stem
    reset_all = torch.ones(4, dtype=torch.bool)
    a_rst_b = dg.bind_dual_morpheme(a, b, seg_reset=reset_all)
    a_rst_c = dg.bind_dual_morpheme(a, c, seg_reset=reset_all)
    a_none = dg.bind_dual_morpheme(a, None)
    assert torch.allclose(a_rst_b, a_rst_c, atol=1e-7)
    assert torch.allclose(a_rst_b, a_none, atol=1e-7)

    # Mixed seg_reset mask in batch
    mixed_rst = torch.tensor([True, False, True, False], dtype=torch.bool)
    idx_m1, val_m1 = dg.dg_lex(a, b, seg_reset=mixed_rst)
    idx_m2, val_m2 = dg.dg_lex(a, c, seg_reset=mixed_rst)
    assert torch.equal(idx_m1[0], idx_m2[0]) and torch.allclose(val_m1[0], val_m2[0], atol=1e-7)
    assert torch.equal(idx_m1[2], idx_m2[2]) and torch.allclose(val_m1[2], val_m2[2], atol=1e-7)
    assert not torch.equal(idx_m1[1], idx_m2[1])
    assert not torch.equal(idx_m1[3], idx_m2[3])


def test_residual_whitening_decorrelation_loss_homeostasis_and_freeze() -> None:
    torch.manual_seed(77)
    cfg = DCPAFlatConfig.flat_micro()
    dg = DualMorphemeWhitenedDG(cfg)

    # Step 0 residual whitening starts near identity (exact orthogonal baseline)
    probe = torch.randn(16, cfg.d_s)
    probe_w = dg.apply_whiten(probe)
    rel_diff = (probe_w - probe).norm() / probe.norm()
    assert rel_diff.item() < 2e-2

    # Construct strongly correlated (anisotropic + common mean) stems to induce heavy DG key crosstalk
    common_dir = F.normalize(torch.randn(1, cfg.d_s), dim=-1) * 5.0
    low_rank = torch.randn(128, 4) @ torch.randn(4, cfg.d_s) * 0.12
    stems = common_dir + low_rank + 0.12 * torch.randn(128, cfg.d_s)
    valid_mask = torch.ones(128, dtype=torch.bool)

    def mean_off_diag_crosstalk() -> float:
        with torch.no_grad():
            pi, pv = dg.dg_lex(stems)
            dense = torch.zeros((128, cfg.d_h)).scatter_add(1, pi, pv)
            gram = dense @ dense.T
            off = 1.0 - torch.eye(128)
            return float((gram * off).sum().item() / off.sum().item())

    init_crosstalk = mean_off_diag_crosstalk()
    init_loss = dg.compute_decorrelation_loss(*dg.dg_lex(stems), valid_mask=valid_mask).item()
    assert init_crosstalk > 0.25
    assert init_loss > 0.05

    opt = torch.optim.Adam(dg.whiten_parameters(), lr=5e-2)
    for _ in range(40):
        opt.zero_grad()
        pi, pv = dg.dg_lex(stems)
        loss = dg.compute_decorrelation_loss(pi, pv, valid_mask=valid_mask)
        loss.backward()
        opt.step()

    final_crosstalk = mean_off_diag_crosstalk()
    final_loss = dg.compute_decorrelation_loss(*dg.dg_lex(stems), valid_mask=valid_mask).item()
    assert final_loss < init_loss * 0.5
    assert final_crosstalk < init_crosstalk * 0.7

    # Test >512 subsampling path and all-False valid_mask edge case
    big_stems = torch.randn(700, cfg.d_s)
    big_valid = torch.arange(700) < 600
    big_pi, big_pv = dg.dg_lex(big_stems)
    big_loss = dg.compute_decorrelation_loss(big_pi, big_pv, valid_mask=big_valid, max_samples=512)
    assert torch.isfinite(big_loss) and big_loss.item() >= 0.0

    empty_loss = dg.compute_decorrelation_loss(big_pi, big_pv, valid_mask=torch.zeros(700, dtype=torch.bool))
    assert torch.isfinite(empty_loss) and empty_loss.item() == 0.0

    # Homeostasis updates b_homeo without autograd: hot neurons drop, cold neurons rise
    hot_idx = torch.arange(cfg.s_active, dtype=torch.int64).unsqueeze(0).expand(32, -1)
    before_hot = dg.b_homeo[: cfg.s_active].clone()
    before_cold = dg.b_homeo[cfg.s_active :].clone()
    dg.update_homeostasis(hot_idx, valid_mask=torch.ones(32, dtype=torch.bool))
    assert (dg.b_homeo[: cfg.s_active] < before_hot).all()
    assert (dg.b_homeo[cfg.s_active :] > before_cold).all()
    assert not dg.b_homeo.requires_grad

    # freeze_whiten(True) freezes ALL whitening parameters; freeze_whiten(False) restores them
    dg.freeze_whiten(True)
    for p in dg.whiten_parameters():
        assert p.requires_grad is False
    dg.freeze_whiten(False)
    for p in dg.whiten_parameters():
        assert p.requires_grad is True


@pytest.mark.parametrize(
    "lane_doc_lengths",
    [
        # Case 1: 4 distinct short docs packed into 1 single lane (S=64, C=16 -> crosses 3 chunk boundaries!)
        [[11, 19, 14, 15]],
        # Case 2: Multi-lane [P=4, S=64] with heterogeneous packed docs + invalid tail padding
        [
            [11, 19, 14, 15],
            [25, 22, 17],
            [9, 10, 12, 13, 14],
            [35, 20],
        ],
    ],
)
def test_multi_doc_lane_packing_exact_parity(lane_doc_lengths: list[list[int]]) -> None:
    """Packed multi-doc lane execution MUST match isolated per-doc dense Gated-Delta reference < 1e-5."""
    torch.manual_seed(2026)
    cfg = DCPAFlatConfig.flat_micro()
    P = len(lane_doc_lengths)
    S = cfg.grid_segs  # 64, C = hip_chunk_size = 16 -> 4 chunks!
    D = cfg.d_concept
    d_s = cfg.d_s
    device = torch.device("cpu")

    hip = HippocampusFlat(cfg).to(device).eval()
    carrier_step_fn = _make_mock_carrier_step_fn(cfg)

    x3 = torch.zeros((P, S, D), device=device)
    f_stem = torch.zeros((P, S, d_s), device=device)
    q_concept = torch.zeros((P, S, D), device=device)
    sem_norm = torch.zeros((P, S), device=device)
    mean_info = torch.zeros((P, S), device=device)
    seg_doc_id = torch.zeros((P, S), device=device, dtype=torch.int32)
    seg_reset = torch.zeros((P, S), device=device, dtype=torch.bool)
    seg_valid = torch.zeros((P, S), device=device, dtype=torch.bool)
    seg_pin = torch.zeros((P, S), device=device, dtype=torch.bool)
    seg_unpin = torch.zeros((P, S), device=device, dtype=torch.bool)

    # Populate packed lanes with unique doc IDs and realistic recurring morphemes & queries
    doc_records: list[tuple[int, int, int, int]] = []  # (lane, start, length, doc_id)
    next_doc_id = 0
    for p, lengths in enumerate(lane_doc_lengths):
        cursor = 0
        for L in lengths:
            assert cursor + L <= S
            # Create a small vocabulary of recurring stems within each doc so writes get queried later!
            vocab_stems = torch.randn((4, d_s), device=device)
            vocab_x3 = torch.randn((4, D), device=device)
            idxs = torch.arange(L, device=device) % 4
            f_doc = vocab_stems[idxs] + 0.05 * torch.randn((L, d_s), device=device)
            x_doc = vocab_x3[idxs] + 0.1 * torch.randn((L, D), device=device)
            # Queries deliberately reuse earlier x3 directions so M^{L<-C} retrieves non-trivial targets
            q_doc = vocab_x3[(idxs + 2) % 4] + 0.05 * torch.randn((L, D), device=device)

            x3[p, cursor : cursor + L] = x_doc
            f_stem[p, cursor : cursor + L] = f_doc
            q_concept[p, cursor : cursor + L] = q_doc
            # High surprisal & moderate sem_norm so ACh > 0.5 -> strong writes (beta > 0.25!)
            mean_info[p, cursor : cursor + L] = 3.2 + 0.4 * torch.rand((L,), device=device)
            sem_norm[p, cursor : cursor + L] = 0.6 + 0.3 * torch.rand((L,), device=device)
            seg_doc_id[p, cursor : cursor + L] = next_doc_id
            seg_reset[p, cursor] = True
            seg_valid[p, cursor : cursor + L] = True
            if L > 5:
                seg_pin[p, cursor + 3] = True
            if L > 10:
                seg_unpin[p, cursor + 9] = True

            doc_records.append((p, cursor, L, next_doc_id))
            cursor += L
            next_doc_id += 1

        # Tail padding uses globally distinct negative sentinel doc IDs (ScatterBridgeOut contract)
        for s_pad in range(cursor, S):
            seg_doc_id[p, s_pad] = -(p * S + s_pad + 2)
            seg_reset[p, s_pad] = True
            seg_valid[p, s_pad] = False

    # Run packed multi-doc interleaved comprehension + chunk-triangular expression
    g2_init = _init_mock_g2_state(cfg, P, device)
    hip_init = hip.init_chunk_state(P, device=device)

    with torch.no_grad():
        inter_out = hip.forward_interleaved_with_carrier_g2(
            x3=x3,
            f_stem=f_stem,
            sem_norm=sem_norm,
            mean_info=mean_info,
            seg_doc_id=seg_doc_id,
            seg_reset=seg_reset,
            seg_valid=seg_valid,
            seg_pin_trig=seg_pin,
            seg_unpin_trig=seg_unpin,
            g2_state=g2_init,
            carrier_step_fn=carrier_step_fn,
            hip_enabled=True,
            chunk_state=hip_init,
        )
        f_hip_packed, rho_packed = hip.read_expression_chunk_triangular(
            q_concept=q_concept,
            seg_doc_id=seg_doc_id,
            seg_valid=seg_valid,
            write_events=inter_out.write_events,
            chunk_state_init=hip_init,
        )

    # Confirm non-trivial hippocampal writes & reads actually happened (not trivially 0 == 0!)
    assert inter_out.max_beta_seq.max().item() > 0.15, "Expected real Write-Before-Merge events!"
    assert inter_out.c_hip_seq.abs().max().item() > 1e-2, "Expected non-trivial M^{C<-L} reads!"
    assert f_hip_packed.abs().max().item() > 1e-2, "Expected non-trivial M^{L<-C} reads!"
    assert inter_out.gamma_hip_seq.max().item() > 0.05, "Expected non-trivial familiarity buildup!"

    # Verify EVERY single document against BOTH independent textbook dense reference AND step_streaming!
    max_err_c_hip = 0.0
    max_err_c_fused = 0.0
    max_err_f_hip = 0.0
    max_err_rho = 0.0
    max_err_stream_c = 0.0
    max_err_stream_f = 0.0

    for p, start, L, doc_id in doc_records:
        sl = slice(start, start + L)
        with torch.no_grad():
            ref = _run_textbook_dense_reference_single_doc(
                hip=hip,
                cfg=cfg,
                x3_doc=x3[p, sl],
                f_stem_doc=f_stem[p, sl],
                q_concept_doc=q_concept[p, sl],
                sem_norm_doc=sem_norm[p, sl],
                mean_info_doc=mean_info[p, sl],
                pin_doc=seg_pin[p, sl],
                unpin_doc=seg_unpin[p, sl],
                carrier_step_fn=carrier_step_fn,
            )

        err_c_hip = (inter_out.c_hip_seq[p, sl] - ref["c_hip"]).abs().max().item()
        err_c_fused = (inter_out.c_fused_seq[p, sl] - ref["c_fused"]).abs().max().item()
        err_prec = (inter_out.prec_bu_seq[p, sl] - ref["prec_bu"]).abs().max().item()
        err_ach = (inter_out.ach_seq[p, sl] - ref["ach"]).abs().max().item()
        err_ghip = (inter_out.gamma_hip_seq[p, sl] - ref["gamma_hip"]).abs().max().item()
        err_f_hip = (f_hip_packed[p, sl] - ref["f_hip"]).abs().max().item()
        err_rho = (rho_packed[p, sl] - ref["rho_out"]).abs().max().item()
        err_slots = (inter_out.g2_scan.slots_seq[p, sl] - ref["slots_prior"]).abs().max().item()
        err_beta = (inter_out.max_beta_seq[p, sl] - ref["max_beta"]).abs().max().item()

        max_err_c_hip = max(max_err_c_hip, err_c_hip)
        max_err_c_fused = max(max_err_c_fused, err_c_fused)
        max_err_f_hip = max(max_err_f_hip, err_f_hip)
        max_err_rho = max(max_err_rho, err_rho)

        assert err_c_hip < 1e-5, f"Doc {doc_id} M^(C<-L) c_hip mismatch: {err_c_hip}"
        assert err_c_fused < 1e-5, f"Doc {doc_id} c_fused mismatch: {err_c_fused}"
        assert err_prec < 1e-5, f"Doc {doc_id} prec_bu mismatch: {err_prec}"
        assert err_ach < 1e-5, f"Doc {doc_id} ach mismatch: {err_ach}"
        assert err_ghip < 1e-5, f"Doc {doc_id} gamma_hip mismatch: {err_ghip}"
        assert err_f_hip < 1e-5, f"Doc {doc_id} M^(L<-C) triangular f_hip mismatch: {err_f_hip}"
        assert err_rho < 1e-5, f"Doc {doc_id} rho_out mismatch: {err_rho}"
        assert err_slots < 1e-5, f"Doc {doc_id} G2_prior slots mismatch: {err_slots}"
        assert err_beta < 1e-5, f"Doc {doc_id} max_beta mismatch: {err_beta}"

        # Also verify Gate 6 single-step streaming parity on this doc
        s_g2 = _init_mock_g2_state(cfg, 1, device)
        s_hip = hip.init_chunk_state(1, device=device)
        s_prev: torch.Tensor | None = None
        stream_c_hip = []
        stream_f_hip = []
        stream_rho = []
        with torch.no_grad():
            for t in range(L):
                step_out = hip.step_streaming(
                    x3=x3[p, start + t : start + t + 1],
                    f_stem=f_stem[p, start + t : start + t + 1],
                    prev_stem=s_prev,
                    sem_norm=sem_norm[p, start + t : start + t + 1],
                    mean_info=mean_info[p, start + t : start + t + 1],
                    doc_id=torch.tensor([doc_id], device=device, dtype=torch.int32),
                    reset=torch.tensor([t == 0], device=device, dtype=torch.bool),
                    valid=torch.tensor([True], device=device, dtype=torch.bool),
                    pin_trig=seg_pin[p, start + t : start + t + 1],
                    unpin_trig=seg_unpin[p, start + t : start + t + 1],
                    g2_state=s_g2,
                    hip_state=s_hip,
                    carrier_step_fn=carrier_step_fn,
                    q_concept=q_concept[p, start + t : start + t + 1],
                )
                s_g2 = step_out.next_g2_state
                s_hip = step_out.next_hip_state
                s_prev = step_out.next_prev_stem
                stream_c_hip.append(step_out.read_out.c_hip[0])
                assert step_out.f_hip is not None and step_out.rho_out is not None
                stream_f_hip.append(step_out.f_hip[0])
                stream_rho.append(step_out.rho_out[0])

        err_sc = (torch.stack(stream_c_hip) - ref["c_hip"]).abs().max().item()
        err_sf = (torch.stack(stream_f_hip) - ref["f_hip"]).abs().max().item()
        err_sr = (torch.stack(stream_rho) - ref["rho_out"]).abs().max().item()
        max_err_stream_c = max(max_err_stream_c, err_sc)
        max_err_stream_f = max(max_err_stream_f, err_sf)
        assert err_sc < 1e-5, f"Streaming c_hip parity failed: {err_sc}"
        assert err_sf < 1e-5, f"Streaming f_hip parity failed: {err_sf}"
        assert err_sr < 1e-5, f"Streaming rho_out parity failed: {err_sr}"

    assert max_err_c_hip < 1e-5
    assert max_err_c_fused < 1e-5
    assert max_err_f_hip < 1e-5
    assert max_err_rho < 1e-5
    assert max_err_stream_c < 1e-5
    assert max_err_stream_f < 1e-5


def test_gradient_flow_scalar_gates_and_whitening() -> None:
    """Backward through beta, rho_out, comprehension/expression readouts, and W_whiten has finite non-zero grads."""
    torch.manual_seed(99)
    cfg = DCPAFlatConfig.flat_micro()
    P, S, D, d_s = 2, 32, cfg.d_concept, cfg.d_s
    device = torch.device("cpu")

    hip = HippocampusFlat(cfg).to(device).train()
    carrier_step_fn = _make_mock_carrier_step_fn(cfg)

    # Recurring stems/concepts within two packed docs per lane so both writes & reads occur in-chunk
    vocab_f = torch.randn((3, d_s), device=device)
    vocab_x = torch.randn((3, D), device=device)
    idxs = torch.arange(S, device=device) % 3
    f_stem = (vocab_f[idxs].unsqueeze(0).expand(P, -1, -1) + 0.05 * torch.randn(P, S, d_s)).requires_grad_(True)
    x3 = (vocab_x[idxs].unsqueeze(0).expand(P, -1, -1) + 0.1 * torch.randn(P, S, D)).requires_grad_(True)
    q_concept = (vocab_x[(idxs + 1) % 3].unsqueeze(0).expand(P, -1, -1) + 0.05 * torch.randn(P, S, D)).requires_grad_(True)

    mean_info = torch.full((P, S), 3.4, device=device)
    sem_norm = torch.full((P, S), 0.7, device=device)
    seg_doc_id = torch.zeros((P, S), device=device, dtype=torch.int32)
    seg_doc_id[:, 16:] = 1
    seg_reset = torch.zeros((P, S), device=device, dtype=torch.bool)
    seg_reset[:, 0] = True
    seg_reset[:, 16] = True
    seg_valid = torch.ones((P, S), device=device, dtype=torch.bool)
    seg_pin = torch.zeros((P, S), device=device, dtype=torch.bool)
    seg_unpin = torch.zeros((P, S), device=device, dtype=torch.bool)

    g2_init = _init_mock_g2_state(cfg, P, device)
    hip_init = hip.init_chunk_state(P, device=device)

    inter_out = hip.forward_interleaved_with_carrier_g2(
        x3=x3,
        f_stem=f_stem,
        sem_norm=sem_norm,
        mean_info=mean_info,
        seg_doc_id=seg_doc_id,
        seg_reset=seg_reset,
        seg_valid=seg_valid,
        seg_pin_trig=seg_pin,
        seg_unpin_trig=seg_unpin,
        g2_state=g2_init,
        carrier_step_fn=carrier_step_fn,
        hip_enabled=True,
        chunk_state=hip_init,
    )
    f_hip, rho_out = hip.read_expression_chunk_triangular(
        q_concept=q_concept,
        seg_doc_id=seg_doc_id,
        seg_valid=seg_valid,
        write_events=inter_out.write_events,
        chunk_state_init=hip_init,
    )

    loss = (
        inter_out.c_fused_seq.pow(2).mean()
        + inter_out.prec_bu_seq.pow(2).mean()
        + f_hip.pow(2).mean()
        + rho_out.pow(2).mean()
        + inter_out.mean_beta
        + inter_out.l_dg
    )
    assert torch.isfinite(loss)
    loss.backward()

    # Check ALL trainable parameters (scalar gates and W_whiten) have finite non-zero gradients
    for name, param in hip.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        assert param.grad.abs().max().item() > 0.0, f"Zero gradient for {name}"

    assert f_stem.grad is not None and torch.isfinite(f_stem.grad).all() and f_stem.grad.abs().max().item() > 0.0
    assert x3.grad is not None and torch.isfinite(x3.grad).all() and x3.grad.abs().max().item() > 0.0
    assert q_concept.grad is not None and torch.isfinite(q_concept.grad).all() and q_concept.grad.abs().max().item() > 0.0

    # Fast weights must NEVER be registered as nn.Parameter
    param_ids = {id(p) for p in hip.parameters()}
    for t in (
        inter_out.final_hip_state.m_c_base,
        inter_out.final_hip_state.m_l_base,
        inter_out.final_hip_state.u_lex,
        inter_out.final_hip_state.u_sem,
    ):
        assert id(t) not in param_ids


def test_read_comprehension_from_state_tuple_and_hip_disabled_bypass() -> None:
    torch.manual_seed(11)
    cfg = DCPAFlatConfig.flat_micro()
    P, S, D, d_s = 2, 16, cfg.d_concept, cfg.d_s
    device = torch.device("cpu")
    hip = HippocampusFlat(cfg).to(device).eval()

    # Verify standalone read_comprehension_from_state works with plain 5-tuple intra-chunk buffer
    state = hip.init_chunk_state(P, device=device)
    f_cur = torch.randn(P, d_s)
    psi_idx, psi_val = hip.dg.dg_lex(f_cur)
    C_buf = 4
    k_idx = torch.zeros((P, C_buf, cfg.k_slots, cfg.s_active), dtype=torch.int64)
    k_val = torch.zeros((P, C_buf, cfg.k_slots, cfg.s_active), dtype=torch.float32)
    delta = torch.zeros((P, C_buf, cfg.k_slots, D), dtype=torch.float32)
    k_idx[:, 1, 0, :] = psi_idx
    k_val[:, 1, 0, :] = psi_val
    target_vec = torch.randn(P, D)
    delta[:, 1, 0, :] = target_vec
    # Also put a distractor write from another doc at step 0 and a future write at step 2
    k_idx[:, 0, 0, :] = psi_idx
    k_val[:, 0, 0, :] = psi_val
    delta[:, 0, 0, :] = torch.randn(P, D) * 100.0
    k_idx[:, 2, 0, :] = psi_idx
    k_val[:, 2, 0, :] = psi_val
    delta[:, 2, 0, :] = torch.randn(P, D) * 100.0

    write_doc_id = torch.tensor([[99, 7, 7, 7], [99, 8, 8, 8]], dtype=torch.int32)
    write_step = torch.arange(C_buf, dtype=torch.int64)
    out = hip.read_comprehension_from_state(
        psi_idx=psi_idx,
        psi_val=psi_val,
        c_slow=torch.randn(P, D),
        sem_norm=torch.ones(P),
        mean_info=torch.ones(P) * 2.0,
        cur_doc_id=torch.tensor([7, 8], dtype=torch.int32),
        cur_step=2,
        chunk_state=state,
        intra_buf=(k_idx, k_val, delta, write_doc_id, write_step),
    )
    # Only step 1 (matching doc_id and step < 2) should contribute, decayed by gamma^(2-1)
    expected_c_hip = target_vec * cfg.gamma_hip
    assert torch.allclose(out.c_hip, expected_c_hip, atol=1e-5)

    # Verify hip_enabled=False clean bypass
    carrier_step_fn = _make_mock_carrier_step_fn(cfg)
    x3 = torch.randn(P, S, D)
    f_stem = torch.randn(P, S, d_s)
    q_c = torch.randn(P, S, D)
    zeros_b = torch.zeros((P, S), dtype=torch.bool)
    ones_b = torch.ones((P, S), dtype=torch.bool)
    doc_ids = torch.zeros((P, S), dtype=torch.int32)

    bypass_out = hip.forward_interleaved_with_carrier_g2(
        x3=x3,
        f_stem=f_stem,
        sem_norm=torch.ones(P, S),
        mean_info=torch.ones(P, S),
        seg_doc_id=doc_ids,
        seg_reset=zeros_b,
        seg_valid=ones_b,
        seg_pin_trig=zeros_b,
        seg_unpin_trig=zeros_b,
        g2_state=_init_mock_g2_state(cfg, P, device),
        carrier_step_fn=carrier_step_fn,
        hip_enabled=False,
    )
    assert torch.equal(bypass_out.c_fused_seq, x3)
    assert bypass_out.max_beta_seq.abs().max().item() == 0.0

    f_hip_b, rho_b = hip.read_expression_chunk_triangular(
        q_concept=q_c,
        seg_doc_id=doc_ids,
        seg_valid=ones_b,
        write_events=bypass_out.write_events,
        chunk_state_init=state,
        hip_enabled=False,
    )
    assert f_hip_b.abs().max().item() == 0.0
    assert rho_b.abs().max().item() == 0.0
    _ = IntraChunkWriteBuffer


def test_multi_slot_simultaneous_eviction_merge_and_chunk_continuation_parity() -> None:
    """Stress-test simultaneous K-slot WY solves, merge-pair eligibility inheritance, and chunk continuation."""
    torch.manual_seed(404)
    cfg = DCPAFlatConfig.flat_micro()
    P, S, D, d_s, K, d_r = 2, 64, cfg.d_concept, cfg.d_s, cfg.k_slots, cfg.d_r
    device = torch.device("cpu")
    hip = HippocampusFlat(cfg).to(device).eval()

    def multi_evict_carrier_step(
        state: CarrierGraphState,
        x_in: torch.Tensor,
        reset: torch.Tensor,
        valid: torch.Tensor,
        pin_trig: torch.Tensor,
        unpin_trig: torch.Tensor,
    ):
        B = x_in.shape[0]
        rst_1 = reset.unsqueeze(-1)
        rst_2 = reset.view(B, 1, 1)
        rst_3 = reset.view(B, 1, 1, 1)

        p_slots = torch.where(rst_2, torch.zeros_like(state.slots), state.slots)
        p_edges = torch.where(rst_3, torch.zeros_like(state.edges), state.edges)
        p_occ = torch.where(rst_1, torch.full_like(state.occ, cfg.occ_min), state.occ)
        p_pin = torch.where(reset, torch.zeros_like(state.slot0_pinned), state.slot0_pinned)
        prior_state = CarrierGraphState(slots=p_slots, edges=p_edges, occ=p_occ, slot0_pinned=p_pin)

        # Simultaneous non-zero eviction on ALL occupied slots (> 0.15) to heavily exercise KxK WY solve
        evict_weight = torch.sigmoid((p_occ - 0.14) / 0.06) * valid.to(torch.float32).unsqueeze(-1)
        evict_target = p_slots

        target_slot = p_occ.argmin(dim=-1)
        merge_dst = p_occ.argmax(dim=-1)
        merge_src = target_slot
        merge_active = p_occ.amin(dim=-1) > 0.25

        slot_ids = torch.arange(K, device=x_in.device, dtype=torch.int64).unsqueeze(0)
        is_tgt = slot_ids == target_slot.unsqueeze(1)
        x_unit = F.normalize(x_in.float(), dim=-1, eps=1e-6) * 0.92
        next_slots = torch.where(
            (valid.unsqueeze(1) & is_tgt).unsqueeze(-1),
            0.25 * p_slots + 0.75 * x_unit.unsqueeze(1),
            p_slots * 0.97,
        )
        next_slots = torch.where(valid.view(B, 1, 1), next_slots, p_slots)
        dots = torch.bmm(next_slots, next_slots.transpose(1, 2)).unsqueeze(-1)
        next_edges = cfg.edge_decay * p_edges + (1.0 - cfg.edge_decay) * torch.tanh(dots.expand(B, K, K, d_r) * 0.6)
        next_edges = torch.where(valid.view(B, 1, 1, 1), next_edges, p_edges)
        next_occ = torch.where(
            is_tgt,
            (p_occ * 0.4 + 0.5).clamp(cfg.occ_min, cfg.occ_max),
            (p_occ * 0.98).clamp(cfg.occ_min, cfg.occ_max),
        )
        next_occ = torch.where(valid.unsqueeze(-1), next_occ, p_occ)
        next_pin = (p_pin | (pin_trig & valid)) & (~unpin_trig)
        next_state = CarrierGraphState(slots=next_slots, edges=next_edges, occ=next_occ, slot0_pinned=next_pin)
        info = {
            "slot_assign": F.one_hot(target_slot, num_classes=K).to(torch.float32),
            "win_i": merge_dst,
            "win_j": merge_src,
            "merge_active": merge_active,
        }
        return next_state, prior_state, evict_weight, evict_target, info

    # Lane 0 has a single 64-step long doc (crosses chunk 0->1->2->3); Lane 1 has 2 docs (28 + 36)
    vocab_f = torch.randn((3, d_s))
    vocab_x = torch.randn((3, D))
    idxs = torch.arange(S) % 3
    f_stem = vocab_f[idxs].unsqueeze(0).expand(P, -1, -1) + 0.04 * torch.randn(P, S, d_s)
    x3 = vocab_x[idxs].unsqueeze(0).expand(P, -1, -1) + 0.08 * torch.randn(P, S, D)
    q_concept = vocab_x[(idxs + 1) % 3].unsqueeze(0).expand(P, -1, -1) + 0.04 * torch.randn(P, S, D)
    mean_info = torch.full((P, S), 3.3)
    sem_norm = torch.full((P, S), 0.7)
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_doc_id[1, :28] = 1
    seg_doc_id[1, 28:] = 2
    seg_reset = torch.zeros((P, S), dtype=torch.bool)
    seg_reset[:, 0] = True
    seg_reset[1, 28] = True
    seg_valid = torch.ones((P, S), dtype=torch.bool)
    zeros_b = torch.zeros((P, S), dtype=torch.bool)

    g2_0 = _init_mock_g2_state(cfg, P, device)
    hip_0 = hip.init_chunk_state(P, device=device)

    with torch.no_grad():
        full_out = hip.forward_interleaved_with_carrier_g2(
            x3=x3,
            f_stem=f_stem,
            sem_norm=sem_norm,
            mean_info=mean_info,
            seg_doc_id=seg_doc_id,
            seg_reset=seg_reset,
            seg_valid=seg_valid,
            seg_pin_trig=zeros_b,
            seg_unpin_trig=zeros_b,
            g2_state=g2_0,
            carrier_step_fn=multi_evict_carrier_step,
            chunk_state=hip_0,
        )
        full_f, full_rho = hip.read_expression_chunk_triangular(
            q_concept=q_concept,
            seg_doc_id=seg_doc_id,
            seg_valid=seg_valid,
            write_events=full_out.write_events,
            chunk_state_init=hip_0,
        )

        # Verify multiple slots actually wrote simultaneously at the same step!
        assert (full_out.write_events.beta > 0.05).sum(dim=-1).max().item() >= 3

        # Compare all 3 docs against textbook dense reference
        for p, sl in ((0, slice(0, 64)), (1, slice(0, 28)), (1, slice(28, 64))):
            ref = _run_textbook_dense_reference_single_doc(
                hip=hip,
                cfg=cfg,
                x3_doc=x3[p, sl],
                f_stem_doc=f_stem[p, sl],
                q_concept_doc=q_concept[p, sl],
                sem_norm_doc=sem_norm[p, sl],
                mean_info_doc=mean_info[p, sl],
                pin_doc=zeros_b[p, sl],
                unpin_doc=zeros_b[p, sl],
                carrier_step_fn=multi_evict_carrier_step,
            )
            assert (full_out.c_hip_seq[p, sl] - ref["c_hip"]).abs().max().item() < 1e-5
            assert (full_out.c_fused_seq[p, sl] - ref["c_fused"]).abs().max().item() < 1e-5
            assert (full_f[p, sl] - ref["f_hip"]).abs().max().item() < 1e-5
            assert (full_rho[p, sl] - ref["rho_out"]).abs().max().item() < 1e-5

        # Also verify two-half chunk continuation [0:32] -> [32:64] matches full [0:64] to < 1e-5!
        part1 = hip.forward_interleaved_with_carrier_g2(
            x3=x3[:, :32],
            f_stem=f_stem[:, :32],
            sem_norm=sem_norm[:, :32],
            mean_info=mean_info[:, :32],
            seg_doc_id=seg_doc_id[:, :32],
            seg_reset=seg_reset[:, :32],
            seg_valid=seg_valid[:, :32],
            seg_pin_trig=zeros_b[:, :32],
            seg_unpin_trig=zeros_b[:, :32],
            g2_state=g2_0,
            carrier_step_fn=multi_evict_carrier_step,
            chunk_state=hip_0,
        )
        part2 = hip.forward_interleaved_with_carrier_g2(
            x3=x3[:, 32:],
            f_stem=f_stem[:, 32:],
            sem_norm=sem_norm[:, 32:],
            mean_info=mean_info[:, 32:],
            seg_doc_id=seg_doc_id[:, 32:],
            seg_reset=seg_reset[:, 32:],
            seg_valid=seg_valid[:, 32:],
            seg_pin_trig=zeros_b[:, 32:],
            seg_unpin_trig=zeros_b[:, 32:],
            g2_state=part1.g2_scan.final_state,
            carrier_step_fn=multi_evict_carrier_step,
            chunk_state=part1.final_hip_state,
            init_prev_stem=f_stem[:, 31],
        )
        assert (part2.c_hip_seq - full_out.c_hip_seq[:, 32:]).abs().max().item() < 1e-5
        assert (part2.c_fused_seq - full_out.c_fused_seq[:, 32:]).abs().max().item() < 1e-5


def test_static_red_lines_source_audit() -> None:
    """Audit `hippocampus_flat.py` AST to guarantee ZERO forbidden hot-path calls & zero graph_carrier import."""
    src_path = _TRAIN_PY / "dcpa" / "modules" / "hippocampus_flat.py"
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    forbidden_attrs = {"nonzero", "repeat_interleave", "item", "cpu", "tolist"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
            pytest.fail(f"Forbidden attribute call `.{node.attr}` found at line {node.lineno} in hippocampus_flat.py")
        if isinstance(node, ast.ImportFrom) and node.module and "graph_carrier" in node.module:
            pytest.fail(f"Forbidden import of graph_carrier at line {node.lineno}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "graph_carrier" in alias.name:
                    pytest.fail(f"Forbidden import of graph_carrier at line {node.lineno}")
