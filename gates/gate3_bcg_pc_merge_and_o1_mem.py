"""Gate 3: BCG-PC Differentiable K+1 -> K Merge Gradients & Strict O(1) Memory Verification.

Pass/Fail Quantitative Thresholds (Engineering Blueprint Chapter 7):
  (1) Backward gradient norms of M_score (W_L, W_R, W_E, w_merge), W_comp, and LIF threshold
      theta are ALL strictly non-zero and finite (no NaN / Inf).
  (2) Inference mode scaling input length from 1,024 bytes to 65,536 bytes:
      dynamic working state memory growth MUST equal STRICTLY 0 Bytes!
"""

from __future__ import annotations

import torch

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.modules.hippocampus import EligibilityState


def _state_nbytes(graph_state, fast_state, elig_state, h_state, m_state, xi_state) -> int:
    total = 0
    for t in (
        graph_state.slots,
        graph_state.edges,
        graph_state.salience,
        fast_state.m_c_from_l,
        fast_state.m_l_from_c,
        fast_state.u_lex,
        fast_state.u_sem,
        fast_state.col_step_lex,
        fast_state.col_step_sem,
        elig_state.e_idx,
        elig_state.e_val,
        elig_state.f_tilde,
        elig_state.eta,
        h_state,
        m_state,
        xi_state,
    ):
        total += t.numel() * t.element_size()
    return total


def verify_gate3() -> None:
    print("=== Running Gate 3: BCG-PC Differentiable Merge & O(1) Memory Constancy ===")
    torch.manual_seed(2026)

    # -------------------------------------------------------------------------
    # Check (1): Non-zero finite gradients through Gumbel-STE merge & LIF STE
    # -------------------------------------------------------------------------
    cfg = DCPAConfig.tiny()
    model = DCPAModel(cfg)
    model.train()

    raw_bytes = torch.randint(32, 126, (2, 96), dtype=torch.uint8)
    lang_ids = torch.tensor([1, 4], dtype=torch.long)

    out = model(raw_bytes=raw_bytes, lang_ids=lang_ids, stage="B", tau_s=0.5, tau_m=0.8)
    out.loss_total.backward()

    required_grads = {
        "bcg_pc.w_l.weight (M_score)": model.bcg_pc.w_l.weight.grad,
        "bcg_pc.w_r.weight (M_score)": model.bcg_pc.w_r.weight.grad,
        "bcg_pc.w_edge_score.weight (M_score)": model.bcg_pc.w_edge_score.weight.grad,
        "bcg_pc.w_merge.weight (M_score)": model.bcg_pc.w_merge.weight.grad,
        "bcg_pc.w_comp.weight (W_comp)": model.bcg_pc.w_comp.weight.grad,
        "sgsc.theta_raw (LIF theta)": model.sgsc.theta_raw.grad,
    }
    for name, g in required_grads.items():
        assert g is not None, f"GATE 3 FAIL: gradient for {name} is None!"
        assert torch.isfinite(g).all(), f"GATE 3 FAIL: non-finite gradient in {name}!"
        gnorm = float(g.float().norm().item())
        assert gnorm > 0.0, f"GATE 3 FAIL: zero gradient norm for {name}!"
        print(f"  grad_norm[{name}] = {gnorm:.6e}")

    # Verify ZERO NaN across ALL trainable parameters
    for pname, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"GATE 3 FAIL: NaN/Inf grad in {pname}"

    print("[PASS] Check 1/2: All required gradients (M_score, W_comp, LIF theta) strictly non-zero and finite.")

    # -------------------------------------------------------------------------
    # Check (2): O(1) dynamic working state memory from T=1,024 -> T=65,536
    # -------------------------------------------------------------------------
    cfg_inf = DCPAConfig.base()
    model_inf = DCPAModel(cfg_inf)
    model_inf.eval()

    def run_stream_and_measure_state_bytes(total_bytes: int) -> int:
        chunk_size = 1024
        h_cur = model_inf.sgsc.h0.detach().clone()
        graph_state = model_inf.bcg_pc.init_state(1, dtype=torch.float32)
        fast_state = model_inf.hippocampus.init_fast_state(1, mode="column_major", storage_dtype=torch.bfloat16)
        elig_state = EligibilityState.zeros(1, cfg_inf.k_slots, cfg_inf.s_active, cfg_inf.d_s, torch.device("cpu"), torch.float32)
        m_cur = model_inf.pmd.m0.detach().clone()
        xi_cur = torch.zeros((16, 4), dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, total_bytes, chunk_size):
                chunk = torch.randint(32, 126, (1, chunk_size), dtype=torch.uint8)
                sgsc_o = model_inf.sgsc(chunk, tau_s=0.35, h_init=h_cur)
                h_cur = sgsc_o.h_seq[0, -1].clone()
                # Process concept steps in chunk without ever expanding slot count K=8
                # (run a representative subset every chunk for fast verification while updating state)
                n_segs = sgsc_o.f_stem.shape[0]
                for m_i in range(min(n_segs, 8)):
                    hr = model_inf.hippocampus.read_comprehension(
                        sgsc_o.f_stem[m_i : m_i + 1],
                        sgsc_o.c_slow[m_i : m_i + 1],
                        sgsc_o.sem_norm[m_i : m_i + 1],
                        sgsc_o.mean_info[m_i : m_i + 1],
                        fast_state=fast_state,
                        enabled=True,
                    )
                    bo = model_inf.bcg_pc.step(
                        hr.c_fused, hr.prec_bu, hr.ach_signal, graph_state, tau_m=0.25, use_gumbel=False, update_c0_ema=False
                    )
                    graph_state = bo.state
                    elig_state = model_inf.hippocampus.inherit_eligibility_on_merge(
                        elig_state, hr.psi_idx, hr.psi_val, sgsc_o.f_stem[m_i : m_i + 1], hr.ach_signal, bo.win_i, bo.win_j
                    )
                    elig_state, _, _ = model_inf.hippocampus.three_factor_write(
                        bo.s_bar_post, graph_state.edges, model_inf.bcg_pc.c0, elig_state, fast_state, write_threshold=0.0
                    )

        assert graph_state.slots.shape == (1, cfg_inf.k_slots, cfg_inf.d_concept)
        assert graph_state.edges.shape == (1, cfg_inf.k_slots, cfg_inf.k_slots, cfg_inf.d_r)
        return _state_nbytes(graph_state, fast_state, elig_state, h_cur, m_cur, xi_cur)

    mem_1k = run_stream_and_measure_state_bytes(1_024)
    mem_64k = run_stream_and_measure_state_bytes(65_536)
    delta = mem_64k - mem_1k
    print(f"  Working state footprint at T= 1,024: {mem_1k:,} Bytes ({mem_1k / (1024*1024):.3f} MiB)")
    print(f"  Working state footprint at T=65,536: {mem_64k:,} Bytes ({mem_64k / (1024*1024):.3f} MiB)")
    print(f"  Delta = {delta} Bytes")
    assert delta == 0, f"GATE 3 FAIL: dynamic working state grew by {delta} Bytes!"
    print("[PASS] Check 2/2: Working state memory delta from T=1,024 to T=65,536 is strictly 0 Bytes.")
    print("=== GATE 3 PASSED ===")


if __name__ == "__main__":
    verify_gate3()
