"""Flat-named safetensors slow-weight exporter & deterministic Golden Test Vector generator.

Implements Engineering Blueprint Section 1.2 Cross-Stack Numerical Parity Contract:
  - Exports all slow parameters & frozen orthogonal buffers (R_lex, R_sem, c0) into a single
    flat-named model.safetensors alongside config.json.
  - Excludes tied duplicates (pmd.embed, pmd.w_out point to sgsc.embed, sgsc.w_out) and
    NEVER exports fast hippocampal matrices as slow weights.
  - Generates a deterministic golden_vectors.safetensors bundle containing input bytes,
    SGSC outputs, Hippocampus read & single-step write outputs, BCG-PC outputs, and PMD
    streaming + Broca CQ outputs so dcpa-core-rs verifies >= 0.9999 cosine similarity
    (max abs err < 1e-5 in float32 mode).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file

from dcpa.compile_utils import get_clean_state_dict
from dcpa.model import DCPAModel
from dcpa.modules.hippocampus import EligibilityState


def extract_flat_state_dict(model: DCPAModel, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Extract canonical flat-named slow weights & frozen buffers for Rust consumption.

    Removes tied duplicate keys (`pmd.embed.weight`, `pmd.w_out.weight`) and derived
    index buffers (`perm_bank`, `pair_i`, `pair_j`, `pair_u`, `pair_v`) which Rust builds
    deterministically from `k_slots`.
    """
    skip_keys = {
        "pmd.embed.weight",
        "pmd.w_out.weight",
        "bcg_pc.perm_bank",
        "bcg_pc.pair_i",
        "bcg_pc.pair_j",
        "bcg_pc.pair_u",
        "bcg_pc.pair_v",
        "bcg_pc.pair_u_onehot",
        "bcg_pc.pin_mask_72",
        "bcg_pc.slot0_onehot",
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in get_clean_state_dict(model).items():
        if name in skip_keys:
            continue
        t = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
        if name == "sgsc.theta_raw":
            # Fold dimensionless lif_theta_scale so Rust core evaluates softplus(sgsc_theta_raw) verbatim
            t = (tensor.detach().float() * model.cfg.lif_theta_scale).to(device="cpu", dtype=dtype).contiguous()
        tensors[name] = t
    return tensors


def export_model_and_golden(
    model: DCPAModel,
    out_dir: str | Path,
    sample_text: str = "A zibber is a tiny frog! 你好🦀",
    lang_id: int = 1,
    export_dtype: torch.dtype = torch.float32,
) -> dict[str, Path]:
    """Export model.safetensors, config.json, and golden.safetensors for Gate 6 verification."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_cpu = model.to("cpu")

    # 1. Save config.json and model.safetensors
    cfg_path = out_path / "config.json"
    model_cpu.cfg.save_json(cfg_path)

    weights_path = out_path / "model.safetensors"
    flat_weights = extract_flat_state_dict(model_cpu, dtype=export_dtype)
    save_file(flat_weights, str(weights_path))

    # 2. Run deterministic single-session forward pass (float32 reference) for Golden Test Vectors
    raw_py = sample_text.encode("utf-8")
    raw_bytes = torch.tensor(list(raw_py), dtype=torch.uint8).unsqueeze(0)  # [1, T]
    lang_t = torch.tensor([lang_id], dtype=torch.long)

    with torch.no_grad():
        # Module 1: SGSC
        sgsc_out = model_cpu.sgsc(raw_bytes, tau_s=0.35)
        m_total = sgsc_out.f_stem.shape[0]
        l_reg = model_cpu.pmd.get_lang_vec(lang_t)  # [1, 64]

        # Initialize single-session working states
        graph_state = model_cpu.bcg_pc.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
        fast_state = model_cpu.hippocampus.init_fast_state(1, mode="column_major", storage_dtype=torch.float32)
        elig_state = EligibilityState.zeros(1, model_cpu.cfg.k_slots, model_cpu.cfg.s_active, model_cpu.cfg.d_s, torch.device("cpu"), torch.float32)

        # Step through all M_total segments identically to Rust streaming engine
        all_c_fused = []
        all_prec_bu = []
        all_ach = []
        all_z_star = []
        all_g_ctx = []
        all_win_pair = []

        for m in range(m_total):
            f_m = sgsc_out.f_stem[m : m + 1]
            c_m = sgsc_out.c_slow[m : m + 1]
            sn_m = sgsc_out.sem_norm[m : m + 1]
            mi_m = sgsc_out.mean_info[m : m + 1]

            hip_r = model_cpu.hippocampus.read_comprehension(
                f_stem=f_m,
                c_slow=c_m,
                sem_norm=sn_m,
                mean_info=mi_m,
                fast_state=fast_state,
                enabled=True,
            )
            bcg_o = model_cpu.bcg_pc.step(
                c_fused=hip_r.c_fused,
                prec_bu=hip_r.prec_bu,
                ach=hip_r.ach_signal,
                state=graph_state,
                tau_m=0.25,
                use_gumbel=False,
                update_c0_ema=False,
            )
            graph_state = bcg_o.state
            elig_state = model_cpu.hippocampus.inherit_eligibility_on_merge(
                elig=elig_state,
                cand_idx=hip_r.psi_idx,
                cand_val=hip_r.psi_val,
                cand_f=f_m,
                cand_ach=hip_r.ach_signal,
                win_i=bcg_o.win_i,
                win_j=bcg_o.win_j,
            )
            elig_state, _, _ = model_cpu.hippocampus.three_factor_write(
                s_bar_post=bcg_o.s_bar_post,
                edges_post=graph_state.edges,
                c0=model_cpu.bcg_pc.c0,
                elig=elig_state,
                fast_state=fast_state,
                write_threshold=0.0,
            )

            all_c_fused.append(hip_r.c_fused[0])
            all_prec_bu.append(hip_r.prec_bu[0])
            all_ach.append(hip_r.ach_signal[0])
            all_z_star.append(bcg_o.z_star[0])
            all_g_ctx.append(bcg_o.g_ctx[0])
            all_win_pair.append(bcg_o.win_pair_idx[0])

        # Also test explicit single-step Hippocampus retrieval on segment 0 after all writes
        psi0_idx, psi0_val = model_cpu.hippocampus.dg_lex(sgsc_out.f_stem[0:1])
        hip_c_after = fast_state.read_c_from_l(psi0_idx, psi0_val)[0]  # [D]

        # Module 4: 1 step of Broca CQ + 4 streaming motor byte steps from the final concept graph
        q_init = torch.sigmoid(graph_state.salience.float())
        a_prev = torch.zeros((1, model_cpu.cfg.k_slots, model_cpu.cfg.d_r), dtype=torch.float32)
        cq_out = model_cpu.pmd.broca_cq_step(
            graph=graph_state,
            q_prev=q_init,
            q_init=q_init,
            a_prev=a_prev,
            l_reg=l_reg,
        )
        f_out_cq, rho_out_cq = model_cpu.hippocampus.read_expression(
            z_out=cq_out.z_out,
            l_reg=l_reg,
            fast_state=fast_state,
            enabled=True,
        )

        # Stream 4 greedy motor bytes
        m_cur = model_cpu.pmd.m0.unsqueeze(0).clone()
        xi_cur = torch.zeros((1, 16, 4), dtype=torch.float32)
        charge_cur = model_cpu.pmd.init_segment_charge(f_out_cq, cq_out.delta_syn, l_reg)
        prev_b = torch.zeros((1,), dtype=torch.long)

        motor_logits_list = []
        motor_bytes_list = []
        motor_charge_list = []
        for tau_step in range(1, 5):
            logits, m_cur, xi_cur, charge_cur, _ = model_cpu.pmd.step_byte_streaming(
                prev_byte=prev_b,
                f_out=f_out_cq,
                delta_syn=cq_out.delta_syn,
                l_reg=l_reg,
                tau_step=tau_step,
                m_prev=m_cur,
                xi_prev=xi_cur,
                charge_prev=charge_cur,
            )
            next_b = logits.argmax(dim=-1)
            motor_logits_list.append(logits[0])
            motor_bytes_list.append(next_b[0])
            motor_charge_list.append(charge_cur[0])
            prev_b = next_b

    golden_tensors = {
        "raw_bytes": raw_bytes[0].to(torch.uint8).contiguous(),
        "lang_id": torch.tensor([lang_id], dtype=torch.int64),
        "sgsc.h_seq": sgsc_out.h_seq[0].float().contiguous(),
        "sgsc.prior_probs": sgsc_out.prior_probs[0].float().contiguous(),
        "sgsc.info": sgsc_out.info[0].float().contiguous(),
        "sgsc.spikes": sgsc_out.spikes[0].to(torch.uint8).contiguous(),
        "sgsc.cu_seqlens": sgsc_out.cu_seqlens.to(torch.int32).contiguous(),
        "sgsc.f_stem": sgsc_out.f_stem.float().contiguous(),
        "sgsc.c_slow": sgsc_out.c_slow.float().contiguous(),
        "sgsc.mean_info": sgsc_out.mean_info.float().contiguous(),
        "hip.c_fused": torch.stack(all_c_fused, dim=0).float().contiguous(),
        "hip.prec_bu": torch.stack(all_prec_bu, dim=0).float().contiguous(),
        "hip.ach": torch.stack(all_ach, dim=0).float().contiguous(),
        "hip.c_after_write": hip_c_after.float().contiguous(),
        "hip.u_lex": fast_state.u_lex.float().contiguous(),
        "bcg.z_star": torch.stack(all_z_star, dim=0).float().contiguous(),
        "bcg.g_ctx": torch.stack(all_g_ctx, dim=0).float().contiguous(),
        "bcg.final_slots": graph_state.slots[0].float().contiguous(),
        "bcg.final_edges": graph_state.edges[0].float().contiguous(),
        "bcg.final_salience": graph_state.salience[0].float().contiguous(),
        "bcg.win_pair": torch.stack(all_win_pair, dim=0).to(torch.int64).contiguous(),
        "pmd.cq_z_out": cq_out.z_out[0].float().contiguous(),
        "pmd.cq_delta_syn": cq_out.delta_syn[0].float().contiguous(),
        "pmd.cq_f_out": f_out_cq[0].float().contiguous(),
        "pmd.cq_rho_out": rho_out_cq.float().contiguous(),
        "pmd.motor_logits": torch.stack(motor_logits_list, dim=0).float().contiguous(),
        "pmd.motor_bytes": torch.stack(motor_bytes_list, dim=0).to(torch.uint8).contiguous(),
        "pmd.motor_charge": torch.stack(motor_charge_list, dim=0).float().contiguous(),
        "pmd.final_stack_xi": xi_cur[0].float().contiguous(),
    }

    golden_path = out_path / "golden.safetensors"
    save_file(golden_tensors, str(golden_path))

    meta_path = out_path / "golden_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "sample_text": sample_text,
                "num_bytes": len(raw_py),
                "num_segments": m_total,
                "lang_id": lang_id,
                "export_dtype": str(export_dtype),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "config": cfg_path,
        "weights": weights_path,
        "golden": golden_path,
        "meta": meta_path,
    }
