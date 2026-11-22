//! Zero-allocation DCPA full-stack streaming inference & online learning engine.
//!
//! Numerically aligned 1-to-1 with `dcpa-train-py` (Gate 6 cosine similarity >= 0.9999,
//! max abs error < 1e-5 in float32 mode) with STRICTLY 0 heap allocations in every hot-path method!

use crate::simd::{
    apply_holographic_rope_f32, axpy_f32, cosine_sim_f32, dot_f32, expf, l2_norm_f32, linear_f32,
    logaddexp, logf, rmsnorm_f32, sigmoid, silu, softmax_inplace_f32, softplus, sqrtf, tanhf,
};
use crate::state::{
    ConceptGraphState, EligibilityTags, FastWeightPrecision, HippocampusFastWeights,
    MotorCheckpoint, MotorState, ScratchPool, SgscState, MAX_SEG_CAP,
};
use crate::weights::DcpaWeights;

/// Scalar telemetry emitted by a single byte ingestion step (zero heap allocation).
#[derive(Debug, Clone, Copy)]
pub struct ByteIngestTrace {
    pub info: f32,
    pub spiked: bool,
    pub seg_len: usize,
    pub mean_info: f32,
    pub sem_norm: f32,
}

/// Scalar telemetry emitted by a single concept graph + hippocampus step.
#[derive(Debug, Clone, Copy)]
pub struct ConceptStepTrace {
    pub ach: f32,
    pub gamma_slow: f32,
    pub gamma_hip: f32,
    pub surprisal: f32,
    pub ponder_steps: usize,
    pub win_pair_idx: usize,
    pub win_i: usize,
    pub win_j: usize,
}

/// Complete self-contained DCPA streaming runtime (all working memory allocated ONCE on `new`).
pub struct DcpaEngine {
    pub weights: DcpaWeights,
    pub sgsc: SgscState,
    pub graph: ConceptGraphState,
    pub elig: EligibilityTags,
    pub fast_mem: HippocampusFastWeights,
    pub motor: MotorState,
    pub scratch: ScratchPool,
}

impl DcpaEngine {
    pub fn new(weights: DcpaWeights, fast_precision: FastWeightPrecision) -> Self {
        let cfg = weights.cfg.clone();
        let sgsc = SgscState::new(&weights);
        let graph = ConceptGraphState::new(&weights);
        let elig = EligibilityTags::new(&cfg);
        let fast_mem = HippocampusFastWeights::new(&cfg, fast_precision);
        let motor = MotorState::new(&weights);
        let scratch = ScratchPool::new(&cfg);
        Self {
            weights,
            sgsc,
            graph,
            elig,
            fast_mem,
            motor,
            scratch,
        }
    }

    pub fn reset_session(&mut self) {
        self.sgsc.reset(&self.weights);
        self.graph.reset(&self.weights);
        self.elig.reset();
        self.motor.reset(&self.weights);
    }

    /// Top-k partial selection into fixed slices `out_idx[..k]` and `out_val[..k]` (descending),
    /// followed by ReLU and unit-L2 normalization. Strictly ZERO heap allocation!
    #[inline]
    fn topk_relu_unit_norm(scores: &[f32], out_idx: &mut [usize], out_val: &mut [f32]) {
        let k = out_idx.len();
        for j in 0..k {
            out_idx[j] = 0;
            out_val[j] = f32::NEG_INFINITY;
        }
        // Maintain sorted descending top-k insertion (for s=16..32 over d_h=4096, < 4 microseconds!)
        for (idx, &val) in scores.iter().enumerate() {
            if val > out_val[k - 1] {
                let mut pos = k - 1;
                while pos > 0 && val > out_val[pos - 1] {
                    out_val[pos] = out_val[pos - 1];
                    out_idx[pos] = out_idx[pos - 1];
                    pos -= 1;
                }
                out_val[pos] = val;
                out_idx[pos] = idx;
            }
        }
        let mut sq_sum = 0.0f32;
        for j in 0..k {
            let r = if out_val[j] > 0.0 { out_val[j] } else { 0.0 };
            out_val[j] = r;
            sq_sum += r * r;
        }
        let inv = 1.0 / (sqrtf(sq_sum) + 1e-8);
        for j in 0..k {
            out_val[j] *= inv;
        }
    }

    /// Module 1 (SGSC): Ingest ONE raw UTF-8 byte in strictly O(d_s^2) time and 0 heap allocations!
    ///
    /// When a spike fires (`trace.spiked == true`):
    ///   - `self.scratch.vec_ds_e[..d_s]` holds `f_stem`
    ///   - `self.scratch.vec_d_a[..D]` holds `c_slow`
    pub fn sgsc_ingest_byte(&mut self, byte: u8, force_end: bool) -> ByteIngestTrace {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d_c = cfg.d_concept;

        // 1. Strictly causal scale-invariant cosine prior p_t = Softmax(8 * norm(W_out) @ norm(h_{t-1}))
        //    Strictly weight-tied to W_out while decoupling PMD logit scale growth from SGSC surprisal.
        let h_norm = sqrtf(dot_f32(&self.sgsc.h, &self.sgsc.h) + 1e-2);
        let inv_h = 8.0 / h_norm;
        for i in 0..d_s {
            self.scratch.vec_ds_a[i] = self.sgsc.h[i] * inv_h;
        }
        for b_i in 0..256 {
            let row = &self.weights.sgsc_w_out[b_i * d_s..(b_i + 1) * d_s];
            let w_norm = sqrtf(dot_f32(row, row) + 1e-6);
            self.scratch.logits_256[b_i] = dot_f32(&self.scratch.vec_ds_a, row) / w_norm;
        }
        let mut max_l = f32::NEG_INFINITY;
        for &v in self.scratch.logits_256.iter() {
            if v > max_l {
                max_l = v;
            }
        }
        let mut sum_exp = 0.0f32;
        for i in 0..256 {
            let e = expf(self.scratch.logits_256[i] - max_l);
            self.scratch.probs_256[i] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        for i in 0..256 {
            self.scratch.probs_256[i] *= inv_sum;
        }
        let log_sum_exp = max_l + logf(sum_exp);
        let info = (log_sum_exp - self.scratch.logits_256[byte as usize]).max(0.0);

        let delta_info = if self.sgsc.has_prev_byte {
            (info - self.sgsc.prev_info).max(0.0)
        } else {
            0.0
        };
        self.sgsc.prev_info = info;
        self.sgsc.has_prev_byte = true;

        // 2. Causal diagonal sensory state update h_t = alpha_t * h_{t-1} + (1 - alpha_t) * SiLU(W_u u_t)
        let emb_off = (byte as usize) * d_s;
        let u_t = &self.weights.sgsc_embed[emb_off..emb_off + d_s];
        linear_f32(
            u_t,
            &self.weights.sgsc_w_alpha,
            Some(&self.weights.sgsc_b_alpha),
            &mut self.scratch.vec_ds_a,
        );
        linear_f32(u_t, &self.weights.sgsc_w_u, None, &mut self.scratch.vec_ds_b);
        for i in 0..d_s {
            let a = sigmoid(self.scratch.vec_ds_a[i]);
            let inj = (1.0 - a) * silu(self.scratch.vec_ds_b[i]);
            self.sgsc.h[i] = a * self.sgsc.h[i] + inj;
        }

        // Record into fixed segment buffer
        let slot_pos = self.sgsc.seg_len.min(MAX_SEG_CAP - 1);
        self.sgsc.seg_h[slot_pos * d_s..(slot_pos + 1) * d_s].copy_from_slice(&self.sgsc.h);
        self.sgsc.seg_bytes[slot_pos] = byte;
        self.sgsc.seg_len += 1;
        self.sgsc.seg_info_sum += info;

        // 3. Strictly causal UTF-8 codepoint completion tracker
        if byte < 0x80 {
            self.sgsc.utf8_rem = 0;
        } else if (byte & 0xE0) == 0xC0 {
            self.sgsc.utf8_rem = 1;
        } else if (byte & 0xF0) == 0xE0 {
            self.sgsc.utf8_rem = 2;
        } else if (byte & 0xF8) == 0xF0 {
            self.sgsc.utf8_rem = 3;
        } else if (byte & 0xC0) == 0x80 {
            self.sgsc.utf8_rem = self.sgsc.utf8_rem.saturating_sub(1);
        } else {
            self.sgsc.utf8_rem = 0;
        }

        // 4. LIF membrane potential & hard UTF-8 + maxSegLen=16 spike gate
        let v_tilde =
            cfg.lif_lambda * self.sgsc.v + cfg.lif_beta1 * info + cfg.lif_beta2 * delta_info;
        let theta = cfg.lif_theta_min + softplus(self.weights.sgsc_theta_raw);
        let can_spike = self.sgsc.utf8_rem == 0;
        let fire = force_end
            || (can_spike && (v_tilde >= theta || self.sgsc.seg_len >= cfg.max_seg_len))
            || self.sgsc.seg_len >= MAX_SEG_CAP;

        if !fire {
            self.sgsc.v = v_tilde;
            return ByteIngestTrace {
                info,
                spiked: false,
                seg_len: self.sgsc.seg_len,
                mean_info: 0.0,
                sem_norm: 0.0,
            };
        }

        // 5. Spike fired! Execute Unified RoPE phase-bound stem extraction (Engineering Doc 2.1)
        let seg_len = self.sgsc.seg_len;
        let mean_info = self.sgsc.seg_info_sum / (seg_len as f32);
        let h_end = &self.sgsc.seg_h[(seg_len - 1) * d_s..seg_len * d_s];

        // Anchor query q_m = RoPE(W_q h_end, seg_len)
        linear_f32(h_end, &self.weights.sgsc_w_q, None, &mut self.scratch.vec_ds_a);
        apply_holographic_rope_f32(
            &self.scratch.vec_ds_a,
            seg_len as f32,
            cfg.rope_period,
            cfg.rope_num_harmonics,
            false,
            &mut self.scratch.vec_ds_c, // q_seg
        );

        let inv_sqrt_ds = 1.0 / sqrtf(d_s as f32);
        for j in 0..seg_len {
            let hj = &self.sgsc.seg_h[j * d_s..(j + 1) * d_s];
            linear_f32(hj, &self.weights.sgsc_w_k, None, &mut self.scratch.vec_ds_a);
            apply_holographic_rope_f32(
                &self.scratch.vec_ds_a,
                (j + 1) as f32,
                cfg.rope_period,
                cfg.rope_num_harmonics,
                false,
                &mut self.scratch.vec_ds_b, // k_j
            );
            self.scratch.seg_scores[j] =
                dot_f32(&self.scratch.vec_ds_b, &self.scratch.vec_ds_c) * inv_sqrt_ds;
        }
        softmax_inplace_f32(&mut self.scratch.seg_scores[..seg_len]);

        // Weighted RoPE value sum agg = sum_j gamma_j * RoPE(W_f h_j, j+1)
        self.scratch.vec_ds_d.fill(0.0);
        for j in 0..seg_len {
            let hj = &self.sgsc.seg_h[j * d_s..(j + 1) * d_s];
            linear_f32(hj, &self.weights.sgsc_w_f, None, &mut self.scratch.vec_ds_a);
            apply_holographic_rope_f32(
                &self.scratch.vec_ds_a,
                (j + 1) as f32,
                cfg.rope_period,
                cfg.rope_num_harmonics,
                false,
                &mut self.scratch.vec_ds_b,
            );
            axpy_f32(self.scratch.seg_scores[j], &self.scratch.vec_ds_b, &mut self.scratch.vec_ds_d);
        }

        // f_stem = RMSNorm(h_end + agg) -> stored in scratch.vec_ds_e
        for i in 0..d_s {
            self.scratch.vec_ds_a[i] = h_end[i] + self.scratch.vec_ds_d[i];
        }
        rmsnorm_f32(
            &self.scratch.vec_ds_a,
            &self.weights.sgsc_norm_f,
            cfg.eps,
            &mut self.scratch.vec_ds_e,
        );

        // sem_proj = W_sem f_stem, c_slow = RMSNorm(W_res h_end + sem_proj) -> scratch.vec_d_a
        linear_f32(&self.scratch.vec_ds_e, &self.weights.sgsc_w_sem, None, &mut self.scratch.vec_d_b);
        let sem_norm = l2_norm_f32(&self.scratch.vec_d_b);
        linear_f32(h_end, &self.weights.sgsc_w_res, None, &mut self.scratch.vec_d_c);
        for i in 0..d_c {
            self.scratch.vec_d_c[i] += self.scratch.vec_d_b[i];
        }
        rmsnorm_f32(
            &self.scratch.vec_d_c,
            &self.weights.sgsc_norm_c,
            cfg.eps,
            &mut self.scratch.vec_d_a,
        );

        // Reset LIF accumulator & segment counters
        self.sgsc.v = 0.0;
        self.sgsc.seg_len = 0;
        self.sgsc.seg_info_sum = 0.0;

        ByteIngestTrace {
            info,
            spiked: true,
            seg_len,
            mean_info,
            sem_norm,
        }
    }

    /// Helper: Message passing over K=8 concept graph slots -> writes `s_bar [K*D]` and `g_ctx [D]`.
    fn message_pass_graph(
        cfg: &crate::config::DcpaConfig,
        w_e: &[f32],
        slots: &[f32],
        edges: &[f32],
        salience: &[f32],
        w_attn: &mut [f32],
        tmp_e: &mut [f32],
        s_bar_out: &mut [f32],
        g_ctx_out: &mut [f32],
    ) {
        let k = cfg.k_slots;
        let d = cfg.d_concept;
        let d_r = cfg.d_r;

        w_attn[..k].copy_from_slice(&salience[..k]);
        softmax_inplace_f32(&mut w_attn[..k]);

        s_bar_out[..k * d].copy_from_slice(&slots[..k * d]);
        for i in 0..k {
            let dst = &mut s_bar_out[i * d..(i + 1) * d];
            for j in 0..k {
                let edge_off = (i * k + j) * d_r;
                linear_f32(&edges[edge_off..edge_off + d_r], w_e, None, tmp_e);
                let sj = &slots[j * d..(j + 1) * d];
                let wj = w_attn[j];
                for m in 0..d {
                    dst[m] += wj * tanhf(tmp_e[m]) * sj[m];
                }
            }
        }

        g_ctx_out.fill(0.0);
        for i in 0..k {
            axpy_f32(w_attn[i], &s_bar_out[i * d..(i + 1) * d], g_ctx_out);
        }
    }

    /// Execute Modules 3 (Read) -> 2 (BCG-PC) -> 3 (Asymmetric Inheritance + 3-Factor Write)
    /// directly from the spiked segment stored in `scratch.vec_ds_e` (f_stem) & `scratch.vec_d_a` (c_slow).
    ///
    /// After return:
    ///   - `scratch.hip_c_fused` holds `c_fused`
    ///   - `scratch.hip_prec_bu` holds `prec_bu`
    ///   - `scratch.vec_d_g` holds `z_star`
    ///   - `scratch.vec_d_h` holds `g_ctx`
    /// Strictly ZERO heap allocations!
    pub fn step_concept_from_scratch(&mut self, mean_info: f32, sem_norm: f32) -> ConceptStepTrace {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d = cfg.d_concept;
        let k = cfg.k_slots;
        let kp1 = k + 1;
        let d_r = cfg.d_r;
        let d_m = cfg.d_merge;
        let s_act = cfg.s_active;

        // =====================================================================
        // 1. Module 3 Comprehension Read (DG_lex + ACh gating + 3-state routing)
        // =====================================================================
        linear_f32(
            &self.scratch.vec_ds_e, // f_stem
            &self.weights.hip_r_lex,
            None,
            &mut self.scratch.dg_scores,
        );
        Self::topk_relu_unit_norm(
            &self.scratch.dg_scores,
            &mut self.scratch.psi_idx,
            &mut self.scratch.psi_val,
        );

        let gamma_slow = sigmoid(
            self.weights.hip_alpha_1 * (self.weights.hip_theta_i - mean_info)
                + self.weights.hip_alpha_2 * (sem_norm - self.weights.hip_theta_n),
        );
        let gamma_hip = self.fast_mem.read_c_from_l(
            &self.scratch.psi_idx,
            &self.scratch.psi_val,
            &mut self.scratch.hip_c_hip,
        );
        let ach = (1.0 - gamma_slow) * (1.0 - gamma_hip);
        let mu = gamma_hip / (gamma_slow + gamma_hip + cfg.eps);

        // pi_cort = Softplus(f_prec_cort(c_slow)) + 1e-4
        linear_f32(
            &self.scratch.vec_d_a, // c_slow
            &self.weights.hip_f_prec_cort_w,
            Some(&self.weights.hip_f_prec_cort_b),
            &mut self.scratch.hip_prec_bu,
        );
        for i in 0..d {
            self.scratch.hip_c_fused[i] =
                (1.0 - mu) * self.scratch.vec_d_a[i] + mu * self.scratch.hip_c_hip[i];
            let pi_cort = softplus(self.scratch.hip_prec_bu[i]) + 1e-4;
            self.scratch.hip_prec_bu[i] = (1.0 - ach) * pi_cort;
        }

        // =====================================================================
        // 2. Module 2 BCG-PC: Top-Down Prior, Masked Pondering & K+1 -> K Merge
        // =====================================================================
        Self::message_pass_graph(
            cfg,
            &self.weights.bcg_w_e,
            &self.graph.slots,
            &self.graph.edges,
            &self.graph.salience,
            &mut self.scratch.w_attn,
            &mut self.scratch.vec_d_b,
            &mut self.scratch.s_bar,
            &mut self.scratch.vec_d_h, // g_ctx
        );

        // c_hat = f_pred_2(silu(f_pred_1(g_ctx))) -> scratch.vec_d_b
        linear_f32(
            &self.scratch.vec_d_h,
            &self.weights.bcg_f_pred_1_w,
            Some(&self.weights.bcg_f_pred_1_b),
            &mut self.scratch.vec_d_c,
        );
        for v in self.scratch.vec_d_c.iter_mut() {
            *v = silu(*v);
        }
        linear_f32(
            &self.scratch.vec_d_c,
            &self.weights.bcg_f_pred_2_w,
            Some(&self.weights.bcg_f_pred_2_b),
            &mut self.scratch.vec_d_b, // c_hat
        );

        // pi_m = softplus(f_prec_2(silu(f_prec_1(g_ctx)))) + 1e-4 -> scratch.vec_d_c
        linear_f32(
            &self.scratch.vec_d_h,
            &self.weights.bcg_f_prec_1_w,
            Some(&self.weights.bcg_f_prec_1_b),
            &mut self.scratch.vec_d_d,
        );
        for v in self.scratch.vec_d_d.iter_mut() {
            *v = silu(*v);
        }
        linear_f32(
            &self.scratch.vec_d_d,
            &self.weights.bcg_f_prec_2_w,
            Some(&self.weights.bcg_f_prec_2_b),
            &mut self.scratch.vec_d_c, // pi_m
        );
        for v in self.scratch.vec_d_c.iter_mut() {
            let p = softplus(*v) + 1e-4;
            *v = 2.0 * tanhf(p * 0.5);
        }

        // eps_0 -> scratch.vec_d_d, z_cur -> scratch.vec_d_g
        let one_minus_ach = (1.0 - ach).clamp(0.0, 1.0);
        for i in 0..d {
            let pi_eff = self.scratch.vec_d_c[i] * one_minus_ach;
            self.scratch.vec_d_d[i] = pi_eff * (self.scratch.hip_c_fused[i] - self.scratch.vec_d_b[i]);
        }
        let inv_sqrt_d = 1.0 / sqrtf(d as f32);
        let surprisal_0 = l2_norm_f32(&self.scratch.vec_d_d) * inv_sqrt_d;

        // Bayesian precision fusion + fast path
        linear_f32(&self.scratch.vec_d_d, &self.weights.bcg_w_fast, None, &mut self.scratch.vec_d_e);
        for i in 0..d {
            let pi_td = self.scratch.vec_d_c[i];
            let pi_bu = self.scratch.hip_prec_bu[i].max(0.0);
            let fused = (pi_td * self.scratch.vec_d_b[i] + pi_bu * self.scratch.hip_c_fused[i])
                / (pi_td + pi_bu + cfg.eps);
            self.scratch.vec_d_g[i] = fused + one_minus_ach * self.scratch.vec_d_e[i];
        }

        // Event-driven pondering loop (L_max = 4; immediate shortcut when surprisal <= tau_skip!)
        let mut surp_cur = surprisal_0;
        let mut ponder_steps = 0usize;
        for l_idx in 0..cfg.l_max {
            let thresh = if l_idx == 0 { cfg.tau_skip } else { cfg.tau_halt };
            if surp_cur <= thresh {
                break;
            }
            ponder_steps += 1;

            // CrossAttn(Q = z_cur, K = S_bar, V = S_bar) -> scratch.vec_d_e
            linear_f32(&self.scratch.vec_d_g, &self.weights.bcg_ca_wq, None, &mut self.scratch.vec_d_f);
            for slot_i in 0..k {
                let sb = &self.scratch.s_bar[slot_i * d..(slot_i + 1) * d];
                linear_f32(sb, &self.weights.bcg_ca_wk, None, &mut self.scratch.vec_d_e);
                self.scratch.ca_scores[slot_i] =
                    dot_f32(&self.scratch.vec_d_f, &self.scratch.vec_d_e) * inv_sqrt_d;
            }
            softmax_inplace_f32(&mut self.scratch.ca_scores[..k]);
            self.scratch.vec_d_f.fill(0.0);
            for slot_i in 0..k {
                let sb = &self.scratch.s_bar[slot_i * d..(slot_i + 1) * d];
                linear_f32(sb, &self.weights.bcg_ca_wv, None, &mut self.scratch.vec_d_e);
                axpy_f32(self.scratch.ca_scores[slot_i], &self.scratch.vec_d_e, &mut self.scratch.vec_d_f);
            }
            linear_f32(&self.scratch.vec_d_f, &self.weights.bcg_ca_wo, None, &mut self.scratch.vec_d_e);

            // MoE input = Pre-RMSNorm(eps_cur + r_a) -> scratch.vec_d_f
            {
                let mut sum_sq = 0.0f32;
                for i in 0..d {
                    let v = self.scratch.vec_d_d[i] + self.scratch.vec_d_e[i];
                    self.scratch.vec_d_f[i] = v;
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / sqrtf(sum_sq / (d as f32) + cfg.eps);
                for i in 0..d {
                    self.scratch.vec_d_f[i] *= inv_rms;
                }
            }

            // Top-2 SparseMoE routing
            linear_f32(
                &self.scratch.vec_d_f,
                &self.weights.bcg_moe_router,
                None,
                &mut self.scratch.router_logits,
            );
            let mut top_idx = [0usize; 2];
            let mut top_val = [f32::NEG_INFINITY; 2];
            for (ei, &lv) in self.scratch.router_logits.iter().enumerate() {
                if lv > top_val[0] {
                    top_val[1] = top_val[0];
                    top_idx[1] = top_idx[0];
                    top_val[0] = lv;
                    top_idx[0] = ei;
                } else if lv > top_val[1] {
                    top_val[1] = lv;
                    top_idx[1] = ei;
                }
            }
            let m_t = top_val[0].max(top_val[1]);
            let e0 = expf(top_val[0] - m_t);
            let e1 = expf(top_val[1] - m_t);
            let g0 = e0 / (e0 + e1);
            let g1 = e1 / (e0 + e1);
            let gates = [g0, g1];

            self.scratch.expert_out.fill(0.0);
            for t_i in 0..2 {
                let exp = &self.weights.bcg_experts[top_idx[t_i]];
                // Layer 1 SwiGLU (input vec_d_f is unit-RMS normalized)
                linear_f32(&self.scratch.vec_d_f, &exp.w_gate1, None, &mut self.scratch.expert_h1);
                linear_f32(&self.scratch.vec_d_f, &exp.w_up1, None, &mut self.scratch.expert_h2);
                for h_i in 0..cfg.expert_hidden {
                    self.scratch.expert_h1[h_i] =
                        silu(self.scratch.expert_h1[h_i]) * self.scratch.expert_h2[h_i];
                }
                linear_f32(&self.scratch.expert_h1, &exp.w_down1, None, &mut self.scratch.vec_d_e);
                let mut sum_sq2 = 0.0f32;
                for i in 0..d {
                    let v = self.scratch.vec_d_e[i] + self.scratch.vec_d_f[i];
                    self.scratch.vec_d_e[i] = v;
                    sum_sq2 += v * v;
                }
                let inv_rms2 = 1.0 / sqrtf(sum_sq2 / (d as f32) + cfg.eps);
                for i in 0..d {
                    self.scratch.vec_d_b[i] = self.scratch.vec_d_e[i] * inv_rms2;
                }
                // Layer 2 Pre-RMSNorm SwiGLU
                linear_f32(&self.scratch.vec_d_b, &exp.w_gate2, None, &mut self.scratch.expert_h1);
                linear_f32(&self.scratch.vec_d_b, &exp.w_up2, None, &mut self.scratch.expert_h2);
                for h_i in 0..cfg.expert_hidden {
                    self.scratch.expert_h1[h_i] =
                        silu(self.scratch.expert_h1[h_i]) * self.scratch.expert_h2[h_i];
                }
                linear_f32(&self.scratch.expert_h1, &exp.w_down2, None, &mut self.scratch.vec_d_b);
                for i in 0..d {
                    let y = self.scratch.vec_d_e[i] + self.scratch.vec_d_b[i];
                    self.scratch.expert_out[i] += gates[t_i] * y;
                }
            }

            // z_next = RMSNorm(z_cur + ponder_eta * expert_out)
            for i in 0..d {
                self.scratch.vec_d_e[i] = self.scratch.vec_d_g[i] + cfg.ponder_eta * self.scratch.expert_out[i];
            }
            rmsnorm_f32(
                &self.scratch.vec_d_e,
                &self.weights.bcg_norm_ponder,
                cfg.eps,
                &mut self.scratch.vec_d_g,
            );

            // Recompute residual eps = pi_eff * (c_fused - W_gen z_next)
            linear_f32(&self.scratch.vec_d_g, &self.weights.bcg_w_gen, None, &mut self.scratch.vec_d_e);
            for i in 0..d {
                let pi_eff = self.scratch.vec_d_c[i] * one_minus_ach;
                self.scratch.vec_d_d[i] = pi_eff * (self.scratch.hip_c_fused[i] - self.scratch.vec_d_e[i]);
            }
            surp_cur = l2_norm_f32(&self.scratch.vec_d_d) * inv_sqrt_d;
        }

        // 3. Candidate slot expansion (K -> K+1): s_tilde [(K+1)*D], a_tilde [(K+1)*(K+1)*d_r]
        self.scratch.s_tilde[..k * d].copy_from_slice(&self.graph.slots);
        self.scratch.s_tilde[k * d..kp1 * d].copy_from_slice(&self.scratch.vec_d_g); // z_star

        for i in 0..kp1 {
            let si = &self.scratch.s_tilde[i * d..(i + 1) * d];
            linear_f32(si, &self.weights.bcg_u_src, None, &mut self.scratch.src_all[i * d_r..(i + 1) * d_r]);
            linear_f32(si, &self.weights.bcg_u_dst, None, &mut self.scratch.dst_all[i * d_r..(i + 1) * d_r]);
            linear_f32(si, &self.weights.bcg_w_l, None, &mut self.scratch.wl_all[i * d_m..(i + 1) * d_m]);
            linear_f32(si, &self.weights.bcg_w_r, None, &mut self.scratch.wr_all[i * d_m..(i + 1) * d_m]);
        }

        // Populate a_tilde: copy KxK from graph.edges, fill row K and col K via tanh(src * dst)
        for i in 0..k {
            for j in 0..k {
                let dst_off = (i * kp1 + j) * d_r;
                let src_off = (i * k + j) * d_r;
                self.scratch.a_tilde[dst_off..dst_off + d_r]
                    .copy_from_slice(&self.graph.edges[src_off..src_off + d_r]);
            }
            // col K: edge i -> K
            let dst_off = (i * kp1 + k) * d_r;
            for r in 0..d_r {
                self.scratch.a_tilde[dst_off + r] =
                    tanhf(self.scratch.src_all[i * d_r + r] * self.scratch.dst_all[k * d_r + r]);
            }
        }
        // row K: edge K -> 0..=K
        for j in 0..kp1 {
            let dst_off = (k * kp1 + j) * d_r;
            for r in 0..d_r {
                self.scratch.a_tilde[dst_off + r] =
                    tanhf(self.scratch.src_all[k * d_r + r] * self.scratch.dst_all[j * d_r + r]);
            }
        }

        // 4. Score all K*(K+1) = 72 ordered pairs & find winning pair (i*, j*)
        let mut best_score = f32::NEG_INFINITY;
        let mut win_pair_idx = 0usize;
        let mut win_i = 0usize;
        let mut win_j = 1usize;
        let mut pair_idx = 0usize;

        for i in 0..kp1 {
            let wl_i = &self.scratch.wl_all[i * d_m..(i + 1) * d_m];
            for j in 0..kp1 {
                if i == j {
                    continue;
                }
                let wr_j = &self.scratch.wr_all[j * d_m..(j + 1) * d_m];
                let a_off = (i * kp1 + j) * d_r;
                linear_f32(
                    &self.scratch.a_tilde[a_off..a_off + d_r],
                    &self.weights.bcg_w_edge_score,
                    None,
                    &mut self.scratch.merge_h,
                );
                let mut score = 0.0f32;
                for m in 0..d_m {
                    score += self.weights.bcg_w_merge[m]
                        * tanhf(wl_i[m] + wr_j[m] + self.scratch.merge_h[m]);
                }
                if score > best_score {
                    best_score = score;
                    win_pair_idx = pair_idx;
                    win_i = i;
                    win_j = j;
                }
                pair_idx += 1;
            }
        }

        // 5. O(D) in-place parent composition ONLY for winning pair (win_i, win_j)!
        let u = win_i.min(win_j);
        let v = win_i.max(win_j);
        let si = &self.scratch.s_tilde[win_i * d..(win_i + 1) * d];
        let sj = &self.scratch.s_tilde[win_j * d..(win_j + 1) * d];
        let a_ij_off = (win_i * kp1 + win_j) * d_r;

        self.scratch.vec_2d[..d].copy_from_slice(si);
        self.scratch.vec_2d[d..2 * d].copy_from_slice(sj);
        self.scratch.vec_2d_dr[..2 * d].copy_from_slice(&self.scratch.vec_2d);
        self.scratch.vec_2d_dr[2 * d..2 * d + d_r]
            .copy_from_slice(&self.scratch.a_tilde[a_ij_off..a_ij_off + d_r]);

        linear_f32(
            &self.scratch.vec_2d_dr,
            &self.weights.bcg_w_g_w,
            Some(&self.weights.bcg_w_g_b),
            &mut self.scratch.vec_d_b, // g_win
        );
        linear_f32(
            &self.scratch.vec_2d,
            &self.weights.bcg_w_comp_w,
            Some(&self.weights.bcg_w_comp_b),
            &mut self.scratch.vec_d_c, // comp_win
        );
        for m in 0..d {
            let g = sigmoid(self.scratch.vec_d_b[m]);
            self.scratch.vec_d_d[m] = g * self.scratch.vec_d_c[m] + (1.0 - g) * si[m];
        }
        // Write merged parent in-place to slot u!
        rmsnorm_f32(
            &self.scratch.vec_d_d,
            &self.weights.bcg_norm_merge,
            cfg.eps,
            &mut self.graph.slots[u * d..(u + 1) * d],
        );
        if v < k {
            // Case B: place z_star into freed slot v!
            self.graph.slots[v * d..(v + 1) * d].copy_from_slice(&self.scratch.vec_d_g);
        }

        // Exact sparse edge contraction A_new = einsum('ka,lc,acr->klr', P, P, A_tilde)
        // First contract axis 0: a_tmp[row, c, r] = sum_a P[row, a] * A_tilde[a, c, r]
        for row in 0..k {
            for c in 0..kp1 {
                let out_off = (row * kp1 + c) * d_r;
                if row == u {
                    let o1 = (win_i * kp1 + c) * d_r;
                    let o2 = (win_j * kp1 + c) * d_r;
                    for r in 0..d_r {
                        self.scratch.a_tmp[out_off + r] =
                            0.5 * (self.scratch.a_tilde[o1 + r] + self.scratch.a_tilde[o2 + r]);
                    }
                } else {
                    let src_row = if row == v && v < k { k } else { row };
                    let src_off = (src_row * kp1 + c) * d_r;
                    self.scratch.a_tmp[out_off..out_off + d_r]
                        .copy_from_slice(&self.scratch.a_tilde[src_off..src_off + d_r]);
                }
            }
        }
        // Second contract axis 1: graph.edges[row, col, r] = sum_c P[col, c] * a_tmp[row, c, r]
        for row in 0..k {
            for col in 0..k {
                let out_off = (row * k + col) * d_r;
                if col == u {
                    let o1 = (row * kp1 + win_i) * d_r;
                    let o2 = (row * kp1 + win_j) * d_r;
                    for r in 0..d_r {
                        self.graph.edges[out_off + r] =
                            0.5 * (self.scratch.a_tmp[o1 + r] + self.scratch.a_tmp[o2 + r]);
                    }
                } else {
                    let src_col = if col == v && v < k { k } else { col };
                    let src_off = (row * kp1 + src_col) * d_r;
                    self.graph.edges[out_off..out_off + d_r]
                        .copy_from_slice(&self.scratch.a_tmp[src_off..src_off + d_r]);
                }
            }
        }

        // Exact LogSumExp Slot Salience Conservation (Engineering Doc 2.3)
        for i in 0..k {
            self.scratch.omega_tilde[i] = self.graph.salience[i] - cfg.delta_decay;
        }
        let omega_cand =
            dot_f32(&self.weights.bcg_w_omega, &self.scratch.vec_d_g) + self.weights.bcg_kappa_omega * surprisal_0;
        self.scratch.omega_tilde[k] = omega_cand;
        for row in 0..k {
            if row == u {
                self.graph.salience[row] = logaddexp(
                    self.scratch.omega_tilde[win_i],
                    self.scratch.omega_tilde[win_j],
                );
            } else if row == v && v < k {
                self.graph.salience[row] = self.scratch.omega_tilde[k];
            } else {
                self.graph.salience[row] = self.scratch.omega_tilde[row];
            }
        }

        // Post-merge message passing S_bar_post
        Self::message_pass_graph(
            cfg,
            &self.weights.bcg_w_e,
            &self.graph.slots,
            &self.graph.edges,
            &self.graph.salience,
            &mut self.scratch.w_attn,
            &mut self.scratch.vec_d_b,
            &mut self.scratch.s_bar_post,
            &mut self.scratch.vec_d_c, // discard post g_ctx
        );

        // =====================================================================
        // 3. Module 3 Asymmetric Eligibility Inheritance (Commandment #8)
        // =====================================================================
        let eta_i = if win_i < k { self.elig.eta[win_i] } else { ach };
        let eta_j = if win_j < k { self.elig.eta[win_j] } else { ach };
        let p_star = if eta_i >= eta_j { win_i } else { win_j };
        let p_eta = if eta_i >= eta_j { eta_i } else { eta_j };

        // Slot u inherits p* verbatim (with lambda_eta decay)
        if p_star == k {
            self.elig.e_idx[u * s_act..(u + 1) * s_act].copy_from_slice(&self.scratch.psi_idx);
            self.elig.e_val[u * s_act..(u + 1) * s_act].copy_from_slice(&self.scratch.psi_val);
            self.elig.f_tilde[u * d_s..(u + 1) * d_s].copy_from_slice(&self.scratch.vec_ds_e);
        } else if p_star != u {
            let src_idx_off = p_star * s_act;
            let dst_idx_off = u * s_act;
            for q in 0..s_act {
                self.elig.e_idx[dst_idx_off + q] = self.elig.e_idx[src_idx_off + q];
                self.elig.e_val[dst_idx_off + q] = self.elig.e_val[src_idx_off + q];
            }
            let src_f_off = p_star * d_s;
            let dst_f_off = u * d_s;
            for q in 0..d_s {
                self.elig.f_tilde[dst_f_off + q] = self.elig.f_tilde[src_f_off + q];
            }
        }
        self.elig.eta[u] = cfg.lambda_eta * p_eta;

        if v < k {
            // Case B: freed slot v gets candidate slot K
            self.elig.e_idx[v * s_act..(v + 1) * s_act].copy_from_slice(&self.scratch.psi_idx);
            self.elig.e_val[v * s_act..(v + 1) * s_act].copy_from_slice(&self.scratch.psi_val);
            self.elig.f_tilde[v * d_s..(v + 1) * d_s].copy_from_slice(&self.scratch.vec_ds_e);
            self.elig.eta[v] = ach;
        }

        // =====================================================================
        // 4. Module 3 Three-Factor Semantic Crystallization & Gated Delta Write
        // =====================================================================
        self.fast_mem.advance_step();
        for slot_i in 0..k {
            let eta_slot = self.elig.eta[slot_i];
            if eta_slot <= 0.0 {
                continue;
            }
            let s_bar_i = &self.scratch.s_bar_post[slot_i * d..(slot_i + 1) * d];
            let cos_c0 = cosine_sim_f32(s_bar_i, &self.weights.bcg_c0);
            let mut edge_norm_sum = 0.0f32;
            for j in 0..k {
                let off = (slot_i * k + j) * d_r;
                edge_norm_sum += l2_norm_f32(&self.graph.edges[off..off + d_r]);
            }
            let q_cryst = sigmoid(
                self.weights.hip_alpha_q
                    * ((1.0 - cos_c0) + self.weights.hip_gamma_e * edge_norm_sum
                        - self.weights.hip_theta_q),
            );
            let beta = eta_slot * q_cryst;
            if beta > 0.0 {
                linear_f32(s_bar_i, &self.weights.hip_r_sem, None, &mut self.scratch.dg_scores);
                Self::topk_relu_unit_norm(
                    &self.scratch.dg_scores,
                    &mut self.scratch.phi_idx,
                    &mut self.scratch.phi_val,
                );
                let ei_off = slot_i * s_act;
                let ft_off = slot_i * d_s;
                self.fast_mem.write_slot_inplace(
                    beta,
                    &self.elig.e_idx[ei_off..ei_off + s_act],
                    &self.elig.e_val[ei_off..ei_off + s_act],
                    s_bar_i,
                    &self.scratch.phi_idx,
                    &self.scratch.phi_val,
                    &self.elig.f_tilde[ft_off..ft_off + d_s],
                    &mut self.scratch.hip_proj_c,
                    &mut self.scratch.hip_proj_l,
                );
                // Quench consumed eligibility trace
                self.elig.eta[slot_i] = ((1.0 - cfg.kappa_quench * beta) * eta_slot).clamp(0.0, 1.0);
            }
        }

        ConceptStepTrace {
            ach,
            gamma_slow,
            gamma_hip,
            surprisal: surprisal_0,
            ponder_steps,
            win_pair_idx,
            win_i,
            win_j,
        }
    }

    /// Initialize Broca CQ queue from current concept graph salience. Zero heap allocation.
    pub fn broca_cq_init(&mut self) {
        let k = self.weights.cfg.k_slots;
        for i in 0..k {
            let q0 = sigmoid(self.graph.salience[i]);
            self.motor.queue_q[i] = q0;
            self.motor.queue_q_init[i] = q0;
        }
        self.motor.a_prev.fill(0.0);
    }

    /// Module 4 (Broca CQ + Hippocampal Expressive Read): Pop next concept & stem prototype!
    ///
    /// Writes:
    ///   - `scratch.pmd_z_out [D]`
    ///   - `scratch.pmd_delta_syn [d_s]`
    ///   - `scratch.pmd_f_out [d_s]`
    /// Returns `(rho_out, queue_done)`. Strictly ZERO heap allocation!
    pub fn broca_cq_step_and_retrieve(&mut self, lang_id: usize) -> (f32, bool) {
        let cfg = &self.weights.cfg;
        let k = cfg.k_slots;
        let d = cfg.d_concept;
        let d_r = cfg.d_r;
        let d_l = cfg.d_l;
        let d_s = cfg.d_s;

        let l_off = lang_id * d_l;
        let l_reg = &self.weights.pmd_lang_embed[l_off..l_off + d_l];

        // 1. Edge precedence matrix P^{ord}_{i,j}(l_reg) = tanh((W_ord l_reg)^T A_{i,j})
        linear_f32(l_reg, &self.weights.pmd_w_ord, None, &mut self.scratch.vec_dr_a);
        linear_f32(l_reg, &self.weights.pmd_w_coh, None, &mut self.scratch.vec_dr_b);

        for i in 0..k {
            for j in 0..k {
                let e_off = (i * k + j) * d_r;
                self.scratch.p_ord[i * k + j] =
                    tanhf(dot_f32(&self.scratch.vec_dr_a, &self.graph.edges[e_off..e_off + d_r]));
            }
        }

        // Readiness potential u_ready[i]
        let mut win_slot = 0usize;
        let mut best_u = f32::NEG_INFINITY;
        for i in 0..k {
            let mut inh = 0.0f32;
            for j in 0..k {
                if i != j {
                    let p = self.scratch.p_ord[i * k + j];
                    if p > 0.0 {
                        inh += self.motor.queue_q[j] * p;
                    }
                }
            }
            let ap = &self.motor.a_prev[i * d_r..(i + 1) * d_r];
            let coh = dot_f32(&self.scratch.vec_dr_b, ap);
            let q_clamped = self.motor.queue_q[i].max(cfg.eps);
            let u_val = logf(q_clamped) - cfg.lambda_inh * inh + coh;
            self.scratch.u_ready[i] = u_val;
            if u_val > best_u {
                best_u = u_val;
                win_slot = i;
            }
        }

        // Hard winner selection -> z_out, r_role, delta_syn
        self.scratch
            .pmd_z_out
            .copy_from_slice(&self.graph.slots[win_slot * d..(win_slot + 1) * d]);

        self.scratch.vec_dr_a.fill(0.0);
        for j in 0..k {
            let e_off = (j * k + win_slot) * d_r;
            axpy_f32(
                self.motor.queue_q_init[j],
                &self.graph.edges[e_off..e_off + d_r],
                &mut self.scratch.vec_dr_a,
            );
        }
        self.scratch.vec_dr_dl[..d_r].copy_from_slice(&self.scratch.vec_dr_a);
        self.scratch.vec_dr_dl[d_r..d_r + d_l].copy_from_slice(l_reg);

        linear_f32(
            &self.scratch.vec_dr_dl,
            &self.weights.pmd_w_syn_w,
            Some(&self.weights.pmd_w_syn_b),
            &mut self.scratch.vec_ds_a,
        );
        linear_f32(
            &self.scratch.pmd_z_out,
            &self.weights.pmd_u_syn,
            None,
            &mut self.scratch.vec_ds_b,
        );
        for i in 0..d_s {
            self.scratch.pmd_delta_syn[i] = silu(self.scratch.vec_ds_a[i] + self.scratch.vec_ds_b[i]);
        }

        // IoR update: zero out selected slot & update a_prev
        self.motor.queue_q[win_slot] = 0.0;
        for i in 0..k {
            let src = (win_slot * k + i) * d_r;
            self.motor.a_prev[i * d_r..(i + 1) * d_r]
                .copy_from_slice(&self.graph.edges[src..src + d_r]);
        }
        let mut max_q = 0.0f32;
        for &qv in self.motor.queue_q.iter() {
            if qv > max_q {
                max_q = qv;
            }
        }
        let done = max_q < cfg.tau_done;

        // Module 3 Expressive Read: f_out = (1 - rho) * W_lex_slow(z_out + W_l l_reg) + rho * M^{L<-C} phi_out
        linear_f32(l_reg, &self.weights.hip_w_lang_lex, None, &mut self.scratch.vec_d_b);
        for i in 0..d {
            self.scratch.vec_d_b[i] += self.scratch.pmd_z_out[i];
        }
        linear_f32(
            &self.scratch.vec_d_b,
            &self.weights.hip_w_lex_slow,
            None,
            &mut self.scratch.vec_ds_a, // f_slow
        );
        // Unit-RMS normalize cortical stem readout to match SGSC.f_stem scale
        {
            let mut sum_sq = 0.0f32;
            for i in 0..d_s {
                let v = self.scratch.vec_ds_a[i];
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / sqrtf(sum_sq / (d_s as f32) + cfg.eps);
            for i in 0..d_s {
                self.scratch.vec_ds_a[i] *= inv_rms;
            }
        }

        linear_f32(&self.scratch.pmd_z_out, &self.weights.hip_r_sem, None, &mut self.scratch.dg_scores);
        Self::topk_relu_unit_norm(
            &self.scratch.dg_scores,
            &mut self.scratch.phi_idx,
            &mut self.scratch.phi_val,
        );
        let gamma_sem = self.fast_mem.read_l_from_c(
            &self.scratch.phi_idx,
            &self.scratch.phi_val,
            &mut self.scratch.vec_ds_b, // f_hip
        );
        let rho_out = sigmoid(self.weights.hip_alpha_out * (gamma_sem - self.weights.hip_theta_out));
        for i in 0..d_s {
            self.scratch.pmd_f_out[i] =
                (1.0 - rho_out) * self.scratch.vec_ds_a[i] + rho_out * self.scratch.vec_ds_b[i];
        }

        // Initialize motor charge E_{n, 0} for this concept segment
        self.scratch.vec_2ds_dl[..d_s].copy_from_slice(&self.scratch.pmd_f_out);
        self.scratch.vec_2ds_dl[d_s..2 * d_s].copy_from_slice(&self.scratch.pmd_delta_syn);
        self.scratch.vec_2ds_dl[2 * d_s..2 * d_s + d_l].copy_from_slice(l_reg);
        let e0_raw =
            dot_f32(&self.weights.pmd_w_e_init_w, &self.scratch.vec_2ds_dl) + self.weights.pmd_w_e_init_b;
        self.motor.charge_e = softplus(e0_raw);

        (rho_out, done)
    }

    /// Module 4 (PMD): Execute ONE streaming byte generation step from `scratch.pmd_f_out` & `scratch.pmd_delta_syn`.
    ///
    /// Writes logits into `scratch.logits_256`. Returns `(greedy_byte, charge_e, p_seg_end)`.
    /// Strictly ZERO heap allocation!
    pub fn pmd_step_byte(&mut self, prev_byte: u8, lang_id: usize, tau_step: usize) -> (u8, f32, f32) {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d_l = cfg.d_l;

        let u_off = (prev_byte as usize) * d_s;
        let u_prev = &self.weights.sgsc_embed[u_off..u_off + d_s];
        let l_off = lang_id * d_l;
        let l_reg = &self.weights.pmd_lang_embed[l_off..l_off + d_l];

        // 1. RoPE inverse phase demodulation: d_stem = R_Theta(-tau)(U_demod f_out)
        linear_f32(&self.scratch.pmd_f_out, &self.weights.pmd_u_demod, None, &mut self.scratch.vec_ds_a);
        apply_holographic_rope_f32(
            &self.scratch.vec_ds_a,
            tau_step as f32,
            cfg.rope_period,
            cfg.rope_num_harmonics,
            true,
            &mut self.scratch.vec_ds_b, // d_stem
        );

        // Stem-affix handoff gate chi
        self.scratch.vec_3ds[..d_s].copy_from_slice(&self.scratch.vec_ds_b);
        self.scratch.vec_3ds[d_s..2 * d_s].copy_from_slice(u_prev);
        self.scratch.vec_3ds[2 * d_s..3 * d_s].copy_from_slice(&self.scratch.pmd_delta_syn);
        let chi = sigmoid(
            dot_f32(&self.weights.pmd_w_chi_w, &self.scratch.vec_3ds) + self.weights.pmd_w_chi_b,
        );
        for i in 0..d_s {
            self.scratch.vec_ds_c[i] =
                chi * self.scratch.vec_ds_b[i] + (1.0 - chi) * self.scratch.pmd_delta_syn[i];
        } // d_drive in vec_ds_c

        // 2. O(1) 128B Differentiable Pushdown Stack shift in float32
        self.scratch.vec_ds_dl[..d_s].copy_from_slice(u_prev);
        self.scratch.vec_ds_dl[d_s..d_s + d_l].copy_from_slice(l_reg);
        let mut act = [0.0f32; 3];
        linear_f32(
            &self.scratch.vec_ds_dl,
            &self.weights.pmd_w_act_w,
            Some(&self.weights.pmd_w_act_b),
            &mut act,
        );
        softmax_inplace_f32(&mut act);
        let (p_push, p_pop, p_hold) = (act[0], act[1], act[2]);

        let mut v_tag = [0.0f32; 4];
        linear_f32(u_prev, &self.weights.pmd_w_tag_w, Some(&self.weights.pmd_w_tag_b), &mut v_tag);
        for t in 0..4 {
            v_tag[t] = tanhf(v_tag[t]);
        }

        let prev_xi = self.motor.stack_xi;
        for row in 0..16 {
            for col in 0..4 {
                let pushed = if row == 0 { v_tag[col] } else { prev_xi[row - 1][col] };
                let popped = if row < 15 { prev_xi[row + 1][col] } else { 0.0 };
                self.motor.stack_xi[row][col] =
                    p_hold * prev_xi[row][col] + p_push * pushed + p_pop * popped;
            }
        }

        // Top-2 stack slice R^8 -> W_st -> R^{d_s}
        let mut xi_top2 = [0.0f32; 8];
        xi_top2[0..4].copy_from_slice(&self.motor.stack_xi[0]);
        xi_top2[4..8].copy_from_slice(&self.motor.stack_xi[1]);
        linear_f32(&xi_top2, &self.weights.pmd_w_st, None, &mut self.scratch.vec_ds_a);

        // 3. Weight-tied causal byte motor micro-core
        for i in 0..d_s {
            self.scratch.vec_ds_d[i] = u_prev[i] + self.scratch.vec_ds_c[i] + self.scratch.vec_ds_a[i];
        } // x_in in vec_ds_d
        linear_f32(
            &self.scratch.vec_ds_d,
            &self.weights.pmd_w_alpha_mot_w,
            Some(&self.weights.pmd_w_alpha_mot_b),
            &mut self.scratch.vec_ds_a,
        );
        linear_f32(&self.scratch.vec_ds_d, &self.weights.pmd_w_m, None, &mut self.scratch.vec_ds_b);
        for i in 0..d_s {
            let a = sigmoid(self.scratch.vec_ds_a[i]);
            let inj = (1.0 - a) * silu(self.scratch.vec_ds_b[i]);
            self.motor.m[i] = a * self.motor.m[i] + inj;
        }

        // Shared byte classification head W_out
        linear_f32(&self.motor.m, &self.weights.sgsc_w_out, None, &mut self.scratch.logits_256);
        let mut best_b = 0u8;
        let mut best_l = f32::NEG_INFINITY;
        for b_i in 0..256 {
            if self.scratch.logits_256[b_i] > best_l {
                best_l = self.scratch.logits_256[b_i];
                best_b = b_i as u8;
            }
        }

        // 4. Dual Motor Charge Depletion Gate (unit-RMS normalize m so it matches d_drive scale)
        {
            let mut sum_sq = 0.0f32;
            for i in 0..d_s {
                let v = self.motor.m[i];
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / sqrtf(sum_sq / (d_s as f32) + cfg.eps);
            for i in 0..d_s {
                self.scratch.vec_3ds[i] = self.motor.m[i] * inv_rms;
            }
        }
        self.scratch.vec_3ds[d_s..2 * d_s].copy_from_slice(&self.scratch.vec_ds_c);
        let drain_raw = dot_f32(&self.weights.pmd_w_drain_w, &self.scratch.vec_3ds[..2 * d_s])
            + self.weights.pmd_w_drain_b;
        self.motor.charge_e -= softplus(drain_raw);
        let p_end = sigmoid(-self.motor.charge_e / cfg.tau_e.max(1e-4));

        self.motor.last_byte = best_b;
        (best_b, self.motor.charge_e, p_end)
    }

    /// Zero-copy Efference Copy self-monitoring: re-encodes emitted segment bytes through L1 SGSC + Hippocampus
    /// into `scratch.mon_c_check` and returns semantic drift error `eps_mon = 1 - cos(z_out, c_check)`.
    fn efference_copy_check(&mut self, seg_bytes: &[u8]) -> f32 {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d = cfg.d_concept;
        let seg_len = seg_bytes.len().min(MAX_SEG_CAP);
        if seg_len == 0 {
            return 2.0;
        }

        self.scratch.mon_h.copy_from_slice(&self.weights.sgsc_h0);
        let mut info_sum = 0.0f32;
        for (j, &byte) in seg_bytes[..seg_len].iter().enumerate() {
            // Prior surprisal
            linear_f32(&self.scratch.mon_h, &self.weights.sgsc_w_out, None, &mut self.scratch.logits_256);
            let mut max_l = f32::NEG_INFINITY;
            for &v in self.scratch.logits_256.iter() {
                if v > max_l {
                    max_l = v;
                }
            }
            let mut sum_e = 0.0f32;
            for i in 0..256 {
                sum_e += expf(self.scratch.logits_256[i] - max_l);
            }
            info_sum += (max_l + logf(sum_e) - self.scratch.logits_256[byte as usize]).max(0.0);

            let emb_off = (byte as usize) * d_s;
            let u_t = &self.weights.sgsc_embed[emb_off..emb_off + d_s];
            linear_f32(
                u_t,
                &self.weights.sgsc_w_alpha,
                Some(&self.weights.sgsc_b_alpha),
                &mut self.scratch.vec_ds_a,
            );
            linear_f32(u_t, &self.weights.sgsc_w_u, None, &mut self.scratch.vec_ds_b);
            for i in 0..d_s {
                let a = sigmoid(self.scratch.vec_ds_a[i]);
                self.scratch.mon_h[i] = a * self.scratch.mon_h[i] + (1.0 - a) * silu(self.scratch.vec_ds_b[i]);
            }
            self.scratch.mon_seg_h[j * d_s..(j + 1) * d_s].copy_from_slice(&self.scratch.mon_h);
        }

        let h_end = &self.scratch.mon_seg_h[(seg_len - 1) * d_s..seg_len * d_s];
        linear_f32(h_end, &self.weights.sgsc_w_q, None, &mut self.scratch.vec_ds_a);
        apply_holographic_rope_f32(
            &self.scratch.vec_ds_a,
            seg_len as f32,
            cfg.rope_period,
            cfg.rope_num_harmonics,
            false,
            &mut self.scratch.vec_ds_c,
        );
        let inv_sqrt_ds = 1.0 / sqrtf(d_s as f32);
        for j in 0..seg_len {
            let hj = &self.scratch.mon_seg_h[j * d_s..(j + 1) * d_s];
            linear_f32(hj, &self.weights.sgsc_w_k, None, &mut self.scratch.vec_ds_a);
            apply_holographic_rope_f32(
                &self.scratch.vec_ds_a,
                (j + 1) as f32,
                cfg.rope_period,
                cfg.rope_num_harmonics,
                false,
                &mut self.scratch.vec_ds_b,
            );
            self.scratch.seg_scores[j] =
                dot_f32(&self.scratch.vec_ds_b, &self.scratch.vec_ds_c) * inv_sqrt_ds;
        }
        softmax_inplace_f32(&mut self.scratch.seg_scores[..seg_len]);
        self.scratch.vec_ds_d.fill(0.0);
        for j in 0..seg_len {
            let hj = &self.scratch.mon_seg_h[j * d_s..(j + 1) * d_s];
            linear_f32(hj, &self.weights.sgsc_w_f, None, &mut self.scratch.vec_ds_a);
            apply_holographic_rope_f32(
                &self.scratch.vec_ds_a,
                (j + 1) as f32,
                cfg.rope_period,
                cfg.rope_num_harmonics,
                false,
                &mut self.scratch.vec_ds_b,
            );
            axpy_f32(self.scratch.seg_scores[j], &self.scratch.vec_ds_b, &mut self.scratch.vec_ds_d);
        }
        for i in 0..d_s {
            self.scratch.vec_ds_a[i] = h_end[i] + self.scratch.vec_ds_d[i];
        }
        rmsnorm_f32(&self.scratch.vec_ds_a, &self.weights.sgsc_norm_f, cfg.eps, &mut self.scratch.mon_f_stem);

        linear_f32(&self.scratch.mon_f_stem, &self.weights.sgsc_w_sem, None, &mut self.scratch.vec_d_b);
        let sem_norm = l2_norm_f32(&self.scratch.vec_d_b);
        linear_f32(h_end, &self.weights.sgsc_w_res, None, &mut self.scratch.vec_d_c);
        for i in 0..d {
            self.scratch.vec_d_c[i] += self.scratch.vec_d_b[i];
        }
        rmsnorm_f32(&self.scratch.vec_d_c, &self.weights.sgsc_norm_c, cfg.eps, &mut self.scratch.mon_c_slow);

        // Hippocampus read head on self-monitored stem
        linear_f32(&self.scratch.mon_f_stem, &self.weights.hip_r_lex, None, &mut self.scratch.dg_scores);
        Self::topk_relu_unit_norm(&self.scratch.dg_scores, &mut self.scratch.psi_idx, &mut self.scratch.psi_val);
        let mean_info = info_sum / (seg_len as f32);
        let gamma_slow = sigmoid(
            self.weights.hip_alpha_1 * (self.weights.hip_theta_i - mean_info)
                + self.weights.hip_alpha_2 * (sem_norm - self.weights.hip_theta_n),
        );
        let gamma_hip = self.fast_mem.read_c_from_l(
            &self.scratch.psi_idx,
            &self.scratch.psi_val,
            &mut self.scratch.hip_c_hip,
        );
        let mu = gamma_hip / (gamma_slow + gamma_hip + cfg.eps);
        for i in 0..d {
            self.scratch.mon_c_check[i] =
                (1.0 - mu) * self.scratch.mon_c_slow[i] + mu * self.scratch.hip_c_hip[i];
        }

        1.0 - cosine_sim_f32(&self.scratch.pmd_z_out, &self.scratch.mon_c_check)
    }

    /// Streaming concept generation with Stack-Allocated Efference Copy Rollback (Engineering Doc 2.5).
    ///
    /// Writes generated UTF-8 bytes into caller-supplied fixed slice `out_buf`, returning bytes written.
    /// Retries up to `max_repair_retries` (2) times with halved temperature on UTF-8 error or
    /// semantic drift `eps_mon > tau_repair`, then greedy fallback. Strictly ZERO heap allocations!
    pub fn generate_concept_with_efference_copy(
        &mut self,
        lang_id: usize,
        out_buf: &mut [u8],
        rng_state: &mut u64,
    ) -> (usize, bool, f32) {
        let (_rho, done) = self.broca_cq_step_and_retrieve(lang_id);
        let init_charge = self.motor.charge_e;
        let checkpoint = MotorCheckpoint::capture(&self.motor);
        let max_retries = self.weights.cfg.max_repair_retries;
        let tau_repair = self.weights.cfg.tau_repair;
        let max_seg = self.weights.cfg.max_seg_len.min(out_buf.len()).min(MAX_SEG_CAP);

        let mut temp = 0.8f32;
        let mut last_len = 0usize;
        let mut last_err = 0.0f32;

        for attempt in 0..=max_retries {
            checkpoint.restore(&mut self.motor);
            self.motor.charge_e = init_charge;
            let greedy = attempt == max_retries;
            let mut len = 0usize;

            for tau_step in 1..=max_seg {
                let prev = self.motor.last_byte;
                let (greedy_b, charge_e, _) = self.pmd_step_byte(prev, lang_id, tau_step);
                let chosen = if greedy || temp < 0.05 {
                    greedy_b
                } else {
                    // Temperature-scaled sampling with deterministic XorShift64 (zero heap alloc)
                    let mut max_l = f32::NEG_INFINITY;
                    for &v in self.scratch.logits_256.iter() {
                        if v > max_l {
                            max_l = v;
                        }
                    }
                    let inv_t = 1.0 / temp;
                    let mut sum = 0.0f32;
                    for i in 0..256 {
                        let e = expf((self.scratch.logits_256[i] - max_l) * inv_t);
                        self.scratch.probs_256[i] = e;
                        sum += e;
                    }
                    let mut x = *rng_state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *rng_state = x;
                    let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
                    let target = u * sum;
                    let mut acc = 0.0f32;
                    let mut picked = greedy_b;
                    for i in 0..256 {
                        acc += self.scratch.probs_256[i];
                        if acc >= target {
                            picked = i as u8;
                            break;
                        }
                    }
                    self.motor.last_byte = picked;
                    picked
                };

                out_buf[len] = chosen;
                len += 1;
                if charge_e <= 0.0 {
                    break;
                }
            }

            last_len = len;
            let utf8_ok = core::str::from_utf8(&out_buf[..len]).is_ok();
            last_err = if utf8_ok {
                self.efference_copy_check(&out_buf[..len])
            } else {
                2.0
            };

            if utf8_ok && last_err <= tau_repair {
                break;
            }
            temp *= 0.5;
        }

        (last_len, done, last_err)
    }
}
