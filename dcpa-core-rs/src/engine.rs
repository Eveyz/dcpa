//! Zero-allocation DCPA full-stack streaming inference & online learning engine.
//!
//! Numerically aligned 1-to-1 with `dcpa-train-py` (Gate 6 cosine similarity >= 0.9999,
//! max abs error < 1e-5 in float32 mode) with STRICTLY 0 heap allocations in every hot-path method!

use crate::simd::{
    apply_holographic_rope_f32, axpy_f32, cosf, cosine_sim_f32, dot_f32, expf, l2_norm_f32,
    linear_f32, logaddexp, logf, powf, rmsnorm_f32, sigmoid, silu, sinf, softmax_inplace_f32,
    softplus, sqrtf, tanhf, unit_rmsnorm_f32,
};
use crate::state::{
    BrainSnapshot, ConceptGraphState, CorticalState, EligibilityTags, FastWeightPrecision,
    HippocampusFastWeights, MotorCheckpoint, MotorState, ScratchPool, SgscState, MAX_SEG_CAP,
};
use crate::weights::{CorticalMoEWeights, DcpaWeights};

/// Scalar telemetry emitted by a single byte ingestion step (zero heap allocation).
#[derive(Debug, Clone, Copy)]
pub struct ByteIngestTrace {
    pub info: f32,
    pub spiked: bool,
    pub seg_len: usize,
    pub mean_info: f32,
    pub sem_norm: f32,
    pub pin_triggered: bool,
    pub unpin_triggered: bool,
    pub seg_end_depth: i32,
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
    pub slot0_pinned: bool,
}

/// Complete self-contained DCPA-Turbo streaming runtime (all working memory allocated ONCE on `new`).
pub struct DcpaEngine {
    pub weights: DcpaWeights,
    pub sgsc: SgscState,
    pub cortical: CorticalState,
    pub graph: ConceptGraphState,
    pub elig: EligibilityTags,
    pub fast_mem: HippocampusFastWeights,
    pub motor: MotorState,
    pub scratch: ScratchPool,
}

impl DcpaEngine {
    pub fn new(mut weights: DcpaWeights, fast_precision: FastWeightPrecision) -> Self {
        if fast_precision == FastWeightPrecision::Bf16 {
            weights.pack_experts_bf16();
        }
        let cfg = weights.cfg.clone();
        let sgsc = SgscState::new(&weights);
        let cortical = CorticalState::new(&weights);
        let graph = ConceptGraphState::new(&weights);
        let elig = EligibilityTags::new(&cfg);
        let fast_mem = HippocampusFastWeights::new(&cfg, fast_precision);
        let motor = MotorState::new(&weights);
        let scratch = ScratchPool::new(&cfg);
        Self {
            weights,
            sgsc,
            cortical,
            graph,
            elig,
            fast_mem,
            motor,
            scratch,
        }
    }

    pub fn reset_session(&mut self) {
        self.sgsc.reset(&self.weights);
        self.cortical.reset(&self.weights);
        self.graph.reset(&self.weights);
        self.elig.reset();
        self.motor.reset(&self.weights);
    }

    /// Capture O(1) active working memory snapshot into pre-allocated `snap` (0 heap allocs!).
    #[inline]
    pub fn capture_snapshot(&self, snap: &mut BrainSnapshot) {
        snap.capture_from(&self.sgsc, &self.cortical, &self.graph, &self.elig, &self.motor);
    }

    /// Restore O(1) active working memory snapshot from `snap` (0 heap allocs!).
    #[inline]
    pub fn restore_snapshot(&mut self, snap: &BrainSnapshot) {
        snap.restore_into(
            &mut self.sgsc,
            &mut self.cortical,
            &mut self.graph,
            &mut self.elig,
            &mut self.motor,
        );
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

        // Record into fixed segment buffer & track Spec 2 Slot-0 Thesis Pinning control sequences
        let slot_pos = self.sgsc.seg_len.min(MAX_SEG_CAP - 1);
        self.sgsc.seg_h[slot_pos * d_s..(slot_pos + 1) * d_s].copy_from_slice(&self.sgsc.h);
        self.sgsc.seg_bytes[slot_pos] = byte;
        self.sgsc.seg_len += 1;
        self.sgsc.seg_info_sum += info;

        if self.sgsc.recent_len < 6 {
            self.sgsc.recent_bytes[self.sgsc.recent_len] = byte;
            self.sgsc.recent_len += 1;
        } else {
            self.sgsc.recent_bytes.copy_within(1..6, 0);
            self.sgsc.recent_bytes[5] = byte;
        }
        let rlen = self.sgsc.recent_len;
        if self.sgsc.pending_asst_pin {
            // Defer `\x1fA\n` pin if immediately followed by `\x1d` (`\x1dS\n`); otherwise trigger!
            if byte != 0x1D {
                self.sgsc.seg_pin_hit = true;
            }
            self.sgsc.pending_asst_pin = false;
            self.sgsc.pending_asst_ticks = 0;
        }
        if rlen >= 3 {
            let b_m2 = self.sgsc.recent_bytes[rlen - 3];
            let b_m1 = self.sgsc.recent_bytes[rlen - 2];
            let b_0 = self.sgsc.recent_bytes[rlen - 1];
            if b_m2 == 0x1D && b_m1 == b'E' && b_0 == 0x0A {
                self.sgsc.seg_pin_hit = true;
            } else if b_m2 == 0x1F && b_m1 == b'A' && b_0 == 0x0A {
                self.sgsc.pending_asst_pin = true;
                self.sgsc.pending_asst_ticks = 1;
            }
        }
        // Turn boundary `0x04` (EOT) or `0x1E` (user RS) schedules unpinning and cancels any earlier pin in this segment.
        if byte == 0x04 || byte == 0x1E {
            self.sgsc.seg_unpin_hit = true;
            self.sgsc.seg_pin_hit = false;
            self.sgsc.pending_asst_pin = false;
            self.sgsc.pending_asst_ticks = 0;
        }

        // Strictly causal AST bracket/scope nesting depth matching `rope.py:compute_causal_byte_scope_depth`
        let post_reset = byte == 0x04
            || byte == 0x13
            || byte == 0x1E
            || (rlen >= 3
                && self.sgsc.recent_bytes[rlen - 3] == 0x1D
                && self.sgsc.recent_bytes[rlen - 2] == b'E'
                && self.sgsc.recent_bytes[rlen - 1] == 0x0A);
        if post_reset {
            self.sgsc.raw_ast_depth = 0;
        } else {
            let delta = match byte {
                b'(' | b'[' | b'{' => 1,
                b')' | b']' | b'}' => -1,
                _ => 0,
            };
            self.sgsc.raw_ast_depth = (self.sgsc.raw_ast_depth + delta).max(0);
        }
        let cur_clamped_depth = self.sgsc.raw_ast_depth.clamp(0, 31);
        self.cortical.ast_depth = cur_clamped_depth;

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
                pin_triggered: false,
                unpin_triggered: false,
                seg_end_depth: cur_clamped_depth,
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
        let pin_triggered = self.sgsc.seg_pin_hit;
        let unpin_triggered = self.sgsc.seg_unpin_hit;
        self.sgsc.last_seg_end_depth = cur_clamped_depth;
        self.sgsc.v = 0.0;
        self.sgsc.seg_len = 0;
        self.sgsc.seg_info_sum = 0.0;
        self.sgsc.seg_pin_hit = false;
        self.sgsc.seg_unpin_hit = false;

        ByteIngestTrace {
            info,
            spiked: true,
            seg_len,
            mean_info,
            sem_norm,
            pin_triggered,
            unpin_triggered,
            seg_end_depth: cur_clamped_depth,
        }
    }

    /// Dual-coordinate Concept-RoPE (`3/4` relative concept order + `1/4` AST scope depth)
    /// across all `n_heads` heads of a `[D]` vector. Strictly 0 heap allocations!
    #[inline]
    fn apply_concept_rope_2d_heads(
        x: &[f32],
        concept_pos: i64,
        scope_depth: i32,
        n_heads: usize,
        theta_base: f32,
        scope_period: f32,
        out: &mut [f32],
    ) {
        let d = x.len();
        let d_head = d / n_heads;
        let half = d_head / 2;
        let scope_pairs = (half / 4).max(1);
        let pos_pairs = half - scope_pairs;
        let pos_dim = (pos_pairs * 2).max(2) as f32;

        let mut cos_a = [0.0f32; 128];
        let mut sin_a = [0.0f32; 128];
        let c_pos = concept_pos as f32;
        let s_dep = scope_depth as f32;

        for i in 0..pos_pairs {
            let freq = powf(theta_base, -(2.0 * (i as f32)) / pos_dim);
            let ang = c_pos * freq;
            cos_a[i] = cosf(ang);
            sin_a[i] = sinf(ang);
        }
        let base_scope_freq = (2.0 * core::f32::consts::PI) / scope_period;
        for j in 0..scope_pairs {
            let freq = base_scope_freq * ((j + 1) as f32);
            let ang = s_dep * freq;
            cos_a[pos_pairs + j] = cosf(ang);
            sin_a[pos_pairs + j] = sinf(ang);
        }

        for h in 0..n_heads {
            let base = h * d_head;
            for p in 0..half {
                let xe = x[base + 2 * p];
                let xo = x[base + 2 * p + 1];
                let ca = cos_a[p];
                let sa = sin_a[p];
                out[base + 2 * p] = xe * ca - xo * sa;
                out[base + 2 * p + 1] = xe * sa + xo * ca;
            }
        }
    }

    /// Execute `SharedAndRoutedMoE` (`1 Shared + Top-K Routed SwiGLU`) with 0 heap allocations.
    #[allow(clippy::too_many_arguments)]
    fn run_cortical_moe(
        cfg: &crate::config::DcpaConfig,
        moe: &CorticalMoEWeights,
        x_in: &[f32],
        y_out: &mut [f32],
        x_norm: &mut [f32],
        y_shared: &mut [f32],
        shared_gu: &mut [f32],
        shared_h: &mut [f32],
        router_logits: &mut [f32],
        routed_h1: &mut [f32],
        routed_h2: &mut [f32],
        routed_tmp: &mut [f32],
        routed_out: &mut [f32],
    ) {
        let d = cfg.d_concept;
        let sh = cfg.shared_expert_hidden;
        let ch = cfg.cort_expert_hidden;
        let n_exp = cfg.num_experts;
        let tk = cfg.top_k_experts.min(8).min(n_exp);

        rmsnorm_f32(x_in, &moe.norm, cfg.eps, x_norm);

        // 1. Shared expert branch
        moe.shared_w_gu.linear_no_bias(x_norm, shared_gu);
        for j in 0..sh {
            shared_h[j] = silu(shared_gu[j]) * shared_gu[sh + j];
        }
        moe.shared_w_d.linear_no_bias(shared_h, y_shared);

        // 2. Top-K router in float32
        linear_f32(x_norm, &moe.router, None, &mut router_logits[..n_exp]);
        let mut top_idx = [0usize; 8];
        let mut top_val = [f32::NEG_INFINITY; 8];
        for (ei, &lv) in router_logits[..n_exp].iter().enumerate() {
            if lv > top_val[tk - 1] {
                let mut p = tk - 1;
                while p > 0 && lv > top_val[p - 1] {
                    top_val[p] = top_val[p - 1];
                    top_idx[p] = top_idx[p - 1];
                    p -= 1;
                }
                top_val[p] = lv;
                top_idx[p] = ei;
            }
        }
        softmax_inplace_f32(&mut top_val[..tk]);

        // 3. Routed specialist experts
        routed_out[..d].fill(0.0);
        for t_i in 0..tk {
            let exp = &moe.experts[top_idx[t_i]];
            exp.w_gate.linear_no_bias(x_norm, routed_h1);
            exp.w_up.linear_no_bias(x_norm, routed_h2);
            for j in 0..ch {
                routed_h1[j] = silu(routed_h1[j]) * routed_h2[j];
            }
            exp.w_down.linear_no_bias(routed_h1, routed_tmp);
            axpy_f32(top_val[t_i], &routed_tmp[..d], &mut routed_out[..d]);
        }

        for i in 0..d {
            y_out[i] = x_in[i] + y_shared[i] + routed_out[i];
        }
    }

    /// Lower Cortical Sandwich streaming step (`CorticalSandwich.encode_step_streaming`):
    /// pushes shared `c_kv` into O(1) FIFO ring and writes `h_enc` into `scratch.cort_h_enc`.
    fn cortical_encode_step_parts(
        weights: &DcpaWeights,
        cortical: &mut CorticalState,
        scratch: &mut ScratchPool,
        seg_end_depth: i32,
        unpin_reset: bool,
    ) {
        let cfg = &weights.cfg;
        let d = cfg.d_concept;
        let d_s = cfg.d_s;
        let w = cfg.window_size;
        let n_heads = cfg.n_heads;
        let dh = d / n_heads;
        let scale = 1.0 / sqrtf(dh as f32);

        // 1. Shared MLA-latent KV compression: c_kv = RMSNorm(f_stem + w_kv_c(c_slow))
        linear_f32(&scratch.vec_d_a, &weights.cort_w_kv_c, None, &mut scratch.vec_ds_a);
        for i in 0..d_s {
            scratch.vec_ds_a[i] += scratch.vec_ds_e[i];
        }
        rmsnorm_f32(&scratch.vec_ds_a, &weights.cort_norm_kv, cfg.eps, &mut scratch.vec_ds_b);

        // 2. O(1) FIFO ring shift & append at index W - 1
        if w > 1 {
            cortical.ring_kv.copy_within(d_s..w * d_s, 0);
            cortical.ring_pos.copy_within(1..w, 0);
            cortical.ring_depth.copy_within(1..w, 0);
            cortical.ring_valid.copy_within(1..w, 0);
        }
        cortical.ring_kv[(w - 1) * d_s..w * d_s].copy_from_slice(&scratch.vec_ds_b);
        cortical.ring_pos[w - 1] = cortical.doc_pos;
        cortical.ring_depth[w - 1] = seg_end_depth;
        cortical.ring_valid[w - 1] = true;
        cortical.doc_pos += 1;
        cortical.ast_depth = if unpin_reset { 0 } else { seg_end_depth };

        // 3. Lower Cortical Encoder layers
        scratch.cort_x.copy_from_slice(&scratch.vec_d_a);
        let cur_pos = cortical.ring_pos[w - 1];
        let cur_dep = cortical.ring_depth[w - 1];

        for layer in &weights.cort_enc_layers {
            rmsnorm_f32(&scratch.cort_x, &layer.norm_attn, cfg.eps, &mut scratch.cort_tmp);
            linear_f32(&scratch.cort_tmp, &layer.w_q, None, &mut scratch.cort_q);
            Self::apply_concept_rope_2d_heads(
                &scratch.cort_q,
                cur_pos,
                cur_dep,
                n_heads,
                cfg.rope_theta_concept,
                cfg.rope_period,
                &mut scratch.cort_q_rot,
            );

            for r in 0..w {
                if cortical.ring_valid[r] {
                    let kv_r = &cortical.ring_kv[r * d_s..(r + 1) * d_s];
                    linear_f32(kv_r, &layer.w_uk, None, &mut scratch.cort_tmp);
                    Self::apply_concept_rope_2d_heads(
                        &scratch.cort_tmp,
                        cortical.ring_pos[r],
                        cortical.ring_depth[r],
                        n_heads,
                        cfg.rope_theta_concept,
                        cfg.rope_period,
                        &mut scratch.cort_k_ring[r * d..(r + 1) * d],
                    );
                    linear_f32(kv_r, &layer.w_uv, None, &mut scratch.cort_v_ring[r * d..(r + 1) * d]);
                }
            }

            scratch.cort_tmp.fill(0.0);
            for h in 0..n_heads {
                let q_h = &scratch.cort_q_rot[h * dh..(h + 1) * dh];
                for r in 0..w {
                    scratch.cort_scores[r] = if cortical.ring_valid[r] {
                        let k_rh = &scratch.cort_k_ring[r * d + h * dh..r * d + (h + 1) * dh];
                        dot_f32(q_h, k_rh) * scale
                    } else {
                        -1.0e9
                    };
                }
                softmax_inplace_f32(&mut scratch.cort_scores[..w]);
                let ctx_h = &mut scratch.cort_tmp[h * dh..(h + 1) * dh];
                for r in 0..w {
                    if cortical.ring_valid[r] {
                        let v_rh = &scratch.cort_v_ring[r * d + h * dh..r * d + (h + 1) * dh];
                        axpy_f32(scratch.cort_scores[r], v_rh, ctx_h);
                    }
                }
            }

            linear_f32(&scratch.cort_tmp, &layer.w_o, None, &mut scratch.cort_tmp2);
            for i in 0..d {
                scratch.cort_tmp2[i] += scratch.cort_x[i];
            }
            Self::run_cortical_moe(
                cfg,
                &layer.moe,
                &scratch.cort_tmp2,
                &mut scratch.cort_x,
                &mut scratch.cort_tmp,
                &mut scratch.vec_d_e,
                &mut scratch.cort_shared_gu,
                &mut scratch.cort_shared_h,
                &mut scratch.router_logits,
                &mut scratch.cort_routed_h1,
                &mut scratch.cort_routed_h2,
                &mut scratch.vec_d_f,
                &mut scratch.cort_routed_out,
            );
        }

        rmsnorm_f32(&scratch.cort_x, &weights.cort_norm_enc_out, cfg.eps, &mut scratch.cort_h_enc);
    }

    /// Upper Cortical Sandwich streaming decoder step (`CorticalSandwich.decode_step_streaming`):
    /// writes `z_dec` into `scratch.cort_z_dec` and `copy_stem` into `scratch.cort_copy_out`.
    fn cortical_decode_step_parts(
        weights: &DcpaWeights,
        cortical: &CorticalState,
        graph: &ConceptGraphState,
        scratch: &mut ScratchPool,
        l_reg: &[f32],
    ) {
        let cfg = &weights.cfg;
        let d = cfg.d_concept;
        let d_s = cfg.d_s;
        let d_r = cfg.d_r;
        let k = cfg.k_slots;
        let w = cfg.window_size;
        let n_heads = cfg.n_heads;
        let dh = d / n_heads;
        let scale = 1.0 / sqrtf(dh as f32);
        let inv_h = 1.0 / (n_heads as f32);

        let z_in = if cortical.has_prev_seg {
            &cortical.prev_z_ref
        } else {
            &weights.cort_bos_concept
        };
        linear_f32(l_reg, &weights.cort_w_lang_dec, None, &mut scratch.cort_x);
        for i in 0..d {
            scratch.cort_x[i] += z_in[i];
        }
        scratch.cort_copy_f.fill(0.0);

        let cur_pos = cortical.doc_pos;
        let cur_start_depth = cortical.ast_depth;

        for layer in &weights.cort_dec_layers {
            // 1. Enrich K=8 graph slots with incoming relation edges & 32D temporal phase
            scratch.w_attn[..k].copy_from_slice(&graph.salience[..k]);
            softmax_inplace_f32(&mut scratch.w_attn[..k]);

            scratch.edge_in_kd[..k * d_r].fill(0.0);
            for j in 0..k {
                let dst = &mut scratch.edge_in_kd[j * d_r..(j + 1) * d_r];
                for i in 0..k {
                    let e_off = (i * k + j) * d_r;
                    axpy_f32(scratch.w_attn[i], &graph.edges[e_off..e_off + d_r], dst);
                }
            }

            for j in 0..k {
                scratch.cort_sal_bias[j] = -softplus(-graph.salience[j]);
                let p = graph.phase[j];
                for idx in 0..16 {
                    let freq = powf(256.0, -(idx as f32) / 16.0);
                    let ang = p * freq;
                    scratch.phase_feat_32[idx] = sinf(ang);
                    scratch.phase_feat_32[16 + idx] = cosf(ang);
                }
                linear_f32(
                    &scratch.edge_in_kd[j * d_r..(j + 1) * d_r],
                    &layer.w_edge_in,
                    None,
                    &mut scratch.cort_tmp,
                );
                linear_f32(&scratch.phase_feat_32, &layer.w_phase, None, &mut scratch.cort_tmp2);
                for m in 0..d {
                    scratch.cort_tmp[m] =
                        graph.slots[j * d + m] + tanhf(scratch.cort_tmp[m]) + scratch.cort_tmp2[m];
                }
                linear_f32(&scratch.cort_tmp, &layer.w_gk, None, &mut scratch.cort_gk[j * d..(j + 1) * d]);
                linear_f32(&scratch.cort_tmp, &layer.w_gv, None, &mut scratch.cort_gv[j * d..(j + 1) * d]);
            }

            // 2. Query & strict-past window K/V projections
            rmsnorm_f32(&scratch.cort_x, &layer.norm_attn, cfg.eps, &mut scratch.cort_tmp);
            linear_f32(&scratch.cort_tmp, &layer.w_q, None, &mut scratch.cort_q);
            Self::apply_concept_rope_2d_heads(
                &scratch.cort_q,
                cur_pos,
                cur_start_depth,
                n_heads,
                cfg.rope_theta_concept,
                cfg.rope_period,
                &mut scratch.cort_q_rot,
            );

            for r in 0..w {
                let diff = cur_pos - cortical.ring_pos[r];
                if cortical.ring_valid[r] && diff >= 1 && diff <= (w as i64) {
                    let kv_r = &cortical.ring_kv[r * d_s..(r + 1) * d_s];
                    linear_f32(kv_r, &layer.w_uk, None, &mut scratch.cort_tmp);
                    Self::apply_concept_rope_2d_heads(
                        &scratch.cort_tmp,
                        cortical.ring_pos[r],
                        cortical.ring_depth[r],
                        n_heads,
                        cfg.rope_theta_concept,
                        cfg.rope_period,
                        &mut scratch.cort_k_ring[r * d..(r + 1) * d],
                    );
                    linear_f32(kv_r, &layer.w_uv, None, &mut scratch.cort_v_ring[r * d..(r + 1) * d]);
                }
            }

            // 3. Single-softmax unified attention over (Strict-Past W Window ∪ K=8 Graph Slots)
            scratch.cort_tmp.fill(0.0);
            scratch.cort_copy_acc.fill(0.0);
            for h in 0..n_heads {
                let q_rot_h = &scratch.cort_q_rot[h * dh..(h + 1) * dh];
                let q_raw_h = &scratch.cort_q[h * dh..(h + 1) * dh];
                for r in 0..w {
                    let diff = cur_pos - cortical.ring_pos[r];
                    scratch.cort_scores[r] = if cortical.ring_valid[r] && diff >= 1 && diff <= (w as i64) {
                        let k_rh = &scratch.cort_k_ring[r * d + h * dh..r * d + (h + 1) * dh];
                        dot_f32(q_rot_h, k_rh) * scale
                    } else {
                        -1.0e9
                    };
                }
                for j in 0..k {
                    let gk_jh = &scratch.cort_gk[j * d + h * dh..j * d + (h + 1) * dh];
                    scratch.cort_scores[w + j] =
                        dot_f32(q_raw_h, gk_jh) * scale + scratch.cort_sal_bias[j];
                }
                softmax_inplace_f32(&mut scratch.cort_scores[..w + k]);
                let ctx_h = &mut scratch.cort_tmp[h * dh..(h + 1) * dh];
                for r in 0..w {
                    let aw = scratch.cort_scores[r];
                    scratch.cort_copy_acc[r] += aw * inv_h;
                    let diff = cur_pos - cortical.ring_pos[r];
                    if cortical.ring_valid[r] && diff >= 1 && diff <= (w as i64) {
                        let v_rh = &scratch.cort_v_ring[r * d + h * dh..r * d + (h + 1) * dh];
                        axpy_f32(aw, v_rh, ctx_h);
                    }
                }
                for j in 0..k {
                    let ag = scratch.cort_scores[w + j];
                    let gv_jh = &scratch.cort_gv[j * d + h * dh..j * d + (h + 1) * dh];
                    axpy_f32(ag, gv_jh, ctx_h);
                }
            }

            scratch.cort_copy_f.fill(0.0);
            for r in 0..w {
                let cw = scratch.cort_copy_acc[r];
                if cw != 0.0 {
                    axpy_f32(cw, &cortical.ring_kv[r * d_s..(r + 1) * d_s], &mut scratch.cort_copy_f);
                }
            }

            linear_f32(&scratch.cort_tmp, &layer.w_o, None, &mut scratch.cort_tmp2);
            for i in 0..d {
                scratch.cort_tmp2[i] += scratch.cort_x[i];
            }
            Self::run_cortical_moe(
                cfg,
                &layer.moe,
                &scratch.cort_tmp2,
                &mut scratch.cort_x,
                &mut scratch.cort_tmp,
                &mut scratch.vec_d_e,
                &mut scratch.cort_shared_gu,
                &mut scratch.cort_shared_h,
                &mut scratch.router_logits,
                &mut scratch.cort_routed_h1,
                &mut scratch.cort_routed_h2,
                &mut scratch.vec_d_f,
                &mut scratch.cort_routed_out,
            );
        }

        rmsnorm_f32(&scratch.cort_x, &weights.cort_norm_dec_out, cfg.eps, &mut scratch.cort_z_dec);
        linear_f32(&scratch.cort_copy_f, &weights.cort_w_copy_out, None, &mut scratch.cort_copy_out);
    }

    /// Public zero-alloc helper running `decode_step_streaming` for `(lang_id, style_id)`.
    pub fn cortical_decode_styled(&mut self, lang_id: usize, style_id: usize) {
        let l_reg_arr = self.build_l_reg(lang_id, style_id);
        let d_l = self.weights.cfg.d_l;
        Self::cortical_decode_step_parts(
            &self.weights,
            &self.cortical,
            &self.graph,
            &mut self.scratch,
            &l_reg_arr[..d_l],
        );
    }

    /// Helper: Phase-aware message passing over K=8 concept graph slots -> writes `s_bar [K*D]` and `g_ctx [D]`.
    #[allow(clippy::too_many_arguments)]
    fn message_pass_graph(
        cfg: &crate::config::DcpaConfig,
        w_e: &[f32],
        w_slot_phase: &[f32],
        slots: &[f32],
        edges: &[f32],
        salience: &[f32],
        phase: &[f32],
        phase_feat_16: &mut [f32; 16],
        s_phased: &mut [f32],
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

        for slot_i in 0..k {
            let p = phase[slot_i];
            for idx in 0..8 {
                let freq = powf(128.0, -(idx as f32) / 8.0);
                let ang = p * freq;
                phase_feat_16[idx] = sinf(ang);
                phase_feat_16[8 + idx] = cosf(ang);
            }
            linear_f32(phase_feat_16, w_slot_phase, None, tmp_e);
            for m in 0..d {
                s_phased[slot_i * d + m] = slots[slot_i * d + m] + tmp_e[m];
            }
        }

        s_bar_out[..k * d].copy_from_slice(&s_phased[..k * d]);
        for i in 0..k {
            let dst = &mut s_bar_out[i * d..(i + 1) * d];
            for j in 0..k {
                let edge_off = (i * k + j) * d_r;
                linear_f32(&edges[edge_off..edge_off + d_r], w_e, None, tmp_e);
                let sj = &s_phased[j * d..(j + 1) * d];
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
        self.step_concept_from_scratch_with_opts(mean_info, sem_norm, false, None)
    }

    /// Full Spec 2 + Spec 5 concept transition handling both turn-boundary unpin and Slot-0 re-pin.
    /// Strictly ZERO heap allocations!
    pub fn step_concept_from_scratch_with_turn_opts(
        &mut self,
        mean_info: f32,
        sem_norm: f32,
        pin_trigger: bool,
        unpin_trigger: bool,
        ach_override: Option<f32>,
    ) -> ConceptStepTrace {
        if unpin_trigger {
            self.graph.slot0_pinned = false;
            self.graph.phase.fill(0.0);
        }
        self.step_concept_from_scratch_internal(mean_info, sem_norm, pin_trigger, unpin_trigger, ach_override)
    }

    /// Full Spec 2 + Spec 5 concept transition with optional `Slot-0 Thesis Pinning` trigger and `ach_override`
    /// (e.g. `ach_override = Some(0.0)` when injecting `0x12` tool observations).
    /// Strictly ZERO heap allocations!
    pub fn step_concept_from_scratch_with_opts(
        &mut self,
        mean_info: f32,
        sem_norm: f32,
        pin_trigger: bool,
        ach_override: Option<f32>,
    ) -> ConceptStepTrace {
        self.step_concept_from_scratch_internal(mean_info, sem_norm, pin_trigger, false, ach_override)
    }

    fn step_concept_from_scratch_internal(
        &mut self,
        mean_info: f32,
        sem_norm: f32,
        pin_trigger: bool,
        unpin_reset: bool,
        ach_override: Option<f32>,
    ) -> ConceptStepTrace {
        // 0. Module 1.5 Lower Cortical Sandwich (`encode_step_streaming` -> `scratch.cort_h_enc`)
        Self::cortical_encode_step_parts(
            &self.weights,
            &mut self.cortical,
            &mut self.scratch,
            self.sgsc.last_seg_end_depth,
            unpin_reset,
        );

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
        let ach = match ach_override {
            Some(ov) => ov.clamp(0.0, 1.0),
            None => (1.0 - gamma_slow) * (1.0 - gamma_hip),
        };
        let mu = gamma_hip / (gamma_slow + gamma_hip + cfg.eps);

        // pi_cort = Softplus(f_prec_cort(h_enc)) + 1e-4
        linear_f32(
            &self.scratch.cort_h_enc, // h_enc from Lower Cortical Sandwich
            &self.weights.hip_f_prec_cort_w,
            Some(&self.weights.hip_f_prec_cort_b),
            &mut self.scratch.hip_prec_bu,
        );
        for i in 0..d {
            self.scratch.hip_c_fused[i] =
                (1.0 - mu) * self.scratch.cort_h_enc[i] + mu * self.scratch.hip_c_hip[i];
            let pi_cort = softplus(self.scratch.hip_prec_bu[i]) + 1e-4;
            self.scratch.hip_prec_bu[i] = (1.0 - ach) * pi_cort;
        }

        // =====================================================================
        // 2. Module 2 BCG-PC: Top-Down Prior, Masked Pondering & K+1 -> K Merge
        // =====================================================================
        Self::message_pass_graph(
            cfg,
            &self.weights.bcg_w_e,
            &self.weights.bcg_w_slot_phase,
            &self.graph.slots,
            &self.graph.edges,
            &self.graph.salience,
            &self.graph.phase,
            &mut self.scratch.phase_feat_16,
            &mut self.scratch.s_phased,
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

        // Bayesian precision fusion + fast path projected onto unit-RMS concept manifold (norm_ponder)
        linear_f32(&self.scratch.vec_d_d, &self.weights.bcg_w_fast, None, &mut self.scratch.vec_d_e);
        for i in 0..d {
            let pi_td = self.scratch.vec_d_c[i];
            let pi_bu = self.scratch.hip_prec_bu[i].max(0.0);
            let fused = (pi_td * self.scratch.vec_d_b[i] + pi_bu * self.scratch.hip_c_fused[i])
                / (pi_td + pi_bu + cfg.eps);
            self.scratch.vec_d_e[i] = fused + one_minus_ach * self.scratch.vec_d_e[i];
        }
        rmsnorm_f32(
            &self.scratch.vec_d_e,
            &self.weights.bcg_norm_ponder,
            cfg.eps,
            &mut self.scratch.vec_d_g,
        );
        // Preserve pre-ponder z_0 for K+1 -> K working-memory merge (100% parity with Mode III parallel training;
        // System-2 pondering refines vec_d_g -> z_star for CorticalDecoder prev_z_ref).
        self.scratch.bcg_z0.copy_from_slice(&self.scratch.vec_d_g);

        // Event-driven pondering loop (L_max = 8; immediate shortcut when surprisal <= tau_skip!)
        let mut surp_cur = surprisal_0;
        let mut ponder_steps = 0usize;
        if surp_cur > cfg.tau_skip && cfg.l_max > 0 {
            // Precompute SlotCrossAttention K & V ONCE across all K=8 slots (8x less KV GEMV work!)
            for slot_i in 0..k {
                let sb = &self.scratch.s_bar[slot_i * d..(slot_i + 1) * d];
                linear_f32(
                    sb,
                    &self.weights.bcg_ca_wk,
                    None,
                    &mut self.scratch.ca_k[slot_i * d..(slot_i + 1) * d],
                );
                linear_f32(
                    sb,
                    &self.weights.bcg_ca_wv,
                    None,
                    &mut self.scratch.ca_v[slot_i * d..(slot_i + 1) * d],
                );
            }
        }
        for l_idx in 0..cfg.l_max {
            let thresh = if l_idx == 0 { cfg.tau_skip } else { cfg.tau_halt };
            if surp_cur <= thresh {
                break;
            }
            ponder_steps += 1;

            // CrossAttn(Q = z_cur, K = ca_k, V = ca_v) -> scratch.vec_d_e
            linear_f32(&self.scratch.vec_d_g, &self.weights.bcg_ca_wq, None, &mut self.scratch.vec_d_f);
            for slot_i in 0..k {
                let ck = &self.scratch.ca_k[slot_i * d..(slot_i + 1) * d];
                self.scratch.ca_scores[slot_i] = dot_f32(&self.scratch.vec_d_f, ck) * inv_sqrt_d;
            }
            softmax_inplace_f32(&mut self.scratch.ca_scores[..k]);
            self.scratch.vec_d_f.fill(0.0);
            for slot_i in 0..k {
                let cv = &self.scratch.ca_v[slot_i * d..(slot_i + 1) * d];
                axpy_f32(self.scratch.ca_scores[slot_i], cv, &mut self.scratch.vec_d_f);
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
                // Layer 1 SwiGLU (input vec_d_f is unit-RMS normalized; uses F32 or 4-row Bf16 SIMD!)
                exp.w_gate1.linear_no_bias(&self.scratch.vec_d_f, &mut self.scratch.expert_h1);
                exp.w_up1.linear_no_bias(&self.scratch.vec_d_f, &mut self.scratch.expert_h2);
                for h_i in 0..cfg.expert_hidden {
                    self.scratch.expert_h1[h_i] =
                        silu(self.scratch.expert_h1[h_i]) * self.scratch.expert_h2[h_i];
                }
                exp.w_down1.linear_no_bias(&self.scratch.expert_h1, &mut self.scratch.vec_d_e);
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
                exp.w_gate2.linear_no_bias(&self.scratch.vec_d_b, &mut self.scratch.expert_h1);
                exp.w_up2.linear_no_bias(&self.scratch.vec_d_b, &mut self.scratch.expert_h2);
                for h_i in 0..cfg.expert_hidden {
                    self.scratch.expert_h1[h_i] =
                        silu(self.scratch.expert_h1[h_i]) * self.scratch.expert_h2[h_i];
                }
                exp.w_down2.linear_no_bias(&self.scratch.expert_h1, &mut self.scratch.vec_d_b);
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

        // 3. Log-Space Contractive Gate Lambda_m in [0.920, 0.998] & Candidate slot expansion (K -> K+1)
        // Uses pre-ponder z_0 (`bcg_z0`) to match Mode II/III parallel graph trajectory 1-to-1!
        let already_pinned = self.graph.slot0_pinned;
        linear_f32(
            &self.scratch.bcg_z0,
            &self.weights.bcg_w_lambda_w,
            Some(&self.weights.bcg_w_lambda_b),
            &mut self.scratch.lam_k,
        );
        for slot_i in 0..k {
            let lam = if already_pinned && slot_i == 0 {
                1.0f32
            } else {
                let gamma_m = 0.002 + 0.081 * sigmoid(-self.scratch.lam_k[slot_i]);
                expf(-gamma_m)
            };
            for m in 0..d {
                self.scratch.s_contracted[slot_i * d + m] = self.graph.slots[slot_i * d + m] * lam;
            }
        }
        self.scratch.s_tilde[..k * d].copy_from_slice(&self.scratch.s_contracted);
        self.scratch.s_tilde[k * d..kp1 * d].copy_from_slice(&self.scratch.bcg_z0);

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
                if already_pinned && (i == 0 || j == 0) {
                    score -= 1.0e4;
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
        // Write contracted slots + merged parent in-place (preserving Slot 0 at 0.0 ULP when already_pinned)!
        for slot_i in 0..k {
            if already_pinned && slot_i == 0 {
                continue;
            }
            if slot_i != u && !(v < k && slot_i == v) {
                self.graph.slots[slot_i * d..(slot_i + 1) * d]
                    .copy_from_slice(&self.scratch.s_contracted[slot_i * d..(slot_i + 1) * d]);
            }
        }
        if !(already_pinned && u == 0) {
            rmsnorm_f32(
                &self.scratch.vec_d_d,
                &self.weights.bcg_norm_merge,
                cfg.eps,
                &mut self.graph.slots[u * d..(u + 1) * d],
            );
        }
        if v < k && !(already_pinned && v == 0) {
            // Case B: place pre-ponder z_0 into freed slot v!
            self.graph.slots[v * d..(v + 1) * d].copy_from_slice(&self.scratch.bcg_z0);
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
            dot_f32(&self.weights.bcg_w_omega, &self.scratch.bcg_z0) + self.weights.bcg_kappa_omega * surprisal_0;
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
        if already_pinned {
            self.graph.salience[0] = self.graph.salience[0].max(2.0);
        }

        // Slot Temporal Phase update (`phase [K]`)
        let pin_slot0_prev_phase = self.graph.phase[0];
        for slot_i in 0..k {
            self.scratch.phase_tilde[slot_i] = self.graph.phase[slot_i] + 1.0;
        }
        self.scratch.phase_tilde[k] = 0.0;
        let p_merged = self.scratch.phase_tilde[win_i].min(self.scratch.phase_tilde[win_j]);
        for row in 0..k {
            if row == u {
                self.graph.phase[row] = p_merged;
            } else if row == v && v < k {
                self.graph.phase[row] = 0.0;
            } else {
                self.graph.phase[row] = self.scratch.phase_tilde[row];
            }
        }
        if already_pinned {
            self.graph.phase[0] = pin_slot0_prev_phase;
        }

        // Post-merge message passing S_bar_post & g_post (in vec_d_c)
        Self::message_pass_graph(
            cfg,
            &self.weights.bcg_w_e,
            &self.weights.bcg_w_slot_phase,
            &self.graph.slots,
            &self.graph.edges,
            &self.graph.salience,
            &self.graph.phase,
            &mut self.scratch.phase_feat_16,
            &mut self.scratch.s_phased,
            &mut self.scratch.w_attn,
            &mut self.scratch.vec_d_b,
            &mut self.scratch.s_bar_post,
            &mut self.scratch.vec_d_c, // g_post
        );

        // Spec 2 Slot-0 Thesis Pinning latch (fires once on `\x1dE\n` or non-scratch `\x1fA\n`, matching bcg_pc.py)
        if pin_trigger && !already_pinned {
            unit_rmsnorm_f32(&self.scratch.vec_d_c, cfg.eps, &mut self.graph.slots[0..d]);
            self.graph.salience[0] = self.graph.salience[0].max(2.0);
            self.graph.phase[0] = 0.0;
            self.graph.slot0_pinned = true;
            Self::message_pass_graph(
                cfg,
                &self.weights.bcg_w_e,
                &self.weights.bcg_w_slot_phase,
                &self.graph.slots,
                &self.graph.edges,
                &self.graph.salience,
                &self.graph.phase,
                &mut self.scratch.phase_feat_16,
                &mut self.scratch.s_phased,
                &mut self.scratch.w_attn,
                &mut self.scratch.vec_d_b,
                &mut self.scratch.s_bar_post,
                &mut self.scratch.vec_d_c,
            );
        }

        // Update CorticalCarryState causal prev_z_ref
        self.cortical.prev_z_ref.copy_from_slice(&self.scratch.vec_d_g);
        self.cortical.has_prev_seg = true;

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
            slot0_pinned: self.graph.slot0_pinned,
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
    /// Construct orthogonal 64D register control vector:
    /// l_reg = [RMSNorm_32(lang_embed[lang_id]) || RMSNorm_32(style_embed[style_id])]
    #[inline]
    fn build_l_reg(&self, lang_id: usize, style_id: usize) -> [f32; 64] {
        let cfg = &self.weights.cfg;
        let half = cfg.d_l / 2;
        let mut out = [0.0f32; 64];
        let lid = lang_id.min(cfg.num_lang_domains.saturating_sub(1));
        let sid = style_id.min(cfg.num_style_domains.saturating_sub(1));
        let l_slice = &self.weights.pmd_lang_embed[lid * half..(lid + 1) * half];
        let s_slice = &self.weights.pmd_style_embed[sid * half..(sid + 1) * half];
        rmsnorm_f32(l_slice, &self.weights.pmd_norm_lang, cfg.eps, &mut out[..half]);
        rmsnorm_f32(s_slice, &self.weights.pmd_norm_style, cfg.eps, &mut out[half..half * 2]);
        out
    }

    pub fn broca_cq_step_and_retrieve(&mut self, lang_id: usize) -> (f32, bool) {
        self.broca_cq_step_and_retrieve_styled(lang_id, 0)
    }

    pub fn broca_cq_step_and_retrieve_styled(&mut self, lang_id: usize, style_id: usize) -> (f32, bool) {
        let cfg = &self.weights.cfg;
        let k = cfg.k_slots;
        let d = cfg.d_concept;
        let d_r = cfg.d_r;
        let d_l = cfg.d_l;
        let d_s = cfg.d_s;

        let l_reg_arr = self.build_l_reg(lang_id, style_id);
        let l_reg = &l_reg_arr[..d_l];

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

    /// Closed-Loop Mode-1 Concept Prediction & Hippocampal Expressive Retrieval!
    ///
    /// Matches Stage 5 SFT (`use_broca_cq=False`) and `cli/chat_m3.py` 1-to-1:
    ///   1. Message-passes current `graph` -> `g_ctx`
    ///   2. Predicts next concept `c_hat = f_pred_2(silu(f_pred_1(g_ctx)))` (`scratch.pmd_z_out`)
    ///   3. Computes `delta_syn = silu(w_syn([w_ctx_role(g_ctx), l_reg]) + u_syn(c_hat))`
    ///   4. Retrieves `f_out` from slow cortex + Hippocampus `M^{L<-C}` and initializes `motor.charge_e`
    /// Strictly ZERO heap allocations! Returns `rho_out`.
    pub fn mode1_step_and_retrieve_styled(&mut self, lang_id: usize, style_id: usize) -> f32 {
        let l_reg_arr = self.build_l_reg(lang_id, style_id);
        let d_l = self.weights.cfg.d_l;
        let l_reg = &l_reg_arr[..d_l];

        // 1. Upper Cortical Sandwich decoder -> `cort_z_dec` + `cort_copy_out`
        Self::cortical_decode_step_parts(
            &self.weights,
            &self.cortical,
            &self.graph,
            &mut self.scratch,
            l_reg,
        );

        let cfg = &self.weights.cfg;
        let d = cfg.d_concept;
        let d_r = cfg.d_r;
        let d_s = cfg.d_s;

        // 2. Phase-aware graph readout -> `g_ctx` (`vec_d_h`) & `c_hat` (`pmd_z_out`)
        Self::message_pass_graph(
            cfg,
            &self.weights.bcg_w_e,
            &self.weights.bcg_w_slot_phase,
            &self.graph.slots,
            &self.graph.edges,
            &self.graph.salience,
            &self.graph.phase,
            &mut self.scratch.phase_feat_16,
            &mut self.scratch.s_phased,
            &mut self.scratch.w_attn,
            &mut self.scratch.vec_d_b,
            &mut self.scratch.s_bar,
            &mut self.scratch.vec_d_h, // g_ctx
        );

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
            &mut self.scratch.pmd_z_out, // c_hat
        );

        // 3. Streaming delta_syn conditioned on g_ctx + z_dec
        linear_f32(
            &self.scratch.vec_d_h,
            &self.weights.pmd_w_ctx_role,
            None,
            &mut self.scratch.vec_dr_a,
        );
        self.scratch.vec_dr_dl[..d_r].copy_from_slice(&self.scratch.vec_dr_a);
        self.scratch.vec_dr_dl[d_r..d_r + d_l].copy_from_slice(l_reg);
        linear_f32(
            &self.scratch.vec_dr_dl,
            &self.weights.pmd_w_syn_w,
            Some(&self.weights.pmd_w_syn_b),
            &mut self.scratch.vec_ds_a,
        );
        linear_f32(
            &self.scratch.cort_z_dec,
            &self.weights.pmd_u_syn,
            None,
            &mut self.scratch.vec_ds_b,
        );
        for i in 0..d_s {
            self.scratch.pmd_delta_syn[i] = silu(self.scratch.vec_ds_a[i] + self.scratch.vec_ds_b[i]);
        }

        // 4. Slow cortical stem + pointer-copy stem + Fast Hippocampal retrieval (`c_hat`)
        linear_f32(l_reg, &self.weights.hip_w_lang_lex, None, &mut self.scratch.vec_d_b);
        for i in 0..d {
            self.scratch.vec_d_b[i] += self.scratch.cort_z_dec[i];
        }
        linear_f32(
            &self.scratch.vec_d_b,
            &self.weights.hip_w_lex_slow,
            None,
            &mut self.scratch.vec_ds_c,
        );
        unit_rmsnorm_f32(&self.scratch.vec_ds_c, cfg.eps, &mut self.scratch.vec_ds_a);
        for i in 0..d_s {
            self.scratch.vec_ds_a[i] += self.scratch.cort_copy_out[i];
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
            &mut self.scratch.vec_ds_b,
        );
        let rho_out = sigmoid(self.weights.hip_alpha_out * (gamma_sem - self.weights.hip_theta_out));
        for i in 0..d_s {
            self.scratch.pmd_f_out[i] =
                (1.0 - rho_out) * self.scratch.vec_ds_a[i] + rho_out * self.scratch.vec_ds_b[i];
        }

        self.scratch.vec_2ds_dl[..d_s].copy_from_slice(&self.scratch.pmd_f_out);
        self.scratch.vec_2ds_dl[d_s..2 * d_s].copy_from_slice(&self.scratch.pmd_delta_syn);
        self.scratch.vec_2ds_dl[2 * d_s..2 * d_s + d_l].copy_from_slice(l_reg);
        let e0_raw =
            dot_f32(&self.weights.pmd_w_e_init_w, &self.scratch.vec_2ds_dl) + self.weights.pmd_w_e_init_b;
        self.motor.charge_e = softplus(e0_raw);
        rho_out
    }

    /// Module 4 (PMD): Execute ONE streaming byte generation step from `scratch.pmd_f_out` & `scratch.pmd_delta_syn`.
    ///
    /// Writes logits into `scratch.logits_256`. Returns `(greedy_byte, charge_e, p_seg_end)`.
    /// Strictly ZERO heap allocation!
    pub fn pmd_step_byte(&mut self, prev_byte: u8, lang_id: usize, tau_step: usize) -> (u8, f32, f32) {
        self.pmd_step_byte_ex(prev_byte, lang_id, 0, tau_step, false)
    }

    pub fn pmd_step_byte_styled(
        &mut self,
        prev_byte: u8,
        lang_id: usize,
        style_id: usize,
        tau_step: usize,
    ) -> (u8, f32, f32) {
        self.pmd_step_byte_ex(prev_byte, lang_id, style_id, tau_step, false)
    }

    pub fn pmd_step_byte_ex(
        &mut self,
        prev_byte: u8,
        lang_id: usize,
        style_id: usize,
        tau_step: usize,
        extra_stack_reset: bool,
    ) -> (u8, f32, f32) {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d_l = cfg.d_l;

        let u_off = (prev_byte as usize) * d_s;
        let u_prev = &self.weights.sgsc_embed[u_off..u_off + d_s];
        let l_reg_arr = self.build_l_reg(lang_id, style_id);
        let l_reg = &l_reg_arr[..d_l];

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

        // Spec 1 multi-frame stack reset: 0x04 (EOT), 0x13 (tool suspend), 0x1E (user turn), or `\x1dE\n`
        if prev_byte == 0x04 || prev_byte == 0x13 || prev_byte == 0x1E || extra_stack_reset {
            self.motor.stack_xi = [[0.0f32; 4]; 16];
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
            // Strictly causal scale-invariant cosine prior matching sgsc_ingest_byte (8.0 * norm(W_out) @ norm(h))
            let h_norm = sqrtf(dot_f32(&self.scratch.mon_h, &self.scratch.mon_h) + 1e-2);
            let inv_h = 8.0 / h_norm;
            for i in 0..d_s {
                self.scratch.vec_ds_a[i] = self.scratch.mon_h[i] * inv_h;
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

#[cfg(test)]
mod engine_tests {
    use super::*;
    use crate::config::DcpaConfig;
    use crate::state::FastWeightPrecision;

    #[test]
    fn test_multiturn_slot0_unpin_and_repin() {
        let cfg = DcpaConfig {
            vocab_size: 256,
            d_s: 64,
            d_concept: 64,
            k_slots: 8,
            d_r: 16,
            d_merge: 32,
            d_h: 128,
            s_active: 8,
            dg_seed_lex: 17,
            dg_seed_sem: 29,
            gamma_hip: 0.995,
            lambda_eta: 0.85,
            kappa_quench: 0.9,
            c0_ema_momentum: 0.99,
            householder_max_writes: 128,
            householder_write_threshold: 0.05,
            num_experts: 4,
            top_k_experts: 2,
            expert_hidden: 64,
            l_max: 4,
            tau_skip: 0.32,
            tau_halt: 0.22,
            ponder_eta: 0.5,
            lambda_moe_bal: 0.01,
            lif_lambda: 0.85,
            lif_beta1: 1.0,
            lif_beta2: 0.5,
            lif_theta_init: 1.5,
            lif_theta_min: 0.5,
            max_seg_len: 16,
            target_spike_rate: 0.105,
            lambda_rate: 1.0,
            delta_decay: 0.05,
            d_l: 64,
            num_lang_domains: 6,
            num_style_domains: 5,
            stack_depth: 16,
            stack_tag_dim: 4,
            tau_e: 0.5,
            tau_q: 0.5,
            lambda_inh: 2.0,
            tau_done: 0.1,
            tau_repair: 0.45,
            max_repair_retries: 2,
            rope_period: 20.0,
            rope_num_harmonics: 8,
            window_size: 32,
            clause_chunk_size: 8,
            n_enc_layers: 1,
            n_dec_layers: 1,
            n_heads: 4,
            cort_expert_hidden: 64,
            shared_expert_hidden: 64,
            rope_theta_concept: 10000.0,
            lambda_var: 1.0,
            lambda_act: 0.01,
            lambda_halt: 0.01,
            eps: 1e-6,
        };
        let w = DcpaWeights::dummy_for_test(cfg);
        let mut engine = DcpaEngine::new(w, FastWeightPrecision::F32);
        let d = engine.weights.cfg.d_concept;

        // Turn 1: pin Slot-0 with evidence A, then emit EOT + Turn 2 user header
        let turn1 = b"\x1dE\nAlpha cobalt quantum fact.\n";
        for (i, &b) in turn1.iter().enumerate() {
            let tr = engine.sgsc_ingest_byte(b, i + 1 == turn1.len());
            if tr.spiked {
                let _ = engine.step_concept_from_scratch_with_turn_opts(
                    tr.mean_info,
                    tr.sem_norm,
                    tr.pin_triggered,
                    tr.unpin_triggered,
                    None,
                );
            }
        }
        assert!(engine.graph.slot0_pinned, "Turn 1 should pin Slot 0");
        let slot0_t1 = engine.graph.slots[0..d].to_vec();

        // Turn boundary + Turn 2: must unpin stale Slot-0 and re-pin Turn 2 evidence B
        let turn2 = b"\x04\x1eU\nNext question\n\x1dE\nBravo amber photon fact.\n";
        for (i, &b) in turn2.iter().enumerate() {
            let tr = engine.sgsc_ingest_byte(b, i + 1 == turn2.len());
            if tr.spiked {
                let _ = engine.step_concept_from_scratch_with_turn_opts(
                    tr.mean_info,
                    tr.sem_norm,
                    tr.pin_triggered,
                    tr.unpin_triggered,
                    None,
                );
            }
        }
        assert!(engine.graph.slot0_pinned, "Turn 2 should re-pin Slot 0");
        let max_diff = engine.graph.slots[0..d]
            .iter()
            .zip(slot0_t1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "Slot 0 remained locked across turn boundary (max_diff={})",
            max_diff
        );
    }
}
