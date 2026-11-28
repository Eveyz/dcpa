//! Pure static `no_std`-compatible, zero-heap-allocation inference engine for DCPA-Flat v3.3.
//!
//! Matches `dcpa-train-py/dcpa/model_flat.py::DCPAModelFlat.forward_streaming_sequence` 1-to-1
//! (`cos_sim >= 0.9999`, `max_abs_err < 1e-5` in float32 mode), with strictly `0 allocs/byte`
//! throughout byte ingestion, segment stepping, hippocampal fast-weight updates, and snapshot
//! capture/restore, plus bit-exact `0.0` drift on non-target inputs when Additive Dead-Zone
//! plugins are hot-mounted.

use alloc::vec;
use alloc::vec::Vec;

use crate::flat::config::DcpaFlatConfig;
use crate::flat::state::{
    BrainSnapshotV3, CarrierGraphStateRs, FlatEligState, FlatPmdState, FlatRingState,
    FlatScratchPool, FlatSgscState,
};
use crate::flat::weights::{
    DcpaFlatWeights, FlatCarrierWeights, FlatCorticalLayerWeights, FlatGraphKvWeights,
    FlatMoEWeights, PluginExpertPackRs,
};
use crate::simd::{
    apply_holographic_rope_f32, axpy_f32, cosf, dot_f32, expf, l2_norm_f32, linear_f32, logf,
    powf, rmsnorm_f32, sigmoid, silu, sinf, softplus, sqrtf, tanhf, unit_rmsnorm_f32,
};
use crate::state::{FastWeightPrecision, HippocampusFastWeights};

/// Scalar telemetry returned by `DcpaFlatEngine::sgsc_ingest_byte` (zero heap allocation).
#[derive(Debug, Clone, Copy)]
pub struct FlatByteStepOut {
    pub info: f32,
    pub delta_info: f32,
    pub spiked: bool,
    pub seg_len: usize,
    pub sem_norm: f32,
    pub mean_info: f32,
    pub seg_start_depth: i32,
    pub seg_end_depth: i32,
    pub seg_pin_trig: bool,
    pub seg_unpin_trig: bool,
}

/// Scalar telemetry returned by `DcpaFlatEngine::predict_segment_prior` (zero heap allocation).
#[derive(Debug, Clone, Copy)]
pub struct FlatPriorStepOut {
    pub u_n: f32,
    pub refined: bool,
    pub refine_gate: f32,
    pub rho_out: f32,
}

/// Scalar telemetry returned by `DcpaFlatEngine::absorb_segment` (zero heap allocation).
#[derive(Debug, Clone, Copy)]
pub struct FlatSegmentStepOut {
    pub max_beta: f32,
    pub ach: f32,
    pub gamma_slow: f32,
    pub gamma_hip: f32,
    pub slot0_pinned: bool,
}

/// Scalar telemetry returned by `DcpaFlatEngine::pmd_step_byte` (zero heap allocation).
#[derive(Debug, Clone, Copy)]
pub struct FlatPmdStepOut {
    pub pred_byte: u8,
    pub charge_e: f32,
    pub p_seg_end: f32,
}

/// Self-contained DCPA-Flat v3.3 streaming engine (all working memory allocated ONCE in `new`).
pub struct DcpaFlatEngine {
    pub weights: DcpaFlatWeights,
    pub plugins: Vec<PluginExpertPackRs>,
    pub sgsc: FlatSgscState,
    pub g1: CarrierGraphStateRs,
    pub g2: CarrierGraphStateRs,
    pub rings: FlatRingState,
    pub prev_stem: Vec<f32>, // [d_s]
    pub prev_x3: Vec<f32>,   // [D]
    pub pmd: FlatPmdState,
    pub fast_mem: HippocampusFastWeights,
    pub elig: FlatEligState,
    pub scratch: FlatScratchPool,
    pub tau_think: f32,
    pub hip_enabled: bool,
}

impl DcpaFlatEngine {
    pub fn new(weights: DcpaFlatWeights, fast_precision: FastWeightPrecision, hip_enabled: bool) -> Self {
        let cfg = weights.cfg.clone();
        let mut sgsc = FlatSgscState::new(&cfg);
        sgsc.reset(&weights);
        let mut g1 = CarrierGraphStateRs::new(&cfg);
        g1.reset_from_carrier(&cfg, &weights.carrier_g1);
        let mut g2 = CarrierGraphStateRs::new(&cfg);
        g2.reset_from_carrier(&cfg, &weights.carrier_g2);
        let rings = FlatRingState::new(&cfg);
        let prev_stem = vec![0.0f32; cfg.d_s];
        let prev_x3 = vec![0.0f32; cfg.d_concept];
        let mut pmd = FlatPmdState::new(&cfg);
        pmd.reset(&weights);
        let fast_mem = HippocampusFastWeights::new_with_dims(
            cfg.d_h,
            cfg.d_concept,
            cfg.d_s,
            cfg.gamma_hip,
            fast_precision,
        );
        let elig = FlatEligState::new(&cfg);
        let scratch = FlatScratchPool::new(&cfg);
        let tau_think = weights.tau_think_init;

        Self {
            weights,
            plugins: Vec::new(),
            sgsc,
            g1,
            g2,
            rings,
            prev_stem,
            prev_x3,
            pmd,
            fast_mem,
            elig,
            scratch,
            tau_think,
            hip_enabled,
        }
    }

    /// Reset all streaming working states & hippocampal fast memory cleanly (0 heap allocs!).
    pub fn reset_session(&mut self) {
        self.sgsc.reset(&self.weights);
        self.g1.reset_from_carrier(&self.weights.cfg, &self.weights.carrier_g1);
        self.g2.reset_from_carrier(&self.weights.cfg, &self.weights.carrier_g2);
        self.rings.reset();
        self.prev_stem.fill(0.0);
        self.prev_x3.fill(0.0);
        self.pmd.reset(&self.weights);
        self.fast_mem.wipe_zero();
        self.elig.reset();
        self.tau_think = self.weights.tau_think_init;
    }

    /// Hot-mount an Additive Dead-Zone plugin pack across all cortical layers.
    pub fn mount_expert_pack(&mut self, mut pack: PluginExpertPackRs) -> Result<(), &'static str> {
        let cfg = &self.weights.cfg;
        pack.validate(cfg.n_layers, cfg.d_concept, cfg.plugin_tau_min)?;
        pack.normalize_proto_keys(cfg.eps);
        self.scratch.ensure_plugin_hidden(pack.hidden_dim);
        if let Some(pos) = self.plugins.iter().position(|p| p.pack_id == pack.pack_id) {
            self.plugins[pos] = pack;
        } else {
            self.plugins.push(pack);
        }
        Ok(())
    }

    /// Unmount a plugin pack by `pack_id`, restoring bit-exact base model behavior.
    pub fn unmount_expert_pack(&mut self, pack_id: &str) -> bool {
        if let Some(pos) = self.plugins.iter().position(|p| p.pack_id == pack_id) {
            self.plugins.remove(pos);
            true
        } else {
            false
        }
    }

    /// Capture O(1) active working-memory snapshot into pre-allocated `snap` (strictly 0 allocs!).
    #[inline]
    pub fn capture_snapshot(&self, snap: &mut BrainSnapshotV3) {
        snap.sgsc.copy_from(&self.sgsc);
        snap.g1.copy_from(&self.g1);
        snap.g2.copy_from(&self.g2);
        snap.rings.copy_from(&self.rings);
        snap.prev_rot_stem.copy_from_slice(&self.prev_stem);
        snap.prev_x3.copy_from_slice(&self.prev_x3);
        snap.pmd.copy_from(&self.pmd);
        snap.tau_think = self.tau_think;
    }

    /// Restore O(1) active working-memory snapshot from `snap` (strictly 0 allocs!).
    #[inline]
    pub fn restore_snapshot(&mut self, snap: &BrainSnapshotV3) {
        self.sgsc.copy_from(&snap.sgsc);
        self.g1.copy_from(&snap.g1);
        self.g2.copy_from(&snap.g2);
        self.rings.copy_from(&snap.rings);
        self.prev_stem.copy_from_slice(&snap.prev_rot_stem);
        self.prev_x3.copy_from_slice(&snap.prev_x3);
        self.pmd.copy_from(&snap.pmd);
        self.tau_think = snap.tau_think;
    }

    /// Compute canonical orthogonal 64D register vector `[RMSNorm_32(lang) || RMSNorm_32(style)]`
    /// into `self.scratch.l_reg` (0 allocs!).
    #[inline]
    pub fn compute_l_reg(&mut self, lang_id: usize, style_id: usize) {
        compute_l_reg_into(&self.weights, lang_id, style_id, &mut self.scratch.l_reg);
    }

    /// Module 1 (`SGSCFlat.step_streaming_byte`): ingest 1 raw UTF-8 byte with strictly 0 allocs!
    ///
    /// On return:
    /// - `self.scratch.prior_logits_256` holds strictly causal byte prior logits.
    /// - When `out.spiked == true`:
    ///   - `self.scratch.f_stem[..d_s]` holds extracted phase-bound holographic stem `f_stem`.
    ///   - `self.scratch.x0[..D]` holds slow cortical concept projection `x0`.
    pub fn sgsc_ingest_byte(&mut self, byte: u8, force_end: bool) -> FlatByteStepOut {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d_c = cfg.d_concept;
        let l_max = cfg.max_seg_len;
        let b_usize = byte as usize;
        let b_i32 = byte as i32;

        // 1. Strictly causal row-L2-normalized cosine prior logits scaled by exp(log_temp_sgsc)
        let temp_sgsc = expf(self.weights.sgsc_log_temp.clamp(-2.0, 4.0));
        let inv_h = temp_sgsc / sqrtf(dot_f32(&self.sgsc.h, &self.sgsc.h) + 1e-2);
        for i in 0..d_s {
            self.scratch.ds_a[i] = self.sgsc.h[i] * inv_h;
        }
        linear_f32(
            &self.scratch.ds_a,
            &self.weights.w_out_row_norm,
            None,
            &mut self.scratch.prior_logits_256,
        );

        let mut max_l = f32::NEG_INFINITY;
        for &v in self.scratch.prior_logits_256.iter() {
            if v > max_l {
                max_l = v;
            }
        }
        let mut sum_exp = 0.0f32;
        for &v in self.scratch.prior_logits_256.iter() {
            sum_exp += expf(v - max_l);
        }
        let info = (max_l + logf(sum_exp)) - self.scratch.prior_logits_256[b_usize];
        let delta_info = if self.sgsc.has_prev_byte {
            let diff = info - self.sgsc.prev_info;
            if diff > 0.0 { diff } else { 0.0 }
        } else {
            0.0
        };

        // 2. Doc-masked causal k=3 depthwise byte conv + diagonal sensory recurrence:
        //    u = w0 * embed[b_t] + w1 * embed[b_{t-1}] + w2 * embed[b_{t-2}] (0 across doc start)
        let u0 = &self.weights.sgsc_embed[b_usize * d_s..(b_usize + 1) * d_s];
        let cw0 = &self.weights.sgsc_conv_w[0..d_s];
        let cw1 = &self.weights.sgsc_conv_w[d_s..2 * d_s];
        let cw2 = &self.weights.sgsc_conv_w[2 * d_s..3 * d_s];
        let b_m1 = self.sgsc.prev2_bytes[1];
        let b_m2 = self.sgsc.prev2_bytes[0];
        for i in 0..d_s {
            let mut val = u0[i] * cw0[i];
            if b_m1 >= 0 {
                val += self.weights.sgsc_embed[(b_m1 as usize) * d_s + i] * cw1[i];
            }
            if b_m2 >= 0 {
                val += self.weights.sgsc_embed[(b_m2 as usize) * d_s + i] * cw2[i];
            }
            self.scratch.ds_c[i] = val;
        }
        linear_f32(
            &self.scratch.ds_c,
            &self.weights.sgsc_w_alpha_w,
            Some(&self.weights.sgsc_w_alpha_b),
            &mut self.scratch.ds_a,
        );
        linear_f32(&self.scratch.ds_c, &self.weights.sgsc_w_u, None, &mut self.scratch.ds_b);
        for i in 0..d_s {
            let a = sigmoid(self.scratch.ds_a[i]);
            let b_inj = (1.0 - a) * silu(self.scratch.ds_b[i]);
            self.sgsc.h[i] = a * self.sgsc.h[i] + b_inj;
        }

        // 3. LIF membrane potential & UTF-8 + max_seg_len guard
        let drive = cfg.lif_beta1 * info + cfg.lif_beta2 * delta_info;
        let v_tilde = cfg.lif_lambda * self.sgsc.v + drive;
        let theta = cfg.lif_theta_min + softplus(cfg.lif_theta_scale * self.weights.sgsc_theta_raw);

        let is_cont = (b_i32 & 0xC0) == 0x80;
        let lead_rem = if (b_i32 & 0xE0) == 0xC0 {
            1
        } else if (b_i32 & 0xF0) == 0xE0 {
            2
        } else if (b_i32 & 0xF8) == 0xF0 {
            3
        } else {
            0
        };
        let utf8_rem_new = if is_cont {
            (self.sgsc.utf8_rem - 1).max(0)
        } else {
            lead_rem
        };
        let can_spike = utf8_rem_new == 0;
        let seg_len_new = self.sgsc.seg_len + 1;
        let fire = force_end || (seg_len_new >= l_max) || (can_spike && v_tilde >= theta);

        // 4. Protocol pin/unpin & causal AST bracket depth
        let pin_hit = self.sgsc.prev2_bytes[0] == 0x1D
            && self.sgsc.prev2_bytes[1] == (b'E' as i32)
            && b_i32 == 0x0A
            && self.sgsc.has_prev_byte;
        let unpin_hit = b_i32 == 0x04 || b_i32 == 0x1E;
        let post_reset = b_i32 == 0x04 || b_i32 == 0x13 || b_i32 == 0x1E || pin_hit;
        let delta_d: i32 = match byte {
            0x28 | 0x5B | 0x7B => 1,
            0x29 | 0x5D | 0x7D => -1,
            _ => 0,
        };
        let depth_new = if post_reset {
            0
        } else {
            (self.sgsc.ast_depth + delta_d).clamp(0, 31)
        };

        // 5. Store h_new into bounded segment hidden buffer
        let slot_idx = (seg_len_new - 1).min(l_max - 1);
        self.sgsc.seg_h_buf[slot_idx * d_s..(slot_idx + 1) * d_s].copy_from_slice(&self.sgsc.h);
        let seg_info_sum_upd = self.sgsc.seg_info_sum + info;
        let seg_pin_upd = self.sgsc.seg_pin || pin_hit;
        let seg_unpin_upd = self.sgsc.seg_unpin || unpin_hit;
        let out_start_depth = self.sgsc.seg_start_depth;

        let mut sem_norm = 0.0f32;
        let mut mean_info = 0.0f32;

        // 6. On spike: exact Holographic RoPE intra-segment attention -> f_stem & x0
        if fire {
            let eff_len = seg_len_new.clamp(1, l_max);
            linear_f32(&self.sgsc.h, &self.weights.sgsc_w_q, None, &mut self.scratch.ds_a);
            apply_holographic_rope_f32(
                &self.scratch.ds_a,
                eff_len as f32,
                cfg.rope_period,
                cfg.rope_num_harmonics,
                false,
                &mut self.scratch.ds_b, // q_seg
            );
            let inv_sqrt_ds = 1.0 / sqrtf(d_s as f32);
            let mut max_sc = f32::NEG_INFINITY;
            for p in 0..eff_len {
                let hp = &self.sgsc.seg_h_buf[p * d_s..(p + 1) * d_s];
                linear_f32(hp, &self.weights.sgsc_w_k, None, &mut self.scratch.ds_a);
                apply_holographic_rope_f32(
                    &self.scratch.ds_a,
                    (p + 1) as f32,
                    cfg.rope_period,
                    cfg.rope_num_harmonics,
                    false,
                    &mut self.scratch.ds_c,
                );
                let sc = dot_f32(&self.scratch.ds_c, &self.scratch.ds_b) * inv_sqrt_ds;
                self.scratch.seg_scores[p] = sc;
                if sc > max_sc {
                    max_sc = sc;
                }
            }
            let mut s_sum = 0.0f32;
            for p in 0..eff_len {
                let e = expf(self.scratch.seg_scores[p] - max_sc);
                self.scratch.seg_scores[p] = e;
                s_sum += e;
            }
            let inv_s = 1.0 / s_sum;
            self.scratch.ds_d.fill(0.0); // agg
            for p in 0..eff_len {
                let gamma_p = self.scratch.seg_scores[p] * inv_s;
                let hp = &self.sgsc.seg_h_buf[p * d_s..(p + 1) * d_s];
                linear_f32(hp, &self.weights.sgsc_w_f, None, &mut self.scratch.ds_a);
                apply_holographic_rope_f32(
                    &self.scratch.ds_a,
                    (p + 1) as f32,
                    cfg.rope_period,
                    cfg.rope_num_harmonics,
                    false,
                    &mut self.scratch.ds_c,
                );
                axpy_f32(gamma_p, &self.scratch.ds_c, &mut self.scratch.ds_d);
            }
            for i in 0..d_s {
                self.scratch.ds_a[i] = self.sgsc.h[i] + self.scratch.ds_d[i];
            }
            rmsnorm_f32(
                &self.scratch.ds_a,
                &self.weights.sgsc_norm_f,
                cfg.eps,
                &mut self.scratch.f_stem,
            );

            // Slow cortical concept projection x0 = norm_c(w_res(h_new) + w_sem(f_stem))
            linear_f32(
                &self.scratch.f_stem,
                &self.weights.sgsc_w_sem,
                None,
                &mut self.scratch.cort_x, // sem_proj
            );
            sem_norm = l2_norm_f32(&self.scratch.cort_x);
            linear_f32(
                &self.sgsc.h,
                &self.weights.sgsc_w_res,
                None,
                &mut self.scratch.cort_xn,
            );
            for i in 0..d_c {
                self.scratch.cort_xn[i] += self.scratch.cort_x[i];
            }
            rmsnorm_f32(
                &self.scratch.cort_xn,
                &self.weights.sgsc_norm_c,
                cfg.eps,
                &mut self.scratch.x0,
            );
            mean_info = seg_info_sum_upd / (seg_len_new.max(1) as f32);
        }

        // 7. Update streaming state
        self.sgsc.prev2_bytes[0] = self.sgsc.prev2_bytes[1];
        self.sgsc.prev2_bytes[1] = b_i32;
        self.sgsc.prev_info = info;
        self.sgsc.has_prev_byte = true;
        self.sgsc.utf8_rem = utf8_rem_new;
        self.sgsc.ast_depth = depth_new;
        if fire {
            self.sgsc.v = 0.0;
            self.sgsc.seg_len = 0;
            self.sgsc.seg_start_depth = depth_new;
            self.sgsc.seg_pin = false;
            self.sgsc.seg_unpin = false;
            self.sgsc.seg_info_sum = 0.0;
        } else {
            self.sgsc.v = v_tilde;
            self.sgsc.seg_len = seg_len_new;
            self.sgsc.seg_pin = seg_pin_upd;
            self.sgsc.seg_unpin = seg_unpin_upd;
            self.sgsc.seg_info_sum = seg_info_sum_upd;
        }

        FlatByteStepOut {
            info,
            delta_info,
            spiked: fire,
            seg_len: seg_len_new,
            sem_norm,
            mean_info,
            seg_start_depth: out_start_depth,
            seg_end_depth: depth_new,
            seg_pin_trig: seg_pin_upd,
            seg_unpin_trig: seg_unpin_upd,
        }
    }

    /// Strictly causal `< s` segment prior prediction BEFORE absorbing segment `s` (0 allocs!).
    ///
    /// Runs L4–L6 (with unified `W + K + 1` single-softmax on L4/L6 when enabled), computes `u_n`,
    /// conditionally runs L7–L8 ONLY when `u_n > tau_think`, computes 3-way stem confluence
    /// (`w_lex_slow(z_hat) + copy_ring + hip M(L<-C)`), computes `delta_syn`, and initializes
    /// `self.pmd.charge_e`.
    ///
    /// Outputs in `self.scratch`: `cort_z_hat`, `f_copy`, `f_hip`, `f_out`, `delta_syn`.
    pub fn predict_segment_prior(
        &mut self,
        lang_id: usize,
        style_id: usize,
        cur_start_depth: i32,
        s_idx: i32,
        unpin_trig: bool,
    ) -> FlatPriorStepOut {
        let cfg = &self.weights.cfg;
        let d_c = cfg.d_concept;
        let d_s = cfg.d_s;
        let d_l = cfg.d_l;

        compute_l_reg_into(&self.weights, lang_id, style_id, &mut self.scratch.l_reg);

        // 1. Build shared G2_prior KV ONCE (reset to G_init at s_idx == 0; otherwise current G2)
        if s_idx == 0 {
            self.scratch
                .car_prior
                .reset_from_carrier(cfg, &self.weights.carrier_g2);
            build_graph_kv_from_car_prior(cfg, &self.weights.kv_g2, &mut self.scratch);
            // Zero x3_prev view for s == 0
            self.prev_x3.fill(0.0);
        } else {
            let _ = unpin_trig; // slot0_pinned is unused by KV projection (only affects update)
            build_graph_kv(
                cfg,
                &self.weights.kv_g2,
                &self.g2.slots,
                &self.g2.edges,
                &self.g2.occ,
                &mut self.scratch,
            );
        }

        // 2. Seed x = (bos if s == 0 else prev_x3) + w_reg_pred(l_reg)
        linear_f32(
            &self.scratch.l_reg,
            &self.weights.w_reg_pred,
            None,
            &mut self.scratch.cort_wo,
        );
        if s_idx == 0 {
            for i in 0..d_c {
                self.scratch.cort_x[i] = self.weights.bos_concept[i] + self.scratch.cort_wo[i];
            }
        } else {
            for i in 0..d_c {
                self.scratch.cort_x[i] = self.prev_x3[i] + self.scratch.cort_wo[i];
            }
        }

        // 3. L4-L6 prediction layers around shared G2_prior KV + optional W=128 strict-past ring
        let pred_start = cfg.n_pre_g1_layers + cfg.n_post_g1_layers;
        let pred_end = pred_start + cfg.n_predict_full_layers;
        for layer_idx in pred_start..pred_end {
            match &self.weights.cortical_layers[layer_idx] {
                FlatCorticalLayerWeights::UnifiedWindowGraph {
                    norm_attn,
                    w_q,
                    w_uk,
                    w_uv,
                    w_reg_bias,
                    w_ior_q,
                    ior_strength,
                    b_sink,
                    w_o,
                    moe,
                } => {
                    run_unified_window_graph_layer(
                        cfg,
                        layer_idx,
                        norm_attn,
                        w_q,
                        w_uk,
                        w_uv,
                        w_reg_bias,
                        w_ior_q,
                        *ior_strength,
                        b_sink,
                        w_o,
                        moe,
                        &self.plugins,
                        &self.prev_x3,
                        &self.rings,
                        s_idx,
                        cur_start_depth,
                        &mut self.scratch,
                    );
                }
                FlatCorticalLayerWeights::GraphCrossAttn {
                    norm_attn,
                    w_q,
                    w_reg_bias,
                    w_o,
                    ior,
                    moe,
                } => {
                    run_graph_cross_attn_layer(
                        cfg,
                        layer_idx,
                        norm_attn,
                        w_q,
                        w_reg_bias,
                        w_o,
                        ior.as_ref(),
                        moe,
                        &self.plugins,
                        Some(&self.prev_x3),
                        &mut self.scratch,
                    );
                }
                FlatCorticalLayerWeights::PreG1 { .. } => unreachable!(),
            }
        }

        rmsnorm_f32(
            &self.scratch.cort_x,
            &self.weights.norm_l6_out,
            cfg.eps,
            &mut self.scratch.cort_z_hat_6,
        );

        // 4. Uncertainty head u_n and conditional L7-L8 refinement (0 FLOPs when u_n <= tau_think!)
        let u_n = dot_f32(&self.weights.w_unc_w, &self.scratch.cort_z_hat_6) + self.weights.w_unc_b;
        let refined = u_n > self.tau_think;
        let refine_gate = if refined {
            tanhf(2.0 * (u_n - self.tau_think))
        } else {
            0.0
        };

        if !refined {
            self.scratch.cort_z_hat.copy_from_slice(&self.scratch.cort_z_hat_6);
        } else {
            self.scratch.cort_x.copy_from_slice(&self.scratch.cort_z_hat_6);
            let refine_end = pred_end + cfg.n_refine_layers;
            for layer_idx in pred_end..refine_end {
                if let FlatCorticalLayerWeights::GraphCrossAttn {
                    norm_attn,
                    w_q,
                    w_reg_bias,
                    w_o,
                    ior,
                    moe,
                } = &self.weights.cortical_layers[layer_idx]
                {
                    run_graph_cross_attn_layer(
                        cfg,
                        layer_idx,
                        norm_attn,
                        w_q,
                        w_reg_bias,
                        w_o,
                        ior.as_ref(),
                        moe,
                        &self.plugins,
                        Some(&self.prev_x3),
                        &mut self.scratch,
                    );
                }
            }
            for i in 0..d_c {
                let z6 = self.scratch.cort_z_hat_6[i];
                self.scratch.cort_z_hat[i] = z6 + refine_gate * (self.scratch.cort_x[i] - z6);
            }
        }

        // 5. Three-way stem confluence: w_lex_slow(z_hat) + CopyRing + Hippocampus M(L<-C)
        linear_f32(
            &self.scratch.cort_z_hat,
            &self.weights.w_lex_slow,
            None,
            &mut self.scratch.f_slow,
        );
        run_copy_ring_step(
            cfg,
            &self.weights,
            &self.rings,
            s_idx,
            cur_start_depth,
            &mut self.scratch,
        );

        let rho_out = if self.hip_enabled {
            if s_idx == 0 {
                self.scratch.f_hip.fill(0.0);
                let g_sem = 0.0f32;
                let rho = g_sem * sigmoid(self.weights.hip_alpha_out * (g_sem - self.weights.hip_theta_out));
                for i in 0..d_s {
                    let f_cort = self.scratch.f_slow[i] + self.scratch.f_copy[i];
                    self.scratch.f_out[i] = (1.0 - rho) * f_cort;
                }
                rho
            } else {
                dg_sem_into(
                    cfg,
                    &self.weights,
                    &self.scratch.cort_z_hat,
                    &mut self.scratch.dg_scores,
                    &mut self.scratch.phi_idx,
                    &mut self.scratch.phi_val,
                );
                let g_sem = self.fast_mem.read_l_from_c(
                    &self.scratch.phi_idx,
                    &self.scratch.phi_val,
                    &mut self.scratch.f_hip,
                );
                let rho = g_sem * sigmoid(self.weights.hip_alpha_out * (g_sem - self.weights.hip_theta_out));
                for i in 0..d_s {
                    let f_cort = self.scratch.f_slow[i] + self.scratch.f_copy[i];
                    self.scratch.f_out[i] = (1.0 - rho) * f_cort + rho * self.scratch.f_hip[i];
                }
                rho
            }
        } else {
            self.scratch.f_hip.fill(0.0);
            for i in 0..d_s {
                self.scratch.f_out[i] = self.scratch.f_slow[i] + self.scratch.f_copy[i];
            }
            0.0
        };

        // 6. delta_syn = w_delta_syn([z_hat, l_reg]) + b_delta_syn
        self.scratch.vec_d_dl[..d_c].copy_from_slice(&self.scratch.cort_z_hat);
        self.scratch.vec_d_dl[d_c..d_c + d_l].copy_from_slice(&self.scratch.l_reg);
        linear_f32(
            &self.scratch.vec_d_dl,
            &self.weights.w_delta_syn_w,
            Some(&self.weights.w_delta_syn_b),
            &mut self.scratch.delta_syn,
        );

        // 7. Initialize PMD segment motor charge E_{n, 0} > 0
        self.scratch.vec_2ds_dl[..d_s].copy_from_slice(&self.scratch.f_out);
        self.scratch.vec_2ds_dl[d_s..2 * d_s].copy_from_slice(&self.scratch.delta_syn);
        self.scratch.vec_2ds_dl[2 * d_s..2 * d_s + d_l].copy_from_slice(&self.scratch.l_reg);
        let e_lin = dot_f32(&self.weights.pmd_w_e_init_w, &self.scratch.vec_2ds_dl)
            + self.weights.pmd_w_e_init_b;
        self.pmd.charge_e = softplus(e_lin);

        FlatPriorStepOut {
            u_n,
            refined,
            refine_gate,
            rho_out,
        }
    }

    /// Module 5 (`PMDFlat.step_streaming_byte`): single-byte motor step using `f_out`, `delta_syn`,
    /// `l_reg`, and `self.pmd` (strictly 0 heap allocs!).
    ///
    /// Calibrated byte logits are written into `self.scratch.byte_logits_256`.
    pub fn pmd_step_byte(
        &mut self,
        prev_byte: u8,
        tau_step: usize,
        extra_stack_reset: bool,
    ) -> FlatPmdStepOut {
        let cfg = &self.weights.cfg;
        let d_s = cfg.d_s;
        let d_l = cfg.d_l;
        let pb = prev_byte as usize;

        let u_prev = &self.weights.sgsc_embed[pb * d_s..(pb + 1) * d_s];

        // 1. Inverse holographic RoPE demodulation at -tau_step & chi gate
        linear_f32(
            &self.scratch.f_out,
            &self.weights.pmd_u_demod,
            None,
            &mut self.scratch.pmd_demod,
        );
        let tau_clamped = tau_step.min(cfg.max_seg_len) as f32;
        apply_holographic_rope_f32(
            &self.scratch.pmd_demod,
            tau_clamped,
            cfg.rope_period,
            cfg.rope_num_harmonics,
            true,
            &mut self.scratch.pmd_dstem,
        );
        self.scratch.vec_3ds[..d_s].copy_from_slice(&self.scratch.pmd_dstem);
        self.scratch.vec_3ds[d_s..2 * d_s].copy_from_slice(u_prev);
        self.scratch.vec_3ds[2 * d_s..3 * d_s].copy_from_slice(&self.scratch.delta_syn);
        let chi = sigmoid(dot_f32(&self.weights.pmd_w_chi_w, &self.scratch.vec_3ds) + self.weights.pmd_w_chi_b);
        for i in 0..d_s {
            self.scratch.pmd_ddrive[i] =
                chi * self.scratch.pmd_dstem[i] + (1.0 - chi) * self.scratch.delta_syn[i];
        }

        // 2. 128B Differentiable Pushdown Stack shift (16x4) with multi-frame resets
        self.scratch.vec_ds_dl[..d_s].copy_from_slice(u_prev);
        self.scratch.vec_ds_dl[d_s..d_s + d_l].copy_from_slice(&self.scratch.l_reg);
        let mut act_logits = [0.0f32; 3];
        linear_f32(
            &self.scratch.vec_ds_dl,
            &self.weights.pmd_w_act_w,
            Some(&self.weights.pmd_w_act_b),
            &mut act_logits,
        );
        let mut max_a = act_logits[0];
        if act_logits[1] > max_a {
            max_a = act_logits[1];
        }
        if act_logits[2] > max_a {
            max_a = act_logits[2];
        }
        let e0 = expf(act_logits[0] - max_a);
        let e1 = expf(act_logits[1] - max_a);
        let e2 = expf(act_logits[2] - max_a);
        let inv_e = 1.0 / (e0 + e1 + e2);
        let p_push = e0 * inv_e;
        let p_pop = e1 * inv_e;
        let p_hold = e2 * inv_e;

        let mut v_tag = [0.0f32; 4];
        linear_f32(
            u_prev,
            &self.weights.pmd_w_tag_w,
            Some(&self.weights.pmd_w_tag_b),
            &mut v_tag,
        );
        for v in v_tag.iter_mut() {
            *v = tanhf(*v);
        }

        let ctrl_reset = prev_byte == 0x04 || prev_byte == 0x13 || prev_byte == 0x1E || extra_stack_reset;
        if ctrl_reset {
            self.pmd.xi = [[0.0f32; 4]; 16];
        }
        let old_xi = self.pmd.xi;
        for row in 0..16 {
            let pushed = if row == 0 { v_tag } else { old_xi[row - 1] };
            let popped = if row + 1 < 16 { old_xi[row + 1] } else { [0.0f32; 4] };
            for c in 0..4 {
                self.pmd.xi[row][c] =
                    p_hold * old_xi[row][c] + p_push * pushed[c] + p_pop * popped[c];
            }
        }
        let mut xi_top2 = [0.0f32; 8];
        xi_top2[0..4].copy_from_slice(&self.pmd.xi[0]);
        xi_top2[4..8].copy_from_slice(&self.pmd.xi[1]);
        linear_f32(&xi_top2, &self.weights.pmd_w_st, None, &mut self.scratch.pmd_st_out);

        // 3. Causal byte motor diagonal recurrence
        for i in 0..d_s {
            self.scratch.pmd_xin[i] =
                u_prev[i] + self.scratch.pmd_ddrive[i] + self.scratch.pmd_st_out[i];
        }
        linear_f32(
            &self.scratch.pmd_xin,
            &self.weights.pmd_w_alpha_mot_w,
            Some(&self.weights.pmd_w_alpha_mot_b),
            &mut self.scratch.ds_a,
        );
        linear_f32(
            &self.scratch.pmd_xin,
            &self.weights.pmd_w_m,
            None,
            &mut self.scratch.ds_b,
        );
        for i in 0..d_s {
            let a = sigmoid(self.scratch.ds_a[i]);
            let b_mot = (1.0 - a) * silu(self.scratch.ds_b[i]);
            self.pmd.m[i] = a * self.pmd.m[i] + b_mot;
        }

        // 4. Calibrated row-L2-normalized shared output vocab logits
        let temp_pmd = expf(self.weights.pmd_log_temp.clamp(-2.0, 4.0));
        let inv_m = temp_pmd / sqrtf(dot_f32(&self.pmd.m, &self.pmd.m) + 1e-2);
        for i in 0..d_s {
            self.scratch.pmd_m_norm[i] = self.pmd.m[i] * inv_m;
        }
        linear_f32(
            &self.scratch.pmd_m_norm,
            &self.weights.w_out_row_norm,
            None,
            &mut self.scratch.byte_logits_256,
        );
        let mut best_b = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in self.scratch.byte_logits_256.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_b = i;
            }
        }

        // 5. Charge depletion gate
        unit_rmsnorm_f32(&self.pmd.m, cfg.eps, &mut self.scratch.pmd_m_rms);
        self.scratch.vec_2ds[..d_s].copy_from_slice(&self.scratch.pmd_m_rms);
        self.scratch.vec_2ds[d_s..2 * d_s].copy_from_slice(&self.scratch.pmd_ddrive);
        let drain_lin =
            dot_f32(&self.weights.pmd_w_drain_w, &self.scratch.vec_2ds) + self.weights.pmd_w_drain_b;
        let drain = softplus(drain_lin);
        self.pmd.charge_e -= drain;
        let tau_e = if cfg.tau_e > 1e-4 { cfg.tau_e } else { 1e-4 };
        let p_seg_end = sigmoid(-self.pmd.charge_e / tau_e);

        FlatPmdStepOut {
            pred_byte: best_b as u8,
            charge_e: self.pmd.charge_e,
            p_seg_end,
        }
    }

    /// Absorb completed segment `s_idx` (using `self.scratch.f_stem`, `self.scratch.x0`) into
    /// stem ring, L1, Carrier G1, L2–L3, Hippocampus + Carrier G2, and optional semantic ring.
    ///
    /// Strictly 0 heap allocations!
    /// Leaves `self.scratch.cort_x1`, `self.scratch.cort_x3`, and `self.scratch.car_prior` (G2_prior)
    /// populated for inspection/parity verification.
    pub fn absorb_segment(
        &mut self,
        s_idx: i32,
        seg_end_depth: i32,
        pin_trig: bool,
        unpin_trig: bool,
        lang_id: usize,
        style_id: usize,
        sem_norm: f32,
        mean_info: f32,
    ) -> FlatSegmentStepOut {
        let cfg = &self.weights.cfg;
        let d_c = cfg.d_concept;
        let k_slots = cfg.k_slots;
        let d_r = cfg.d_r;
        let s_act = cfg.s_active;
        let d_s = cfg.d_s;
        let reset_s = s_idx == 0;

        compute_l_reg_into(&self.weights, lang_id, style_id, &mut self.scratch.l_reg);

        // 1. Push pure f_stem into stem ring (matches `CopyRing.push_ring` BEFORE L1)
        self.rings.push_stem(&self.scratch.f_stem, s_idx, seg_end_depth);

        // 2. L1 (pre-G1): optional W=128 causal local window (`0 <= Δs < W`) over `f_stem` + MoE
        self.scratch.cort_x.copy_from_slice(&self.scratch.x0);
        if let FlatCorticalLayerWeights::PreG1 { window, moe } = &self.weights.cortical_layers[0] {
            if let Some(win) = window {
                run_pre_g1_window_attn(
                    cfg,
                    win,
                    &self.rings,
                    s_idx,
                    seg_end_depth,
                    &mut self.scratch,
                );
            }
            run_decoupled_moe(cfg, 0, moe, &self.plugins, &mut self.scratch);
        }
        self.scratch.cort_x1.copy_from_slice(&self.scratch.cort_x);

        // 3. Carrier G1: free-flowing post-update contractive graph step
        step_carrier_single(
            cfg,
            &self.weights.carrier_g1,
            false, // pin_enabled = false
            &mut self.g1,
            CarrierInput::CortX1,
            reset_s,
            false,
            false,
            &mut self.scratch,
        );

        // 4. L2–L3 (post-G1): shared G1 KV graph cross-attention + MoE + optional c_kv_sem
        build_graph_kv(
            cfg,
            &self.weights.kv_g1,
            &self.g1.slots,
            &self.g1.edges,
            &self.g1.occ,
            &mut self.scratch,
        );
        self.scratch.cort_x.copy_from_slice(&self.scratch.cort_x1);
        for layer_idx in 1..(1 + cfg.n_post_g1_layers) {
            if let FlatCorticalLayerWeights::GraphCrossAttn {
                norm_attn,
                w_q,
                w_reg_bias,
                w_o,
                ior,
                moe,
            } = &self.weights.cortical_layers[layer_idx]
            {
                run_graph_cross_attn_layer(
                    cfg,
                    layer_idx,
                    norm_attn,
                    w_q,
                    w_reg_bias,
                    w_o,
                    ior.as_ref(),
                    moe,
                    &self.plugins,
                    None,
                    &mut self.scratch,
                );
            }
        }
        rmsnorm_f32(
            &self.scratch.cort_x,
            &self.weights.norm_l3_out,
            cfg.eps,
            &mut self.scratch.cort_x3,
        );
        if let (Some(w_down), Some(norm_sem)) = (&self.weights.w_sem_down, &self.weights.norm_sem_ring) {
            linear_f32(
                &self.scratch.cort_x3,
                w_down,
                None,
                &mut self.scratch.cort_sem_tmp,
            );
            rmsnorm_f32(
                &self.scratch.cort_sem_tmp,
                norm_sem,
                cfg.eps,
                &mut self.scratch.cort_c_kv_sem,
            );
        }

        // 5. Hippocampus + Carrier G2 causal interleaved step
        let mut max_beta = 0.0f32;
        let mut ach_out = 0.0f32;
        let mut gamma_slow_out = 0.0f32;
        let mut gamma_hip_out = 0.0f32;

        if self.hip_enabled {
            if reset_s {
                self.fast_mem.wipe_zero();
                self.elig.reset();
            }

            // 5a. Dual-morpheme whitened lexical DG
            dg_lex_from_scratch(
                cfg,
                &self.weights,
                &self.prev_stem,
                reset_s,
                &mut self.scratch,
            );

            // 5b. Comprehension read HIPR strictly BEFORE this step's writes
            let gamma_hip = self.fast_mem.read_c_from_l(
                &self.scratch.psi_idx,
                &self.scratch.psi_val,
                &mut self.scratch.hip_c_hip,
            );
            let gamma_slow = sigmoid(
                self.weights.hip_alpha_1 * (self.weights.hip_theta_i - mean_info)
                    + self.weights.hip_alpha_2 * (sem_norm - self.weights.hip_theta_n),
            );
            let ach = (1.0 - gamma_slow) * (1.0 - gamma_hip);
            let mu = gamma_hip / (gamma_slow + gamma_hip + cfg.eps);
            for i in 0..d_c {
                self.scratch.hip_c_fused[i] =
                    (1.0 - mu) * self.scratch.cort_x3[i] + mu * self.scratch.hip_c_hip[i];
            }
            ach_out = ach;
            gamma_slow_out = gamma_slow;
            gamma_hip_out = gamma_hip;

            // 5c. Carrier G2 single step with c_fused (populates `scratch.car_prior` & `car_evict_w`)
            step_carrier_single(
                cfg,
                &self.weights.carrier_g2,
                true, // pin_enabled = true for G2!
                &mut self.g2,
                CarrierInput::HipCFused,
                reset_s,
                pin_trig,
                unpin_trig,
                &mut self.scratch,
            );

            // 5d. Write-Before-Merge 3-factor crystallization beta[0..K]
            for k in 0..k_slots {
                let slot_slice = &self.scratch.car_prior.slots[k * d_c..(k + 1) * d_c];
                let slot_norm = l2_norm_f32(slot_slice);
                let mut edge_norm_sum = 0.0f32;
                for j in 0..k_slots {
                    let e_off = (k * k_slots + j) * d_r;
                    edge_norm_sum += l2_norm_f32(&self.scratch.car_prior.edges[e_off..e_off + d_r]);
                }
                let q_cryst = sigmoid(
                    self.weights.hip_alpha_q
                        * (slot_norm + self.weights.hip_gamma_e * edge_norm_sum - self.weights.hip_theta_q),
                );
                let ev_w = self.scratch.car_evict_w[k].clamp(0.0, 1.0);
                let beta_raw = self.elig.elig_eta[k] * q_cryst * ev_w;
                let b = if beta_raw > cfg.hip_write_threshold {
                    beta_raw
                } else {
                    0.0
                };
                self.scratch.hip_beta[k] = b;
                if b > max_beta {
                    max_beta = b;
                }
            }

            // 5e. Exact sequential K-slot Gated-Delta updates on M(C<-L) and M(L<-C) + familiarity
            for k in 0..k_slots {
                let bk = self.scratch.hip_beta[k];
                if bk > 0.0 {
                    let s_bar = &self.scratch.car_prior.slots[k * d_c..(k + 1) * d_c];
                    dg_sem_into(
                        cfg,
                        &self.weights,
                        s_bar,
                        &mut self.scratch.dg_scores,
                        &mut self.scratch.phi_idx,
                        &mut self.scratch.phi_val,
                    );
                    let e_idx = &self.elig.elig_idx[k * s_act..(k + 1) * s_act];
                    let e_val = &self.elig.elig_val[k * s_act..(k + 1) * s_act];
                    let f_tilde = &self.elig.elig_f[k * d_s..(k + 1) * d_s];
                    self.fast_mem.write_slot_inplace(
                        bk,
                        e_idx,
                        e_val,
                        s_bar,
                        &self.scratch.phi_idx,
                        &self.scratch.phi_val,
                        f_tilde,
                        &mut self.scratch.hip_proj_c,
                        &mut self.scratch.hip_proj_l,
                    );
                }
            }

            // 5f. End-of-step lazy gamma_hip decay on M(C<-L), M(L<-C), u_lex, u_sem
            self.fast_mem.advance_step();

            // 5g. Update eligibility traces after Write-Before-Merge using true G2 slot attribution & merge pair
            let kappa = cfg.kappa_quench;
            let lam = cfg.lambda_eta;
            for k in 0..k_slots {
                let eta_q = ((1.0 - kappa * self.scratch.hip_beta[k]) * self.elig.elig_eta[k]).clamp(0.0, 1.0);
                self.elig.elig_eta[k] = eta_q * lam;
            }
            // Asymmetric argmax inheritance into merged parent slot `win_i` (u_star) when merge active
            if self.scratch.car_merge_active {
                let wi = self.scratch.car_win_i.min(k_slots - 1);
                let wj = self.scratch.car_win_j.min(k_slots - 1);
                let p_star = if self.elig.elig_eta[wi] >= self.elig.elig_eta[wj] { wi } else { wj };
                if p_star != wi {
                    let (src_i, dst_i) = (p_star * s_act, wi * s_act);
                    self.elig.elig_idx.copy_within(src_i..src_i + s_act, dst_i);
                    self.elig.elig_val.copy_within(src_i..src_i + s_act, dst_i);
                    let (src_f, dst_f) = (p_star * d_s, wi * d_s);
                    self.elig.elig_f.copy_within(src_f..src_f + d_s, dst_f);
                    self.elig.elig_eta[wi] = self.elig.elig_eta[p_star];
                }
            }
            let slot_star = self.scratch.car_slot_star.min(k_slots - 1);
            if ach >= self.elig.elig_eta[slot_star] || self.scratch.car_evict_w[slot_star] > 0.5 {
                self.elig.elig_idx[slot_star * s_act..(slot_star + 1) * s_act]
                    .copy_from_slice(&self.scratch.psi_idx);
                self.elig.elig_val[slot_star * s_act..(slot_star + 1) * s_act]
                    .copy_from_slice(&self.scratch.psi_val);
                self.elig.elig_f[slot_star * d_s..(slot_star + 1) * d_s]
                    .copy_from_slice(&self.scratch.f_stem);
                self.elig.elig_eta[slot_star] = ach;
            }

            self.prev_stem.copy_from_slice(&self.scratch.f_stem);
        } else {
            step_carrier_single(
                cfg,
                &self.weights.carrier_g2,
                true,
                &mut self.g2,
                CarrierInput::CortX3,
                reset_s,
                pin_trig,
                unpin_trig,
                &mut self.scratch,
            );
        }

        // 6. Push optional semantic ring & update prev_x3
        if self.rings.sem_ring.is_some() {
            self.rings.push_sem(&self.scratch.cort_c_kv_sem);
        }
        self.prev_x3.copy_from_slice(&self.scratch.cort_x3);

        FlatSegmentStepOut {
            max_beta,
            ach: ach_out,
            gamma_slow: gamma_slow_out,
            gamma_hip: gamma_hip_out,
            slot0_pinned: self.g2.slot0_pinned,
        }
    }
}

// =============================================================================
// Internal Zero-Alloc Mathematical Kernels (1-to-1 with Python Reference)
// =============================================================================

#[inline]
fn compute_l_reg_into(weights: &DcpaFlatWeights, lang_id: usize, style_id: usize, out: &mut [f32]) {
    let cfg = &weights.cfg;
    let half = cfg.d_l / 2;
    let lid = lang_id.min(cfg.num_lang_domains - 1);
    let sid = style_id.min(cfg.num_style_domains - 1);
    let l_emb = &weights.sgsc_lang_embed[lid * half..(lid + 1) * half];
    let s_emb = &weights.sgsc_style_embed[sid * half..(sid + 1) * half];
    rmsnorm_f32(l_emb, &weights.sgsc_norm_lang, cfg.eps, &mut out[..half]);
    rmsnorm_f32(s_emb, &weights.sgsc_norm_style, cfg.eps, &mut out[half..2 * half]);
}

/// Apply dual-coordinate 2D Concept-RoPE (`3/4` relative concept order + `1/4` AST scope depth)
/// to a single head vector `x_head [d_head]` matching `dcpa/ops/rope.py::apply_concept_rope_2d` 1-to-1.
#[inline]
fn apply_concept_rope_2d_head(
    x_head: &[f32],
    pos: f32,
    depth: f32,
    theta_base: f32,
    scope_period: f32,
    out: &mut [f32],
) {
    let d_head = x_head.len();
    let half = d_head / 2;
    let scope_pairs = (half / 4).max(1);
    let pos_pairs = half - scope_pairs;
    let pos_dim = (pos_pairs * 2).max(2) as f32;
    let scope_omega = (2.0 * core::f32::consts::PI) / scope_period;

    for i in 0..pos_pairs {
        let freq = powf(theta_base, -2.0 * (i as f32) / pos_dim);
        let angle = pos * freq;
        let c = cosf(angle);
        let s = sinf(angle);
        let xe = x_head[2 * i];
        let xo = x_head[2 * i + 1];
        out[2 * i] = xe * c - xo * s;
        out[2 * i + 1] = xe * s + xo * c;
    }
    for j in 0..scope_pairs {
        let k = pos_pairs + j;
        let angle = depth * scope_omega * ((j + 1) as f32);
        let c = cosf(angle);
        let s = sinf(angle);
        let xe = x_head[2 * k];
        let xo = x_head[2 * k + 1];
        out[2 * k] = xe * c - xo * s;
        out[2 * k + 1] = xe * s + xo * c;
    }
}

/// Apply 2D Concept-RoPE across all `n_heads` heads stored contiguously as `[H * d_head]`.
#[inline]
fn apply_concept_rope_2d_all_heads(
    x: &[f32],
    n_heads: usize,
    d_head: usize,
    pos: f32,
    depth: f32,
    theta_base: f32,
    scope_period: f32,
    out: &mut [f32],
) {
    for h in 0..n_heads {
        let off = h * d_head;
        apply_concept_rope_2d_head(
            &x[off..off + d_head],
            pos,
            depth,
            theta_base,
            scope_period,
            &mut out[off..off + d_head],
        );
    }
}

/// Exact softmax over `scores[..n]` handling `-inf` safely (0 allocs!).
#[inline]
fn softmax_slice_exact(scores: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in scores.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in scores.iter_mut() {
        let e = if v.is_finite() { expf(*v - max_v) } else { 0.0 };
        *v = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in scores.iter_mut() {
            *v *= inv;
        }
    }
}

/// Top-k selection (descending, tie-break toward lowest index) + ReLU + unit L2 norm + energy mask.
#[inline]
fn topk_relu_unit_norm_masked(
    scores: &[f32],
    has_energy: bool,
    out_idx: &mut [usize],
    out_val: &mut [f32],
) {
    let k = out_idx.len();
    for j in 0..k {
        out_idx[j] = 0;
        out_val[j] = f32::NEG_INFINITY;
    }
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
    if !has_energy {
        out_val.fill(0.0);
        return;
    }
    let mut sq = 0.0f32;
    for j in 0..k {
        let r = if out_val[j] > 0.0 { out_val[j] } else { 0.0 };
        out_val[j] = r;
        sq += r * r;
    }
    let inv = 1.0 / (sqrtf(sq) + 1e-8);
    for j in 0..k {
        out_val[j] *= inv;
    }
}

fn dg_lex_from_scratch(
    cfg: &DcpaFlatConfig,
    weights: &DcpaFlatWeights,
    f_prev: &[f32],
    seg_reset: bool,
    scratch: &mut FlatScratchPool,
) {
    let d_s = cfg.d_s;
    let d_h = cfg.d_h;
    if seg_reset {
        scratch.dg_f_bound.copy_from_slice(&scratch.f_stem);
    } else {
        linear_f32(f_prev, &weights.dg_q_rot, None, &mut scratch.dg_rot_prev);
        let inv_sqrt2 = sqrtf(0.5);
        for i in 0..d_s {
            scratch.dg_f_bound[i] = (scratch.f_stem[i] + scratch.dg_rot_prev[i]) * inv_sqrt2;
        }
    }
    // Residual whitening preconditioner: f_w = f_bound + w2(silu(w1(f_bound)))
    linear_f32(
        &scratch.dg_f_bound,
        &weights.dg_w_whiten_1_w,
        Some(&weights.dg_w_whiten_1_b),
        &mut scratch.dg_w1_h,
    );
    for v in scratch.dg_w1_h.iter_mut() {
        *v = silu(*v);
    }
    linear_f32(
        &scratch.dg_w1_h,
        &weights.dg_w_whiten_2_w,
        Some(&weights.dg_w_whiten_2_b),
        &mut scratch.dg_f_w,
    );
    for i in 0..d_s {
        scratch.dg_f_w[i] += scratch.dg_f_bound[i];
    }

    linear_f32(&scratch.dg_f_w, &weights.dg_r_lex, None, &mut scratch.dg_scores);
    for i in 0..d_h {
        scratch.dg_scores[i] += weights.dg_b_homeo[i];
    }
    let has_energy = dot_f32(&scratch.dg_f_bound, &scratch.dg_f_bound) > 1e-12;
    topk_relu_unit_norm_masked(
        &scratch.dg_scores,
        has_energy,
        &mut scratch.psi_idx,
        &mut scratch.psi_val,
    );
}

fn dg_sem_into(
    cfg: &DcpaFlatConfig,
    weights: &DcpaFlatWeights,
    concept: &[f32],
    dg_scores: &mut [f32],
    out_idx: &mut [usize],
    out_val: &mut [f32],
) {
    let d_h = cfg.d_h;
    linear_f32(concept, &weights.dg_r_sem, None, dg_scores);
    for i in 0..d_h {
        dg_scores[i] += weights.dg_b_homeo_sem[i];
    }
    let has_energy = dot_f32(concept, concept) > 1e-12;
    topk_relu_unit_norm_masked(dg_scores, has_energy, out_idx, out_val);
}

/// Shared multi-head Graph KV projection computed ONCE from `scratch.car_prior`.
fn build_graph_kv_from_car_prior(
    cfg: &DcpaFlatConfig,
    kv_w: &FlatGraphKvWeights,
    scratch: &mut FlatScratchPool,
) {
    build_graph_kv_split(
        cfg,
        kv_w,
        &scratch.car_prior.slots,
        &scratch.car_prior.edges,
        &scratch.car_prior.occ,
        &mut scratch.gkv_k,
        &mut scratch.gkv_v,
        &mut scratch.gkv_occ_log,
        &mut scratch.gkv_rel_bias,
        &mut scratch.gkv_edge_in,
        &mut scratch.gkv_edge_out,
        &mut scratch.gkv_tmp_d1,
        &mut scratch.gkv_tmp_d2,
        &mut scratch.gkv_s_enr,
        &mut scratch.gkv_proj_d,
        &mut scratch.gkv_rb_h,
    );
}

/// Shared multi-head Graph KV projection computed ONCE per graph (`G1` or `G2_prior`).
fn build_graph_kv(
    cfg: &DcpaFlatConfig,
    kv_w: &FlatGraphKvWeights,
    slots: &[f32],
    edges: &[f32],
    occ: &[f32],
    scratch: &mut FlatScratchPool,
) {
    build_graph_kv_split(
        cfg,
        kv_w,
        slots,
        edges,
        occ,
        &mut scratch.gkv_k,
        &mut scratch.gkv_v,
        &mut scratch.gkv_occ_log,
        &mut scratch.gkv_rel_bias,
        &mut scratch.gkv_edge_in,
        &mut scratch.gkv_edge_out,
        &mut scratch.gkv_tmp_d1,
        &mut scratch.gkv_tmp_d2,
        &mut scratch.gkv_s_enr,
        &mut scratch.gkv_proj_d,
        &mut scratch.gkv_rb_h,
    );
}

#[allow(clippy::too_many_arguments)]
fn build_graph_kv_split(
    cfg: &DcpaFlatConfig,
    kv_w: &FlatGraphKvWeights,
    slots: &[f32],
    edges: &[f32],
    occ: &[f32],
    gkv_k: &mut [f32],
    gkv_v: &mut [f32],
    gkv_occ_log: &mut [f32],
    gkv_rel_bias: &mut [f32],
    gkv_edge_in: &mut [f32],
    gkv_edge_out: &mut [f32],
    gkv_tmp_d1: &mut [f32],
    gkv_tmp_d2: &mut [f32],
    gkv_s_enr: &mut [f32],
    gkv_proj_d: &mut [f32],
    gkv_rb_h: &mut [f32],
) {
    let k_slots = cfg.k_slots;
    let d_c = cfg.d_concept;
    let d_r = cfg.d_r;
    let n_heads = cfg.n_heads;
    let d_head = cfg.d_head();

    let mut occ_sum = 0.0f32;
    let mut w_occ = [0.0f32; 16];
    for k in 0..k_slots {
        let oc = occ[k].clamp(cfg.occ_min, cfg.occ_max);
        w_occ[k] = oc;
        occ_sum += oc;
        gkv_occ_log[k] = logf(oc);
    }
    let inv_occ = 1.0 / if occ_sum > cfg.eps { occ_sum } else { cfg.eps };
    for k in 0..k_slots {
        w_occ[k] *= inv_occ;
    }

    gkv_edge_in.fill(0.0);
    gkv_edge_out.fill(0.0);
    for i in 0..k_slots {
        let wi = w_occ[i];
        for j in 0..k_slots {
            let wj = w_occ[j];
            let e_off = (i * k_slots + j) * d_r;
            let e_ij = &edges[e_off..e_off + d_r];
            axpy_f32(wi, e_ij, &mut gkv_edge_in[j * d_r..(j + 1) * d_r]);
            axpy_f32(wj, e_ij, &mut gkv_edge_out[i * d_r..(i + 1) * d_r]);
        }
    }

    for k in 0..k_slots {
        let s_k = &slots[k * d_c..(k + 1) * d_c];
        let ein_k = &gkv_edge_in[k * d_r..(k + 1) * d_r];
        let eout_k = &gkv_edge_out[k * d_r..(k + 1) * d_r];

        rmsnorm_f32(s_k, &kv_w.norm_slot, cfg.eps, gkv_s_enr);
        linear_f32(ein_k, &kv_w.w_edge_in, None, gkv_tmp_d1);
        linear_f32(eout_k, &kv_w.w_edge_out, None, gkv_tmp_d2);
        for c in 0..d_c {
            gkv_s_enr[c] += tanhf(gkv_tmp_d1[c]) + tanhf(gkv_tmp_d2[c]);
        }

        linear_f32(gkv_s_enr, &kv_w.w_gk, None, gkv_proj_d);
        for h in 0..n_heads {
            let dst_off = (h * k_slots + k) * d_head;
            gkv_k[dst_off..dst_off + d_head]
                .copy_from_slice(&gkv_proj_d[h * d_head..(h + 1) * d_head]);
        }

        linear_f32(gkv_s_enr, &kv_w.w_gv, None, gkv_proj_d);
        for h in 0..n_heads {
            let dst_off = (h * k_slots + k) * d_head;
            gkv_v[dst_off..dst_off + d_head]
                .copy_from_slice(&gkv_proj_d[h * d_head..(h + 1) * d_head]);
        }

        linear_f32(ein_k, &kv_w.w_rel_bias, None, gkv_rb_h);
        for h in 0..n_heads {
            gkv_rel_bias[h * k_slots + k] = 2.0 * tanhf(gkv_rb_h[h]);
        }
    }
}

/// Decoupled Base MoE (1 Shared + E Routed Top-K) + Additive Dead-Zone Plugin Pool.
///
/// Guarantees:
/// - Strictly 0 heap allocations.
/// - When no plugin exceeds its dead-zone `tau` (`d == 0.0`), skips ALL plugin matmuls ->
///   bit-exact `0.0` drift and `0` extra FLOPs!
fn run_decoupled_moe(
    cfg: &DcpaFlatConfig,
    layer_idx: usize,
    moe: &FlatMoEWeights,
    plugins: &[PluginExpertPackRs],
    scratch: &mut FlatScratchPool,
) {
    let d_c = cfg.d_concept;
    let n_exp = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let sh = cfg.shared_hidden;
    let rh = cfg.routed_hidden;

    rmsnorm_f32(&scratch.cort_x, &moe.norm, cfg.eps, &mut scratch.moe_xn);

    // 1. Always-on Shared SwiGLU Expert
    linear_f32(&scratch.moe_xn, &moe.shared_w_gu, None, &mut scratch.moe_shared_gu);
    for i in 0..sh {
        scratch.moe_shared_h[i] = silu(scratch.moe_shared_gu[i]) * scratch.moe_shared_gu[sh + i];
    }
    linear_f32(
        &scratch.moe_shared_h,
        &moe.shared_w_d,
        None,
        &mut scratch.moe_y_shared,
    );

    // 2. Base Router: `b_bal` used ONLY for selection, NEVER multiplied into synthesis gate!
    linear_f32(
        &scratch.moe_xn,
        &moe.router_w,
        Some(&moe.router_b),
        &mut scratch.moe_router,
    );
    for e in 0..n_exp {
        let clamped = if scratch.moe_router[e] < cfg.router_logit_min {
            cfg.router_logit_min
        } else {
            scratch.moe_router[e]
        };
        scratch.moe_router[e] = clamped;
        scratch.moe_sel[e] = clamped + moe.b_bal[e];
    }

    // Select top_k experts (stack arrays support up to top_k=8 cleanly with 0 allocs)
    let mut top_idx = [0usize; 8];
    let mut top_sel = [f32::NEG_INFINITY; 8];
    for e in 0..n_exp {
        let sc = scratch.moe_sel[e];
        if sc > top_sel[top_k - 1] {
            let mut p = top_k - 1;
            while p > 0 && sc > top_sel[p - 1] {
                top_sel[p] = top_sel[p - 1];
                top_idx[p] = top_idx[p - 1];
                p -= 1;
            }
            top_sel[p] = sc;
            top_idx[p] = e;
        }
    }
    let mut g_base = [0.0f32; 8];
    if top_k == 1 {
        g_base[0] = sigmoid(scratch.moe_router[top_idx[0]]);
    } else {
        let mut mx = f32::NEG_INFINITY;
        for k_i in 0..top_k {
            let l = scratch.moe_router[top_idx[k_i]];
            g_base[k_i] = l;
            if l > mx {
                mx = l;
            }
        }
        let mut s = 0.0f32;
        for k_i in 0..top_k {
            g_base[k_i] = expf(g_base[k_i] - mx);
            s += g_base[k_i];
        }
        let inv = 1.0 / s;
        for k_i in 0..top_k {
            g_base[k_i] *= inv;
        }
    }

    // 3. Base Routed SwiGLU evaluation
    scratch.moe_y_routed.fill(0.0);
    let w_gu_stride = (2 * rh) * d_c;
    let w_d_stride = d_c * rh;
    for k_i in 0..top_k {
        let e = top_idx[k_i];
        let w_gu_e = &moe.w_gu[e * w_gu_stride..(e + 1) * w_gu_stride];
        let w_d_e = &moe.w_d[e * w_d_stride..(e + 1) * w_d_stride];
        linear_f32(&scratch.moe_xn, w_gu_e, None, &mut scratch.moe_routed_gu);
        for j in 0..rh {
            scratch.moe_routed_h[j] = silu(scratch.moe_routed_gu[j]) * scratch.moe_routed_gu[rh + j];
        }
        linear_f32(&scratch.moe_routed_h, w_d_e, None, &mut scratch.moe_y_tmp);
        axpy_f32(g_base[k_i], &scratch.moe_y_tmp, &mut scratch.moe_y_routed);
    }

    // Base residual output: y_base = x + y_shared + y_routed
    for i in 0..d_c {
        scratch.cort_x[i] = scratch.cort_x[i] + scratch.moe_y_shared[i] + scratch.moe_y_routed[i];
    }

    // 4. Orthogonal Additive Dead-Zone Plugin Pool (0.0 bit drift & 0 extra FLOPs when inactive!)
    if plugins.is_empty() {
        return;
    }
    let norm_x = sqrtf(dot_f32(&scratch.moe_xn, &scratch.moe_xn));
    let denom = if norm_x > cfg.eps { norm_x } else { cfg.eps };
    let inv_norm = 1.0 / denom;
    for i in 0..d_c {
        scratch.moe_h_hat[i] = scratch.moe_xn[i] * inv_norm;
    }

    let k_plug = cfg.plugin_top_k.max(1);
    let mut n_active = 0usize;
    scratch.plug_sel_g[..k_plug].fill(0.0);

    for (p_idx, pack) in plugins.iter().enumerate() {
        let ep = pack.experts_per_layer;
        let base_e = layer_idx * ep;
        for e in 0..ep {
            let pk_off = (base_e + e) * d_c;
            let cos_sim = dot_f32(&scratch.moe_h_hat, &pack.proto_keys[pk_off..pk_off + d_c]);
            let d_val = cos_sim - pack.tau[base_e + e];
            if d_val > 0.0 {
                let g_val = tanhf(pack.alpha[base_e + e] * d_val);
                if g_val > 0.0 {
                    if n_active < k_plug {
                        let mut pos = n_active;
                        while pos > 0 && g_val > scratch.plug_sel_g[pos - 1] {
                            scratch.plug_sel_g[pos] = scratch.plug_sel_g[pos - 1];
                            scratch.plug_sel_pack[pos] = scratch.plug_sel_pack[pos - 1];
                            scratch.plug_sel_exp[pos] = scratch.plug_sel_exp[pos - 1];
                            pos -= 1;
                        }
                        scratch.plug_sel_g[pos] = g_val;
                        scratch.plug_sel_pack[pos] = p_idx;
                        scratch.plug_sel_exp[pos] = e;
                        n_active += 1;
                    } else if g_val > scratch.plug_sel_g[k_plug - 1] {
                        let mut pos = k_plug - 1;
                        while pos > 0 && g_val > scratch.plug_sel_g[pos - 1] {
                            scratch.plug_sel_g[pos] = scratch.plug_sel_g[pos - 1];
                            scratch.plug_sel_pack[pos] = scratch.plug_sel_pack[pos - 1];
                            scratch.plug_sel_exp[pos] = scratch.plug_sel_exp[pos - 1];
                            pos -= 1;
                        }
                        scratch.plug_sel_g[pos] = g_val;
                        scratch.plug_sel_pack[pos] = p_idx;
                        scratch.plug_sel_exp[pos] = e;
                    }
                }
            }
        }
    }

    // Hard dead-zone early return: strictly 0.0 drift & 0 plugin matmuls!
    if n_active == 0 {
        return;
    }

    scratch.moe_y_plug.fill(0.0);
    for idx in 0..n_active {
        let g_val = scratch.plug_sel_g[idx];
        let pack = &plugins[scratch.plug_sel_pack[idx]];
        let e = scratch.plug_sel_exp[idx];
        let hp = pack.hidden_dim;
        let flat_e = layer_idx * pack.experts_per_layer + e;
        let gu_stride = (2 * hp) * d_c;
        let d_stride = d_c * hp;
        let w_gu = &pack.w_gu[flat_e * gu_stride..(flat_e + 1) * gu_stride];
        let w_d = &pack.w_d[flat_e * d_stride..(flat_e + 1) * d_stride];
        linear_f32(&scratch.moe_xn, w_gu, None, &mut scratch.plug_gu[..2 * hp]);
        for j in 0..hp {
            scratch.plug_h[j] = silu(scratch.plug_gu[j]) * scratch.plug_gu[hp + j];
        }
        linear_f32(&scratch.plug_h[..hp], w_d, None, &mut scratch.moe_y_tmp);
        axpy_f32(g_val, &scratch.moe_y_tmp, &mut scratch.moe_y_plug);
    }
    for i in 0..d_c {
        scratch.cort_x[i] += scratch.moe_y_plug[i];
    }
}

/// L1 `PreG1CorticalLayer` causal local window attention (`0 <= Δs < W`) over `f_stem`.
fn run_pre_g1_window_attn(
    cfg: &DcpaFlatConfig,
    win: &crate::flat::weights::PreG1WindowWeights,
    rings: &FlatRingState,
    cur_pos: i32,
    cur_end_depth: i32,
    scratch: &mut FlatScratchPool,
) {
    let d_c = cfg.d_concept;
    let d_s = cfg.d_s;
    let w = cfg.window_size;
    let n_heads = cfg.n_heads;
    let d_head = cfg.d_head();
    let scale = 1.0 / sqrtf(d_head as f32);

    rmsnorm_f32(&scratch.cort_x, &win.norm_attn, cfg.eps, &mut scratch.cort_xn);
    linear_f32(&scratch.cort_xn, &win.w_q, None, &mut scratch.cort_q_raw);
    apply_concept_rope_2d_all_heads(
        &scratch.cort_q_raw,
        n_heads,
        d_head,
        cur_pos as f32,
        cur_end_depth as f32,
        cfg.rope_theta_concept,
        cfg.rope_period,
        &mut scratch.cort_q_rot,
    );

    // Project & rotate only valid ring entries (`0 <= cur_pos - ring_pos < W`)
    for r in 0..w {
        let diff = cur_pos - rings.ring_pos[r];
        if rings.ring_valid[r] && diff >= 0 && diff < (w as i32) {
            let stem_r = &rings.stem_ring[r * d_s..(r + 1) * d_s];
            linear_f32(stem_r, &win.w_uk, None, &mut scratch.cort_wo);
            apply_concept_rope_2d_all_heads(
                &scratch.cort_wo,
                n_heads,
                d_head,
                rings.ring_pos[r] as f32,
                rings.ring_depth[r] as f32,
                cfg.rope_theta_concept,
                cfg.rope_period,
                &mut scratch.cort_k_ring[r * d_c..(r + 1) * d_c],
            );
            linear_f32(
                stem_r,
                &win.w_uv,
                None,
                &mut scratch.cort_v_ring[r * d_c..(r + 1) * d_c],
            );
        }
    }

    scratch.cort_ctx.fill(0.0);
    for h in 0..n_heads {
        let q_h = &scratch.cort_q_rot[h * d_head..(h + 1) * d_head];
        for r in 0..w {
            let diff = cur_pos - rings.ring_pos[r];
            if rings.ring_valid[r] && diff >= 0 && diff < (w as i32) {
                let k_rh = &scratch.cort_k_ring[r * d_c + h * d_head..r * d_c + (h + 1) * d_head];
                scratch.cort_attn_scores[r] = dot_f32(q_h, k_rh) * scale;
            } else {
                scratch.cort_attn_scores[r] = f32::NEG_INFINITY;
            }
        }
        softmax_slice_exact(&mut scratch.cort_attn_scores[..w]);
        let ctx_h = &mut scratch.cort_ctx[h * d_head..(h + 1) * d_head];
        for r in 0..w {
            let a = scratch.cort_attn_scores[r];
            if a > 0.0 {
                let v_rh = &scratch.cort_v_ring[r * d_c + h * d_head..r * d_c + (h + 1) * d_head];
                axpy_f32(a, v_rh, ctx_h);
            }
        }
    }

    linear_f32(&scratch.cort_ctx, &win.w_o, None, &mut scratch.cort_wo);
    for i in 0..d_c {
        scratch.cort_x[i] += scratch.cort_wo[i];
    }
}

/// Compute soft IoR penalty `[H * K]` into `scratch.cort_ior_pen` matching `_compute_ior_penalty` 1-to-1.
fn compute_ior_penalty_into(
    cfg: &DcpaFlatConfig,
    w_ior_q: &[f32],
    ior_strength: f32,
    x3_prev: &[f32],
    scratch: &mut FlatScratchPool,
) {
    let n_heads = cfg.n_heads;
    let d_head = cfg.d_head();
    let k_slots = cfg.k_slots;
    let strength = softplus(ior_strength);

    linear_f32(x3_prev, w_ior_q, None, &mut scratch.cort_ior_q);
    for h in 0..n_heads {
        let q_h = &scratch.cort_ior_q[h * d_head..(h + 1) * d_head];
        let q_inv = 1.0 / sqrtf(dot_f32(q_h, q_h) + 1e-8);
        for k in 0..k_slots {
            let k_off = (h * k_slots + k) * d_head;
            let k_hk = &scratch.gkv_k[k_off..k_off + d_head];
            let k_norm = sqrtf(dot_f32(k_hk, k_hk));
            let k_inv = 1.0 / if k_norm > 1e-6 { k_norm } else { 1e-6 };
            let sim_prev = dot_f32(q_h, k_hk) * q_inv * k_inv;
            scratch.cort_ior_pen[h * k_slots + k] =
                if sim_prev > 0.0 { -strength * sim_prev } else { 0.0 };
        }
    }
}

/// Pure `K`-slot graph cross-attention cortical layer (`L2-L3`, `L5`, `L7-L8`, or `L4/L6` without window).
#[allow(clippy::too_many_arguments)]
fn run_graph_cross_attn_layer(
    cfg: &DcpaFlatConfig,
    layer_idx: usize,
    norm_attn: &[f32],
    w_q: &[f32],
    w_reg_bias: &[f32],
    w_o: &[f32],
    ior: Option<&crate::flat::weights::IorWeights>,
    moe: &FlatMoEWeights,
    plugins: &[PluginExpertPackRs],
    x3_prev: Option<&[f32]>,
    scratch: &mut FlatScratchPool,
) {
    let d_c = cfg.d_concept;
    let n_heads = cfg.n_heads;
    let d_head = cfg.d_head();
    let k_slots = cfg.k_slots;
    let scale = 1.0 / sqrtf(d_head as f32);

    rmsnorm_f32(&scratch.cort_x, norm_attn, cfg.eps, &mut scratch.cort_xn);
    linear_f32(&scratch.cort_xn, w_q, None, &mut scratch.cort_q_raw);
    linear_f32(&scratch.l_reg, w_reg_bias, None, &mut scratch.cort_reg_bias);
    for v in scratch.cort_reg_bias.iter_mut() {
        *v = 2.0 * tanhf(*v);
    }

    let has_ior = match (ior, x3_prev) {
        (Some(iw), Some(xp)) => {
            compute_ior_penalty_into(cfg, &iw.w_ior_q, iw.ior_strength, xp, scratch);
            true
        }
        _ => false,
    };

    scratch.cort_ctx.fill(0.0);
    for h in 0..n_heads {
        let q_h = &scratch.cort_q_raw[h * d_head..(h + 1) * d_head];
        for k in 0..k_slots {
            let k_off = (h * k_slots + k) * d_head;
            let k_hk = &scratch.gkv_k[k_off..k_off + d_head];
            let hk = h * k_slots + k;
            let mut sc = dot_f32(q_h, k_hk) * scale
                + scratch.gkv_occ_log[k]
                + scratch.gkv_rel_bias[hk]
                + scratch.cort_reg_bias[hk];
            if has_ior {
                sc += scratch.cort_ior_pen[hk];
            }
            scratch.cort_attn_scores[k] = sc;
        }
        softmax_slice_exact(&mut scratch.cort_attn_scores[..k_slots]);
        let ctx_h = &mut scratch.cort_ctx[h * d_head..(h + 1) * d_head];
        for k in 0..k_slots {
            let a = scratch.cort_attn_scores[k];
            let v_off = (h * k_slots + k) * d_head;
            axpy_f32(a, &scratch.gkv_v[v_off..v_off + d_head], ctx_h);
        }
    }

    linear_f32(&scratch.cort_ctx, w_o, None, &mut scratch.cort_wo);
    for i in 0..d_c {
        scratch.cort_x[i] += scratch.cort_wo[i];
    }

    run_decoupled_moe(cfg, layer_idx, moe, plugins, scratch);
}

/// L4 / L6 Unified `(strict-past W ∪ K=8 G2_prior ∪ 1 sink)` single-softmax layer.
#[allow(clippy::too_many_arguments)]
fn run_unified_window_graph_layer(
    cfg: &DcpaFlatConfig,
    layer_idx: usize,
    norm_attn: &[f32],
    w_q: &[f32],
    w_uk: &[f32],
    w_uv: &[f32],
    w_reg_bias: &[f32],
    w_ior_q: &[f32],
    ior_strength: f32,
    b_sink: &[f32],
    w_o: &[f32],
    moe: &FlatMoEWeights,
    plugins: &[PluginExpertPackRs],
    x3_prev: &[f32],
    rings: &FlatRingState,
    cur_pos: i32,
    cur_start_depth: i32,
    scratch: &mut FlatScratchPool,
) {
    let d_c = cfg.d_concept;
    let w = cfg.window_size;
    let n_heads = cfg.n_heads;
    let d_head = cfg.d_head();
    let k_slots = cfg.k_slots;
    let scale = 1.0 / sqrtf(d_head as f32);

    rmsnorm_f32(&scratch.cort_x, norm_attn, cfg.eps, &mut scratch.cort_xn);
    linear_f32(&scratch.cort_xn, w_q, None, &mut scratch.cort_q_raw);
    apply_concept_rope_2d_all_heads(
        &scratch.cort_q_raw,
        n_heads,
        d_head,
        cur_pos as f32,
        cur_start_depth as f32,
        cfg.rope_theta_concept,
        cfg.rope_period,
        &mut scratch.cort_q_rot,
    );

    let (ring_buf, ring_dim) = match &rings.sem_ring {
        Some(sem) => (sem.as_slice(), rings.d_sem_ring),
        None => (rings.stem_ring.as_slice(), rings.d_s),
    };

    for r in 0..w {
        let diff = cur_pos - rings.ring_pos[r];
        if rings.ring_valid[r] && diff >= 1 && diff <= (w as i32) {
            let rv = &ring_buf[r * ring_dim..(r + 1) * ring_dim];
            linear_f32(rv, w_uk, None, &mut scratch.cort_wo);
            apply_concept_rope_2d_all_heads(
                &scratch.cort_wo,
                n_heads,
                d_head,
                rings.ring_pos[r] as f32,
                rings.ring_depth[r] as f32,
                cfg.rope_theta_concept,
                cfg.rope_period,
                &mut scratch.cort_k_ring[r * d_c..(r + 1) * d_c],
            );
            linear_f32(rv, w_uv, None, &mut scratch.cort_v_ring[r * d_c..(r + 1) * d_c]);
        }
    }

    linear_f32(&scratch.l_reg, w_reg_bias, None, &mut scratch.cort_reg_bias);
    for v in scratch.cort_reg_bias.iter_mut() {
        *v = 2.0 * tanhf(*v);
    }
    compute_ior_penalty_into(cfg, w_ior_q, ior_strength, x3_prev, scratch);

    let total_cols = w + k_slots + 1;
    scratch.cort_ctx.fill(0.0);
    for h in 0..n_heads {
        let q_rot_h = &scratch.cort_q_rot[h * d_head..(h + 1) * d_head];
        let q_raw_h = &scratch.cort_q_raw[h * d_head..(h + 1) * d_head];
        // Columns 0..W: strict-past local window
        for r in 0..w {
            let diff = cur_pos - rings.ring_pos[r];
            if rings.ring_valid[r] && diff >= 1 && diff <= (w as i32) {
                let k_rh = &scratch.cort_k_ring[r * d_c + h * d_head..r * d_c + (h + 1) * d_head];
                scratch.cort_attn_scores[r] = dot_f32(q_rot_h, k_rh) * scale;
            } else {
                scratch.cort_attn_scores[r] = f32::NEG_INFINITY;
            }
        }
        // Columns W..W+K: G2_prior shared graph KV (uses UNROTATED q_raw!)
        for k in 0..k_slots {
            let k_off = (h * k_slots + k) * d_head;
            let k_hk = &scratch.gkv_k[k_off..k_off + d_head];
            let hk = h * k_slots + k;
            scratch.cort_attn_scores[w + k] = dot_f32(q_raw_h, k_hk) * scale
                + scratch.gkv_occ_log[k]
                + scratch.gkv_rel_bias[hk]
                + scratch.cort_reg_bias[hk]
                + scratch.cort_ior_pen[hk];
        }
        // Column W+K: always-open learnable sink (zero value vector)
        scratch.cort_attn_scores[w + k_slots] = b_sink[h];

        softmax_slice_exact(&mut scratch.cort_attn_scores[..total_cols]);

        let ctx_h = &mut scratch.cort_ctx[h * d_head..(h + 1) * d_head];
        for r in 0..w {
            let a = scratch.cort_attn_scores[r];
            if a > 0.0 {
                let v_rh = &scratch.cort_v_ring[r * d_c + h * d_head..r * d_c + (h + 1) * d_head];
                axpy_f32(a, v_rh, ctx_h);
            }
        }
        for k in 0..k_slots {
            let a = scratch.cort_attn_scores[w + k];
            let v_off = (h * k_slots + k) * d_head;
            axpy_f32(a, &scratch.gkv_v[v_off..v_off + d_head], ctx_h);
        }
    }

    linear_f32(&scratch.cort_ctx, w_o, None, &mut scratch.cort_wo);
    for i in 0..d_c {
        scratch.cort_x[i] += scratch.cort_wo[i];
    }

    run_decoupled_moe(cfg, layer_idx, moe, plugins, scratch);
}

/// Pure `f_stem` Holographic CopyRing single-step streaming readout (0 allocs!).
fn run_copy_ring_step(
    cfg: &DcpaFlatConfig,
    weights: &DcpaFlatWeights,
    rings: &FlatRingState,
    cur_pos: i32,
    cur_start_depth: i32,
    scratch: &mut FlatScratchPool,
) {
    let w = cfg.window_size;
    let d_s = cfg.d_s;
    let n_heads = cfg.n_heads;
    let d_k = cfg.copy_ring_dk();
    let proj_dim = n_heads * d_k;
    let scale = 1.0 / sqrtf(d_k as f32);

    rmsnorm_f32(
        &scratch.cort_z_hat,
        &weights.copy_norm_q,
        cfg.eps,
        &mut scratch.copy_qn,
    );
    linear_f32(&scratch.copy_qn, &weights.copy_w_q, None, &mut scratch.copy_q_raw);
    apply_concept_rope_2d_all_heads(
        &scratch.copy_q_raw,
        n_heads,
        d_k,
        cur_pos as f32,
        cur_start_depth as f32,
        cfg.rope_theta_concept,
        cfg.rope_period,
        &mut scratch.copy_q_rot,
    );

    for r in 0..w {
        let diff = cur_pos - rings.ring_pos[r];
        if rings.ring_valid[r] && diff >= 1 && diff <= (w as i32) {
            let stem_r = &rings.stem_ring[r * d_s..(r + 1) * d_s];
            linear_f32(stem_r, &weights.copy_w_k, None, &mut scratch.copy_k_tmp);
            apply_concept_rope_2d_all_heads(
                &scratch.copy_k_tmp,
                n_heads,
                d_k,
                rings.ring_pos[r] as f32,
                rings.ring_depth[r] as f32,
                cfg.rope_theta_concept,
                cfg.rope_period,
                &mut scratch.copy_k_rot[r * proj_dim..(r + 1) * proj_dim],
            );
        }
    }

    scratch.copy_w.fill(0.0);
    let inv_h = 1.0 / (n_heads as f32);
    for h in 0..n_heads {
        let q_h = &scratch.copy_q_rot[h * d_k..(h + 1) * d_k];
        for r in 0..w {
            let diff = cur_pos - rings.ring_pos[r];
            if rings.ring_valid[r] && diff >= 1 && diff <= (w as i32) {
                let k_rh = &scratch.copy_k_rot[r * proj_dim + h * d_k..r * proj_dim + (h + 1) * d_k];
                scratch.copy_scores[r] = dot_f32(q_h, k_rh) * scale;
            } else {
                scratch.copy_scores[r] = f32::NEG_INFINITY;
            }
        }
        scratch.copy_scores[w] = weights.copy_b_sink[h];
        softmax_slice_exact(&mut scratch.copy_scores[..w + 1]);
        for r in 0..w {
            scratch.copy_w[r] += scratch.copy_scores[r] * inv_h;
        }
    }

    scratch.f_copy.fill(0.0);
    for r in 0..w {
        let cw = scratch.copy_w[r];
        if cw > 0.0 {
            let stem_r = &rings.stem_ring[r * d_s..(r + 1) * d_s];
            axpy_f32(cw, stem_r, &mut scratch.f_copy);
        }
    }
}

#[derive(Clone, Copy)]
enum CarrierInput {
    CortX1,
    CortX3,
    HipCFused,
}

/// True-contractive closed-unit-ball graph carrier single step (`G1` or `G2`),
/// matching `dcpa/modules/graph_carrier.py::ContractiveGraphCarrier.step_single` 1-to-1!
///
/// Updates `state` in-place to `next_state`, saves post-reset pre-update `prior_state` into
/// `scratch.car_prior`, and writes Write-Before-Merge intensity into `scratch.car_evict_w`.
#[allow(clippy::too_many_arguments)]
fn step_carrier_single(
    cfg: &DcpaFlatConfig,
    w: &FlatCarrierWeights,
    pin_enabled: bool,
    state: &mut CarrierGraphStateRs,
    x_in_src: CarrierInput,
    seg_reset: bool,
    pin_trig: bool,
    unpin_trig: bool,
    scratch: &mut FlatScratchPool,
) {
    let k_slots = cfg.k_slots;
    let d_c = cfg.d_concept;
    let d_r = cfg.d_r;
    let d_m = cfg.d_merge;
    let write_occ = cfg.occ_min + 0.80 * (cfg.occ_max - cfg.occ_min);

    // 0. Pre-update document reset + optional unpin (G2 only) -> snapshot `car_prior`
    if seg_reset {
        scratch.car_prior.reset_from_carrier(cfg, w);
    } else {
        scratch.car_prior.copy_from(state);
        if pin_enabled {
            if unpin_trig {
                scratch.car_prior.slot0_pinned = false;
            }
        } else {
            scratch.car_prior.slot0_pinned = false;
        }
    }
    let pinned_pre = scratch.car_prior.slot0_pinned;

    // 1. Input side-path projection onto closed unit ball B_D
    let x_in: &[f32] = match x_in_src {
        CarrierInput::CortX1 => &scratch.cort_x1,
        CarrierInput::CortX3 => &scratch.cort_x3,
        CarrierInput::HipCFused => &scratch.hip_c_fused,
    };
    rmsnorm_f32(x_in, &w.norm_in, cfg.eps, &mut scratch.car_xn);
    linear_f32(&scratch.car_xn, &w.w_in, None, &mut scratch.car_vin);
    let v_scale = 0.96 / sqrtf(dot_f32(&scratch.car_vin, &scratch.car_vin) + 0.04);
    for c in 0..d_c {
        scratch.car_uin[c] = scratch.car_vin[c] * v_scale;
    }

    linear_f32(&scratch.car_xn, &w.w_edge_in_src, None, &mut scratch.car_ein_src);
    linear_f32(&scratch.car_xn, &w.w_edge_in_dst, None, &mut scratch.car_ein_dst);
    linear_f32(
        &scratch.car_xn,
        &w.w_lam_w,
        Some(&w.w_lam_b),
        &mut scratch.car_lam_logits,
    );

    for k in 0..k_slots {
        let s_k = &scratch.car_prior.slots[k * d_c..(k + 1) * d_c];
        linear_f32(
            s_k,
            &w.w_edge_slot_src,
            None,
            &mut scratch.car_eslot_src[k * d_r..(k + 1) * d_r],
        );
        linear_f32(
            s_k,
            &w.w_edge_slot_dst,
            None,
            &mut scratch.car_eslot_dst[k * d_r..(k + 1) * d_r],
        );
        linear_f32(
            s_k,
            &w.w_pair_si,
            None,
            &mut scratch.car_h_si[k * d_m..(k + 1) * d_m],
        );
        linear_f32(
            s_k,
            &w.w_pair_sj,
            None,
            &mut scratch.car_h_sj[k * d_m..(k + 1) * d_m],
        );
        for r in 0..d_r {
            scratch.car_eout_cand[k * d_r + r] =
                tanhf(scratch.car_ein_src[r] * scratch.car_eslot_dst[k * d_r + r]);
            scratch.car_ein_cand[k * d_r + r] =
                tanhf(scratch.car_eslot_src[k * d_r + r] * scratch.car_ein_dst[r]);
        }
    }

    // Explicit edge decay & occupancy relaxation
    for i in 0..(k_slots * k_slots * d_r) {
        scratch.car_edges_dec[i] = scratch.car_prior.edges[i] * cfg.edge_decay;
    }
    for k in 0..k_slots {
        let rt = cfg.occ_min + cfg.occ_relax * (scratch.car_prior.occ[k] - cfg.occ_min);
        scratch.car_rho_tilde[k] = rt.clamp(cfg.occ_min, cfg.occ_max);
    }
    let eta_e = 0.25 * sigmoid(w.edge_write_logit);

    // Free-slot mask
    let mut free_mask = [true; 16];
    if pin_enabled && pinned_pre {
        free_mask[0] = false;
    }

    // ------------------------------------------------------------------
    // Stage 1: O(K) Coreference Assimilation
    // ------------------------------------------------------------------
    let occ_span = (cfg.occ_max - cfg.occ_min).max(1e-6);
    let mut k_star = 0usize;
    let mut best_assim_logit = f32::NEG_INFINITY;
    for k in 0..k_slots {
        let s_k = &scratch.car_prior.slots[k * d_c..(k + 1) * d_c];
        let sim = dot_f32(&scratch.car_uin, s_k);
        let occ_frac = ((scratch.car_rho_tilde[k] - cfg.occ_min) / occ_span).clamp(0.0, 1.0);
        let occ_act = (occ_frac * 4.0).clamp(0.0, 1.0);
        let asc = sim * occ_act;
        scratch.car_sim[k] = sim;
        scratch.car_occ_act[k] = occ_act;
        scratch.car_assim_sc[k] = asc;
        let logit = if free_mask[k] { asc * 2.0 } else { -1.0e9 };
        if logit > best_assim_logit {
            best_assim_logit = logit;
            k_star = k;
        }
    }
    let g_assim = scratch.car_occ_act[k_star]
        * sigmoid(w.assim_scale.clamp(0.5, 6.0) * (scratch.car_assim_sc[k_star] - w.assim_threshold));
    let lam_kstar = cfg.slot_keep_max
        * sigmoid(scratch.car_lam_logits[k_star] + w.lam_sim_gain.clamp(-2.0, 2.0) * scratch.car_sim[k_star]);

    scratch.car_slots_s1.copy_from_slice(&scratch.car_prior.slots);
    for c in 0..d_c {
        let idx = k_star * d_c + c;
        scratch.car_slots_s1[idx] =
            lam_kstar * scratch.car_prior.slots[idx] + (1.0 - lam_kstar) * scratch.car_uin[c];
    }
    scratch.car_occ_s1.copy_from_slice(&scratch.car_rho_tilde);
    scratch.car_occ_s1[k_star] = (scratch.car_rho_tilde[k_star]
        + 0.35 * (cfg.occ_max - scratch.car_rho_tilde[k_star]))
    .clamp(cfg.occ_min, cfg.occ_max);

    scratch.car_edges_s1.copy_from_slice(&scratch.car_edges_dec);
    for j in 0..k_slots {
        let off = (k_star * k_slots + j) * d_r;
        let cand = &scratch.car_eout_cand[j * d_r..(j + 1) * d_r];
        for r in 0..d_r {
            scratch.car_edges_s1[off + r] =
                (1.0 - eta_e) * scratch.car_edges_dec[off + r] + eta_e * cand[r];
        }
    }
    for i in 0..k_slots {
        let off = (i * k_slots + k_star) * d_r;
        let cand = &scratch.car_ein_cand[i * d_r..(i + 1) * d_r];
        for r in 0..d_r {
            scratch.car_edges_s1[off + r] =
                (1.0 - eta_e) * scratch.car_edges_s1[off + r] + eta_e * cand[r];
        }
    }
    for k in 0..k_slots {
        let diag = (k * k_slots + k) * d_r;
        scratch.car_edges_s1[diag..diag + d_r].fill(0.0);
    }

    // ------------------------------------------------------------------
    // Stage 2A: Lossless empty-slot direct write into coldest free slot k_cold
    // ------------------------------------------------------------------
    let mut k_cold = 0usize;
    let mut min_free_rho = f32::INFINITY;
    for k in 0..k_slots {
        let rv = if free_mask[k] { scratch.car_rho_tilde[k] } else { 2.0 };
        if rv < min_free_rho {
            min_free_rho = rv;
            k_cold = k;
        }
    }
    let sigma_full = sigmoid((min_free_rho - cfg.full_gate_center) / cfg.full_gate_temp);

    scratch.car_slots_emp.copy_from_slice(&scratch.car_prior.slots);
    scratch.car_slots_emp[k_cold * d_c..(k_cold + 1) * d_c].copy_from_slice(&scratch.car_uin);
    scratch.car_occ_emp.copy_from_slice(&scratch.car_rho_tilde);
    scratch.car_occ_emp[k_cold] = write_occ;

    scratch.car_edges_emp.copy_from_slice(&scratch.car_edges_dec);
    for j in 0..k_slots {
        let off = (k_cold * k_slots + j) * d_r;
        scratch.car_edges_emp[off..off + d_r]
            .copy_from_slice(&scratch.car_eout_cand[j * d_r..(j + 1) * d_r]);
    }
    for i in 0..k_slots {
        let off = (i * k_slots + k_cold) * d_r;
        scratch.car_edges_emp[off..off + d_r]
            .copy_from_slice(&scratch.car_ein_cand[i * d_r..(i + 1) * d_r]);
    }
    for k in 0..k_slots {
        let diag = (k * k_slots + k) * d_r;
        scratch.car_edges_emp[diag..diag + d_r].fill(0.0);
    }

    // ------------------------------------------------------------------
    // Stage 2B: Full-load K*(K-1) Probabilistic-OR Merge
    // ------------------------------------------------------------------
    let cos_gain = w.pair_cos_gain.clamp(-2.0, 2.0);
    let mut win_i = 0usize;
    let mut win_j = 1usize;
    let mut best_pair_logit = f32::NEG_INFINITY;

    for i in 0..k_slots {
        for j in 0..k_slots {
            if i == j {
                continue;
            }
            if pin_enabled && pinned_pre && (i == 0 || j == 0) {
                let logit = -1.0e9f32;
                if logit > best_pair_logit {
                    best_pair_logit = logit;
                    win_i = i;
                    win_j = j;
                }
                continue;
            }
            let e_off = (i * k_slots + j) * d_r;
            linear_f32(
                &scratch.car_edges_dec[e_off..e_off + d_r],
                &w.w_pair_edge,
                None,
                &mut scratch.car_h_edge,
            );
            let h_si_i = &scratch.car_h_si[i * d_m..(i + 1) * d_m];
            let h_sj_j = &scratch.car_h_sj[j * d_m..(j + 1) * d_m];
            let rho_i = scratch.car_rho_tilde[i];
            let rho_j = scratch.car_rho_tilde[j];
            for m in 0..d_m {
                let pre = h_si_i[m]
                    + h_sj_j[m]
                    + scratch.car_h_edge[m]
                    + rho_i * w.w_pair_rho_i[m]
                    + rho_j * w.w_pair_rho_j[m]
                    + w.pair_bias[m];
                scratch.car_h_pair[m] = silu(pre);
            }
            let pair_cos = dot_f32(
                &scratch.car_prior.slots[i * d_c..(i + 1) * d_c],
                &scratch.car_prior.slots[j * d_c..(j + 1) * d_c],
            );
            let raw_logit =
                dot_f32(&scratch.car_h_pair, &w.w_pair_score_w) + w.w_pair_score_b + cos_gain * pair_cos;
            let logit = 4.0 * tanhf(raw_logit * 0.25);
            if logit > best_pair_logit {
                best_pair_logit = logit;
                win_i = i;
                win_j = j;
                scratch.car_h_star.copy_from_slice(&scratch.car_h_pair);
            }
        }
    }

    let u_star = win_i.min(win_j);
    let v_star = win_i.max(win_j);

    let beta_logit = dot_f32(&scratch.car_h_star, &w.w_merge_beta_w) + w.w_merge_beta_b;
    let beta_m = 0.55 + 0.30 * sigmoid(beta_logit);
    let sb = sqrtf(beta_m);
    let s1mb = sqrtf(1.0 - beta_m);
    linear_f32(&scratch.car_h_star, &w.w_merge_delta, None, &mut scratch.car_delta_m);
    for c in 0..d_c {
        let si = scratch.car_prior.slots[win_i * d_c + c];
        let sj = scratch.car_prior.slots[win_j * d_c + c];
        let dm = 0.12 * tanhf(scratch.car_delta_m[c]);
        scratch.car_s_merged[c] = cfg.slot_keep_max * (sb * si + s1mb * sj) + dm;
    }
    let m_norm = sqrtf(dot_f32(&scratch.car_s_merged, &scratch.car_s_merged) + cfg.eps);
    let m_scale = 0.995 / if m_norm > 0.995 { m_norm } else { 0.995 };
    for c in 0..d_c {
        scratch.car_s_merged[c] *= m_scale;
    }

    let ri = scratch.car_rho_tilde[win_i];
    let rj = scratch.car_rho_tilde[win_j];
    let rho_or = (ri + rj - ri * rj).clamp(cfg.occ_min, 0.90);

    scratch.car_slots_mrg.copy_from_slice(&scratch.car_prior.slots);
    scratch.car_slots_mrg[u_star * d_c..(u_star + 1) * d_c].copy_from_slice(&scratch.car_s_merged);
    scratch.car_slots_mrg[v_star * d_c..(v_star + 1) * d_c].copy_from_slice(&scratch.car_uin);
    scratch.car_occ_mrg.copy_from_slice(&scratch.car_rho_tilde);
    scratch.car_occ_mrg[u_star] = rho_or;
    scratch.car_occ_mrg[v_star] = write_occ;

    // Edge merge: row step then col step then zero diagonal
    scratch.car_edges_row.copy_from_slice(&scratch.car_edges_dec);
    for j in 0..k_slots {
        let off_u = (u_star * k_slots + j) * d_r;
        let off_v = (v_star * k_slots + j) * d_r;
        let off_wi = (win_i * k_slots + j) * d_r;
        let off_wj = (win_j * k_slots + j) * d_r;
        let cand_out = &scratch.car_eout_cand[j * d_r..(j + 1) * d_r];
        for r in 0..d_r {
            scratch.car_edges_row[off_u + r] =
                0.5 * (scratch.car_edges_dec[off_wi + r] + scratch.car_edges_dec[off_wj + r]);
            scratch.car_edges_row[off_v + r] = cand_out[r];
        }
    }
    scratch.car_edges_mrg.copy_from_slice(&scratch.car_edges_row);
    for i in 0..k_slots {
        let off_u = (i * k_slots + u_star) * d_r;
        let off_v = (i * k_slots + v_star) * d_r;
        let off_wi = (i * k_slots + win_i) * d_r;
        let off_wj = (i * k_slots + win_j) * d_r;
        let cand_in = &scratch.car_ein_cand[i * d_r..(i + 1) * d_r];
        for r in 0..d_r {
            scratch.car_edges_mrg[off_u + r] =
                0.5 * (scratch.car_edges_row[off_wi + r] + scratch.car_edges_row[off_wj + r]);
            scratch.car_edges_mrg[off_v + r] = cand_in[r];
        }
    }
    for k in 0..k_slots {
        let diag = (k * k_slots + k) * d_r;
        scratch.car_edges_mrg[diag..diag + d_r].fill(0.0);
    }

    // Write-Before-Merge eviction weight & exact dominant slot attribution (G2 -> Hippocampus)
    let ev_int = sigma_full * (1.0 - g_assim);
    scratch.car_evict_w.fill(0.0);
    scratch.car_evict_w[u_star] = ev_int;
    scratch.car_evict_w[v_star] = ev_int;
    let is_assim = g_assim >= 0.5;
    let is_merge = !is_assim && (sigma_full >= 0.5);
    scratch.car_slot_star = if is_assim {
        k_star
    } else if is_merge {
        v_star
    } else {
        k_cold
    };
    scratch.car_win_i = u_star;
    scratch.car_win_j = v_star;
    scratch.car_merge_active = is_merge;

    // ------------------------------------------------------------------
    // Smooth interpolation: Stage 1 (assim) vs Stage 2 (empty vs merge)
    // ------------------------------------------------------------------
    let one_m_sf = 1.0 - sigma_full;
    let one_m_ga = 1.0 - g_assim;
    for i in 0..(k_slots * d_c) {
        let s2 = one_m_sf * scratch.car_slots_emp[i] + sigma_full * scratch.car_slots_mrg[i];
        state.slots[i] = g_assim * scratch.car_slots_s1[i] + one_m_ga * s2;
    }
    for k in 0..k_slots {
        let o2 = one_m_sf * scratch.car_occ_emp[k] + sigma_full * scratch.car_occ_mrg[k];
        state.occ[k] = (g_assim * scratch.car_occ_s1[k] + one_m_ga * o2).clamp(cfg.occ_min, cfg.occ_max);
    }
    for i in 0..(k_slots * k_slots * d_r) {
        let e2 = one_m_sf * scratch.car_edges_emp[i] + sigma_full * scratch.car_edges_mrg[i];
        state.edges[i] = (g_assim * scratch.car_edges_s1[i] + one_m_ga * e2).clamp(-0.9999, 0.9999);
    }

    // Bit-exact Slot-0 preservation while pinned, or post-update pinning on `\x1dE\n` (G2 only)
    if pin_enabled {
        if pinned_pre {
            state.slots[..d_c].copy_from_slice(&scratch.car_prior.slots[..d_c]);
            state.occ[0] = scratch.car_prior.occ[0];
            state.slot0_pinned = true;
        } else if pin_trig {
            let mut occ_sum = 0.0f32;
            for k in 0..k_slots {
                occ_sum += state.occ[k];
            }
            let inv_sum = 1.0 / if occ_sum > cfg.eps { occ_sum } else { cfg.eps };
            scratch.car_g_sum.fill(0.0);
            for k in 0..k_slots {
                axpy_f32(
                    state.occ[k] * inv_sum,
                    &state.slots[k * d_c..(k + 1) * d_c],
                    &mut scratch.car_g_sum,
                );
            }
            let g_scale = 0.92 / sqrtf(dot_f32(&scratch.car_g_sum, &scratch.car_g_sum) + 0.01);
            for c in 0..d_c {
                state.slots[c] = scratch.car_g_sum[c] * g_scale;
            }
            if state.occ[0] < 0.85 {
                state.occ[0] = 0.85;
            }
            state.slot0_pinned = true;
        } else {
            state.slot0_pinned = false;
        }
    } else {
        state.slot0_pinned = false;
    }
}
