//! Pre-allocated zero-heap-allocation runtime states, `BrainSnapshotV3`, `FlatScratchPool`,
//! and `SleepReplayCaptureRing` for DCPA-Flat v3.3.
//!
//! Strictly `no_std` + `alloc` compatible (file writing gated behind `#[cfg(feature = "std")]`).

use alloc::vec;
use alloc::vec::Vec;

use crate::flat::config::DcpaFlatConfig;
use crate::flat::weights::{DcpaFlatWeights, FlatCarrierWeights};
use crate::simd::{dot_f32, sqrtf, tanhf};

/// Closed-unit-ball contractive concept graph state (`G1` or `G2`, ~16 KB in BF16).
#[derive(Clone)]
pub struct CarrierGraphStateRs {
    pub slots: Vec<f32>, // [K * D]
    pub edges: Vec<f32>, // [K * K * d_r]
    pub occ: Vec<f32>,   // [K]
    pub slot0_pinned: bool,
}

impl CarrierGraphStateRs {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        let k = cfg.k_slots;
        let d = cfg.d_concept;
        let d_r = cfg.d_r;
        Self {
            slots: vec![0.0f32; k * d],
            edges: vec![0.0f32; k * k * d_r],
            occ: vec![cfg.occ_min; k],
            slot0_pinned: false,
        }
    }

    /// Initialize from carrier learnable `G_init` tensors (exact match to `_get_init_tensors`).
    pub fn reset_from_carrier(&mut self, cfg: &DcpaFlatConfig, w: &FlatCarrierWeights) {
        let k = cfg.k_slots;
        let d = cfg.d_concept;
        let d_r = cfg.d_r;
        for slot in 0..k {
            let src = &w.init_slots_raw[slot * d..(slot + 1) * d];
            let dst = &mut self.slots[slot * d..(slot + 1) * d];
            let norm = sqrtf(dot_f32(src, src) + cfg.eps);
            let scale = 0.50 / if norm > 0.50 { norm } else { 0.50 };
            for c in 0..d {
                dst[c] = src[c] * scale;
            }
        }
        for i in 0..k {
            for j in 0..k {
                let off = (i * k + j) * d_r;
                if i == j {
                    self.edges[off..off + d_r].fill(0.0);
                } else {
                    for r in 0..d_r {
                        self.edges[off + r] = 0.5 * tanhf(w.init_edges_raw[off + r]);
                    }
                }
            }
        }
        self.occ.fill(cfg.occ_min);
        self.slot0_pinned = false;
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.slots.copy_from_slice(&src.slots);
        self.edges.copy_from_slice(&src.edges);
        self.occ.copy_from_slice(&src.occ);
        self.slot0_pinned = src.slot0_pinned;
    }
}

/// O(1) single-stream SGSC state matching `SGSCFlatStreamState` 1-to-1.
#[derive(Clone)]
pub struct FlatSgscState {
    pub h: Vec<f32>,         // [d_s]
    pub v: f32,
    pub prev_info: f32,
    pub has_prev_byte: bool,
    pub seg_len: usize,
    pub utf8_rem: i32,
    pub ast_depth: i32,
    pub seg_start_depth: i32,
    pub prev2_bytes: [i32; 2],
    pub seg_pin: bool,
    pub seg_unpin: bool,
    pub seg_h_buf: Vec<f32>, // [max_seg_len * d_s]
    pub seg_info_sum: f32,
}

impl FlatSgscState {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        Self {
            h: vec![0.0f32; cfg.d_s],
            v: 0.0,
            prev_info: 0.0,
            has_prev_byte: false,
            seg_len: 0,
            utf8_rem: 0,
            ast_depth: 0,
            seg_start_depth: 0,
            prev2_bytes: [0, 0],
            seg_pin: false,
            seg_unpin: false,
            seg_h_buf: vec![0.0f32; cfg.max_seg_len * cfg.d_s],
            seg_info_sum: 0.0,
        }
    }

    pub fn reset(&mut self, weights: &DcpaFlatWeights) {
        self.h.copy_from_slice(&weights.sgsc_h0);
        self.v = 0.0;
        self.prev_info = 0.0;
        self.has_prev_byte = false;
        self.seg_len = 0;
        self.utf8_rem = 0;
        self.ast_depth = 0;
        self.seg_start_depth = 0;
        self.prev2_bytes = [0, 0];
        self.seg_pin = false;
        self.seg_unpin = false;
        self.seg_h_buf.fill(0.0);
        self.seg_info_sum = 0.0;
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.h.copy_from_slice(&src.h);
        self.v = src.v;
        self.prev_info = src.prev_info;
        self.has_prev_byte = src.has_prev_byte;
        self.seg_len = src.seg_len;
        self.utf8_rem = src.utf8_rem;
        self.ast_depth = src.ast_depth;
        self.seg_start_depth = src.seg_start_depth;
        self.prev2_bytes = src.prev2_bytes;
        self.seg_pin = src.seg_pin;
        self.seg_unpin = src.seg_unpin;
        self.seg_h_buf.copy_from_slice(&src.seg_h_buf);
        self.seg_info_sum = src.seg_info_sum;
    }
}

/// O(1) single-stream PMDFlat motor & 128B differentiable pushdown stack state.
#[derive(Clone)]
pub struct FlatPmdState {
    pub m: Vec<f32>,             // [d_s]
    pub xi: [[f32; 4]; 16],      // 16x4 differentiable pushdown stack
    pub charge_e: f32,
}

impl FlatPmdState {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        Self {
            m: vec![0.0f32; cfg.d_s],
            xi: [[0.0f32; 4]; 16],
            charge_e: 0.0,
        }
    }

    pub fn reset(&mut self, weights: &DcpaFlatWeights) {
        self.m.copy_from_slice(&weights.pmd_m0);
        self.xi = [[0.0f32; 4]; 16];
        self.charge_e = 0.0;
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.m.copy_from_slice(&src.m);
        self.xi = src.xi;
        self.charge_e = src.charge_e;
    }
}

/// O(1) FIFO local window rings: pure 48KB `f_stem` ring + optional +32KB `c_kv_sem` ring.
#[derive(Clone)]
pub struct FlatRingState {
    pub window_size: usize,
    pub d_s: usize,
    pub d_sem_ring: usize,
    pub stem_ring: Vec<f32>,        // [W * d_s]
    pub sem_ring: Option<Vec<f32>>, // [W * d_sem_ring] when !share_stem_ring
    pub ring_pos: Vec<i32>,         // [W]
    pub ring_depth: Vec<i32>,       // [W]
    pub ring_valid: Vec<bool>,      // [W]
    pub doc_pos: i32,
}

impl FlatRingState {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        let w = cfg.window_size;
        Self {
            window_size: w,
            d_s: cfg.d_s,
            d_sem_ring: cfg.d_sem_ring,
            stem_ring: vec![0.0f32; w * cfg.d_s],
            sem_ring: if cfg.share_stem_ring {
                None
            } else {
                Some(vec![0.0f32; w * cfg.d_sem_ring])
            },
            ring_pos: vec![0i32; w],
            ring_depth: vec![0i32; w],
            ring_valid: vec![false; w],
            doc_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.stem_ring.fill(0.0);
        if let Some(ref mut sem) = self.sem_ring {
            sem.fill(0.0);
        }
        self.ring_pos.fill(0);
        self.ring_depth.fill(0);
        self.ring_valid.fill(false);
        self.doc_pos = 0;
    }

    /// Zero-alloc O(1) FIFO shift pushing a completed `f_stem` into the ring (matches `CopyRing.push_ring`).
    #[inline]
    pub fn push_stem(&mut self, f_stem: &[f32], pos: i32, end_depth: i32) {
        let w = self.window_size;
        let d_s = self.d_s;
        if w > 1 {
            self.stem_ring.copy_within(d_s..w * d_s, 0);
            self.ring_pos.copy_within(1..w, 0);
            self.ring_depth.copy_within(1..w, 0);
            self.ring_valid.copy_within(1..w, 0);
        }
        let last = (w - 1) * d_s;
        self.stem_ring[last..last + d_s].copy_from_slice(f_stem);
        self.ring_pos[w - 1] = pos;
        self.ring_depth[w - 1] = end_depth;
        self.ring_valid[w - 1] = true;
        self.doc_pos = pos + 1;
    }

    /// Zero-alloc O(1) FIFO shift pushing L3 `c_kv_sem` into the optional semantic ring.
    #[inline]
    pub fn push_sem(&mut self, c_kv_sem: &[f32]) {
        if let Some(ref mut sem) = self.sem_ring {
            let w = self.window_size;
            let d = self.d_sem_ring;
            if w > 1 {
                sem.copy_within(d..w * d, 0);
            }
            let last = (w - 1) * d;
            sem[last..last + d].copy_from_slice(c_kv_sem);
        }
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.stem_ring.copy_from_slice(&src.stem_ring);
        if let (Some(dst), Some(s)) = (&mut self.sem_ring, &src.sem_ring) {
            dst.copy_from_slice(s);
        }
        self.ring_pos.copy_from_slice(&src.ring_pos);
        self.ring_depth.copy_from_slice(&src.ring_depth);
        self.ring_valid.copy_from_slice(&src.ring_valid);
        self.doc_pos = src.doc_pos;
    }
}

/// Per-slot hippocampal synaptic eligibility traces for Write-Before-Merge (`[K, s]`, `[K, d_s]`, `[K]`).
#[derive(Clone)]
pub struct FlatEligState {
    pub k_slots: usize,
    pub s_active: usize,
    pub d_s: usize,
    pub elig_idx: Vec<usize>, // [K * s]
    pub elig_val: Vec<f32>,   // [K * s]
    pub elig_f: Vec<f32>,     // [K * d_s]
    pub elig_eta: Vec<f32>,   // [K]
}

impl FlatEligState {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        Self {
            k_slots: cfg.k_slots,
            s_active: cfg.s_active,
            d_s: cfg.d_s,
            elig_idx: vec![0usize; cfg.k_slots * cfg.s_active],
            elig_val: vec![0.0f32; cfg.k_slots * cfg.s_active],
            elig_f: vec![0.0f32; cfg.k_slots * cfg.d_s],
            elig_eta: vec![0.0f32; cfg.k_slots],
        }
    }

    pub fn reset(&mut self) {
        self.elig_idx.fill(0);
        self.elig_val.fill(0.0);
        self.elig_f.fill(0.0);
        self.elig_eta.fill(0.0);
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.elig_idx.copy_from_slice(&src.elig_idx);
        self.elig_val.copy_from_slice(&src.elig_val);
        self.elig_f.copy_from_slice(&src.elig_f);
        self.elig_eta.copy_from_slice(&src.elig_eta);
    }
}

/// WS-6 O(1) zero-heap-allocation full brain snapshot (`~114 KB` in BF16 for 206M; `~82 KB` when `share_stem_ring=true`).
#[derive(Clone)]
pub struct BrainSnapshotV3 {
    pub sgsc: FlatSgscState,
    pub g1: CarrierGraphStateRs,
    pub g2: CarrierGraphStateRs,
    pub rings: FlatRingState,
    pub prev_rot_stem: Vec<f32>, // [d_s]
    pub prev_x3: Vec<f32>,       // [D]
    pub pmd: FlatPmdState,
    pub tau_think: f32,
}

impl BrainSnapshotV3 {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        Self {
            sgsc: FlatSgscState::new(cfg),
            g1: CarrierGraphStateRs::new(cfg),
            g2: CarrierGraphStateRs::new(cfg),
            rings: FlatRingState::new(cfg),
            prev_rot_stem: vec![0.0f32; cfg.d_s],
            prev_x3: vec![0.0f32; cfg.d_concept],
            pmd: FlatPmdState::new(cfg),
            tau_think: 0.0,
        }
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.sgsc.copy_from(&src.sgsc);
        self.g1.copy_from(&src.g1);
        self.g2.copy_from(&src.g2);
        self.rings.copy_from(&src.rings);
        self.prev_rot_stem.copy_from_slice(&src.prev_rot_stem);
        self.prev_x3.copy_from_slice(&src.prev_x3);
        self.pmd.copy_from(&src.pmd);
        self.tau_think = src.tau_think;
    }

    /// Estimated compact BF16 wire/RAM footprint in bytes (matches `docs/DCPA_FLAT_V3_3_SPEC.md`).
    pub fn byte_size_estimate_bf16(cfg: &DcpaFlatConfig) -> usize {
        let sgsc_bytes = cfg.d_s * 2 + 32;
        let single_carrier =
            cfg.k_slots * cfg.d_concept * 2 + cfg.k_slots * cfg.k_slots * cfg.d_r * 2 + cfg.k_slots * 4 + 1;
        let carriers_bytes = 2 * single_carrier;
        let stem_ring_bytes = cfg.window_size * cfg.d_s * 2;
        let sem_ring_bytes = if cfg.share_stem_ring {
            0
        } else {
            cfg.window_size * cfg.d_sem_ring * 2
        };
        let ring_meta_bytes = cfg.window_size * 6; // pos (i32) + depth (u8) + valid (u8)
        let prev_bytes = cfg.d_s * 2 + cfg.d_concept * 2;
        let pmd_bytes = cfg.d_s * 2 + cfg.stack_depth * cfg.stack_tag_dim * 2 + 8;
        let scalar_bytes = 8;
        sgsc_bytes
            + carriers_bytes
            + stem_ring_bytes
            + sem_ring_bytes
            + ring_meta_bytes
            + prev_bytes
            + pmd_bytes
            + scalar_bytes
    }
}

/// Pre-allocated working scratch pool guaranteeing STRICTLY 0 heap allocations in the hot path!
pub struct FlatScratchPool {
    // Byte-level logits & probs
    pub prior_logits_256: [f32; 256],
    pub byte_logits_256: [f32; 256],

    // SGSC scratch
    pub ds_a: Vec<f32>,
    pub ds_b: Vec<f32>,
    pub ds_c: Vec<f32>,
    pub ds_d: Vec<f32>,
    pub f_stem: Vec<f32>,
    pub seg_scores: Vec<f32>, // [max_seg_len]
    pub x0: Vec<f32>,         // [D]
    pub l_reg: Vec<f32>,      // [64]

    // Carrier scratch (shared by G1 & G2)
    pub car_prior: CarrierGraphStateRs,
    pub car_xn: Vec<f32>,         // [D]
    pub car_vin: Vec<f32>,        // [D]
    pub car_uin: Vec<f32>,        // [D]
    pub car_ein_src: Vec<f32>,    // [d_r]
    pub car_ein_dst: Vec<f32>,    // [d_r]
    pub car_eslot_src: Vec<f32>,  // [K * d_r]
    pub car_eslot_dst: Vec<f32>,  // [K * d_r]
    pub car_eout_cand: Vec<f32>,  // [K * d_r]
    pub car_ein_cand: Vec<f32>,   // [K * d_r]
    pub car_h_si: Vec<f32>,       // [K * d_merge]
    pub car_h_sj: Vec<f32>,       // [K * d_merge]
    pub car_h_edge: Vec<f32>,     // [d_merge]
    pub car_h_pair: Vec<f32>,     // [d_merge]
    pub car_h_star: Vec<f32>,     // [d_merge]
    pub car_lam_logits: Vec<f32>, // [K]
    pub car_rho_tilde: Vec<f32>,  // [K]
    pub car_sim: Vec<f32>,        // [K]
    pub car_occ_act: Vec<f32>,    // [K]
    pub car_assim_sc: Vec<f32>,   // [K]
    pub car_edges_dec: Vec<f32>,  // [K * K * d_r]
    pub car_slots_s1: Vec<f32>,   // [K * D]
    pub car_occ_s1: Vec<f32>,     // [K]
    pub car_edges_s1: Vec<f32>,   // [K * K * d_r]
    pub car_slots_emp: Vec<f32>,  // [K * D]
    pub car_occ_emp: Vec<f32>,    // [K]
    pub car_edges_emp: Vec<f32>,  // [K * K * d_r]
    pub car_slots_mrg: Vec<f32>,  // [K * D]
    pub car_occ_mrg: Vec<f32>,    // [K]
    pub car_edges_mrg: Vec<f32>,  // [K * K * d_r]
    pub car_edges_row: Vec<f32>,  // [K * K * d_r]
    pub car_s_merged: Vec<f32>,   // [D]
    pub car_delta_m: Vec<f32>,    // [D]
    pub car_evict_w: Vec<f32>,    // [K]
    pub car_g_sum: Vec<f32>,      // [D]

    // Shared Graph KV scratch (`K_G`, `V_G`, `occ_log_prior`, `rel_bias`)
    pub gkv_k: Vec<f32>,        // [H * K * Dh]
    pub gkv_v: Vec<f32>,        // [H * K * Dh]
    pub gkv_occ_log: Vec<f32>,  // [K]
    pub gkv_rel_bias: Vec<f32>, // [H * K]
    pub gkv_edge_in: Vec<f32>,  // [K * d_r]
    pub gkv_edge_out: Vec<f32>, // [K * d_r]
    pub gkv_tmp_d1: Vec<f32>,   // [D]
    pub gkv_tmp_d2: Vec<f32>,   // [D]
    pub gkv_s_enr: Vec<f32>,    // [D]
    pub gkv_proj_d: Vec<f32>,   // [D]
    pub gkv_rb_h: Vec<f32>,     // [H]

    // Cortical stack & attention scratch
    pub cort_x: Vec<f32>,         // [D]
    pub cort_x1: Vec<f32>,        // [D]
    pub cort_x3: Vec<f32>,        // [D]
    pub cort_z_hat_6: Vec<f32>,   // [D]
    pub cort_z_hat: Vec<f32>,     // [D]
    pub cort_h_ref: Vec<f32>,     // [D]
    pub cort_xn: Vec<f32>,        // [D]
    pub cort_q_raw: Vec<f32>,     // [D]
    pub cort_q_rot: Vec<f32>,     // [D]
    pub cort_k_ring: Vec<f32>,    // [W * D]
    pub cort_v_ring: Vec<f32>,    // [W * D]
    pub cort_ctx: Vec<f32>,       // [D]
    pub cort_wo: Vec<f32>,        // [D]
    pub cort_reg_bias: Vec<f32>,  // [H * K]
    pub cort_ior_q: Vec<f32>,     // [D]
    pub cort_ior_pen: Vec<f32>,   // [H * K]
    pub cort_attn_scores: Vec<f32>, // [W + K + 1]
    pub cort_c_kv_sem: Vec<f32>,  // [d_sem_ring]
    pub cort_sem_tmp: Vec<f32>,   // [d_sem_ring]

    // Base MoE + Plugin MoE scratch
    pub moe_xn: Vec<f32>,        // [D]
    pub moe_shared_gu: Vec<f32>, // [2 * shared_hidden]
    pub moe_shared_h: Vec<f32>,  // [shared_hidden]
    pub moe_y_shared: Vec<f32>,  // [D]
    pub moe_router: Vec<f32>,    // [E]
    pub moe_sel: Vec<f32>,       // [E]
    pub moe_routed_gu: Vec<f32>, // [2 * routed_hidden]
    pub moe_routed_h: Vec<f32>,  // [routed_hidden]
    pub moe_y_routed: Vec<f32>,  // [D]
    pub moe_y_tmp: Vec<f32>,     // [D]
    pub moe_h_hat: Vec<f32>,     // [D]
    pub moe_y_plug: Vec<f32>,    // [D]
    pub plug_gu: Vec<f32>,       // [2 * max_plug_hidden] (resized on mount_expert_pack only!)
    pub plug_h: Vec<f32>,        // [max_plug_hidden]
    pub plug_sel_g: Vec<f32>,    // [plugin_top_k]
    pub plug_sel_pack: Vec<usize>,// [plugin_top_k]
    pub plug_sel_exp: Vec<usize>, // [plugin_top_k]

    // CopyRing scratch
    pub copy_qn: Vec<f32>,       // [D]
    pub copy_q_raw: Vec<f32>,    // [H * d_k]
    pub copy_q_rot: Vec<f32>,    // [H * d_k]
    pub copy_k_tmp: Vec<f32>,    // [H * d_k]
    pub copy_k_rot: Vec<f32>,    // [W * H * d_k]
    pub copy_scores: Vec<f32>,   // [W + 1]
    pub copy_w: Vec<f32>,        // [W]
    pub f_copy: Vec<f32>,        // [d_s]

    // Stem confluence & PMD scratch
    pub f_slow: Vec<f32>,        // [d_s]
    pub f_hip: Vec<f32>,         // [d_s]
    pub f_out: Vec<f32>,         // [d_s]
    pub delta_syn: Vec<f32>,     // [d_s]
    pub vec_d_dl: Vec<f32>,      // [D + d_l]
    pub vec_2ds_dl: Vec<f32>,    // [2*d_s + d_l]
    pub vec_3ds: Vec<f32>,       // [3*d_s]
    pub vec_ds_dl: Vec<f32>,     // [d_s + d_l]
    pub vec_2ds: Vec<f32>,       // [2*d_s]
    pub pmd_demod: Vec<f32>,     // [d_s]
    pub pmd_dstem: Vec<f32>,     // [d_s]
    pub pmd_ddrive: Vec<f32>,    // [d_s]
    pub pmd_xin: Vec<f32>,       // [d_s]
    pub pmd_st_out: Vec<f32>,    // [d_s]
    pub pmd_m_rms: Vec<f32>,     // [d_s]
    pub pmd_m_norm: Vec<f32>,    // [d_s]

    // Hippocampus & DG scratch
    pub dg_rot_prev: Vec<f32>,   // [d_s]
    pub dg_f_bound: Vec<f32>,    // [d_s]
    pub dg_w1_h: Vec<f32>,       // [dg_whiten_hidden]
    pub dg_f_w: Vec<f32>,        // [d_s]
    pub dg_scores: Vec<f32>,     // [d_h]
    pub psi_idx: Vec<usize>,     // [s_active]
    pub psi_val: Vec<f32>,       // [s_active]
    pub phi_idx: Vec<usize>,     // [s_active]
    pub phi_val: Vec<f32>,       // [s_active]
    pub hip_c_hip: Vec<f32>,     // [D]
    pub hip_c_fused: Vec<f32>,   // [D]
    pub hip_beta: Vec<f32>,      // [K]
    pub hip_proj_c: Vec<f32>,    // [D]
    pub hip_proj_l: Vec<f32>,    // [d_s]
}

impl FlatScratchPool {
    pub fn new(cfg: &DcpaFlatConfig) -> Self {
        let d_s = cfg.d_s;
        let d = cfg.d_concept;
        let k = cfg.k_slots;
        let d_r = cfg.d_r;
        let d_m = cfg.d_merge;
        let h = cfg.n_heads;
        let w = cfg.window_size;
        let d_l = cfg.d_l;
        let e = cfg.num_experts;
        let sh = cfg.shared_hidden;
        let rh = cfg.routed_hidden;
        let copy_proj = h * cfg.copy_ring_dk();
        let d_h = cfg.d_h;
        let s_act = cfg.s_active;
        let plug_k = cfg.plugin_top_k.max(1);

        Self {
            prior_logits_256: [0.0; 256],
            byte_logits_256: [0.0; 256],

            ds_a: vec![0.0; d_s],
            ds_b: vec![0.0; d_s],
            ds_c: vec![0.0; d_s],
            ds_d: vec![0.0; d_s],
            f_stem: vec![0.0; d_s],
            seg_scores: vec![0.0; cfg.max_seg_len.max(1)],
            x0: vec![0.0; d],
            l_reg: vec![0.0; d_l],

            car_prior: CarrierGraphStateRs::new(cfg),
            car_xn: vec![0.0; d],
            car_vin: vec![0.0; d],
            car_uin: vec![0.0; d],
            car_ein_src: vec![0.0; d_r],
            car_ein_dst: vec![0.0; d_r],
            car_eslot_src: vec![0.0; k * d_r],
            car_eslot_dst: vec![0.0; k * d_r],
            car_eout_cand: vec![0.0; k * d_r],
            car_ein_cand: vec![0.0; k * d_r],
            car_h_si: vec![0.0; k * d_m],
            car_h_sj: vec![0.0; k * d_m],
            car_h_edge: vec![0.0; d_m],
            car_h_pair: vec![0.0; d_m],
            car_h_star: vec![0.0; d_m],
            car_lam_logits: vec![0.0; k],
            car_rho_tilde: vec![0.0; k],
            car_sim: vec![0.0; k],
            car_occ_act: vec![0.0; k],
            car_assim_sc: vec![0.0; k],
            car_edges_dec: vec![0.0; k * k * d_r],
            car_slots_s1: vec![0.0; k * d],
            car_occ_s1: vec![0.0; k],
            car_edges_s1: vec![0.0; k * k * d_r],
            car_slots_emp: vec![0.0; k * d],
            car_occ_emp: vec![0.0; k],
            car_edges_emp: vec![0.0; k * k * d_r],
            car_slots_mrg: vec![0.0; k * d],
            car_occ_mrg: vec![0.0; k],
            car_edges_mrg: vec![0.0; k * k * d_r],
            car_edges_row: vec![0.0; k * k * d_r],
            car_s_merged: vec![0.0; d],
            car_delta_m: vec![0.0; d],
            car_evict_w: vec![0.0; k],
            car_g_sum: vec![0.0; d],

            gkv_k: vec![0.0; h * k * cfg.d_head()],
            gkv_v: vec![0.0; h * k * cfg.d_head()],
            gkv_occ_log: vec![0.0; k],
            gkv_rel_bias: vec![0.0; h * k],
            gkv_edge_in: vec![0.0; k * d_r],
            gkv_edge_out: vec![0.0; k * d_r],
            gkv_tmp_d1: vec![0.0; d],
            gkv_tmp_d2: vec![0.0; d],
            gkv_s_enr: vec![0.0; d],
            gkv_proj_d: vec![0.0; d],
            gkv_rb_h: vec![0.0; h],

            cort_x: vec![0.0; d],
            cort_x1: vec![0.0; d],
            cort_x3: vec![0.0; d],
            cort_z_hat_6: vec![0.0; d],
            cort_z_hat: vec![0.0; d],
            cort_h_ref: vec![0.0; d],
            cort_xn: vec![0.0; d],
            cort_q_raw: vec![0.0; d],
            cort_q_rot: vec![0.0; d],
            cort_k_ring: vec![0.0; w * d],
            cort_v_ring: vec![0.0; w * d],
            cort_ctx: vec![0.0; d],
            cort_wo: vec![0.0; d],
            cort_reg_bias: vec![0.0; h * k],
            cort_ior_q: vec![0.0; d],
            cort_ior_pen: vec![0.0; h * k],
            cort_attn_scores: vec![0.0; w + k + 1],
            cort_c_kv_sem: vec![0.0; cfg.d_sem_ring],
            cort_sem_tmp: vec![0.0; cfg.d_sem_ring],

            moe_xn: vec![0.0; d],
            moe_shared_gu: vec![0.0; 2 * sh],
            moe_shared_h: vec![0.0; sh],
            moe_y_shared: vec![0.0; d],
            moe_router: vec![0.0; e],
            moe_sel: vec![0.0; e],
            moe_routed_gu: vec![0.0; 2 * rh],
            moe_routed_h: vec![0.0; rh],
            moe_y_routed: vec![0.0; d],
            moe_y_tmp: vec![0.0; d],
            moe_h_hat: vec![0.0; d],
            moe_y_plug: vec![0.0; d],
            plug_gu: vec![0.0; 2 * rh],
            plug_h: vec![0.0; rh],
            plug_sel_g: vec![0.0; plug_k],
            plug_sel_pack: vec![0usize; plug_k],
            plug_sel_exp: vec![0usize; plug_k],

            copy_qn: vec![0.0; d],
            copy_q_raw: vec![0.0; copy_proj],
            copy_q_rot: vec![0.0; copy_proj],
            copy_k_tmp: vec![0.0; copy_proj],
            copy_k_rot: vec![0.0; w * copy_proj],
            copy_scores: vec![0.0; w + 1],
            copy_w: vec![0.0; w],
            f_copy: vec![0.0; d_s],

            f_slow: vec![0.0; d_s],
            f_hip: vec![0.0; d_s],
            f_out: vec![0.0; d_s],
            delta_syn: vec![0.0; d_s],
            vec_d_dl: vec![0.0; d + d_l],
            vec_2ds_dl: vec![0.0; 2 * d_s + d_l],
            vec_3ds: vec![0.0; 3 * d_s],
            vec_ds_dl: vec![0.0; d_s + d_l],
            vec_2ds: vec![0.0; 2 * d_s],
            pmd_demod: vec![0.0; d_s],
            pmd_dstem: vec![0.0; d_s],
            pmd_ddrive: vec![0.0; d_s],
            pmd_xin: vec![0.0; d_s],
            pmd_st_out: vec![0.0; d_s],
            pmd_m_rms: vec![0.0; d_s],
            pmd_m_norm: vec![0.0; d_s],

            dg_rot_prev: vec![0.0; d_s],
            dg_f_bound: vec![0.0; d_s],
            dg_w1_h: vec![0.0; cfg.dg_whiten_hidden],
            dg_f_w: vec![0.0; d_s],
            dg_scores: vec![0.0; d_h],
            psi_idx: vec![0usize; s_act],
            psi_val: vec![0.0; s_act],
            phi_idx: vec![0usize; s_act],
            phi_val: vec![0.0; s_act],
            hip_c_hip: vec![0.0; d],
            hip_c_fused: vec![0.0; d],
            hip_beta: vec![0.0; k],
            hip_proj_c: vec![0.0; d],
            hip_proj_l: vec![0.0; d_s],
        }
    }

    /// Ensure plugin scratch buffers accommodate a newly mounted pack's hidden dimension.
    pub fn ensure_plugin_hidden(&mut self, hidden_dim: usize) {
        if self.plug_h.len() < hidden_dim {
            self.plug_h.resize(hidden_dim, 0.0);
        }
        if self.plug_gu.len() < 2 * hidden_dim {
            self.plug_gu.resize(2 * hidden_dim, 0.0);
        }
    }
}

/// One real UTF-8 fragment recorded during awake high-beta or explicit-correction events (R6).
#[derive(Debug, Clone)]
pub struct SleepCapturedFragment {
    pub bytes: Vec<u8>,
    pub lang_id: u8,
    pub style_id: u8,
    pub anchor_byte_ranges: Vec<(usize, usize)>,
    pub peak_beta: f32,
}

/// Fixed-capacity awake replay ring capturing high-beta real fragments for `dcpa-sleep-daemon`.
///
/// Writes canonical `<QIBBBB` `.idx`, raw `.bin`, and fixed-point x10 `.loss_mask.bin`
/// (`10` normal, `15` syntax/protocol delimiters, `30` high-beta anchor spans).
#[derive(Debug, Clone)]
pub struct SleepReplayCaptureRing {
    pub max_fragments: usize,
    pub max_total_bytes: usize,
    pub beta_threshold: f32,
    fragments: Vec<SleepCapturedFragment>,
    total_bytes: usize,
}

impl SleepReplayCaptureRing {
    pub fn new(max_fragments: usize, max_total_bytes: usize, beta_threshold: f32) -> Self {
        Self {
            max_fragments: max_fragments.max(1),
            max_total_bytes: max_total_bytes.max(64),
            beta_threshold,
            fragments: Vec::new(),
            total_bytes: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn fragments(&self) -> &[SleepCapturedFragment] {
        &self.fragments
    }

    pub fn clear(&mut self) {
        self.fragments.clear();
        self.total_bytes = 0;
    }

    /// Record a fragment if `peak_beta > self.beta_threshold` OR `explicit_correction == true`.
    /// Returns `true` if accepted into the ring.
    pub fn maybe_record(
        &mut self,
        bytes: &[u8],
        lang_id: u8,
        style_id: u8,
        anchor_byte_ranges: &[(usize, usize)],
        peak_beta: f32,
        explicit_correction: bool,
    ) -> bool {
        if bytes.is_empty() || (!explicit_correction && !(peak_beta > self.beta_threshold)) {
            return false;
        }
        // Truncate overly large fragment on a valid UTF-8 boundary if needed
        let mut slice_len = bytes.len().min(self.max_total_bytes);
        if slice_len < bytes.len() {
            while slice_len > 0 && (bytes[slice_len] & 0xC0) == 0x80 {
                slice_len -= 1;
            }
        }
        if slice_len == 0 {
            return false;
        }
        while !self.fragments.is_empty()
            && (self.fragments.len() >= self.max_fragments
                || self.total_bytes + slice_len > self.max_total_bytes)
        {
            let evicted = self.fragments.remove(0);
            self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes.len());
        }
        let clamped_ranges: Vec<(usize, usize)> = anchor_byte_ranges
            .iter()
            .filter_map(|&(s, e)| {
                let cs = s.min(slice_len);
                let ce = e.min(slice_len);
                if cs < ce {
                    Some((cs, ce))
                } else {
                    None
                }
            })
            .collect();
        self.total_bytes += slice_len;
        self.fragments.push(SleepCapturedFragment {
            bytes: bytes[..slice_len].to_vec(),
            lang_id,
            style_id,
            anchor_byte_ranges: clamped_ranges,
            peak_beta,
        });
        true
    }

    #[cfg(feature = "std")]
    pub fn write_shard_triplet(&self, out_dir: &std::path::Path, stem: &str) -> std::io::Result<usize> {
        use alloc::format;
        use std::fs::{self, File};
        use std::io::Write;

        fs::create_dir_all(out_dir)?;
        let bin_path = out_dir.join(format!("{}.bin", stem));
        let idx_path = out_dir.join(format!("{}.idx", stem));
        let mask_path = out_dir.join(format!("{}.loss_mask.bin", stem));

        let mut f_bin = File::create(&bin_path)?;
        let mut f_idx = File::create(&idx_path)?;
        let mut f_mask = File::create(&mask_path)?;

        let mut offset: u64 = 0;
        let mut mask_buf: Vec<u8> = Vec::new();
        for frag in &self.fragments {
            let len = frag.bytes.len();
            if len == 0 {
                continue;
            }
            f_bin.write_all(&frag.bytes)?;

            // Build fixed-point x10 loss mask: 10 normal, 15 delimiter/protocol, 30 high-beta anchor span
            mask_buf.clear();
            mask_buf.resize(len, 10u8);
            let mut cur_depth: i32 = 0;
            let mut max_depth: i32 = 0;
            for (i, &b) in frag.bytes.iter().enumerate() {
                let is_delim = matches!(
                    b,
                    b'\n' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b':' | b';' | b',' | b'.' | b'='
                        | 0x04 | 0x11 | 0x12 | 0x13 | 0x14 | 0x1D | 0x1E | 0x1F
                );
                if is_delim {
                    mask_buf[i] = 15u8;
                }
                if matches!(b, b'(' | b'[' | b'{') {
                    cur_depth = (cur_depth + 1).min(31);
                    if cur_depth > max_depth {
                        max_depth = cur_depth;
                    }
                } else if matches!(b, b')' | b']' | b'}') {
                    cur_depth = (cur_depth - 1).max(0);
                } else if matches!(b, 0x04 | 0x13 | 0x1E) {
                    cur_depth = 0;
                }
            }
            for &(s, e) in &frag.anchor_byte_ranges {
                let cs = s.min(len);
                let ce = e.min(len);
                for v in &mut mask_buf[cs..ce] {
                    *v = 30u8;
                }
            }
            f_mask.write_all(&mask_buf)?;

            // Standard 16-byte `<QIBBBB` index record: (offset: u64, length: u32, lang_id, source_id, max_depth, style_id)
            let mut rec = [0u8; 16];
            rec[0..8].copy_from_slice(&offset.to_le_bytes());
            rec[8..12].copy_from_slice(&(len as u32).to_le_bytes());
            rec[12] = frag.lang_id;
            rec[13] = 99u8; // sleep-replay captured awake fragment source_id
            rec[14] = max_depth.clamp(0, 255) as u8;
            rec[15] = frag.style_id;
            f_idx.write_all(&rec)?;

            offset += len as u64;
        }
        f_bin.flush()?;
        f_idx.flush()?;
        f_mask.flush()?;
        Ok(self.fragments.len())
    }
}
