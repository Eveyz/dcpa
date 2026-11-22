//! Zero-heap-allocation pre-allocated runtime working states for DCPA.
//!
//! Enforces Engineering Blueprint Chapter 6 Laws #1 and #2:
//!   - Column-Major 64-byte cache-line aligned hippocampal fast weights (7.52 MB in L3 for Base).
//!   - Constant L1 working memory concept graph (17.5 KB for Base), eligibility tags (4.1 KB),
//!     and motor decoder + 128B differentiable pushdown stack (528 B).
//!   - Stack-allocated MotorCheckpoint for zero-allocation Efference Copy local rollback.
//!   - Reusable ScratchPool so the hot loop performs STRICTLY 0 heap allocations (`0 allocs/byte`).

use alloc::vec;
use alloc::vec::Vec;

use crate::bf16::{AlignedVec, Bf16};
use crate::config::DcpaConfig;
use crate::simd::{
    accumulate_bf16_col_f32, axpy_f32, powf, scale_bf16_col_inplace, update_bf16_col_delta,
};
use crate::weights::DcpaWeights;

/// Maximum buffered segment length (hard cap 16 + up to 3 continuation bytes for in-flight UTF-8 char).
pub const MAX_SEG_CAP: usize = 20;
/// Maximum supported d_s for stack-allocated MotorCheckpoint (supports up to d_s = 256).
pub const MAX_STACK_DS: usize = 256;

/// Module 1 SGSC constant-space streaming state (~388 B active core + fixed segment ring).
pub struct SgscState {
    pub h: Vec<f32>,         // [d_s]
    pub v: f32,              // LIF membrane potential
    pub prev_info: f32,      // I_{t-1}
    pub has_prev_byte: bool, // False before byte 0 (so Delta_I_0 = 0.0)
    pub seg_len: usize,      // Current segment byte count (0..MAX_SEG_CAP)
    pub utf8_rem: u8,        // Strictly causal remaining UTF-8 continuation bytes
    pub seg_h: Vec<f32>,     // [MAX_SEG_CAP * d_s] flat pre-allocated segment hidden states
    pub seg_info_sum: f32,   // Running sum of I_t within current segment
    pub seg_bytes: [u8; MAX_SEG_CAP],
}

impl SgscState {
    pub fn new(weights: &DcpaWeights) -> Self {
        let d_s = weights.cfg.d_s;
        Self {
            h: weights.sgsc_h0.clone(),
            v: 0.0,
            prev_info: 0.0,
            has_prev_byte: false,
            seg_len: 0,
            utf8_rem: 0,
            seg_h: vec![0.0; MAX_SEG_CAP * d_s],
            seg_info_sum: 0.0,
            seg_bytes: [0u8; MAX_SEG_CAP],
        }
    }

    #[inline]
    pub fn reset(&mut self, weights: &DcpaWeights) {
        self.h.copy_from_slice(&weights.sgsc_h0);
        self.v = 0.0;
        self.prev_info = 0.0;
        self.has_prev_byte = false;
        self.seg_len = 0;
        self.utf8_rem = 0;
        self.seg_info_sum = 0.0;
    }
}

/// Module 2 BCG-PC constant-capacity working memory graph (K=8 -> 17.5 KB in L1!).
pub struct ConceptGraphState {
    pub slots: Vec<f32>,    // [K * D]
    pub edges: Vec<f32>,    // [K * K * d_r]
    pub salience: Vec<f32>, // [K]
}

impl ConceptGraphState {
    pub fn new(weights: &DcpaWeights) -> Self {
        Self {
            slots: weights.bcg_s0.clone(),
            edges: weights.bcg_a0.clone(),
            salience: weights.bcg_omega0.clone(),
        }
    }

    pub fn reset(&mut self, weights: &DcpaWeights) {
        self.slots.copy_from_slice(&weights.bcg_s0);
        self.edges.copy_from_slice(&weights.bcg_a0);
        self.salience.copy_from_slice(&weights.bcg_omega0);
    }
}

/// Per-slot synaptic eligibility tags (K=8 -> 4.1 KB in L1!).
pub struct EligibilityTags {
    pub k_slots: usize,
    pub s_active: usize,
    pub d_s: usize,
    pub e_idx: Vec<usize>, // [K * s]
    pub e_val: Vec<f32>,   // [K * s]
    pub f_tilde: Vec<f32>, // [K * d_s]
    pub eta: Vec<f32>,     // [K]
}

impl EligibilityTags {
    pub fn new(cfg: &DcpaConfig) -> Self {
        Self {
            k_slots: cfg.k_slots,
            s_active: cfg.s_active,
            d_s: cfg.d_s,
            e_idx: vec![0usize; cfg.k_slots * cfg.s_active],
            e_val: vec![0.0f32; cfg.k_slots * cfg.s_active],
            f_tilde: vec![0.0f32; cfg.k_slots * cfg.d_s],
            eta: vec![0.0f32; cfg.k_slots],
        }
    }

    pub fn reset(&mut self) {
        self.e_idx.fill(0);
        self.e_val.fill(0.0);
        self.f_tilde.fill(0.0);
        self.eta.fill(0.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastWeightPrecision {
    /// Production L3-resident 7.52 MB column-major bf16 storage.
    Bf16,
    /// Full float32 column-major storage for Gate 6 < 1e-5 parity testing.
    F32,
}

enum ColumnStorage {
    Bf16(AlignedVec<Bf16>),
    F32(AlignedVec<f32>),
}

/// Module 3 Column-Major 64-byte cache-line aligned Hippocampal Fast Weights (7.52 MB in L3).
pub struct HippocampusFastWeights {
    pub d_h: usize,
    pub d_concept: usize,
    pub d_s: usize,
    pub gamma_hip: f32,
    pub step_count: u64,
    m_c_from_l: ColumnStorage, // [d_h][D] column-major contiguous
    m_l_from_c: ColumnStorage, // [d_h][d_s] column-major contiguous
    pub u_lex: AlignedVec<f32>,
    pub u_sem: AlignedVec<f32>,
    pub col_step_lex: Vec<u64>,
    pub col_step_sem: Vec<u64>,
    pub col_hits_lex: Vec<u32>,
}

impl HippocampusFastWeights {
    pub fn new(cfg: &DcpaConfig, precision: FastWeightPrecision) -> Self {
        let m_c_from_l = match precision {
            FastWeightPrecision::Bf16 => ColumnStorage::Bf16(AlignedVec::zeroed(cfg.d_h * cfg.d_concept)),
            FastWeightPrecision::F32 => ColumnStorage::F32(AlignedVec::zeroed(cfg.d_h * cfg.d_concept)),
        };
        let m_l_from_c = match precision {
            FastWeightPrecision::Bf16 => ColumnStorage::Bf16(AlignedVec::zeroed(cfg.d_h * cfg.d_s)),
            FastWeightPrecision::F32 => ColumnStorage::F32(AlignedVec::zeroed(cfg.d_h * cfg.d_s)),
        };
        Self {
            d_h: cfg.d_h,
            d_concept: cfg.d_concept,
            d_s: cfg.d_s,
            gamma_hip: cfg.gamma_hip,
            step_count: 0,
            m_c_from_l,
            m_l_from_c,
            u_lex: AlignedVec::zeroed(cfg.d_h),
            u_sem: AlignedVec::zeroed(cfg.d_h),
            col_step_lex: vec![0u64; cfg.d_h],
            col_step_sem: vec![0u64; cfg.d_h],
            col_hits_lex: vec![0u32; cfg.d_h],
        }
    }

    pub fn byte_footprint(&self) -> usize {
        let c_bytes = match &self.m_c_from_l {
            ColumnStorage::Bf16(v) => v.len() * 2,
            ColumnStorage::F32(v) => v.len() * 4,
        };
        let l_bytes = match &self.m_l_from_c {
            ColumnStorage::Bf16(v) => v.len() * 2,
            ColumnStorage::F32(v) => v.len() * 4,
        };
        c_bytes + l_bytes + (self.u_lex.len() + self.u_sem.len()) * 4
    }

    #[inline(always)]
    pub fn advance_step(&mut self) {
        self.step_count += 1;
    }

    #[inline(always)]
    fn sync_lex_col(&mut self, k: usize) {
        if self.gamma_hip == 1.0 {
            return;
        }
        let prev = self.col_step_lex[k];
        if self.step_count > prev {
            let dt = (self.step_count - prev) as f32;
            let scale = powf(self.gamma_hip, dt);
            let off = k * self.d_concept;
            match &mut self.m_c_from_l {
                ColumnStorage::Bf16(buf) => {
                    scale_bf16_col_inplace(scale, &mut buf.as_mut_slice()[off..off + self.d_concept]);
                }
                ColumnStorage::F32(buf) => {
                    for v in &mut buf.as_mut_slice()[off..off + self.d_concept] {
                        *v *= scale;
                    }
                }
            }
            self.u_lex.as_mut_slice()[k] *= scale;
            self.col_step_lex[k] = self.step_count;
        }
    }

    #[inline(always)]
    fn sync_sem_col(&mut self, k: usize) {
        if self.gamma_hip == 1.0 {
            return;
        }
        let prev = self.col_step_sem[k];
        if self.step_count > prev {
            let dt = (self.step_count - prev) as f32;
            let scale = powf(self.gamma_hip, dt);
            let off = k * self.d_s;
            match &mut self.m_l_from_c {
                ColumnStorage::Bf16(buf) => {
                    scale_bf16_col_inplace(scale, &mut buf.as_mut_slice()[off..off + self.d_s]);
                }
                ColumnStorage::F32(buf) => {
                    for v in &mut buf.as_mut_slice()[off..off + self.d_s] {
                        *v *= scale;
                    }
                }
            }
            self.u_sem.as_mut_slice()[k] *= scale;
            self.col_step_sem[k] = self.step_count;
        }
    }

    /// Sparse column gather: c_out = M^{C<-L} @ psi and gamma_hip = sum_k psi_k^2 * u_lex[k].
    /// Touches ONLY the s active contiguous 64B-aligned columns. Zero heap allocation!
    pub fn read_c_from_l(
        &mut self,
        psi_idx: &[usize],
        psi_val: &[f32],
        c_out: &mut [f32],
    ) -> f32 {
        c_out.fill(0.0);
        let mut gamma_hip = 0.0f32;
        let s = psi_idx.len();
        for i in 0..s {
            let k = psi_idx[i];
            let w = psi_val[i];
            self.sync_lex_col(k);
            let uk = self.u_lex.as_slice()[k];
            gamma_hip += w * w * uk;
            if uk > 0.1 {
                self.col_hits_lex[k] = self.col_hits_lex[k].saturating_add(1);
            }
            let off = k * self.d_concept;
            match &self.m_c_from_l {
                ColumnStorage::Bf16(buf) => {
                    accumulate_bf16_col_f32(w, &buf.as_slice()[off..off + self.d_concept], c_out);
                }
                ColumnStorage::F32(buf) => {
                    axpy_f32(w, &buf.as_slice()[off..off + self.d_concept], c_out);
                }
            }
        }
        gamma_hip.clamp(0.0, 1.0)
    }

    /// Sparse column gather: f_out = M^{L<-C} @ phi and gamma_sem = sum_k phi_k^2 * u_sem[k].
    pub fn read_l_from_c(
        &mut self,
        phi_idx: &[usize],
        phi_val: &[f32],
        f_out: &mut [f32],
    ) -> f32 {
        f_out.fill(0.0);
        let mut gamma_sem = 0.0f32;
        let s = phi_idx.len();
        for i in 0..s {
            let k = phi_idx[i];
            let w = phi_val[i];
            self.sync_sem_col(k);
            gamma_sem += w * w * self.u_sem.as_slice()[k];
            let off = k * self.d_s;
            match &self.m_l_from_c {
                ColumnStorage::Bf16(buf) => {
                    accumulate_bf16_col_f32(w, &buf.as_slice()[off..off + self.d_s], f_out);
                }
                ColumnStorage::F32(buf) => {
                    axpy_f32(w, &buf.as_slice()[off..off + self.d_s], f_out);
                }
            }
        }
        gamma_sem.clamp(0.0, 1.0)
    }

    /// In-place sparse Column-Major Householder Gated Delta erase-write for ONE slot.
    /// Zero heap allocations! Uses caller-provided scratch buffers `proj_c [D]` & `proj_l [d_s]`.
    pub fn write_slot_inplace(
        &mut self,
        beta: f32,
        e_idx: &[usize],
        e_val: &[f32],
        s_bar: &[f32],
        phi_idx: &[usize],
        phi_val: &[f32],
        f_tilde: &[f32],
        proj_c: &mut [f32],
        proj_l: &mut [f32],
    ) {
        if beta <= 0.0 {
            return;
        }
        let s = e_idx.len();

        // 1. Comprehension write M^{C<-L}: read proj_c = M @ e, then err = beta * (s_bar - proj_c)
        proj_c.fill(0.0);
        for i in 0..s {
            let k = e_idx[i];
            let w = e_val[i];
            self.sync_lex_col(k);
            let off = k * self.d_concept;
            match &self.m_c_from_l {
                ColumnStorage::Bf16(buf) => {
                    accumulate_bf16_col_f32(w, &buf.as_slice()[off..off + self.d_concept], proj_c);
                }
                ColumnStorage::F32(buf) => {
                    axpy_f32(w, &buf.as_slice()[off..off + self.d_concept], proj_c);
                }
            }
        }
        for d in 0..self.d_concept {
            proj_c[d] = beta * (s_bar[d] - proj_c[d]);
        }
        for i in 0..s {
            let k = e_idx[i];
            let w = e_val[i];
            let off = k * self.d_concept;
            match &mut self.m_c_from_l {
                ColumnStorage::Bf16(buf) => {
                    update_bf16_col_delta(w, proj_c, &mut buf.as_mut_slice()[off..off + self.d_concept]);
                }
                ColumnStorage::F32(buf) => {
                    axpy_f32(w, proj_c, &mut buf.as_mut_slice()[off..off + self.d_concept]);
                }
            }
            if w > 0.0 {
                let u_cur = self.u_lex.as_slice()[k];
                self.u_lex.as_mut_slice()[k] = (u_cur + beta * (1.0 - u_cur)).clamp(0.0, 1.0);
            }
        }

        // 2. Expression write M^{L<-C}: read proj_l = M @ phi, then err = beta * (f_tilde - proj_l)
        proj_l.fill(0.0);
        for i in 0..s {
            let k = phi_idx[i];
            let w = phi_val[i];
            self.sync_sem_col(k);
            let off = k * self.d_s;
            match &self.m_l_from_c {
                ColumnStorage::Bf16(buf) => {
                    accumulate_bf16_col_f32(w, &buf.as_slice()[off..off + self.d_s], proj_l);
                }
                ColumnStorage::F32(buf) => {
                    axpy_f32(w, &buf.as_slice()[off..off + self.d_s], proj_l);
                }
            }
        }
        for d in 0..self.d_s {
            proj_l[d] = beta * (f_tilde[d] - proj_l[d]);
        }
        for i in 0..s {
            let k = phi_idx[i];
            let w = phi_val[i];
            let off = k * self.d_s;
            match &mut self.m_l_from_c {
                ColumnStorage::Bf16(buf) => {
                    update_bf16_col_delta(w, proj_l, &mut buf.as_mut_slice()[off..off + self.d_s]);
                }
                ColumnStorage::F32(buf) => {
                    axpy_f32(w, proj_l, &mut buf.as_mut_slice()[off..off + self.d_s]);
                }
            }
            if w > 0.0 {
                let u_cur = self.u_sem.as_slice()[k];
                self.u_sem.as_mut_slice()[k] = (u_cur + beta * (1.0 - u_cur)).clamp(0.0, 1.0);
            }
        }
    }

    /// Flush lazy decay across all columns (used by sleep daemon inspection / golden export parity).
    pub fn sync_all_u_lex(&mut self) {
        if self.gamma_hip == 1.0 {
            return;
        }
        for k in 0..self.d_h {
            self.sync_lex_col(k);
        }
    }

    /// Extract single column k of M^{C<-L} into f32 slice (for SWR sleep distillation).
    pub fn read_lex_column_f32(&mut self, k: usize, out: &mut [f32]) {
        self.sync_lex_col(k);
        let off = k * self.d_concept;
        match &self.m_c_from_l {
            ColumnStorage::Bf16(buf) => {
                let s = &buf.as_slice()[off..off + self.d_concept];
                for i in 0..self.d_concept {
                    out[i] = s[i].to_f32();
                }
            }
            ColumnStorage::F32(buf) => {
                out.copy_from_slice(&buf.as_slice()[off..off + self.d_concept]);
            }
        }
    }
}

/// Module 4 PMD constant-space streaming state (~528 B in L1!).
pub struct MotorState {
    pub m: Vec<f32>,             // [d_s]
    pub stack_xi: [[f32; 4]; 16],// 16 x 4 differentiable pushdown stack
    pub queue_q: Vec<f32>,       // [K] Broca CQ remaining salience
    pub queue_q_init: Vec<f32>,  // [K] q_0 = sigmoid(omega)
    pub a_prev: Vec<f32>,        // [K * d_r]
    pub last_byte: u8,
    pub charge_e: f32,
}

impl MotorState {
    pub fn new(weights: &DcpaWeights) -> Self {
        let k = weights.cfg.k_slots;
        let d_r = weights.cfg.d_r;
        Self {
            m: weights.pmd_m0.clone(),
            stack_xi: [[0.0f32; 4]; 16],
            queue_q: vec![0.0f32; k],
            queue_q_init: vec![0.0f32; k],
            a_prev: vec![0.0f32; k * d_r],
            last_byte: 0,
            charge_e: 0.0,
        }
    }

    pub fn reset(&mut self, weights: &DcpaWeights) {
        self.m.copy_from_slice(&weights.pmd_m0);
        self.stack_xi = [[0.0f32; 4]; 16];
        self.queue_q.fill(0.0);
        self.queue_q_init.fill(0.0);
        self.a_prev.fill(0.0);
        self.last_byte = 0;
        self.charge_e = 0.0;
    }
}

/// Stack-allocated Efference Copy local rollback snapshot (zero heap allocation!).
#[derive(Clone, Copy)]
pub struct MotorCheckpoint {
    pub m_prev: [f32; MAX_STACK_DS],
    pub stack_xi_prev: [[f32; 4]; 16],
    pub last_byte_prev: u8,
}

impl MotorCheckpoint {
    #[inline]
    pub fn capture(motor: &MotorState) -> Self {
        let mut m_prev = [0.0f32; MAX_STACK_DS];
        let d_s = motor.m.len();
        m_prev[..d_s].copy_from_slice(&motor.m);
        Self {
            m_prev,
            stack_xi_prev: motor.stack_xi,
            last_byte_prev: motor.last_byte,
        }
    }

    #[inline]
    pub fn restore(&self, motor: &mut MotorState) {
        let d_s = motor.m.len();
        motor.m.copy_from_slice(&self.m_prev[..d_s]);
        motor.stack_xi = self.stack_xi_prev;
        motor.last_byte = self.last_byte_prev;
    }
}

/// Pre-allocated flat temporary vector pool (~64 KB) guaranteeing 0 allocs/byte in hot loop!
pub struct ScratchPool {
    // SGSC scratch
    pub logits_256: [f32; 256],
    pub probs_256: [f32; 256],
    pub vec_ds_a: Vec<f32>,
    pub vec_ds_b: Vec<f32>,
    pub vec_ds_c: Vec<f32>,
    pub vec_ds_d: Vec<f32>,
    pub vec_ds_e: Vec<f32>,
    pub seg_scores: [f32; MAX_SEG_CAP],

    // Concept / BCG-PC scratch
    pub vec_d_a: Vec<f32>,
    pub vec_d_b: Vec<f32>,
    pub vec_d_c: Vec<f32>,
    pub vec_d_d: Vec<f32>,
    pub vec_d_e: Vec<f32>,
    pub vec_d_f: Vec<f32>,
    pub vec_d_g: Vec<f32>,
    pub vec_d_h: Vec<f32>,
    pub vec_2d: Vec<f32>,
    pub vec_2d_dr: Vec<f32>,
    pub expert_h1: Vec<f32>,
    pub expert_h2: Vec<f32>,
    pub expert_out: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub s_bar: Vec<f32>,        // [K * D]
    pub s_bar_post: Vec<f32>,   // [K * D]
    pub s_tilde: Vec<f32>,      // [(K+1) * D]
    pub a_tilde: Vec<f32>,      // [(K+1) * (K+1) * d_r]
    pub a_tmp: Vec<f32>,        // [K * (K+1) * d_r]
    pub src_all: Vec<f32>,      // [(K+1) * d_r]
    pub dst_all: Vec<f32>,      // [(K+1) * d_r]
    pub wl_all: Vec<f32>,       // [(K+1) * d_merge]
    pub wr_all: Vec<f32>,       // [(K+1) * d_merge]
    pub merge_h: Vec<f32>,      // [d_merge]
    pub omega_tilde: Vec<f32>,  // [K+1]
    pub ca_scores: Vec<f32>,    // [K]
    pub w_attn: Vec<f32>,       // [K]

    // Hippocampus scratch
    pub dg_scores: Vec<f32>,    // [d_h]
    pub psi_idx: Vec<usize>,    // [s]
    pub psi_val: Vec<f32>,      // [s]
    pub phi_idx: Vec<usize>,    // [s]
    pub phi_val: Vec<f32>,      // [s]
    pub hip_c_fused: Vec<f32>,  // [D]
    pub hip_prec_bu: Vec<f32>,  // [D]
    pub hip_c_hip: Vec<f32>,    // [D]
    pub hip_proj_c: Vec<f32>,   // [D]
    pub hip_proj_l: Vec<f32>,   // [d_s]

    // PMD scratch
    pub vec_3ds: Vec<f32>,      // [3 * d_s]
    pub vec_ds_dl: Vec<f32>,    // [d_s + d_l]
    pub vec_2ds_dl: Vec<f32>,   // [2 * d_s + d_l]
    pub vec_dr_dl: Vec<f32>,    // [d_r + d_l]
    pub vec_dr_a: Vec<f32>,     // [d_r]
    pub vec_dr_b: Vec<f32>,     // [d_r]
    pub p_ord: Vec<f32>,        // [K * K]
    pub u_ready: Vec<f32>,      // [K]
    pub pmd_z_out: Vec<f32>,    // [D]
    pub pmd_delta_syn: Vec<f32>,// [d_s]
    pub pmd_f_out: Vec<f32>,    // [d_s]

    // Efference Copy isolated monitor SGSC scratch
    pub mon_h: Vec<f32>,        // [d_s]
    pub mon_seg_h: Vec<f32>,    // [MAX_SEG_CAP * d_s]
    pub mon_f_stem: Vec<f32>,   // [d_s]
    pub mon_c_slow: Vec<f32>,   // [D]
    pub mon_c_check: Vec<f32>,  // [D]
}

impl ScratchPool {
    pub fn new(cfg: &DcpaConfig) -> Self {
        let d_s = cfg.d_s;
        let d = cfg.d_concept;
        let k = cfg.k_slots;
        let kp1 = k + 1;
        let d_r = cfg.d_r;
        let d_l = cfg.d_l;
        let s = cfg.s_active;
        Self {
            logits_256: [0.0; 256],
            probs_256: [0.0; 256],
            vec_ds_a: vec![0.0; d_s],
            vec_ds_b: vec![0.0; d_s],
            vec_ds_c: vec![0.0; d_s],
            vec_ds_d: vec![0.0; d_s],
            vec_ds_e: vec![0.0; d_s],
            seg_scores: [0.0; MAX_SEG_CAP],

            vec_d_a: vec![0.0; d],
            vec_d_b: vec![0.0; d],
            vec_d_c: vec![0.0; d],
            vec_d_d: vec![0.0; d],
            vec_d_e: vec![0.0; d],
            vec_d_f: vec![0.0; d],
            vec_d_g: vec![0.0; d],
            vec_d_h: vec![0.0; d],
            vec_2d: vec![0.0; 2 * d],
            vec_2d_dr: vec![0.0; 2 * d + d_r],
            expert_h1: vec![0.0; cfg.expert_hidden],
            expert_h2: vec![0.0; cfg.expert_hidden],
            expert_out: vec![0.0; d],
            router_logits: vec![0.0; cfg.num_experts],
            s_bar: vec![0.0; k * d],
            s_bar_post: vec![0.0; k * d],
            s_tilde: vec![0.0; kp1 * d],
            a_tilde: vec![0.0; kp1 * kp1 * d_r],
            a_tmp: vec![0.0; k * kp1 * d_r],
            src_all: vec![0.0; kp1 * d_r],
            dst_all: vec![0.0; kp1 * d_r],
            wl_all: vec![0.0; kp1 * cfg.d_merge],
            wr_all: vec![0.0; kp1 * cfg.d_merge],
            merge_h: vec![0.0; cfg.d_merge],
            omega_tilde: vec![0.0; kp1],
            ca_scores: vec![0.0; k],
            w_attn: vec![0.0; k],

            dg_scores: vec![0.0; cfg.d_h],
            psi_idx: vec![0; s],
            psi_val: vec![0.0; s],
            phi_idx: vec![0; s],
            phi_val: vec![0.0; s],
            hip_c_fused: vec![0.0; d],
            hip_prec_bu: vec![0.0; d],
            hip_c_hip: vec![0.0; d],
            hip_proj_c: vec![0.0; d],
            hip_proj_l: vec![0.0; d_s],

            vec_3ds: vec![0.0; 3 * d_s],
            vec_ds_dl: vec![0.0; d_s + d_l],
            vec_2ds_dl: vec![0.0; 2 * d_s + d_l],
            vec_dr_dl: vec![0.0; d_r + d_l],
            vec_dr_a: vec![0.0; d_r],
            vec_dr_b: vec![0.0; d_r],
            p_ord: vec![0.0; k * k],
            u_ready: vec![0.0; k],
            pmd_z_out: vec![0.0; d],
            pmd_delta_syn: vec![0.0; d_s],
            pmd_f_out: vec![0.0; d_s],

            mon_h: vec![0.0; d_s],
            mon_seg_h: vec![0.0; MAX_SEG_CAP * d_s],
            mon_f_stem: vec![0.0; d_s],
            mon_c_slow: vec![0.0; d],
            mon_c_check: vec![0.0; d],
        }
    }
}
