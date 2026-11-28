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
/// Maximum supported d_s for stack-allocated MotorCheckpoint (supports d_s = 384..512 in v2 presets).
pub const MAX_STACK_DS: usize = 512;

/// Module 1 SGSC constant-space streaming state (~388 B active core + fixed segment ring).
#[derive(Clone)]
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
    pub recent_bytes: [u8; 6],
    pub recent_len: usize,
    pub seg_pin_hit: bool,
    pub seg_unpin_hit: bool,
    pub pending_asst_pin: bool,
    pub pending_asst_ticks: u8,
    pub raw_ast_depth: i32,
    pub last_seg_end_depth: i32,
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
            recent_bytes: [0u8; 6],
            recent_len: 0,
            seg_pin_hit: false,
            seg_unpin_hit: false,
            pending_asst_pin: false,
            pending_asst_ticks: 0,
            raw_ast_depth: 0,
            last_seg_end_depth: 0,
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
        self.recent_bytes = [0u8; 6];
        self.recent_len = 0;
        self.seg_pin_hit = false;
        self.seg_unpin_hit = false;
        self.pending_asst_pin = false;
        self.pending_asst_ticks = 0;
        self.raw_ast_depth = 0;
        self.last_seg_end_depth = 0;
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.h.copy_from_slice(&src.h);
        self.v = src.v;
        self.prev_info = src.prev_info;
        self.has_prev_byte = src.has_prev_byte;
        self.seg_len = src.seg_len;
        self.utf8_rem = src.utf8_rem;
        self.seg_h.copy_from_slice(&src.seg_h);
        self.seg_info_sum = src.seg_info_sum;
        self.seg_bytes = src.seg_bytes;
        self.recent_bytes = src.recent_bytes;
        self.recent_len = src.recent_len;
        self.seg_pin_hit = src.seg_pin_hit;
        self.seg_unpin_hit = src.seg_unpin_hit;
        self.pending_asst_pin = src.pending_asst_pin;
        self.pending_asst_ticks = src.pending_asst_ticks;
        self.raw_ast_depth = src.raw_ast_depth;
        self.last_seg_end_depth = src.last_seg_end_depth;
    }
}

/// Shared O(1) MLA-compressed local morpheme/concept ring (`W=128`, ~48 KB in BF16 / ~96 KB in FP32).
#[derive(Clone)]
pub struct CorticalState {
    pub ring_kv: Vec<f32>,    // [W * d_s]
    pub ring_pos: Vec<i64>,   // [W]
    pub ring_depth: Vec<i32>, // [W]
    pub ring_valid: Vec<bool>,// [W]
    pub doc_pos: i64,
    pub ast_depth: i32,
    pub prev_z_ref: Vec<f32>, // [D]
    pub has_prev_seg: bool,
}

impl CorticalState {
    pub fn new(weights: &DcpaWeights) -> Self {
        let w = weights.cfg.window_size;
        let d_s = weights.cfg.d_s;
        Self {
            ring_kv: vec![0.0f32; w * d_s],
            ring_pos: vec![0i64; w],
            ring_depth: vec![0i32; w],
            ring_valid: vec![false; w],
            doc_pos: 0,
            ast_depth: 0,
            prev_z_ref: weights.cort_bos_concept.clone(),
            has_prev_seg: false,
        }
    }

    pub fn reset(&mut self, weights: &DcpaWeights) {
        self.ring_kv.fill(0.0);
        self.ring_pos.fill(0);
        self.ring_depth.fill(0);
        self.ring_valid.fill(false);
        self.doc_pos = 0;
        self.ast_depth = 0;
        self.prev_z_ref.copy_from_slice(&weights.cort_bos_concept);
        self.has_prev_seg = false;
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.ring_kv.copy_from_slice(&src.ring_kv);
        self.ring_pos.copy_from_slice(&src.ring_pos);
        self.ring_depth.copy_from_slice(&src.ring_depth);
        self.ring_valid.copy_from_slice(&src.ring_valid);
        self.doc_pos = src.doc_pos;
        self.ast_depth = src.ast_depth;
        self.prev_z_ref.copy_from_slice(&src.prev_z_ref);
        self.has_prev_seg = src.has_prev_seg;
    }
}

/// Module 2 BCG-PC constant-capacity working memory graph (K=8 -> 17.5 KB in L1!).
#[derive(Clone)]
pub struct ConceptGraphState {
    pub slots: Vec<f32>,    // [K * D]
    pub edges: Vec<f32>,    // [K * K * d_r]
    pub salience: Vec<f32>, // [K]
    pub phase: Vec<f32>,    // [K]
    pub slot0_pinned: bool, // Spec 2 Slot-0 Thesis Pinning latch
}

impl ConceptGraphState {
    pub fn new(weights: &DcpaWeights) -> Self {
        let k = weights.cfg.k_slots;
        Self {
            slots: weights.bcg_s0.clone(),
            edges: weights.bcg_a0.clone(),
            salience: weights.bcg_omega0.clone(),
            phase: vec![0.0f32; k],
            slot0_pinned: false,
        }
    }

    pub fn reset(&mut self, weights: &DcpaWeights) {
        self.slots.copy_from_slice(&weights.bcg_s0);
        self.edges.copy_from_slice(&weights.bcg_a0);
        self.salience.copy_from_slice(&weights.bcg_omega0);
        self.phase.fill(0.0);
        self.slot0_pinned = false;
    }

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.slots.copy_from_slice(&src.slots);
        self.edges.copy_from_slice(&src.edges);
        self.salience.copy_from_slice(&src.salience);
        self.phase.copy_from_slice(&src.phase);
        self.slot0_pinned = src.slot0_pinned;
    }
}

/// Per-slot synaptic eligibility tags (K=8 -> 4.1 KB in L1!).
#[derive(Clone)]
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

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.e_idx.copy_from_slice(&src.e_idx);
        self.e_val.copy_from_slice(&src.e_val);
        self.f_tilde.copy_from_slice(&src.f_tilde);
        self.eta.copy_from_slice(&src.eta);
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
        Self::new_with_dims(cfg.d_h, cfg.d_concept, cfg.d_s, cfg.gamma_hip, precision)
    }

    /// Construct column-major hippocampal fast weights directly from dimensions (shared by v2 & Flat v3.3).
    pub fn new_with_dims(
        d_h: usize,
        d_concept: usize,
        d_s: usize,
        gamma_hip: f32,
        precision: FastWeightPrecision,
    ) -> Self {
        let m_c_from_l = match precision {
            FastWeightPrecision::Bf16 => ColumnStorage::Bf16(AlignedVec::zeroed(d_h * d_concept)),
            FastWeightPrecision::F32 => ColumnStorage::F32(AlignedVec::zeroed(d_h * d_concept)),
        };
        let m_l_from_c = match precision {
            FastWeightPrecision::Bf16 => ColumnStorage::Bf16(AlignedVec::zeroed(d_h * d_s)),
            FastWeightPrecision::F32 => ColumnStorage::F32(AlignedVec::zeroed(d_h * d_s)),
        };
        Self {
            d_h,
            d_concept,
            d_s,
            gamma_hip,
            step_count: 0,
            m_c_from_l,
            m_l_from_c,
            u_lex: AlignedVec::zeroed(d_h),
            u_sem: AlignedVec::zeroed(d_h),
            col_step_lex: vec![0u64; d_h],
            col_step_sem: vec![0u64; d_h],
            col_hits_lex: vec![0u32; d_h],
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

    /// Zero all hippocampal fast weights and familiarity counters (`/wipe`).
    pub fn wipe_zero(&mut self) {
        self.step_count = 0;
        match &mut self.m_c_from_l {
            ColumnStorage::Bf16(buf) => buf.as_mut_slice().fill(Bf16(0)),
            ColumnStorage::F32(buf) => buf.as_mut_slice().fill(0.0),
        }
        match &mut self.m_l_from_c {
            ColumnStorage::Bf16(buf) => buf.as_mut_slice().fill(Bf16(0)),
            ColumnStorage::F32(buf) => buf.as_mut_slice().fill(0.0),
        }
        self.u_lex.as_mut_slice().fill(0.0);
        self.u_sem.as_mut_slice().fill(0.0);
        self.col_step_lex.fill(0);
        self.col_step_sem.fill(0);
        self.col_hits_lex.fill(0);
    }

    /// Deep-clone hippocampal fast weights (used once at startup for `M_tools_base` snapshot).
    pub fn clone_snapshot(&self) -> Self {
        let m_c_from_l = match &self.m_c_from_l {
            ColumnStorage::Bf16(src) => {
                let mut dst = AlignedVec::zeroed(src.len());
                dst.as_mut_slice().copy_from_slice(src.as_slice());
                ColumnStorage::Bf16(dst)
            }
            ColumnStorage::F32(src) => {
                let mut dst = AlignedVec::zeroed(src.len());
                dst.as_mut_slice().copy_from_slice(src.as_slice());
                ColumnStorage::F32(dst)
            }
        };
        let m_l_from_c = match &self.m_l_from_c {
            ColumnStorage::Bf16(src) => {
                let mut dst = AlignedVec::zeroed(src.len());
                dst.as_mut_slice().copy_from_slice(src.as_slice());
                ColumnStorage::Bf16(dst)
            }
            ColumnStorage::F32(src) => {
                let mut dst = AlignedVec::zeroed(src.len());
                dst.as_mut_slice().copy_from_slice(src.as_slice());
                ColumnStorage::F32(dst)
            }
        };
        let mut u_lex = AlignedVec::zeroed(self.u_lex.len());
        u_lex.as_mut_slice().copy_from_slice(self.u_lex.as_slice());
        let mut u_sem = AlignedVec::zeroed(self.u_sem.len());
        u_sem.as_mut_slice().copy_from_slice(self.u_sem.as_slice());
        Self {
            d_h: self.d_h,
            d_concept: self.d_concept,
            d_s: self.d_s,
            gamma_hip: self.gamma_hip,
            step_count: self.step_count,
            m_c_from_l,
            m_l_from_c,
            u_lex,
            u_sem,
            col_step_lex: self.col_step_lex.clone(),
            col_step_sem: self.col_step_sem.clone(),
            col_hits_lex: self.col_hits_lex.clone(),
        }
    }

    /// Zero-allocation in-place restore from a saved `HippocampusFastWeights` snapshot (`/reset`).
    pub fn restore_from(&mut self, snap: &Self) {
        self.step_count = snap.step_count;
        match (&mut self.m_c_from_l, &snap.m_c_from_l) {
            (ColumnStorage::Bf16(d), ColumnStorage::Bf16(s)) => d.as_mut_slice().copy_from_slice(s.as_slice()),
            (ColumnStorage::F32(d), ColumnStorage::F32(s)) => d.as_mut_slice().copy_from_slice(s.as_slice()),
            _ => panic!("FastWeightPrecision mismatch in restore_from"),
        }
        match (&mut self.m_l_from_c, &snap.m_l_from_c) {
            (ColumnStorage::Bf16(d), ColumnStorage::Bf16(s)) => d.as_mut_slice().copy_from_slice(s.as_slice()),
            (ColumnStorage::F32(d), ColumnStorage::F32(s)) => d.as_mut_slice().copy_from_slice(s.as_slice()),
            _ => panic!("FastWeightPrecision mismatch in restore_from"),
        }
        self.u_lex.as_mut_slice().copy_from_slice(snap.u_lex.as_slice());
        self.u_sem.as_mut_slice().copy_from_slice(snap.u_sem.as_slice());
        self.col_step_lex.copy_from_slice(&snap.col_step_lex);
        self.col_step_sem.copy_from_slice(&snap.col_step_sem);
        self.col_hits_lex.copy_from_slice(&snap.col_hits_lex);
    }
}

/// Module 4 PMD constant-space streaming state (~528 B in L1!).
#[derive(Clone)]
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

    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        self.m.copy_from_slice(&src.m);
        self.stack_xi = src.stack_xi;
        self.queue_q.copy_from_slice(&src.queue_q);
        self.queue_q_init.copy_from_slice(&src.queue_q_init);
        self.a_prev.copy_from_slice(&src.a_prev);
        self.last_byte = src.last_byte;
        self.charge_e = src.charge_e;
    }
}

/// O(1) ~22 KB Forkable Brain Working-Memory Snapshot (foundation for zero-KV-cache MCTS/Beam search!).
/// Captures & restores SGSC + K=8 ConceptGraph + EligibilityTags + Motor/128B Stack in microseconds
/// with STRICTLY 0 heap allocations during `capture_from` / `restore_into`!
#[derive(Clone)]
pub struct BrainSnapshot {
    pub sgsc: SgscState,
    pub cortical: CorticalState,
    pub graph: ConceptGraphState,
    pub elig: EligibilityTags,
    pub motor: MotorState,
}

impl BrainSnapshot {
    pub fn new(weights: &DcpaWeights) -> Self {
        Self {
            sgsc: SgscState::new(weights),
            cortical: CorticalState::new(weights),
            graph: ConceptGraphState::new(weights),
            elig: EligibilityTags::new(&weights.cfg),
            motor: MotorState::new(weights),
        }
    }

    #[inline]
    pub fn capture_from(
        &mut self,
        sgsc: &SgscState,
        cortical: &CorticalState,
        graph: &ConceptGraphState,
        elig: &EligibilityTags,
        motor: &MotorState,
    ) {
        self.sgsc.copy_from(sgsc);
        self.cortical.copy_from(cortical);
        self.graph.copy_from(graph);
        self.elig.copy_from(elig);
        self.motor.copy_from(motor);
    }

    #[inline]
    pub fn restore_into(
        &self,
        sgsc: &mut SgscState,
        cortical: &mut CorticalState,
        graph: &mut ConceptGraphState,
        elig: &mut EligibilityTags,
        motor: &mut MotorState,
    ) {
        sgsc.copy_from(&self.sgsc);
        cortical.copy_from(&self.cortical);
        graph.copy_from(&self.graph);
        elig.copy_from(&self.elig);
        motor.copy_from(&self.motor);
    }

    /// Exact active working-memory footprint in bytes.
    pub fn byte_footprint(&self) -> usize {
        let f32_elems = self.sgsc.h.len()
            + self.sgsc.seg_h.len()
            + self.cortical.ring_kv.len()
            + self.cortical.prev_z_ref.len()
            + self.graph.slots.len()
            + self.graph.edges.len()
            + self.graph.salience.len()
            + self.graph.phase.len()
            + self.elig.e_val.len()
            + self.elig.f_tilde.len()
            + self.elig.eta.len()
            + self.motor.m.len()
            + 64
            + self.motor.queue_q.len()
            + self.motor.queue_q_init.len()
            + self.motor.a_prev.len();
        let usize_elems = self.elig.e_idx.len();
        let ring_meta_bytes = self.cortical.ring_pos.len() * 8
            + self.cortical.ring_depth.len() * 4
            + self.cortical.ring_valid.len();
        f32_elems * core::mem::size_of::<f32>()
            + usize_elems * core::mem::size_of::<usize>()
            + ring_meta_bytes
            + 64
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
    pub bcg_z0: Vec<f32>,
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
    pub ca_k: Vec<f32>,         // [K * D] precomputed SlotCrossAttention keys
    pub ca_v: Vec<f32>,         // [K * D] precomputed SlotCrossAttention values
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

    // Cortical Sandwich & Contractive Phase Graph scratch (0 allocs/byte!)
    pub cort_h_enc: Vec<f32>,       // [D]
    pub cort_z_dec: Vec<f32>,       // [D]
    pub cort_x: Vec<f32>,           // [D]
    pub cort_tmp: Vec<f32>,         // [D]
    pub cort_tmp2: Vec<f32>,        // [D]
    pub cort_q: Vec<f32>,           // [D]
    pub cort_q_rot: Vec<f32>,       // [D]
    pub cort_k_ring: Vec<f32>,      // [W * D]
    pub cort_v_ring: Vec<f32>,      // [W * D]
    pub cort_gk: Vec<f32>,          // [K * D]
    pub cort_gv: Vec<f32>,          // [K * D]
    pub cort_sal_bias: Vec<f32>,    // [K]
    pub cort_scores: Vec<f32>,      // [W + K]
    pub cort_copy_acc: Vec<f32>,    // [W]
    pub cort_copy_f: Vec<f32>,      // [d_s]
    pub cort_copy_out: Vec<f32>,    // [d_s]
    pub cort_shared_gu: Vec<f32>,   // [2 * shared_expert_hidden]
    pub cort_shared_h: Vec<f32>,    // [shared_expert_hidden]
    pub cort_routed_h1: Vec<f32>,   // [cort_expert_hidden]
    pub cort_routed_h2: Vec<f32>,   // [cort_expert_hidden]
    pub cort_routed_out: Vec<f32>,  // [D]
    pub phase_feat_16: [f32; 16],
    pub phase_feat_32: [f32; 32],
    pub lam_k: Vec<f32>,            // [K]
    pub phase_tilde: Vec<f32>,      // [K + 1]
    pub s_phased: Vec<f32>,         // [K * D]
    pub s_contracted: Vec<f32>,     // [K * D]
    pub edge_in_kd: Vec<f32>,       // [K * d_r]
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
        let w = cfg.window_size;
        let sh = cfg.shared_expert_hidden;
        let ch = cfg.cort_expert_hidden;
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
            bcg_z0: vec![0.0; d],
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
            ca_k: vec![0.0; k * d],
            ca_v: vec![0.0; k * d],
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

            cort_h_enc: vec![0.0; d],
            cort_z_dec: vec![0.0; d],
            cort_x: vec![0.0; d],
            cort_tmp: vec![0.0; d],
            cort_tmp2: vec![0.0; d],
            cort_q: vec![0.0; d],
            cort_q_rot: vec![0.0; d],
            cort_k_ring: vec![0.0; w * d],
            cort_v_ring: vec![0.0; w * d],
            cort_gk: vec![0.0; k * d],
            cort_gv: vec![0.0; k * d],
            cort_sal_bias: vec![0.0; k],
            cort_scores: vec![0.0; w + k],
            cort_copy_acc: vec![0.0; w],
            cort_copy_f: vec![0.0; d_s],
            cort_copy_out: vec![0.0; d_s],
            cort_shared_gu: vec![0.0; 2 * sh],
            cort_shared_h: vec![0.0; sh],
            cort_routed_h1: vec![0.0; ch],
            cort_routed_h2: vec![0.0; ch],
            cort_routed_out: vec![0.0; d],
            phase_feat_16: [0.0; 16],
            phase_feat_32: [0.0; 32],
            lam_k: vec![0.0; k],
            phase_tilde: vec![0.0; kp1],
            s_phased: vec![0.0; k * d],
            s_contracted: vec![0.0; k * d],
            edge_in_kd: vec![0.0; k * d_r],
        }
    }
}
