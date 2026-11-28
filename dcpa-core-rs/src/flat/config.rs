//! DCPA-Flat v3.3 configuration matching Python `dcpa.interfaces.DCPAFlatConfig` 1-to-1.
//!
//! Strictly `no_std` + `alloc` compatible.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize {
    256
}
fn default_grid_lanes() -> usize {
    32
}
fn default_grid_bytes() -> usize {
    4096
}
fn default_grid_segs() -> usize {
    640
}
fn default_max_seg_len() -> usize {
    16
}
fn default_doc_meta_cap_per_lane() -> usize {
    64
}
fn default_d_s() -> usize {
    192
}
fn default_d_concept() -> usize {
    512
}
fn default_n_heads() -> usize {
    8
}
fn default_k_slots() -> usize {
    8
}
fn default_d_r() -> usize {
    32
}
fn default_d_merge() -> usize {
    256
}
fn default_c_scan() -> usize {
    16
}
fn default_occ_min() -> f32 {
    0.05
}
fn default_occ_max() -> f32 {
    0.95
}
fn default_occ_relax() -> f32 {
    0.96
}
fn default_slot_keep_max() -> f32 {
    0.990
}
fn default_full_gate_center() -> f32 {
    0.20
}
fn default_full_gate_temp() -> f32 {
    0.10
}
fn default_edge_decay() -> f32 {
    0.97
}
fn default_carrier_grad_guard() -> String {
    String::from("none")
}
fn default_carrier_grad_clip_rms() -> f32 {
    1.0
}
fn default_n_layers() -> usize {
    8
}
fn default_n_pre_g1_layers() -> usize {
    1
}
fn default_n_post_g1_layers() -> usize {
    2
}
fn default_n_predict_full_layers() -> usize {
    3
}
fn default_n_refine_layers() -> usize {
    2
}
fn default_refine_frac() -> f32 {
    0.25
}
fn default_tau_think_ema() -> f32 {
    0.99
}
fn default_use_local_window_layers() -> bool {
    true
}
fn default_local_window_layers() -> Vec<usize> {
    vec![0, 3, 5]
}
fn default_window_size() -> usize {
    128
}
fn default_share_stem_ring() -> bool {
    false
}
fn default_d_sem_ring() -> usize {
    128
}
fn default_num_experts() -> usize {
    8
}
fn default_top_k_experts() -> usize {
    1
}
fn default_shared_hidden() -> usize {
    256
}
fn default_routed_hidden() -> usize {
    384
}
fn default_router_bias_lr() -> f32 {
    1e-3
}
fn default_router_logit_min() -> f32 {
    -4.0
}
fn default_plugin_top_k() -> usize {
    2
}
fn default_plugin_tau_min() -> f32 {
    0.35
}
fn default_d_h() -> usize {
    4096
}
fn default_s_active() -> usize {
    32
}
fn default_dg_seed_lex() -> u64 {
    42
}
fn default_dg_seed_sem() -> u64 {
    43
}
fn default_dg_whiten_hidden() -> usize {
    192
}
fn default_dg_homeo_lr() -> f32 {
    5e-3
}
fn default_hip_chunk_size() -> usize {
    32
}
fn default_gamma_hip() -> f32 {
    0.999
}
fn default_lambda_eta() -> f32 {
    0.95
}
fn default_kappa_quench() -> f32 {
    0.95
}
fn default_hip_write_threshold() -> f32 {
    0.05
}
fn default_hip_sleep_beta_threshold() -> f32 {
    0.25
}
fn default_d_l() -> usize {
    64
}
fn default_num_lang_domains() -> usize {
    6
}
fn default_num_style_domains() -> usize {
    5
}
fn default_stack_depth() -> usize {
    16
}
fn default_stack_tag_dim() -> usize {
    4
}
fn default_lif_lambda() -> f32 {
    0.85
}
fn default_lif_beta1() -> f32 {
    0.25
}
fn default_lif_beta2() -> f32 {
    0.75
}
fn default_lif_theta_init() -> f32 {
    2.2
}
fn default_lif_theta_min() -> f32 {
    0.05
}
fn default_lif_theta_scale() -> f32 {
    12.0
}
fn default_target_spike_rate() -> f32 {
    1.0 / 9.5
}
fn default_vocab_temp_init() -> f32 {
    8.0
}
fn default_tau_e() -> f32 {
    0.5
}
fn default_rope_period() -> f32 {
    33.0
}
fn default_rope_num_harmonics() -> usize {
    16
}
fn default_rope_theta_concept() -> f32 {
    10000.0
}
fn default_jepa_ema_momentum() -> f32 {
    0.996
}
fn default_lambda_jepa() -> f32 {
    0.5
}
fn default_lambda_unc() -> f32 {
    0.1
}
fn default_lambda_dg() -> f32 {
    0.05
}
fn default_lambda_rate() -> f32 {
    2.0
}
fn default_lambda_halt() -> f32 {
    1.0
}
fn default_lambda_var() -> f32 {
    1.0
}
fn default_lambda_sgsc_prior() -> f32 {
    0.2
}
fn default_lr_micro() -> f32 {
    6e-4
}
fn default_lr_concept() -> f32 {
    4e-4
}
fn default_lr_scalar_gate() -> f32 {
    1e-3
}
fn default_weight_decay() -> f32 {
    0.05
}
fn default_grad_clip_norm() -> f32 {
    1.0
}
fn default_eps() -> f32 {
    1e-6
}

/// Architectural configuration for DCPA-Flat v3.3 (`DCPAFlatConfig` JSON 1-to-1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcpaFlatConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_grid_lanes")]
    pub grid_lanes: usize,
    #[serde(default = "default_grid_bytes")]
    pub grid_bytes: usize,
    #[serde(default = "default_grid_segs")]
    pub grid_segs: usize,
    #[serde(default = "default_max_seg_len")]
    pub max_seg_len: usize,
    #[serde(default = "default_doc_meta_cap_per_lane")]
    pub doc_meta_cap_per_lane: usize,

    #[serde(default = "default_d_s")]
    pub d_s: usize,
    #[serde(default = "default_d_concept")]
    pub d_concept: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,

    #[serde(default = "default_k_slots")]
    pub k_slots: usize,
    #[serde(default = "default_d_r")]
    pub d_r: usize,
    #[serde(default = "default_d_merge")]
    pub d_merge: usize,
    #[serde(default = "default_c_scan")]
    pub c_scan: usize,
    #[serde(default = "default_occ_min")]
    pub occ_min: f32,
    #[serde(default = "default_occ_max")]
    pub occ_max: f32,
    #[serde(default = "default_occ_relax")]
    pub occ_relax: f32,
    #[serde(default = "default_slot_keep_max")]
    pub slot_keep_max: f32,
    #[serde(default = "default_full_gate_center")]
    pub full_gate_center: f32,
    #[serde(default = "default_full_gate_temp")]
    pub full_gate_temp: f32,
    #[serde(default = "default_edge_decay")]
    pub edge_decay: f32,
    #[serde(default = "default_carrier_grad_guard")]
    pub carrier_grad_guard: String,
    #[serde(default = "default_carrier_grad_clip_rms")]
    pub carrier_grad_clip_rms: f32,

    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_n_pre_g1_layers")]
    pub n_pre_g1_layers: usize,
    #[serde(default = "default_n_post_g1_layers")]
    pub n_post_g1_layers: usize,
    #[serde(default = "default_n_predict_full_layers")]
    pub n_predict_full_layers: usize,
    #[serde(default = "default_n_refine_layers")]
    pub n_refine_layers: usize,
    #[serde(default = "default_refine_frac")]
    pub refine_frac: f32,
    #[serde(default = "default_tau_think_ema")]
    pub tau_think_ema: f32,

    #[serde(default = "default_use_local_window_layers")]
    pub use_local_window_layers: bool,
    #[serde(default = "default_local_window_layers")]
    pub local_window_layers: Vec<usize>,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_share_stem_ring")]
    pub share_stem_ring: bool,
    #[serde(default = "default_d_sem_ring")]
    pub d_sem_ring: usize,

    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_top_k_experts")]
    pub top_k_experts: usize,
    #[serde(default = "default_shared_hidden")]
    pub shared_hidden: usize,
    #[serde(default = "default_routed_hidden")]
    pub routed_hidden: usize,
    #[serde(default = "default_router_bias_lr")]
    pub router_bias_lr: f32,
    #[serde(default = "default_router_logit_min")]
    pub router_logit_min: f32,
    #[serde(default = "default_plugin_top_k")]
    pub plugin_top_k: usize,
    #[serde(default = "default_plugin_tau_min")]
    pub plugin_tau_min: f32,

    #[serde(default = "default_d_h")]
    pub d_h: usize,
    #[serde(default = "default_s_active")]
    pub s_active: usize,
    #[serde(default = "default_dg_seed_lex")]
    pub dg_seed_lex: u64,
    #[serde(default = "default_dg_seed_sem")]
    pub dg_seed_sem: u64,
    #[serde(default = "default_dg_whiten_hidden")]
    pub dg_whiten_hidden: usize,
    #[serde(default = "default_dg_homeo_lr")]
    pub dg_homeo_lr: f32,
    #[serde(default = "default_hip_chunk_size")]
    pub hip_chunk_size: usize,
    #[serde(default = "default_gamma_hip")]
    pub gamma_hip: f32,
    #[serde(default = "default_lambda_eta")]
    pub lambda_eta: f32,
    #[serde(default = "default_kappa_quench")]
    pub kappa_quench: f32,
    #[serde(default = "default_hip_write_threshold")]
    pub hip_write_threshold: f32,
    #[serde(default = "default_hip_sleep_beta_threshold")]
    pub hip_sleep_beta_threshold: f32,

    #[serde(default = "default_d_l")]
    pub d_l: usize,
    #[serde(default = "default_num_lang_domains")]
    pub num_lang_domains: usize,
    #[serde(default = "default_num_style_domains")]
    pub num_style_domains: usize,
    #[serde(default = "default_stack_depth")]
    pub stack_depth: usize,
    #[serde(default = "default_stack_tag_dim")]
    pub stack_tag_dim: usize,
    #[serde(default = "default_lif_lambda")]
    pub lif_lambda: f32,
    #[serde(default = "default_lif_beta1")]
    pub lif_beta1: f32,
    #[serde(default = "default_lif_beta2")]
    pub lif_beta2: f32,
    #[serde(default = "default_lif_theta_init")]
    pub lif_theta_init: f32,
    #[serde(default = "default_lif_theta_min")]
    pub lif_theta_min: f32,
    #[serde(default = "default_lif_theta_scale")]
    pub lif_theta_scale: f32,
    #[serde(default = "default_target_spike_rate")]
    pub target_spike_rate: f32,
    #[serde(default = "default_vocab_temp_init")]
    pub vocab_temp_init: f32,
    #[serde(default = "default_tau_e")]
    pub tau_e: f32,
    #[serde(default = "default_rope_period")]
    pub rope_period: f32,
    #[serde(default = "default_rope_num_harmonics")]
    pub rope_num_harmonics: usize,
    #[serde(default = "default_rope_theta_concept")]
    pub rope_theta_concept: f32,

    #[serde(default = "default_jepa_ema_momentum")]
    pub jepa_ema_momentum: f32,
    #[serde(default = "default_lambda_jepa")]
    pub lambda_jepa: f32,
    #[serde(default = "default_lambda_unc")]
    pub lambda_unc: f32,
    #[serde(default = "default_lambda_dg")]
    pub lambda_dg: f32,
    #[serde(default = "default_lambda_rate")]
    pub lambda_rate: f32,
    #[serde(default = "default_lambda_halt")]
    pub lambda_halt: f32,
    #[serde(default = "default_lambda_var")]
    pub lambda_var: f32,
    #[serde(default = "default_lambda_sgsc_prior")]
    pub lambda_sgsc_prior: f32,

    #[serde(default = "default_lr_micro")]
    pub lr_micro: f32,
    #[serde(default = "default_lr_concept")]
    pub lr_concept: f32,
    #[serde(default = "default_lr_scalar_gate")]
    pub lr_scalar_gate: f32,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip_norm")]
    pub grad_clip_norm: f32,
    #[serde(default = "default_eps")]
    pub eps: f32,
}

impl Default for DcpaFlatConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            grid_lanes: default_grid_lanes(),
            grid_bytes: default_grid_bytes(),
            grid_segs: default_grid_segs(),
            max_seg_len: default_max_seg_len(),
            doc_meta_cap_per_lane: default_doc_meta_cap_per_lane(),
            d_s: default_d_s(),
            d_concept: default_d_concept(),
            n_heads: default_n_heads(),
            k_slots: default_k_slots(),
            d_r: default_d_r(),
            d_merge: default_d_merge(),
            c_scan: default_c_scan(),
            occ_min: default_occ_min(),
            occ_max: default_occ_max(),
            occ_relax: default_occ_relax(),
            slot_keep_max: default_slot_keep_max(),
            full_gate_center: default_full_gate_center(),
            full_gate_temp: default_full_gate_temp(),
            edge_decay: default_edge_decay(),
            carrier_grad_guard: default_carrier_grad_guard(),
            carrier_grad_clip_rms: default_carrier_grad_clip_rms(),
            n_layers: default_n_layers(),
            n_pre_g1_layers: default_n_pre_g1_layers(),
            n_post_g1_layers: default_n_post_g1_layers(),
            n_predict_full_layers: default_n_predict_full_layers(),
            n_refine_layers: default_n_refine_layers(),
            refine_frac: default_refine_frac(),
            tau_think_ema: default_tau_think_ema(),
            use_local_window_layers: default_use_local_window_layers(),
            local_window_layers: default_local_window_layers(),
            window_size: default_window_size(),
            share_stem_ring: default_share_stem_ring(),
            d_sem_ring: default_d_sem_ring(),
            num_experts: default_num_experts(),
            top_k_experts: default_top_k_experts(),
            shared_hidden: default_shared_hidden(),
            routed_hidden: default_routed_hidden(),
            router_bias_lr: default_router_bias_lr(),
            router_logit_min: default_router_logit_min(),
            plugin_top_k: default_plugin_top_k(),
            plugin_tau_min: default_plugin_tau_min(),
            d_h: default_d_h(),
            s_active: default_s_active(),
            dg_seed_lex: default_dg_seed_lex(),
            dg_seed_sem: default_dg_seed_sem(),
            dg_whiten_hidden: default_dg_whiten_hidden(),
            dg_homeo_lr: default_dg_homeo_lr(),
            hip_chunk_size: default_hip_chunk_size(),
            gamma_hip: default_gamma_hip(),
            lambda_eta: default_lambda_eta(),
            kappa_quench: default_kappa_quench(),
            hip_write_threshold: default_hip_write_threshold(),
            hip_sleep_beta_threshold: default_hip_sleep_beta_threshold(),
            d_l: default_d_l(),
            num_lang_domains: default_num_lang_domains(),
            num_style_domains: default_num_style_domains(),
            stack_depth: default_stack_depth(),
            stack_tag_dim: default_stack_tag_dim(),
            lif_lambda: default_lif_lambda(),
            lif_beta1: default_lif_beta1(),
            lif_beta2: default_lif_beta2(),
            lif_theta_init: default_lif_theta_init(),
            lif_theta_min: default_lif_theta_min(),
            lif_theta_scale: default_lif_theta_scale(),
            target_spike_rate: default_target_spike_rate(),
            vocab_temp_init: default_vocab_temp_init(),
            tau_e: default_tau_e(),
            rope_period: default_rope_period(),
            rope_num_harmonics: default_rope_num_harmonics(),
            rope_theta_concept: default_rope_theta_concept(),
            jepa_ema_momentum: default_jepa_ema_momentum(),
            lambda_jepa: default_lambda_jepa(),
            lambda_unc: default_lambda_unc(),
            lambda_dg: default_lambda_dg(),
            lambda_rate: default_lambda_rate(),
            lambda_halt: default_lambda_halt(),
            lambda_var: default_lambda_var(),
            lambda_sgsc_prior: default_lambda_sgsc_prior(),
            lr_micro: default_lr_micro(),
            lr_concept: default_lr_concept(),
            lr_scalar_gate: default_lr_scalar_gate(),
            weight_decay: default_weight_decay(),
            grad_clip_norm: default_grad_clip_norm(),
            eps: default_eps(),
        }
    }
}

impl DcpaFlatConfig {
    /// Tiny fast preset matching Python `DCPAFlatConfig.flat_micro()`.
    pub fn flat_micro() -> Self {
        Self {
            grid_lanes: 4,
            grid_bytes: 256,
            grid_segs: 64,
            window_size: 32,
            c_scan: 16,
            hip_chunk_size: 16,
            d_s: 64,
            d_concept: 128,
            n_heads: 4,
            k_slots: 8,
            d_r: 16,
            d_merge: 64,
            d_sem_ring: 64,
            num_experts: 4,
            top_k_experts: 1,
            shared_hidden: 64,
            routed_hidden: 96,
            d_h: 512,
            s_active: 16,
            dg_whiten_hidden: 64,
            ..Self::default()
        }
    }

    /// Flagship 206M preset matching Python `DCPAFlatConfig.flat_206m()`.
    pub fn flat_206m() -> Self {
        Self {
            d_s: 192,
            d_concept: 768,
            n_heads: 8,
            k_slots: 8,
            d_r: 32,
            d_merge: 256,
            d_sem_ring: 128,
            num_experts: 12,
            top_k_experts: 1,
            shared_hidden: 640,
            routed_hidden: 768,
            d_h: 4096,
            s_active: 32,
            dg_whiten_hidden: 192,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.vocab_size != 256 {
            return Err("vocab_size must be 256 (raw UTF-8 bytes)");
        }
        if self.k_slots < 2 || self.k_slots > 12 {
            return Err("k_slots must be in [2, 12]");
        }
        if self.d_s == 0 || self.d_s % 64 != 0 || self.d_concept == 0 || self.d_concept % 64 != 0 {
            return Err("d_s and d_concept must be positive multiples of 64");
        }
        if self.n_heads == 0
            || self.d_concept % self.n_heads != 0
            || (self.d_concept / self.n_heads) % 8 != 0
        {
            return Err("d_concept must be divisible by n_heads with d_head % 8 == 0");
        }
        if self.n_pre_g1_layers
            + self.n_post_g1_layers
            + self.n_predict_full_layers
            + self.n_refine_layers
            != self.n_layers
        {
            return Err("Layer counts (pre_g1 + post_g1 + predict_full + refine) must equal n_layers");
        }
        if self.n_pre_g1_layers != 1 {
            return Err("n_pre_g1_layers must be 1 (L1)");
        }
        if self.window_size == 0 || self.grid_segs % self.window_size != 0 {
            return Err("grid_segs must be a multiple of window_size");
        }
        if self.c_scan == 0 || self.grid_segs % self.c_scan != 0 {
            return Err("grid_segs must be a multiple of c_scan");
        }
        if !(self.edge_decay > 0.0 && self.edge_decay <= 0.995) {
            return Err("edge_decay must be in (0, 0.995]");
        }
        if !matches!(
            self.carrier_grad_guard.as_str(),
            "none" | "router_sg" | "soft_clip"
        ) {
            return Err("carrier_grad_guard must be 'none', 'router_sg', or 'soft_clip'");
        }
        if self.d_l != 64 {
            return Err("d_l must be 64 (32D lang + 32D style)");
        }
        if self.stack_depth != 16 || self.stack_tag_dim != 4 {
            return Err("PMD pushdown stack must be 16x4 (128B in BF16)");
        }
        if self.num_experts == 0 || self.top_k_experts == 0 || self.top_k_experts > self.num_experts
        {
            return Err("top_k_experts must be in 1..=num_experts");
        }
        if self.s_active == 0 || self.s_active > self.d_h {
            return Err("s_active must be in 1..=d_h");
        }
        Ok(())
    }

    #[inline]
    pub fn d_head(&self) -> usize {
        self.d_concept / self.n_heads
    }

    #[inline]
    pub fn ring_dim(&self) -> usize {
        if self.share_stem_ring {
            self.d_s
        } else {
            self.d_sem_ring
        }
    }

    #[inline]
    pub fn is_window_layer(&self, layer_idx: usize) -> bool {
        self.use_local_window_layers && self.local_window_layers.contains(&layer_idx)
    }

    /// Key/query head dimension of `HolographicCopyRing` matching `dcpa/modules/copy_ring.py` 1-to-1.
    #[inline]
    pub fn copy_ring_dk(&self) -> usize {
        if self.d_s % self.n_heads == 0
            && (self.d_s / self.n_heads) >= 8
            && (self.d_s / self.n_heads) % 2 == 0
        {
            self.d_s / self.n_heads
        } else {
            self.d_head()
        }
    }

    #[inline]
    pub fn num_merge_pairs(&self) -> usize {
        self.k_slots * (self.k_slots - 1)
    }

    #[cfg(feature = "std")]
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
