//! DCPA architecture configuration matching Python `dcpa/config.py` 1-to-1.

use serde::{Deserialize, Serialize};

fn default_num_style_domains() -> usize {
    5
}

fn default_lambda_moe_bal() -> f32 {
    0.01
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcpaConfig {
    pub vocab_size: usize,
    pub d_s: usize,
    pub d_concept: usize,
    pub k_slots: usize,
    pub d_r: usize,
    pub d_merge: usize,
    pub d_h: usize,
    pub s_active: usize,
    pub dg_seed_lex: u64,
    pub dg_seed_sem: u64,
    pub gamma_hip: f32,
    pub lambda_eta: f32,
    pub kappa_quench: f32,
    pub c0_ema_momentum: f32,
    pub householder_max_writes: usize,
    pub householder_write_threshold: f32,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub expert_hidden: usize,
    pub l_max: usize,
    pub tau_skip: f32,
    pub tau_halt: f32,
    pub ponder_eta: f32,
    #[serde(default = "default_lambda_moe_bal")]
    pub lambda_moe_bal: f32,
    pub lif_lambda: f32,
    pub lif_beta1: f32,
    pub lif_beta2: f32,
    pub lif_theta_init: f32,
    pub lif_theta_min: f32,
    pub max_seg_len: usize,
    pub target_spike_rate: f32,
    pub lambda_rate: f32,
    pub delta_decay: f32,
    pub d_l: usize,
    pub num_lang_domains: usize,
    #[serde(default = "default_num_style_domains")]
    pub num_style_domains: usize,
    pub stack_depth: usize,
    pub stack_tag_dim: usize,
    pub tau_e: f32,
    pub tau_q: f32,
    pub lambda_inh: f32,
    pub tau_done: f32,
    pub tau_repair: f32,
    pub max_repair_retries: usize,
    pub rope_period: f32,
    pub rope_num_harmonics: usize,
    pub lambda_var: f32,
    pub lambda_act: f32,
    pub lambda_halt: f32,
    pub eps: f32,
}

impl DcpaConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.vocab_size != 256 {
            return Err("Commandment #1 violation: vocab_size must be 256 (UTF-8 bytes)");
        }
        if self.k_slots < 2 || self.k_slots > 12 {
            return Err("Commandment #2 violation: k_slots must be in [2, 12]");
        }
        if self.d_s % 64 != 0 || self.d_concept % 64 != 0 {
            return Err("d_s and d_concept must be multiples of 64 for RoPE & SIMD");
        }
        if self.stack_depth != 16 || self.stack_tag_dim != 4 {
            return Err("PMD stack must be 16x4 (128B in bf16)");
        }
        if self.d_l != 64 {
            return Err("d_l must be 64");
        }
        Ok(())
    }

    pub fn num_merge_pairs(&self) -> usize {
        self.k_slots * (self.k_slots + 1)
    }

    #[cfg(feature = "std")]
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
