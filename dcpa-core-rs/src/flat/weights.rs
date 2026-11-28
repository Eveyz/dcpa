//! DCPA-Flat v3.3 slow weight container and hot-mountable `PluginExpertPackRs`.
//!
//! Loads every tensor exported by `dcpa.export.extract_flat_v3_state_dict` and unified
//! `.dcpa_expert.safetensors` files via the existing pure-Rust `parse_safetensors` parser.
//! Strictly `no_std` + `alloc` compatible (`from_safetensors` gated behind `#[cfg(feature = "std")]`).

#[cfg(feature = "std")]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use alloc::format;
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::flat::config::DcpaFlatConfig;
use crate::simd::{dot_f32, sqrtf};
#[cfg(feature = "std")]
use crate::weights::{parse_safetensors, RawTensor};

/// Read-only weights for one `ContractiveGraphCarrier` (`G1` or `G2`).
pub struct FlatCarrierWeights {
    pub init_slots_raw: Vec<f32>,  // [K, D]
    pub init_edges_raw: Vec<f32>,  // [K, K, d_r]
    pub norm_in: Vec<f32>,         // [D]
    pub w_in: Vec<f32>,            // [D, D]
    pub w_lam_w: Vec<f32>,         // [K, D]
    pub w_lam_b: Vec<f32>,         // [K]
    pub assim_scale: f32,
    pub assim_threshold: f32,
    pub lam_sim_gain: f32,
    pub w_edge_in_src: Vec<f32>,   // [d_r, D]
    pub w_edge_in_dst: Vec<f32>,   // [d_r, D]
    pub w_edge_slot_src: Vec<f32>, // [d_r, D]
    pub w_edge_slot_dst: Vec<f32>, // [d_r, D]
    pub edge_write_logit: f32,
    pub w_pair_si: Vec<f32>,       // [d_merge, D]
    pub w_pair_sj: Vec<f32>,       // [d_merge, D]
    pub w_pair_edge: Vec<f32>,     // [d_merge, d_r]
    pub w_pair_rho_i: Vec<f32>,    // [d_merge]
    pub w_pair_rho_j: Vec<f32>,    // [d_merge]
    pub pair_bias: Vec<f32>,       // [d_merge]
    pub w_pair_score_w: Vec<f32>,  // [d_merge]
    pub w_pair_score_b: f32,
    pub pair_cos_gain: f32,
    pub w_merge_beta_w: Vec<f32>,  // [d_merge]
    pub w_merge_beta_b: f32,
    pub w_merge_delta: Vec<f32>,   // [D, d_merge]
}

/// Read-only weights for `_GraphKVProjection` (`kv_g1` or `kv_g2`).
pub struct FlatGraphKvWeights {
    pub norm_slot: Vec<f32>,  // [D]
    pub w_edge_in: Vec<f32>,  // [D, d_r]
    pub w_edge_out: Vec<f32>, // [D, d_r]
    pub w_gk: Vec<f32>,       // [D, D]
    pub w_gv: Vec<f32>,       // [D, D]
    pub w_rel_bias: Vec<f32>, // [H, d_r]
}

/// Read-only weights for `DecoupledPluggableMoE` base pool (1 shared + E routed).
pub struct FlatMoEWeights {
    pub norm: Vec<f32>,        // [D]
    pub shared_w_gu: Vec<f32>, // [2*shared_hidden, D]
    pub shared_w_d: Vec<f32>,  // [D, shared_hidden]
    pub router_w: Vec<f32>,    // [E, D]
    pub router_b: Vec<f32>,    // [E]
    pub b_bal: Vec<f32>,       // [E]
    pub w_gu: Vec<f32>,        // [E, 2*routed_hidden, D]
    pub w_d: Vec<f32>,         // [E, D, routed_hidden]
}

/// Weights for one cortical layer in the 8-layer DCPA-Flat v3.3 stack.
pub enum FlatCorticalLayerWeights {
    /// L1 (layer 0): optional W=128 causal local window over `f_stem` + MoE.
    PreG1 {
        window: Option<PreG1WindowWeights>,
        moe: FlatMoEWeights,
    },
    /// L2-L3, L5, L7-L8 (and L4/L6 when window layers off): pure K=8 graph cross-attention + MoE.
    GraphCrossAttn {
        norm_attn: Vec<f32>,       // [D]
        w_q: Vec<f32>,             // [D, D]
        w_reg_bias: Vec<f32>,      // [H*K, d_l]
        w_o: Vec<f32>,             // [D, D]
        ior: Option<IorWeights>,
        moe: FlatMoEWeights,
    },
    /// L4 / L6 when `use_local_window_layers`: unified `(W ∪ K ∪ sink)` single-softmax + MoE.
    UnifiedWindowGraph {
        norm_attn: Vec<f32>,  // [D]
        w_q: Vec<f32>,        // [D, D]
        w_uk: Vec<f32>,       // [D, ring_dim]
        w_uv: Vec<f32>,       // [D, ring_dim]
        w_reg_bias: Vec<f32>, // [H*K, d_l]
        w_ior_q: Vec<f32>,    // [D, D]
        ior_strength: f32,
        b_sink: Vec<f32>,     // [H]
        w_o: Vec<f32>,        // [D, D]
        moe: FlatMoEWeights,
    },
}

pub struct PreG1WindowWeights {
    pub norm_attn: Vec<f32>, // [D]
    pub w_q: Vec<f32>,       // [D, D]
    pub w_uk: Vec<f32>,      // [D, d_s]
    pub w_uv: Vec<f32>,      // [D, d_s]
    pub w_o: Vec<f32>,       // [D, D]
}

pub struct IorWeights {
    pub w_ior_q: Vec<f32>, // [D, D]
    pub ior_strength: f32,
}

/// Unified hot-mountable Additive Dead-Zone plugin pack (`.dcpa_expert.safetensors`).
#[derive(Debug, Clone)]
pub struct PluginExpertPackRs {
    pub pack_id: String,
    pub n_layers: usize,
    pub experts_per_layer: usize,
    pub d_concept: usize,
    pub hidden_dim: usize,
    /// Unit-L2 normalized prototype keys `[n_layers, E_p, D]`.
    pub proto_keys: Vec<f32>,
    /// Dead-zone thresholds `[n_layers, E_p]`, each `>= plugin_tau_min`.
    pub tau: Vec<f32>,
    /// Positive tanh gains `[n_layers, E_p]`.
    pub alpha: Vec<f32>,
    /// SwiGLU gate+up weights `[n_layers, E_p, 2*H_p, D]`.
    pub w_gu: Vec<f32>,
    /// SwiGLU down weights `[n_layers, E_p, D, H_p]`.
    pub w_d: Vec<f32>,
}

impl PluginExpertPackRs {
    /// Normalize prototype keys in-place to unit L2 norm matching `F.normalize(..., dim=-1, eps=eps)`.
    pub fn normalize_proto_keys(&mut self, eps: f32) {
        let d = self.d_concept;
        let total = self.n_layers * self.experts_per_layer;
        for idx in 0..total {
            let row = &mut self.proto_keys[idx * d..(idx + 1) * d];
            let norm = sqrtf(dot_f32(row, row));
            let denom = if norm > eps { norm } else { eps };
            let inv = 1.0 / denom;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    pub fn validate(&self, n_layers: usize, d_concept: usize, tau_min: f32) -> Result<(), &'static str> {
        if self.n_layers != n_layers || self.d_concept != d_concept {
            return Err("PluginExpertPackRs layer count or d_concept mismatch");
        }
        let ep = self.experts_per_layer;
        let hp = self.hidden_dim;
        if ep == 0 || hp == 0 {
            return Err("PluginExpertPackRs experts_per_layer and hidden_dim must be > 0");
        }
        if self.proto_keys.len() != n_layers * ep * d_concept {
            return Err("PluginExpertPackRs proto_keys length mismatch");
        }
        if self.tau.len() != n_layers * ep || self.alpha.len() != n_layers * ep {
            return Err("PluginExpertPackRs tau/alpha length mismatch");
        }
        if self.w_gu.len() != n_layers * ep * (2 * hp) * d_concept {
            return Err("PluginExpertPackRs w_gu length mismatch");
        }
        if self.w_d.len() != n_layers * ep * d_concept * hp {
            return Err("PluginExpertPackRs w_d length mismatch");
        }
        if tau_min > 0.0 {
            for &t in &self.tau {
                if t < tau_min - 1e-6 {
                    return Err("PluginExpertPackRs tau fell below plugin_tau_min");
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "std")]
    pub fn from_safetensors(pack_id: &str, bytes: &[u8]) -> Result<Self, String> {
        let meta_pack_id = extract_safetensors_meta_string(bytes, "pack_id");
        let resolved_id = match meta_pack_id {
            Some(s) if !s.is_empty() => s,
            _ => pack_id.to_string(),
        };

        let mut map = parse_safetensors(bytes)?;
        let pk_t = map
            .remove("proto_keys")
            .ok_or_else(|| "missing tensor 'proto_keys'".to_string())?;
        if pk_t.shape.len() != 3 {
            return Err(format!("proto_keys must be 3D, got {:?}", pk_t.shape));
        }
        let n_layers = pk_t.shape[0];
        let experts_per_layer = pk_t.shape[1];
        let d_concept = pk_t.shape[2];

        let tau = take_vec(&mut map, "tau")?;
        let alpha = take_vec(&mut map, "alpha")?;
        let w_gu_t = map
            .remove("w_gu")
            .ok_or_else(|| "missing tensor 'w_gu'".to_string())?;
        if w_gu_t.shape.len() != 4 {
            return Err(format!("w_gu must be 4D, got {:?}", w_gu_t.shape));
        }
        let hidden_dim = w_gu_t.shape[2] / 2;
        let w_d = take_vec(&mut map, "w_d")?;

        let mut pack = Self {
            pack_id: resolved_id,
            n_layers,
            experts_per_layer,
            d_concept,
            hidden_dim,
            proto_keys: pk_t.data_f32,
            tau,
            alpha,
            w_gu: w_gu_t.data_f32,
            w_d,
        };
        pack.validate(n_layers, d_concept, 0.0)
            .map_err(|e| e.to_string())?;
        pack.normalize_proto_keys(1e-6);
        Ok(pack)
    }
}

/// All read-only slow weights and frozen orthogonal buffers for DCPA-Flat v3.3.
pub struct DcpaFlatWeights {
    pub cfg: DcpaFlatConfig,

    // Module 1: SGSCFlat (embed & w_out tied with PMDFlat)
    pub sgsc_embed: Vec<f32>,      // [256, d_s]
    pub sgsc_w_out: Vec<f32>,      // [256, d_s]
    pub w_out_row_norm: Vec<f32>,  // [256, d_s] pre-normalized w_out rows (L2 eps=1e-6)
    pub sgsc_log_temp: f32,
    pub sgsc_lang_embed: Vec<f32>, // [num_lang, 32]
    pub sgsc_style_embed: Vec<f32>,// [num_style, 32]
    pub sgsc_norm_lang: Vec<f32>,  // [32]
    pub sgsc_norm_style: Vec<f32>, // [32]
    pub sgsc_h0: Vec<f32>,         // [d_s]
    pub sgsc_w_alpha_w: Vec<f32>,  // [d_s, d_s]
    pub sgsc_w_alpha_b: Vec<f32>,  // [d_s]
    pub sgsc_w_u: Vec<f32>,        // [d_s, d_s]
    pub sgsc_conv_w: Vec<f32>,     // [3 * d_s] doc-masked causal k=3 depthwise conv
    pub sgsc_theta_raw: f32,
    pub sgsc_w_q: Vec<f32>,        // [d_s, d_s]
    pub sgsc_w_k: Vec<f32>,        // [d_s, d_s]
    pub sgsc_w_f: Vec<f32>,        // [d_s, d_s]
    pub sgsc_norm_f: Vec<f32>,     // [d_s]
    pub sgsc_w_res: Vec<f32>,      // [D, d_s]
    pub sgsc_w_sem: Vec<f32>,      // [D, d_s]
    pub sgsc_norm_c: Vec<f32>,     // [D]

    // Module 2: Dual contractive concept-graph carriers
    pub carrier_g1: FlatCarrierWeights,
    pub carrier_g2: FlatCarrierWeights,

    // Module 3: 8-layer Cortical Stack
    pub kv_g1: FlatGraphKvWeights,
    pub kv_g2: FlatGraphKvWeights,
    pub cortical_layers: Vec<FlatCorticalLayerWeights>,
    pub norm_l3_out: Vec<f32>,            // [D]
    pub w_sem_down: Option<Vec<f32>>,     // [d_sem_ring, D] when !share_stem_ring
    pub norm_sem_ring: Option<Vec<f32>>,  // [d_sem_ring] when !share_stem_ring
    pub w_reg_pred: Vec<f32>,             // [D, d_l]
    pub norm_l6_out: Vec<f32>,            // [D]
    pub w_unc_w: Vec<f32>,                // [D]
    pub w_unc_b: f32,
    pub tau_think_init: f32,

    // Module 4: HippocampusFlat + DualMorphemeWhitenedDG
    pub dg_q_rot: Vec<f32>,       // [d_s, d_s]
    pub dg_r_lex: Vec<f32>,       // [d_h, d_s]
    pub dg_r_sem: Vec<f32>,       // [d_h, D]
    pub dg_b_homeo: Vec<f32>,     // [d_h]
    pub dg_b_homeo_sem: Vec<f32>, // [d_h]
    pub dg_w_whiten_1_w: Vec<f32>,// [dg_whiten_hidden, d_s]
    pub dg_w_whiten_1_b: Vec<f32>,// [dg_whiten_hidden]
    pub dg_w_whiten_2_w: Vec<f32>,// [d_s, dg_whiten_hidden]
    pub dg_w_whiten_2_b: Vec<f32>,// [d_s]
    pub hip_alpha_1: f32,
    pub hip_alpha_2: f32,
    pub hip_theta_i: f32,
    pub hip_theta_n: f32,
    pub hip_alpha_q: f32,
    pub hip_gamma_e: f32,
    pub hip_theta_q: f32,
    pub hip_alpha_out: f32,
    pub hip_theta_out: f32,

    // Module 4.5: Pure f_stem HolographicCopyRing
    pub copy_norm_q: Vec<f32>,  // [D]
    pub copy_w_q: Vec<f32>,     // [H*d_k, D]
    pub copy_w_k: Vec<f32>,     // [H*d_k, d_s]
    pub copy_b_sink: Vec<f32>,  // [H]

    // Top-level causal seed & stem confluence heads
    pub bos_concept: Vec<f32>,   // [D]
    pub w_lex_slow: Vec<f32>,    // [d_s, D]
    pub w_delta_syn_w: Vec<f32>, // [d_s, D + d_l]
    pub w_delta_syn_b: Vec<f32>, // [d_s]

    // Module 5: PMDFlat
    pub pmd_log_temp: f32,
    pub pmd_w_syn_reg_w: Vec<f32>, // [d_s, d_s + d_l]
    pub pmd_w_syn_reg_b: Vec<f32>, // [d_s]
    pub pmd_u_demod: Vec<f32>,     // [d_s, d_s]
    pub pmd_w_chi_w: Vec<f32>,     // [3*d_s]
    pub pmd_w_chi_b: f32,
    pub pmd_w_act_w: Vec<f32>,     // [3, d_s + d_l]
    pub pmd_w_act_b: Vec<f32>,     // [3]
    pub pmd_w_tag_w: Vec<f32>,     // [4, d_s]
    pub pmd_w_tag_b: Vec<f32>,     // [4]
    pub pmd_w_st: Vec<f32>,        // [d_s, 8]
    pub pmd_m0: Vec<f32>,          // [d_s]
    pub pmd_w_alpha_mot_w: Vec<f32>, // [d_s, d_s]
    pub pmd_w_alpha_mot_b: Vec<f32>, // [d_s]
    pub pmd_w_m: Vec<f32>,         // [d_s, d_s]
    pub pmd_w_e_init_w: Vec<f32>,  // [2*d_s + d_l]
    pub pmd_w_e_init_b: f32,
    pub pmd_w_drain_w: Vec<f32>,   // [2*d_s]
    pub pmd_w_drain_b: f32,
}

/// Precompute row-L2-normalized `w_out` (`w_out[r] / sqrt(||w_out[r]||^2 + 1e-6)`).
pub fn normalize_w_out_rows(w_out: &[f32], vocab_size: usize, d_s: usize) -> Vec<f32> {
    let mut out = alloc::vec![0.0f32; vocab_size * d_s];
    for r in 0..vocab_size {
        let src = &w_out[r * d_s..(r + 1) * d_s];
        let inv = 1.0 / sqrtf(dot_f32(src, src) + 1e-6);
        let dst = &mut out[r * d_s..(r + 1) * d_s];
        for c in 0..d_s {
            dst[c] = src[c] * inv;
        }
    }
    out
}

#[cfg(feature = "std")]
fn extract_safetensors_meta_string(bytes: &[u8], key: &str) -> Option<String> {
    if bytes.len() < 8 {
        return None;
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
    if 8 + header_len > bytes.len() {
        return None;
    }
    let header_str = core::str::from_utf8(&bytes[8..8 + header_len]).ok()?;
    let header_json: serde_json::Value = serde_json::from_str(header_str).ok()?;
    header_json
        .get("__metadata__")?
        .get(key)?
        .as_str()
        .map(|s| s.to_string())
}

#[cfg(feature = "std")]
fn take_vec(map: &mut BTreeMap<String, RawTensor>, k: &str) -> Result<Vec<f32>, String> {
    map.remove(k)
        .map(|t| t.data_f32)
        .ok_or_else(|| format!("missing tensor '{}'", k))
}

#[cfg(feature = "std")]
fn take_scalar(map: &mut BTreeMap<String, RawTensor>, k: &str) -> Result<f32, String> {
    let v = map
        .remove(k)
        .map(|t| t.data_f32)
        .ok_or_else(|| format!("missing scalar tensor '{}'", k))?;
    v.first()
        .copied()
        .ok_or_else(|| format!("empty scalar tensor '{}'", k))
}

#[cfg(feature = "std")]
fn load_carrier(map: &mut BTreeMap<String, RawTensor>, prefix: &str) -> Result<FlatCarrierWeights, String> {
    Ok(FlatCarrierWeights {
        init_slots_raw: take_vec(map, &format!("{}init_slots_raw", prefix))?,
        init_edges_raw: take_vec(map, &format!("{}init_edges_raw", prefix))?,
        norm_in: take_vec(map, &format!("{}norm_in.weight", prefix))?,
        w_in: take_vec(map, &format!("{}w_in.weight", prefix))?,
        w_lam_w: take_vec(map, &format!("{}w_lam.weight", prefix))?,
        w_lam_b: take_vec(map, &format!("{}w_lam.bias", prefix))?,
        assim_scale: take_scalar(map, &format!("{}assim_scale", prefix))?,
        assim_threshold: take_scalar(map, &format!("{}assim_threshold", prefix))?,
        lam_sim_gain: take_scalar(map, &format!("{}lam_sim_gain", prefix))?,
        w_edge_in_src: take_vec(map, &format!("{}w_edge_in_src.weight", prefix))?,
        w_edge_in_dst: take_vec(map, &format!("{}w_edge_in_dst.weight", prefix))?,
        w_edge_slot_src: take_vec(map, &format!("{}w_edge_slot_src.weight", prefix))?,
        w_edge_slot_dst: take_vec(map, &format!("{}w_edge_slot_dst.weight", prefix))?,
        edge_write_logit: take_scalar(map, &format!("{}edge_write_logit", prefix))?,
        w_pair_si: take_vec(map, &format!("{}w_pair_si.weight", prefix))?,
        w_pair_sj: take_vec(map, &format!("{}w_pair_sj.weight", prefix))?,
        w_pair_edge: take_vec(map, &format!("{}w_pair_edge.weight", prefix))?,
        w_pair_rho_i: take_vec(map, &format!("{}w_pair_rho_i", prefix))?,
        w_pair_rho_j: take_vec(map, &format!("{}w_pair_rho_j", prefix))?,
        pair_bias: take_vec(map, &format!("{}pair_bias", prefix))?,
        w_pair_score_w: take_vec(map, &format!("{}w_pair_score.weight", prefix))?,
        w_pair_score_b: take_scalar(map, &format!("{}w_pair_score.bias", prefix))?,
        pair_cos_gain: take_scalar(map, &format!("{}pair_cos_gain", prefix))?,
        w_merge_beta_w: take_vec(map, &format!("{}w_merge_beta.weight", prefix))?,
        w_merge_beta_b: take_scalar(map, &format!("{}w_merge_beta.bias", prefix))?,
        w_merge_delta: take_vec(map, &format!("{}w_merge_delta.weight", prefix))?,
    })
}

#[cfg(feature = "std")]
fn load_graph_kv(map: &mut BTreeMap<String, RawTensor>, prefix: &str) -> Result<FlatGraphKvWeights, String> {
    Ok(FlatGraphKvWeights {
        norm_slot: take_vec(map, &format!("{}norm_slot.weight", prefix))?,
        w_edge_in: take_vec(map, &format!("{}w_edge_in.weight", prefix))?,
        w_edge_out: take_vec(map, &format!("{}w_edge_out.weight", prefix))?,
        w_gk: take_vec(map, &format!("{}w_gk.weight", prefix))?,
        w_gv: take_vec(map, &format!("{}w_gv.weight", prefix))?,
        w_rel_bias: take_vec(map, &format!("{}w_rel_bias.weight", prefix))?,
    })
}

#[cfg(feature = "std")]
fn load_moe(map: &mut BTreeMap<String, RawTensor>, prefix: &str) -> Result<FlatMoEWeights, String> {
    Ok(FlatMoEWeights {
        norm: take_vec(map, &format!("{}norm.weight", prefix))?,
        shared_w_gu: take_vec(map, &format!("{}shared_w_gu.weight", prefix))?,
        shared_w_d: take_vec(map, &format!("{}shared_w_d.weight", prefix))?,
        router_w: take_vec(map, &format!("{}router.weight", prefix))?,
        router_b: take_vec(map, &format!("{}router.bias", prefix))?,
        b_bal: take_vec(map, &format!("{}b_bal", prefix))?,
        w_gu: take_vec(map, &format!("{}w_gu", prefix))?,
        w_d: take_vec(map, &format!("{}w_d", prefix))?,
    })
}

#[cfg(feature = "std")]
impl DcpaFlatWeights {
    pub fn from_safetensors(cfg: DcpaFlatConfig, bytes: &[u8]) -> Result<Self, String> {
        cfg.validate().map_err(|e| e.to_string())?;
        let mut map = parse_safetensors(bytes)?;

        let sgsc_w_out = take_vec(&mut map, "sgsc.w_out.weight")?;
        let w_out_row_norm = normalize_w_out_rows(&sgsc_w_out, cfg.vocab_size, cfg.d_s);

        let carrier_g1 = load_carrier(&mut map, "carrier_g1.")?;
        let carrier_g2 = load_carrier(&mut map, "carrier_g2.")?;
        let kv_g1 = load_graph_kv(&mut map, "cortical.kv_g1.")?;
        let kv_g2 = load_graph_kv(&mut map, "cortical.kv_g2.")?;

        let pred_start = cfg.n_pre_g1_layers + cfg.n_post_g1_layers;
        let pred_end = pred_start + cfg.n_predict_full_layers;
        let mut cortical_layers = Vec::with_capacity(cfg.n_layers);

        for idx in 0..cfg.n_layers {
            let p = format!("cortical.layers.{}.", idx);
            let moe = load_moe(&mut map, &format!("{}moe.", p))?;
            if idx == 0 {
                let window = if cfg.is_window_layer(0) {
                    Some(PreG1WindowWeights {
                        norm_attn: take_vec(&mut map, &format!("{}norm_attn.weight", p))?,
                        w_q: take_vec(&mut map, &format!("{}w_q.weight", p))?,
                        w_uk: take_vec(&mut map, &format!("{}w_uk.weight", p))?,
                        w_uv: take_vec(&mut map, &format!("{}w_uv.weight", p))?,
                        w_o: take_vec(&mut map, &format!("{}w_o.weight", p))?,
                    })
                } else {
                    None
                };
                cortical_layers.push(FlatCorticalLayerWeights::PreG1 { window, moe });
            } else if idx < pred_start {
                cortical_layers.push(FlatCorticalLayerWeights::GraphCrossAttn {
                    norm_attn: take_vec(&mut map, &format!("{}norm_attn.weight", p))?,
                    w_q: take_vec(&mut map, &format!("{}w_q.weight", p))?,
                    w_reg_bias: take_vec(&mut map, &format!("{}w_reg_bias.weight", p))?,
                    w_o: take_vec(&mut map, &format!("{}w_o.weight", p))?,
                    ior: None,
                    moe,
                });
            } else if idx < pred_end && cfg.is_window_layer(idx) {
                cortical_layers.push(FlatCorticalLayerWeights::UnifiedWindowGraph {
                    norm_attn: take_vec(&mut map, &format!("{}norm_attn.weight", p))?,
                    w_q: take_vec(&mut map, &format!("{}w_q.weight", p))?,
                    w_uk: take_vec(&mut map, &format!("{}w_uk.weight", p))?,
                    w_uv: take_vec(&mut map, &format!("{}w_uv.weight", p))?,
                    w_reg_bias: take_vec(&mut map, &format!("{}w_reg_bias.weight", p))?,
                    w_ior_q: take_vec(&mut map, &format!("{}w_ior_q.weight", p))?,
                    ior_strength: take_scalar(&mut map, &format!("{}ior_strength", p))?,
                    b_sink: take_vec(&mut map, &format!("{}b_sink", p))?,
                    w_o: take_vec(&mut map, &format!("{}w_o.weight", p))?,
                    moe,
                });
            } else {
                cortical_layers.push(FlatCorticalLayerWeights::GraphCrossAttn {
                    norm_attn: take_vec(&mut map, &format!("{}norm_attn.weight", p))?,
                    w_q: take_vec(&mut map, &format!("{}w_q.weight", p))?,
                    w_reg_bias: take_vec(&mut map, &format!("{}w_reg_bias.weight", p))?,
                    w_o: take_vec(&mut map, &format!("{}w_o.weight", p))?,
                    ior: Some(IorWeights {
                        w_ior_q: take_vec(&mut map, &format!("{}w_ior_q.weight", p))?,
                        ior_strength: take_scalar(&mut map, &format!("{}ior_strength", p))?,
                    }),
                    moe,
                });
            }
        }

        let (w_sem_down, norm_sem_ring) = if !cfg.share_stem_ring {
            (
                Some(take_vec(&mut map, "cortical.w_sem_down.weight")?),
                Some(take_vec(&mut map, "cortical.norm_sem_ring.weight")?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            cfg,
            sgsc_embed: take_vec(&mut map, "sgsc.embed.weight")?,
            sgsc_w_out,
            w_out_row_norm,
            sgsc_log_temp: take_scalar(&mut map, "sgsc.log_temp_sgsc")?,
            sgsc_lang_embed: take_vec(&mut map, "sgsc.lang_embed.weight")?,
            sgsc_style_embed: take_vec(&mut map, "sgsc.style_embed.weight")?,
            sgsc_norm_lang: take_vec(&mut map, "sgsc.norm_lang.weight")?,
            sgsc_norm_style: take_vec(&mut map, "sgsc.norm_style.weight")?,
            sgsc_h0: take_vec(&mut map, "sgsc.h0")?,
            sgsc_w_alpha_w: take_vec(&mut map, "sgsc.w_alpha.weight")?,
            sgsc_w_alpha_b: take_vec(&mut map, "sgsc.w_alpha.bias")?,
            sgsc_w_u: take_vec(&mut map, "sgsc.w_u.weight")?,
            sgsc_conv_w: take_vec(&mut map, "sgsc.conv_w")?,
            sgsc_theta_raw: take_scalar(&mut map, "sgsc.theta_raw")?,
            sgsc_w_q: take_vec(&mut map, "sgsc.w_q.weight")?,
            sgsc_w_k: take_vec(&mut map, "sgsc.w_k.weight")?,
            sgsc_w_f: take_vec(&mut map, "sgsc.w_f.weight")?,
            sgsc_norm_f: take_vec(&mut map, "sgsc.norm_f.weight")?,
            sgsc_w_res: take_vec(&mut map, "sgsc.w_res.weight")?,
            sgsc_w_sem: take_vec(&mut map, "sgsc.w_sem.weight")?,
            sgsc_norm_c: take_vec(&mut map, "sgsc.norm_c.weight")?,

            carrier_g1,
            carrier_g2,

            kv_g1,
            kv_g2,
            cortical_layers,
            norm_l3_out: take_vec(&mut map, "cortical.norm_l3_out.weight")?,
            w_sem_down,
            norm_sem_ring,
            w_reg_pred: take_vec(&mut map, "cortical.w_reg_pred.weight")?,
            norm_l6_out: take_vec(&mut map, "cortical.norm_l6_out.weight")?,
            w_unc_w: take_vec(&mut map, "cortical.w_unc.weight")?,
            w_unc_b: take_scalar(&mut map, "cortical.w_unc.bias")?,
            tau_think_init: take_scalar(&mut map, "cortical.tau_think")?,

            dg_q_rot: take_vec(&mut map, "hippocampus.dg.q_rot")?,
            dg_r_lex: take_vec(&mut map, "hippocampus.dg.r_lex")?,
            dg_r_sem: take_vec(&mut map, "hippocampus.dg.r_sem")?,
            dg_b_homeo: take_vec(&mut map, "hippocampus.dg.b_homeo")?,
            dg_b_homeo_sem: take_vec(&mut map, "hippocampus.dg.b_homeo_sem")?,
            dg_w_whiten_1_w: take_vec(&mut map, "hippocampus.dg.w_whiten_1.weight")?,
            dg_w_whiten_1_b: take_vec(&mut map, "hippocampus.dg.w_whiten_1.bias")?,
            dg_w_whiten_2_w: take_vec(&mut map, "hippocampus.dg.w_whiten_2.weight")?,
            dg_w_whiten_2_b: take_vec(&mut map, "hippocampus.dg.w_whiten_2.bias")?,
            hip_alpha_1: take_scalar(&mut map, "hippocampus.alpha_1")?,
            hip_alpha_2: take_scalar(&mut map, "hippocampus.alpha_2")?,
            hip_theta_i: take_scalar(&mut map, "hippocampus.theta_i")?,
            hip_theta_n: take_scalar(&mut map, "hippocampus.theta_n")?,
            hip_alpha_q: take_scalar(&mut map, "hippocampus.alpha_q")?,
            hip_gamma_e: take_scalar(&mut map, "hippocampus.gamma_e")?,
            hip_theta_q: take_scalar(&mut map, "hippocampus.theta_q")?,
            hip_alpha_out: take_scalar(&mut map, "hippocampus.alpha_out")?,
            hip_theta_out: take_scalar(&mut map, "hippocampus.theta_out")?,

            copy_norm_q: take_vec(&mut map, "copy_ring.norm_q.weight")?,
            copy_w_q: take_vec(&mut map, "copy_ring.w_q.weight")?,
            copy_w_k: take_vec(&mut map, "copy_ring.w_k.weight")?,
            copy_b_sink: take_vec(&mut map, "copy_ring.b_sink")?,

            bos_concept: take_vec(&mut map, "bos_concept")?,
            w_lex_slow: take_vec(&mut map, "w_lex_slow.weight")?,
            w_delta_syn_w: take_vec(&mut map, "w_delta_syn.weight")?,
            w_delta_syn_b: take_vec(&mut map, "w_delta_syn.bias")?,

            pmd_log_temp: take_scalar(&mut map, "pmd.log_temp_pmd")?,
            pmd_w_syn_reg_w: take_vec(&mut map, "pmd.w_syn_reg.weight")?,
            pmd_w_syn_reg_b: take_vec(&mut map, "pmd.w_syn_reg.bias")?,
            pmd_u_demod: take_vec(&mut map, "pmd.u_demod.weight")?,
            pmd_w_chi_w: take_vec(&mut map, "pmd.w_chi.weight")?,
            pmd_w_chi_b: take_scalar(&mut map, "pmd.w_chi.bias")?,
            pmd_w_act_w: take_vec(&mut map, "pmd.w_act.weight")?,
            pmd_w_act_b: take_vec(&mut map, "pmd.w_act.bias")?,
            pmd_w_tag_w: take_vec(&mut map, "pmd.w_tag.weight")?,
            pmd_w_tag_b: take_vec(&mut map, "pmd.w_tag.bias")?,
            pmd_w_st: take_vec(&mut map, "pmd.w_st.weight")?,
            pmd_m0: take_vec(&mut map, "pmd.m0")?,
            pmd_w_alpha_mot_w: take_vec(&mut map, "pmd.w_alpha_mot.weight")?,
            pmd_w_alpha_mot_b: take_vec(&mut map, "pmd.w_alpha_mot.bias")?,
            pmd_w_m: take_vec(&mut map, "pmd.w_m.weight")?,
            pmd_w_e_init_w: take_vec(&mut map, "pmd.w_e_init.weight")?,
            pmd_w_e_init_b: take_scalar(&mut map, "pmd.w_e_init.bias")?,
            pmd_w_drain_w: take_vec(&mut map, "pmd.w_drain.weight")?,
            pmd_w_drain_b: take_scalar(&mut map, "pmd.w_drain.bias")?,
        })
    }
}
