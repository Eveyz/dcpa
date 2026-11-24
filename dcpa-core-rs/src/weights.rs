//! Pure static Rust safetensors parser & flat-named slow weight container.
//!
//! Zero Python, zero LibTorch, zero external ML runtime dependencies!
//! Parses the standard safetensors binary layout:
//!   [8-byte little-endian header_len] [UTF-8 JSON header] [contiguous little-endian tensor bytes]

#[cfg(feature = "std")]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use alloc::format;
#[cfg(feature = "std")]
use alloc::string::ToString;
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::vec;
use alloc::vec::Vec;

use crate::bf16::{AlignedVec, Bf16};
use crate::config::DcpaConfig;
use crate::simd::{linear_bf16_f32, linear_f32};

#[derive(Debug, Clone)]
pub struct RawTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_f32: Vec<f32>,
    pub data_u8: Vec<u8>,
    pub data_i32: Vec<i32>,
    pub data_i64: Vec<i64>,
}

/// Parse a raw `.safetensors` byte buffer into a map of named tensors (supports F32, BF16, U8, I32, I64).
#[cfg(feature = "std")]
pub fn parse_safetensors(bytes: &[u8]) -> Result<BTreeMap<String, RawTensor>, String> {
    if bytes.len() < 8 {
        return Err("safetensors file too short".to_string());
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        return Err("invalid safetensors header length".to_string());
    }
    let header_str = core::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|_| "non-UTF8 safetensors header".to_string())?;
    let header_json: serde_json::Value =
        serde_json::from_str(header_str).map_err(|e| format!("header JSON error: {}", e))?;
    let obj = header_json
        .as_object()
        .ok_or_else(|| "header must be JSON object".to_string())?;

    let data_base = 8 + header_len;
    let mut out = BTreeMap::new();

    for (name, val) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = val["dtype"]
            .as_str()
            .ok_or_else(|| format!("missing dtype for {}", name))?
            .to_string();
        let shape: Vec<usize> = val["shape"]
            .as_array()
            .ok_or_else(|| format!("missing shape for {}", name))?
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let offsets = val["data_offsets"]
            .as_array()
            .ok_or_else(|| format!("missing data_offsets for {}", name))?;
        let start = data_base + offsets[0].as_u64().unwrap() as usize;
        let end = data_base + offsets[1].as_u64().unwrap() as usize;
        let raw = &bytes[start..end];

        let mut rt = RawTensor {
            dtype: dtype.clone(),
            shape,
            data_f32: Vec::new(),
            data_u8: Vec::new(),
            data_i32: Vec::new(),
            data_i64: Vec::new(),
        };

        match dtype.as_str() {
            "F32" => {
                let n = raw.len() / 4;
                rt.data_f32 = Vec::with_capacity(n);
                for i in 0..n {
                    rt.data_f32.push(f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
                }
            }
            "BF16" => {
                let n = raw.len() / 2;
                rt.data_f32 = Vec::with_capacity(n);
                for i in 0..n {
                    let bits = u16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into().unwrap());
                    rt.data_f32.push(Bf16(bits).to_f32());
                }
            }
            "U8" => {
                rt.data_u8 = raw.to_vec();
            }
            "I32" => {
                let n = raw.len() / 4;
                rt.data_i32 = Vec::with_capacity(n);
                for i in 0..n {
                    rt.data_i32.push(i32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
                }
            }
            "I64" => {
                let n = raw.len() / 8;
                rt.data_i64 = Vec::with_capacity(n);
                for i in 0..n {
                    rt.data_i64.push(i64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()));
                }
            }
            other => {
                return Err(format!("unsupported safetensors dtype {} for {}", other, name));
            }
        }
        out.insert(name.clone(), rt);
    }

    Ok(out)
}

/// Write a flat map of f32 tensors into standard `.safetensors` format (used by dcpa-sleep-daemon).
#[cfg(feature = "std")]
pub fn serialize_safetensors_f32(tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut body: Vec<u8> = Vec::new();

    for (name, (shape, data)) in tensors {
        let start = body.len();
        for &v in data {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let end = body.len();
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".to_string(), serde_json::Value::String("F32".to_string()));
        entry.insert(
            "shape".to_string(),
            serde_json::Value::Array(shape.iter().map(|&d| serde_json::json!(d)).collect()),
        );
        entry.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(vec![serde_json::json!(start), serde_json::json!(end)]),
        );
        header.insert(name.clone(), serde_json::Value::Object(entry));
    }

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + body.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&body);
    out
}

/// Dual-precision slow expert matrix storage:
/// - `F32`: exact float32 parity mode (Gate 6a `< 1e-5` cross-stack verification).
/// - `Bf16`: 64-byte cache-line aligned compact bfloat16 mode (halves memory bandwidth & footprint!).
pub enum ExpertMatrix {
    F32(Vec<f32>),
    Bf16(AlignedVec<Bf16>),
}

impl ExpertMatrix {
    pub fn from_f32_vec(v: Vec<f32>) -> Self {
        Self::F32(v)
    }

    pub fn pack_bf16(&mut self) {
        if let Self::F32(src) = self {
            let mut dst = AlignedVec::<Bf16>::zeroed(src.len());
            let dst_slice = dst.as_mut_slice();
            for (i, &val) in src.iter().enumerate() {
                dst_slice[i] = Bf16::from_f32(val);
            }
            *self = Self::Bf16(dst);
        }
    }

    #[inline]
    pub fn linear_no_bias(&self, x: &[f32], out: &mut [f32]) {
        match self {
            Self::F32(w) => linear_f32(x, w, None, out),
            Self::Bf16(w) => linear_bf16_f32(x, w.as_slice(), None, out),
        }
    }

    pub fn byte_footprint(&self) -> usize {
        match self {
            Self::F32(w) => w.len() * 4,
            Self::Bf16(w) => w.len() * 2,
        }
    }
}

pub struct ExpertWeights {
    pub w_gate1: ExpertMatrix,
    pub w_up1: ExpertMatrix,
    pub w_down1: ExpertMatrix,
    pub w_gate2: ExpertMatrix,
    pub w_up2: ExpertMatrix,
    pub w_down2: ExpertMatrix,
}

impl ExpertWeights {
    pub fn pack_bf16(&mut self) {
        self.w_gate1.pack_bf16();
        self.w_up1.pack_bf16();
        self.w_down1.pack_bf16();
        self.w_gate2.pack_bf16();
        self.w_up2.pack_bf16();
        self.w_down2.pack_bf16();
    }
}

/// All read-only slow weights and frozen orthogonal buffers of DCPA.
pub struct DcpaWeights {
    pub cfg: DcpaConfig,

    // Module 1: SGSC (Embed & W_out shared with PMD)
    pub sgsc_embed: Vec<f32>,      // [256, d_s]
    pub sgsc_w_out: Vec<f32>,      // [256, d_s]
    pub sgsc_h0: Vec<f32>,         // [d_s]
    pub sgsc_w_alpha: Vec<f32>,    // [d_s, d_s]
    pub sgsc_b_alpha: Vec<f32>,    // [d_s]
    pub sgsc_w_u: Vec<f32>,        // [d_s, d_s]
    pub sgsc_theta_raw: f32,
    pub sgsc_w_q: Vec<f32>,        // [d_s, d_s]
    pub sgsc_w_k: Vec<f32>,        // [d_s, d_s]
    pub sgsc_w_f: Vec<f32>,        // [d_s, d_s]
    pub sgsc_norm_f: Vec<f32>,     // [d_s]
    pub sgsc_w_res: Vec<f32>,      // [D, d_s]
    pub sgsc_w_sem: Vec<f32>,      // [D, d_s]
    pub sgsc_norm_c: Vec<f32>,     // [D]

    // Module 2: BCG-PC
    pub bcg_s0: Vec<f32>,          // [K, D]
    pub bcg_a0: Vec<f32>,          // [K, K, d_r]
    pub bcg_omega0: Vec<f32>,      // [K]
    pub bcg_c0: Vec<f32>,          // [D]
    pub bcg_w_e: Vec<f32>,         // [D, d_r]
    pub bcg_f_pred_1_w: Vec<f32>,  // [D, D]
    pub bcg_f_pred_1_b: Vec<f32>,  // [D]
    pub bcg_f_pred_2_w: Vec<f32>,  // [D, D]
    pub bcg_f_pred_2_b: Vec<f32>,  // [D]
    pub bcg_f_prec_1_w: Vec<f32>,  // [D, D]
    pub bcg_f_prec_1_b: Vec<f32>,  // [D]
    pub bcg_f_prec_2_w: Vec<f32>,  // [D, D]
    pub bcg_f_prec_2_b: Vec<f32>,  // [D]
    pub bcg_w_fast: Vec<f32>,      // [D, D]
    pub bcg_w_gen: Vec<f32>,       // [D, D]
    pub bcg_ca_wq: Vec<f32>,       // [D, D]
    pub bcg_ca_wk: Vec<f32>,       // [D, D]
    pub bcg_ca_wv: Vec<f32>,       // [D, D]
    pub bcg_ca_wo: Vec<f32>,       // [D, D]
    pub bcg_moe_router: Vec<f32>,  // [num_experts, D]
    pub bcg_experts: Vec<ExpertWeights>,
    pub bcg_norm_ponder: Vec<f32>, // [D]
    pub bcg_u_src: Vec<f32>,       // [d_r, D]
    pub bcg_u_dst: Vec<f32>,       // [d_r, D]
    pub bcg_w_l: Vec<f32>,         // [d_merge, D]
    pub bcg_w_r: Vec<f32>,         // [d_merge, D]
    pub bcg_w_edge_score: Vec<f32>,// [d_merge, d_r]
    pub bcg_w_merge: Vec<f32>,     // [d_merge]
    pub bcg_w_g_w: Vec<f32>,       // [D, 2*D + d_r]
    pub bcg_w_g_b: Vec<f32>,       // [D]
    pub bcg_w_comp_w: Vec<f32>,    // [D, 2*D]
    pub bcg_w_comp_b: Vec<f32>,    // [D]
    pub bcg_norm_merge: Vec<f32>,  // [D]
    pub bcg_w_omega: Vec<f32>,     // [D]
    pub bcg_kappa_omega: f32,

    // Module 3: Hippocampus frozen projections & slow gates
    pub hip_r_lex: Vec<f32>,       // [d_h, d_s]
    pub hip_r_sem: Vec<f32>,       // [d_h, D]
    pub hip_alpha_1: f32,
    pub hip_alpha_2: f32,
    pub hip_theta_i: f32,
    pub hip_theta_n: f32,
    pub hip_alpha_q: f32,
    pub hip_gamma_e: f32,
    pub hip_theta_q: f32,
    pub hip_alpha_out: f32,
    pub hip_theta_out: f32,
    pub hip_f_prec_cort_w: Vec<f32>, // [D, D]
    pub hip_f_prec_cort_b: Vec<f32>, // [D]
    pub hip_w_lex_slow: Vec<f32>,    // [d_s, D]
    pub hip_w_lang_lex: Vec<f32>,    // [D, d_l]

    // Module 4: PMD (32D lang ⊕ 32D style orthogonal control vector -> 64D)
    pub pmd_lang_embed: Vec<f32>,   // [num_lang, d_l/2]
    pub pmd_style_embed: Vec<f32>,  // [num_style, d_l/2]
    pub pmd_norm_lang: Vec<f32>,    // [d_l/2]
    pub pmd_norm_style: Vec<f32>,   // [d_l/2]
    pub pmd_w_ord: Vec<f32>,        // [d_r, d_l]
    pub pmd_w_coh: Vec<f32>,        // [d_r, d_l]
    pub pmd_w_syn_w: Vec<f32>,      // [d_s, d_r + d_l]
    pub pmd_w_syn_b: Vec<f32>,      // [d_s]
    pub pmd_u_syn: Vec<f32>,        // [d_s, D]
    pub pmd_w_ctx_role: Vec<f32>,   // [d_r, D]
    pub pmd_u_demod: Vec<f32>,      // [d_s, d_s]
    pub pmd_w_chi_w: Vec<f32>,      // [3*d_s]
    pub pmd_w_chi_b: f32,
    pub pmd_w_act_w: Vec<f32>,      // [3, d_s + d_l]
    pub pmd_w_act_b: Vec<f32>,      // [3]
    pub pmd_w_tag_w: Vec<f32>,      // [4, d_s]
    pub pmd_w_tag_b: Vec<f32>,      // [4]
    pub pmd_w_st: Vec<f32>,         // [d_s, 8]
    pub pmd_m0: Vec<f32>,           // [d_s]
    pub pmd_w_alpha_mot_w: Vec<f32>,// [d_s, d_s]
    pub pmd_w_alpha_mot_b: Vec<f32>,// [d_s]
    pub pmd_w_m: Vec<f32>,          // [d_s, d_s]
    pub pmd_w_e_init_w: Vec<f32>,   // [2*d_s + d_l]
    pub pmd_w_e_init_b: f32,
    pub pmd_w_drain_w: Vec<f32>,    // [2*d_s]
    pub pmd_w_drain_b: f32,
}

#[cfg(test)]
impl DcpaWeights {
    pub fn dummy_for_test(cfg: DcpaConfig) -> Self {
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i + seed) % 11) as f32 - 5.0) * 0.04)
                .collect()
        };
        let d_s = cfg.d_s;
        let d = cfg.d_concept;
        let k = cfg.k_slots;
        let d_r = cfg.d_r;
        let d_m = cfg.d_merge;
        let d_h = cfg.d_h;
        let d_l = cfg.d_l;
        let eh = cfg.expert_hidden;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            experts.push(ExpertWeights {
                w_gate1: ExpertMatrix::F32(fill(eh * d, e + 1)),
                w_up1: ExpertMatrix::F32(fill(eh * d, e + 2)),
                w_down1: ExpertMatrix::F32(fill(d * eh, e + 3)),
                w_gate2: ExpertMatrix::F32(fill(eh * d, e + 4)),
                w_up2: ExpertMatrix::F32(fill(eh * d, e + 5)),
                w_down2: ExpertMatrix::F32(fill(d * eh, e + 6)),
            });
        }
        Self {
            sgsc_embed: fill(256 * d_s, 11),
            sgsc_w_out: fill(256 * d_s, 13),
            sgsc_h0: fill(d_s, 17),
            sgsc_w_alpha: fill(d_s * d_s, 19),
            sgsc_b_alpha: fill(d_s, 23),
            sgsc_w_u: fill(d_s * d_s, 29),
            sgsc_theta_raw: 0.5,
            sgsc_w_q: fill(d_s * d_s, 31),
            sgsc_w_k: fill(d_s * d_s, 37),
            sgsc_w_f: fill(d_s * d_s, 41),
            sgsc_norm_f: alloc::vec![1.0f32; d_s],
            sgsc_w_res: fill(d * d_s, 43),
            sgsc_w_sem: fill(d * d_s, 47),
            sgsc_norm_c: alloc::vec![1.0f32; d],
            bcg_s0: fill(k * d, 53),
            bcg_a0: fill(k * k * d_r, 59),
            bcg_omega0: alloc::vec![0.0f32; k],
            bcg_c0: fill(d, 61),
            bcg_w_e: fill(d * d_r, 67),
            bcg_f_pred_1_w: fill(d * d, 71),
            bcg_f_pred_1_b: fill(d, 73),
            bcg_f_pred_2_w: fill(d * d, 79),
            bcg_f_pred_2_b: fill(d, 83),
            bcg_f_prec_1_w: fill(d * d, 89),
            bcg_f_prec_1_b: fill(d, 97),
            bcg_f_prec_2_w: fill(d * d, 101),
            bcg_f_prec_2_b: fill(d, 103),
            bcg_w_fast: fill(d * d, 107),
            bcg_w_gen: fill(d * d, 109),
            bcg_ca_wq: fill(d * d, 113),
            bcg_ca_wk: fill(d * d, 127),
            bcg_ca_wv: fill(d * d, 131),
            bcg_ca_wo: fill(d * d, 137),
            bcg_moe_router: fill(cfg.num_experts * d, 139),
            bcg_experts: experts,
            bcg_norm_ponder: alloc::vec![1.0f32; d],
            bcg_u_src: fill(d_r * d, 149),
            bcg_u_dst: fill(d_r * d, 151),
            bcg_w_l: fill(d_m * d, 157),
            bcg_w_r: fill(d_m * d, 163),
            bcg_w_edge_score: fill(d_m * d_r, 167),
            bcg_w_merge: fill(d_m, 173),
            bcg_w_g_w: fill(d * (2 * d + d_r), 179),
            bcg_w_g_b: fill(d, 181),
            bcg_w_comp_w: fill(d * 2 * d, 191),
            bcg_w_comp_b: fill(d, 193),
            bcg_norm_merge: alloc::vec![1.0f32; d],
            bcg_w_omega: fill(d, 197),
            bcg_kappa_omega: 0.5,
            hip_r_lex: fill(d_h * d_s, 199),
            hip_r_sem: fill(d_h * d, 211),
            hip_alpha_1: 1.0,
            hip_alpha_2: 1.0,
            hip_theta_i: 2.0,
            hip_theta_n: 1.0,
            hip_alpha_q: 1.0,
            hip_gamma_e: 0.5,
            hip_theta_q: 0.5,
            hip_alpha_out: 1.0,
            hip_theta_out: 0.5,
            hip_f_prec_cort_w: fill(d * d, 223),
            hip_f_prec_cort_b: fill(d, 227),
            hip_w_lex_slow: fill(d_s * d, 229),
            hip_w_lang_lex: fill(d * d_l, 233),
            pmd_lang_embed: fill(cfg.num_lang_domains * (d_l / 2), 239),
            pmd_style_embed: fill(cfg.num_style_domains * (d_l / 2), 241),
            pmd_norm_lang: alloc::vec![1.0f32; d_l / 2],
            pmd_norm_style: alloc::vec![1.0f32; d_l / 2],
            pmd_w_ord: fill(d_r * d_l, 251),
            pmd_w_coh: fill(d_r * d_l, 257),
            pmd_w_syn_w: fill(d_s * (d_r + d_l), 263),
            pmd_w_syn_b: fill(d_s, 269),
            pmd_u_syn: fill(d_s * d, 271),
            pmd_w_ctx_role: fill(d_r * d, 277),
            pmd_u_demod: fill(d_s * d_s, 281),
            pmd_w_chi_w: fill(3 * d_s, 283),
            pmd_w_chi_b: 1.0,
            pmd_w_act_w: fill(3 * (d_s + d_l), 293),
            pmd_w_act_b: alloc::vec![-1.5, -1.5, 2.0],
            pmd_w_tag_w: fill(4 * d_s, 307),
            pmd_w_tag_b: fill(4, 311),
            pmd_w_st: fill(d_s * 8, 313),
            pmd_m0: fill(d_s, 317),
            pmd_w_alpha_mot_w: fill(d_s * d_s, 331),
            pmd_w_alpha_mot_b: fill(d_s, 337),
            pmd_w_m: fill(d_s * d_s, 347),
            pmd_w_e_init_w: fill(2 * d_s + d_l, 349),
            pmd_w_e_init_b: 2.5,
            pmd_w_drain_w: fill(2 * d_s, 353),
            pmd_w_drain_b: -0.5,
            cfg,
        }
    }
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
impl DcpaWeights {
    pub fn from_safetensors(cfg: DcpaConfig, bytes: &[u8]) -> Result<Self, String> {
        cfg.validate().map_err(|e| e.to_string())?;
        let mut map = parse_safetensors(bytes)?;

        let mut bcg_experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            bcg_experts.push(ExpertWeights {
                w_gate1: ExpertMatrix::from_f32_vec(take_vec(&mut map, &format!("bcg_pc.moe.experts.{}.w_gate1.weight", e))?),
                w_up1: ExpertMatrix::from_f32_vec(take_vec(&mut map, &format!("bcg_pc.moe.experts.{}.w_up1.weight", e))?),
                w_down1: ExpertMatrix::from_f32_vec(take_vec(&mut map, &format!("bcg_pc.moe.experts.{}.w_down1.weight", e))?),
                w_gate2: ExpertMatrix::from_f32_vec(take_vec(&mut map, &format!("bcg_pc.moe.experts.{}.w_gate2.weight", e))?),
                w_up2: ExpertMatrix::from_f32_vec(take_vec(&mut map, &format!("bcg_pc.moe.experts.{}.w_up2.weight", e))?),
                w_down2: ExpertMatrix::from_f32_vec(take_vec(&mut map, &format!("bcg_pc.moe.experts.{}.w_down2.weight", e))?),
            });
        }

        Ok(Self {
            cfg,
            sgsc_embed: take_vec(&mut map, "sgsc.embed.weight")?,
            sgsc_w_out: take_vec(&mut map, "sgsc.w_out.weight")?,
            sgsc_h0: take_vec(&mut map, "sgsc.h0")?,
            sgsc_w_alpha: take_vec(&mut map, "sgsc.w_alpha.weight")?,
            sgsc_b_alpha: take_vec(&mut map, "sgsc.w_alpha.bias")?,
            sgsc_w_u: take_vec(&mut map, "sgsc.w_u.weight")?,
            sgsc_theta_raw: take_scalar(&mut map, "sgsc.theta_raw")?,
            sgsc_w_q: take_vec(&mut map, "sgsc.w_q.weight")?,
            sgsc_w_k: take_vec(&mut map, "sgsc.w_k.weight")?,
            sgsc_w_f: take_vec(&mut map, "sgsc.w_f.weight")?,
            sgsc_norm_f: take_vec(&mut map, "sgsc.norm_f.weight")?,
            sgsc_w_res: take_vec(&mut map, "sgsc.w_res.weight")?,
            sgsc_w_sem: take_vec(&mut map, "sgsc.w_sem.weight")?,
            sgsc_norm_c: take_vec(&mut map, "sgsc.norm_c.weight")?,

            bcg_s0: take_vec(&mut map, "bcg_pc.s0")?,
            bcg_a0: take_vec(&mut map, "bcg_pc.a0")?,
            bcg_omega0: take_vec(&mut map, "bcg_pc.omega0")?,
            bcg_c0: take_vec(&mut map, "bcg_pc.c0")?,
            bcg_w_e: take_vec(&mut map, "bcg_pc.w_e.weight")?,
            bcg_f_pred_1_w: take_vec(&mut map, "bcg_pc.f_pred_1.weight")?,
            bcg_f_pred_1_b: take_vec(&mut map, "bcg_pc.f_pred_1.bias")?,
            bcg_f_pred_2_w: take_vec(&mut map, "bcg_pc.f_pred_2.weight")?,
            bcg_f_pred_2_b: take_vec(&mut map, "bcg_pc.f_pred_2.bias")?,
            bcg_f_prec_1_w: take_vec(&mut map, "bcg_pc.f_prec_1.weight")?,
            bcg_f_prec_1_b: take_vec(&mut map, "bcg_pc.f_prec_1.bias")?,
            bcg_f_prec_2_w: take_vec(&mut map, "bcg_pc.f_prec_2.weight")?,
            bcg_f_prec_2_b: take_vec(&mut map, "bcg_pc.f_prec_2.bias")?,
            bcg_w_fast: take_vec(&mut map, "bcg_pc.w_fast.weight")?,
            bcg_w_gen: take_vec(&mut map, "bcg_pc.w_gen.weight")?,
            bcg_ca_wq: take_vec(&mut map, "bcg_pc.cross_attn.w_q.weight")?,
            bcg_ca_wk: take_vec(&mut map, "bcg_pc.cross_attn.w_k.weight")?,
            bcg_ca_wv: take_vec(&mut map, "bcg_pc.cross_attn.w_v.weight")?,
            bcg_ca_wo: take_vec(&mut map, "bcg_pc.cross_attn.w_o.weight")?,
            bcg_moe_router: take_vec(&mut map, "bcg_pc.moe.router.weight")?,
            bcg_experts,
            bcg_norm_ponder: take_vec(&mut map, "bcg_pc.norm_ponder.weight")?,
            bcg_u_src: take_vec(&mut map, "bcg_pc.u_src.weight")?,
            bcg_u_dst: take_vec(&mut map, "bcg_pc.u_dst.weight")?,
            bcg_w_l: take_vec(&mut map, "bcg_pc.w_l.weight")?,
            bcg_w_r: take_vec(&mut map, "bcg_pc.w_r.weight")?,
            bcg_w_edge_score: take_vec(&mut map, "bcg_pc.w_edge_score.weight")?,
            bcg_w_merge: take_vec(&mut map, "bcg_pc.w_merge.weight")?,
            bcg_w_g_w: take_vec(&mut map, "bcg_pc.w_g.weight")?,
            bcg_w_g_b: take_vec(&mut map, "bcg_pc.w_g.bias")?,
            bcg_w_comp_w: take_vec(&mut map, "bcg_pc.w_comp.weight")?,
            bcg_w_comp_b: take_vec(&mut map, "bcg_pc.w_comp.bias")?,
            bcg_norm_merge: take_vec(&mut map, "bcg_pc.norm_merge.weight")?,
            bcg_w_omega: take_vec(&mut map, "bcg_pc.w_omega.weight")?,
            bcg_kappa_omega: take_scalar(&mut map, "bcg_pc.kappa_omega")?,

            hip_r_lex: take_vec(&mut map, "hippocampus.r_lex")?,
            hip_r_sem: take_vec(&mut map, "hippocampus.r_sem")?,
            hip_alpha_1: take_scalar(&mut map, "hippocampus.alpha_1")?,
            hip_alpha_2: take_scalar(&mut map, "hippocampus.alpha_2")?,
            hip_theta_i: take_scalar(&mut map, "hippocampus.theta_i")?,
            hip_theta_n: take_scalar(&mut map, "hippocampus.theta_n")?,
            hip_alpha_q: take_scalar(&mut map, "hippocampus.alpha_q")?,
            hip_gamma_e: take_scalar(&mut map, "hippocampus.gamma_e")?,
            hip_theta_q: take_scalar(&mut map, "hippocampus.theta_q")?,
            hip_alpha_out: take_scalar(&mut map, "hippocampus.alpha_out")?,
            hip_theta_out: take_scalar(&mut map, "hippocampus.theta_out")?,
            hip_f_prec_cort_w: take_vec(&mut map, "hippocampus.f_prec_cort.weight")?,
            hip_f_prec_cort_b: take_vec(&mut map, "hippocampus.f_prec_cort.bias")?,
            hip_w_lex_slow: take_vec(&mut map, "hippocampus.w_lex_slow.weight")?,
            hip_w_lang_lex: take_vec(&mut map, "hippocampus.w_lang_lex.weight")?,

            pmd_lang_embed: take_vec(&mut map, "pmd.lang_embed.weight")?,
            pmd_style_embed: take_vec(&mut map, "pmd.style_embed.weight")?,
            pmd_norm_lang: take_vec(&mut map, "pmd.norm_lang.weight")?,
            pmd_norm_style: take_vec(&mut map, "pmd.norm_style.weight")?,
            pmd_w_ord: take_vec(&mut map, "pmd.w_ord.weight")?,
            pmd_w_coh: take_vec(&mut map, "pmd.w_coh.weight")?,
            pmd_w_syn_w: take_vec(&mut map, "pmd.w_syn.weight")?,
            pmd_w_syn_b: take_vec(&mut map, "pmd.w_syn.bias")?,
            pmd_u_syn: take_vec(&mut map, "pmd.u_syn.weight")?,
            pmd_w_ctx_role: take_vec(&mut map, "pmd.w_ctx_role.weight")?,
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

impl DcpaWeights {
    /// Convert all MoE expert matrices into 64-byte aligned `Bf16` storage (~50% lower slow-weight memory!).
    pub fn pack_experts_bf16(&mut self) {
        for exp in &mut self.bcg_experts {
            exp.pack_bf16();
        }
    }
}
