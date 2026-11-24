//! dcpa-sleep-daemon: Idle-Time Sharp-Wave Ripple (SWR) Hippocampus -> Cortex Consolidation Daemon.
//!
//! Implements Engineering Blueprint Chapter 6 Law #3 & Theory Doc Part 3.III.2:
//!   - Triggers during idle periods when hippocampal active familiarity slot occupancy > 35%.
//!   - Filters high-familiarity prototypes (`u_lex[k] > 0.85` & hit count >= min_hits).
//!   - Performs rank-1 local least-squares micro-step updates on slow cortical matrices `W_sem` [D, d_s]
//!     and `W_lex_slow` [d_s, D], interleaved 1:4 with random cortical old anchor vectors to prevent
//!     catastrophic forgetting!
//!   - Verifies post-consolidation cortical familiarity `gamma_slow > 0.9`, decays consolidated
//!     hippocampal familiarity trace by 0.2 to free fast-weight space, and exports an incremental
//!     safetensors cortical weight slice (`cortical_sleep_slice.safetensors`).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use dcpa_core_rs::config::DcpaConfig;
use dcpa_core_rs::engine::DcpaEngine;
use dcpa_core_rs::simd::{cosine_sim_f32, l2_norm_f32, linear_f32, sigmoid};
use dcpa_core_rs::state::FastWeightPrecision;
use dcpa_core_rs::weights::{serialize_safetensors_f32, DcpaWeights};

pub struct SleepConsolidationConfig {
    pub active_ratio_threshold: f32, // 0.35 (35%)
    pub familiarity_threshold: f32,  // 0.85
    pub min_hits: u32,
    pub step_size: f32,
    pub anchor_replay_ratio: usize,  // 4 old anchors per 1 new prototype (1:4)
    pub target_gamma_slow: f32,      // 0.90
    pub hip_release_decay: f32,      // 0.20
    pub max_micro_steps: usize,
}

impl Default for SleepConsolidationConfig {
    fn default() -> Self {
        Self {
            active_ratio_threshold: 0.35,
            familiarity_threshold: 0.85,
            min_hits: 1,
            step_size: 0.05,
            anchor_replay_ratio: 4,
            target_gamma_slow: 0.90,
            hip_release_decay: 0.20,
            max_micro_steps: 40,
        }
    }
}

#[derive(Debug)]
pub struct SleepReport {
    pub active_slot_ratio: f32,
    pub candidates_found: usize,
    pub consolidated_count: usize,
    pub avg_post_gamma_slow: f32,
    pub avg_anchor_retention_cos: f32,
}

/// Rank-1 local least-squares micro-step on matrix W [rows, cols]:
///   W <- W + eta * (target - W x) x^T / (||x||^2 + eps)
fn rank1_rls_update(w: &mut [f32], rows: usize, cols: usize, x: &[f32], target: &[f32], eta: f32) {
    let mut x_norm_sq = 0.0f32;
    for &v in x {
        x_norm_sq += v * v;
    }
    let scale = eta / (x_norm_sq + 1e-6);
    for r in 0..rows {
        let row = &mut w[r * cols..(r + 1) * cols];
        let mut pred = 0.0f32;
        for c in 0..cols {
            pred += row[c] * x[c];
        }
        let err = (target[r] - pred) * scale;
        for c in 0..cols {
            row[c] += err * x[c];
        }
    }
}

/// Deterministic pseudo-random cortical anchor vector (unit norm) for 1:4 anti-forgetting replay.
fn make_anchor_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut v = Vec::with_capacity(dim);
    let mut sq = 0.0f32;
    for _ in 0..dim {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let u = ((x >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
        v.push(u);
        sq += u * u;
    }
    let inv = 1.0 / (sq.sqrt() + 1e-8);
    for val in v.iter_mut() {
        *val *= inv;
    }
    v
}

/// Run one Sharp-Wave Ripple (SWR) offline sleep distillation pass on an active `DcpaEngine`.
pub fn run_swr_sleep_replay(
    engine: &mut DcpaEngine,
    scfg: &SleepConsolidationConfig,
    force_run: bool,
) -> SleepReport {
    engine.fast_mem.sync_all_u_lex();
    let d_h = engine.weights.cfg.d_h;
    let d_s = engine.weights.cfg.d_s;
    let d_c = engine.weights.cfg.d_concept;

    let active_slots = engine
        .fast_mem
        .u_lex
        .as_slice()
        .iter()
        .filter(|&&u| u > 0.1)
        .count();
    let active_ratio = (active_slots as f32) / (d_h as f32);

    if !force_run && active_ratio <= scfg.active_ratio_threshold {
        return SleepReport {
            active_slot_ratio: active_ratio,
            candidates_found: 0,
            consolidated_count: 0,
            avg_post_gamma_slow: 0.0,
            avg_anchor_retention_cos: 1.0,
        };
    }

    // Identify high-familiarity hippocampal indices (u_lex > 0.85 & hit count >= min_hits)
    let mut candidate_cols: Vec<usize> = Vec::new();
    for k in 0..d_h {
        if engine.fast_mem.u_lex.as_slice()[k] > scfg.familiarity_threshold
            && engine.fast_mem.col_hits_lex[k] >= scfg.min_hits
        {
            candidate_cols.push(k);
        }
    }

    if candidate_cols.is_empty() {
        return SleepReport {
            active_slot_ratio: active_ratio,
            candidates_found: 0,
            consolidated_count: 0,
            avg_post_gamma_slow: 0.0,
            avg_anchor_retention_cos: 1.0,
        };
    }

    // Snapshot random cortical anchor input-output pairs BEFORE updating (1:4 anti-forgetting replay!)
    let num_anchors = (candidate_cols.len() * scfg.anchor_replay_ratio).min(64).max(4);
    let mut old_anchors: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(num_anchors);
    for a_i in 0..num_anchors {
        let f_old = make_anchor_vec(d_s, 0xABCD_0001 + a_i as u64);
        let mut c_old = vec![0.0f32; d_c];
        linear_f32(&f_old, &engine.weights.sgsc_w_sem, None, &mut c_old);
        let mut f_rec = vec![0.0f32; d_s];
        linear_f32(&c_old, &engine.weights.hip_w_lex_slow, None, &mut f_rec);
        old_anchors.push((f_old, c_old, f_rec));
    }

    // Reconstruct prototype pairs (f_proto, c_proto) from R_lex row k & M^{C<-L} column k
    let mut consolidated = 0usize;
    let mut post_gamma_sum = 0.0f32;
    let mut c_proto = vec![0.0f32; d_c];
    let mut c_pred = vec![0.0f32; d_c];

    for (idx_pos, &k) in candidate_cols.iter().enumerate() {
        engine.fast_mem.read_lex_column_f32(k, &mut c_proto);
        let scale = (engine.weights.cfg.s_active as f32).sqrt();
        for v in c_proto.iter_mut() {
            *v *= scale;
        }
        let f_proto = &engine.weights.hip_r_lex[k * d_s..(k + 1) * d_s];

        // Local SWR Replay micro-steps mixing 1 new prototype : 4 old cortical anchors
        let mut gamma_slow_after = 0.0f32;
        for _step in 0..scfg.max_micro_steps {
            rank1_rls_update(
                &mut engine.weights.sgsc_w_sem,
                d_c,
                d_s,
                f_proto,
                &c_proto,
                scfg.step_size,
            );
            rank1_rls_update(
                &mut engine.weights.hip_w_lex_slow,
                d_s,
                d_c,
                &c_proto,
                f_proto,
                scfg.step_size,
            );

            // Replay 4 old anchors to protect slow cortical manifold against forgetting
            for r in 0..scfg.anchor_replay_ratio {
                let (ref f_old, ref c_old, ref f_rec) =
                    old_anchors[(idx_pos * scfg.anchor_replay_ratio + r) % old_anchors.len()];
                rank1_rls_update(
                    &mut engine.weights.sgsc_w_sem,
                    d_c,
                    d_s,
                    f_old,
                    c_old,
                    scfg.step_size * 0.5,
                );
                rank1_rls_update(
                    &mut engine.weights.hip_w_lex_slow,
                    d_s,
                    d_c,
                    c_old,
                    f_rec,
                    scfg.step_size * 0.5,
                );
            }

            linear_f32(f_proto, &engine.weights.sgsc_w_sem, None, &mut c_pred);
            let sem_norm = l2_norm_f32(&c_pred);
            let align_cos = cosine_sim_f32(&c_pred, &c_proto);
            // Effective cortical familiarity check combining norm & semantic alignment
            let gamma_norm = sigmoid(
                engine.weights.hip_alpha_1 * (engine.weights.hip_theta_i - 1.0)
                    + engine.weights.hip_alpha_2 * (sem_norm - engine.weights.hip_theta_n),
            );
            gamma_slow_after = (0.5 * gamma_norm + 0.5 * align_cos.max(0.0)).min(1.0);
            if align_cos > scfg.target_gamma_slow && gamma_slow_after > scfg.target_gamma_slow {
                break;
            }
        }

        post_gamma_sum += gamma_slow_after;
        if gamma_slow_after > scfg.target_gamma_slow {
            // Release fast memory capacity: u_lex[k] *= 0.2
            engine.fast_mem.u_lex.as_mut_slice()[k] *= scfg.hip_release_decay;
            engine.fast_mem.col_hits_lex[k] = 0;
            consolidated += 1;
        }
    }

    // Measure old cortical anchor retention
    let mut retention_sum = 0.0f32;
    for (f_old, c_old, _) in &old_anchors {
        linear_f32(f_old, &engine.weights.sgsc_w_sem, None, &mut c_pred);
        retention_sum += cosine_sim_f32(&c_pred, c_old);
    }

    SleepReport {
        active_slot_ratio: active_ratio,
        candidates_found: candidate_cols.len(),
        consolidated_count: consolidated,
        avg_post_gamma_slow: post_gamma_sum / (candidate_cols.len() as f32),
        avg_anchor_retention_cos: retention_sum / (old_anchors.len() as f32),
    }
}

/// Export incrementally updated slow cortical slices (`sgsc.w_sem.weight`, `hippocampus.w_lex_slow.weight`).
pub fn export_cortical_slice(weights: &DcpaWeights, out_path: &Path) -> std::io::Result<()> {
    let mut map: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    map.insert(
        "sgsc.w_sem.weight".to_string(),
        (
            vec![weights.cfg.d_concept, weights.cfg.d_s],
            weights.sgsc_w_sem.clone(),
        ),
    );
    map.insert(
        "hippocampus.w_lex_slow.weight".to_string(),
        (
            vec![weights.cfg.d_s, weights.cfg.d_concept],
            weights.hip_w_lex_slow.clone(),
        ),
    );
    let raw = serialize_safetensors_f32(&map);
    fs::write(out_path, raw)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dcpa-sleep-daemon <model_dir> <out_slice.safetensors>");
        std::process::exit(1);
    }
    let dir = Path::new(&args[1]);
    let out_slice = Path::new(&args[2]);

    let cfg_str = fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let cfg = DcpaConfig::from_json_str(&cfg_str).expect("parse config.json");
    let w_bytes = fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let weights = DcpaWeights::from_safetensors(cfg, &w_bytes).expect("load model.safetensors");
    let mut engine = DcpaEngine::new(weights, FastWeightPrecision::Bf16);

    // Demo/diagnostic consolidation cycle: ingest a novel phrase repeatedly to populate u_lex > 0.85
    let text = b"A zibber is a nocturnal emerald frog. That zibber jumped onto the wet leaf.";
    for _ in 0..4 {
        for (i, &b) in text.iter().enumerate() {
            let tr = engine.sgsc_ingest_byte(b, i + 1 == text.len());
            if tr.spiked {
                let _ = engine.step_concept_from_scratch(tr.mean_info, tr.sem_norm);
            }
        }
    }
    // Elevate top active slot to simulate repeated user encounter above 0.85 threshold
    for k in 0..engine.weights.cfg.s_active {
        let idx = engine.scratch.psi_idx[k];
        engine.fast_mem.u_lex.as_mut_slice()[idx] = 0.95;
        engine.fast_mem.col_hits_lex[idx] = 5;
    }

    let report = run_swr_sleep_replay(&mut engine, &SleepConsolidationConfig::default(), true);
    println!("SWR Sleep Consolidation Report: {:?}", report);
    export_cortical_slice(&engine.weights, out_slice).expect("failed to export cortical slice");
    println!("Exported incremental cortical weight slice to {:?}", out_slice);
}
