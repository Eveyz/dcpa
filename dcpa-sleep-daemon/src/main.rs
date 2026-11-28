//! dcpa-sleep-daemon (v3.3 R6): Idle-Time Sharp-Wave Ripple (SWR) Real-Fragment Replay ->
//! Dedicated Additive Dead-Zone Personal Sleep Plugin Consolidation Daemon.
//!
//! Architectural invariants (docs/DCPA_FLAT_V3_3_SPEC.md Section 3.5):
//!   1. ZERO Gaussian / pseudo-random anchor noise (`make_anchor_vec` permanently retired).
//!   2. Base model slow weights (`model.safetensors`) are 100% READ-ONLY and NEVER modified
//!      (eliminating catastrophic forgetting & base weight drift at the architecture level).
//!   3. Awake phase records real UTF-8 interaction fragments whose hippocampal write strength
//!      exceeds `beta > 0.25` (or explicit user corrections) into `SleepReplayCaptureRing`,
//!      writing canonical 16-byte `<QIBBBB` `.bin/.idx` + fixed-point x10 `.loss_mask.bin`.
//!   4. Idle sleep phase invokes the unified end-to-end plugin trainer backend
//!      (Phase 1: `python -m dcpa.train_plugin --mode sleep`; Phase 2: drop-in Candle via
//!      `--trainer-cmd`) to update ONLY the isolated 8-layer personal sleep plugin pack
//!      (`personal_sleep.dcpa_expert.safetensors`).
//!   5. Before atomic promotion, Rust `DcpaFlatEngine` hot-mounts the candidate plugin onto
//!      the read-only base and verifies:
//!        - General-domain EN/ZH probe anchor max logit drift `< 1e-6` (exact `0.0` dead-zone).
//!        - Base weights file byte-for-byte untouched.
//!      Upon pass, atomically replaces the live plugin file and decays consolidated hippocampal
//!      familiarity traces (`u_lex *= 0.2`) to free fast-weight capacity.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use dcpa_core_rs::flat::{
    DcpaFlatConfig, DcpaFlatEngine, DcpaFlatWeights, PluginExpertPackRs, SleepReplayCaptureRing,
};
use dcpa_core_rs::state::FastWeightPrecision;

#[derive(Debug, Clone)]
pub struct FlatSleepConfig {
    pub active_ratio_threshold: f32, // 0.35 (35% active u_lex slots)
    pub beta_capture_threshold: f32, // 0.25 (awake capture threshold)
    pub max_ring_fragments: usize,   // e.g. 256 fragments
    pub max_ring_bytes: usize,       // 4 MiB default ring capacity
    pub hip_release_decay: f32,      // 0.20 familiarity decay after consolidation
    pub familiarity_release_min: f32,// 0.50 slots above this decay after consolidation
    pub max_anchor_drift: f32,       // 1e-6 hard zero-drift gate
    pub train_steps: usize,
    pub experts_per_layer: usize,
    pub pack_id: String,
    pub device: String,              // "cpu" | "auto" | "cuda" | "mps"
}

impl Default for FlatSleepConfig {
    fn default() -> Self {
        Self {
            active_ratio_threshold: 0.35,
            beta_capture_threshold: 0.25,
            max_ring_fragments: 256,
            max_ring_bytes: 4 * 1024 * 1024,
            hip_release_decay: 0.20,
            familiarity_release_min: 0.50,
            max_anchor_drift: 1e-6,
            train_steps: 20,
            experts_per_layer: 1,
            pack_id: "personal_sleep".to_string(),
            device: "cpu".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlatSleepReport {
    pub active_slot_ratio: f32,
    pub fragments_captured: usize,
    pub replay_total_bytes: usize,
    pub anchor_max_logit_drift: f32,
    pub target_plugin_delta: f32,
    pub released_hip_slots: usize,
    pub base_weights_untouched: bool,
    pub promoted: bool,
}

/// Built-in general-domain probes matching `dcpa/train_plugin.py` anchors for Rust zero-drift gating.
const GENERAL_PROBES: &[(usize, usize, &[u8])] = &[
    (
        0,
        0,
        b"The scientific method relies on reproducible experiments, clear hypotheses, and falsifiable measurements.\n\n",
    ),
    (
        1,
        0,
        "科学方法依赖可重复实验、清晰假设与可证伪的量化检验，任何结论都必须经得起独立复核。\n\n".as_bytes(),
    ),
];

/// Run one full sequence through `DcpaFlatEngine` (returning flattened byte logits `[T * 256]`,
/// peak hippocampal `max_beta`, and high-beta anchor byte ranges `(start, end)`).
pub fn run_flat_sequence_trace(
    engine: &mut DcpaFlatEngine,
    raw_bytes: &[u8],
    lang_id: usize,
    style_id: usize,
    beta_threshold: f32,
) -> (Vec<f32>, f32, Vec<(usize, usize)>) {
    engine.reset_session();
    let t_len = raw_bytes.len();
    let mut logits_out = vec![0.0f32; t_len * 256];
    let mut peak_beta = 0.0f32;
    let mut anchor_ranges: Vec<(usize, usize)> = Vec::new();

    // First pass: segment boundaries via streaming SGSC (or single-pass buffered per segment)
    // Using exact same causal segment cadence as `DCPAModelFlat.forward_streaming_sequence`:
    let mut s_idx: i32 = 0;
    let mut seg_start_byte: usize = 0;

    // Collect SGSC segments deterministically byte-by-byte, then run causal predict -> PMD -> absorb
    // Notice: SGSC depends strictly on raw bytes & sensory weights (independent of cortical/hip state).
    let mut seg_bounds: Vec<(usize, usize, i32, i32, bool, bool, f32, f32, Vec<f32>, Vec<f32>)> =
        Vec::new();
    for (t, &b) in raw_bytes.iter().enumerate() {
        let bo = engine.sgsc_ingest_byte(b, t + 1 == t_len);
        if bo.spiked {
            seg_bounds.push((
                seg_start_byte,
                t + 1,
                bo.seg_start_depth,
                bo.seg_end_depth,
                bo.seg_pin_trig,
                bo.seg_unpin_trig,
                bo.sem_norm,
                bo.mean_info,
                engine.scratch.f_stem.clone(),
                engine.scratch.x0.clone(),
            ));
            seg_start_byte = t + 1;
        }
    }

    for (t0, t1, st_dep, en_dep, pin_t, unpin_t, sem_n, mean_i, f_stem, x0) in seg_bounds {
        let _prior = engine.predict_segment_prior(lang_id, style_id, st_dep, s_idx, unpin_t);
        for (local_i, t) in (t0..t1).enumerate() {
            let pb = if t == 0 { 0u8 } else { raw_bytes[t - 1] };
            let scratch_end_reset = t >= 3
                && raw_bytes[t - 3] == 0x1D
                && raw_bytes[t - 2] == b'E'
                && raw_bytes[t - 1] == b'\n';
            let _pmd = engine.pmd_step_byte(pb, local_i + 1, scratch_end_reset || t == 0);
            logits_out[t * 256..(t + 1) * 256].copy_from_slice(&engine.scratch.byte_logits_256);
        }
        engine.scratch.f_stem.copy_from_slice(&f_stem);
        engine.scratch.x0.copy_from_slice(&x0);
        let seg_o = engine.absorb_segment(
            s_idx, en_dep, pin_t, unpin_t, lang_id, style_id, sem_n, mean_i,
        );
        if seg_o.max_beta > peak_beta {
            peak_beta = seg_o.max_beta;
        }
        if seg_o.max_beta > beta_threshold {
            anchor_ranges.push((t0, t1));
        }
        s_idx += 1;
    }

    (logits_out, peak_beta, anchor_ranges)
}

/// Awake capture hook: ingest real interaction fragments into `DcpaFlatEngine` and record any
/// fragment triggering `peak_beta > beta_threshold` (or `explicit_correction == true`) into
/// `SleepReplayCaptureRing` with high-beta anchor spans marked for `3.0x` loss weighting.
pub fn record_awake_real_fragment(
    engine: &mut DcpaFlatEngine,
    ring: &mut SleepReplayCaptureRing,
    raw_bytes: &[u8],
    lang_id: u8,
    style_id: u8,
    explicit_correction: bool,
) -> (bool, f32) {
    let (_, peak_beta, anchors) = run_flat_sequence_trace(
        engine,
        raw_bytes,
        lang_id as usize,
        style_id as usize,
        ring.beta_threshold,
    );
    let recorded = ring.maybe_record(
        raw_bytes,
        lang_id,
        style_id,
        &anchors,
        peak_beta,
        explicit_correction,
    );
    (recorded, peak_beta)
}

/// Dual-gate Rust verification of a newly trained sleep plugin pack on the read-only base model.
/// Returns `(max_anchor_drift, max_target_delta)`.
pub fn verify_sleep_plugin_in_rust(
    base_dir: &Path,
    candidate_plugin_path: &Path,
    sample_target: Option<(&[u8], usize, usize)>,
) -> Result<(f32, f32), String> {
    let cfg_str = fs::read_to_string(base_dir.join("config.json"))
        .map_err(|e| format!("read config.json: {}", e))?;
    let cfg = DcpaFlatConfig::from_json_str(&cfg_str)
        .map_err(|e| format!("parse config.json: {}", e))?;
    let w_bytes = fs::read(base_dir.join("model.safetensors"))
        .map_err(|e| format!("read model.safetensors: {}", e))?;
    let weights = DcpaFlatWeights::from_safetensors(cfg, &w_bytes)?;
    let mut engine = DcpaFlatEngine::new(weights, FastWeightPrecision::F32, false);

    // 1. Collect base probe logits before mounting plugin
    let mut base_probes: Vec<Vec<f32>> = Vec::with_capacity(GENERAL_PROBES.len());
    for &(lid, sid, bytes) in GENERAL_PROBES {
        let (logits, _, _) = run_flat_sequence_trace(&mut engine, bytes, lid, sid, 0.25);
        base_probes.push(logits);
    }
    let base_target = sample_target.map(|(b, lid, sid)| {
        run_flat_sequence_trace(&mut engine, b, lid, sid, 0.25).0
    });

    // 2. Hot-mount candidate personal sleep plugin pack
    let plug_bytes = fs::read(candidate_plugin_path)
        .map_err(|e| format!("read candidate plugin: {}", e))?;
    let pack = PluginExpertPackRs::from_safetensors("personal_sleep", &plug_bytes)?;
    engine
        .mount_expert_pack(pack)
        .map_err(|e| format!("mount candidate plugin: {}", e))?;

    // 3. Verify general-domain probe drift (strictly < max_anchor_drift; exact 0.0 under hard dead-zone)
    let mut max_anchor_drift = 0.0f32;
    for (i, &(lid, sid, bytes)) in GENERAL_PROBES.iter().enumerate() {
        let (plug_logits, _, _) = run_flat_sequence_trace(&mut engine, bytes, lid, sid, 0.25);
        for (a, b) in plug_logits.iter().zip(base_probes[i].iter()) {
            let d = (a - b).abs();
            if d > max_anchor_drift {
                max_anchor_drift = d;
            }
        }
    }

    let mut max_target_delta = 0.0f32;
    if let (Some((b, lid, sid)), Some(base_t)) = (sample_target, base_target) {
        let (plug_t, _, _) = run_flat_sequence_trace(&mut engine, b, lid, sid, 0.25);
        for (a, bv) in plug_t.iter().zip(base_t.iter()) {
            let d = (a - bv).abs();
            if d > max_target_delta {
                max_target_delta = d;
            }
        }
    }

    Ok((max_anchor_drift, max_target_delta))
}

/// Decay consolidated hippocampal familiarity slots (`u_lex *= decay`) to free fast-weight space.
pub fn release_consolidated_hippocampal_slots(
    engine: &mut DcpaFlatEngine,
    familiarity_min: f32,
    release_decay: f32,
) -> usize {
    engine.fast_mem.sync_all_u_lex();
    let d_h = engine.weights.cfg.d_h;
    let mut released = 0usize;
    for k in 0..d_h {
        if engine.fast_mem.u_lex.as_slice()[k] > familiarity_min {
            engine.fast_mem.u_lex.as_mut_slice()[k] *= release_decay;
            engine.fast_mem.col_hits_lex[k] = 0;
            released += 1;
        }
    }
    released
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Full v3.3 idle sleep consolidation pipeline:
///   1. Flush captured real fragments (`SleepReplayCaptureRing`) to `<QIBBBB` `.bin/.idx/.loss_mask.bin`.
///   2. Invoke unified plugin trainer backend (Phase 1 PyTorch `dcpa.train_plugin --mode sleep`,
///      or Phase 2 Candle via custom `trainer_cmd`) writing a staging `.tmp.dcpa_expert.safetensors`.
///   3. Verify read-only base weight immutability + Rust `DcpaFlatEngine` zero-drift gate (`< 1e-6`).
///   4. Atomically promote plugin to `out_plugin_path` & release consolidated hippocampal `u_lex` slots.
pub fn run_v3_sleep_consolidation(
    engine: &mut DcpaFlatEngine,
    ring: &SleepReplayCaptureRing,
    base_dir: &Path,
    work_dir: &Path,
    out_plugin_path: &Path,
    scfg: &FlatSleepConfig,
    python_bin: &Path,
    trainer_cmd_override: Option<&str>,
    force_run: bool,
) -> Result<FlatSleepReport, String> {
    engine.fast_mem.sync_all_u_lex();
    let d_h = engine.weights.cfg.d_h;
    let active_slots = engine
        .fast_mem
        .u_lex
        .as_slice()
        .iter()
        .filter(|&&u| u > 0.1)
        .count();
    let active_ratio = (active_slots as f32) / (d_h.max(1) as f32);

    if ring.is_empty() || (!force_run && active_ratio <= scfg.active_ratio_threshold) {
        return Ok(FlatSleepReport {
            active_slot_ratio: active_ratio,
            fragments_captured: ring.len(),
            replay_total_bytes: ring.total_bytes(),
            anchor_max_logit_drift: 0.0,
            target_plugin_delta: 0.0,
            released_hip_slots: 0,
            base_weights_untouched: true,
            promoted: false,
        });
    }

    let base_weight_path = base_dir.join("model.safetensors");
    let base_hash_before = fnv1a_hash(
        &fs::read(&base_weight_path)
            .map_err(|e| format!("read base model.safetensors before sleep: {}", e))?,
    );

    // 1. Write captured real fragments into canonical `<QIBBBB` shard triplet
    fs::create_dir_all(work_dir).map_err(|e| format!("create work_dir: {}", e))?;
    let n_written = ring
        .write_shard_triplet(work_dir, "sleep_replay_ring")
        .map_err(|e| format!("write sleep_replay_ring triplet: {}", e))?;
    if n_written == 0 {
        return Err("sleep_replay_ring wrote 0 valid fragments".to_string());
    }
    let replay_bin = work_dir.join("sleep_replay_ring.bin");
    let replay_idx = work_dir.join("sleep_replay_ring.idx");
    let staging_plugin = work_dir.join("personal_sleep.tmp.dcpa_expert.safetensors");
    let report_json = work_dir.join("sleep_train_report.json");

    // 2. Invoke unified end-to-end plugin trainer backend
    let mut cmd = if let Some(custom) = trainer_cmd_override {
        let mut c = Command::new(custom);
        c.arg("--mode").arg("sleep");
        c
    } else {
        let mut c = Command::new(python_bin);
        c.arg("-m").arg("dcpa.train_plugin").arg("--mode").arg("sleep");
        // Ensure `dcpa-train-py` is on PYTHONPATH when invoked from repo root or installed venv
        if let Ok(cur_dir) = env::current_dir() {
            let train_py = cur_dir.join("dcpa-train-py");
            if train_py.is_dir() {
                let prev = env::var("PYTHONPATH").unwrap_or_default();
                let merged = if prev.is_empty() {
                    format!("{}:{}", train_py.display(), cur_dir.display())
                } else {
                    format!("{}:{}:{}", train_py.display(), cur_dir.display(), prev)
                };
                c.env("PYTHONPATH", merged);
            }
        }
        c
    };

    cmd.arg("--base-dir")
        .arg(base_dir)
        .arg("--replay-bin")
        .arg(&replay_bin)
        .arg("--replay-idx")
        .arg(&replay_idx)
        .arg("--out-plugin")
        .arg(&staging_plugin)
        .arg("--pack-id")
        .arg(&scfg.pack_id)
        .arg("--steps")
        .arg(scfg.train_steps.to_string())
        .arg("--experts-per-layer")
        .arg(scfg.experts_per_layer.to_string())
        .arg("--device")
        .arg(&scfg.device)
        .arg("--report-json")
        .arg(&report_json);

    let output = cmd
        .output()
        .map_err(|e| format!("failed to launch sleep plugin trainer: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "sleep plugin trainer exited with {}: stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // 3. Verify base weights are byte-for-byte untouched + Rust zero-drift verification
    let base_hash_after = fnv1a_hash(
        &fs::read(&base_weight_path)
            .map_err(|e| format!("read base model.safetensors after sleep: {}", e))?,
    );
    let base_untouched = base_hash_before == base_hash_after;
    if !base_untouched {
        return Err("SECURITY VIOLATION: read-only base model.safetensors was modified!".to_string());
    }

    let sample_target = ring
        .fragments()
        .first()
        .map(|f| (f.bytes.as_slice(), f.lang_id as usize, f.style_id as usize));
    let (anchor_drift, target_delta) =
        verify_sleep_plugin_in_rust(base_dir, &staging_plugin, sample_target)?;

    if !(anchor_drift < scfg.max_anchor_drift) {
        let _ = fs::remove_file(&staging_plugin);
        return Err(format!(
            "Sleep plugin rejected by zero-drift gate: anchor_max_logit_drift={:.3e} >= {:.1e}",
            anchor_drift, scfg.max_anchor_drift
        ));
    }

    // 4. Atomic promotion & hot-mount + hippocampal slot release
    if let Some(parent) = out_plugin_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create plugin parent dir: {}", e))?;
    }
    fs::rename(&staging_plugin, out_plugin_path)
        .or_else(|_| {
            fs::copy(&staging_plugin, out_plugin_path)?;
            fs::remove_file(&staging_plugin)
        })
        .map_err(|e| format!("promote personal_sleep plugin: {}", e))?;

    let final_bytes = fs::read(out_plugin_path)
        .map_err(|e| format!("reload promoted sleep plugin: {}", e))?;
    let final_pack = PluginExpertPackRs::from_safetensors(&scfg.pack_id, &final_bytes)?;
    engine
        .mount_expert_pack(final_pack)
        .map_err(|e| format!("hot-mount promoted sleep plugin: {}", e))?;

    let released = release_consolidated_hippocampal_slots(
        engine,
        scfg.familiarity_release_min,
        scfg.hip_release_decay,
    );

    Ok(FlatSleepReport {
        active_slot_ratio: active_ratio,
        fragments_captured: ring.len(),
        replay_total_bytes: ring.total_bytes(),
        anchor_max_logit_drift: anchor_drift,
        target_plugin_delta: target_delta,
        released_hip_slots: released,
        base_weights_untouched: base_untouched,
        promoted: true,
    })
}

fn default_python_bin() -> PathBuf {
    if let Ok(p) = env::var("DCPA_PYTHON") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let venv_py = PathBuf::from("dcpa-train-py/.venv/bin/python");
    if venv_py.is_file() {
        return venv_py;
    }
    PathBuf::from("python3")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: dcpa-sleep-daemon <base_model_dir> <out_personal_sleep.dcpa_expert.safetensors> \
             [--steps N] [--python PATH] [--trainer-cmd PATH] [--work-dir DIR] [--fragment-file FILE]"
        );
        std::process::exit(1);
    }

    let base_dir = PathBuf::from(&args[1]);
    let out_plugin = PathBuf::from(&args[2]);
    let mut scfg = FlatSleepConfig::default();
    let mut python_bin = default_python_bin();
    let mut trainer_override: Option<String> = None;
    let mut work_dir = out_plugin
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("sleep_work");
    let mut fragment_file: Option<PathBuf> = None;

    let mut i = 3usize;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" if i + 1 < args.len() => {
                scfg.train_steps = args[i + 1].parse().expect("invalid --steps");
                i += 2;
            }
            "--python" if i + 1 < args.len() => {
                python_bin = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--trainer-cmd" if i + 1 < args.len() => {
                trainer_override = Some(args[i + 1].clone());
                i += 2;
            }
            "--work-dir" if i + 1 < args.len() => {
                work_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--fragment-file" if i + 1 < args.len() => {
                fragment_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(2);
            }
        }
    }

    let cfg_str = fs::read_to_string(base_dir.join("config.json")).expect("read config.json");
    let cfg = DcpaFlatConfig::from_json_str(&cfg_str).expect("parse Flat config.json");
    let w_bytes = fs::read(base_dir.join("model.safetensors")).expect("read model.safetensors");
    let weights = DcpaFlatWeights::from_safetensors(cfg, &w_bytes).expect("load Flat model.safetensors");
    let mut engine = DcpaFlatEngine::new(weights, FastWeightPrecision::Bf16, true);

    let mut ring = SleepReplayCaptureRing::new(
        scfg.max_ring_fragments,
        scfg.max_ring_bytes,
        scfg.beta_capture_threshold,
    );

    if let Some(frag_path) = fragment_file {
        let raw = fs::read(&frag_path).expect("read --fragment-file");
        let _ = record_awake_real_fragment(&mut engine, &mut ring, &raw, 4, 4, true);
    } else {
        // Default awake consolidation episode (real structured fragments, zero Gaussian noise!)
        let episodes: &[(u8, u8, &[u8])] = &[
            (
                4,
                4,
                b"SELECT acct_id, SUM(amount) AS net_total FROM ledger_entries WHERE status = 'POSTED' GROUP BY acct_id;\n",
            ),
            (
                4,
                4,
                b"CREATE INDEX idx_ledger_acct_status ON ledger_entries (acct_id, status);\n",
            ),
        ];
        for &(lid, sid, text) in episodes {
            let _ = record_awake_real_fragment(&mut engine, &mut ring, text, lid, sid, true);
        }
    }

    let report = run_v3_sleep_consolidation(
        &mut engine,
        &ring,
        &base_dir,
        &work_dir,
        &out_plugin,
        &scfg,
        &python_bin,
        trainer_override.as_deref(),
        true,
    )
    .expect("v3.3 sleep consolidation failed");

    println!(
        "[SleepDaemon v3.3] promoted={} | fragments={} ({}B) | anchor_drift={:.3e} | target_delta={:.3e} | base_untouched={} | released_slots={} | out={}",
        report.promoted,
        report.fragments_captured,
        report.replay_total_bytes,
        report.anchor_max_logit_drift,
        report.target_plugin_delta,
        report.base_weights_untouched,
        report.released_hip_slots,
        out_plugin.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_immutability_hash_and_ring_capture_contract() {
        let a = fnv1a_hash(b"base_weights_v3");
        let b = fnv1a_hash(b"base_weights_v3");
        let c = fnv1a_hash(b"base_weights_v4");
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut ring = SleepReplayCaptureRing::new(8, 4096, 0.25);
        // Low beta without explicit correction is rejected
        assert!(!ring.maybe_record(b"routine text", 0, 0, &[], 0.10, false));
        assert!(ring.is_empty());
        // High beta (> 0.25) is captured
        assert!(ring.maybe_record(b"SELECT * FROM vault;", 4, 4, &[(14, 19)], 0.42, false));
        assert_eq!(ring.len(), 1);
    }
}
