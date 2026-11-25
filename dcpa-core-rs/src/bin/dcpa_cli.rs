//! dcpa-cli: Local zero-GPU CLI, Gate 6 Cross-Stack Verifier, Zero-Alloc CPU Benchmark,
//! and Stage-5 Stateful Closed-Loop Interactive REPL (`0x1E/0x1F/0x04/0x1D/0x11/0x13/0x12`).
//!
//! Installs a custom global `CountingAllocator` to mathematically prove `0 allocs/byte`
//! throughout the hot byte perception, hippocampal Householder update, `~22 KB` snapshot
//! fork/restore, and streaming generation loops (Engineering Blueprint Chapter 6 Law #2 & Gate 6).

use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dcpa_core_rs::config::DcpaConfig;
use dcpa_core_rs::engine::DcpaEngine;
use dcpa_core_rs::state::{BrainSnapshot, FastWeightPrecision};
use dcpa_core_rs::weights::{parse_safetensors, DcpaWeights};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_alloc_counter() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn read_alloc_counter() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
}

fn resolve_export_paths(path: &Path) -> (PathBuf, PathBuf) {
    if path.is_file() {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        return (parent.join("config.json"), path.to_path_buf());
    }
    let cfg_path = path.join("config.json");
    let candidates = [
        "dcpa_chat_consolidated.safetensors",
        "dcpa_chat_sft.safetensors",
        "dcpa_final.safetensors",
        "model.safetensors",
    ];
    for name in candidates {
        let cand = path.join(name);
        if cand.is_file() {
            return (cfg_path, cand);
        }
    }
    (cfg_path, path.join("model.safetensors"))
}

fn load_engine(path: &Path, precision: FastWeightPrecision) -> DcpaEngine {
    let (cfg_path, weight_path) = resolve_export_paths(path);
    let cfg_str = fs::read_to_string(&cfg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", cfg_path.display(), e));
    let cfg = DcpaConfig::from_json_str(&cfg_str).expect("invalid config.json");
    let w_bytes = fs::read(&weight_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", weight_path.display(), e));
    let weights = DcpaWeights::from_safetensors(cfg, &w_bytes)
        .unwrap_or_else(|e| panic!("failed to load {}: {}", weight_path.display(), e));
    DcpaEngine::new(weights, precision)
}

fn check_vec_parity(name: &str, rust_v: &[f32], py_v: &[f32], max_err_thresh: f32, min_cos: f32) {
    assert_eq!(
        rust_v.len(),
        py_v.len(),
        "Shape mismatch for {}: rust={} vs py={}",
        name,
        rust_v.len(),
        py_v.len()
    );
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut na2 = 0.0f64;
    let mut nb2 = 0.0f64;
    for i in 0..rust_v.len() {
        let a = rust_v[i] as f64;
        let b = py_v[i] as f64;
        let d = (rust_v[i] - py_v[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        dot += a * b;
        na2 += a * a;
        nb2 += b * b;
    }
    let denom = na2.sqrt() * nb2.sqrt();
    let cos = if denom == 0.0 {
        1.0f32
    } else {
        (dot / denom) as f32
    };
    println!(
        "  [PARITY] {:24} | cos_sim = {:.7} | max_abs_err = {:.3e}",
        name, cos, max_abs
    );
    assert!(
        cos >= min_cos,
        "GATE 6 FAIL on {}: cos_sim {:.7} < {:.4}",
        name,
        cos,
        min_cos
    );
    assert!(
        max_abs < max_err_thresh,
        "GATE 6 FAIL on {}: max_abs_err {:.3e} >= {:.1e}",
        name,
        max_abs,
        max_err_thresh
    );
}

fn cmd_verify_golden(dir: &Path) {
    println!("=== Rust Gate 6a: Cross-Stack Python <-> Rust Numerical Parity ===");
    let cfg_str = fs::read_to_string(dir.join("config.json")).expect("failed to read config.json");
    let cfg = DcpaConfig::from_json_str(&cfg_str).expect("invalid config.json");
    let w_bytes = fs::read(dir.join("model.safetensors")).expect("failed to read model.safetensors");
    let weights = DcpaWeights::from_safetensors(cfg, &w_bytes).expect("failed to load model.safetensors");
    let mut engine = DcpaEngine::new(weights, FastWeightPrecision::F32);

    let g_bytes = fs::read(dir.join("golden.safetensors")).expect("failed to read golden.safetensors");
    let golden = parse_safetensors(&g_bytes).expect("failed to parse golden.safetensors");

    let raw_bytes = &golden["raw_bytes"].data_u8;
    let lang_id = golden["lang_id"].data_i64[0] as usize;
    let t_len = raw_bytes.len();
    let d_s = engine.weights.cfg.d_s;
    let d_c = engine.weights.cfg.d_concept;

    // Buffers to collect outputs for post-run comparison (allocated BEFORE hot loop!)
    let mut rust_h_seq = Vec::with_capacity(t_len * d_s);
    let mut rust_prior_probs = Vec::with_capacity(t_len * 256);
    let mut rust_info = Vec::with_capacity(t_len);
    let mut rust_spikes = Vec::with_capacity(t_len);
    let mut rust_f_stem = Vec::new();
    let mut rust_c_slow = Vec::new();
    let mut rust_h_enc = Vec::new();
    let mut rust_c_fused = Vec::new();
    let mut rust_prec_bu = Vec::new();
    let mut rust_ach = Vec::new();
    let mut rust_z_star = Vec::new();
    let mut rust_g_ctx = Vec::new();
    let mut rust_win_pair = Vec::new();
    let mut first_f_stem = vec![0.0f32; d_s];
    let mut have_first = false;

    for (t, &byte) in raw_bytes.iter().enumerate() {
        let force_end = t == t_len - 1;
        let tr = engine.sgsc_ingest_byte(byte, force_end);

        rust_h_seq.extend_from_slice(&engine.sgsc.h);
        rust_prior_probs.extend_from_slice(&engine.scratch.probs_256);
        rust_info.push(tr.info);
        rust_spikes.push(if tr.spiked { 1u8 } else { 0u8 });

        if tr.spiked {
            if !have_first {
                first_f_stem.copy_from_slice(&engine.scratch.vec_ds_e);
                have_first = true;
            }
            rust_f_stem.extend_from_slice(&engine.scratch.vec_ds_e);
            rust_c_slow.extend_from_slice(&engine.scratch.vec_d_a);

            let ctr = engine.step_concept_from_scratch(tr.mean_info, tr.sem_norm);
            rust_h_enc.extend_from_slice(&engine.scratch.cort_h_enc);
            rust_c_fused.extend_from_slice(&engine.scratch.hip_c_fused);
            rust_prec_bu.extend_from_slice(&engine.scratch.hip_prec_bu);
            rust_ach.push(ctr.ach);
            rust_z_star.extend_from_slice(&engine.scratch.vec_d_g);
            rust_g_ctx.extend_from_slice(&engine.scratch.vec_d_h);
            rust_win_pair.push(ctr.win_pair_idx as i64);
        }
    }

    assert_eq!(
        rust_spikes, golden["sgsc.spikes"].data_u8,
        "GATE 6 FAIL: SGSC discrete spike sequence diverged from Python!"
    );
    assert_eq!(
        rust_win_pair, golden["bcg.win_pair"].data_i64,
        "GATE 6 FAIL: BCG-PC winning merge pair sequence diverged from Python!"
    );

    check_vec_parity("sgsc.h_seq", &rust_h_seq, &golden["sgsc.h_seq"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.prior_probs", &rust_prior_probs, &golden["sgsc.prior_probs"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.info", &rust_info, &golden["sgsc.info"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.f_stem", &rust_f_stem, &golden["sgsc.f_stem"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.c_slow", &rust_c_slow, &golden["sgsc.c_slow"].data_f32, 1e-5, 0.9999);

    check_vec_parity("cortical.h_enc", &rust_h_enc, &golden["cortical.h_enc"].data_f32, 1e-5, 0.9999);
    engine.cortical_decode_styled(lang_id, 0);
    check_vec_parity("cortical.z_dec", &engine.scratch.cort_z_dec, &golden["cortical.z_dec"].data_f32, 1e-5, 0.9999);
    check_vec_parity("cortical.copy_stem", &engine.scratch.cort_copy_out, &golden["cortical.copy_stem"].data_f32, 1e-5, 0.9999);

    check_vec_parity("hip.c_fused", &rust_c_fused, &golden["hip.c_fused"].data_f32, 1e-5, 0.9999);
    check_vec_parity("hip.prec_bu", &rust_prec_bu, &golden["hip.prec_bu"].data_f32, 1e-5, 0.9999);
    check_vec_parity("hip.ach", &rust_ach, &golden["hip.ach"].data_f32, 1e-5, 0.9999);

    dcpa_core_rs::simd::linear_f32(
        &first_f_stem,
        &engine.weights.hip_r_lex,
        None,
        &mut engine.scratch.dg_scores,
    );
    let s = engine.weights.cfg.s_active;
    let mut idx = vec![0usize; s];
    let mut val = vec![f32::NEG_INFINITY; s];
    for (k_i, &sc) in engine.scratch.dg_scores.iter().enumerate() {
        if sc > val[s - 1] {
            let mut p = s - 1;
            while p > 0 && sc > val[p - 1] {
                val[p] = val[p - 1];
                idx[p] = idx[p - 1];
                p -= 1;
            }
            val[p] = sc;
            idx[p] = k_i;
        }
    }
    let mut sq = 0.0f32;
    for v in val.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
        sq += *v * *v;
    }
    let inv = 1.0 / (sq.sqrt() + 1e-8);
    for v in val.iter_mut() {
        *v *= inv;
    }
    let mut hip_c_after = vec![0.0f32; d_c];
    engine.fast_mem.read_c_from_l(&idx, &val, &mut hip_c_after);
    check_vec_parity("hip.c_after_write", &hip_c_after, &golden["hip.c_after_write"].data_f32, 1e-5, 0.9999);

    check_vec_parity("bcg.z_star", &rust_z_star, &golden["bcg.z_star"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.g_ctx", &rust_g_ctx, &golden["bcg.g_ctx"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_slots", &engine.graph.slots, &golden["bcg.final_slots"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_edges", &engine.graph.edges, &golden["bcg.final_edges"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_salience", &engine.graph.salience, &golden["bcg.final_salience"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_phase", &engine.graph.phase, &golden["bcg.final_phase"].data_f32, 1e-5, 0.9999);

    engine.broca_cq_init();
    let (rho_out, _done) = engine.broca_cq_step_and_retrieve(lang_id);
    check_vec_parity("pmd.cq_z_out", &engine.scratch.pmd_z_out, &golden["pmd.cq_z_out"].data_f32, 1e-5, 0.9999);
    check_vec_parity("pmd.cq_delta_syn", &engine.scratch.pmd_delta_syn, &golden["pmd.cq_delta_syn"].data_f32, 1e-5, 0.9999);
    check_vec_parity("pmd.cq_f_out", &engine.scratch.pmd_f_out, &golden["pmd.cq_f_out"].data_f32, 1e-5, 0.9999);
    check_vec_parity("pmd.cq_rho_out", &[rho_out], &golden["pmd.cq_rho_out"].data_f32, 1e-5, 0.9999);

    let mut motor_logits = Vec::with_capacity(4 * 256);
    let mut motor_bytes = Vec::with_capacity(4);
    let mut motor_charge = Vec::with_capacity(4);
    let mut prev_b = 0u8;
    for tau_step in 1..=4 {
        let (pred_b, charge_e, _) = engine.pmd_step_byte(prev_b, lang_id, tau_step);
        motor_logits.extend_from_slice(&engine.scratch.logits_256);
        motor_bytes.push(pred_b);
        motor_charge.push(charge_e);
        prev_b = pred_b;
    }
    assert_eq!(motor_bytes, golden["pmd.motor_bytes"].data_u8, "GATE 6 FAIL: motor emitted bytes differ!");
    check_vec_parity("pmd.motor_logits", &motor_logits, &golden["pmd.motor_logits"].data_f32, 1e-5, 0.9999);
    check_vec_parity("pmd.motor_charge", &motor_charge, &golden["pmd.motor_charge"].data_f32, 1e-5, 0.9999);

    let mut flat_xi = Vec::with_capacity(64);
    for row in 0..16 {
        flat_xi.extend_from_slice(&engine.motor.stack_xi[row]);
    }
    check_vec_parity("pmd.final_stack_xi", &flat_xi, &golden["pmd.final_stack_xi"].data_f32, 1e-5, 0.9999);

    println!("[PASS] Rust Gate 6a: ALL 4 modules match Python golden vectors with cos_sim >= 0.9999 & err < 1e-5!");
}

fn cmd_bench(dir: &Path) {
    println!("=== Rust Gate 6b: Single-Core CPU Latency & Zero-Heap-Allocation Proof ===");
    let mut engine = load_engine(dir, FastWeightPrecision::Bf16);
    let cfg = engine.weights.cfg.clone();
    let mut snap = BrainSnapshot::new(&engine.weights);
    println!(
        "Loaded DCPA config: d_s={}, D={}, K={}, d_h={}, s={}, Hippocampus L3={:.2} MB, BrainSnapshot={:.1} KB",
        cfg.d_s,
        cfg.d_concept,
        cfg.k_slots,
        cfg.d_h,
        cfg.s_active,
        (engine.fast_mem.byte_footprint() as f64) / (1024.0 * 1024.0),
        (snap.byte_footprint() as f64) / 1024.0,
    );

    // 1. Benchmark single-step Column-Major 64B-aligned Hippocampus sparse erase-write (< 50 us!)
    let s = cfg.s_active;
    let e_idx: Vec<usize> = (0..s).map(|i| (i * 97) % cfg.d_h).collect();
    let e_val: Vec<f32> = vec![1.0 / (s as f32).sqrt(); s];
    let phi_idx: Vec<usize> = (0..s).map(|i| (i * 131 + 7) % cfg.d_h).collect();
    let phi_val: Vec<f32> = vec![1.0 / (s as f32).sqrt(); s];
    let s_bar: Vec<f32> = vec![0.123; cfg.d_concept];
    let f_tilde: Vec<f32> = vec![-0.456; cfg.d_s];

    for _ in 0..100 {
        engine.fast_mem.advance_step();
        engine.fast_mem.write_slot_inplace(
            0.9,
            &e_idx,
            &e_val,
            &s_bar,
            &phi_idx,
            &phi_val,
            &f_tilde,
            &mut engine.scratch.hip_proj_c,
            &mut engine.scratch.hip_proj_l,
        );
    }

    let iters = 2000usize;
    reset_alloc_counter();
    let t0 = Instant::now();
    for _ in 0..iters {
        engine.fast_mem.advance_step();
        engine.fast_mem.write_slot_inplace(
            0.9,
            &e_idx,
            &e_val,
            &s_bar,
            &phi_idx,
            &phi_val,
            &f_tilde,
            &mut engine.scratch.hip_proj_c,
            &mut engine.scratch.hip_proj_l,
        );
    }
    let hip_us = (t0.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    let hip_allocs = read_alloc_counter();
    println!(
        "  [BENCH] Hippocampus sparse column Householder erase-write: {:.2} us/step (threshold < 50 us), allocs = {}",
        hip_us, hip_allocs
    );
    assert_eq!(hip_allocs, 0, "GATE 6 FAIL: Hippocampus write allocated {} times!", hip_allocs);
    assert!(
        hip_us < 50.0,
        "GATE 6 FAIL: Hippocampus sparse erase-write took {:.2} us >= 50 us!",
        hip_us
    );

    // 2. Benchmark O(1) ~22 KB BrainSnapshot fork/restore (< 5 us, 0 allocs!)
    reset_alloc_counter();
    let t_snap = Instant::now();
    for _ in 0..iters {
        engine.capture_snapshot(&mut snap);
        engine.restore_snapshot(&snap);
    }
    let snap_us = (t_snap.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    let snap_allocs = read_alloc_counter();
    println!(
        "  [BENCH] O(1) BrainSnapshot capture+restore ({:.1} KB): {:.2} us/roundtrip, allocs = {}",
        (snap.byte_footprint() as f64) / 1024.0,
        snap_us,
        snap_allocs
    );
    assert_eq!(snap_allocs, 0, "GATE 6 FAIL: BrainSnapshot allocated {} times!", snap_allocs);

    // 3. Benchmark streaming byte ingest + Mode-1 closed-loop + Efference Copy generation (0 allocs/byte!)
    let sample = b"A zibber is a tiny nocturnal emerald frog! Meanwhile the river flowed past the cedar bridge. ";
    let mut out_seg = [0u8; 20];
    let mut rng_state = 0x1234_5678_9ABC_DEF1u64;
    let mut generated_bytes = 0usize;

    for &b in sample.iter() {
        let tr = engine.sgsc_ingest_byte(b, false);
        if tr.spiked {
            let _ = engine.step_concept_from_scratch_with_turn_opts(
                tr.mean_info,
                tr.sem_norm,
                tr.pin_triggered,
                tr.unpin_triggered,
                None,
            );
        }
    }

    reset_alloc_counter();
    let t_stream = Instant::now();
    for i in 0..512 {
        let b = sample[i % sample.len()];
        let tr = engine.sgsc_ingest_byte(b, i == 511);
        if tr.spiked {
            let _ = engine.step_concept_from_scratch_with_turn_opts(
                tr.mean_info,
                tr.sem_norm,
                tr.pin_triggered,
                tr.unpin_triggered,
                None,
            );
        }
    }
    // Exercise closed-loop Mode-1 motor drive + PMD byte steps (strictly 0 allocs)
    let _ = engine.mode1_step_and_retrieve_styled(0, 0);
    let mut prev_b = b'\n';
    for tau in 1..=16 {
        let (nb, _, p_end) = engine.pmd_step_byte_ex(prev_b, 0, 0, tau, false);
        prev_b = nb;
        if p_end >= 0.5 {
            break;
        }
    }
    while generated_bytes < 256 {
        engine.broca_cq_init();
        for _ in 0..cfg.k_slots {
            let (n_written, done, _err) =
                engine.generate_concept_with_efference_copy(1, &mut out_seg, &mut rng_state);
            generated_bytes += n_written;
            if done || generated_bytes >= 256 {
                break;
            }
        }
    }
    let stream_secs = t_stream.elapsed().as_secs_f64();
    let stream_allocs = read_alloc_counter();
    let gen_bps = (generated_bytes as f64) / stream_secs;
    println!(
        "  [BENCH] Hot loop (512B ingest + Mode-1 + {}B Efference-Copy stream gen): {:.1} bytes/sec | heap allocs = {} (0 allocs/byte)",
        generated_bytes, gen_bps, stream_allocs
    );
    assert_eq!(
        stream_allocs, 0,
        "GATE 6 FAIL: Hot streaming loop performed {} heap allocations (must be strictly 0)!",
        stream_allocs
    );
    println!("[PASS] Rust Gate 6b: Single-core latency < 50 us & strictly 0 heap allocations in hot loop!");
}

// =============================================================================
// Stage-5 Protocol & Pure-Rust Safe Whitelist Tool Sandbox (`0x11/0x13/0x12`)
// =============================================================================

const TURN_USER_BYTE: u8 = 0x1E;
const TURN_ASST_BYTE: u8 = 0x1F;
const TURN_EOT_BYTE: u8 = 0x04;
const SCRATCH_CTRL_BYTE: u8 = 0x1D;
const TOOL_CALL_BYTE: u8 = 0x11;
const TOOL_OBS_BYTE: u8 = 0x12;
const TOOL_SUSPEND_BYTE: u8 = 0x13;
const MAX_TOOL_HOPS_PER_TURN: usize = 5;

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DIM: &str = "\x1b[90m";

const BUILTIN_TOOL_SIGNATURES: &[&str] = &[
    "tool calc(expr: str) -> float : evaluate exact math and statistics expression",
    "tool now_hk() -> str : return current Hong Kong UTC+8 ISO timestamp",
    "tool list_files(path: str) -> list : list workspace directory entries",
    "tool read_file_span(path: str, start: int, end: int) -> str : read line range from workspace file",
    "tool word_char_count(text: str) -> dict : count words chars and UTF-8 bytes",
];

fn sanitize_raw_text(text: &str) -> String {
    text.chars()
        .filter(|&ch| {
            let code = ch as u32;
            code >= 0x20 || ch == '\n' || ch == '\t'
        })
        .collect()
}

fn format_user_turn_prompt(user_text: &str) -> Vec<u8> {
    let clean = sanitize_raw_text(user_text);
    let trimmed = clean.trim();
    let mut out = Vec::with_capacity(trimmed.len() + 8);
    out.extend_from_slice(&[TURN_USER_BYTE, b'U', b'\n']);
    out.extend_from_slice(trimmed.as_bytes());
    out.extend_from_slice(&[b'\n', TURN_ASST_BYTE, b'A', b'\n']);
    out
}

fn format_tool_observation_after_suspend(result_text: &str) -> Vec<u8> {
    let clean = sanitize_raw_text(result_text);
    let trimmed = clean.trim();
    let mut out = Vec::with_capacity(trimmed.len() + 10);
    out.extend_from_slice(&[b'\n', TOOL_OBS_BYTE, b'O', b'\n']);
    out.extend_from_slice(trimmed.as_bytes());
    out.extend_from_slice(&[b'\n', TURN_ASST_BYTE, b'A', b'\n']);
    out
}

/// Exact language/domain ID mapping matching `utils/chat_protocol.py` (`4=Code/SQL`, `2=Formal/JA`, etc.).
fn detect_lang_id(text: &str) -> usize {
    let s = text.trim();
    let low = s.to_ascii_lowercase();
    let js_hits = [
        "function ",
        "const ",
        "=>",
        "console.log",
        "export ",
        "async function",
        "=== ",
        "!== ",
    ]
    .iter()
    .filter(|&&kw| s.contains(kw))
    .count();
    let rust_hits = ["fn ", "let mut ", "impl ", "struct ", "-> ", "usize", "Vec<", "pub fn", "match "]
        .iter()
        .filter(|&&kw| s.contains(kw))
        .count();
    let sql_hits = [
        "select ",
        " from ",
        " where ",
        "insert into ",
        "update ",
        "create table ",
        "join ",
        "group by ",
        "order by ",
        "inner join ",
        "left join ",
    ]
    .iter()
    .filter(|&&kw| low.contains(kw))
    .count();
    let py_hits = ["def ", "class ", "import ", "return ", "elif ", "lambda ", "-> int", "-> list"]
        .iter()
        .filter(|&&kw| s.contains(kw))
        .count();
    let fol_hits = [
        "forall ",
        "exists ",
        "|-",
        "|=",
        "<->",
        "==>",
        "\\forall",
        "\\exists",
        "\\implies",
        "\\vdash",
        "theorem ",
        "lemma ",
    ]
    .iter()
    .filter(|&&kw| low.contains(kw))
    .count();
    if low.contains("```javascript")
        || low.contains("```js")
        || js_hits >= 2
        || ((low.contains("javascript") || low.contains("nodejs"))
            && (s.contains("function") || s.contains("const ") || s.contains("=>") || s.contains("函数")))
    {
        return 5;
    }
    if rust_hits >= 2
        || sql_hits >= 2
        || low.contains("```rust")
        || low.contains("```sql")
        || (low.contains("rust") && (s.contains("struct ") || s.contains("fn ")))
    {
        return 4;
    }
    if py_hits >= 2
        || low.contains("```python")
        || (low.contains("python") && (s.contains("def ") || s.contains("函数") || low.contains("function")))
    {
        return 3;
    }
    if ["∀", "∃", "⊢", "⊨", "⟹", "⟺"].iter().any(|&sym| s.contains(sym))
        || low.contains("```lean")
        || low.contains("```coq")
        || low.contains("```fol")
        || fol_hits >= 2
    {
        return 2;
    }
    let ja_kana = s
        .chars()
        .filter(|&c| ('\u{3040}'..='\u{309f}').contains(&c) || ('\u{30a0}'..='\u{30ff}').contains(&c))
        .count();
    if ja_kana >= 2 {
        return 2;
    }
    let cjk = s
        .chars()
        .filter(|&c| ('\u{4e00}'..='\u{9fff}').contains(&c) || ('\u{3400}'..='\u{4dbf}').contains(&c))
        .count();
    if cjk >= 2 {
        return 1;
    }
    0
}

fn lang_name(id: usize) -> &'static str {
    match id {
        0 => "en",
        1 => "zh",
        2 => "ja",
        3 => "py",
        4 => "rs",
        5 => "js",
        _ => "en",
    }
}

fn style_name(id: usize) -> &'static str {
    match id {
        0 => "neutral",
        1 => "memo",
        2 => "expository",
        3 => "narrative",
        4 => "strict_code_tool",
        _ => "neutral",
    }
}

struct MathParser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> MathParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            s: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut v = self.parse_term()?;
        loop {
            self.skip_ws();
            if self.pos < self.s.len() && self.s[self.pos] == b'+' {
                self.pos += 1;
                v += self.parse_term()?;
            } else if self.pos < self.s.len() && self.s[self.pos] == b'-' {
                self.pos += 1;
                v -= self.parse_term()?;
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut v = self.parse_power()?;
        loop {
            self.skip_ws();
            if self.pos + 1 < self.s.len() && self.s[self.pos] == b'/' && self.s[self.pos + 1] == b'/' {
                self.pos += 2;
                let rhs = self.parse_power()?;
                if rhs == 0.0 {
                    return Err("ZeroDivisionError".to_string());
                }
                v = (v / rhs).floor();
            } else if self.pos < self.s.len() && self.s[self.pos] == b'*' && (self.pos + 1 >= self.s.len() || self.s[self.pos + 1] != b'*') {
                self.pos += 1;
                v *= self.parse_power()?;
            } else if self.pos < self.s.len() && self.s[self.pos] == b'/' {
                self.pos += 1;
                let rhs = self.parse_power()?;
                if rhs == 0.0 {
                    return Err("ZeroDivisionError".to_string());
                }
                v /= rhs;
            } else if self.pos < self.s.len() && self.s[self.pos] == b'%' {
                self.pos += 1;
                let rhs = self.parse_power()?;
                if rhs == 0.0 {
                    return Err("ZeroDivisionError".to_string());
                }
                v %= rhs;
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn parse_power(&mut self) -> Result<f64, String> {
        let base = self.parse_unary()?;
        self.skip_ws();
        if self.pos + 1 < self.s.len() && self.s[self.pos] == b'*' && self.s[self.pos + 1] == b'*' {
            self.pos += 2;
            let exp = self.parse_power()?;
            if exp.abs() > 100.0 {
                return Err("Exponent magnitude capped at 100".to_string());
            }
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<f64, String> {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b'+' {
            self.pos += 1;
            return self.parse_unary();
        }
        if self.pos < self.s.len() && self.s[self.pos] == b'-' {
            self.pos += 1;
            return Ok(-self.parse_unary()?);
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<f64, String> {
        self.skip_ws();
        if self.pos >= self.s.len() {
            return Err("Unexpected end of math expression".to_string());
        }
        if self.s[self.pos] == b'(' {
            self.pos += 1;
            let v = self.parse_expr()?;
            self.skip_ws();
            if self.pos >= self.s.len() || self.s[self.pos] != b')' {
                return Err("Unclosed '(' in math expression".to_string());
            }
            self.pos += 1;
            return Ok(v);
        }
        if self.s[self.pos].is_ascii_alphabetic() || self.s[self.pos] == b'_' {
            let start = self.pos;
            while self.pos < self.s.len() && (self.s[self.pos].is_ascii_alphanumeric() || self.s[self.pos] == b'_') {
                self.pos += 1;
            }
            let ident = std::str::from_utf8(&self.s[start..self.pos]).unwrap_or("");
            self.skip_ws();
            if self.pos < self.s.len() && self.s[self.pos] == b'(' {
                self.pos += 1;
                let mut args = Vec::new();
                loop {
                    self.skip_ws();
                    if self.pos < self.s.len() && self.s[self.pos] == b'[' {
                        self.pos += 1;
                        loop {
                            self.skip_ws();
                            if self.pos < self.s.len() && self.s[self.pos] == b']' {
                                self.pos += 1;
                                break;
                            }
                            args.push(self.parse_expr()?);
                            self.skip_ws();
                            if self.pos < self.s.len() && self.s[self.pos] == b',' {
                                self.pos += 1;
                            }
                        }
                    } else if self.pos < self.s.len() && self.s[self.pos] != b')' {
                        args.push(self.parse_expr()?);
                    }
                    self.skip_ws();
                    if self.pos < self.s.len() && self.s[self.pos] == b',' {
                        self.pos += 1;
                        continue;
                    }
                    break;
                }
                if self.pos >= self.s.len() || self.s[self.pos] != b')' {
                    return Err("Unclosed function '(' in math expression".to_string());
                }
                self.pos += 1;
                return match (ident, args.as_slice()) {
                    ("abs", &[x]) => Ok(x.abs()),
                    ("sqrt", &[x]) => Ok(x.sqrt()),
                    ("sin", &[x]) => Ok(x.sin()),
                    ("cos", &[x]) => Ok(x.cos()),
                    ("tan", &[x]) => Ok(x.tan()),
                    ("exp", &[x]) => Ok(x.exp()),
                    ("log" | "ln", &[x]) => Ok(x.ln()),
                    ("log10", &[x]) => Ok(x.log10()),
                    ("floor", &[x]) => Ok(x.floor()),
                    ("ceil", &[x]) => Ok(x.ceil()),
                    ("round", &[x]) => Ok(x.round()),
                    ("pow", &[a, b]) if b.abs() <= 100.0 => Ok(a.powf(b)),
                    ("sum", xs) => Ok(xs.iter().sum()),
                    ("mean", xs) if !xs.is_empty() => Ok(xs.iter().sum::<f64>() / (xs.len() as f64)),
                    ("min", xs) if !xs.is_empty() => Ok(xs.iter().copied().fold(f64::INFINITY, f64::min)),
                    ("max", xs) if !xs.is_empty() => Ok(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                    _ => Err(format!("Unsupported math call `{}`", ident)),
                };
            }
            return match ident {
                "pi" => Ok(std::f64::consts::PI),
                "e" => Ok(std::f64::consts::E),
                "tau" => Ok(std::f64::consts::TAU),
                _ => Err(format!("Disallowed identifier `{}`", ident)),
            };
        }
        let start = self.pos;
        while self.pos < self.s.len()
            && (self.s[self.pos].is_ascii_digit()
                || self.s[self.pos] == b'.'
                || self.s[self.pos] == b'e'
                || self.s[self.pos] == b'E'
                || ((self.s[self.pos] == b'+' || self.s[self.pos] == b'-')
                    && self.pos > start
                    && (self.s[self.pos - 1] == b'e' || self.s[self.pos - 1] == b'E')))
        {
            self.pos += 1;
        }
        let num_s = std::str::from_utf8(&self.s[start..self.pos]).unwrap_or("");
        num_s
            .parse::<f64>()
            .map_err(|_| format!("Invalid numeric literal `{}`", num_s))
    }
}

fn eval_math_safe(expr: &str) -> Result<f64, String> {
    let mut p = MathParser::new(expr.trim());
    let val = p.parse_expr()?;
    p.skip_ws();
    if p.pos != p.s.len() {
        return Err("Trailing characters in math expression".to_string());
    }
    if !val.is_finite() {
        return Err("Non-finite math result".to_string());
    }
    Ok((val * 1e10).round() / 1e10)
}

fn unquote_arg(raw: &str) -> Result<String, String> {
    let s = raw.trim();
    let s = if let Some((_, rhs)) = s.split_once('=') {
        rhs.trim()
    } else {
        s
    };
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        Ok(s[1..s.len() - 1].replace("\\\"", "\"").replace("\\'", "'").replace("\\n", "\n"))
    } else {
        Err("Expected quoted string literal argument".to_string())
    }
}

fn now_hk_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
        + 8 * 3600;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    // Howard Hinnant civil-from-days algorithm
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+08:00", year, m, d, hh, mm, ss)
}

fn execute_tool_call_safe_rust(call_str: &str, workspace_root: &Path) -> (bool, String) {
    let raw = sanitize_raw_text(call_str);
    let s = raw.trim();
    let Some(open) = s.find('(') else {
        return (false, "Error: SyntaxError: missing '(' in tool call".to_string());
    };
    if !s.ends_with(')') {
        return (false, "Error: SyntaxError: missing closing ')' in tool call".to_string());
    }
    let fn_name = s[..open].trim();
    let args_str = s[open + 1..s.len() - 1].trim();

    match fn_name {
        "calc" => match unquote_arg(args_str).and_then(|e| eval_math_safe(&e)) {
            Ok(v) => {
                if (v - v.trunc()).abs() < 1e-10 {
                    (true, format!("{:.1}", v))
                } else {
                    (true, format!("{}", v))
                }
            }
            Err(e) => (false, format!("Error: ValueError: {}", e)),
        },
        "now_hk" => {
            if !args_str.is_empty() {
                return (false, "Error: ValueError: now_hk() takes no arguments".to_string());
            }
            (true, now_hk_iso())
        }
        "word_char_count" => match unquote_arg(args_str) {
            Ok(txt) => {
                let words = txt.split_whitespace().count();
                let chars = txt.chars().count();
                let bytes = txt.len();
                (
                    true,
                    format!("{{'words': {}, 'chars': {}, 'bytes': {}}}", words, chars, bytes),
                )
            }
            Err(e) => (false, format!("Error: ValueError: {}", e)),
        },
        "list_files" => {
            let rel = if args_str.is_empty() {
                ".".to_string()
            } else {
                match unquote_arg(args_str) {
                    Ok(v) => v,
                    Err(e) => return (false, format!("Error: ValueError: {}", e)),
                }
            };
            let Ok(root_canon) = workspace_root.canonicalize() else {
                return (false, "Error: IOError: invalid workspace root".to_string());
            };
            let Ok(target) = root_canon.join(&rel).canonicalize() else {
                return (false, format!("Error: FileNotFoundError: {}", rel));
            };
            if !target.starts_with(&root_canon) || !target.is_dir() {
                return (false, format!("Error: SecurityPolicy: path outside workspace `{}`", rel));
            }
            let mut entries = Vec::new();
            if let Ok(rd) = fs::read_dir(&target) {
                for entry in rd.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !name.starts_with('.') {
                        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                        entries.push(if is_dir { format!("{}/", name) } else { name });
                    }
                }
            }
            entries.sort();
            entries.truncate(64);
            (true, format!("{:?}", entries))
        }
        "read_file_span" => {
            let parts: Vec<&str> = args_str.split(',').collect();
            if parts.is_empty() {
                return (false, "Error: ValueError: missing path argument".to_string());
            }
            let Ok(rel) = unquote_arg(parts[0]) else {
                return (false, "Error: ValueError: invalid path string".to_string());
            };
            let parse_int = |s: &str, default: usize| -> usize {
                let rhs = s.split_once('=').map(|x| x.1).unwrap_or(s).trim();
                rhs.parse::<usize>().unwrap_or(default)
            };
            let start = if parts.len() >= 2 { parse_int(parts[1], 1) } else { 1 };
            let end = if parts.len() >= 3 { parse_int(parts[2], 40) } else { 40 };
            if start < 1 || end < start || end - start > 200 {
                return (false, "Error: ValueError: invalid line span (max 200 lines)".to_string());
            }
            let Ok(root_canon) = workspace_root.canonicalize() else {
                return (false, "Error: IOError: invalid workspace root".to_string());
            };
            let Ok(target) = root_canon.join(&rel).canonicalize() else {
                return (false, format!("Error: FileNotFoundError: {}", rel));
            };
            if !target.starts_with(&root_canon) || !target.is_file() {
                return (false, format!("Error: SecurityPolicy: path outside workspace `{}`", rel));
            }
            let Ok(txt) = fs::read_to_string(&target) else {
                return (false, "Error: IOError: failed to read UTF-8 file".to_string());
            };
            let out: Vec<String> = txt
                .lines()
                .skip(start - 1)
                .take(end - start + 1)
                .enumerate()
                .map(|(i, ln)| format!("L{}: {}", start + i, ln))
                .collect();
            (true, out.join("\n"))
        }
        _ => (
            false,
            format!("Error: SecurityPolicy: unregistered or forbidden tool `{}`", fn_name),
        ),
    }
}

struct IngestStats {
    spikes: usize,
    max_ach: f32,
    ponder_skips: usize,
}

fn ingest_slice(engine: &mut DcpaEngine, bytes: &[u8], ach_override: Option<f32>) -> IngestStats {
    let mut spikes = 0usize;
    let mut max_ach = 0.0f32;
    let mut ponder_skips = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        let tr = engine.sgsc_ingest_byte(b, i + 1 == bytes.len());
        if tr.spiked {
            spikes += 1;
            let ctr = engine.step_concept_from_scratch_with_turn_opts(
                tr.mean_info,
                tr.sem_norm,
                tr.pin_triggered,
                tr.unpin_triggered,
                ach_override,
            );
            if ctr.ach > max_ach {
                max_ach = ctr.ach;
            }
            if ctr.ponder_steps == 0 {
                ponder_skips += 1;
            }
        }
    }
    IngestStats {
        spikes,
        max_ach,
        ponder_skips,
    }
}

fn sample_next_byte(logits: &[f32; 256], probs_scratch: &mut [f32; 256], temp: f32, top_p: f32, rng: &mut u64) -> u8 {
    if temp < 0.02 {
        let mut best_b = 0u8;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_b = i as u8;
            }
        }
        return best_b;
    }

    let inv_t = 1.0 / temp;
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max_l {
            max_l = v;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..256 {
        let e = ((logits[i] - max_l) * inv_t).exp();
        probs_scratch[i] = e;
        sum += e;
    }
    let inv_sum = 1.0 / sum.max(1e-12);
    let mut order = [0u8; 256];
    for i in 0..256 {
        probs_scratch[i] *= inv_sum;
        order[i] = i as u8;
    }

    if top_p < 0.999 {
        order.sort_unstable_by(|&a, &b| {
            probs_scratch[b as usize]
                .partial_cmp(&probs_scratch[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut keep_sum = 0.0f32;
        let mut cutoff = 256usize;
        for (rank, &idx) in order.iter().enumerate() {
            let p = probs_scratch[idx as usize];
            cum += p;
            keep_sum += p;
            if cum >= top_p {
                cutoff = rank + 1;
                break;
            }
        }
        for &idx in &order[cutoff..] {
            probs_scratch[idx as usize] = 0.0;
        }
        let inv_keep = 1.0 / keep_sum.max(1e-12);
        for p in probs_scratch.iter_mut() {
            *p *= inv_keep;
        }
    }

    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
    let mut acc = 0.0f32;
    for i in 0..256 {
        acc += probs_scratch[i];
        if acc >= u {
            return i as u8;
        }
    }
    order[0]
}

fn cmd_chat(dir: &Path) {
    let (_, weight_path) = resolve_export_paths(dir);
    let mut engine = load_engine(dir, FastWeightPrecision::Bf16);
    let workspace_root = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Startup Hippocampus Tool Pre-Registration (Spec 5.3)
    for sig in BUILTIN_TOOL_SIGNATURES {
        let mut line = Vec::with_capacity(sig.len() + 1);
        line.extend_from_slice(sig.as_bytes());
        line.push(b'\n');
        let _ = ingest_slice(&mut engine, &line, None);
    }
    let m_tools_base = engine.fast_mem.clone_snapshot();
    engine.reset_session();

    println!(
        "DCPA Local Rust Engine Ready ({} | zero GPU, zero Python, {:.2} MB BF16 L3 Hippocampus + 4-row SIMD).",
        weight_path.file_name().and_then(|s| s.to_str()).unwrap_or("safetensors"),
        (engine.fast_mem.byte_footprint() as f64) / (1024.0 * 1024.0)
    );
    println!("Commands: /style <name> | /lang auto|en|zh|ja|formal|py|rs|sql|js | /think show|dim|hide | /probe on|off | /teach <text> | /tools | /reset | /wipe | /quit");

    let stdin = io::stdin();
    let mut rng_state = 0xCAFE_BABE_1337_0001u64;
    let mut lang_override: Option<usize> = None;
    let mut style_id: usize = 0;
    let mut think_mode = "dim".to_string();
    let mut probe_on = true;
    let temp: f32 = 0.55;
    let top_p: f32 = 0.92;
    let max_new_bytes: usize = 512;

    loop {
        print!("\ndcpa-rs> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        let Ok(n_read) = stdin.lock().read_line(&mut line) else {
            break;
        };
        if n_read == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Slash command router
        if trimmed.starts_with('/') {
            let mut parts = trimmed.splitn(2, char::is_whitespace);
            let cmd = parts.next().unwrap_or("");
            let arg = parts.next().unwrap_or("").trim();
            match cmd {
                "/quit" | "/exit" => break,
                "/tools" => {
                    for sig in BUILTIN_TOOL_SIGNATURES {
                        println!("  - {}", sig);
                    }
                }
                "/reset" => {
                    engine.reset_session();
                    engine.fast_mem.restore_from(&m_tools_base);
                    println!("[reset] 8-slot graph cleared; Hippocampus restored to M_tools_base.");
                }
                "/wipe" => {
                    engine.reset_session();
                    engine.fast_mem.wipe_zero();
                    println!("[wipe] 8-slot graph AND Hippocampus fast weights zeroed.");
                }
                "/teach" => {
                    if arg.is_empty() {
                        println!("Usage: /teach <definition or fact>");
                    } else {
                        let clean = sanitize_raw_text(arg);
                        let st = ingest_slice(&mut engine, clean.as_bytes(), None);
                        println!(
                            "[taught] {} bytes -> {} concepts (max_ACh={:.3})",
                            clean.len(),
                            st.spikes,
                            st.max_ach
                        );
                    }
                }
                "/style" => {
                    style_id = match arg {
                        "neutral" | "0" => 0,
                        "memo" | "1" => 1,
                        "expository" | "2" => 2,
                        "narrative" | "3" => 3,
                        "strict_code_tool" | "code" | "tool" | "4" => 4,
                        _ => {
                            println!("Choices: neutral | memo | expository | narrative | strict_code_tool");
                            continue;
                        }
                    };
                    println!("[style] set to {} ({})", style_name(style_id), style_id);
                }
                "/lang" => match arg {
                    "auto" => {
                        lang_override = None;
                        println!("[lang] auto");
                    }
                    "en" => {
                        lang_override = Some(0);
                        println!("[lang] en (0)");
                    }
                    "zh" => {
                        lang_override = Some(1);
                        println!("[lang] zh (1)");
                    }
                    "ja" => {
                        lang_override = Some(2);
                        println!("[lang] ja (2)");
                    }
                    "formal" | "fol" | "math" | "lean" => {
                        lang_override = Some(2);
                        println!("[lang] formal (2)");
                    }
                    "py" | "python" => {
                        lang_override = Some(3);
                        println!("[lang] py (3)");
                    }
                    "rs" | "rust" => {
                        lang_override = Some(4);
                        println!("[lang] rs (4)");
                    }
                    "sql" => {
                        lang_override = Some(4);
                        println!("[lang] sql (4)");
                    }
                    "js" | "javascript" => {
                        lang_override = Some(5);
                        println!("[lang] js (5)");
                    }
                    _ => println!("Choices: auto | en | zh | ja | formal | py | rs | sql | js"),
                },
                "/think" => {
                    if matches!(arg, "show" | "dim" | "hide") {
                        think_mode = arg.to_string();
                        println!("[think] {}", think_mode);
                    } else {
                        println!("Choices: show | dim | hide");
                    }
                }
                "/probe" => {
                    probe_on = matches!(arg, "on" | "1" | "true");
                    println!("[probe] {}", if probe_on { "on" } else { "off" });
                }
                _ => println!("Unknown command `{}`", cmd),
            }
            continue;
        }

        let t0 = Instant::now();
        let lang_id = lang_override.unwrap_or_else(|| detect_lang_id(trimmed));
        let mut active_style_id = style_id;
        let prompt_bytes = format_user_turn_prompt(trimmed);
        let in_stats = ingest_slice(&mut engine, &prompt_bytes, None);

        let _ = engine.mode1_step_and_retrieve_styled(lang_id, active_style_id);
        let mut prev_b = *prompt_bytes.last().unwrap_or(&b'\n');
        let mut recent_bytes: Vec<u8> = prompt_bytes[prompt_bytes.len().saturating_sub(3)..].to_vec();
        let mut emitted_raw: Vec<u8> = Vec::new();
        let mut utf8_pending: Vec<u8> = Vec::new();
        let mut utf8_rem = 0usize;
        let mut tau_step = 1usize;

        let mut in_tool_call = false;
        let mut tool_call_buf: Vec<u8> = Vec::new();
        let mut tool_hops = 0usize;
        let mut in_scratch = false;
        let mut scratch_buf: Vec<u8> = Vec::new();
        let mut stopped_on_eot = false;
        let mut total_emitted = 0usize;
        let mut probs_buf = [0.0f32; 256];

        for _ in 0..max_new_bytes {
            let extra_reset = recent_bytes.len() >= 3
                && recent_bytes[recent_bytes.len() - 3..] == [SCRATCH_CTRL_BYTE, b'E', b'\n'];
            let (_, _, p_end) =
                engine.pmd_step_byte_ex(prev_b, lang_id, active_style_id, tau_step, extra_reset);

            let eff_temp = if in_tool_call || lang_id == 3 || lang_id == 4 || lang_id == 5 {
                0.0
            } else {
                temp
            };
            let next_b = sample_next_byte(
                &engine.scratch.logits_256,
                &mut probs_buf,
                eff_temp,
                top_p,
                &mut rng_state,
            );

            if next_b == TURN_EOT_BYTE {
                stopped_on_eot = true;
                // Advance 1 stack-reset tick on 0x04 so 128B stack Xi_t clears cleanly
                let _ = engine.pmd_step_byte_ex(TURN_EOT_BYTE, lang_id, active_style_id, 1, false);
                break;
            }

            emitted_raw.push(next_b);
            total_emitted += 1;
            recent_bytes.push(next_b);
            if recent_bytes.len() > 6 {
                recent_bytes.drain(..recent_bytes.len() - 6);
            }

            if next_b < 0x80 {
                utf8_rem = 0;
            } else if (next_b & 0xE0) == 0xC0 {
                utf8_rem = 1;
            } else if (next_b & 0xF0) == 0xE0 {
                utf8_rem = 2;
            } else if (next_b & 0xF8) == 0xF0 {
                utf8_rem = 3;
            } else if (next_b & 0xC0) == 0x80 {
                utf8_rem = utf8_rem.saturating_sub(1);
            } else {
                utf8_rem = 0;
            }

            if next_b == TOOL_CALL_BYTE {
                in_tool_call = true;
                tool_call_buf.clear();
                active_style_id = 4;
                prev_b = next_b;
                tau_step = 1;
                continue;
            }

            if next_b == TOOL_SUSPEND_BYTE {
                let mut call_slice = tool_call_buf.as_slice();
                if call_slice.starts_with(b"C\n") {
                    call_slice = &call_slice[2..];
                } else if call_slice.starts_with(b"C") {
                    call_slice = &call_slice[1..];
                }
                let call_str = String::from_utf8_lossy(call_slice).trim().to_string();
                let (circuit_broken, res_str) = if tool_hops >= MAX_TOOL_HOPS_PER_TURN {
                    (
                        true,
                        format!(
                            "Error: CircuitBreaker: exceeded max {} consecutive tool calls in a single turn",
                            MAX_TOOL_HOPS_PER_TURN
                        ),
                    )
                } else {
                    tool_hops += 1;
                    let (_ok, r) = execute_tool_call_safe_rust(&call_str, &workspace_root);
                    (false, r)
                };
                if probe_on {
                    print!(
                        "{}[tool #{}: {} -> {}]{} ",
                        ANSI_DIM, tool_hops, call_str, res_str, ANSI_RESET
                    );
                    let _ = io::stdout().flush();
                }

                let obs_bytes = format_tool_observation_after_suspend(&res_str);
                if !emitted_raw.is_empty() {
                    let _ = ingest_slice(&mut engine, &emitted_raw, None);
                    emitted_raw.clear();
                }
                let _ = ingest_slice(&mut engine, &obs_bytes, Some(0.0));

                in_tool_call = false;
                tool_call_buf.clear();
                utf8_pending.clear();
                utf8_rem = 0;
                active_style_id = style_id;
                engine.motor.stack_xi = [[0.0f32; 4]; 16];
                let _ = engine.mode1_step_and_retrieve_styled(lang_id, active_style_id);
                prev_b = *obs_bytes.last().unwrap_or(&b'\n');
                recent_bytes = obs_bytes[obs_bytes.len().saturating_sub(3)..].to_vec();
                tau_step = 1;
                if circuit_broken {
                    break;
                }
                continue;
            }

            if in_tool_call {
                tool_call_buf.push(next_b);
            } else if next_b == SCRATCH_CTRL_BYTE {
                if !in_scratch {
                    in_scratch = true;
                    scratch_buf.clear();
                } else {
                    let s_txt = String::from_utf8_lossy(&scratch_buf);
                    let body = s_txt.strip_prefix("S\n").unwrap_or(&s_txt);
                    if think_mode == "show" {
                        print!("[think: {}] ", body.trim());
                        let _ = io::stdout().flush();
                    } else if think_mode == "dim" {
                        print!("{}[think: {}]{} ", ANSI_DIM, body.trim(), ANSI_RESET);
                        let _ = io::stdout().flush();
                    }
                    scratch_buf.clear();
                    in_scratch = false;
                }
            } else if in_scratch {
                scratch_buf.push(next_b);
            } else {
                let skip_close_e = recent_bytes.len() >= 2
                    && recent_bytes[recent_bytes.len() - 2] == SCRATCH_CTRL_BYTE
                    && next_b == b'E';
                let skip_close_nl = recent_bytes.len() >= 3
                    && recent_bytes[recent_bytes.len() - 3..] == [SCRATCH_CTRL_BYTE, b'E', b'\n'];
                if !skip_close_e && !skip_close_nl {
                    utf8_pending.push(next_b);
                    if utf8_rem == 0 {
                        print!("{}", String::from_utf8_lossy(&utf8_pending));
                        let _ = io::stdout().flush();
                        utf8_pending.clear();
                    }
                }
            }

            prev_b = next_b;
            tau_step += 1;
            let hit_scratch_close = recent_bytes.len() >= 3
                && recent_bytes[recent_bytes.len() - 3..] == [SCRATCH_CTRL_BYTE, b'E', b'\n'];
            let seg_done = p_end >= 0.5 || tau_step > engine.weights.cfg.max_seg_len;
            if hit_scratch_close || seg_done {
                if !in_tool_call && !emitted_raw.is_empty() && utf8_rem == 0 {
                    let _ = ingest_slice(&mut engine, &emitted_raw, None);
                    emitted_raw.clear();
                }
                if utf8_rem == 0 || tau_step > engine.weights.cfg.max_seg_len + 3 {
                    let _ = engine.mode1_step_and_retrieve_styled(lang_id, active_style_id);
                    tau_step = 1;
                }
            }
        }

        if !utf8_pending.is_empty() {
            print!("{}", String::from_utf8_lossy(&utf8_pending));
        }
        if !emitted_raw.is_empty() || stopped_on_eot {
            emitted_raw.push(TURN_EOT_BYTE);
            emitted_raw.push(b'\n');
            let _ = ingest_slice(&mut engine, &emitted_raw, None);
        }
        println!();
        if probe_on {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let skip_pct = 100.0 * (in_stats.ponder_skips as f64) / (in_stats.spikes.max(1) as f64);
            println!(
                "{}[{}|{} | in={}B->{} segs | out={}B | max_ACh={:.2} | ponder_skip={:.0}% | pinned={} | tools={} | {:.1}ms]{}",
                ANSI_DIM,
                lang_name(lang_id),
                style_name(style_id),
                prompt_bytes.len(),
                in_stats.spikes,
                total_emitted,
                in_stats.max_ach,
                skip_pct,
                engine.graph.slot0_pinned,
                tool_hops,
                ms,
                ANSI_RESET
            );
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  dcpa-cli verify-golden <export_dir>");
        eprintln!("  dcpa-cli bench <export_dir_or_safetensors>");
        eprintln!("  dcpa-cli chat <export_dir_or_safetensors>");
        std::process::exit(1);
    }
    let dir = Path::new(&args[2]);
    match args[1].as_str() {
        "verify-golden" => cmd_verify_golden(dir),
        "bench" => cmd_bench(dir),
        "chat" => cmd_chat(dir),
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn test_safe_math_and_tool_sandbox() {
        assert!((eval_math_safe("17 * 23 + sqrt(144) - 2**3").unwrap() - 395.0).abs() < 1e-9);
        assert!((eval_math_safe("mean([10, 20, 30, 40])").unwrap() - 25.0).abs() < 1e-9);
        assert!(eval_math_safe("2 ** 500").is_err());
        assert!(eval_math_safe("__import__('os')").is_err());

        let cwd = Path::new(".");
        let (ok, res) = execute_tool_call_safe_rust("calc(\"6 * 7\")", cwd);
        assert!(ok, "res={}", res);
        assert_eq!(res, "42.0");

        let (ok_wc, res_wc) = execute_tool_call_safe_rust("word_char_count(\"你好 world\")", cwd);
        assert!(ok_wc);
        assert!(res_wc.contains("'words': 2") && res_wc.contains("'chars': 8"));

        let (ok_bad, _) = execute_tool_call_safe_rust("system(\"ls\")", cwd);
        assert!(!ok_bad);
    }

    #[test]
    fn test_lang_detection_matches_python_protocol() {
        assert_eq!(detect_lang_id("Explain causal byte segmentation."), 0);
        assert_eq!(detect_lang_id("请解释海马体三因子可塑性机制"), 1);
        assert_eq!(detect_lang_id("海馬の高速重みについて説明してください"), 2);
        assert_eq!(detect_lang_id("∀x ∈ S, ∃y: P(x, y) ⟹ Q(x)"), 2);
        assert_eq!(detect_lang_id("forall x in S, exists y: P(x, y) |- Q(x)"), 2);
        assert_eq!(detect_lang_id("def foo(x):\n    return x + 1"), 3);
        assert_eq!(detect_lang_id("pub fn step(&mut self) -> usize { let mut x = 1; x }"), 4);
        assert_eq!(
            detect_lang_id("SELECT u.id, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id;"),
            4
        );
        assert_eq!(detect_lang_id("const sum = (xs) => xs.reduce((a, b) => a + b, 0);"), 5);
    }
}
