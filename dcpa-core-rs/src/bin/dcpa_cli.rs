//! dcpa-cli: Local zero-GPU CLI, Gate 6 Cross-Stack Verifier, and Zero-Alloc CPU Benchmark.
//!
//! Installs a custom global `CountingAllocator` to mathematically prove `0 allocs/byte`
//! throughout the hot byte perception, hippocampal Householder update, and streaming
//! generation loops (Engineering Blueprint Chapter 6 Law #2 & Gate 6).

use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use dcpa_core_rs::config::DcpaConfig;
use dcpa_core_rs::engine::DcpaEngine;
use dcpa_core_rs::state::FastWeightPrecision;
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

fn load_engine(dir: &Path, precision: FastWeightPrecision) -> DcpaEngine {
    let cfg_str = fs::read_to_string(dir.join("config.json")).expect("failed to read config.json");
    let cfg = DcpaConfig::from_json_str(&cfg_str).expect("invalid config.json");
    let w_bytes = fs::read(dir.join("model.safetensors")).expect("failed to read model.safetensors");
    let weights = DcpaWeights::from_safetensors(cfg, &w_bytes).expect("failed to load model.safetensors");
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
    let mut engine = load_engine(dir, FastWeightPrecision::F32);
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
            rust_c_fused.extend_from_slice(&engine.scratch.hip_c_fused);
            rust_prec_bu.extend_from_slice(&engine.scratch.hip_prec_bu);
            rust_ach.push(ctr.ach);
            rust_z_star.extend_from_slice(&engine.scratch.vec_d_g);
            rust_g_ctx.extend_from_slice(&engine.scratch.vec_d_h);
            rust_win_pair.push(ctr.win_pair_idx as i64);
        }
    }

    // Check exact discrete spike sequence match!
    assert_eq!(
        rust_spikes, golden["sgsc.spikes"].data_u8,
        "GATE 6 FAIL: SGSC discrete spike sequence diverged from Python!"
    );
    assert_eq!(
        rust_win_pair, golden["bcg.win_pair"].data_i64,
        "GATE 6 FAIL: BCG-PC winning merge pair sequence diverged from Python!"
    );

    // Check Module 1 SGSC parity
    check_vec_parity("sgsc.h_seq", &rust_h_seq, &golden["sgsc.h_seq"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.prior_probs", &rust_prior_probs, &golden["sgsc.prior_probs"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.info", &rust_info, &golden["sgsc.info"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.f_stem", &rust_f_stem, &golden["sgsc.f_stem"].data_f32, 1e-5, 0.9999);
    check_vec_parity("sgsc.c_slow", &rust_c_slow, &golden["sgsc.c_slow"].data_f32, 1e-5, 0.9999);

    // Check Module 3 Hippocampus Read & Post-Write parity
    check_vec_parity("hip.c_fused", &rust_c_fused, &golden["hip.c_fused"].data_f32, 1e-5, 0.9999);
    check_vec_parity("hip.prec_bu", &rust_prec_bu, &golden["hip.prec_bu"].data_f32, 1e-5, 0.9999);
    check_vec_parity("hip.ach", &rust_ach, &golden["hip.ach"].data_f32, 1e-5, 0.9999);

    // Query segment 0 after all writes
    dcpa_core_rs::simd::linear_f32(
        &first_f_stem,
        &engine.weights.hip_r_lex,
        None,
        &mut engine.scratch.dg_scores,
    );
    // Re-use exact topk selection
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

    // Check Module 2 BCG-PC parity
    check_vec_parity("bcg.z_star", &rust_z_star, &golden["bcg.z_star"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.g_ctx", &rust_g_ctx, &golden["bcg.g_ctx"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_slots", &engine.graph.slots, &golden["bcg.final_slots"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_edges", &engine.graph.edges, &golden["bcg.final_edges"].data_f32, 1e-5, 0.9999);
    check_vec_parity("bcg.final_salience", &engine.graph.salience, &golden["bcg.final_salience"].data_f32, 1e-5, 0.9999);

    // Check Module 4 PMD Broca CQ + 4 greedy motor byte steps parity
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
    println!(
        "Loaded DCPA config: d_s={}, D={}, K={}, d_h={}, s={}, Hippocampus L3 footprint={:.2} MB",
        cfg.d_s,
        cfg.d_concept,
        cfg.k_slots,
        cfg.d_h,
        cfg.s_active,
        (engine.fast_mem.byte_footprint() as f64) / (1024.0 * 1024.0)
    );

    // 1. Benchmark single-step Column-Major 64B-aligned Hippocampus sparse erase-write (< 50 us!)
    let s = cfg.s_active;
    let e_idx: Vec<usize> = (0..s).map(|i| (i * 97) % cfg.d_h).collect();
    let e_val: Vec<f32> = vec![1.0 / (s as f32).sqrt(); s];
    let phi_idx: Vec<usize> = (0..s).map(|i| (i * 131 + 7) % cfg.d_h).collect();
    let phi_val: Vec<f32> = vec![1.0 / (s as f32).sqrt(); s];
    let s_bar: Vec<f32> = vec![0.123; cfg.d_concept];
    let f_tilde: Vec<f32> = vec![-0.456; cfg.d_s];

    // Warmup CPU L3 cache
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

    // 2. Benchmark streaming byte ingest + Efference Copy streaming generation with STRICT 0 allocs/byte!
    let sample = b"A zibber is a tiny nocturnal emerald frog! Meanwhile the river flowed past the cedar bridge. ";
    let mut out_seg = [0u8; 20];
    let mut rng_state = 0x1234_5678_9ABC_DEF1u64;
    let mut generated_bytes = 0usize;

    // Warmup
    for &b in sample.iter() {
        let tr = engine.sgsc_ingest_byte(b, false);
        if tr.spiked {
            let _ = engine.step_concept_from_scratch(tr.mean_info, tr.sem_norm);
        }
    }

    reset_alloc_counter();
    let t_stream = Instant::now();
    // Ingest 512 bytes
    for i in 0..512 {
        let b = sample[i % sample.len()];
        let tr = engine.sgsc_ingest_byte(b, i == 511);
        if tr.spiked {
            let _ = engine.step_concept_from_scratch(tr.mean_info, tr.sem_norm);
        }
    }
    // Stream generate 256+ bytes via Broca CQ + PMD + Efference Copy rollback loop
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
        "  [BENCH] Hot loop (512B ingest + {}B Efference-Copy stream gen): {:.1} bytes/sec | heap allocs = {} (0 allocs/byte)",
        generated_bytes, gen_bps, stream_allocs
    );
    assert_eq!(
        stream_allocs, 0,
        "GATE 6 FAIL: Hot streaming loop performed {} heap allocations (must be strictly 0)!",
        stream_allocs
    );
    println!("[PASS] Rust Gate 6b: Single-core latency < 50 us & strictly 0 heap allocations in hot loop!");
}

fn cmd_chat(dir: &Path) {
    let mut engine = load_engine(dir, FastWeightPrecision::Bf16);
    println!(
        "DCPA Local Rust Engine Ready (zero GPU, zero Python, {:.2} MB L3 Hippocampus). Type text and press Enter (Ctrl+D to quit):",
        (engine.fast_mem.byte_footprint() as f64) / (1024.0 * 1024.0)
    );
    let stdin = io::stdin();
    let mut rng_state = 0xCAFE_BABE_1337_0001u64;
    let mut out_seg = [0u8; 20];

    for line in stdin.lock().lines() {
        let Ok(text) = line else { break };
        if text.trim().is_empty() {
            continue;
        }
        let bytes = text.as_bytes();
        let mut spikes = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            let tr = engine.sgsc_ingest_byte(b, i + 1 == bytes.len());
            if tr.spiked {
                spikes += 1;
                let _ = engine.step_concept_from_scratch(tr.mean_info, tr.sem_norm);
            }
        }
        print!("[ingested {} bytes -> {} concepts] reply: ", bytes.len(), spikes);
        engine.broca_cq_init();
        let mut reply = Vec::new();
        for _ in 0..engine.weights.cfg.k_slots {
            let (n, done, _err) = engine.generate_concept_with_efference_copy(1, &mut out_seg, &mut rng_state);
            reply.extend_from_slice(&out_seg[..n]);
            if done {
                break;
            }
        }
        println!("{}", String::from_utf8_lossy(&reply));
        let _ = io::stdout().flush();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  dcpa-cli verify-golden <export_dir>");
        eprintln!("  dcpa-cli bench <export_dir>");
        eprintln!("  dcpa-cli chat <export_dir>");
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
