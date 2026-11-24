//! dcpa-core-rs: 100% Pure Static Rust Inference & Continual Learning Engine for DCPA.
//!
//! - `no_std` compatible core (compile with `--no-default-features` for embedded / bare-metal).
//! - Hardware SIMD acceleration (ARM64 NEON / x86_64 AVX).
//! - Column-Major 64-byte cache-line aligned Hippocampal fast weights (7.52 MB in CPU L3).
//! - Strictly 0 heap allocations in the hot perception & streaming generation loop (`0 allocs/byte`).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bf16;
pub mod config;
pub mod engine;
pub mod simd;
pub mod state;
pub mod weights;

pub use config::DcpaConfig;
pub use engine::{ByteIngestTrace, ConceptStepTrace, DcpaEngine};
pub use state::{BrainSnapshot, FastWeightPrecision, HippocampusFastWeights, MotorCheckpoint};
pub use weights::DcpaWeights;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::bf16::{AlignedVec, Bf16};
    use crate::config::DcpaConfig;
    use crate::simd::{dot_bf16_f32, dot_f32, linear_bf16_f32, linear_f32, unit_rmsnorm_f32};
    use crate::state::{FastWeightPrecision, HippocampusFastWeights};

    #[test]
    fn test_simd_linear_4row_and_bf16_parity() {
        // Test unaligned odd out_features (7) and non-multiple-of-16 in_features (53)
        let in_dim = 53usize;
        let out_dim = 7usize;
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.13 - 1.7).sin()).collect();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.07 + 0.3).cos() * 0.5)
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.1 - 0.2).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        linear_f32(&x, &w, Some(&bias), &mut out_simd);

        for r in 0..out_dim {
            let mut expected = bias[r];
            for c in 0..in_dim {
                expected += w[r * in_dim + c] * x[c];
            }
            assert!(
                (out_simd[r] - expected).abs() < 1e-5,
                "row {} mismatch: got {} vs expected {}",
                r,
                out_simd[r],
                expected
            );
            let d = dot_f32(&w[r * in_dim..(r + 1) * in_dim], &x) + bias[r];
            assert!((out_simd[r] - d).abs() < 1e-5);
        }

        // Verify BF16 packed linear matches quantized FP32 reference within BF16 precision
        let mut w_bf16 = AlignedVec::<Bf16>::zeroed(w.len());
        let mut w_q_f32 = vec![0.0f32; w.len()];
        for (i, &val) in w.iter().enumerate() {
            let b = Bf16::from_f32(val);
            w_bf16.as_mut_slice()[i] = b;
            w_q_f32[i] = b.to_f32();
        }
        let mut out_bf16 = vec![0.0f32; out_dim];
        let mut out_ref = vec![0.0f32; out_dim];
        linear_bf16_f32(&x, w_bf16.as_slice(), Some(&bias), &mut out_bf16);
        linear_f32(&x, &w_q_f32, Some(&bias), &mut out_ref);
        for r in 0..out_dim {
            assert!(
                (out_bf16[r] - out_ref[r]).abs() < 1e-5,
                "bf16 row {} mismatch: got {} vs ref {}",
                r,
                out_bf16[r],
                out_ref[r]
            );
            let d_bf16 = dot_bf16_f32(&w_bf16.as_slice()[r * in_dim..(r + 1) * in_dim], &x) + bias[r];
            assert!((out_bf16[r] - d_bf16).abs() < 1e-5);
        }
    }

    #[test]
    fn test_unit_rmsnorm_and_hippocampus_snapshot_restore() {
        let x = [3.0f32, -4.0, 0.0, 5.0];
        let mut out = [0.0f32; 4];
        unit_rmsnorm_f32(&x, 1e-6, &mut out);
        let rms = ((out.iter().map(|v| v * v).sum::<f32>()) / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);

        let cfg = DcpaConfig {
            vocab_size: 256,
            d_s: 64,
            d_concept: 64,
            k_slots: 8,
            d_r: 16,
            d_merge: 32,
            d_h: 128,
            s_active: 8,
            dg_seed_lex: 17,
            dg_seed_sem: 29,
            gamma_hip: 0.995,
            lambda_eta: 0.85,
            kappa_quench: 0.9,
            c0_ema_momentum: 0.99,
            householder_max_writes: 128,
            householder_write_threshold: 0.05,
            num_experts: 4,
            top_k_experts: 2,
            expert_hidden: 64,
            l_max: 4,
            tau_skip: 0.32,
            tau_halt: 0.22,
            ponder_eta: 0.5,
            lambda_moe_bal: 0.01,
            lif_lambda: 0.85,
            lif_beta1: 1.0,
            lif_beta2: 0.5,
            lif_theta_init: 1.5,
            lif_theta_min: 0.5,
            max_seg_len: 16,
            target_spike_rate: 0.105,
            lambda_rate: 1.0,
            delta_decay: 0.05,
            d_l: 64,
            num_lang_domains: 5,
            num_style_domains: 5,
            stack_depth: 16,
            stack_tag_dim: 4,
            tau_e: 0.5,
            tau_q: 0.5,
            lambda_inh: 2.0,
            tau_done: 0.1,
            tau_repair: 0.45,
            max_repair_retries: 2,
            rope_period: 20.0,
            rope_num_harmonics: 8,
            lambda_var: 1.0,
            lambda_act: 0.01,
            lambda_halt: 0.01,
            eps: 1e-6,
        };
        assert!(cfg.validate().is_ok());
        let mut mem = HippocampusFastWeights::new(&cfg, FastWeightPrecision::Bf16);
        let idx = [1usize, 5, 9, 13, 17, 21, 25, 29];
        let val = [0.35355338f32; 8];
        let s_bar = vec![0.25f32; 64];
        let f_t = vec![-0.5f32; 64];
        let mut pc = vec![0.0f32; 64];
        let mut pl = vec![0.0f32; 64];
        mem.advance_step();
        mem.write_slot_inplace(0.8, &idx, &val, &s_bar, &idx, &val, &f_t, &mut pc, &mut pl);

        let snap = mem.clone_snapshot();
        let mut out1 = vec![0.0f32; 64];
        let g1 = mem.read_c_from_l(&idx, &val, &mut out1);
        assert!(g1 > 0.05);

        mem.wipe_zero();
        let mut out_zero = vec![0.0f32; 64];
        let gz = mem.read_c_from_l(&idx, &val, &mut out_zero);
        assert_eq!(gz, 0.0);

        mem.restore_from(&snap);
        let mut out2 = vec![0.0f32; 64];
        let g2 = mem.read_c_from_l(&idx, &val, &mut out2);
        assert_eq!(g1, g2);
        assert_eq!(out1, out2);
    }
}
