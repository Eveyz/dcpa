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
pub use state::{FastWeightPrecision, HippocampusFastWeights, MotorCheckpoint};
pub use weights::DcpaWeights;
