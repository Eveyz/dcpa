# DCPA — Developmental Concept-Predictive Architecture

Dual-stack implementation of the **Developmental Concept-Predictive Architecture (DCPA)** following the *Engineering Implementation Blueprint & Anti-Drift Specification* and *Complete Architecture & Mathematical Derivation*.

## Ten Commandments Compliance Matrix

1. **Zero Tokenizer**: Raw UTF-8 bytes (`uint8`, `V = 256`) only — no BPE, WordPiece, tiktoken, or SentencePiece.
2. **Zero KV Cache / Fixed Working Memory**: Constant working memory graph hard-capped at `K = 8` slots (`K <= 12`).
3. **Parallel Associative Scan**: Diagonal linear recurrences (SGSC, PMD motor core) and the 128B differentiable pushdown stack (`Xi_t`) use `O(log T)` Blelloch / associative parallel scans + OpenAI Triton CUDA kernels (zero Python byte loops).
4. **Zero-Padding Ragged Aggregation**: Variable-length segments aggregate directly on 1D flat tensors via `cu_seqlens` (`int32`).
5. **Strict Causality**: SGSC prior `p_t` is computed exclusively from shifted previous states `[h_0, ..., h_{T-1}] @ W_out^T` (`max |p_{1:t}' - p_{1:t}| == 0.0`).
6. **STE / Gumbel-STE Surrogates**: Hard discrete forward decisions (`s_t`, 72-pair merge `alpha_ij`, Broca CQ winner `pi_sel`) with smooth surrogate backward gradients.
7. **Non-Parametric Hippocampal Fast Weights**: `M^{C<-L}`, `M^{L<-C}`, `u^{lex}`, `u^{sem}` are never optimizer parameters; updated purely forward via 3-factor Gated Delta Householder erase-write (using compact `W_max = 32` rank-1 trace buffers in batched GPU training).
8. **Asymmetric Eligibility Inheritance**: `K+1 -> K` merges inherit eligibility traces verbatim from `argmax eta` (strictly preserving `s`-sparse unit-norm orthogonality).
9. **Pragmatic Motor Decoder (PMD)**: Broca Competitive Queuing + Inhibition of Return, Holographic RoPE inverse phase demodulation, 128B differentiable pushdown stack, weight-tied byte motor core, Dual Motor Charge Depletion Gate, and Efference Copy local rollback.
10. **Four-Stage Developmental Curriculum & 6 Acceptance Gates**: Staged warmup (`Stage A -> B -> C -> D`), real-time `AntiCollapseMonitor`, and automated Gates 1–6.

---

## Repository Structure

```
dcpa/
├── Cargo.toml                  # Rust workspace (dcpa-core-rs, dcpa-data-forge, dcpa-sleep-daemon)
├── dcpa-train-py/              # Python 3.11 + PyTorch + OpenAI Triton training plane
│   ├── dcpa/
│   │   ├── config.py           # Micro / Tiny (48M) / Base (152M) presets & invariant checks
│   │   ├── ops/                # Blelloch/Triton scans, Holographic RoPE, Ragged cu_seqlens, RMSNorm
│   │   ├── modules/            # SGSC, BCG-PC, HippocampusDualMemory, PMD
│   │   ├── data/               # Mmap .bin/.idx shard loader & online Category 1/4 generators
│   │   ├── train/              # 4-stage CurriculumSchedule, AntiCollapseMonitor, trainer CLI
│   │   ├── model.py            # Unified DCPAModel + weight tying + joint developmental losses
│   │   └── export.py           # Flat-named model.safetensors + Golden Test Vector exporter
│   └── tests/                  # Pytest unit test suite
├── dcpa-core-rs/               # 100% Pure Static Rust (`no_std` core + ARM NEON / AVX SIMD)
│   └── src/
│       ├── bf16.rs             # IEEE-754 bf16 & 64-byte cache-line aligned storage
│       ├── simd.rs             # ARM64 NEON intrinsics & scalar fallback
│       ├── weights.rs          # Pure Rust flat-named safetensors loader (zero Python/LibTorch)
│       ├── state.rs            # L1 Concept Graph (17.5KB), L3 Column-Major Hippocampus (7.52MB), ScratchPool
│       ├── engine.rs           # Zero-alloc hot path + 512B stack Efference Copy rollback
│       └── bin/dcpa_cli.rs     # Global CountingAllocator verifier (`verify-golden`, `bench`, `chat`)
├── dcpa-data-forge/            # Multi-threaded Rust CLI: NFC norm, AST balance filter, .bin/.idx sharding
├── dcpa-sleep-daemon/          # Rust SWR Sleep Replay consolidation daemon & incremental safetensors slice exporter
└── gates/                      # Automated quantitative Gate 1..6 verification scripts
```

---

## Quickstart

### 1. Run Python Unit Tests & All 6 Acceptance Gates
```bash
dcpa-train-py/.venv/bin/pytest -v dcpa-train-py/tests/test_dcpa.py
dcpa-train-py/.venv/bin/python gates/run_all_gates.py
```

### 2. Build & Test Pure Rust Workspace (`no_std` + Release CLI)
```bash
cargo test --workspace
cargo check -p dcpa-core-rs --no-default-features
cargo build --release --workspace
```

### 3. Generate Developmental Binary Shards (`dcpa-data-forge`)
```bash
cargo run --release -p dcpa-data-forge -- synth-stage1 500 500 data/stage1
cargo run --release -p dcpa-data-forge -- synth-fastmap 500 data/stage4
cargo run --release -p dcpa-data-forge -- inspect data/stage1
```

### 4. Train Developmental Curriculum (`dcpa-train-py`)
```bash
dcpa-train-py/.venv/bin/python -m dcpa.train.trainer \
  --preset micro --steps 200 --batch-size 4 --seq-len 128 \
  --shard-prefix data/stage1 --out-dir checkpoints/micro_demo
```

### 5. Verify Cross-Stack Parity, Run CPU Bench, Sleep Consolidation, or Local Chat
```bash
./target/release/dcpa-cli verify-golden checkpoints/micro_demo
./target/release/dcpa-cli bench checkpoints/micro_demo
./target/release/dcpa-sleep-daemon checkpoints/micro_demo checkpoints/micro_demo/cortical_sleep_slice.safetensors
./target/release/dcpa-cli chat checkpoints/micro_demo
```
