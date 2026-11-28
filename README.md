# DCPA-Flat v3.3 — Developmental Concept-Predictive Architecture

Dual-stack (**Python/PyTorch** training plane + **100% Pure `no_std` Rust** zero-allocation inference plane) implementation of **DCPA-Flat v3.3**, a byte-native, KV-cache-free cognitive architecture with static grid execution, contractive dual-graph carriers, selective local window attention, zero-drift additive MoE plugins, and chunk-folded hippocampal fast memory.

- **Architecture Specification**: [docs/DCPA_FLAT_V3_3_SPEC.md](docs/DCPA_FLAT_V3_3_SPEC.md)
- **Colab All-in-One Notebook**: [notebooks/Colab_DCPA_Flat_v3.ipynb](notebooks/Colab_DCPA_Flat_v3.ipynb)

---

## Key Architectural Upgrades in DCPA-Flat v3.3

| Subsystem | DCPA-Flat v3.3 Design | Key Guarantee |
| :--- | :--- | :--- |
| **Input & Packing** | Byte-native (`uint8`, `V = 256`, zero tokenizer) + 6-domain lossless syntax/paragraph pre-chunking + fixed-point `x10` `.loss_mask.bin` + FFD Static Grid `[P, T]` (`S+1` trash slot) | `torch.compile(..., fullgraph=True, dynamic=False)`, 0 host syncs, `< 6%` pad waste |
| **Sensory & Motor** | `SGSCFlat` + `PMDFlat` (weight-tied byte embeddings, strict causal shift, dual-threshold hysteresis) + `HolographicCopyRing` (`W=128`, `2W+1` zero-sink column) | Exact verbatim OOV copying with bounded constant memory |
| **Working Memory** | Dual `ContractiveGraphCarrier` (`G1` Fast + `G2` Slow with Slot-0 anchor pinning, `K=8` slots, closed unit ball, `edge_decay=0.97`, `c_scan=16` chunk compile) | Provable spectral stability across 512+ steps, constant `O(K)` memory |
| **8-Layer Cortex** | `L1..L8` Flat Stack with selective `W=128` local windows on `L1 / L4 / L6`, unified 265-col single-softmax on `L4 / L6`, adaptive `u_n` refinement (`L7..L8`) | Zero global KV cache; `BrainSnapshotV3` = `102.4 KB` (48M) / `114.4 KB` (206M) |
| **Pluggable MoE** | Base **Top-1** routed + shared SwiGLU decoupled from **Additive Dead-Zone Plugins** (`ReLU(cos(h, p) - tau)`) | Exact `0.000e0` logit drift on non-target inputs; hot-plug/unplug `.dcpa_expert.safetensors` |
| **Hippocampus** | `DualMorphemeWhitenedDG` (`D_h=4096, s=32`) + chunk-folded `HippocampusFlat` (`HIPR <-> G2`, chunk unit-lower-triangular `M(L<-C)` solve) | Batched GPU parity `< 1e-5` vs sequential streaming |
| **Sleep Daemon** | `dcpa-sleep-daemon` captures awake high-surprise `SleepReplayCaptureRing` -> trains orthogonal sleep plugin via frozen-base `dcpa.train_plugin --mode sleep` -> verifies base immutability & `0.0` anchor drift -> atomic swap | Zero catastrophic forgetting; base model stays strictly read-only |
| **Rust Inference** | `dcpa-core-rs` (`#![no_std]` core + `dcpa-cli verify-golden-flat` / `bench-flat`) | `cos_sim = 1.0000000`, `max_abs_err <= 1.43e-6`, `0` heap allocs/byte |

---

## Model Presets & YAML Configs (`configs/v3/`)

| Preset | Config File | Trainable / Active Params | `d_concept` / Experts | Recommended Corpus Profile |
| :--- | :--- | :--- | :--- | :--- |
| **48M Killgate** | [configs/v3/dcpa_flat_48m_killgate.yaml](configs/v3/dcpa_flat_48m_killgate.yaml) | `51.8M` / `16.1M` | `512` / `8E Top-1` | `v3_flat_5g` (`~5.5 GB`) |
| **48M Edge** | [configs/v3/dcpa_flat_48m_v3.yaml](configs/v3/dcpa_flat_48m_v3.yaml) | `51.8M` / `16.1M` | `512` / `8E Top-1` | `v3_flat_18g` (`~18 GB` -> `32 GB` stream) |
| **206M Flagship** | [configs/v3/dcpa_flat_206m_v3.yaml](configs/v3/dcpa_flat_206m_v3.yaml) | `203.9M` / `40.6M` | `768` / `12E Top-1` | `v3_flat_45g` (Takeoff) / `v3_flat_90g` (Full) |
| **650M Scaled** | [configs/v3/dcpa_flat_650m_v3.yaml](configs/v3/dcpa_flat_650m_v3.yaml) | `~650M` / `~115M` | `1024` / `16E Top-1` | `v3_flat_160g` (`~160 GB` -> `300 GB` stream) |

---

## Repository Structure

```text
dcpa/
├── Cargo.toml                           # Rust workspace (dcpa-core-rs, dcpa-data-forge, dcpa-sleep-daemon)
├── docs/DCPA_FLAT_V3_3_SPEC.md          # Frozen DCPA-Flat v3.3 engineering & mathematical spec
├── configs/v3/                          # Flat v3.3 YAML configs (48m_killgate, 48m, 206m, 650m)
├── notebooks/
│   ├── Colab_DCPA_Flat_v3.ipynb         # Primary v3.3 Colab notebook (CPU forge -> 5k killgate -> 48M/206M -> Gate 6)
│   └── Colab_DCPA_Train.ipynb           # Legacy v1/v2 Colab notebook
├── data/
│   ├── cognitive_v2_sources.py          # 12 curated 2025-2026 6-domain HF sources + lossless pre-chunkers
│   ├── build_corpus.py                  # Streaming shard builder (--v3-flat --write-loss-mask)
│   ├── prepare_vast_corpus.py           # Parallel CPU orchestrator (v3_flat_5g..160g) + .tar.zst pack/restore
│   └── ragged_dataloader.py             # StaticGridByteDataset (FFD [P, T] packer) + legacy ragged loader
├── train/
│   ├── train_flat.py                    # Production 5-stage Flat v3.3 trainer + A/B Drive sync + W&B + Rust export
│   └── colab_drive_sync.py              # Atomic A/B Ping-Pong Google Drive sync & health guard
├── gates/
│   ├── run_flat_killgate_5k.py          # 3-way 5,000-step Killgate (Run A legacy vs Run B K=8 vs Run C +W=128)
│   └── gate6_flat_v3_alignment.py       # Python <-> Rust no_std parity, 0-alloc, plugin & sleep daemon gate
├── dcpa-train-py/dcpa/
│   ├── interfaces.py                    # DCPAFlatConfig & 6 frozen tensor/snapshot contracts
│   ├── ops/static_grid.py               # S+1 trash-slot segmented ops & static LIF scan (fullgraph compile)
│   ├── modules/                         # SGSCFlat, PMDFlat, HolographicCopyRing, ContractiveGraphCarrier,
│   │                                    # DecoupledPluggableMoE, CorticalStackFlat, HippocampusFlat
│   ├── model_flat.py                    # DCPAModelFlat + streaming eval parity
│   ├── train_plugin.py                  # Unified frozen-base Additive Dead-Zone plugin trainer (sleep / domain)
│   └── export.py                        # Flat v3.3 safetensors + golden vector exporter
├── dcpa-core-rs/src/flat/               # Pure no_std Rust Flat v3.3 engine, weights, state & dcpa-cli commands
├── dcpa-data-forge/                     # Rust <QIBBBB 16B shard forge, UTF-8/protocol sanitizer & syntax depth tagger
├── dcpa-sleep-daemon/                   # v3.3 awake capture ring + Python plugin bridge + atomic promotion daemon
└── tests/                               # Flat v3.3 (test_flat_*) & legacy v1/v2 unit/integration suites
```

---

## Quickstart

### 1. Verify Python & Rust Dual-Stack Locally
```bash
# Python Flat v3.3 unit & integration tests
dcpa-train-py/.venv/bin/pytest -q tests/test_flat_v3_interfaces.py tests/test_flat_ws1_data.py \
  tests/test_flat_ws2_grid_and_ring.py tests/test_flat_ws3_carrier_and_stack.py \
  tests/test_flat_ws4_hippocampus.py tests/test_flat_ws5_integration.py tests/test_flat_ws6_rust_and_sleep.py

# Rust workspace tests + strict no_std verification
cargo test --workspace
cargo check -p dcpa-core-rs --no-default-features

# End-to-end Gate 6 cross-stack alignment (Python <-> Rust + Plugin + 0-alloc + Sleep Daemon)
dcpa-train-py/.venv/bin/python gates/gate6_flat_v3_alignment.py
```

### 2. Phase 0 — Multi-Core CPU Corpus Forge (Zero GPU Credits)
Run on a cheap/no-GPU multi-core CPU box (or Colab CPU High-RAM via `Colab_DCPA_Flat_v3.ipynb` with `RUN_MODE="cpu_forge"`):
```bash
# Inspect plan offline first
python data/prepare_vast_corpus.py --profile v3_flat_90g --nvme-root /content/nvme_cache --dry-run

# Build & pack .tar.zst archive (choose v3_flat_5g / v3_flat_18g / v3_flat_45g / v3_flat_90g / v3_flat_160g)
python data/prepare_vast_corpus.py \
  --profile v3_flat_90g \
  --nvme-root /content/nvme_cache \
  --max-parallel-jobs 6 \
  --pack-archive /content/nvme_cache/dcpa_flat_v3_flat_90g.tar.zst
```

### 3. Phase 1 — 48M 5,000-Step Three-Way Killgate (GPU ~25 min)
```bash
# Quick local CPU smoke test
dcpa-train-py/.venv/bin/python gates/run_flat_killgate_5k.py --smoke

# Full GPU 5k killgate against forged NVMe shards
python gates/run_flat_killgate_5k.py \
  --bin /content/nvme_cache/data/train.bin --idx /content/nvme_cache/data/train.idx \
  --val-bin /content/nvme_cache/data/val.bin --val-idx /content/nvme_cache/data/val.idx \
  --steps 5000 --warmup-steps 800 --plugin-steps 200 \
  --out-json /content/nvme_cache/killgate_5k_report.json
```

### 4. Phase 2 — Production 5-Stage Training (`train/train_flat.py`)
```bash
# 4-step end-to-end smoke check
dcpa-train-py/.venv/bin/python train/train_flat.py --smoke --local-cache-dir /tmp/dcpa_flat_smoke

# Full 206M flagship training (supports A/B Drive Ping-Pong --resume auto & W&B)
python train/train_flat.py \
  --config configs/v3/dcpa_flat_206m_v3.yaml \
  --data-dir /content/nvme_cache/data \
  --cap-data-dir /content/nvme_cache/cap_data \
  --sft-data-dir /content/nvme_cache/sft_data \
  --local-cache-dir /content/nvme_cache/run_206m \
  --drive-sync-dir /content/drive/MyDrive/dcpa_flat/checkpoints/flat_206m \
  --drive-export-dir /content/drive/MyDrive/dcpa_flat/export/flat_206m \
  --resume auto --wandb
```

### 5. Phase 3 — Rust Verification, Benchmarking & Sleep / Domain Plugins
```bash
cargo build --release --bin dcpa-cli --bin dcpa-sleep-daemon

# Cross-stack golden parity & zero-alloc byte-throughput benchmark
./target/release/dcpa-cli verify-golden-flat export/flat_206m
./target/release/dcpa-cli bench-flat export/flat_206m

# Train a zero-drift Additive Dead-Zone domain plugin on frozen base weights
python -m dcpa.train_plugin \
  --mode domain --base-dir export/flat_206m \
  --replay-bin cap_data/cap_code_train.bin --replay-idx cap_data/cap_code_train.idx \
  --out-plugin export/flat_206m/code_domain.dcpa_expert.safetensors

# Run v3.3 sleep consolidation daemon on captured awake replay ring
./target/release/dcpa-sleep-daemon export/flat_206m sleep_state/sleep_replay_ring.bin \
  export/flat_206m/personal_sleep.dcpa_expert.safetensors
```

---

## Legacy v1 / v2 Compatibility
All legacy `DCPAModel` (v1/v2), `dcpa-core-rs` legacy engine (`verify-golden` / `bench` / `chat`), `configs/v1/`, `configs/v2/`, and `notebooks/Colab_DCPA_Train.ipynb` remain intact and verified under `tests/test_sanity.py` and `gates/run_all_gates.py`.
