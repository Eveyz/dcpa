"""Frozen cross-workstream contracts and canonical configuration for DCPA-Flat v3.3.

This module is the Single Source of Truth (SSOT) shared across WS-1..WS-6:
  - WS-1: Unified `<QIBBBB` binary `.idx` protocol, 6-domain / 5-style IDs, and StaticGridBatch.
  - WS-2: Static prefix-scatter bridge (`S+1` overflow trash-slot) and CopyRing contracts.
  - WS-3: True-contractive CarrierGraphState/ScanOut, 8-layer Cortical Stack, and Additive Dead-Zone PluginExpertPack.
  - WS-4: Sandwiched chunk-folded Hippocampus (`HIPR <-> G2` causal loop & lower-triangular `M(L<-C)`).
  - WS-5: Top-level `DCPAModelFlat` integration and 5k 3-way killgate.
  - WS-6: `dcpa-core-rs` zero-allocation `BrainSnapshotV3` & unified sleep/domain plugin trainer.

Do NOT alter field names, shapes, or sentinel semantics without updating `docs/DCPA_FLAT_V3_3_SPEC.md`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch

# ---------------------------------------------------------------------------
# 1. Canonical Full-Stack Protocol Constants (WS-1 SSOT)
# ---------------------------------------------------------------------------

IDX_RECORD_FMT: str = "<QIBBBB"
IDX_RECORD_SIZE: int = 16

LANG_DOMAINS_V3: dict[str, int] = {
    "en": 0,
    "zh": 1,
    "formal": 2,
    "python": 3,
    "sql_struct": 4,
    "javascript": 5,
}
LANG_ID_TO_NAME_V3: dict[int, str] = {v: k for k, v in LANG_DOMAINS_V3.items()}

STYLE_DOMAINS_V3: dict[str, int] = {
    "rigorous": 0,
    "concise_memo": 1,
    "expository": 2,
    "literary": 3,
    "strict_eng": 4,
}
STYLE_ID_TO_NAME_V3: dict[int, str] = {v: k for k, v in STYLE_DOMAINS_V3.items()}

# Protected v3 protocol control bytes (all other ASCII < 0x20 except \t=0x09, \n=0x0A are stripped pre-framing)
ALLOWED_PROTOCOL_CTRL_BYTES: frozenset[int] = frozenset(
    {
        0x04,  # EOT: assistant turn end (\x04\n)
        0x11,  # DC1: tool call start (\x11C\n)
        0x12,  # DC2: sandbox observation (\x12O\n)
        0x13,  # DC3: suspend decoding for sandbox (\x13\n)
        0x14,  # DC4: reserved external repo/doc paging frame
        0x1D,  # GS:  scratchpad start/end (\x1dS\n ... \x1dE\n -> G2 Slot-0 pin)
        0x1E,  # RS:  user turn start (\x1eU\n)
        0x1F,  # US:  assistant turn start (\x1fA\n)
    }
)

# Fixed-point x10 encoding for `.loss_mask.bin`:
#   0 -> 0.0x, 1 (legacy) or 10 -> 1.0x, 15 -> 1.5x, 30 -> 3.0x
LOSS_MASK_LEGACY_ONE: int = 1
LOSS_MASK_NORMAL_X10: int = 10
LOSS_MASK_SYNTAX_X10: int = 15
LOSS_MASK_ANCHOR_X10: int = 30

CarrierGradGuardMode = Literal["none", "router_sg", "soft_clip"]


def decode_fixed_point_loss_mask(raw_mask: torch.Tensor) -> torch.Tensor:
    """Convert uint8 `.loss_mask.bin` values (`0, 1, 10, 15, 30`) into float32 weights (`0.0, 1.0, 1.0, 1.5, 3.0`)."""
    m = raw_mask.to(dtype=torch.float32)
    return torch.where(raw_mask == LOSS_MASK_LEGACY_ONE, torch.ones_like(m), m * 0.1)


# ---------------------------------------------------------------------------
# 2. DCPA-Flat v3.3 Configuration
# ---------------------------------------------------------------------------


@dataclass
class DCPAFlatConfig:
    """Architectural configuration for DCPA-Flat v3.3."""

    # Vocabulary: strictly raw UTF-8 bytes 0..255
    vocab_size: int = 256

    # Static 2D batch grid [P, T] -> [P, S] (allocated as S+1 with trash slot S)
    grid_lanes: int = 32
    grid_bytes: int = 4096
    grid_segs: int = 640
    max_seg_len: int = 16
    doc_meta_cap_per_lane: int = 64

    # Core dimensions (multiples of 64 for SIMD & RoPE)
    d_s: int = 192
    d_concept: int = 512
    n_heads: int = 8

    # Dual contractive concept-graph carriers (G1 free-flowing, G2 Slot-0 pinnable)
    k_slots: int = 8
    d_r: int = 32
    d_merge: int = 256
    c_scan: int = 16
    occ_min: float = 0.05
    occ_max: float = 0.95
    occ_relax: float = 0.995
    slot_keep_max: float = 0.990
    full_gate_center: float = 0.20
    full_gate_temp: float = 0.05  # R3: relaxed from 0.04 -> 0.10 (max derivative 2.5)
    edge_decay: float = 0.97      # R3: explicit edge spectral contraction gamma_A in (0, 0.99]
    carrier_grad_guard: CarrierGradGuardMode = "none"
    carrier_grad_clip_rms: float = 1.0

    # 8-layer deep cortical stack (L1 before G1, L2-L3 after G1, L4-L6 after G2_prior, L7-L8 refine)
    n_layers: int = 8
    n_pre_g1_layers: int = 1
    n_post_g1_layers: int = 2
    n_predict_full_layers: int = 3
    n_refine_layers: int = 2
    refine_frac: float = 0.25
    tau_think_ema: float = 0.99

    # R1: Selective W=128 local window attention on L1 (idx 0), L4 (idx 3), L6 (idx 5)
    use_local_window_layers: bool = True
    local_window_layers: tuple[int, ...] = (0, 3, 5)
    window_size: int = 128
    share_stem_ring: bool = False  # False -> +32KB c_kv_sem [128, 128]; True -> 0KB reuse f_stem
    d_sem_ring: int = 128

    # R2: Decoupled Base MoE (1 Shared + E Routed Top-1) + Additive Dead-Zone Plugin Pool
    num_experts: int = 8
    top_k_experts: int = 1
    shared_hidden: int = 256
    routed_hidden: int = 384
    router_bias_lr: float = 1e-3
    router_logit_min: float = -4.0
    plugin_top_k: int = 2
    plugin_tau_min: float = 0.35

    # Module 4: Sandwiched chunk-folded Hippocampus + Dual-Morpheme Whitened DG
    d_h: int = 4096
    s_active: int = 32
    dg_seed_lex: int = 42
    dg_seed_sem: int = 43
    dg_whiten_hidden: int = 192
    dg_homeo_lr: float = 5e-3
    hip_chunk_size: int = 32
    gamma_hip: float = 0.999
    lambda_eta: float = 0.95
    hip_write_threshold: float = 0.05
    hip_sleep_beta_threshold: float = 0.25

    # Module 1 & 5: SGSC & PMD sensorimotor micro-cores
    d_l: int = 64
    num_lang_domains: int = 6
    num_style_domains: int = 5
    stack_depth: int = 16
    stack_tag_dim: int = 4
    lif_lambda: float = 0.85
    lif_beta1: float = 0.25
    lif_beta2: float = 0.75
    lif_theta_init: float = 2.2
    lif_theta_min: float = 0.05
    lif_theta_scale: float = 12.0
    target_spike_rate: float = 1.0 / 9.5
    vocab_temp_init: float = 8.0
    rope_period: float = 33.0
    rope_num_harmonics: int = 16
    rope_theta_concept: float = 10000.0

    # Loss & EMA weights
    jepa_ema_momentum: float = 0.996
    lambda_jepa: float = 0.5
    lambda_unc: float = 0.1
    lambda_dg: float = 0.05
    lambda_rate: float = 2.0
    lambda_halt: float = 0.2
    lambda_var: float = 1.0
    lambda_sgsc_prior: float = 0.2

    # Optimizer defaults
    lr_micro: float = 6e-4
    lr_concept: float = 4e-4
    lr_scalar_gate: float = 1e-3
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.vocab_size != 256:
            raise ValueError(f"vocab_size must be 256 (raw UTF-8 bytes), got {self.vocab_size}")
        if not (2 <= self.k_slots <= 12):
            raise ValueError(f"k_slots must be in [2, 12], got {self.k_slots}")
        if self.d_s % 64 != 0 or self.d_concept % 64 != 0:
            raise ValueError(f"d_s ({self.d_s}) and d_concept ({self.d_concept}) must be multiples of 64")
        if self.d_concept % self.n_heads != 0 or (self.d_concept // self.n_heads) % 8 != 0:
            raise ValueError(
                f"d_concept={self.d_concept} must be divisible by n_heads={self.n_heads} with d_head % 8 == 0"
            )
        if (
            self.n_pre_g1_layers + self.n_post_g1_layers + self.n_predict_full_layers + self.n_refine_layers
            != self.n_layers
        ):
            raise ValueError("Layer counts (pre_g1 + post_g1 + predict_full + refine) must equal n_layers")
        if self.grid_segs % self.window_size != 0:
            raise ValueError(f"grid_segs ({self.grid_segs}) must be a multiple of window_size ({self.window_size})")
        if self.grid_segs % self.c_scan != 0:
            raise ValueError(f"grid_segs ({self.grid_segs}) must be a multiple of c_scan ({self.c_scan})")
        if not (0.0 < self.edge_decay <= 0.995):
            raise ValueError(f"edge_decay must be in (0, 0.995], got {self.edge_decay}")
        if self.carrier_grad_guard not in ("none", "router_sg", "soft_clip"):
            raise ValueError(f"Invalid carrier_grad_guard: {self.carrier_grad_guard}")
        if self.d_l != 64:
            raise ValueError("d_l must be 64 (32D lang + 32D style)")

    @property
    def d_head(self) -> int:
        return self.d_concept // self.n_heads

    @property
    def n_flat(self) -> int:
        return self.grid_lanes * self.grid_segs

    @property
    def refine_k(self) -> int:
        return max(1, int(round(self.n_flat * self.refine_frac)))

    @property
    def num_merge_pairs_full(self) -> int:
        """Ordered pairs (i, j) with 0 <= i != j < K: K*(K-1) = 56 for K=8."""
        return self.k_slots * (self.k_slots - 1)

    @classmethod
    def flat_micro(cls) -> DCPAFlatConfig:
        """Tiny fast config for local CPU/MPS unit tests & compile verification."""
        return cls(
            grid_lanes=4,
            grid_bytes=256,
            grid_segs=64,
            window_size=32,
            c_scan=16,
            hip_chunk_size=16,
            d_s=64,
            d_concept=128,
            n_heads=4,
            k_slots=8,
            d_r=16,
            d_merge=64,
            d_sem_ring=64,
            num_experts=4,
            top_k_experts=1,
            shared_hidden=64,
            routed_hidden=96,
            d_h=512,
            s_active=16,
            dg_whiten_hidden=64,
        )

    @classmethod
    def flat_48m(cls) -> DCPAFlatConfig:
        """DCPA-Flat-48M v3.3 production / 5k killgate preset (~50M trainable base)."""
        return cls(
            grid_lanes=32,
            grid_bytes=4096,
            grid_segs=640,
            window_size=128,
            c_scan=16,
            hip_chunk_size=32,
            d_s=192,
            d_concept=512,
            n_heads=8,
            k_slots=8,
            d_r=32,
            d_merge=256,
            d_sem_ring=128,
            num_experts=8,
            top_k_experts=1,
            shared_hidden=256,
            routed_hidden=384,
            d_h=4096,
            s_active=32,
            dg_whiten_hidden=192,
            lr_micro=6e-4,
            lr_concept=5e-4,
        )

    @classmethod
    def flat_206m(cls) -> DCPAFlatConfig:
        """DCPA-Flat-206M v3.3 flagship preset (~204M trainable base, 12 routed Top-1)."""
        return cls(
            grid_lanes=32,
            grid_bytes=4096,
            grid_segs=640,
            window_size=128,
            c_scan=16,
            hip_chunk_size=32,
            d_s=192,
            d_concept=768,
            n_heads=12,
            k_slots=8,
            d_r=32,
            d_merge=256,
            d_sem_ring=128,
            num_experts=12,
            top_k_experts=1,
            shared_hidden=640,
            routed_hidden=768,
            d_h=4096,
            s_active=32,
            dg_whiten_hidden=192,
            lr_micro=5e-4,
            lr_concept=4e-4,
        )

    @classmethod
    def flat_650m(cls) -> DCPAFlatConfig:
        """DCPA-Flat-650M v3.3 scaled flagship preset (~650M trainable base, 16 routed Top-1)."""
        return cls(
            grid_lanes=32,
            grid_bytes=4096,
            grid_segs=640,
            window_size=128,
            c_scan=16,
            hip_chunk_size=32,
            d_s=256,
            d_concept=1024,
            n_heads=16,
            k_slots=8,
            d_r=32,
            d_merge=384,
            d_sem_ring=192,
            num_experts=16,
            top_k_experts=1,
            shared_hidden=1024,
            routed_hidden=1280,
            d_h=4096,
            s_active=32,
            dg_whiten_hidden=256,
            lr_micro=4e-4,
            lr_concept=3e-4,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["local_window_layers"] = list(self.local_window_layers)
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> DCPAFlatConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "local_window_layers" in data and isinstance(data["local_window_layers"], list):
            data["local_window_layers"] = tuple(int(x) for x in data["local_window_layers"])
        return cls(**data)


# ---------------------------------------------------------------------------
# 3. Six Frozen Cross-Module Contracts (WS-1 .. WS-6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticGridBatch:
    """WS-1 DataLoader -> Model top-level input contract (zero mid-doc truncation)."""

    x_bytes: torch.Tensor         # [P, T] uint8
    doc_id_byte: torch.Tensor     # [P, T] int32 (0..N_docs-1 for valid, -1 for lane-tail pad)
    doc_start_byte: torch.Tensor  # [P, T] bool  (True on first byte of each packed doc)
    doc_end_byte: torch.Tensor    # [P, T] bool  (True on last byte of each packed doc)
    loss_weight: torch.Tensor     # [P, T] float32 (0.0 / 1.0 / 1.5 / 3.0)
    doc_meta: torch.Tensor        # [P * cap, 3] int32 (lang_id, style_id, source_id)

    def to(self, device: torch.device, non_blocking: bool = False) -> StaticGridBatch:
        return StaticGridBatch(
            x_bytes=self.x_bytes.to(device=device, non_blocking=non_blocking),
            doc_id_byte=self.doc_id_byte.to(device=device, non_blocking=non_blocking),
            doc_start_byte=self.doc_start_byte.to(device=device, non_blocking=non_blocking),
            doc_end_byte=self.doc_end_byte.to(device=device, non_blocking=non_blocking),
            loss_weight=self.loss_weight.to(device=device, non_blocking=non_blocking),
            doc_meta=self.doc_meta.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> StaticGridBatch:
        """Enable PyTorch `DataLoader(pin_memory=True)` page-locked host memory for async H2D."""
        return StaticGridBatch(
            x_bytes=self.x_bytes.pin_memory(),
            doc_id_byte=self.doc_id_byte.pin_memory(),
            doc_start_byte=self.doc_start_byte.pin_memory(),
            doc_end_byte=self.doc_end_byte.pin_memory(),
            loss_weight=self.loss_weight.pin_memory(),
            doc_meta=self.doc_meta.pin_memory(),
        )

    def validate(self) -> None:
        if self.x_bytes.ndim != 2:
            raise ValueError(f"x_bytes must be 2D [P, T], got {tuple(self.x_bytes.shape)}")
        P, T = self.x_bytes.shape
        for name, t, dt in (
            ("x_bytes", self.x_bytes, torch.uint8),
            ("doc_id_byte", self.doc_id_byte, torch.int32),
            ("doc_start_byte", self.doc_start_byte, torch.bool),
            ("doc_end_byte", self.doc_end_byte, torch.bool),
            ("loss_weight", self.loss_weight, torch.float32),
        ):
            if tuple(t.shape) != (P, T):
                raise ValueError(f"{name} shape {tuple(t.shape)} != ({P}, {T})")
            if t.dtype != dt:
                raise ValueError(f"{name} dtype {t.dtype} != {dt}")
        if self.doc_meta.ndim != 2 or self.doc_meta.shape[1] != 3 or self.doc_meta.dtype != torch.int32:
            raise ValueError(f"doc_meta must be [M_cap, 3] int32, got {tuple(self.doc_meta.shape)} {self.doc_meta.dtype}")


@dataclass(frozen=True)
class ScatterBridgeOut:
    """WS-2 static_grid + SGSC -> downstream segment-grid contract (trash slot S already sliced off)."""

    f_stem: torch.Tensor          # [P, S, d_s]
    x0: torch.Tensor              # [P, S, D]
    seg_valid: torch.Tensor       # [P, S] bool
    seg_doc_id: torch.Tensor      # [P, S] int32 (valid >= 0; invalid = -(p*S + s + 2))
    seg_reset: torch.Tensor       # [P, S] bool  (R_{p,0}=True, R_{p,s}=(D_{p,s} != D_{p,s-1}))
    seg_pin_trig: torch.Tensor    # [P, S] bool  (\x1dE\n hit in segment -> G2 Slot-0 pin)
    seg_unpin_trig: torch.Tensor  # [P, S] bool  (0x04 / 0x1E hit in segment -> G2 Slot-0 unpin)
    seg_start_depth: torch.Tensor # [P, S] int32 causal AST bracket depth at segment start
    seg_end_depth: torch.Tensor   # [P, S] int32 causal AST bracket depth at segment end
    seg_l_reg: torch.Tensor       # [P, S, 64]   orthogonal 32D lang + 32D style vector
    byte_to_seg: torch.Tensor     # [P, T] int32 in [0, S] (S == overflow trash slot)
    intra_seg_pos: torch.Tensor   # [P, T] int32 in [1, max_seg_len]
    byte_overflow: torch.Tensor   # [P, T] bool  True when s_raw >= S
    seg_counts: torch.Tensor      # [P] int32    number of valid segments per lane (clamped to S)
    # Auxiliary SGSC outputs needed for losses & monitoring
    prior_logits: torch.Tensor    # [P, T, 256] float32
    spikes: torch.Tensor          # [P, T] bool
    s_tilde: torch.Tensor         # [P, T] float32
    mean_info: torch.Tensor       # [P, S] float32
    sem_norm: torch.Tensor        # [P, S] float32

    def validate(self, P: int, T: int, S: int, d_s: int, D: int) -> None:
        checks = (
            ("f_stem", self.f_stem, (P, S, d_s)),
            ("x0", self.x0, (P, S, D)),
            ("seg_valid", self.seg_valid, (P, S)),
            ("seg_doc_id", self.seg_doc_id, (P, S)),
            ("seg_reset", self.seg_reset, (P, S)),
            ("seg_pin_trig", self.seg_pin_trig, (P, S)),
            ("seg_unpin_trig", self.seg_unpin_trig, (P, S)),
            ("seg_start_depth", self.seg_start_depth, (P, S)),
            ("seg_end_depth", self.seg_end_depth, (P, S)),
            ("seg_l_reg", self.seg_l_reg, (P, S, 64)),
            ("byte_to_seg", self.byte_to_seg, (P, T)),
            ("intra_seg_pos", self.intra_seg_pos, (P, T)),
            ("byte_overflow", self.byte_overflow, (P, T)),
            ("seg_counts", self.seg_counts, (P,)),
            ("prior_logits", self.prior_logits, (P, T, 256)),
            ("spikes", self.spikes, (P, T)),
            ("s_tilde", self.s_tilde, (P, T)),
            ("mean_info", self.mean_info, (P, S)),
            ("sem_norm", self.sem_norm, (P, S)),
        )
        for name, tensor, expected in checks:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"ScatterBridgeOut.{name} shape {tuple(tensor.shape)} != {expected}")


@dataclass(frozen=True)
class CarrierGraphState:
    """WS-3 bounded-ball contractive concept graph state (~16 KB per carrier in BF16)."""

    slots: torch.Tensor        # [P, K, D]      closed unit ball ||S_k|| <= 1
    edges: torch.Tensor        # [P, K, K, d_r] bounded (-1, 1)
    occ: torch.Tensor          # [P, K]         float32 bounded occupancy rho in [0.05, 0.95]
    slot0_pinned: torch.Tensor # [P]            bool (only G2 may set True; G1 always False)

    def validate(self, P: int, K: int, D: int, d_r: int) -> None:
        if tuple(self.slots.shape) != (P, K, D):
            raise ValueError(f"slots shape {tuple(self.slots.shape)} != ({P}, {K}, {D})")
        if tuple(self.edges.shape) != (P, K, K, d_r):
            raise ValueError(f"edges shape {tuple(self.edges.shape)} != ({P}, {K}, {K}, {d_r})")
        if tuple(self.occ.shape) != (P, K) or self.occ.dtype != torch.float32:
            raise ValueError(f"occ must be ({P}, {K}) float32, got {tuple(self.occ.shape)} {self.occ.dtype}")
        if tuple(self.slot0_pinned.shape) != (P,) or self.slot0_pinned.dtype != torch.bool:
            raise ValueError(
                f"slot0_pinned must be ({P},) bool, got {tuple(self.slot0_pinned.shape)} {self.slot0_pinned.dtype}"
            )


@dataclass(frozen=True)
class CarrierScanOut:
    """WS-3 Carrier chunked-scan trajectory -> Cortical Stack & Hippocampus."""

    slots_seq: torch.Tensor                # [P, S, K, D]      G1=post, G2=strictly causal prior
    edges_seq: torch.Tensor                # [P, S, K, K, d_r] G1=post, G2=strictly causal prior
    occ_seq: torch.Tensor                  # [P, S, K] float32 G1=post, G2=strictly causal prior
    final_state: CarrierGraphState
    evict_weight: Optional[torch.Tensor] = None  # [P, S, K] float32 Write-Before-Merge intensity (G2)
    evict_target: Optional[torch.Tensor] = None  # [P, S, K, D]      pre-eviction slot concept snapshot (G2)
    empty_write_rate: Optional[torch.Tensor] = None  # scalar float32 diagnostic for Gate C


@dataclass(frozen=True)
class HippocampusChunkState:
    """WS-4 lane-resident hippocampal fast-weight base state with lazy doc-ID masking (zero W_max eviction)."""

    m_c_base: torch.Tensor    # [P, d_h, D]   float32 folded M^{C<-L}
    m_l_base: torch.Tensor    # [P, d_h, d_s] float32 folded M^{L<-C}
    u_lex: torch.Tensor       # [P, d_h]      float32 lexical familiarity trace
    u_sem: torch.Tensor       # [P, d_h]      float32 semantic familiarity trace
    base_doc_id: torch.Tensor # [P]           int32   document ID currently residing in base
    elig_idx: torch.Tensor    # [P, K, s]     int64   per-slot DG sparse indices
    elig_val: torch.Tensor    # [P, K, s]     float32 per-slot DG sparse values
    elig_f: torch.Tensor      # [P, K, d_s]   float32 per-slot stem trace
    elig_eta: torch.Tensor    # [P, K]        float32 per-slot eligibility trace


@dataclass(frozen=True)
class PluginExpertPack:
    """WS-3 / WS-6 unified domain & personal-sleep Additive Dead-Zone plugin pack."""

    pack_id: str
    proto_keys: torch.Tensor  # [n_layers, E_p, D]        unit-L2 normalized prototype keys
    tau: torch.Tensor         # [n_layers, E_p]           dead-zone margin >= plugin_tau_min
    alpha: torch.Tensor       # [n_layers, E_p]           positive tanh gain
    w_gu: torch.Tensor        # [n_layers, E_p, 2*H_p, D] SwiGLU gate+up weights
    w_d: torch.Tensor         # [n_layers, E_p, D, H_p]   SwiGLU down weights (zero-init)

    def validate(self, n_layers: int, D: int, tau_min: float = 0.0) -> None:
        if self.proto_keys.ndim != 3 or self.proto_keys.shape[0] != n_layers or self.proto_keys.shape[2] != D:
            raise ValueError(f"proto_keys shape {tuple(self.proto_keys.shape)} incompatible with ({n_layers}, E_p, {D})")
        E_p = self.proto_keys.shape[1]
        if tuple(self.tau.shape) != (n_layers, E_p) or tuple(self.alpha.shape) != (n_layers, E_p):
            raise ValueError("tau/alpha shapes must match (n_layers, E_p)")
        if self.w_gu.ndim != 4 or self.w_gu.shape[:2] != (n_layers, E_p) or self.w_gu.shape[3] != D:
            raise ValueError(f"Invalid w_gu shape {tuple(self.w_gu.shape)}")
        H_p = self.w_gu.shape[2] // 2
        if tuple(self.w_d.shape) != (n_layers, E_p, D, H_p):
            raise ValueError(f"Invalid w_d shape {tuple(self.w_d.shape)}, expected ({n_layers}, {E_p}, {D}, {H_p})")
        if tau_min > 0.0 and bool((self.tau < tau_min - 1e-6).any()):
            raise ValueError(f"Plugin tau fell below tau_min={tau_min}")


@dataclass(frozen=True)
class BrainSnapshotV3:
    """WS-6 single-stream O(1) runtime snapshot contract (Python <-> dcpa-core-rs Gate 6 parity)."""

    sgsc_h: torch.Tensor            # [1, d_s]
    g1_state: CarrierGraphState     # P=1 free-flowing carrier
    g2_state: CarrierGraphState     # P=1 Slot-0 pinnable carrier
    stem_ring: torch.Tensor         # [1, W, d_s]        pure 48 KB f_stem ring (W=128)
    stem_ring_depth: torch.Tensor   # [1, W] int32       AST scope depth per ring slot
    stem_ring_valid: torch.Tensor   # [1, W] bool
    sem_ring: Optional[torch.Tensor]# [1, W, d_sem_ring] optional +32 KB L3 semantic ring
    prev_rot_stem: torch.Tensor     # [1, d_s]           previous segment stem for dual-morpheme DG
    prev_x3: torch.Tensor           # [1, D]             previous L3 output for soft IoR
    pmd_m: torch.Tensor             # [1, d_s]
    pmd_xi: torch.Tensor            # [1, 16, 4]         128B differentiable pushdown stack
    tau_think: float                # scalar online threshold for L7-L8 refinement


__all__ = [
    "IDX_RECORD_FMT",
    "IDX_RECORD_SIZE",
    "LANG_DOMAINS_V3",
    "LANG_ID_TO_NAME_V3",
    "STYLE_DOMAINS_V3",
    "STYLE_ID_TO_NAME_V3",
    "ALLOWED_PROTOCOL_CTRL_BYTES",
    "LOSS_MASK_LEGACY_ONE",
    "LOSS_MASK_NORMAL_X10",
    "LOSS_MASK_SYNTAX_X10",
    "LOSS_MASK_ANCHOR_X10",
    "CarrierGradGuardMode",
    "decode_fixed_point_loss_mask",
    "DCPAFlatConfig",
    "StaticGridBatch",
    "ScatterBridgeOut",
    "CarrierGraphState",
    "CarrierScanOut",
    "HippocampusChunkState",
    "PluginExpertPack",
    "BrainSnapshotV3",
]
