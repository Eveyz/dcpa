"""Cross-step stateful stream carry for DCPA-Turbo (A1: stream-TBPTT across optimizer steps).

Instead of feeding whole documents (serial BCG-PC depth `m_max ~ max_doc_bytes / rho`) every step, the
trainer feeds `S` parallel byte streams of `chunk_bytes` each and carries the per-stream recurrent state
(detached) into the next optimizer step. Because `bcg_tbptt_window` already truncates the graph-recurrence
gradient, this keeps the gradient horizon unchanged while cutting the per-step serial depth to
`~chunk_bytes / rho` and widening the batch to `S` streams.

Packed varlen layout: every batch "document" (`doc_cu_seqlens` entry) is one stream piece. An entry is either
a *continuation* (`cont_mask=True`, initial state gathered from the carry bank) or *fresh* (learned init).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Sequence

import torch

__all__ = [
    "DocCarryState",
    "RowLayout",
    "StreamCarryBank",
    "advance_ring",
    "bucket_pow2",
    "bucket_multiple",
    "build_row_layout",
    "from_rows",
    "to_rows",
]


def bucket_multiple(n: int, mult: int) -> int:
    """Round `n >= 1` up to a multiple of `mult` (static-shape bucketing for compiled kernels)."""
    n = max(int(n), 1)
    m = max(int(mult), 1)
    return ((n + m - 1) // m) * m


def bucket_pow2(n: int, floor: int = 16) -> int:
    """Round `n` up to the next power of two (at least `floor`)."""
    n = max(int(n), 1)
    p = max(int(floor), 1)
    while p < n:
        p <<= 1
    return p


@dataclass
class DocCarryState:
    """Per-entry (`[V, ...]`) recurrent state crossing a stream-chunk boundary.

    Values for fresh entries (`cont_mask=False`) are ignored by the model (learned inits are used instead).
    `max_entry_bytes` is a static upper bound on any entry length (the packer's `chunk_bytes`).
    """

    cont_mask: torch.Tensor  # [V] bool
    sgsc_h: torch.Tensor  # [V, d_s] fp32
    last_info: torch.Tensor  # [V] fp32 (SGSC surprisal of the previous byte, for exact delta_info)
    pmd_m: torch.Tensor  # [V, d_s] fp32
    pmd_xi: torch.Tensor  # [V, stack_depth, stack_tag_dim] fp32
    last_byte: torch.Tensor  # [V] int64 (previous emitted byte for the PMD teacher-forced input)
    scope_depth: torch.Tensor  # [V] int32 raw (unclamped) bracket depth after the previous byte
    slots: torch.Tensor  # [V, K, D]
    edges: torch.Tensor  # [V, K, K, d_r]
    salience: torch.Tensor  # [V, K] fp32
    slot0_pinned: torch.Tensor  # [V] fp32
    phase: torch.Tensor  # [V, K] fp32
    ring_kv: torch.Tensor  # [V, W, d_s] shared MLA latent of the last W segments
    ring_pos: torch.Tensor  # [V, W] int64 doc positions
    ring_depth: torch.Tensor  # [V, W] int32 segment end depths
    ring_valid: torch.Tensor  # [V, W] bool (right-aligned valid entries)
    doc_pos: torch.Tensor  # [V] int64 number of segments already consumed in this document
    prev_z_ref: torch.Tensor  # [V, D] refined concept of the previous segment (decoder causal shift)
    max_entry_bytes: int = 0

    @property
    def num_entries(self) -> int:
        return int(self.cont_mask.shape[0])

    def tensor_field_names(self) -> list[str]:
        return [f.name for f in fields(self) if f.name not in ("cont_mask", "max_entry_bytes")]

    def detached(self) -> "DocCarryState":
        kw = {name: getattr(self, name).detach().clone() for name in self.tensor_field_names()}
        return replace(self, cont_mask=self.cont_mask.detach().clone(), **kw)


@dataclass
class RowLayout:
    """Mapping between the packed `[T]` byte stream and a right-padded `[n_rows, row_len]` row layout.

    Causal scans are exact under right padding, so per-entry initial states can be injected as per-row
    `h0` without touching the scan kernels.
    """

    pad_pos: torch.Tensor  # [T] int64 flat byte -> row * row_len + col
    starts: torch.Tensor  # [V] int64
    ends: torch.Tensor  # [V] int64 (exclusive)
    entry_of_byte: torch.Tensor  # [T] int64
    n_entries: int
    n_rows: int
    row_len: int


def build_row_layout(
    doc_cu_seqlens: torch.Tensor,
    total_bytes: int,
    row_len: int,
    row_bucket: int = 16,
) -> RowLayout:
    """Build a `RowLayout` for packed entries `doc_cu_seqlens [V + 1]` (every entry length `<= row_len`)."""
    device = doc_cu_seqlens.device
    cu = doc_cu_seqlens.to(dtype=torch.int64)
    starts = cu[:-1]
    ends = cu[1:]
    n_entries = int(starts.numel())
    lens = ends - starts
    entry_of_byte = torch.repeat_interleave(
        torch.arange(n_entries, device=device, dtype=torch.int64), lens, output_size=int(total_bytes)
    )
    col = torch.arange(int(total_bytes), device=device, dtype=torch.int64) - starts[entry_of_byte]
    rl = max(int(row_len), 1)
    pad_pos = entry_of_byte * rl + col
    return RowLayout(
        pad_pos=pad_pos,
        starts=starts,
        ends=ends,
        entry_of_byte=entry_of_byte,
        n_entries=n_entries,
        n_rows=bucket_pow2(max(n_entries, 1), row_bucket),
        row_len=rl,
    )


def to_rows(x_flat: torch.Tensor, layout: RowLayout, fill: float | int | bool = 0) -> torch.Tensor:
    """Scatter packed `[T, ...]` into right-padded `[n_rows, row_len, ...]` (differentiable w.r.t. `x_flat`)."""
    rest = tuple(x_flat.shape[1:])
    base = x_flat.new_full((layout.n_rows * layout.row_len, *rest), fill)
    out = base.index_copy(0, layout.pad_pos, x_flat)
    return out.view(layout.n_rows, layout.row_len, *rest)


def from_rows(x_rows: torch.Tensor, layout: RowLayout) -> torch.Tensor:
    """Gather right-padded `[n_rows, row_len, ...]` back into packed `[T, ...]`."""
    rest = tuple(x_rows.shape[2:])
    return x_rows.reshape(layout.n_rows * layout.row_len, *rest).index_select(0, layout.pad_pos)


def pad_entries(x: torch.Tensor, n_rows: int, fill: float | int | bool = 0) -> torch.Tensor:
    """Pad per-entry `[V, ...]` to `[n_rows, ...]`."""
    v = int(x.shape[0])
    if v >= n_rows:
        return x[:n_rows]
    pad = x.new_full((n_rows - v, *x.shape[1:]), fill)
    return torch.cat([x, pad], dim=0)


def advance_ring(
    ring_kv: torch.Tensor,
    ring_pos: torch.Tensor,
    ring_depth: torch.Tensor,
    ring_valid: torch.Tensor,
    c_kv: torch.Tensor,
    pos: torch.Tensor,
    depth: torch.Tensor,
    seg_pad_pos: torch.Tensor,
    q_pad_len: int,
    seg_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append each entry's chunk segments to its right-aligned FIFO ring and keep the last `W`.

    Ring valid entries occupy `[W - r, W)` and chunk entries `[0, n)` of the padded chunk block, so the
    concatenation's valid items are contiguous and the new ring is simply `combined[n + i]` for `i < W`.
    """
    V, W = ring_pos.shape
    d_s = ring_kv.shape[-1]
    Qp = int(q_pad_len)
    n_slots = V * Qp + 1
    dev = ring_kv.device

    ck_pad = ring_kv.new_zeros((n_slots, d_s)).index_copy(0, seg_pad_pos, c_kv.to(ring_kv.dtype))
    ck_pad = ck_pad[: V * Qp].view(V, Qp, d_s)
    pos_pad = torch.zeros((n_slots,), device=dev, dtype=ring_pos.dtype).index_copy(0, seg_pad_pos, pos.to(ring_pos.dtype))
    pos_pad = pos_pad[: V * Qp].view(V, Qp)
    dep_pad = torch.zeros((n_slots,), device=dev, dtype=ring_depth.dtype).index_copy(
        0, seg_pad_pos, depth.to(ring_depth.dtype)
    )
    dep_pad = dep_pad[: V * Qp].view(V, Qp)
    local = torch.arange(Qp, device=dev, dtype=torch.int64).unsqueeze(0)
    val_pad = local < seg_counts.to(torch.int64).unsqueeze(1)

    comb_kv = torch.cat([ring_kv, ck_pad], dim=1)  # [V, W + Qp, d_s]
    comb_pos = torch.cat([ring_pos, pos_pad], dim=1)
    comb_dep = torch.cat([ring_depth, dep_pad], dim=1)
    comb_val = torch.cat([ring_valid, val_pad], dim=1)

    gidx = seg_counts.to(torch.int64).clamp(0, Qp).unsqueeze(1) + torch.arange(W, device=dev, dtype=torch.int64).unsqueeze(0)
    new_kv = comb_kv.gather(1, gidx.unsqueeze(-1).expand(V, W, d_s))
    new_pos = comb_pos.gather(1, gidx)
    new_dep = comb_dep.gather(1, gidx)
    new_val = comb_val.gather(1, gidx)
    return new_kv, new_pos, new_dep, new_val


class StreamCarryBank:
    """Per-stream (`[S, ...]`) detached carry storage living on the training device.

    `gather(carry_from)` builds the per-entry `DocCarryState` for a packed batch (`carry_from[v] = s` gathers
    stream `s`, `-1` = fresh entry). `store(carry_out, carry_save)` writes entry `v` back into stream
    `carry_save[v]` (`-1` = discard, e.g. finished documents).
    """

    def __init__(
        self,
        n_streams: int,
        *,
        d_s: int,
        d_concept: int,
        k_slots: int,
        d_r: int,
        window: int,
        stack_depth: int,
        stack_tag_dim: int,
        device: torch.device,
    ) -> None:
        self.n_streams = max(1, int(n_streams))
        self.d_s = int(d_s)
        self.d_concept = int(d_concept)
        self.k_slots = int(k_slots)
        self.d_r = int(d_r)
        self.window = int(window)
        self.stack_depth = int(stack_depth)
        self.stack_tag_dim = int(stack_tag_dim)
        self.device = device
        self._store: Optional[DocCarryState] = None
        self._valid = torch.zeros((self.n_streams,), dtype=torch.bool)

    @classmethod
    def from_config(cls, cfg, n_streams: int, device: torch.device) -> "StreamCarryBank":
        return cls(
            n_streams,
            d_s=cfg.d_s,
            d_concept=cfg.d_concept,
            k_slots=cfg.k_slots,
            d_r=cfg.d_r,
            window=cfg.window_size,
            stack_depth=cfg.stack_depth,
            stack_tag_dim=cfg.stack_tag_dim,
            device=device,
        )

    def reset(self) -> None:
        self._store = None
        self._valid.zero_()

    def _zeros(self, n: int, device: torch.device) -> DocCarryState:
        K, D, d_r, d_s, W = self.k_slots, self.d_concept, self.d_r, self.d_s, self.window
        f32 = torch.float32
        return DocCarryState(
            cont_mask=torch.zeros((n,), device=device, dtype=torch.bool),
            sgsc_h=torch.zeros((n, d_s), device=device, dtype=f32),
            last_info=torch.zeros((n,), device=device, dtype=f32),
            pmd_m=torch.zeros((n, d_s), device=device, dtype=f32),
            pmd_xi=torch.zeros((n, self.stack_depth, self.stack_tag_dim), device=device, dtype=f32),
            last_byte=torch.zeros((n,), device=device, dtype=torch.int64),
            scope_depth=torch.zeros((n,), device=device, dtype=torch.int32),
            slots=torch.zeros((n, K, D), device=device, dtype=f32),
            edges=torch.zeros((n, K, K, d_r), device=device, dtype=f32),
            salience=torch.zeros((n, K), device=device, dtype=f32),
            slot0_pinned=torch.zeros((n,), device=device, dtype=f32),
            phase=torch.zeros((n, K), device=device, dtype=f32),
            ring_kv=torch.zeros((n, W, d_s), device=device, dtype=f32),
            ring_pos=torch.zeros((n, W), device=device, dtype=torch.int64),
            ring_depth=torch.zeros((n, W), device=device, dtype=torch.int32),
            ring_valid=torch.zeros((n, W), device=device, dtype=torch.bool),
            doc_pos=torch.zeros((n,), device=device, dtype=torch.int64),
            prev_z_ref=torch.zeros((n, D), device=device, dtype=f32),
        )

    def gather(self, carry_from: Sequence[int] | torch.Tensor, max_entry_bytes: int) -> DocCarryState:
        cf = torch.as_tensor(carry_from, dtype=torch.int64).reshape(-1).cpu()
        n = int(cf.numel())
        if self._store is None:
            out = self._zeros(n, self.device)
            out.max_entry_bytes = int(max_entry_bytes)
            return out
        cont_cpu = (cf >= 0) & (cf < self.n_streams)
        if bool(cont_cpu.any()):
            cont_cpu = cont_cpu & self._valid[cf.clamp(0, self.n_streams - 1)]
        idx_dev = cf.clamp(0, self.n_streams - 1).to(self.device, non_blocking=True)
        cont_dev = cont_cpu.to(self.device, non_blocking=True)
        kw = {}
        for name in self._store.tensor_field_names():
            src = getattr(self._store, name)
            g = src.index_select(0, idx_dev)
            mask = cont_dev.view(n, *([1] * (g.ndim - 1)))
            kw[name] = torch.where(mask, g, torch.zeros_like(g))
        return DocCarryState(cont_mask=cont_dev, max_entry_bytes=int(max_entry_bytes), **kw)

    @torch.no_grad()
    def store(self, carry_out: Optional[DocCarryState], carry_save: Sequence[int] | torch.Tensor) -> None:
        cs = torch.as_tensor(carry_save, dtype=torch.int64).reshape(-1).cpu()
        keep = (cs >= 0) & (cs < self.n_streams)
        # Streams not continued this step lose their state (their document finished).
        new_valid = torch.zeros_like(self._valid)
        if carry_out is None or not bool(keep.any()):
            self._valid = new_valid
            return
        if self._store is None:
            self._store = self._zeros(self.n_streams, self.device)
        ent_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        dst_idx = cs[ent_idx]
        ent_dev = ent_idx.to(self.device, non_blocking=True)
        dst_dev = dst_idx.to(self.device, non_blocking=True)
        for name in self._store.tensor_field_names():
            dst = getattr(self._store, name)
            src = getattr(carry_out, name).detach().index_select(0, ent_dev).to(dtype=dst.dtype)
            dst.index_copy_(0, dst_dev, src)
        new_valid[dst_idx] = True
        self._valid = new_valid

    @property
    def num_valid(self) -> int:
        return int(self._valid.sum().item())
