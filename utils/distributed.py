"""Zero-wrapper-overhead Multi-GPU Distributed Training primitives for DCPA (`nccl` & `gloo`).

Why DCPA uses coalesced flat-bucket gradient all-reduce instead of `nn.DistributedDataParallel` wrapper:
  1. DCPA shares 98.5% of its parameters inside the unrolled `BCG-PC` recurrent loop (`m_step = M..0`),
     so recurrent parameter gradients only finalize at `m_step = 0` at the end of `backward()`.
  2. Avoiding the `DDP` module wrapper preserves 100% direct attribute & state-dict compatibility across
     selective `torch.compile`, dynamic 4-stage parameter freezing (`requires_grad_(False)` + `grad=None`),
     Stage-3 conditional `Broca CQ` (`broca_cq_prob=0.20`), Stage-5 surgical freeze, and Watchdog rollbacks.
  3. Synchronizes `has_grad` masks across ranks before bucketed BF16/FP32 `all_reduce`:
       - Parameters untouched on ALL ranks keep `p.grad = None` (strictly preserving Red Line #2 so
         AdamW decoupled weight decay never corrodes frozen or inactive weights).
       - Parameters touched on ANY rank materialize zeros on silent ranks so cross-rank `1/W` averaging
         is 100% mathematically exact.
       - 1D / scalar gates (`theta_raw`, Hippocampus gates) always reduce in exact `float32`, while
         large 2D+ matrices reduce in `bfloat16` over NVLink when `use_bf16_comm=True` (2x less traffic).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class DistContext:
    """Immutable distributed process-group descriptor."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def get_active_dist_context() -> DistContext:
    """Inspect current `torch.distributed` state without initializing a new process group."""
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world = int(dist.get_world_size())
        dev_cnt = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        local_rk = int(os.environ.get("LOCAL_RANK", rank % dev_cnt)) if torch.cuda.is_available() else 0
        return DistContext(
            rank=rank,
            local_rank=local_rk,
            world_size=world,
            backend=str(dist.get_backend()),
        )
    return DistContext()


def init_distributed_context(
    preferred_device: str = "auto",
    timeout_minutes: int = 30,
) -> tuple[DistContext, bool]:
    """Attach to an existing process group or initialize one from `torchrun` env vars.

    Returns `(ctx, initialized_here)` so callers only destroy groups they created.
    """
    if dist.is_available() and dist.is_initialized():
        ctx = get_active_dist_context()
        if ctx.backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(ctx.local_rank)
        return ctx, False

    env_world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if env_world <= 1 or not dist.is_available():
        return DistContext(), False

    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)) or str(rank))
    pref = preferred_device.strip().lower()
    want_cuda = pref in ("auto", "cuda") or pref.startswith("cuda:")
    use_nccl = bool(want_cuda and torch.cuda.is_available() and dist.is_nccl_available())
    backend = "nccl" if use_nccl else "gloo"
    timeout = timedelta(minutes=max(1, int(timeout_minutes)))

    if use_nccl:
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend=backend,
                timeout=timeout,
                device_id=torch.device("cuda", local_rank),
            )
        except TypeError:
            dist.init_process_group(backend=backend, timeout=timeout)
    else:
        dist.init_process_group(backend=backend, timeout=timeout)

    return DistContext(rank=rank, local_rank=local_rank, world_size=env_world, backend=backend), True


def cleanup_distributed_context(initialized_here: bool) -> None:
    """Cleanly tear down the process group if `init_distributed_context` created it."""
    if initialized_here and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def dist_barrier(ctx: DistContext) -> None:
    if not ctx.is_distributed or not dist.is_initialized():
        return
    if ctx.backend == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[ctx.local_rank])
    else:
        dist.barrier()


def _comm_device(ctx: DistContext, fallback_device: torch.device) -> torch.device:
    if ctx.backend == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", ctx.local_rank)
    if ctx.backend == "gloo":
        return torch.device("cpu")
    return fallback_device


@torch.no_grad()
def broadcast_model_state(model: nn.Module, ctx: DistContext, src: int = 0) -> None:
    """Broadcast all parameters and buffers from `src` rank so all GPUs start bit-identical."""
    if not ctx.is_distributed or not dist.is_initialized():
        return
    seen: set[int] = set()
    tensors: list[torch.Tensor] = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            tensors.append(p.data)
    for b in model.buffers():
        if id(b) not in seen:
            seen.add(id(b))
            tensors.append(b.data)

    for t in tensors:
        if ctx.backend == "gloo" and t.device.type != "cpu":
            cpu_t = t.detach().to("cpu").contiguous()
            dist.broadcast(cpu_t, src=src)
            t.copy_(cpu_t.to(t.device))
        else:
            if not t.is_contiguous():
                contig = t.contiguous()
                dist.broadcast(contig, src=src)
                t.copy_(contig)
            else:
                dist.broadcast(t, src=src)


@torch.no_grad()
def sync_bool_any(flag: bool, ctx: DistContext, device: torch.device) -> bool:
    """Return True on ALL ranks if `flag` is True on ANY rank (prevents asymmetric backward hangs)."""
    if not ctx.is_distributed or not dist.is_initialized():
        return bool(flag)
    cdev = _comm_device(ctx, device)
    val = torch.tensor([1 if flag else 0], dtype=torch.int32, device=cdev)
    dist.all_reduce(val, op=dist.ReduceOp.MAX)
    return bool(int(val.item()) > 0)


@torch.no_grad()
def sync_tensor_avg_(tensor: torch.Tensor, ctx: DistContext) -> None:
    """Average `tensor` in-place across all ranks (used for periodic `bcg_pc.c0` EMA sync)."""
    if not ctx.is_distributed or not dist.is_initialized():
        return
    orig_dtype = tensor.dtype
    if ctx.backend == "gloo" and tensor.device.type != "cpu":
        work = tensor.detach().float().to("cpu").contiguous()
        dist.all_reduce(work, op=dist.ReduceOp.SUM)
        work.mul_(1.0 / float(ctx.world_size))
        tensor.copy_(work.to(device=tensor.device, dtype=orig_dtype))
    else:
        work = tensor.detach().float().contiguous()
        dist.all_reduce(work, op=dist.ReduceOp.SUM)
        work.mul_(1.0 / float(ctx.world_size))
        tensor.copy_(work.to(dtype=orig_dtype))


@torch.no_grad()
def all_reduce_metrics_dict(
    metrics: Mapping[str, float],
    ctx: DistContext,
    device: torch.device,
) -> dict[str, float]:
    """Single-vector cross-rank average of scalar metrics for bit-identical Watchdog consensus."""
    if not ctx.is_distributed or not dist.is_initialized() or not metrics:
        return {str(k): float(v) for k, v in metrics.items()}

    keys = sorted(str(k) for k in metrics.keys())
    cdev = _comm_device(ctx, device)
    # Replace local NaN/Inf with NaN safely; IEEE NaN propagates across SUM so all ranks see non-finite!
    vals = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=cdev)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals.mul_(1.0 / float(ctx.world_size))
    out_list = vals.to("cpu").tolist()
    return {k: float(v) for k, v in zip(keys, out_list)}


@torch.no_grad()
def all_reduce_gradients_coalesced(
    params: Sequence[nn.Parameter],
    ctx: DistContext,
    *,
    use_bf16_comm: bool = True,
    bucket_cap_mb: float = 64.0,
) -> None:
    """Coalesced flat-bucket cross-rank gradient averaging with exact `grad=None` preservation.

    - Deduplicates tied parameters (`SGSC <-> PMD` shared `embed` and `w_out`).
    - Synchronizes `has_grad` across ranks so silent conditional branches on one rank average as `0.0`
      while parameters unused on ALL ranks remain `grad=None` (zero AdamW weight-decay drift).
    - Reduces scalar/1D parameters (`< 4096` elements) in exact `float32`, and large 2D+ matrices in
      `bfloat16` over NCCL when `use_bf16_comm=True` (cutting NVLink/PCIe volume by 50%).
    """
    if not ctx.is_distributed or not dist.is_initialized() or not params:
        return

    uniq_params: list[nn.Parameter] = []
    seen_ids: set[int] = set()
    for p in params:
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            uniq_params.append(p)
    if not uniq_params:
        return

    first_dev = uniq_params[0].device
    cdev = _comm_device(ctx, first_dev)

    # 1. Consensus on which parameters received a gradient on at least one rank
    has_g = torch.tensor(
        [1 if p.grad is not None else 0 for p in uniq_params],
        dtype=torch.int32,
        device=cdev,
    )
    dist.all_reduce(has_g, op=dist.ReduceOp.MAX)
    has_g_list = has_g.to("cpu").tolist()

    fp32_group: list[nn.Parameter] = []
    bf16_group: list[nn.Parameter] = []
    can_bf16 = bool(use_bf16_comm and ctx.backend == "nccl")

    for p, hg in zip(uniq_params, has_g_list):
        if int(hg) == 0:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
        if can_bf16 and p.ndim >= 2 and p.numel() >= 4096:
            bf16_group.append(p)
        else:
            fp32_group.append(p)

    inv_w = 1.0 / float(ctx.world_size)

    def _reduce_bucket_list(group: list[nn.Parameter], comm_dtype: torch.dtype) -> None:
        if not group:
            return
        elem_bytes = 2 if comm_dtype == torch.bfloat16 else 4
        max_elems = max(1, int((bucket_cap_mb * 1024 * 1024) // elem_bytes))

        cur_batch: list[nn.Parameter] = []
        cur_elems = 0

        def _flush(batch_p: list[nn.Parameter]) -> None:
            if not batch_p:
                return
            total_n = sum(p.numel() for p in batch_p)
            flat = torch.empty((total_n,), dtype=comm_dtype, device=cdev)
            offset = 0
            for p in batch_p:
                assert p.grad is not None
                n = p.numel()
                g_flat = p.grad.detach().reshape(-1)
                if g_flat.device != cdev or g_flat.dtype != comm_dtype:
                    flat[offset : offset + n].copy_(g_flat.to(device=cdev, dtype=comm_dtype))
                else:
                    flat[offset : offset + n].copy_(g_flat)
                offset += n

            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            if comm_dtype == torch.bfloat16:
                # Multiply scale in fp32 before scattering back to avoid bf16 subnormal rounding bias
                flat_scaled = (flat.float() * inv_w)
                offset = 0
                for p in batch_p:
                    assert p.grad is not None
                    n = p.numel()
                    p.grad.copy_(flat_scaled[offset : offset + n].view_as(p.grad).to(dtype=p.grad.dtype, device=p.grad.device))
                    offset += n
            else:
                flat.mul_(inv_w)
                offset = 0
                for p in batch_p:
                    assert p.grad is not None
                    n = p.numel()
                    p.grad.copy_(flat[offset : offset + n].view_as(p.grad).to(dtype=p.grad.dtype, device=p.grad.device))
                    offset += n

        for p in group:
            p_n = p.numel()
            if cur_batch and (cur_elems + p_n > max_elems):
                _flush(cur_batch)
                cur_batch = []
                cur_elems = 0
            cur_batch.append(p)
            cur_elems += p_n
        _flush(cur_batch)

    _reduce_bucket_list(fp32_group, torch.float32)
    _reduce_bucket_list(bf16_group, torch.bfloat16)
