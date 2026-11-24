"""Stage 5 Pragmatic Instruction Fine-Tuning (SFT) Trainer for DCPA (4,000 steps / ~15 min on H100/A100).

Implements Stage 5 Addendum Section 3:
  - Loads Stage 4 slow-weight export (`dcpa_final.safetensors` + `config.json`, or `.pt` ckpt).
  - Enforces surgical parameter freezing (`requires_grad_(False)` + `grad = None`):
      * FROZEN: SGSC recurrence & segmentation (`w_u, w_alpha, theta_raw, w_q, w_k, w_f, h0, norm_f, w_res, norm_c`)
                and all Hippocampus novelty/crystallization gates (`alpha_1, alpha_2, theta_i, theta_n, alpha_q, theta_q, gamma_e, f_prec_cort`).
      * UNFROZEN:
          - BCG-PC concept core: `lr = 1.5e-4, weight_decay = 0.01`
          - PMD + slow stem projections (`w_sem, w_lex_slow, w_lang_lex`) + shared byte head (`embed, w_out`)
            + expression gate bias (`theta_out, alpha_out`): `lr = 3.0e-4, weight_decay = 0.01`
  - Schedule: `total_steps = 4000`, 200-step linear warmup, cosine decay to `3.0e-5`;
    `target_tokens_bytes_per_step = 65536`, `max_doc_bytes = 2048`.
  - Transient zero-initialized per-dialogue Hippocampus fast weights (`stage="D"`).
  - Exact SFT loss: `L_sft = L_lang(sft_loss_mask, 2.0x on 0x04) + 0.5 * L_PC + 0.5 * L_halt + 0.02 * L_ponder`.
  - Atomically exports `dcpa_chat_sft.safetensors` (+ `config.json`) & optional Drive mirror.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from safetensors.torch import load_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.build_sft_stage5 import SFTByteDataset
from dcpa.compile_utils import configure_model_compile, strip_orig_mod_prefix, unwrap_compiled_module
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from train.colab_drive_sync import assert_not_drive_path, sync_file_to_drive
from train.curriculum_state_machine import _canonical_param_name
from train.train_dcpa import export_final_safetensors
from utils.chat_protocol import build_sft_byte_loss_weights
from utils.device import (
    amp_autocast,
    configure_cuda_tf32,
    describe_hardware_profile,
    detect_hardware_profile,
    get_peak_memory_gb,
    pin_batch_if_cuda,
    reset_peak_memory_stats,
    seed_everything,
    synchronize,
    to_device_async,
)
from utils.distributed import (
    all_reduce_gradients_coalesced,
    all_reduce_metrics_dict,
    broadcast_model_state,
    cleanup_distributed_context,
    dist_barrier,
    init_distributed_context,
    maybe_relaunch_distributed,
    sync_bool_any,
    sync_tensor_avg_,
)


STAGE5_FROZEN_NAMES: frozenset[str] = frozenset(
    {
        # SGSC recurrence & LIF segmentation
        "sgsc.h0",
        "sgsc.w_alpha.weight",
        "sgsc.w_alpha.bias",
        "sgsc.w_u.weight",
        "sgsc.theta_raw",
        "sgsc.w_q.weight",
        "sgsc.w_k.weight",
        "sgsc.w_f.weight",
        "sgsc.norm_f.weight",
        "sgsc.w_res.weight",
        "sgsc.norm_c.weight",
        # Hippocampus novelty & crystallization gates
        "hippocampus.alpha_1",
        "hippocampus.alpha_2",
        "hippocampus.theta_i",
        "hippocampus.theta_n",
        "hippocampus.alpha_q",
        "hippocampus.gamma_e",
        "hippocampus.theta_q",
        "hippocampus.f_prec_cort.weight",
        "hippocampus.f_prec_cort.bias",
    }
)


@dataclass
class Stage5SFTSummary:
    completed_steps: int
    total_steps: int
    final_loss: float
    final_loss_lang: float
    val_bpc: Optional[float]
    export_path: Path
    active_params: int
    frozen_params: int
    eot_grad_norm_max: float


def load_dcpa_model_from_weights(
    weights_path: str | Path,
    config_path: Optional[str | Path] = None,
    device: torch.device | str = "cpu",
) -> DCPAModel:
    """Load `DCPAModel` cleanly from `.safetensors` (+ `config.json`) or `.pt` checkpoint."""
    wp = Path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"Weights file not found: {wp}")

    if wp.suffix == ".safetensors":
        cp = Path(config_path) if config_path is not None else (wp.parent / "config.json")
        if not cp.exists():
            raise FileNotFoundError(f"Matching config.json not found next to {wp}: {cp}")
        cfg = DCPAConfig.load_json(cp)
        model = DCPAModel(cfg)
        raw_tensors = strip_orig_mod_prefix(load_file(str(wp)))
        if "sgsc.theta_raw" in raw_tensors:
            # Undo export's lif_theta_scale folding so model.sgsc.theta matches verbatim
            raw_tensors["sgsc.theta_raw"] = raw_tensors["sgsc.theta_raw"].float() / float(cfg.lif_theta_scale)
        missing, unexpected = model.load_state_dict(raw_tensors, strict=False)
        allowed_missing = {
            "pmd.embed.weight",
            "pmd.w_out.weight",
            "bcg_pc.perm_bank",
            "bcg_pc.pair_i",
            "bcg_pc.pair_j",
            "bcg_pc.pair_u",
            "bcg_pc.pair_v",
            "bcg_pc.pair_u_onehot",
            "bcg_pc.pin_mask_72",
            "bcg_pc.slot0_onehot",
        }
        bad_missing = set(missing) - allowed_missing
        if bad_missing or unexpected:
            raise RuntimeError(f"Weight load mismatch from {wp}: missing={bad_missing}, unexpected={unexpected}")
        # Verify strict SGSC<->PMD weight tying
        assert model.pmd.embed.weight is model.sgsc.embed.weight
        assert model.pmd.w_out.weight is model.sgsc.w_out.weight
        return model.to(device)

    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
    cfg = DCPAConfig(**ckpt["config"])
    model = DCPAModel(cfg)
    sd = strip_orig_mod_prefix(ckpt["model_state_dict"])
    model.load_state_dict(sd, strict=True)
    return model.to(device)


def apply_stage5_surgical_freeze(model: DCPAModel) -> dict[str, Any]:
    """Apply Stage 5 surgical freeze (`requires_grad_(False)` + `grad = None`) and return param groups."""
    bcg_decay: list[torch.nn.Parameter] = []
    bcg_nodecay: list[torch.nn.Parameter] = []
    pmd_decay: list[torch.nn.Parameter] = []
    pmd_nodecay: list[torch.nn.Parameter] = []

    active_names: set[str] = set()
    frozen_names: set[str] = set()
    active_elems = 0
    frozen_elems = 0
    seen_ids: set[int] = set()

    for raw_name, param in model.named_parameters():
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        pname = _canonical_param_name(raw_name)

        if pname in STAGE5_FROZEN_NAMES:
            param.requires_grad_(False)
            param.grad = None
            frozen_names.add(pname)
            frozen_elems += param.numel()
            continue

        param.requires_grad_(True)
        active_names.add(pname)
        active_elems += param.numel()

        no_decay = param.ndim < 2 or pname.endswith(".bias") or "embed" in pname or "norm" in pname
        if pname.startswith("bcg_pc."):
            (bcg_nodecay if no_decay else bcg_decay).append(param)
        else:
            # PMD all params + w_sem, w_lex_slow, w_lang_lex + shared embed/w_out + theta_out/alpha_out
            (pmd_nodecay if no_decay else pmd_decay).append(param)

    param_groups = [
        {"name": "bcg_pc_decay", "params": bcg_decay, "base_lr": 1.5e-4, "lr": 1.5e-4, "weight_decay": 0.01},
        {"name": "bcg_pc_nodecay", "params": bcg_nodecay, "base_lr": 1.5e-4, "lr": 1.5e-4, "weight_decay": 0.0},
        {"name": "pmd_head_decay", "params": pmd_decay, "base_lr": 3.0e-4, "lr": 3.0e-4, "weight_decay": 0.01},
        {"name": "pmd_head_nodecay", "params": pmd_nodecay, "base_lr": 3.0e-4, "lr": 3.0e-4, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    return {
        "param_groups": param_groups,
        "active_names": active_names,
        "frozen_names": frozen_names,
        "active_params": active_elems,
        "frozen_params": frozen_elems,
    }


def _stage5_lr_for_group(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 3.0e-5) -> float:
    if step < warmup_steps:
        frac = float(step + 1) / float(max(warmup_steps, 1))
        return base_lr * max(0.01, frac)
    rem = max(total_steps - warmup_steps, 1)
    prog = min(max(float(step - warmup_steps) / float(rem), 0.0), 1.0)
    cos_f = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_lr + (base_lr - min_lr) * cos_f


@torch.no_grad()
def _eval_sft_val_bpc(
    model: DCPAModel,
    val_ds: Optional[SFTByteDataset],
    device: torch.device,
    use_bf16: bool,
    n_batches: int = 4,
) -> Optional[float]:
    if val_ds is None:
        return None
    was_training = model.training
    model.eval()
    tot_nll = 0.0
    tot_sup = 0.0
    try:
        for b_i in range(n_batches):
            batch = val_ds[b_i]
            x_b = batch["x_bytes"].to(device)
            msk = batch["sft_loss_mask"].to(device)
            cu = batch["cu_seqlens"].to(device)
            lids = batch["doc_lang_ids"].to(device)
            sids = batch.get("doc_style_ids")
            sids_t = sids.to(device) if sids is not None else None
            pts = batch.get("doc_ponder_tax_scale")
            pts_t = pts.to(device) if pts is not None else None
            w = build_sft_byte_loss_weights(x_b, msk, eot_weight=1.0, tool_ctrl_weight=1.0).unsqueeze(0)
            n_sup = float(msk.float().sum().item())
            if n_sup <= 0:
                continue
            sft_lam_act = min(0.02, 0.5 * float(model.cfg.lambda_act))
            with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(
                    raw_bytes=x_b.unsqueeze(0),
                    lang_ids=lids,
                    style_ids=sids_t,
                    ponder_tax_scale=pts_t,
                    stage="D",
                    tau_s=0.25,
                    tau_m=0.25,
                    use_broca_cq=False,
                    loss_weights=w,
                    lambda_rate_override=0.0,
                    lambda_var_override=0.0,
                    lambda_pc_override=0.5,
                    lambda_halt_override=0.5,
                    lambda_act_override=sft_lam_act,
                    doc_cu_seqlens=cu,
                )
            tot_nll += float(out.loss_lang.detach().item()) * n_sup
            tot_sup += n_sup
    finally:
        if was_training:
            model.train()
    if tot_sup <= 0:
        return None
    return (tot_nll / tot_sup) / math.log(2.0)


def run_stage5_sft(
    *,
    weights_path: str | Path,
    data_dir: str | Path = "artifacts/sft_data",
    cap_data_dir: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    export_dir: str | Path = "artifacts/export",
    drive_export_dir: Optional[str | Path] = None,
    total_steps: int = 8000,
    warmup_steps: Optional[int] = None,
    target_tokens_bytes_per_step: int = 65536,
    micro_batch_bytes: Optional[int] = None,
    fixed_global_batch: bool = False,
    max_doc_bytes: int = 2048,
    eot_weight: float = 2.0,
    tool_ctrl_weight: float = 2.5,
    device_str: str = "auto",
    compile_mode: str = "auto",
    deterministic: bool = False,
    seed: int = 202605,
    log_every: int = 50,
    step_hook: Optional[Callable[[int, DCPAModel, dict[str, float]], None]] = None,
) -> Stage5SFTSummary:
    """Run Stage 5 surgical pragmatic + capability SFT (default 8,000 steps) and export `dcpa_chat_sft.safetensors`."""
    dist_ctx, dist_owned = init_distributed_context(preferred_device=device_str)
    is_primary = dist_ctx.is_primary

    def _r0_print(*args: Any, **kwargs: Any) -> None:
        if is_primary:
            print(*args, **kwargs)

    assert_not_drive_path(data_dir, role="--data-dir")
    if cap_data_dir is not None:
        assert_not_drive_path(cap_data_dir, role="--cap-data-dir")
    profile = detect_hardware_profile(
        preferred=device_str,
        deterministic=deterministic,
        target_tokens_bytes_per_step=target_tokens_bytes_per_step,
    )
    device = profile.device
    configure_cuda_tf32(device, allow=not deterministic)
    reset_peak_memory_stats(device)
    use_bf16 = bool(profile.use_bf16 and not deterministic)

    if dist_ctx.is_distributed:
        if (
            fixed_global_batch
            or int(target_tokens_bytes_per_step) >= 65536 * dist_ctx.world_size
            or int(target_tokens_bytes_per_step) < 65536
        ):
            per_rank_target = max(128, math.ceil(int(target_tokens_bytes_per_step) / dist_ctx.world_size))
        else:
            # Default 64 KB/GPU per-rank scaling across N GPUs without starving GPUs!
            per_rank_target = int(target_tokens_bytes_per_step)
    else:
        per_rank_target = max(128, int(target_tokens_bytes_per_step))

    eff_micro_bytes = (
        int(micro_batch_bytes)
        if micro_batch_bytes is not None
        else min(int(profile.micro_batch_bytes), per_rank_target)
    )
    eff_micro_bytes = max(128, eff_micro_bytes)
    grad_accum = max(1, math.ceil(per_rank_target / eff_micro_bytes))
    global_step_bytes = eff_micro_bytes * grad_accum * dist_ctx.world_size
    sft_lr_scale = (
        min(2.0, math.sqrt(float(global_step_bytes) / 65536.0))
        if (dist_ctx.is_distributed and global_step_bytes > 65536)
        else 1.0
    )
    eff_warmup = (
        int(warmup_steps)
        if warmup_steps is not None
        else (300 if total_steps == 8000 else (200 if total_steps == 4000 else max(1, int(round(total_steps * 0.05)))))
    )

    _r0_print(describe_hardware_profile(profile), flush=True)
    seed_everything(seed, device=device)

    model = load_dcpa_model_from_weights(weights_path, config_path=config_path, device=device)
    broadcast_model_state(model, dist_ctx)
    if dist_ctx.is_distributed:
        seed_everything(seed + dist_ctx.rank * 104729, device=device)
    if device.type == "cuda" and profile.vram_gb >= 75.0:
        default_wm = 48.0 if model.cfg.num_experts >= 16 else 52.0
        if profile.vram_gb < 88.0:
            default_wm = 40.0 if model.cfg.num_experts >= 32 else 44.0
        model.bcg_checkpoint_vram_watermark_gb = default_wm
    freeze_info = apply_stage5_surgical_freeze(model)
    if sft_lr_scale != 1.0:
        for pg in freeze_info["param_groups"]:
            pg["base_lr"] = float(pg["base_lr"]) * sft_lr_scale
            pg["lr"] = float(pg["lr"]) * sft_lr_scale
    optimizer = torch.optim.AdamW(freeze_info["param_groups"], betas=(0.9, 0.95), eps=1.0e-8)

    comp_rep = configure_model_compile(model, compile_mode=compile_mode, device=device)
    _r0_print(
        f"[Stage 5 SFT] steps={total_steps:,} (warmup={eff_warmup}) | micro={eff_micro_bytes:,}B x {grad_accum} x {dist_ctx.world_size}GPU "
        f"(global={global_step_bytes:,}B/step) | active={freeze_info['active_params']:,} | "
        f"frozen={freeze_info['frozen_params']:,} | compile={comp_rep.effective_mode}",
        flush=True,
    )

    train_datasets: list[SFTByteDataset] = [
        SFTByteDataset(
            data_dir=data_dir,
            split="train",
            target_batch_bytes=eff_micro_bytes,
            max_doc_bytes=max_doc_bytes,
            num_batches=max((total_steps * grad_accum + 100) * dist_ctx.world_size, 1000),
            seed=seed,
            prefix="sft",
            bucket_sync_world_size=dist_ctx.world_size,
        )
    ]
    if cap_data_dir is not None:
        cdir = Path(cap_data_dir)
        for idx_c, pref in enumerate(("cap_reasoning", "cap_instruct", "cap_toolcall", "cap_writing"), start=1):
            if (cdir / f"{pref}_train.bin").exists() and (cdir / f"{pref}_train.loss_mask.bin").exists():
                train_datasets.append(
                    SFTByteDataset(
                        data_dir=cdir,
                        split="train",
                        target_batch_bytes=eff_micro_bytes,
                        max_doc_bytes=max(max_doc_bytes, 4096 if pref in ("cap_writing", "cap_instruct") else max_doc_bytes),
                        num_batches=max((total_steps * grad_accum + 100) * dist_ctx.world_size, 1000),
                        seed=seed + idx_c * 313,
                        prefix=pref,
                        bucket_sync_world_size=dist_ctx.world_size,
                    )
                )
    if len(train_datasets) > 1:
        _r0_print(
            f"[Stage 5 Mix] Balanced multi-capability mixture active ({len(train_datasets)} shards: sft + cap_*)",
            flush=True,
        )

    val_ds: Optional[SFTByteDataset] = None
    if is_primary and (Path(data_dir) / "sft_val.bin").exists():
        val_ds = SFTByteDataset(
            data_dir=data_dir,
            split="val",
            target_batch_bytes=min(eff_micro_bytes, 4096),
            max_doc_bytes=max_doc_bytes,
            num_batches=64,
            seed=seed + 77,
            prefix="sft",
        )

    batch_cursor = 0
    last_metrics: dict[str, float] = {}
    eot_grad_norm_max = 0.0
    t_start = time.perf_counter()

    try:
        model.train()
        for step_idx in range(total_steps):
            completed = step_idx + 1
            for g in optimizer.param_groups:
                g["lr"] = _stage5_lr_for_group(
                    step_idx, total_steps, eff_warmup, base_lr=float(g["base_lr"]), min_lr=3.0e-5
                )

            optimizer.zero_grad(set_to_none=True)
            accum_m: dict[str, float] = {}
            step_finite = True
            for _ in range(grad_accum):
                # Synchronize shard & length-tier selection across all ranks (`batch_cursor`) while sharding dialogue index!
                ds_sel = train_datasets[batch_cursor % len(train_datasets)]
                logical_ds_idx = batch_cursor // len(train_datasets)
                phys_ds_idx = logical_ds_idx * dist_ctx.world_size + dist_ctx.rank
                batch = pin_batch_if_cuda(ds_sel.get_batch(phys_ds_idx, sync_group=batch_cursor), device)
                batch_cursor += 1
                x_b = to_device_async(batch["x_bytes"], device)
                msk = to_device_async(batch["sft_loss_mask"], device)
                cu = to_device_async(batch["cu_seqlens"], device)
                lids = to_device_async(batch["doc_lang_ids"], device)
                sids_t = to_device_async(batch.get("doc_style_ids"), device)
                pts_t = to_device_async(batch.get("doc_ponder_tax_scale"), device)
                assert x_b is not None and msk is not None and cu is not None and lids is not None

                loss_w = build_sft_byte_loss_weights(
                    x_b, msk, eot_weight=eot_weight, tool_ctrl_weight=tool_ctrl_weight
                ).unsqueeze(0)
                sft_lam_act = min(0.02, 0.5 * float(model.cfg.lambda_act))
                with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(
                        raw_bytes=x_b.unsqueeze(0),
                        lang_ids=lids,
                        style_ids=sids_t,
                        ponder_tax_scale=pts_t,
                        stage="D",
                        tau_s=0.25,
                        tau_m=0.25,
                        use_broca_cq=False,
                        loss_weights=loss_w,
                        lambda_rate_override=0.0,
                        lambda_var_override=0.0,
                        lambda_pc_override=0.5,
                        lambda_halt_override=0.5,
                        lambda_act_override=sft_lam_act,
                        doc_cu_seqlens=cu,
                        record_traces=False,
                    )
                if sync_bool_any(not bool(torch.isfinite(out.loss_total).item()), dist_ctx, device):
                    step_finite = False
                    break
                scaled_loss = out.loss_total / float(grad_accum)
                scaled_loss.backward()
                for k, v in out.metrics.items():
                    accum_m[k] = accum_m.get(k, 0.0) + float(v) / float(grad_accum)

            if not step_finite:
                optimizer.zero_grad(set_to_none=True)
                continue

            active_p = [p for p in model.parameters() if p.requires_grad]
            if dist_ctx.is_distributed:
                all_reduce_gradients_coalesced(active_p, dist_ctx, use_bf16_comm=use_bf16)
                accum_m = all_reduce_metrics_dict(accum_m, dist_ctx, device)

            # Track EOT row W_out[0x04] gradient norm (verified non-zero in Stage 5 audit!)
            if model.sgsc.w_out.weight.grad is not None:
                gn_eot = float(model.sgsc.w_out.weight.grad[0x04].detach().float().norm().item())
                if math.isfinite(gn_eot):
                    eot_grad_norm_max = max(eot_grad_norm_max, gn_eot)

            gn_t = torch.nn.utils.clip_grad_norm_(active_p, max_norm=1.0)
            bad_grad = not math.isfinite(float(gn_t.detach().item()))
            if sync_bool_any(bad_grad, dist_ctx, device):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            last_metrics = accum_m

            if dist_ctx.is_distributed and (completed % 50 == 0 or completed == total_steps):
                sync_tensor_avg_(unwrap_compiled_module(model.bcg_pc).c0, dist_ctx)

            if step_hook is not None:
                step_hook(completed, model, last_metrics)

            if is_primary and (completed == 1 or completed == total_steps or (log_every > 0 and completed % log_every == 0)):
                elapsed = max(time.perf_counter() - t_start, 1e-6)
                kb_s = (completed * global_step_bytes) / (elapsed * 1024.0)
                bpc = last_metrics.get("loss_lang", 0.0) / math.log(2.0)
                _r0_print(
                    f"[SFT {completed:04d}/{total_steps:04d}] loss={last_metrics.get('loss_total', 0.0):.4f} "
                    f"bpc={bpc:.3f} pc={last_metrics.get('loss_pc', 0.0):.4f} "
                    f"halt={last_metrics.get('loss_halt', 0.0):.4f} "
                    f"lr={optimizer.param_groups[-1]['lr']:.2e} "
                    f"mem={get_peak_memory_gb(device):.2f}GB speed={kb_s:.1f}KB/s",
                    flush=True,
                )

        if dist_ctx.is_distributed:
            sync_tensor_avg_(unwrap_compiled_module(model.bcg_pc).c0, dist_ctx)
        val_bpc = _eval_sft_val_bpc(model, val_ds, device=device, use_bf16=use_bf16, n_batches=2) if is_primary else None
        out_sf = Path(export_dir) / "dcpa_chat_sft.safetensors"
        if is_primary:
            synchronize(device)
            exported = export_final_safetensors(model, export_dir, filename="dcpa_chat_sft.safetensors")
            out_sf = exported["weights"]
            _r0_print(
                f"[Stage 5 Export] Atomically exported {out_sf} "
                f"({out_sf.stat().st_size / (1024 * 1024):.2f} MB) | val_bpc={val_bpc}",
                flush=True,
            )
            if drive_export_dir is not None:
                out_sf = sync_file_to_drive(exported["weights"], drive_export_dir, dest_name="dcpa_chat_sft.safetensors")
                sync_file_to_drive(exported["config"], drive_export_dir, dest_name="config.json")
                _r0_print(f"[Stage 5 Export] Synced to Drive -> {out_sf}", flush=True)
            if hasattr(os, "sync"):
                os.sync()
        dist_barrier(dist_ctx)

        return Stage5SFTSummary(
            completed_steps=total_steps,
            total_steps=total_steps,
            final_loss=float(last_metrics.get("loss_total", 0.0)),
            final_loss_lang=float(last_metrics.get("loss_lang", 0.0)),
            val_bpc=val_bpc,
            export_path=out_sf,
            active_params=int(freeze_info["active_params"]),
            frozen_params=int(freeze_info["frozen_params"]),
            eot_grad_norm_max=eot_grad_norm_max,
        )
    finally:
        for ds in train_datasets:
            ds.close()
        if val_ds is not None:
            val_ds.close()
        cleanup_distributed_context(dist_owned)


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA Stage 5 Pragmatic & Capability SFT Trainer")
    ap.add_argument("--weights", type=str, default="artifacts/export/dcpa_final.safetensors")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default="artifacts/sft_data")
    ap.add_argument("--cap-data-dir", type=str, default=None, help="Optional capability shards dir (cap_reasoning/writing/toolcall)")
    ap.add_argument("--export-dir", type=str, default="artifacts/export")
    ap.add_argument("--drive-export-dir", type=str, default=None)
    ap.add_argument("--total-steps", type=int, default=8000)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--target-tokens-bytes-per-step", type=int, default=65536)
    ap.add_argument("--micro-batch-bytes", type=int, default=None)
    ap.add_argument(
        "--fixed-global-batch",
        action="store_true",
        help="Keep exact global step bytes across ranks instead of scaling 64 KB per GPU.",
    )
    ap.add_argument("--max-doc-bytes", type=int, default=2048)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="GPU count: 'auto' (all visible GPUs; 1 process if <=1 GPU), '1' (force single-GPU), or N (e.g. 2, 4, 8).",
    )
    ap.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional comma-separated physical GPU IDs (e.g. '2' or '0,2').",
    )
    ap.add_argument("--compile-mode", choices=["none", "auto", "reduce-overhead"], default="auto")
    ap.add_argument("--seed", type=int, default=202605)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    spawn_rc = maybe_relaunch_distributed(
        __file__,
        sys.argv[1:],
        gpus=args.gpus,
        gpu_ids=args.gpu_ids,
        device=args.device,
    )
    if spawn_rc is not None:
        sys.exit(spawn_rc)

    run_stage5_sft(
        weights_path=args.weights,
        config_path=args.config,
        data_dir=args.data_dir,
        cap_data_dir=args.cap_data_dir,
        export_dir=args.export_dir,
        drive_export_dir=args.drive_export_dir,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        target_tokens_bytes_per_step=args.target_tokens_bytes_per_step,
        micro_batch_bytes=args.micro_batch_bytes,
        fixed_global_batch=args.fixed_global_batch,
        max_doc_bytes=args.max_doc_bytes,
        device_str=args.device,
        compile_mode=args.compile_mode,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
