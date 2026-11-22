"""End-to-end developmental training entry point for DCPA (Stages A -> B -> C -> D).

Supports:
  - Tier 1 (DCPA-Tiny 48M, RTX 4090 / single GPU)
  - Tier 2 (DCPA-Base 152M, 2x4090 / A100 / H100)
  - Local Micro smoke runs on Apple Silicon MPS / CPU
  - Binary mmap shards produced by dcpa-data-forge OR online synthetic developmental stream
  - Mandatory float32 protection zones + bfloat16 autocast on linear projections
  - Real-time AntiCollapseMonitor emergency rescue rules
  - Safetensors + Golden Test Vector export on completion
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dcpa.config import DCPAConfig
from dcpa.data.shard_loader import BinaryShardDataset
from dcpa.data.synthetic import (
    generate_dyck_sequence,
    generate_fast_mapping_episode,
    generate_grid_physics_episode,
)
from dcpa.export import export_model_and_golden
from dcpa.model import DCPAModel
from dcpa.train.curriculum import CurriculumSchedule
from dcpa.train.monitor import AntiCollapseMonitor, project_spectral_norm
from utils.device import amp_autocast, get_device, is_bf16_enabled


def _sample_synthetic_batch(
    rng: random.Random,
    batch_size: int,
    seq_len: int,
    stage: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    raw_rows = []
    lang_ids = []
    weight_rows = []
    has_weights = stage == "D"

    for _ in range(batch_size):
        if stage == "D":
            ep = generate_fast_mapping_episode(rng, lang="en")
            b = ep.text.encode("utf-8")[:seq_len]
            w = torch.ones(seq_len, dtype=torch.float32)
            # High cross-entropy supervision on target sentence (Section 4.3 Step 5)
            s_idx = min(ep.target_byte_start, seq_len)
            e_idx = min(ep.target_byte_end, seq_len)
            w[s_idx:e_idx] = 4.0
            lid = ep.lang_id
        elif rng.random() < 0.5:
            txt = generate_grid_physics_episode(rng, num_steps=16)
            b = txt.encode("utf-8")[:seq_len]
            w = torch.ones(seq_len, dtype=torch.float32)
            lid = 5
        else:
            txt, _ = generate_dyck_sequence(rng, max_depth=8, min_bytes=seq_len, rust_flavored=(stage == "C"))
            b = txt.encode("utf-8")[:seq_len]
            w = torch.ones(seq_len, dtype=torch.float32)
            lid = 4 if stage == "C" else 5

        if len(b) < seq_len:
            b = b + b"\n" * (seq_len - len(b))
        raw_rows.append(torch.tensor(list(b), dtype=torch.uint8))
        lang_ids.append(lid)
        weight_rows.append(w)

    raw_t = torch.stack(raw_rows, dim=0).to(device)
    lang_t = torch.tensor(lang_ids, dtype=torch.long, device=device)
    w_t = torch.stack(weight_rows, dim=0).to(device) if has_weights else None
    return raw_t, lang_t, w_t


def train_loop(
    preset: str = "micro",
    total_steps: int = 200,
    batch_size: int = 4,
    seq_len: int = 128,
    shard_prefix: str | None = None,
    out_dir: str = "checkpoints/run",
    device_str: str = "auto",
    seed: int = 1337,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    device = get_device(preferred=device_str, deterministic=False)

    cfg = {"base": DCPAConfig.base, "tiny": DCPAConfig.tiny, "micro": DCPAConfig.micro}[preset]()
    model = DCPAModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameter_groups())
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    schedule = CurriculumSchedule(total_steps=total_steps)
    monitor = AntiCollapseMonitor(target_spike_rate=cfg.target_spike_rate)

    shard_ds = None
    if shard_prefix is not None:
        shard_ds = BinaryShardDataset(f"{shard_prefix}.bin", f"{shard_prefix}.idx", seq_len=seq_len)

    last_stage = None
    last_metrics: dict[str, float] = {}
    directives = monitor.observe({}, stage="A", step_in_stage=0)

    use_bf16 = is_bf16_enabled(device)

    for step in range(total_steps):
        sc = schedule.get(step)
        if sc.stage != last_stage:
            model.set_stage_freeze(sc.stage)
            last_stage = sc.stage

        for g, blr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = blr * sc.lr_scale

        if shard_ds is not None and len(shard_ds) >= batch_size and sc.stage != "D":
            items = [shard_ds[rng.randrange(len(shard_ds))] for _ in range(batch_size)]
            raw_bytes = torch.stack([x[0] for x in items], dim=0).to(device)
            lang_ids = torch.stack([x[1] for x in items], dim=0).to(device)
            loss_w = None
        else:
            raw_bytes, lang_ids, loss_w = _sample_synthetic_batch(rng, batch_size, seq_len, sc.stage, device)

        optimizer.zero_grad(set_to_none=True)
        tau_m = sc.tau_m if directives.tau_m_override is None else directives.tau_m_override

        with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                raw_bytes=raw_bytes,
                lang_ids=lang_ids,
                stage=sc.stage,
                tau_s=sc.tau_s,
                tau_m=tau_m,
                use_broca_cq=sc.use_broca_cq,
                loss_weights=loss_w,
                lambda_rate_override=directives.lambda_rate_override,
                dual_rate_lagrange=directives.dual_rate_lagrange,
                lambda_var_override=directives.lambda_var_override,
                merge_kl_weight=directives.merge_kl_weight,
                ach_hint_weight=directives.ach_hint_weight,
            )

        out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        directives = monitor.observe(out.metrics, stage=sc.stage, step_in_stage=sc.step_in_stage)
        if directives.apply_wsem_spectral_norm:
            project_spectral_norm(model.sgsc.w_sem.weight, max_sigma=1.0)

        last_metrics = out.metrics
        if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            alert_str = f" ALERTS={directives.alerts}" if directives.alerts else ""
            print(
                f"[step {step:05d}/{total_steps} | Stage {sc.stage}] "
                f"total={out.metrics['loss_total']:.4f} lang={out.metrics['loss_lang']:.4f} "
                f"pc={out.metrics['loss_pc']:.4f} rate={out.metrics['spike_rate']:.3f} "
                f"comp={out.metrics['compression_ratio']:.1f}x std={out.metrics['c_slow_std_mean']:.2f}"
                f"{alert_str}"
            )

    if shard_ds is not None:
        shard_ds.close()

    out_path = Path(out_dir)
    export_model_and_golden(model, out_path)
    return last_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA Four-Stage Developmental Trainer")
    ap.add_argument("--preset", choices=["micro", "tiny", "base"], default="micro")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--shard-prefix", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="checkpoints/run")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    train_loop(
        preset=args.preset,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shard_prefix=args.shard_prefix,
        out_dir=args.out_dir,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
