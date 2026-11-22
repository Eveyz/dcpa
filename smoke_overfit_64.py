"""64-byte rapid overfit smoke test for DCPA (cross-platform: CUDA / Apple MPS / CPU).

Implements Iron Law #1.2 & #1.4:
  - Auto-selects device via `utils/device.py` (`cuda -> mps -> cpu`).
  - Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple MPS.
  - Enables `bfloat16` autocast ONLY on CUDA (pure `float32` on MPS/CPU).
  - Supports reduced `target_tokens_bytes_per_step=8192` local verification budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from utils.device import (
    amp_autocast,
    describe_hardware_profile,
    detect_hardware_profile,
    resolve_micro_batch_plan,
    synchronize,
)


def run_smoke_overfit_64(
    steps: int = 80,
    device_str: str = "auto",
    target_tokens_bytes_per_step: int = 8192,
    seed: int = 2026,
) -> dict[str, float]:
    profile = detect_hardware_profile(
        preferred=device_str,
        deterministic=False,
        target_tokens_bytes_per_step=target_tokens_bytes_per_step,
    )
    device = profile.device
    seq_len = 64
    micro_bs, grad_accum, eff_bytes = resolve_micro_batch_plan(
        profile=profile,
        seq_len=seq_len,
        target_tokens_bytes_per_step=target_tokens_bytes_per_step,
    )
    # Cap micro_bs for rapid single-sample 64B memorization smoke
    batch_size = min(micro_bs, 8)

    print(describe_hardware_profile(profile), flush=True)
    print(
        f"[smoke_overfit_64] seq_len={seq_len} | batch_size={batch_size} | "
        f"planned_micro_bs={micro_bs} | planned_accum={grad_accum} | "
        f"target_step_bytes={eff_bytes}",
        flush=True,
    )

    torch.manual_seed(seed)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(), lr=3e-3)
    for g in optimizer.param_groups:
        g["lr"] = 3e-3 if g["name"] != "scalar_gates" else 5e-4

    sample_text = "A zibber is a frog! 你好🦀 fn main() { let x = 42; }\n"
    raw_py = sample_text.encode("utf-8")[:seq_len]
    if len(raw_py) < seq_len:
        raw_py = raw_py + b" " * (seq_len - len(raw_py))

    raw_batch = torch.tensor(list(raw_py), dtype=torch.uint8, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    lang_batch = torch.ones((batch_size,), dtype=torch.long, device=device)

    t0 = time.perf_counter()
    initial_loss: float | None = None
    final_loss = float("inf")
    final_lang = float("inf")

    model.train()
    for step in range(1, steps + 1):
        stage = "A" if step <= steps // 2 else "B"
        model.set_stage_freeze(stage)
        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device, dtype=torch.bfloat16, enabled=profile.use_bf16):
            out = model(
                raw_bytes=raw_batch,
                lang_ids=lang_batch,
                stage=stage,
                tau_s=0.5,
                tau_m=0.5,
                use_broca_cq=False,
            )

        assert torch.isfinite(out.loss_total), f"Non-finite loss at step {step}"
        out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        cur_loss = float(out.loss_total.detach().item())
        cur_lang = float(out.loss_lang.detach().item())
        if initial_loss is None:
            initial_loss = cur_loss
        final_loss = cur_loss
        final_lang = cur_lang

        if step in (1, steps // 2, steps):
            print(
                f"  [step {step:03d}/{steps:03d} | Stage {stage}] "
                f"loss_total={cur_loss:.4f} loss_lang={cur_lang:.4f} "
                f"spike_rate={out.metrics['spike_rate']:.3f}",
                flush=True,
            )

    synchronize(device)
    elapsed = time.perf_counter() - t0
    assert initial_loss is not None
    assert final_lang < initial_loss * 0.65, (
        f"Smoke overfit did not converge enough: initial_total={initial_loss:.4f}, final_lang={final_lang:.4f}"
    )

    print(
        f"[PASS] smoke_overfit_64 converged in {elapsed:.2f}s on {device.type}: "
        f"initial_loss={initial_loss:.4f} -> final_total={final_loss:.4f} (lang={final_lang:.4f})",
        flush=True,
    )
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_lang": final_lang,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA 64-byte cross-platform smoke overfit test.")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--target-tokens-bytes-per-step", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    run_smoke_overfit_64(
        steps=args.steps,
        device_str=args.device,
        target_tokens_bytes_per_step=args.target_tokens_bytes_per_step,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
