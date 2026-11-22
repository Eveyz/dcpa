"""Gate 4: 128-Byte Differentiable Pushdown Stack Long-Range Bracket Closure Test.

Pass/Fail Quantitative Threshold (Engineering Blueprint Chapter 7):
  On synthetic Dyck-k sequences AND Rust nested code blocks reaching nesting depth 12
  and spanning > 2,000 bytes (zero KV cache, strictly 16x4 = 128B stack Xi_t):
  Closing bracket {}()[] pairing accuracy MUST be >= 99.5%.
"""

from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.data.synthetic import generate_dyck_sequence
from dcpa.model import DCPAModel


_CLOSE_BYTES = [ord("}"), ord(")"), ord("]")]


def _make_training_batch(
    rng: random.Random,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    close_masks = []
    lang_ids = []
    for _ in range(batch_size):
        rust_flav = rng.random() < 0.5
        depth = rng.randint(4, 14)
        # Keep min_bytes shorter than seq_len so full closing cascade (" end " / " ok; ") is seen during training!
        min_b = rng.randint(max(64, seq_len // 3), max(96, seq_len - 48))
        txt, close_pos = generate_dyck_sequence(rng, max_depth=depth, min_bytes=min_b, rust_flavored=rust_flav)
        raw = txt.encode("utf-8")[:seq_len]
        if len(raw) < seq_len:
            # Pad remainder with diverse printable non-bracket ASCII filler so all non-bracket bytes learn p_hold ~ 1
            pad_chars = b"abcdefghijklmnopqrstuvwxyz0123456789_=+;,.: \n"
            pad = bytes(rng.choices(pad_chars, k=seq_len - len(raw)))
            raw = raw + pad
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for pos, _ in close_pos:
            if pos < seq_len:
                mask[pos] = True
        rows.append(torch.tensor(list(raw), dtype=torch.uint8))
        close_masks.append(mask)
        lang_ids.append(4 if rust_flav else 5)
    return (
        torch.stack(rows, dim=0).to(device),
        torch.stack(close_masks, dim=0).to(device),
        torch.tensor(lang_ids, dtype=torch.long, device=device),
    )


def verify_gate4(steps: int = 800, batch_size: int = 16, train_seq_len: int = 256) -> None:
    print(f"=== Running Gate 4: 128B Differentiable Stack Long-Range Bracket Closure (steps={steps}) ===")
    torch.manual_seed(2026)
    rng = random.Random(2026)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    cfg = DCPAConfig.tiny()
    model = DCPAModel(cfg).to(device)

    # Train PMD 128B pushdown stack + byte motor micro-core (zero KV cache!)
    params = list(model.pmd.parameters())
    seen: set[int] = set()
    unique_params = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            unique_params.append(p)

    opt = torch.optim.AdamW(unique_params, lr=5e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)

    t0 = time.perf_counter()
    model.train()
    close_ids = torch.tensor(_CLOSE_BYTES, device=device, dtype=torch.long)

    for step in range(1, steps + 1):
        raw_bytes, close_mask, lang_ids = _make_training_batch(rng, batch_size, train_seq_len, device)
        B, T = raw_bytes.shape

        # Single flat segment per batch item (no concept-slot hint at all — pure 128B stack test!)
        cu_seqlens = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int32)
        f_zero = torch.zeros((B, cfg.d_s), device=device, dtype=torch.float32)
        d_zero = torch.zeros((B, cfg.d_s), device=device, dtype=torch.float32)
        l_reg = model.pmd.get_lang_vec(lang_ids)

        opt.zero_grad(set_to_none=True)
        pmd_out = model.pmd.decode_teacher_forced(
            target_bytes=raw_bytes,
            cu_seqlens=cu_seqlens,
            f_out_seg=f_zero,
            delta_syn_seg=d_zero,
            l_reg=l_reg,
        )

        # Standard next-byte CE + upweighted exact closing-bracket CE (focuses 128B stack dynamics)
        ce_all = F.cross_entropy(pmd_out.byte_logits.reshape(-1, 256), raw_bytes.reshape(-1).long(), reduction="none").view(B, T)
        weights = torch.where(close_mask, torch.full_like(ce_all, 30.0), torch.ones_like(ce_all))
        loss_all = (ce_all * weights).sum() / weights.sum()

        # Also directly supervise 3-way closing bracket discrimination at closing sites
        if close_mask.any():
            c_logits = pmd_out.byte_logits[close_mask][:, close_ids]  # [N_c, 3]
            c_targets_byte = raw_bytes[close_mask].long()
            c_targets_3 = torch.where(
                c_targets_byte == ord("}"),
                torch.zeros_like(c_targets_byte),
                torch.where(c_targets_byte == ord(")"), torch.ones_like(c_targets_byte), torch.full_like(c_targets_byte, 2)),
            )
            loss_close = F.cross_entropy(c_logits, c_targets_3)
        else:
            loss_close = torch.tensor(0.0, device=device)

        # Encourage crisp discrete stack actions (low entropy) so p_hold -> 1.0 over 2000+ byte spans!
        bos = torch.zeros((B, 1), device=device, dtype=torch.long)
        prev_bytes = torch.cat([bos, raw_bytes[:, :-1].long()], dim=1)
        u_prev = model.pmd.embed(prev_bytes)
        l_bt = l_reg.unsqueeze(1).expand(B, T, cfg.d_l)
        act_p = F.softmax(model.pmd.w_act(torch.cat([u_prev, l_bt], dim=-1)).float(), dim=-1).clamp_min(1e-7)
        act_entropy = -(act_p * torch.log(act_p)).sum(dim=-1).mean()

        loss = loss_all + 2.0 * loss_close + 0.05 * act_entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unique_params, 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == 1:
            acc = 0.0
            if close_mask.any():
                acc = float((c_logits.argmax(dim=-1) == c_targets_3).float().mean().item()) * 100.0
            print(
                f"  step {step:04d}/{steps} | loss={loss.item():.4f} close_loss={loss_close.item():.4f} "
                f"act_ent={act_entropy.item():.4f} train_close_acc={acc:.2f}% | {time.perf_counter()-t0:.1f}s"
            )

    # -------------------------------------------------------------------------
    # Zero-Shot Length Generalization Evaluation:
    # Nesting depth >= 12, span > 2,000 bytes (both Dyck-k and Rust nested code blocks)
    # -------------------------------------------------------------------------
    model.eval()
    eval_rng = random.Random(99999)
    total_closes = 0
    correct_closes = 0
    long_span_closes = 0
    long_span_correct = 0

    with torch.no_grad():
        for ep_idx in range(20):
            rust_flav = ep_idx >= 10
            txt, close_positions = generate_dyck_sequence(
                eval_rng,
                max_depth=12,
                min_bytes=2100,
                rust_flavored=rust_flav,
            )
            raw = txt.encode("utf-8")
            assert len(raw) > 2000, f"Test sequence too short: {len(raw)}"
            T = len(raw)
            x = torch.tensor(list(raw), dtype=torch.uint8, device=device).unsqueeze(0)
            cu = torch.tensor([0, T], device=device, dtype=torch.int32)
            f_zero = torch.zeros((1, cfg.d_s), device=device, dtype=torch.float32)
            d_zero = torch.zeros((1, cfg.d_s), device=device, dtype=torch.float32)
            lid = torch.tensor([4 if rust_flav else 5], device=device, dtype=torch.long)
            l_reg = model.pmd.get_lang_vec(lid)

            out = model.pmd.decode_teacher_forced(
                target_bytes=x,
                cu_seqlens=cu,
                f_out_seg=f_zero,
                delta_syn_seg=d_zero,
                l_reg=l_reg,
            )
            logits = out.byte_logits[0]  # [T, 256]

            for pos, expected_ch in close_positions:
                sub_logits = logits[pos, close_ids]
                pred_ch = chr(_CLOSE_BYTES[int(sub_logits.argmax().item())])
                ok = pred_ch == expected_ch
                total_closes += 1
                if ok:
                    correct_closes += 1
                if pos >= 2000:
                    long_span_closes += 1
                    if ok:
                        long_span_correct += 1

    acc_all = 100.0 * correct_closes / max(total_closes, 1)
    acc_long = 100.0 * long_span_correct / max(long_span_closes, 1)
    print(
        f"[RESULT] Depth-12, Span > 2,000B Dyck-k + Rust Nested Blocks:\n"
        f"  Overall bracket closure accuracy   = {acc_all:.2f}% ({correct_closes}/{total_closes})\n"
        f"  Span >= 2,000B closure accuracy    = {acc_long:.2f}% ({long_span_correct}/{long_span_closes})"
    )
    assert acc_all >= 99.5, f"GATE 4 FAIL: closing accuracy {acc_all:.2f}% < 99.5%"
    assert acc_long >= 99.5, f"GATE 4 FAIL: long-span (>=2000B) closing accuracy {acc_long:.2f}% < 99.5%"
    print("=== GATE 4 PASSED ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    verify_gate4(steps=args.steps, batch_size=args.batch_size)
