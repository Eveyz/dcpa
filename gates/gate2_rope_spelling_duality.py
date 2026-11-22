"""Gate 2: Holographic RoPE Phase Binding & Inverse Phase Spelling Duality.

Pass/Fail Quantitative Threshold (Engineering Blueprint Chapter 7):
  Train SGSC stem code f_m -> PMD RoPE inverse phase demodulation micro-core for 5000 steps
  on 100,000 random Chinese/English words & code identifiers (byte lengths 1..14).
  Exact byte-for-byte full-word autoregressive recovery on UNSEEN test words MUST be >= 99.2%.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from utils.device import get_device


_EN_SYL = [
    "ba", "be", "bi", "bo", "bu", "ca", "ce", "co", "da", "de", "di", "do",
    "fa", "fe", "fi", "fo", "ga", "ge", "gi", "go", "ha", "he", "hi", "ka",
    "ke", "ki", "la", "le", "li", "lo", "ma", "me", "mi", "mo", "na", "ne",
    "ni", "no", "pa", "pe", "pi", "po", "ra", "re", "ri", "ro", "sa", "se",
    "si", "so", "ta", "te", "ti", "to", "va", "ve", "vi", "wa", "we", "wi",
    "za", "zi", "bl", "cr", "dr", "fl", "gl", "pr", "sp", "st", "tr", "ck",
    "nd", "ng", "nt", "rk", "rm", "sh", "th", "ch", "er", "in", "on", "en",
]
_ZH_CHARS = list(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
)
_CODE_WORDS = [
    "buf", "ctx", "ptr", "idx", "len", "cap", "val", "key", "map", "vec",
    "node", "slot", "edge", "span", "tag", "mem", "err", "res", "cfg", "ops",
    "scan", "rope", "gate", "step", "load", "save", "read", "sync", "push", "pop",
    "init", "eval", "grad", "norm", "head", "tail", "fast", "slow", "core", "byte",
]


def build_word_dataset(num_words: int = 100_000, seed: int = 42) -> tuple[dict[int, list[bytes]], list[bytes]]:
    """Build 100,000 random Chinese/English/code tokens (1..14 bytes) + strictly disjoint 1,000 unseen test set."""
    rng = random.Random(seed)
    all_unique: set[bytes] = set()
    ordered: list[bytes] = []

    target_total = num_words + 1000
    while len(ordered) < target_total:
        kind = rng.random()
        if kind < 0.40:
            # English / pseudo-word (1..14 bytes)
            n_syl = rng.randint(1, 5)
            w = "".join(rng.choices(_EN_SYL, k=n_syl))
            if rng.random() < 0.25:
                w = w[: rng.randint(1, len(w))]
            b = w.encode("utf-8")
        elif kind < 0.72:
            # Chinese 1..4 chars (3..12 UTF-8 bytes)
            n_ch = rng.randint(1, 4)
            w = "".join(rng.choices(_ZH_CHARS, k=n_ch))
            b = w.encode("utf-8")
        else:
            # Code identifier (snake_case / camelCase / short symbol, 1..14 bytes)
            if rng.random() < 0.5:
                w = rng.choice(_CODE_WORDS) + "_" + rng.choice(_CODE_WORDS)
            else:
                c1, c2 = rng.choice(_CODE_WORDS), rng.choice(_CODE_WORDS)
                w = c1 + c2.capitalize()
            if rng.random() < 0.3:
                w = w + str(rng.randint(0, 99))
            b = w.encode("utf-8")

        if 1 <= len(b) <= 14 and b not in all_unique:
            all_unique.add(b)
            ordered.append(b)

    rng.shuffle(ordered)
    test_words = ordered[:1000]
    train_words = ordered[1000:]

    by_len: dict[int, list[bytes]] = {l: [] for l in range(1, 15)}
    for w in train_words:
        by_len[len(w)].append(w)
    # Ensure every bucket 1..14 is non-empty
    for l in range(1, 15):
        if not by_len[l]:
            by_len[l].append(("a" * l).encode("utf-8"))
    return by_len, test_words


def verify_gate2(steps: int = 5000, batch_size: int = 256, preset: str = "base") -> None:
    print(f"=== Running Gate 2: Holographic RoPE Spelling Duality (steps={steps}, preset={preset}) ===")
    torch.manual_seed(2026)
    rng = random.Random(2026)

    device = get_device("auto", deterministic=False)
    print(f"Device: {device}")

    cfg = {"base": DCPAConfig.base, "tiny": DCPAConfig.tiny, "micro": DCPAConfig.micro}[preset]()
    model = DCPAModel(cfg).to(device)

    # Optimize ONLY SGSC stem encoder + PMD inverse RoPE demodulator & byte motor micro-core
    params = (
        list(model.sgsc.parameters())
        + list(model.pmd.u_demod.parameters())
        + list(model.pmd.w_chi.parameters())
        + list(model.pmd.w_act.parameters())
        + list(model.pmd.w_tag.parameters())
        + list(model.pmd.w_st.parameters())
        + [model.pmd.m0]
        + list(model.pmd.w_alpha_mot.parameters())
        + list(model.pmd.w_m.parameters())
        + list(model.pmd.w_e_init.parameters())
        + list(model.pmd.w_drain.parameters())
        + list(model.pmd.lang_embed.parameters())
    )
    # Deduplicate (embed / w_out shared)
    seen_ids: set[int] = set()
    unique_params = []
    for p in params:
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_params.append(p)

    opt = torch.optim.AdamW(unique_params, lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    train_buckets, test_words = build_word_dataset(num_words=100_000, seed=42)
    valid_lens = [l for l in range(1, 15) if len(train_buckets[l]) > 0]
    len_weights = [len(train_buckets[l]) for l in valid_lens]

    # Pre-convert buckets into uint8 tensors for fast zero-copy batching
    bucket_tensors = {
        l: torch.tensor([list(w) for w in ws], dtype=torch.uint8)
        for l, ws in train_buckets.items()
    }

    t0 = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        # Sample an exact length bucket (zero padding!)
        seg_len = rng.choices(valid_lens, weights=len_weights, k=1)[0]
        pool = bucket_tensors[seg_len]
        idx = torch.randint(0, pool.shape[0], (batch_size,))
        batch_bytes = pool[idx].to(device)  # [B, L] uint8

        # Single segment per word: forced spike strictly at last byte L-1
        forced_spikes = torch.zeros((batch_size, seg_len), device=device, dtype=torch.bool)
        forced_spikes[:, seg_len - 1] = True

        opt.zero_grad(set_to_none=True)
        sgsc_out = model.sgsc(batch_bytes, tau_s=0.35, forced_spikes=forced_spikes)

        l_reg = model.pmd.get_lang_vec(torch.ones((batch_size,), device=device, dtype=torch.long))
        delta_zero = torch.zeros_like(sgsc_out.f_stem)

        pmd_out = model.pmd.decode_teacher_forced(
            target_bytes=batch_bytes,
            cu_seqlens=sgsc_out.cu_seqlens,
            f_out_seg=sgsc_out.f_stem,
            delta_syn_seg=delta_zero,
            l_reg=l_reg,
        )

        loss_ce = F.cross_entropy(pmd_out.byte_logits.reshape(-1, 256), batch_bytes.reshape(-1).long())
        loss_halt = F.binary_cross_entropy(
            pmd_out.p_seg_end.clamp(1e-6, 1.0 - 1e-6),
            forced_spikes.float(),
        )
        loss = loss_ce + 0.5 * loss_halt
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unique_params, 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:04d}/{steps} | L={seg_len:2d} | ce={loss_ce.item():.5f} "
                f"halt={loss_halt.item():.5f} | {elapsed:.1f}s"
            )

    # -------------------------------------------------------------------------
    # Evaluation: Free-running autoregressive spelling recovery on 1,000 UNSEEN words
    # (Starting ONLY from f_stem, halting strictly when motor charge E_{n,tau} <= 0!)
    # -------------------------------------------------------------------------
    model.eval()
    exact_matches = 0
    byte_matches = 0
    byte_total = 0

    # Group unseen test words by length for vectorized encoding, then decode autoregressively
    test_by_len: dict[int, list[bytes]] = {}
    for w in test_words:
        test_by_len.setdefault(len(w), []).append(w)

    with torch.no_grad():
        for seg_len, ws in sorted(test_by_len.items()):
            B_t = len(ws)
            x = torch.tensor([list(w) for w in ws], dtype=torch.uint8, device=device)
            forced_spikes = torch.zeros((B_t, seg_len), device=device, dtype=torch.bool)
            forced_spikes[:, seg_len - 1] = True
            f_stem = model.sgsc(x, tau_s=0.35, forced_spikes=forced_spikes).f_stem  # [B_t, d_s]

            l_reg = model.pmd.get_lang_vec(torch.ones((B_t,), device=device, dtype=torch.long))
            delta_zero = torch.zeros_like(f_stem)

            m_cur = model.pmd.m0.unsqueeze(0).expand(B_t, -1).clone()
            xi_cur = torch.zeros((B_t, 16, 4), device=device, dtype=torch.float32)
            charge_cur = model.pmd.init_segment_charge(f_stem, delta_zero, l_reg)
            prev_b = torch.zeros((B_t,), device=device, dtype=torch.long)

            finished = [False] * B_t
            emitted: list[list[int]] = [[] for _ in range(B_t)]

            for tau_step in range(1, 17):
                logits, m_cur, xi_cur, charge_cur, _ = model.pmd.step_byte_streaming(
                    prev_byte=prev_b,
                    f_out=f_stem,
                    delta_syn=delta_zero,
                    l_reg=l_reg,
                    tau_step=tau_step,
                    m_prev=m_cur,
                    xi_prev=xi_cur,
                    charge_prev=charge_cur,
                )
                pred_b = logits.argmax(dim=-1)
                pred_list = pred_b.tolist()
                charge_list = charge_cur.tolist()
                for bi in range(B_t):
                    if not finished[bi]:
                        emitted[bi].append(pred_list[bi])
                        if charge_list[bi] <= 0.0:
                            finished[bi] = True
                prev_b = pred_b
                if all(finished):
                    break

            for bi, gt_bytes in enumerate(ws):
                pred_bytes = bytes(emitted[bi])
                if pred_bytes == gt_bytes:
                    exact_matches += 1
                byte_total += len(gt_bytes)
                for a_b, b_b in zip(pred_bytes, gt_bytes):
                    if a_b == b_b:
                        byte_matches += 1

    exact_rate = 100.0 * exact_matches / len(test_words)
    byte_rate = 100.0 * byte_matches / max(byte_total, 1)
    print(
        f"[RESULT] Unseen Test Set ({len(test_words)} words): "
        f"Exact Full-Word Byte Recovery = {exact_rate:.2f}% ({exact_matches}/{len(test_words)}), "
        f"Per-Byte Accuracy = {byte_rate:.2f}%"
    )
    assert exact_rate >= 99.2, f"GATE 2 FAIL: exact recovery {exact_rate:.2f}% < 99.2%"
    print("=== GATE 2 PASSED ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--preset", choices=["micro", "tiny", "base"], default="base")
    args = ap.parse_args()
    verify_gate2(steps=args.steps, batch_size=args.batch_size, preset=args.preset)
