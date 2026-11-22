"""Stage 5 Pragmatic SFT & M3 Pro O(1) Stateful Interactive REPL Verification Suite.

Covers Stage 5 Addendum Section 5:
  1. `test_1_chat_protocol_mask_alignment_and_sft_corpus_utf8`:
       - Exact `0x1E / 0x1F / 0x04` control byte `sft_loss_mask` alignment & 100% strict UTF-8
         validity across EN / ZH / JA / Python / Rust in 4-bucket `--smoke-quota` corpus.
       - Zero Train / Val blake2b overlap & 100% `tree-sitter` AST pass on Bucket B code samples.
       - Exact PMD 128B pushdown stack Xi_t zero-reset on the tick immediately after `0x04`.
  2. `test_2_stage5_surgical_freeze_and_eot_gradient`:
       - 20 steps of Stage 5 SFT from exported `dcpa_final.safetensors`:
         * Bit-identical `blake2b` hashes before/after on SGSC `theta_raw` (`vartheta`),
           Hippocampus `theta_i`, `alpha_q`, and all frozen SGSC/Hippocampus gates.
         * Non-zero gradient and weight update on `W_out[0x04]` (EOT termination row).
         * Clean atomic export & reload of `dcpa_chat_sft.safetensors`.
  3. `test_3_m3_repl_stateful_multiturn_eot_and_sleep_consolidation`:
       - Headless `DCPAChatSession` two-turn stateful interaction (Turn 1 novel word teaching ->
         Turn 2 query -> clean `0x04` EOT halt -> `/reset` preserves Hippocampus -> `/sleep`
         50-step SWR consolidation updating ONLY `W_sem` & `W_lex_slow` and exporting
         `dcpa_chat_consolidated.safetensors` -> `/wipe` clears everything).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cli.chat_m3 import DCPAChatSession
from data.build_sft_stage5 import SFTByteDataset, build_stage5_sft_corpus, is_valid_code_ast
from data.ragged_dataloader import IDX_DTYPE
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from train.curriculum_state_machine import hash_named_parameters
from train.train_dcpa import export_final_safetensors
from train.train_stage5_sft import (
    STAGE5_FROZEN_NAMES,
    load_dcpa_model_from_weights,
    run_stage5_sft,
)
from utils.chat_protocol import (
    TURN_ASST_PREFIX,
    TURN_EOT_BYTE,
    TURN_USER_PREFIX,
    build_sft_byte_loss_weights,
    encode_dialogue_turns,
    verify_mask_alignment,
)
from utils.device import get_deterministic_test_device, seed_everything


# =============================================================================
# Test 1: Control-Byte Framing, UTF-8 Integrity, 4-Bucket Corpus & 0x04 Stack Reset
# =============================================================================


def test_1_chat_protocol_mask_alignment_and_sft_corpus_utf8(tmp_path: Path):
    sft_dir = tmp_path / "sft_smoke"
    meta = build_stage5_sft_corpus(
        out_dir=sft_dir,
        smoke_quota=True,
        offline=True,
        seed=202605,
    )
    assert meta["train"]["total_bytes"] >= 900_000
    assert meta["val"]["total_bytes"] >= 90_000

    # Audit every single dialogue in train & val splits
    split_hashes: dict[str, set[bytes]] = {"train": set(), "val": set()}
    buckets_seen: set[int] = set()
    langs_seen: set[int] = set()
    code_checked = 0

    for split in ("train", "val"):
        bin_mm = np.memmap(sft_dir / f"sft_{split}.bin", dtype=np.uint8, mode="r")
        msk_mm = np.memmap(sft_dir / f"sft_{split}.loss_mask.bin", dtype=np.uint8, mode="r")
        idx_mm = np.memmap(sft_dir / f"sft_{split}.idx", dtype=IDX_DTYPE, mode="r")
        assert len(bin_mm) == len(msk_mm)

        for rec in idx_mm:
            off = int(rec["offset"])
            length = int(rec["length"])
            lid = int(rec["lang_id"])
            bkt = int(rec["source_id"])
            raw_b = bytes(bin_mm[off : off + length])
            msk_b = bytes(msk_mm[off : off + length])

            # 100% strict UTF-8 decoding
            text = raw_b.decode("utf-8", errors="strict")
            assert raw_b.startswith(TURN_USER_PREFIX)
            assert TURN_ASST_PREFIX in raw_b
            assert TURN_EOT_BYTE in raw_b

            # 100% exact sft_loss_mask alignment against 0x1E / 0x1F / 0x04 control bytes
            assert verify_mask_alignment(raw_b, msk_b), f"Mask alignment failed in {split} offset={off}"

            # Verify 2.0x EOT focal weight helper
            x_t = torch.from_numpy(np.frombuffer(raw_b, dtype=np.uint8).copy())
            m_t = torch.from_numpy(np.frombuffer(msk_b, dtype=np.uint8).copy().astype(np.bool_))
            w_t = build_sft_byte_loss_weights(x_t, m_t, eot_weight=2.0)
            eot_idx = torch.nonzero(x_t == TURN_EOT_BYTE, as_tuple=False).squeeze(-1)
            assert eot_idx.numel() >= 1
            assert torch.all(w_t[eot_idx] == 2.0)
            assert torch.all(w_t[~m_t] == 0.0)

            # Verify Bucket B code blocks pass tree-sitter 100%
            if bkt == 1:
                m_code = re.search(r"```(python|rust)\n(.*?)```", text, re.DOTALL)
                assert m_code is not None
                assert is_valid_code_ast(m_code.group(2), m_code.group(1))
                code_checked += 1

            buckets_seen.add(bkt)
            langs_seen.add(lid)
            split_hashes[split].add(hashlib.blake2b(raw_b, digest_size=16).digest())

    assert buckets_seen == {0, 1, 2, 3}, f"Missing SFT buckets: {buckets_seen}"
    assert langs_seen == {0, 1, 2, 3, 4}, f"Missing languages: {langs_seen}"
    assert code_checked > 50
    assert len(split_hashes["train"].intersection(split_hashes["val"])) == 0, "Train/Val leakage detected!"

    # Verify PMD 128B pushdown stack Xi_t resets cleanly on the tick immediately after 0x04
    dev = get_deterministic_test_device()
    seed_everything(42, device=dev)
    model = DCPAModel(DCPAConfig.micro()).to(dev).eval()
    dlg = encode_dialogue_turns(
        [
            ("Check {{{ brackets.", "Reply text here."),
            ("Now write Rust.", "fn f() { }"),
        ],
        lang_id=4,
    )
    x_raw = torch.tensor(list(dlg.raw_bytes), dtype=torch.uint8, device=dev).unsqueeze(0)
    with torch.no_grad():
        out = model(raw_bytes=x_raw, lang_ids=torch.tensor([4], device=dev), stage="A")
    # Compare stack at `t = eot_pos + 1` against a fresh 1-step reset from xi=0
    eot0 = dlg.eot_positions[0]
    xi_after_eot = out.pmd.stack_xi[0, eot0 + 1]  # [16, 4]
    # Because Xi_{eot0} was zeroed before push/pop/hold at eot0+1, slots 1..15 MUST be strictly 0.0!
    assert float(xi_after_eot[1:].abs().max().item()) == 0.0, (
        f"PMD 128B stack did NOT reset after 0x04! slots[1:] max={xi_after_eot[1:].abs().max().item()}"
    )

    # Also verify streaming `step_single_byte` resets stack identically when prev_byte == 0x04
    xi_nonzero = torch.ones((1, 16, 4), device=dev, dtype=torch.float32) * 0.75
    f_dummy = torch.zeros((1, model.cfg.d_s), device=dev)
    l_dummy = model.pmd.get_lang_vec(torch.tensor([4], device=dev))
    _, _, xi_next_stream, _, _ = model.pmd.step_single_byte(
        prev_byte=torch.tensor([TURN_EOT_BYTE], dtype=torch.long, device=dev),
        f_out=f_dummy,
        delta_syn=f_dummy,
        l_reg=l_dummy,
        tau_step=1,
        m_prev=f_dummy,
        xi_prev=xi_nonzero,
        charge_prev=torch.ones((1,), device=dev),
    )
    assert float(xi_next_stream[0, 1:].abs().max().item()) == 0.0
    print(
        f"\n[Stage 5 Test 1] {meta['train']['num_docs']} train + {meta['val']['num_docs']} val dialogues | "
        f"100% UTF-8 & 0x1E/0x1F/0x04 mask aligned | {code_checked} AST-OK code samples | "
        "0 Train/Val overlap | PMD 0x04 stack reset=0.000e+00"
    )


# =============================================================================
# Test 2: 20-Step Stage 5 Surgical Freeze Audit & W_out[0x04] Gradient Verification
# =============================================================================


def test_2_stage5_surgical_freeze_and_eot_gradient(tmp_path: Path):
    dev = get_deterministic_test_device()
    seed_everything(77, device=dev)

    # 1. Build compact offline SFT dataset
    sft_dir = tmp_path / "sft_data"
    build_stage5_sft_corpus(out_dir=sft_dir, smoke_quota=True, offline=True, seed=77)

    # 2. Export a base Stage-4 micro model to dcpa_final.safetensors
    base_dir = tmp_path / "base_export"
    base_model = DCPAModel(DCPAConfig.micro()).to(dev)
    base_exp = export_final_safetensors(base_model, base_dir, filename="dcpa_final.safetensors")

    # Hash ALL frozen parameters before SFT (esp. sgsc.theta_raw, hippocampus.theta_i, hippocampus.alpha_q)
    loaded_before = load_dcpa_model_from_weights(base_exp["weights"], device=dev)
    frozen_hashes_before = hash_named_parameters(loaded_before, STAGE5_FROZEN_NAMES)
    w_out_eot_before = loaded_before.sgsc.w_out.weight[TURN_EOT_BYTE].detach().clone()

    # 3. Run 20 steps of Stage 5 surgical SFT
    sft_export_dir = tmp_path / "sft_export"
    summary = run_stage5_sft(
        weights_path=base_exp["weights"],
        data_dir=sft_dir,
        export_dir=sft_export_dir,
        total_steps=20,
        warmup_steps=4,
        target_tokens_bytes_per_step=1024,
        micro_batch_bytes=1024,
        max_doc_bytes=2048,
        device_str="cpu",
        compile_mode="none",
        deterministic=True,
        seed=77,
        log_every=10,
    )
    assert summary.completed_steps == 20
    assert summary.export_path.exists()
    assert summary.eot_grad_norm_max > 0.0, f"W_out[0x04] received zero gradient!"

    # 4. Reload dcpa_chat_sft.safetensors and assert bit-exact blake2b invariance on frozen params!
    loaded_after = load_dcpa_model_from_weights(summary.export_path, device=dev)
    frozen_hashes_after = hash_named_parameters(loaded_after, STAGE5_FROZEN_NAMES)

    for key in ("sgsc.theta_raw", "hippocampus.theta_i", "hippocampus.alpha_q"):
        assert key in frozen_hashes_before and key in frozen_hashes_after
    mismatches = [k for k in frozen_hashes_before if frozen_hashes_before[k] != frozen_hashes_after[k]]
    assert not mismatches, f"Surgical freeze violated! Mutated frozen params: {mismatches}"

    w_out_eot_delta = float((loaded_after.sgsc.w_out.weight[TURN_EOT_BYTE] - w_out_eot_before).abs().max().item())
    assert w_out_eot_delta > 0.0, "W_out[0x04] did not update during Stage 5 SFT!"

    print(
        f"\n[Stage 5 Test 2] 20-step SFT OK | frozen blake2b identical: {len(frozen_hashes_before)}/{len(frozen_hashes_before)} "
        f"(incl. sgsc.theta_raw, hippocampus.theta_i, hippocampus.alpha_q) | "
        f"max ||grad(W_out[0x04])||={summary.eot_grad_norm_max:.4e} | "
        f"max |dW_out[0x04]|={w_out_eot_delta:.4e} | export={summary.export_path.stat().st_size / (1024**2):.2f} MB"
    )


# =============================================================================
# Test 3: Headless M3 Pro O(1) Stateful REPL + EOT Stop + `/sleep` Consolidation
# =============================================================================


def test_3_m3_repl_stateful_multiturn_eot_and_sleep_consolidation(tmp_path: Path):
    dev = get_deterministic_test_device()
    seed_everything(99, device=dev)

    # Export a micro SFT weight file to load cleanly into DCPAChatSession
    exp_dir = tmp_path / "repl_weights"
    model = DCPAModel(DCPAConfig.micro()).to(dev)
    exp = export_final_safetensors(model, exp_dir, filename="dcpa_chat_sft.safetensors")

    loaded = load_dcpa_model_from_weights(exp["weights"], device=dev)
    session = DCPAChatSession(loaded, device=dev, temp=0.3, top_p=0.9, max_new_bytes=32)

    # Verify slash commands `/probe on` and `/lang zh`
    ex, msg = session.handle_command("/probe on")
    assert ex is False and session.probe_mode == "on"
    ex, msg = session.handle_command("/lang zh")
    assert ex is False and session.lang_mode == "zh"

    # Turn 1: Teach novel pseudo-word -> verify persistent Hippocampus write
    chunks_t1: list[str] = []
    reply1, tel1 = session.chat_turn(
        "记住新词：「狴晶」是一种在极低温下散发蓝光的超导陶瓷。请用它造一个因果句。",
        on_chunk=chunks_t1.append,
        force_eot_after=12,
    )
    assert tel1.stopped_on_eot is True
    assert tel1.num_spikes > 0
    assert float(session.carry.fast_state.u_lex.sum().item()) > 0.0
    assert float(session.carry.fast_state.m_c_from_l.abs().max().item()) > 0.0

    # Turn 2: Follow-up query sharing persistent O(1) carry state -> clean 0x04 halt
    u_lex_before_t2 = float(session.carry.fast_state.u_lex.sum().item())
    reply2, tel2 = session.chat_turn(
        "我说错了，狴晶发的是紫光不是蓝光，请修正刚才的句子。",
        force_eot_after=10,
    )
    assert tel2.stopped_on_eot is True
    assert float(session.carry.fast_state.u_lex.sum().item()) >= u_lex_before_t2 * 0.9

    # `/reset` clears 8-slot working memory & restores Hippocampus to `M_tools_base` (non-zero built-in tools!)
    session.handle_command("/reset")
    assert torch.allclose(session.carry.fast_state.u_lex, session.m_tools_base.u_lex)
    assert float(session.carry.fast_state.u_lex.sum().item()) > 0.0

    # `/sleep` 50-step local SWR consolidation updates ONLY `W_sem` & `W_lex_slow`!
    all_names = {n for n, _ in session.model.named_parameters()}
    swr_targets = {"sgsc.w_sem.weight", "hippocampus.w_lex_slow.weight"}
    non_swr_names = all_names - swr_targets
    hashes_before_sleep = hash_named_parameters(session.model, non_swr_names)
    w_sem_before = session.model.sgsc.w_sem.weight.detach().clone()
    w_lex_before = session.model.hippocampus.w_lex_slow.weight.detach().clone()

    cons_path = tmp_path / "export" / "dcpa_chat_consolidated.safetensors"
    ex, sleep_msg = session.handle_command(f"/sleep 50 {cons_path}")
    assert ex is False
    assert cons_path.exists()

    hashes_after_sleep = hash_named_parameters(session.model, non_swr_names)
    assert hashes_before_sleep == hashes_after_sleep, "SWR /sleep mutated parameters outside W_sem & W_lex_slow!"
    assert float((session.model.sgsc.w_sem.weight - w_sem_before).abs().max().item()) > 0.0
    assert float((session.model.hippocampus.w_lex_slow.weight - w_lex_before).abs().max().item()) > 0.0

    # Reload consolidated weights cleanly
    reloaded = load_dcpa_model_from_weights(cons_path, device=dev)
    assert reloaded is not None

    # `/wipe` clears both WM and Hippocampus fast weights back to factory zero
    session.handle_command("/wipe")
    assert float(session.carry.fast_state.u_lex.abs().max().item()) == 0.0
    assert float(session.carry.fast_state.m_c_from_l.abs().max().item()) == 0.0

    print(
        f"\n[Stage 5 Test 3] REPL 2-turn O(1) stateful flow + 0x04 EOT stop OK | "
        f"T1 telemetry: {tel1.format_line()} | {sleep_msg}"
    )
