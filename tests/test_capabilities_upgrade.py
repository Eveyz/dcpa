"""Deterministic Acceptance Test Suite for DCPA Capability Upgrade Patch (Spec 6).

Verifies all 5 mandated capability invariants on deterministic CPU float32:
  1. `test_slot0_thesis_pinning_invariance`:
       T=2048 sequence with `\\x1dS\\noutline\\x1dE\\n` prefix triggering >100 K+1->K merges;
       asserts `slot0_pinned == 1.0` after `\\x1dE\\n`, max abs drift of `S[:, 0, :] == 0.0`,
       and ZERO Dynamo graph recompilations when `bcg_step_kernel` is compiled.
  2. `test_style_lang_orthogonal_compatibility`:
       Bidirectional 16-byte `.idx` compatibility (`<QIBBH` <-> `<QIBBBB`), deterministic non-zero
       PMD distribution shift across `style_id` (1 vs 3) on fixed `(G, lang_id)`, and
       `||lang_half||_2 / ||style_half||_2` ratio error `< 1e-4`.
  3. `test_toolcall_mask_and_oov_aliasing`:
       Inspects 200 samples from `--smoke-quota` `cap_toolcall`: 100% of `\\x12O\\n...` spans
       masked `False`, 100% of `\\x11C\\n...\\x13\\n` calls pass `ast.parse(..., mode='eval')`,
       and >= 25% contain OOV synthetic tool aliases strictly matching top signature <-> call site.
  4. `test_pushdown_stack_multi_frame_reset`:
       Composite stream with unclosed-paren scratchpad + tool call + nested Rust block; verifies
       exact stack zeroing at every control boundary and `< 1e-5` parity vs serial reference.
  5. `test_m3_ast_sandbox_end_to_end_and_security`:
       End-to-end `0x11 -> calc(...) -> 0x13 -> 0x12 -> 0x04` flow in `cli/chat_m3.py`, 100% AST
       whitelist block on `__import__('os').system('echo pwned')`, and 5-hop circuit breaker.
"""

from __future__ import annotations

import ast
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cli.chat_m3 import DCPAChatSession, MAX_TOOL_HOPS_PER_TURN, execute_tool_call_safe
from data.build_cap_toolcall import (
    STANDARD_TOOL_NAMES,
    build_cap_toolcall_corpus,
    validate_tool_call_ast,
)
from data.ragged_dataloader import IDX_DTYPE, IDX_RECORD_FMT, IDX_RECORD_SIZE
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from utils.chat_protocol import (
    SCRATCH_END,
    SCRATCH_START,
    TOOL_CALL_BYTE,
    TOOL_CALL_PREFIX,
    TOOL_OBS_PREFIX,
    TOOL_SUSPEND_BYTE,
    TOOL_SUSPEND_END,
    TURN_ASST_PREFIX,
    TURN_EOT_BYTE,
    TURN_EOT_SUFFIX,
    TURN_USER_BYTE,
    build_sft_byte_loss_weights,
    format_tool_observation_after_suspend,
    verify_mask_alignment,
)
from utils.device import get_deterministic_test_device, seed_everything


# =============================================================================
# Test 1: Slot-0 Thesis Pinning Invariance & Zero Compile Recompilations
# =============================================================================


def test_slot0_thesis_pinning_invariance():
    dev = get_deterministic_test_device()
    seed_everything(202601, device=dev)

    cfg = DCPAConfig.micro()
    cfg.l_max = 8
    model = DCPAModel(cfg).to(device=dev, dtype=torch.float32)
    model.eval()

    # Construct T = 2048 sequence with `\x1dS\noutline\x1dE\n` prefix triggering > 100 K+1 -> K merges
    prefix = SCRATCH_START + b"1.Core thesis -> 2.Evidence -> 3.Synthesis" + SCRATCH_END
    body_unit = (
        b"Systems with bounded working memory preserve global coherence across long spans "
        b"because slot zero remains anchored while remaining slots merge incoming clauses.\n"
    )
    raw = prefix + (body_unit * 24)
    raw = raw[:2048]
    assert len(raw) == 2048
    scratch_end_byte_idx = raw.index(SCRATCH_END) + len(SCRATCH_END) - 1

    x = torch.tensor(list(raw), dtype=torch.uint8, device=dev).unsqueeze(0)  # [1, 2048]
    lang_ids = torch.tensor([0], dtype=torch.long, device=dev)

    with torch.no_grad():
        out_eval = model(raw_bytes=x, lang_ids=lang_ids, stage="C", tau_s=0.25, tau_m=0.25)

    assert out_eval.slot0_pinned_trace is not None and out_eval.s_trace is not None
    cu = out_eval.sgsc.cu_seqlens.cpu().tolist()
    m_total = len(cu) - 1
    # Find the exact segment m_pin containing the closing byte of `\x1dE\n`
    m_pin = next(m for m in range(m_total) if cu[m] <= scratch_end_byte_idx < cu[m + 1])
    merges_after_pin = (m_total - 1) - m_pin
    assert merges_after_pin > 100, f"Expected > 100 merges after \\x1dE\\n, got {merges_after_pin}"

    # Assert slot0_pinned == 0.0 before m_pin and strictly 1.0 from m_pin to end
    if m_pin > 0:
        assert float(out_eval.slot0_pinned_trace[:m_pin].max().item()) == 0.0
    assert torch.equal(
        out_eval.slot0_pinned_trace[m_pin:],
        torch.ones((m_total - m_pin,), dtype=torch.float32, device=dev),
    )

    # Assert Slot-0 vector S[:, 0, :] has STRICTLY 0.0 max absolute drift across all 100+ subsequent merges!
    slot0_ref = out_eval.s_trace[m_pin, 0, :]  # [D]
    slot0_tail = out_eval.s_trace[m_pin:, 0, :]  # [M_tail, D]
    max_drift_eval = float((slot0_tail - slot0_ref.unsqueeze(0)).abs().max().item())
    assert max_drift_eval == 0.0, f"Slot-0 drifted in eval mode! max_drift={max_drift_eval}"

    # Verify ALSO in train() mode with Gumbel-STE + ponder_tax_scale=0.0 vs 1.0 + MoE bal loss
    model.train()
    out_train_zero = model(
        raw_bytes=x,
        lang_ids=lang_ids,
        ponder_tax_scale=torch.tensor([0.0], device=dev),
        stage="C",
        tau_s=0.25,
        tau_m=0.25,
    )
    assert float(out_train_zero.loss_ponder.detach().item()) == 0.0
    assert out_train_zero.loss_moe_bal is not None and float(out_train_zero.loss_moe_bal.detach().item()) >= 0.0
    slot0_ref_tr = out_train_zero.s_trace[m_pin, 0, :]
    max_drift_train = float((out_train_zero.s_trace[m_pin:, 0, :] - slot0_ref_tr.unsqueeze(0)).abs().max().item())
    assert max_drift_train == 0.0, f"Slot-0 drifted in train mode! max_drift={max_drift_train}"

    # Verify ZERO Dynamo graph recompilations when bcg_step_kernel runs compiled across unpinned -> latch -> pinned!
    torch._dynamo.reset()
    compile_calls = {"count": 0}

    def _counting_backend(gm, example_inputs):
        compile_calls["count"] += 1
        return gm.forward

    model.eval()
    model.bcg_pc.set_static_kernel_mode(True)
    compiled_bcg = torch.compile(model.bcg_pc, dynamic=False, backend=_counting_backend)

    B_c, D_c = 2, cfg.d_concept
    st = model.bcg_pc.init_state(B_c, device=dev, dtype=torch.float32)
    tau_t = torch.tensor(0.25, device=dev, dtype=torch.float32)
    mix_t = torch.tensor(0.0, device=dev, dtype=torch.float32)
    am = torch.ones((B_c,), device=dev, dtype=torch.bool)
    ach = torch.zeros((B_c,), device=dev, dtype=torch.float32)
    prec = torch.ones((B_c, D_c), device=dev, dtype=torch.float32)

    pinned_slot0_after_latch = None
    with torch.no_grad():
        for step_i in range(12):
            c_in = torch.randn((B_c, D_c), device=dev)
            trig = torch.tensor([1.0, 0.0], device=dev) if step_i == 2 else torch.zeros((B_c,), device=dev)
            res = compiled_bcg(
                st.slots, st.edges, st.salience, c_in, prec, ach, tau_t, am, mix_t,
                st.slot0_pinned, trig, False, False,
            )
            st_slots, st_edges, st_sal, st_pin = res[6], res[7], res[8], res[15]
            st = type(st)(slots=st_slots, edges=st_edges, salience=st_sal, slot0_pinned=st_pin)
            if step_i == 2:
                pinned_slot0_after_latch = st.slots[0, 0].clone()
            elif step_i > 2:
                assert float((st.slots[0, 0] - pinned_slot0_after_latch).abs().max().item()) == 0.0

    # Exactly 1 graph compilation on step 0, ZERO recompilations across unpinned -> latch -> pinned!
    recompiles = max(0, compile_calls["count"] - 1)
    assert compile_calls["count"] == 1, f"Expected 1 compile (0 recompiles), got {compile_calls['count']}"
    print(
        f"\n[Cap Test 1] merges_after_pin={merges_after_pin} | max_drift_eval={max_drift_eval:.1e} | "
        f"max_drift_train={max_drift_train:.1e} | bcg_step_kernel recompiles={recompiles}"
    )


# =============================================================================
# Test 2: 32D Lang ⊕ 32D Style Orthogonality & 16-Byte `.idx` Compatibility
# =============================================================================


def test_style_lang_orthogonal_compatibility(tmp_path: Path):
    dev = get_deterministic_test_device()
    seed_everything(202602, device=dev)

    # 2A. Bidirectional 16-byte `.idx` binary compatibility (`<QIBBH` -> `<QIBBBB` & new `<QIBBBB`)
    assert IDX_RECORD_SIZE == 16 and struct.calcsize(IDX_RECORD_FMT) == 16
    legacy_path = tmp_path / "legacy.idx"
    # Legacy format packed max_depth=12 as little-endian uint16 -> byte14=12, byte15=0x00!
    legacy_bytes = struct.pack("<QIBBH", 4096, 512, 4, 2, 12) + struct.pack("<QIBBH", 4608, 300, 1, 0, 0)
    legacy_path.write_bytes(legacy_bytes)

    arr_leg = np.memmap(legacy_path, dtype=IDX_DTYPE, mode="r")
    assert len(arr_leg) == 2
    assert int(arr_leg[0]["offset"]) == 4096 and int(arr_leg[0]["length"]) == 512
    assert int(arr_leg[0]["lang_id"]) == 4 and int(arr_leg[0]["max_depth"]) == 12 and int(arr_leg[0]["style_id"]) == 0
    assert int(arr_leg[1]["lang_id"]) == 1 and int(arr_leg[1]["max_depth"]) == 0 and int(arr_leg[1]["style_id"]) == 0
    del arr_leg

    new_path = tmp_path / "styled.idx"
    new_bytes = struct.pack(IDX_RECORD_FMT, 0, 1024, 0, 12, 0, 1) + struct.pack(IDX_RECORD_FMT, 1024, 980, 1, 13, 9, 4)
    new_path.write_bytes(new_bytes)
    arr_new = np.memmap(new_path, dtype=IDX_DTYPE, mode="r")
    assert int(arr_new[0]["style_id"]) == 1 and int(arr_new[0]["max_depth"]) == 0
    assert int(arr_new[1]["style_id"]) == 4 and int(arr_new[1]["max_depth"]) == 9
    del arr_new

    # 2B. Energy balance: ||RMSNorm_32(lang)||_2 / ||RMSNorm_32(style)||_2 ratio error < 1e-4
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=dev, dtype=torch.float32)
    model.eval()

    max_ratio_err = 0.0
    with torch.no_grad():
        for lid in range(5):
            for sid in range(5):
                l_reg = model.pmd.get_lang_vec(
                    torch.tensor([lid], device=dev),
                    torch.tensor([sid], device=dev),
                )
                assert l_reg.shape == (1, 64)
                n_lang = float(l_reg[0, :32].norm(p=2).item())
                n_style = float(l_reg[0, 32:].norm(p=2).item())
                ratio_err = abs((n_lang / max(n_style, 1e-12)) - 1.0)
                max_ratio_err = max(max_ratio_err, ratio_err)
    assert max_ratio_err < 1e-4, f"Lang/Style half-vector L2 norm ratio error {max_ratio_err:.2e} >= 1e-4"

    # 2C. Fixed concept graph G + fixed lang_id=1: switching style_id 1 (memo) vs 3 (narrative)
    # produces deterministic, strictly non-zero PMD output logit divergence!
    g_state = model.bcg_pc.init_state(1, device=dev, dtype=torch.float32)
    g_state.slots.normal_(0.0, 0.5)
    g_state.edges.normal_(0.0, 0.3)
    g_state.salience.copy_(torch.linspace(0.2, 1.8, cfg.k_slots, device=dev).unsqueeze(0))

    def _run_pmd_for_style(sid: int) -> torch.Tensor:
        with torch.no_grad():
            l_reg = model.pmd.get_lang_vec(torch.tensor([1], device=dev), torch.tensor([sid], device=dev))
            q_init = torch.sigmoid(g_state.salience.float())
            a_prev = torch.zeros((1, cfg.k_slots, cfg.d_r), device=dev)
            cq = model.pmd.broca_cq_step(g_state, q_init, q_init, a_prev, l_reg)
            f_out, _ = model.hippocampus.read_expression(cq.z_out, l_reg, fast_state=None, enabled=False)
            ch0 = model.pmd.init_segment_charge(f_out, cq.delta_syn, l_reg)
            logits, _, _, _, _ = model.pmd.step_single_byte(
                prev_byte=torch.tensor([ord("A")], device=dev),
                f_out=f_out,
                delta_syn=cq.delta_syn,
                l_reg=l_reg,
                tau_step=1,
                m_prev=model.pmd.m0.unsqueeze(0),
                xi_prev=torch.zeros((1, 16, 4), device=dev),
                charge_prev=ch0,
            )
            return logits

    logits_s1_a = _run_pmd_for_style(1)
    logits_s1_b = _run_pmd_for_style(1)
    logits_s3 = _run_pmd_for_style(3)

    det_diff = float((logits_s1_a - logits_s1_b).abs().max().item())
    style_diff = float((logits_s1_a - logits_s3).abs().max().item())
    assert det_diff == 0.0, f"PMD style output non-deterministic: {det_diff}"
    assert style_diff > 1e-3, f"Style 1 vs 3 produced negligible PMD shift: {style_diff}"

    print(
        f"\n[Cap Test 2] 16B .idx legacy/new OK | max ||lang||/||style|| ratio err={max_ratio_err:.2e} | "
        f"style(1 vs 3) max logit diff={style_diff:.4f} (det={det_diff:.1e})"
    )


# =============================================================================
# Test 3: Tool-Call Loss Mask Alignment, 100% AST Validity, & >=25% OOV Aliasing
# =============================================================================


def test_toolcall_mask_and_oov_aliasing(tmp_path: Path):
    out_dir = tmp_path / "cap_toolcall_smoke"
    meta = build_cap_toolcall_corpus(out_dir=out_dir, smoke_quota=True, offline=True, seed=202613)
    assert meta["total_bytes"] >= 1_800_000

    bin_mm = np.memmap(out_dir / "cap_toolcall_train.bin", dtype=np.uint8, mode="r")
    msk_mm = np.memmap(out_dir / "cap_toolcall_train.loss_mask.bin", dtype=np.uint8, mode="r")
    idx_mm = np.memmap(out_dir / "cap_toolcall_train.idx", dtype=IDX_DTYPE, mode="r")

    n_sample = min(200, len(idx_mm))
    assert n_sample == 200

    total_obs_spans = 0
    obs_false_spans = 0
    total_calls = 0
    ast_ok_calls = 0
    oov_matched_docs = 0

    header_tool_re = re.compile(r"^tool\s+([A-Za-z_][A-Za-z0-9_]*)\(", re.MULTILINE)

    for i in range(n_sample):
        off = int(idx_mm[i]["offset"])
        ln = int(idx_mm[i]["length"])
        assert int(idx_mm[i]["style_id"]) == 4, f"Expected style_id=4 for toolcall, got {idx_mm[i]['style_id']}"
        raw = bytes(bin_mm[off : off + ln])
        msk = np.array(msk_mm[off : off + ln], dtype=np.uint8, copy=True)

        assert verify_mask_alignment(raw, msk), f"Sample {i} failed verify_mask_alignment"

        # 1. Verify EVERY `\x12O\n...\n\x1fA\n` observation span has loss_mask strictly 100% False
        pos = 0
        while True:
            o_idx = raw.find(TOOL_OBS_PREFIX, pos)
            if o_idx < 0:
                break
            r_idx = raw.find(TURN_ASST_PREFIX, o_idx + len(TOOL_OBS_PREFIX))
            assert r_idx > o_idx, f"Observation without \\x1fA\\n resume in sample {i}"
            span_end = r_idx + len(TURN_ASST_PREFIX)
            total_obs_spans += 1
            assert not msk[o_idx:span_end].any(), f"Observation span masked True in sample {i}!"
            obs_false_spans += 1
            pos = span_end

        # Extract registered tool names from top header (before `\x1eU\n`)
        hdr_end = raw.find(b"\x1eU\n")
        assert hdr_end > 0
        assert not msk[:hdr_end].any(), f"Top tool header was NOT masked False in sample {i}!"
        hdr_text = raw[:hdr_end].decode("utf-8", errors="strict")
        registered_names = set(header_tool_re.findall(hdr_text))
        assert len(registered_names) >= 3, f"Header missing distractors: {registered_names}"

        # 2. Verify EVERY `\x11C\n...\x13\n` call body passes ast.parse(..., mode='eval') & matches header
        doc_has_oov_match = False
        c_pos = 0
        while True:
            c_idx = raw.find(TOOL_CALL_PREFIX, c_pos)
            if c_idx < 0:
                break
            s_idx = raw.find(TOOL_SUSPEND_END, c_idx + len(TOOL_CALL_PREFIX))
            assert s_idx > c_idx, f"Tool call missing \\x13\\n terminator in sample {i}"
            call_str = raw[c_idx + len(TOOL_CALL_PREFIX) : s_idx].decode("utf-8", errors="strict")
            ok, fn_name, depth = validate_tool_call_ast(call_str)
            total_calls += 1
            assert ok and depth <= 16, f"Tool call failed AST validation: {call_str!r}"
            assert fn_name in registered_names, f"Called `{fn_name}` not registered in header {registered_names}!"
            ast_ok_calls += 1
            if fn_name not in STANDARD_TOOL_NAMES:
                doc_has_oov_match = True
            c_pos = s_idx + len(TOOL_SUSPEND_END)

        if doc_has_oov_match:
            oov_matched_docs += 1

        # 3. Verify 2.5x loss weights on 0x11 & 0x13 and 2.0x on 0x04
        x_t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
        m_t = torch.from_numpy(msk.astype(np.bool_))
        w = build_sft_byte_loss_weights(x_t, m_t, eot_weight=2.0, tool_ctrl_weight=2.5)
        if (x_t == TOOL_CALL_BYTE).any():
            assert torch.allclose(w[x_t == TOOL_CALL_BYTE], torch.full_like(w[x_t == TOOL_CALL_BYTE], 2.5))
        if (x_t == TOOL_SUSPEND_BYTE).any():
            assert torch.allclose(w[x_t == TOOL_SUSPEND_BYTE], torch.full_like(w[x_t == TOOL_SUSPEND_BYTE], 2.5))
        assert torch.allclose(w[x_t == TURN_EOT_BYTE], torch.full_like(w[x_t == TURN_EOT_BYTE], 2.0))
        assert float(w[~m_t].abs().max().item()) == 0.0

    oov_ratio = oov_matched_docs / float(n_sample)
    assert total_obs_spans > 0 and obs_false_spans == total_obs_spans
    assert total_calls > 0 and ast_ok_calls == total_calls
    assert oov_ratio >= 0.25, f"OOV alias ratio {oov_ratio:.3f} < 0.25 ({oov_matched_docs}/{n_sample})"

    print(
        f"\n[Cap Test 3] 200 docs inspected | obs_mask_False={obs_false_spans}/{total_obs_spans} (100%) | "
        f"AST_valid={ast_ok_calls}/{total_calls} (100%) | OOV_header_call_match={oov_matched_docs}/{n_sample} ({oov_ratio*100:.1f}%)"
    )


# =============================================================================
# Test 4: Multi-Frame Pushdown Stack Reset & Parallel vs Serial Parity (< 1e-5)
# =============================================================================


def test_pushdown_stack_multi_frame_reset():
    dev = get_deterministic_test_device()
    seed_everything(202604, device=dev)

    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=dev, dtype=torch.float32)
    model.eval()
    # Bias stack action toward push/hold so stack carries clear non-zero energy prior to every boundary
    with torch.no_grad():
        model.pmd.w_act.bias.copy_(torch.tensor([2.5, -2.0, 1.5], device=dev))

    # Composite stream containing ALL 4 control boundaries:
    #   1) Natural language scratchpad with unclosed parenthesis -> `\x1dE\n`
    #   2) Tool call -> `\x13\n` + observation
    #   3) Nested Rust code block -> `\x04\n`
    #   4) Next user turn -> `\x1eU\n`
    composite = (
        SCRATCH_START
        + b"1) plan (unclosed bracket in draft"
        + SCRATCH_END
        + TOOL_CALL_PREFIX
        + b'calc("(1+2)*3")'
        + TOOL_SUSPEND_END
        + TOOL_OBS_PREFIX
        + b"9\n"
        + TURN_ASST_PREFIX
        + b"fn solve(xs: &[i32]) -> i32 { if (xs.len() > 0) { return (xs[0] + 1); } 0 }\n"
        + TURN_EOT_SUFFIX
        + b"\x1eU\nnext turn\n"
    )
    T = len(composite)
    x_raw = torch.tensor(list(composite), dtype=torch.uint8, device=dev).unsqueeze(0)  # [1, T]
    lang_t = torch.tensor([4], dtype=torch.long, device=dev)
    l_reg = model.pmd.get_lang_vec(lang_t)

    # Single whole-sequence segment so parallel decode_teacher_forced and serial step_single_byte share identical drives
    cu_seg = torch.tensor([0, T], dtype=torch.int32, device=dev)
    f_seg = torch.randn((1, cfg.d_s), device=dev)
    d_seg = torch.randn((1, cfg.d_s), device=dev)

    with torch.no_grad():
        par_out = model.pmd.decode_teacher_forced(
            target_bytes=x_raw,
            cu_seqlens=cu_seg,
            f_out_seg=f_seg,
            delta_syn_seg=d_seg,
            l_reg=l_reg,
        )
        xi_par = par_out.stack_xi[0]  # [T, 16, 4]

        # Serial byte-by-byte reference using PMD.step_single_byte with exact Spec 1 reset boundaries
        m_cur = model.pmd.m0.unsqueeze(0).clone()
        xi_cur = torch.zeros((1, 16, 4), device=dev, dtype=torch.float32)
        ch_cur = model.pmd.init_segment_charge(f_seg, d_seg, l_reg)
        xi_serial_list: list[torch.Tensor] = []

        for t in range(T):
            pb = 0 if t == 0 else composite[t - 1]
            is_after_scratch_end = (t >= 3) and (composite[t - 3 : t] == SCRATCH_END)
            _, m_cur, xi_cur, ch_cur, _ = model.pmd.step_single_byte(
                prev_byte=torch.tensor([pb], dtype=torch.long, device=dev),
                f_out=f_seg,
                delta_syn=d_seg,
                l_reg=l_reg,
                tau_step=t + 1,
                m_prev=m_cur,
                xi_prev=xi_cur,
                charge_prev=ch_cur,
                extra_stack_reset=torch.tensor([is_after_scratch_end], dtype=torch.bool, device=dev),
            )
            xi_serial_list.append(xi_cur[0].clone())
        xi_ser = torch.stack(xi_serial_list, dim=0)  # [T, 16, 4]

    max_err = float((xi_par - xi_ser).abs().max().item())
    assert max_err < 1e-5, f"Parallel vs serial pushdown stack mismatch: {max_err:.3e}"

    # Verify exact reset to 0.0 on rows 1..15 at EVERY Spec 1 boundary (after `\x1dE\n`, `0x13`, `0x04`, `0x1E`)!
    reset_ticks = {
        "after_scratch_end": composite.index(SCRATCH_END) + len(SCRATCH_END),
        "after_0x13": composite.index(bytes([TOOL_SUSPEND_BYTE])) + 1,
        "after_0x04": composite.index(bytes([TURN_EOT_BYTE])) + 1,
        "after_0x1E": composite.index(bytes([TURN_USER_BYTE])) + 1,
    }
    for label, tick in reset_ticks.items():
        # Confirm stack carried non-zero deep rows BEFORE the reset (non-vacuous check!)
        if label != "after_0x1E":
            assert float(xi_par[tick - 1, 1:].abs().max().item()) > 1e-4, f"Stack was trivially zero before {label}"
        tail_max = float(xi_par[tick, 1:].abs().max().item())
        assert tail_max == 0.0, f"Stack did NOT zero rows 1..15 at {label} (tick={tick}): max={tail_max}"

    print(
        f"\n[Cap Test 4] Multi-frame stack resets verified at {list(reset_ticks.keys())} (rows[1:]=0.0) | "
        f"parallel vs serial max_err={max_err:.2e}"
    )


# =============================================================================
# Test 5: M3 Pro AST Whitelist Sandbox End-to-End, Security Block & 5-Hop Fuse
# =============================================================================


def test_m3_ast_sandbox_end_to_end_and_security(tmp_path: Path):
    dev = get_deterministic_test_device()
    seed_everything(202605, device=dev)

    # 5A. Static AST audit of cli/chat_m3.py: STRICTLY ZERO bare `eval()` or `exec()` calls (Red Line #4)!
    chat_src = (REPO_ROOT / "cli" / "chat_m3.py").read_text(encoding="utf-8")
    chat_ast = ast.parse(chat_src, filename="cli/chat_m3.py")
    for node in ast.walk(chat_ast):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec"), (
                f"Red Line #4 VIOLATION: forbidden `{node.func.id}()` call found in cli/chat_m3.py!"
            )

    # 5B. Initialize DCPAChatSession -> verify 5 stdlib tools preloaded & M_tools_base snapshot
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device=dev, dtype=torch.float32)
    session = DCPAChatSession(model, device=dev, temp=0.3, top_p=0.9, max_new_bytes=256, workspace_root=tmp_path)
    assert len(session.tool_signatures) == 5
    assert set(session.tool_registry.keys()) == {"calc", "now_hk", "list_files", "read_file_span", "word_char_count"}
    assert float(session.m_tools_base.u_lex.sum().item()) > 0.0

    # Verify new slash commands `/style`, `/think`, `/tools`, `/reset`
    ex, msg = session.handle_command("/style memo")
    assert not ex and session.style_id == 1
    ex, msg = session.handle_command("/think hide")
    assert not ex and session.think_mode == "hide"
    ex, tools_msg = session.handle_command("/tools")
    assert not ex and "calc(" in tools_msg and "now_hk(" in tools_msg

    # 5C. End-to-end 0x11 -> calc("(12 + 8) * 3") -> 0x13 -> 0x12 injection -> 0x04 halt
    # Also verify temp automatically drops to 0.0 inside 0x11..0x13!
    script_1 = list(TOOL_CALL_PREFIX + b'calc("(12 + 8) * 3")' + bytes([TOOL_SUSPEND_BYTE]) + b"Result is 60." + bytes([TURN_EOT_BYTE]))
    seen_temps_in_call: list[float] = []
    idx_1 = {"k": 0}
    in_call_flag = {"on": False}

    def _sampler_calc(_logits, _step, eff_temp):
        b = script_1[idx_1["k"]]
        idx_1["k"] += 1
        if in_call_flag["on"]:
            seen_temps_in_call.append(float(eff_temp))
        if b == TOOL_CALL_BYTE:
            in_call_flag["on"] = True
        elif b == TOOL_SUSPEND_BYTE:
            in_call_flag["on"] = False
        return b

    reply_1, tel_1 = session.chat_turn("Compute (12 + 8) * 3", byte_sampler=_sampler_calc, max_new_bytes=128)
    assert tel_1.stopped_on_eot is True
    assert tel_1.tool_calls_executed == 1
    assert tel_1.tool_events[0]["ok"] is True
    assert tel_1.tool_events[0]["result"] == "60.0"
    assert tel_1.tool_observations[0] == format_tool_observation_after_suspend("60.0")
    assert tel_1.tool_observations[0] == b"\n\x12O\n60.0\n\x1fA\n"
    assert reply_1.strip() == "Result is 60."
    assert all(t == 0.0 for t in seen_temps_in_call), f"Expected temp=0.0 inside 0x11..0x13, got {seen_temps_in_call}"

    # 5D. Malicious payload blocking: 100% rejected by AST whitelist (ZERO execution, controlled Error string)
    malicious_payloads = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "calc(\"__import__('os').system('echo pwned')\")",
        "__builtins__['eval']('1+1')",
        "(lambda x: x)(1)",
        "calc(expr=__import__('os').getcwd())",
    ]
    blocked_cnt = 0
    for payload in malicious_payloads:
        ok, res = execute_tool_call_safe(payload, session.tool_registry)
        assert ok is False and res.startswith("Error:"), f"Malicious payload NOT blocked: {payload} -> {ok}, {res}"
        blocked_cnt += 1

    # Also test malicious payload end-to-end inside chat_turn (`0x11 -> malicious -> 0x13 -> 0x04`)
    script_mal = list(
        TOOL_CALL_PREFIX
        + b"__import__('os').system('echo pwned')"
        + bytes([TOOL_SUSPEND_BYTE])
        + b"Blocked safely."
        + bytes([TURN_EOT_BYTE])
    )
    idx_m = {"k": 0}
    _, tel_mal = session.chat_turn(
        "Run shell command",
        byte_sampler=lambda _l, _s, _t: script_mal[idx_m.__setitem__("k", idx_m["k"] + 1) or idx_m["k"] - 1],
        max_new_bytes=128,
    )
    assert tel_mal.stopped_on_eot is True
    assert tel_mal.tool_events[0]["ok"] is False
    assert tel_mal.tool_events[0]["result"].startswith("Error: SecurityPolicy:")

    # 5E. 5-hop infinite loop circuit breaker: attempt 8 consecutive calls -> trips cleanly at 5!
    single_call = list(TOOL_CALL_PREFIX + b'calc("1+1")' + bytes([TOOL_SUSPEND_BYTE]))
    script_loop = single_call * 8 + [TURN_EOT_BYTE]
    idx_l = {"k": 0}
    _, tel_loop = session.chat_turn(
        "Loop forever",
        byte_sampler=lambda _l, _s, _t: script_loop[idx_l.__setitem__("k", idx_l["k"] + 1) or idx_l["k"] - 1],
        max_new_bytes=512,
    )
    assert tel_loop.circuit_breaker_tripped is True
    assert tel_loop.tool_calls_executed == MAX_TOOL_HOPS_PER_TURN == 5
    assert "CircuitBreaker" in tel_loop.tool_events[-1]["result"]

    print(
        f"\n[Cap Test 5] calc E2E=60.0 (exact \\n\\x12O\\n60.0\\n\\x1fA\\n) | "
        f"malicious blocked={blocked_cnt}/{len(malicious_payloads)} (100%) | "
        f"5-hop circuit breaker tripped at hop={tel_loop.tool_calls_executed}"
    )
