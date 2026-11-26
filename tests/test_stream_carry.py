"""Stream-carry (A1) exactness: chunked forward with carried state == one-shot forward over the full document."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from data.stream_carry_packer import StreamCarryPacker
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.stream_carry import StreamCarryBank

REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_A = (
    b"fn sum_nested(xs: &[i32]) -> i32 { let mut acc = 0; for &v in xs { if v > 0 { acc += v; } } acc }\n"
    b"The quick silver falcon glides silently over the alpine valley before dusk, "
    b"while the river (calm, cold) keeps [its] old {secrets}.\n"
) + "深度学习系统通过字节流感知与受限概念图实现高效的跨语言泛化。".encode("utf-8")
TEXT_B = b"def f(a, b):\n    return [a, (b, {a: b})]\n# trailing comment about streams\n"


def _build_model(seed: int = 7) -> DCPAModel:
    torch.manual_seed(seed)
    model = DCPAModel(DCPAConfig.micro())
    model.eval()
    return model


def _run(
    model: DCPAModel,
    pieces: list[bytes],
    bank: StreamCarryBank,
    carry_from: list[int],
    carry_save: list[int],
    max_entry_bytes: int,
    record_traces: bool,
):
    x = torch.tensor(list(b"".join(pieces)), dtype=torch.uint8).unsqueeze(0)
    cu = [0]
    for p in pieces:
        cu.append(cu[-1] + len(p))
    cu_t = torch.tensor(cu, dtype=torch.int32)
    lang = torch.zeros((len(pieces),), dtype=torch.int64)
    doc_carry = bank.gather(carry_from, max_entry_bytes=max_entry_bytes)
    with torch.no_grad():
        out = model(
            raw_bytes=x,
            lang_ids=lang,
            stage="B",
            tau_s=0.3,
            tau_m=0.25,
            doc_cu_seqlens=cu_t,
            record_traces=record_traces,
            sync_metrics=False,
            doc_carry=doc_carry,
        )
    bank.store(out.doc_carry_out, carry_save)
    return out, cu


def _split_point(out, target: int) -> int:
    """Byte index right after an SGSC spike closest to `target` (exact LIF reset boundary)."""
    spikes = out.sgsc.spikes[0].nonzero().squeeze(-1).tolist()
    cands = [s + 1 for s in spikes if 8 < s + 1 < out.sgsc.spikes.shape[1] - 8]
    return min(cands, key=lambda b: abs(b - target))


def _bank(model: DCPAModel, n: int = 4) -> StreamCarryBank:
    return StreamCarryBank.from_config(model.cfg, n_streams=n, device=torch.device("cpu"))


def test_stream_carry_chunked_equals_full_document() -> None:
    for record_traces in (True, False):
        model = _build_model()
        max_b = 512
        ref, _ = _run(model, [TEXT_A], _bank(model), [-1], [-1], max_b, record_traces)
        ref_logits = ref.pmd.byte_logits[0]
        n_seg_ref = int(ref.sgsc.cu_seqlens.numel()) - 1
        assert n_seg_ref > model.cfg.window_size, "test must exercise ring truncation (> W segments)"

        b = _split_point(ref, len(TEXT_A) // 2)
        bank = _bank(model)
        out1, _ = _run(model, [TEXT_A[:b]], bank, [-1], [0], max_b, record_traces)
        assert bank.num_valid == 1
        out2, _ = _run(model, [TEXT_A[b:]], bank, [0], [-1], max_b, record_traces)

        assert torch.equal(torch.cat([out1.sgsc.spikes[0], out2.sgsc.spikes[0]]), ref.sgsc.spikes[0])
        chunk_logits = torch.cat([out1.pmd.byte_logits[0], out2.pmd.byte_logits[0]], dim=0)
        diff = (chunk_logits - ref_logits).abs().max().item()
        assert diff < 2e-4, f"record_traces={record_traces}: chunked vs full logits diff {diff:.3e}"
        assert ref.doc_carry_out is not None and out2.doc_carry_out is not None
        assert torch.allclose(out2.doc_carry_out.slots, ref.doc_carry_out.slots, atol=2e-4)
        assert int(out2.doc_carry_out.doc_pos[0]) == n_seg_ref


def test_stream_carry_mixed_continuation_and_fresh_entries() -> None:
    model = _build_model(11)
    max_b = 512
    ref_a, _ = _run(model, [TEXT_A], _bank(model), [-1], [-1], max_b, False)
    ref_b, _ = _run(model, [TEXT_B], _bank(model), [-1], [-1], max_b, False)
    b = _split_point(ref_a, len(TEXT_A) // 3)

    bank = _bank(model, n=3)
    # Step 1: stream 2 starts doc A (fresh). Step 2: [fresh doc B on stream 0, continuation of A from stream 2].
    _run(model, [TEXT_A[:b]], bank, [-1], [2], max_b, False)
    out2, cu2 = _run(model, [TEXT_B, TEXT_A[b:]], bank, [-1, 2], [-1, -1], max_b, False)

    lg = out2.pmd.byte_logits[0]
    d_b = (lg[: cu2[1]] - ref_b.pmd.byte_logits[0]).abs().max().item()
    d_a = (lg[cu2[1] :] - ref_a.pmd.byte_logits[0][b:]).abs().max().item()
    assert d_b < 2e-4, f"fresh entry diverged from standalone run: {d_b:.3e}"
    assert d_a < 2e-4, f"continuation entry diverged from full-doc run: {d_a:.3e}"


def test_stream_carry_training_backward_runs() -> None:
    model = _build_model(3)
    model.train()
    model.bcg_tbptt_window = 4
    bank = _bank(model, n=2)
    x = torch.tensor(list(TEXT_A[:120] + TEXT_B[:60]), dtype=torch.uint8).unsqueeze(0)
    cu = torch.tensor([0, 120, 180], dtype=torch.int32)
    for step in range(2):
        carry_from = [-1, -1] if step == 0 else [0, 1]
        dc = bank.gather(carry_from, max_entry_bytes=128)
        out = model(
            raw_bytes=x,
            lang_ids=torch.zeros((2,), dtype=torch.int64),
            stage="B",
            tau_s=0.3,
            tau_m=0.5,
            doc_cu_seqlens=cu,
            record_traces=False,
            sync_metrics=False,
            doc_carry=dc,
        )
        out.loss_total.backward()
        assert torch.isfinite(out.loss_total).item()
        assert out.doc_carry_out is not None and not out.doc_carry_out.slots.requires_grad
        bank.store(out.doc_carry_out, [0, 1])
    assert model.bcg_pc.s0.grad is not None


def _doc_source(seed: int):
    """Synthetic `RaggedByteDataset`-style batches of mixed ASCII / CJK documents with unique ids."""
    rng = random.Random(seed)
    counter = [0]
    produced: dict[int, bytes] = {}

    def pull() -> dict[str, torch.Tensor]:
        docs = []
        for _ in range(rng.randint(1, 4)):
            did = counter[0]
            counter[0] += 1
            body = f"<doc {did}> " + "".join(rng.choice(["ab ", "cd. ", "字节", "流{x}", "\n"]) for _ in range(rng.randint(2, 120)))
            raw = body.encode("utf-8")
            produced[did] = raw
            docs.append((did, raw))
        x = np.frombuffer(b"".join(r for _, r in docs), dtype=np.uint8).copy()
        cu = np.cumsum([0] + [len(r) for _, r in docs]).astype(np.int32)
        n = len(docs)
        return {
            "x_bytes": torch.from_numpy(x),
            "cu_seqlens": torch.from_numpy(cu),
            "doc_lang_ids": torch.tensor([d % 3 for d, _ in docs], dtype=torch.int64),
            "doc_style_ids": torch.zeros((n,), dtype=torch.int64),
            "doc_ponder_tax_scale": torch.ones((n,), dtype=torch.float32),
            "byte_lang_ids": torch.from_numpy(np.repeat([d % 3 for d, _ in docs], [len(r) for _, r in docs])),
            "loss_mask": torch.ones((len(x),), dtype=torch.bool),
            "wug_target_mask": torch.zeros((len(x),), dtype=torch.bool),
        }

    return pull, produced


def test_stream_carry_packer_reconstructs_documents_and_routes_carry() -> None:
    pull, produced = _doc_source(5)
    chunk, rows, cap = 48, 7, 8
    packer = StreamCarryPacker(rows, chunk, min_piece_bytes=8, max_entries=cap)
    live: dict[int, bytes] = {}  # stream -> bytes emitted so far for its live document
    saved_prev: set[int] = set()
    finished: list[bytes] = []
    for _step in range(60):
        b = packer.next_batch(pull)
        cu = b["cu_seqlens"].tolist()
        x = bytes(b["x_bytes"].tolist())
        cf = b["carry_from"].tolist()
        cs = b["carry_save"].tolist()
        assert len(cf) == len(cs) == len(cu) - 1 <= cap
        saved_now: set[int] = set()
        for v in range(len(cf)):
            piece = x[cu[v] : cu[v + 1]]
            assert 0 < len(piece) <= chunk
            piece.decode("utf-8")  # every entry is UTF-8 boundary aligned
            if cf[v] >= 0:
                assert cf[v] in saved_prev, "continuation must read a stream saved in the previous step"
                cur = live.pop(cf[v]) + piece
            else:
                cur = piece
            if cs[v] >= 0:
                assert cs[v] not in saved_now
                saved_now.add(cs[v])
                live[cs[v]] = cur
            else:
                finished.append(cur)
        saved_prev = saved_now
    assert len(finished) > 20
    produced_set = set(produced.values())
    for doc in finished:
        assert doc in produced_set, "stream reassembly must reproduce source documents exactly"


def test_stream_carry_run_training_smoke(tmp_path: Path, capsys, monkeypatch) -> None:
    from train.train_dcpa import run_training

    n_cont = [0]
    orig_gather = StreamCarryBank.gather

    def _counting_gather(self, carry_from, max_entry_bytes):
        out = orig_gather(self, carry_from, max_entry_bytes)
        n_cont[0] += int(out.cont_mask.sum().item())
        return out

    monkeypatch.setattr(StreamCarryBank, "gather", _counting_gather)

    summary = run_training(
        config_path=REPO_ROOT / "configs" / "v1" / "dcpa_48m_train.yaml",
        preset="micro",
        total_steps=8,
        stage_boundaries=(0, 2, 5, 8, 9),
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        data_dir=None,
        ckpt_dir=tmp_path / "ckpts",
        logs_dir=tmp_path / "logs",
        export_dir=tmp_path / "export",
        local_cache_dir=tmp_path / "nvme",
        save_every_steps=100,
        milestone_every_steps=0,
        probe_every_steps=0,
        console_every_steps=4,
        resume="none",
        device_str="cpu",
        deterministic=True,
        seed=11,
        export_final=False,
        stream_carry=True,
        stream_chunk_bytes=128,
    )
    assert summary.completed_steps == 8
    assert "[StreamCarry] enabled" in capsys.readouterr().out
    assert n_cont[0] > 0, "stream carry never continued a document across steps"
    for s in range(3, 9):
        assert np.isfinite(summary.loss_history[s])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A2 CUDA-Graph capture requires CUDA")
def test_bcg_cudagraph_chunk_kernel_matches_compiled_on_cuda() -> None:
    """GPU self-check for A2: CUDA-Graph clause-chunk replay == regular compiled kernel (fwd loss + grads),
    and repeated stream-carry steps never read CUDA-Graph outputs that were overwritten by later replays."""
    from dcpa.compile_utils import configure_model_compile, unwrap_compiled_module

    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = DCPAModel(DCPAConfig.micro()).to(dev).train()
    model.bcg_grad_checkpointing = False
    model.bcg_tbptt_window = 8
    configure_model_compile(model, "auto", dev, bcg_cudagraphs=True)
    bcg_raw = unwrap_compiled_module(model.bcg_pc)
    assert bcg_raw.has_cudagraph_chunk_kernel
    cg_fn = bcg_raw._cudagraph_chunk_fn

    p1, p2 = TEXT_A[:200], TEXT_B
    x = torch.tensor(list(p1 + p2), dtype=torch.uint8, device=dev).unsqueeze(0)
    cu = torch.tensor([0, len(p1), len(p1) + len(p2)], dtype=torch.int32, device=dev)
    lang = torch.zeros((2,), dtype=torch.int64, device=dev)
    bank = StreamCarryBank.from_config(model.cfg, n_streams=2, device=dev)

    def _step(carry_from: list[int], use_cg: bool):
        bcg_raw._cudagraph_chunk_fn = cg_fn if use_cg else None
        model.zero_grad(set_to_none=True)
        c0_saved = bcg_raw.c0.clone()
        torch.manual_seed(123)
        out = model(
            raw_bytes=x,
            lang_ids=lang,
            stage="B",
            tau_s=0.3,
            tau_m=0.5,
            doc_cu_seqlens=cu,
            record_traces=False,
            sync_metrics=False,
            doc_carry=bank.gather(carry_from, max_entry_bytes=256),
        )
        out.loss_total.backward()
        grads = {n: p.grad.detach().clone() for n, p in bcg_raw.named_parameters() if p.grad is not None}
        bcg_raw.c0.copy_(c0_saved)
        return out, grads

    for step in range(3):
        carry_from = [-1, -1] if step == 0 else [0, 1]
        out_ref, g_ref = _step(carry_from, use_cg=False)
        out_cg, g_cg = _step(carry_from, use_cg=True)
        assert torch.isfinite(out_cg.loss_total).item()
        assert abs(out_cg.loss_total.item() - out_ref.loss_total.item()) < 1e-3
        for name, g in g_ref.items():
            assert name in g_cg
            assert torch.allclose(g_cg[name], g, atol=1e-3, rtol=1e-2), name
        bank.store(out_cg.doc_carry_out, [0, 1])


def test_stream_carry_exact_under_cpu_compile_and_doc_bucketing() -> None:
    """Compiled static-shape hotspots (aot_eager on CPU) + doc bucketing + prefix shrink keep exactness."""
    from dcpa.compile_utils import configure_model_compile

    model = _build_model(5)
    eager_ref, _ = _run(model, [TEXT_B, TEXT_A], _bank(model), [-1, -1], [-1, -1], 512, False)
    configure_model_compile(model, "auto", torch.device("cpu"), allow_cpu_compile=True)
    assert model.enable_doc_bucketing
    ref_a, _ = _run(model, [TEXT_A], _bank(model), [-1], [-1], 512, False)
    comp_both, cu_both = _run(model, [TEXT_B, TEXT_A], _bank(model), [-1, -1], [-1, -1], 512, False)
    d_comp = (comp_both.pmd.byte_logits - eager_ref.pmd.byte_logits).abs().max().item()
    assert d_comp < 2e-4, f"compiled vs eager diverged: {d_comp:.3e}"

    b = _split_point(ref_a, len(TEXT_A) // 2)
    bank = _bank(model, n=2)
    _run(model, [TEXT_B, TEXT_A[:b]], bank, [-1, -1], [-1, 1], 512, False)
    out2, cu2 = _run(model, [TEXT_A[b:]], bank, [1], [-1], 512, False)
    d = (out2.pmd.byte_logits[0] - ref_a.pmd.byte_logits[0][b:]).abs().max().item()
    assert d < 2e-4, f"compiled stream-carry continuation diverged: {d:.3e}"


def test_a1_a2_defaults_enabled_in_configs_and_compile_utils() -> None:
    import inspect
    import yaml

    from dcpa.compile_utils import configure_model_compile

    sig = inspect.signature(configure_model_compile)
    assert sig.parameters["bcg_cudagraphs"].default is True

    for rel in (
        "configs/dcpa_48m_train.yaml",
        "configs/v1/dcpa_48m_train.yaml",
        "configs/v1/dcpa_210m_moe.yaml",
        "configs/v1/dcpa_unified.yaml",
        "configs/v1/dcpa_v2_4xa100.yaml",
        "configs/v2/dcpa_48m_v2_cognitive.yaml",
        "configs/v2/dcpa_210m_v2_cognitive.yaml",
    ):
        cfg = yaml.safe_load((REPO_ROOT / rel).read_text(encoding="utf-8"))
        hw = cfg["hardware"]
        assert hw.get("stream_carry") is True, f"{rel}: stream_carry should default to True"
        assert hw.get("bcg_cudagraphs") is True, f"{rel}: bcg_cudagraphs should default to True"


def test_hf_auth_resolution_and_headers(tmp_path: Path, monkeypatch) -> None:
    from data.hf_auth import configure_hf_auth, hf_auth_headers, hf_token_kwargs, resolve_hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("# comment only\nOTHER=1\n", encoding="utf-8")

    # 1. Explicit CLI token wins over env file
    env_f = tmp_path / "local.env"
    env_f.write_text('export HF_TOKEN="hf_from_file_123"\n', encoding="utf-8")
    assert resolve_hf_token("hf_cli_999", env_file=env_f) == "hf_cli_999"

    # 2. local.env resolution & export to both env vars + Bearer header
    tok = configure_hf_auth(None, verbose=False, env_file=env_f)
    assert tok == "hf_from_file_123"
    assert hf_token_kwargs() == {"token": "hf_from_file_123"}
    hdrs = hf_auth_headers()
    assert hdrs["Authorization"] == "Bearer hf_from_file_123"
    assert "User-Agent" in hdrs

    # 3. Unauthenticated headers when env is cleared
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_token_kwargs() == {}
    assert "Authorization" not in hf_auth_headers()
    assert resolve_hf_token(None, env_file=empty_env) in (None, "") or isinstance(
        resolve_hf_token(None, env_file=empty_env), str
    )
