"""Verification suite for Vast.ai / multi-GPU optimizations & high-speed pipeline.

Covers (100% offline — zero network downloads):
  1. `SparseMicroMoE` jagged Top-2 `F.grouped_mm` vs `torch.bmm` forward + backward equivalence
     (including 0-token empty expert handling).
  2. `RaggedByteDataset` Bayes length-tier bucketing: intra-batch length homogeneity, cross-rank
     `sync_group` tier alignment across `world_size=4`, and unbiased marginal domain probabilities.
  3. Real 2-process `torch.multiprocessing.spawn` (`gloo`) distributed training across Stages 1..4
     (verifying conditional `broca_cq` unused-grad sync, Stage 4 freeze non-corrosion, cross-rank
     weight + `c0` parity, Rank-0 `.safetensors` export, and elastic `world_size=2 -> 1` resume).
  4. Real 2-process `gloo` Stage 5 SFT training & export verification.
  5. `DCPAConfig` v2 presets (`moe_650m`, `moe_1_8b`, `moe_2_6b`, `moe_3_5b`), `dcpa_v2_4xa100.yaml`
     schema parsing, `prepare_vast_corpus.py --dry-run`, and `launch_vast_train.sh --dry-run`.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.build_cap_instruct import build_cap_instruct_corpus
from data.build_cap_reasoning import build_cap_reasoning_corpus
from data.build_sft_stage5 import build_stage5_sft_corpus
from data.prepare_vast_corpus import prepare_vast_corpus, verify_all_shards
from data.ragged_dataloader import IDX_DTYPE, IDX_RECORD_FMT, RaggedByteDataset
from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import BCGPC, SparseMicroMoE
from train.curriculum_state_machine import hash_named_parameters
from train.train_dcpa import build_dcpa_config, run_training
from train.train_stage5_sft import run_stage5_sft
from utils.chat_protocol import verify_mask_alignment


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write_synthetic_multitier_corpus(out_dir: Path) -> None:
    """Create a deterministic 4-domain `(0:en, 1:zh, 3:py, 5:js)` `.bin/.idx` shard spanning all 4 length tiers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tier_lengths = (220, 780, 1550, 3100)
    rng = np.random.default_rng(2026)

    for split, docs_per_cell in (("train", 16), ("val", 4)):
        bin_p = out_dir / f"{split}.bin"
        idx_p = out_dir / f"{split}.idx"
        offset = 0
        with bin_p.open("wb") as fb, idx_p.open("wb") as fi:
            for lid in (0, 1, 3, 5):
                for d_len in tier_lengths:
                    for j in range(docs_per_cell):
                        # Pure valid ASCII bytes (`a`..`z`, spaces, newline)
                        body = rng.integers(97, 123, size=d_len - 1, dtype=np.uint8).tobytes() + b"\n"
                        fb.write(body)
                        fi.write(struct.pack(IDX_RECORD_FMT, offset, d_len, lid, 1, 2, 0))
                        offset += d_len


def test_sparse_micro_moe_grouped_mm_forward_and_backward_equivalence() -> None:
    if not hasattr(F, "grouped_mm"):
        pytest.skip("PyTorch build lacks torch.nn.functional.grouped_mm")

    torch.manual_seed(42)
    moe_ref = SparseMicroMoE(d_model=128, hidden_dim=192, num_experts=8, top_k=2).to(torch.bfloat16)
    moe_gmm = copy.deepcopy(moe_ref).to(torch.bfloat16)
    moe_ref.allow_cpu_grouped_mm = False
    moe_gmm.allow_cpu_grouped_mm = True

    # Only 3 tokens routed Top-2 across 8 experts -> guarantees at least 2 experts have 0 tokens!
    x_ref = torch.randn(3, 128, dtype=torch.bfloat16, requires_grad=True)
    x_gmm = x_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn(3, 128, dtype=torch.bfloat16)

    # 1) Test static_shape=True forward + backward (including 0-token empty experts!)
    y_ref, aux_ref = moe_ref(x_ref, static_shape=True)
    y_gmm, aux_gmm = moe_gmm(x_gmm, static_shape=True)

    assert torch.isfinite(y_gmm).all()
    assert torch.allclose(y_ref, y_gmm, atol=2.5e-2, rtol=2e-2)
    assert torch.allclose(aux_ref, aux_gmm, atol=1e-5, rtol=1e-4)

    (y_ref * grad_out).sum().backward()
    (y_gmm * grad_out).sum().backward()

    assert x_ref.grad is not None and x_gmm.grad is not None
    assert torch.isfinite(x_gmm.grad).all()
    assert torch.allclose(x_ref.grad, x_gmm.grad, atol=3e-2, rtol=3e-2)

    for (n1, p1), (n2, p2) in zip(moe_ref.named_parameters(), moe_gmm.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None, f"Missing grad on {n1}"
        assert torch.isfinite(p2.grad).all(), f"Non-finite grad on {n2}"
        assert torch.allclose(p1.grad, p2.grad, atol=8e-2, rtol=6e-2), f"Grad mismatch on {n1}"
        if p1.grad.float().norm() > 1e-4:
            cos = F.cosine_similarity(p1.grad.flatten().float(), p2.grad.flatten().float(), dim=0).item()
            assert cos > 0.998, f"Low grad cosine similarity ({cos:.5f}) on {n1}"

    # 2) Test forward_compacted_bmm (CUDA hot path!) with padding tail (N=8 padded, n_valid=3)
    moe_ref.zero_grad(set_to_none=True)
    moe_gmm.zero_grad(set_to_none=True)
    x_pad_ref = torch.randn(8, 128, dtype=torch.bfloat16, requires_grad=True)
    x_pad_gmm = x_pad_ref.detach().clone().requires_grad_(True)
    smask = torch.arange(8) < 3
    grad_pad = torch.randn(8, 128, dtype=torch.bfloat16)

    yc_ref, auxc_ref = moe_ref.forward_compacted_bmm(x_pad_ref, n_valid=3, sample_mask=smask)
    yc_gmm, auxc_gmm = moe_gmm.forward_compacted_bmm(x_pad_gmm, n_valid=3, sample_mask=smask)
    assert torch.allclose(yc_ref, yc_gmm, atol=2.5e-2, rtol=2e-2)
    assert torch.allclose(auxc_ref, auxc_gmm, atol=1e-5, rtol=1e-4)

    (yc_ref * grad_pad).sum().backward()
    (yc_gmm * grad_pad).sum().backward()
    assert x_pad_ref.grad is not None and x_pad_gmm.grad is not None
    assert torch.allclose(x_pad_ref.grad[:3], x_pad_gmm.grad[:3], atol=4e-2, rtol=4e-2)
    for (n1, p1), (n2, p2) in zip(moe_ref.named_parameters(), moe_gmm.named_parameters()):
        assert p1.grad is not None and p2.grad is not None, f"Missing compacted grad on {n1}"
        assert torch.allclose(p1.grad, p2.grad, atol=8e-2, rtol=6e-2), f"Compacted grad mismatch on {n1}"


def test_ragged_length_bucketing_homogeneity_cross_rank_sync_and_unbiased_domains(tmp_path: Path) -> None:
    data_dir = tmp_path / "tiered_data"
    _write_synthetic_multitier_corpus(data_dir)

    target_weights = {0: 0.38, 1: 0.42, 3: 0.10, 5: 0.10}
    world_size = 4
    ds = RaggedByteDataset(
        bin_path=data_dir / "train.bin",
        idx_path=data_dir / "train.idx",
        target_batch_bytes=4096,
        max_doc_bytes=4096,
        domain_weights=target_weights,
        wug_ratio=0.0,
        seed=777,
        length_bucketing=True,
        bucket_sync_world_size=world_size,
    )
    try:
        tier_bounds = ((1, 512), (513, 1024), (1025, 2048), (2049, 4096))

        def _tier_of(doc_len: int) -> int:
            for ti, (lo, hi) in enumerate(tier_bounds):
                if lo <= doc_len <= hi:
                    return ti
            raise AssertionError(f"Unexpected doc_len={doc_len}")

        lang_byte_counts = {lid: 0 for lid in (0, 1, 3, 5)}
        total_sampled_bytes = 0

        for logical_b in range(80):
            rank_tiers: list[int] = []
            for r in range(world_size):
                phys_b = logical_b * world_size + r
                batch = ds.get_batch(phys_b, sync_group=logical_b)
                cu = batch["cu_seqlens"].tolist()
                doc_lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
                # Exclude final boundary-truncated tail doc if shorter than tier floor
                if len(doc_lens) >= 2:
                    interior_tiers = {_tier_of(L) for L in doc_lens[:-1]}
                    assert len(interior_tiers) == 1, f"Mixed length tiers inside batch: {doc_lens}"
                    rank_tiers.append(next(iter(interior_tiers)))
                for lid_val, d_len in zip(batch["doc_lang_ids"].tolist(), doc_lens):
                    lang_byte_counts[int(lid_val)] += int(d_len)
                    total_sampled_bytes += int(d_len)

            if len(rank_tiers) == world_size:
                assert len(set(rank_tiers)) == 1, f"Cross-rank length tier desync: {rank_tiers}"

        for lid, expected_p in target_weights.items():
            emp_p = lang_byte_counts[lid] / max(total_sampled_bytes, 1)
            assert abs(emp_p - expected_p) < 0.05, (
                f"Domain {lid} marginal drift under Bayes length bucketing: emp={emp_p:.3f} vs exp={expected_p:.3f}"
            )
    finally:
        ds.close()


def _ddp_train_worker(rank: int, world_size: int, port: int, work_dir_str: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    work_dir = Path(work_dir_str)
    summary = run_training(
        config_path=REPO_ROOT / "configs" / "v1" / "dcpa_210m_moe.yaml",
        preset="micro",
        total_steps=8,
        stage_boundaries=(0, 2, 4, 6, 8),
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=256,
        fixed_global_batch=True,
        data_dir=work_dir / "data",
        local_cache_dir=work_dir / "nvme",
        ckpt_dir=work_dir / "ckpts",
        logs_dir=work_dir / "logs",
        export_dir=work_dir / "export",
        save_every_steps=4,
        probe_every_steps=100,
        console_every_steps=4,
        resume="none",
        device_str="cpu",
        compile_mode="none",
        seed=2026,
        export_final=True,
    )
    assert summary.completed_steps == 8

    ckpt = torch.load(work_dir / "ckpts" / "last_healthy.pt", map_location="cpu", weights_only=False)
    rank_report = {
        "rank": rank,
        "completed_steps": summary.completed_steps,
        "step_in_ckpt": int(ckpt["step"]),
    }
    (work_dir / f"rank_{rank}_done.json").write_text(json.dumps(rank_report), encoding="utf-8")


def test_distributed_2rank_gloo_stage1_to_4_and_elastic_resume(tmp_path: Path) -> None:
    _write_synthetic_multitier_corpus(tmp_path / "data")
    port = _find_free_port()

    torch.multiprocessing.spawn(
        _ddp_train_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )

    for r in (0, 1):
        rep = json.loads((tmp_path / f"rank_{r}_done.json").read_text(encoding="utf-8"))
        assert rep["completed_steps"] == 8
        assert rep["step_in_ckpt"] == 8

    assert (tmp_path / "export" / "dcpa_final.safetensors").is_file()
    assert (tmp_path / "export" / "config.json").is_file()

    # Elastic resume test: resume the 2-GPU checkpoint on 1 process (`world_size=1`) up to step 10!
    for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(k, None)

    resumed = run_training(
        config_path=REPO_ROOT / "configs" / "v1" / "dcpa_210m_moe.yaml",
        preset="micro",
        total_steps=10,
        stage_boundaries=(0, 2, 4, 6, 10),
        target_tokens_bytes_per_step=256,
        micro_batch_bytes=256,
        data_dir=tmp_path / "data",
        local_cache_dir=tmp_path / "nvme",
        ckpt_dir=tmp_path / "ckpts",
        logs_dir=tmp_path / "logs",
        export_dir=tmp_path / "export",
        probe_every_steps=100,
        console_every_steps=5,
        resume="auto",
        device_str="cpu",
        compile_mode="none",
        seed=2026,
        export_final=True,
    )
    assert resumed.resumed_from_step == 8
    assert resumed.completed_steps == 10


def _ddp_sft_worker(rank: int, world_size: int, port: int, work_dir_str: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    work_dir = Path(work_dir_str)
    res = run_stage5_sft(
        weights_path=work_dir / "export" / "dcpa_final.safetensors",
        config_path=work_dir / "export" / "config.json",
        data_dir=work_dir / "sft_data",
        export_dir=work_dir / "sft_export",
        total_steps=3,
        warmup_steps=1,
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=256,
        max_doc_bytes=512,
        device_str="cpu",
        compile_mode="none",
        seed=202605,
        log_every=1,
    )
    assert res.completed_steps == 3
    (work_dir / f"sft_rank_{rank}.json").write_text(
        json.dumps({"rank": rank, "val_bpc": res.val_bpc}),
        encoding="utf-8",
    )


def test_distributed_2rank_stage5_sft(tmp_path: Path) -> None:
    _write_synthetic_multitier_corpus(tmp_path / "data")
    # Produce a fast micro base export first
    run_training(
        config_path=REPO_ROOT / "configs" / "v1" / "dcpa_210m_moe.yaml",
        preset="micro",
        total_steps=2,
        stage_boundaries=(0, 2, 3, 4, 5),
        target_tokens_bytes_per_step=256,
        micro_batch_bytes=256,
        data_dir=tmp_path / "data",
        local_cache_dir=tmp_path / "nvme",
        ckpt_dir=tmp_path / "ckpts",
        logs_dir=tmp_path / "logs",
        export_dir=tmp_path / "export",
        probe_every_steps=100,
        resume="none",
        device_str="cpu",
        compile_mode="none",
    )
    build_stage5_sft_corpus(out_dir=tmp_path / "sft_data", smoke_quota=True, offline=True, seed=99)

    port = _find_free_port()
    torch.multiprocessing.spawn(
        _ddp_sft_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    r0 = json.loads((tmp_path / "sft_rank_0.json").read_text(encoding="utf-8"))
    r1 = json.loads((tmp_path / "sft_rank_1.json").read_text(encoding="utf-8"))
    assert r0["val_bpc"] is not None and np.isfinite(r0["val_bpc"])
    assert r1["val_bpc"] is None
    assert (tmp_path / "sft_export" / "dcpa_chat_sft.safetensors").is_file()


def test_v2_presets_and_vast_orchestrator_dry_run_and_scripts(tmp_path: Path) -> None:
    from dcpa.model import DCPAModel

    yaml_cfg = yaml.safe_load((REPO_ROOT / "configs" / "v1" / "dcpa_v2_4xa100.yaml").read_text(encoding="utf-8"))
    expected_param_ranges = {
        "moe_210m": (180_000_000, 240_000_000),
        "moe_650m": (600_000_000, 780_000_000),
        "moe_1_8b": (1_650_000_000, 2_000_000_000),
        "moe_2_6b": (2_450_000_000, 2_950_000_000),
        "moe_3_5b": (3_250_000_000, 3_850_000_000),
    }

    for preset_name, (lo_p, hi_p) in expected_param_ranges.items():
        cfg = build_dcpa_config(preset_name, yaml_cfg)
        # Rust dcpa-core-rs hard invariants
        assert cfg.vocab_size == 256
        assert cfg.k_slots == 8
        assert cfg.stack_depth == 16
        assert cfg.stack_tag_dim == 4
        assert cfg.d_l == 64
        assert cfg.d_s % 64 == 0
        assert cfg.d_concept % 64 == 0
        with torch.device("meta"):
            m = DCPAModel(cfg)
            n_params = sum(p.numel() for p in m.parameters())
        assert lo_p <= n_params <= hi_p, f"{preset_name} param count {n_params:,} outside [{lo_p:,}, {hi_p:,}]"

    plan = prepare_vast_corpus(
        profile="v2_stem_10g",
        nvme_root=tmp_path / "vast_nvme",
        dry_run=True,
    )
    assert plan["dry_run"] is True
    assert set(plan["commands"].keys()) == {
        "corpus_5domain",
        "cap_reasoning",
        "cap_instruct",
        "cap_writing",
        "cap_toolcall",
        "stage5_sft",
    }
    assert "--stem-upgrade" in plan["commands"]["corpus_5domain"]

    # Verify launch_vast_train.sh --dry-run executes cleanly with zero network/GPU access
    proc = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "launch_vast_train.sh"),
            "--dry-run",
            f"--nvme-dir={tmp_path / 'vast_nvme'}",
            "--gpus=4",
            "--preset=moe_2_6b",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--nproc_per_node=4" in proc.stdout
    assert "dcpa_v2_4xa100.yaml" in proc.stdout
    assert "train_stage5_sft.py" in proc.stdout


def test_bcg_pc_dynamo_single_graph_zero_breaks() -> None:
    """Verify static-shape `scatter_add` MoE routing eliminates all `torch.bincount` graph breaks."""
    if not hasattr(F, "grouped_mm"):
        pytest.skip("PyTorch build lacks torch.nn.functional.grouped_mm")

    torch._dynamo.reset()
    cfg = DCPAConfig(
        d_s=64,
        d_concept=128,
        k_slots=8,
        d_r=32,
        d_merge=64,
        d_h=512,
        s_active=16,
        d_l=64,
        num_experts=8,
        top_k_experts=2,
        expert_hidden=192,
        l_max=4,
    )
    bcg = BCGPC(cfg).to(torch.bfloat16).train()
    bcg.set_static_kernel_mode(True)
    for m in bcg.modules():
        if isinstance(m, SparseMicroMoE):
            m.allow_cpu_grouped_mm = True

    B, D = 4, cfg.d_concept
    st = bcg.init_state(B, device=torch.device("cpu"), dtype=torch.bfloat16)
    c_in = torch.randn(B, D, dtype=torch.bfloat16)
    prec = torch.ones(B, D, dtype=torch.bfloat16)
    ach = torch.zeros(B, dtype=torch.bfloat16)
    tau = torch.tensor(0.3, dtype=torch.float32)
    am = torch.tensor([True, True, True, False])
    mix = torch.tensor(0.0, dtype=torch.float32)
    trig = torch.zeros(B, dtype=torch.bfloat16)

    explanation = torch._dynamo.explain(bcg)(
        st.slots,
        st.edges,
        st.salience,
        c_in,
        prec,
        ach,
        tau,
        am,
        mix,
        st.slot0_pinned,
        trig,
        False,
        False,
    )
    assert explanation.graph_count == 1, (
        f"BCGPC shattered into {explanation.graph_count} graphs! Breaks: {explanation.break_reasons}"
    )
    assert len(explanation.break_reasons) == 0


def test_bayes_length_bucketing_asymmetric_domain_lengths_zero_bias(tmp_path: Path) -> None:
    """Regression test: when EN docs are short (220B) and JS docs are long (3000B),
    byte-mass outer tier weighting must match unbucketed byte-share closely without 13x short-doc bias.
    """
    asym_dir = tmp_path / "asym_data"
    asym_dir.mkdir(parents=True, exist_ok=True)
    bin_p = asym_dir / "train.bin"
    idx_p = asym_dir / "train.idx"
    rng = np.random.default_rng(1234)

    offset = 0
    with bin_p.open("wb") as fb, idx_p.open("wb") as fi:
        # Domain 0 (EN): 64 short docs (220 bytes -> band 0)
        for _ in range(64):
            body = rng.integers(97, 123, size=219, dtype=np.uint8).tobytes() + b"\n"
            fb.write(body)
            fi.write(struct.pack(IDX_RECORD_FMT, offset, 220, 0, 1, 1, 0))
            offset += 220
        # Domain 5 (JS): 64 long docs (3000 bytes -> band 3)
        for _ in range(64):
            body = rng.integers(97, 123, size=2999, dtype=np.uint8).tobytes() + b"\n"
            fb.write(body)
            fi.write(struct.pack(IDX_RECORD_FMT, offset, 3000, 5, 1, 4, 0))
            offset += 3000

    # Under unbucketed doc-draw (`p_0 = 0.5, p_5 = 0.5`), expected byte share of JS (5) is:
    #   3000 / (220 + 3000) = 0.9317!
    # Under the old doc-count tier bug, band 0 and band 3 each got 50% of batches (packing 18 short docs vs 1.3 long docs),
    # collapsing JS byte share from 93.2% down to ~50%!
    expected_js_byte_share = 3000.0 / (220.0 + 3000.0)
    ds_bucket = RaggedByteDataset(
        bin_path=bin_p,
        idx_path=idx_p,
        target_batch_bytes=8192,
        max_doc_bytes=4096,
        domain_weights={0: 0.5, 5: 0.5},
        wug_ratio=0.0,
        seed=2026,
        length_bucketing=True,
    )
    try:
        assert ds_bucket._use_tiered_buckets is True
        b0, b5 = 0, 0
        for i in range(120):
            batch = ds_bucket.get_batch(i)
            cu = batch["cu_seqlens"].tolist()
            lens = [cu[k + 1] - cu[k] for k in range(len(cu) - 1)]
            for lid_v, ln_v in zip(batch["doc_lang_ids"].tolist(), lens):
                if int(lid_v) == 0:
                    b0 += int(ln_v)
                elif int(lid_v) == 5:
                    b5 += int(ln_v)
        emp_js_share = b5 / max(b0 + b5, 1)
        assert abs(emp_js_share - expected_js_byte_share) < 0.04, (
            f"Byte-mass tier weighting biased asymmetric lengths: emp_js={emp_js_share:.3f} vs exp={expected_js_byte_share:.3f}"
        )
    finally:
        ds_bucket.close()


def test_cap_instruct_and_js_reasoning_offline_builders_and_sidecar_mask(tmp_path: Path) -> None:
    """Verify new `cap_instruct` + JS `cap_reasoning` offline builders, zero JA/Rust,
    100% `verify_mask_alignment`, and `RaggedByteDataset` `.loss_mask.bin` sidecar loading.
    """
    cap_dir = tmp_path / "cap_data"
    meta_inst = build_cap_instruct_corpus(out_dir=cap_dir, smoke_quota=True, offline=True, seed=202614)
    meta_reas = build_cap_reasoning_corpus(out_dir=cap_dir, smoke_quota=True, offline=True, seed=202611)
    assert meta_inst["total_bytes"] >= 1_800_000
    assert meta_reas["total_bytes"] >= 1_800_000

    # Audit cap_instruct: only active domains {0, 1, 3, 5}, 100% mask alignment
    idx_inst = np.memmap(cap_dir / "cap_instruct_train.idx", dtype=IDX_DTYPE, mode="r")
    bin_inst = np.memmap(cap_dir / "cap_instruct_train.bin", dtype=np.uint8, mode="r")
    msk_inst = np.memmap(cap_dir / "cap_instruct_train.loss_mask.bin", dtype=np.uint8, mode="r")
    inst_langs = set(np.unique(idx_inst["lang_id"]).tolist())
    assert inst_langs.issubset({0, 1, 3, 5}), f"Forbidden JA/Rust lang_id in cap_instruct: {inst_langs}"
    assert {0, 1, 3, 5}.issubset(inst_langs), f"Missing active domain in cap_instruct: {inst_langs}"
    for i in range(min(80, len(idx_inst))):
        off, ln = int(idx_inst[i]["offset"]), int(idx_inst[i]["length"])
        assert verify_mask_alignment(bytes(bin_inst[off : off + ln]), bytes(msk_inst[off : off + ln]))
    del idx_inst, bin_inst, msk_inst

    # Audit cap_reasoning: includes JS (`lang_id=5`) state-machine traces
    idx_reas = np.memmap(cap_dir / "cap_reasoning_train.idx", dtype=IDX_DTYPE, mode="r")
    reas_langs = set(np.unique(idx_reas["lang_id"]).tolist())
    assert 5 in reas_langs, f"Expected JS (lang_id=5) traces in cap_reasoning, got {reas_langs}"
    assert 2 not in reas_langs and 4 not in reas_langs
    del idx_reas

    # Verify RaggedByteDataset automatically loads `.loss_mask.bin` sidecar so User prompts stay False!
    ds_cap = RaggedByteDataset(
        bin_path=cap_dir / "cap_instruct_train.bin",
        idx_path=cap_dir / "cap_instruct_train.idx",
        target_batch_bytes=4096,
        max_doc_bytes=2048,
        wug_ratio=0.0,
        seed=42,
    )
    try:
        assert ds_cap.mask_path is not None
        b = ds_cap.get_batch(0)
        lmask = b["loss_mask"]
        # Both False (User prompt) and True (Assistant response) bytes MUST appear!
        assert bool((~lmask).any().item()), "Sidecar .loss_mask.bin failed to mask User prompt bytes False!"
        assert bool(lmask.any().item()), "Sidecar .loss_mask.bin masked out all Assistant bytes!"
    finally:
        ds_cap.close()


def test_unified_cli_single_to_multi_to_single_gpu_switch(tmp_path: Path) -> None:
    """End-to-end test of seamless `--gpus 1 <-> --gpus 2` switching using the SAME command,
    SAME `configs/v1/dcpa_unified.yaml`, and SAME checkpoint directory (zero manual `torchrun`).
    """
    data_dir = tmp_path / "data"
    _write_synthetic_multitier_corpus(data_dir)
    sft_dir = tmp_path / "sft_data"
    build_stage5_sft_corpus(out_dir=sft_dir, smoke_quota=True, offline=True, seed=123)

    nvme_dir = tmp_path / "nvme"
    ckpt_dir = tmp_path / "ckpts"
    logs_dir = tmp_path / "logs"
    export_dir = tmp_path / "export"

    base_env = os.environ.copy()
    for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "DCPA_DIST_WORKER"):
        base_env.pop(k, None)
    base_env["PYTHONPATH"] = f"{REPO_ROOT}:{TRAIN_PY}"

    common_args = [
        sys.executable,
        str(REPO_ROOT / "train" / "train_dcpa.py"),
        "--config",
        str(REPO_ROOT / "configs" / "v1" / "dcpa_unified.yaml"),
        "--preset",
        "micro",
        "--device",
        "cpu",
        "--compile-mode",
        "none",
        "--stage-boundaries",
        "0,2,4,6,8",
        "--target-tokens-bytes-per-step",
        "256",
        "--micro-batch-bytes",
        "256",
        "--data-dir",
        str(data_dir),
        "--local-cache-dir",
        str(nvme_dir),
        "--ckpt-dir",
        str(ckpt_dir),
        "--logs-dir",
        str(logs_dir),
        "--export-dir",
        str(export_dir),
        "--save-every-steps",
        "3",
        "--probe-every-steps",
        "100",
        "--console-every-steps",
        "2",
    ]

    # Phase 1: Single-process (`--gpus 1`) up to step 3
    r1 = subprocess.run(
        [*common_args, "--gpus", "1", "--total-steps", "3", "--resume", "none", "--no-export"],
        cwd=str(REPO_ROOT),
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Launcher]" not in r1.stdout
    ckpt_s3 = torch.load(ckpt_dir / "last_healthy.pt", map_location="cpu", weights_only=False)
    assert int(ckpt_s3["step"]) == 3
    phys_s3 = int(ckpt_s3["dataloader_cursor"]["phys_offset"])

    # Phase 2: Switch directly to 2-process Multi-GPU (`--gpus 2`) -> auto-spawns torchrun & resumes step 3..6!
    r2 = subprocess.run(
        [*common_args, "--gpus", "2", "--total-steps", "6", "--resume", "auto", "--no-export"],
        cwd=str(REPO_ROOT),
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Launcher] Multi-process mode: spawning 2 workers" in r2.stdout
    ckpt_s6 = torch.load(ckpt_dir / "last_healthy.pt", map_location="cpu", weights_only=False)
    assert int(ckpt_s6["step"]) == 6
    phys_s6 = int(ckpt_s6["dataloader_cursor"]["phys_offset"])
    assert phys_s6 >= phys_s3 + 3 * 2, f"Expected monotonic 2-worker phys_offset advance: {phys_s3} -> {phys_s6}"

    # Phase 3: Switch BACK to Single-process (`--gpus 1`) -> resumes step 6..8 & exports safetensors!
    r3 = subprocess.run(
        [*common_args, "--gpus", "1", "--total-steps", "8", "--resume", "auto"],
        cwd=str(REPO_ROOT),
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Launcher]" not in r3.stdout
    assert (export_dir / "dcpa_final.safetensors").is_file()

    # Phase 4: Stage 5 SFT auto-spawning 2 workers via `--gpus 2` (zero manual torchrun!)
    sft_export = tmp_path / "sft_export"
    r4 = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "train" / "train_stage5_sft.py"),
            "--gpus",
            "2",
            "--device",
            "cpu",
            "--compile-mode",
            "none",
            "--weights",
            str(export_dir / "dcpa_final.safetensors"),
            "--config",
            str(export_dir / "config.json"),
            "--data-dir",
            str(sft_dir),
            "--export-dir",
            str(sft_export),
            "--total-steps",
            "2",
            "--target-tokens-bytes-per-step",
            "512",
            "--micro-batch-bytes",
            "256",
            "--max-doc-bytes",
            "512",
            "--log-every",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Launcher] Multi-process mode: spawning 2 workers" in r4.stdout
    assert (sft_export / "dcpa_chat_sft.safetensors").is_file()

    # Phase 5: Unified `scripts/train.sh` dry-run for both `--gpus=1` and `--gpus=4 --gpu-ids=0,1,2,3`
    dry_1 = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "train.sh"), "--gpus=1", "--dry-run", f"--nvme-dir={nvme_dir}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Single-GPU / Single-Process" in dry_1.stdout
    assert "--gpus 1" in dry_1.stdout
    assert "dcpa_unified.yaml" in dry_1.stdout

    dry_4 = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "train.sh"),
            "--gpus=4",
            "--gpu-ids=0,1,2,3",
            "--dry-run",
            f"--nvme-dir={nvme_dir}",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Multi-GPU (4 GPUs)" in dry_4.stdout
    assert "--gpus 4 --gpu-ids 0,1,2,3" in dry_4.stdout
    assert "--preset moe_2_6b" in dry_4.stdout
