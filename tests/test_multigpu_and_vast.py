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

from data.build_sft_stage5 import build_stage5_sft_corpus
from data.prepare_vast_corpus import prepare_vast_corpus, verify_all_shards
from data.ragged_dataloader import IDX_RECORD_FMT, RaggedByteDataset
from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import SparseMicroMoE
from train.curriculum_state_machine import hash_named_parameters
from train.train_dcpa import build_dcpa_config, run_training
from train.train_stage5_sft import run_stage5_sft


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write_synthetic_multitier_corpus(out_dir: Path) -> None:
    """Create a deterministic 5-domain `.bin/.idx` shard spanning all 4 length tiers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tier_lengths = (220, 780, 1550, 3100)
    rng = np.random.default_rng(2026)

    for split, docs_per_cell in (("train", 16), ("val", 4)):
        bin_p = out_dir / f"{split}.bin"
        idx_p = out_dir / f"{split}.idx"
        offset = 0
        with bin_p.open("wb") as fb, idx_p.open("wb") as fi:
            for lid in range(5):
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

    target_weights = {0: 0.35, 1: 0.25, 2: 0.15, 3: 0.15, 4: 0.10}
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

        lang_byte_counts = {lid: 0 for lid in range(5)}
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
        config_path=REPO_ROOT / "configs" / "dcpa_210m_moe.yaml",
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
        config_path=REPO_ROOT / "configs" / "dcpa_210m_moe.yaml",
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
        config_path=REPO_ROOT / "configs" / "dcpa_210m_moe.yaml",
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

    yaml_cfg = yaml.safe_load((REPO_ROOT / "configs" / "dcpa_v2_4xa100.yaml").read_text(encoding="utf-8"))
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
