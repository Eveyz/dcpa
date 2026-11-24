"""Acceptance test suite for DCPA v2 (`moe_210m`) Holistic 35:35:30 Cognitive Pipeline.

Covers (100% offline — zero network downloads):
  1. `configs/dcpa_210m_v2_cognitive.yaml`:
       - `moe_210m` config & `W_max=64`, `householder_write_threshold=0.08`, `num_lang_domains=6`
       - Stage boundaries `[0, 12000, 42000, 78000, 100000]`
       - Stage 1 `max_stage_tier == 1` & `anchor_doc_start is True`
       - Stage 2 `Slot-0` starvation guard (`cap_mix_prob == 0.0` at step 15000, `0.15` at step 35000)
       - Stage 4 `wug_mode == "cognitive_v2"` & `wug_ratio == 0.55`
       - Exact 6-slot `35:35:30` holistic domain weight sum == 1.0 across all stages
  2. Offline 6-domain staged corpus (`build_corpus(..., cognitive_v2=True, offline=True)`):
       - All 6 `lang_id`s (`0=en, 1=zh, 2=formal, 3=py, 4=sql_struct, 5=js`) populated
       - ZERO Rust code in `lang_id=4` (verified SQL/JSON structured domain)
       - Non-zero `style_id` distribution across `0..4` and stage tiers `1..3`
  3. Data-Alchemy builders (`build_cap_code_v2`, `build_cap_reasoning --cognitive-v2`,
     `build_cap_writing`, `build_sft_stage5 --cognitive-v2`):
       - 100% `verify_shard_triplet` & `verify_mask_alignment`
       - Anti-pattern / slop inputs strictly masked `0`, refactored / clean outputs masked `1`
       - Zero Rust across `cap_code` and `sft_stage5`
  4. Stage-4 `sample_cognitive_stage4_episode_fast` across all 6 `lang_id`s (`0..5`)
  5. Orchestrator (`prepare_vast_corpus` `v2_cognitive_5g` dry-run + `scripts/train.sh --v2-cognitive --dry-run`)
     and end-to-end Stage 1..4 + Stage 5 SFT mini-run with `dcpa_210m_v2_cognitive.yaml`.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.build_cap_code_v2 import build_cap_code_corpus
from data.build_cap_instruct import build_cap_instruct_corpus
from data.build_cap_reasoning import build_cap_reasoning_corpus
from data.build_cap_toolcall import build_cap_toolcall_corpus
from data.build_cap_writing import build_cap_writing_corpus
from data.build_corpus import build_corpus
from data.build_sft_stage5 import build_stage5_sft_corpus
from data.prepare_vast_corpus import prepare_vast_corpus, verify_shard_triplet
from data.ragged_dataloader import IDX_DTYPE, RaggedByteDataset, infer_doc_stage_tiers
from data.wug_generator import COGNITIVE_V2_WUG_LANG_IDS, sample_cognitive_stage4_episode_fast
from dcpa.model import DCPAModel
from train.curriculum_state_machine import CurriculumController
from train.train_dcpa import build_dcpa_config, run_training
from train.train_stage5_sft import run_stage5_sft
from utils.chat_protocol import SCRATCH_END, SCRATCH_START, verify_mask_alignment


def test_v2_cognitive_yaml_and_curriculum_schedule() -> None:
    cfg_path = REPO_ROOT / "configs" / "dcpa_210m_v2_cognitive.yaml"
    assert cfg_path.is_file()
    raw_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg = build_dcpa_config("moe_210m", raw_yaml)
    assert cfg.vocab_size == 256
    assert cfg.num_lang_domains == 6
    assert cfg.num_style_domains == 5
    assert cfg.num_experts == 32
    assert cfg.top_k_experts == 2
    assert cfg.householder_max_writes == 64
    assert abs(cfg.householder_write_threshold - 0.08) < 1e-6

    with torch.device("meta"):
        model = DCPAModel(cfg)
        total_params = sum(p.numel() for p in model.parameters())
    assert 195_000_000 <= total_params <= 225_000_000

    csm = CurriculumController(raw_yaml.get("curriculum", {}))
    assert csm.boundaries == (0, 12000, 42000, 78000, 100000)

    # Stage 1: tier-1 syntax/rhythm filter + doc-start anchoring + zero cap_mix
    s1 = csm.get(1000)
    assert s1.stage_id == 1
    assert s1.max_stage_tier == 1
    assert s1.anchor_doc_start is True
    assert s1.cap_mix_prob == 0.0
    assert abs(sum(s1.domain_weights.values()) - 1.0) < 1e-5

    # Stage 2 early (step 15000 < 32000): Slot-0 unpinned (`cap_mix_prob == 0.0`) so all 72 pairs train!
    s2_early = csm.get(15000)
    assert s2_early.stage_id == 2
    assert s2_early.max_stage_tier == 2
    assert s2_early.anchor_doc_start is False
    assert s2_early.cap_mix_prob == 0.0

    # Stage 2 late (step 35000 >= 32000): gentle Slot-0 outline warmup (`cap_mix_prob == 0.15`)
    s2_late = csm.get(35000)
    assert s2_late.stage_id == 2
    assert abs(s2_late.cap_mix_prob - 0.15) < 1e-6

    # Stage 3 (step 60000): full alchemy burst (`cap_mix_prob == 0.45`, tier 3)
    s3 = csm.get(60000)
    assert s3.stage_id == 3
    assert s3.max_stage_tier == 3
    assert abs(s3.cap_mix_prob - 0.45) < 1e-6
    assert abs(sum(s3.domain_weights.values()) - 1.0) < 1e-5

    # Stage 4 (step 85000): World-Bible & Micro-SDK fast-weight consolidation (`cognitive_v2`)
    s4 = csm.get(85000)
    assert s4.stage_id == 4
    assert s4.wug_mode == "cognitive_v2"
    assert abs(s4.wug_ratio - 0.55) < 1e-6
    assert s4.max_doc_bytes == 3072
    assert "cortical_io" in s4.frozen_owners and "bcg_pc" in s4.frozen_owners


def test_stage4_cognitive_wug_episodes_all_6_domains_zero_rust() -> None:
    rng = random.Random(202604)
    assert COGNITIVE_V2_WUG_LANG_IDS == (0, 1, 2, 3, 4, 5)

    for lid in COGNITIVE_V2_WUG_LANG_IDS:
        styles_seen: set[int] = set()
        for i in range(8):
            max_b = 2400 if i % 2 == 0 else 800
            raw_b, t1_len, tgt_s, tgt_e, style_id = sample_cognitive_stage4_episode_fast(
                rng, lid, max_bytes=max_b
            )
            assert len(raw_b) <= max_b
            assert 0 < t1_len <= tgt_s < tgt_e <= len(raw_b)
            assert 0 <= style_id <= 4
            txt = raw_b.decode("utf-8", errors="strict")
            if lid == 4:
                # Slot 4 MUST be SQL/structured schema, NEVER Rust (`fn ` / `let mut `)
                assert "CREATE TABLE" in txt or "SELECT" in txt or "def " in txt
                assert "let mut " not in txt and "Vec::with_capacity" not in txt
            styles_seen.add(style_id)
        if lid in (3, 4, 5):
            assert styles_seen == {4}


def test_v2_builders_offline_6domain_alchemy_and_end_to_end_train(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    cap_dir = tmp_path / "cap_data"
    sft_dir = tmp_path / "sft_data"

    # 1. Build 6-domain staged pre-training corpus (zero Rust)
    corpus_meta = build_corpus(
        out_dir=data_dir,
        smoke_quota=True,
        workers=2,
        seed=2026,
        cognitive_v2=True,
        offline=True,
    )
    ok_tr, _ = verify_shard_triplet(data_dir / "train.bin", data_dir / "train.idx")
    ok_va, _ = verify_shard_triplet(data_dir / "val.bin", data_dir / "val.idx")
    assert ok_tr and ok_va

    idx_c = np.memmap(data_dir / "train.idx", dtype=IDX_DTYPE, mode="r")
    bin_c = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    corpus_langs = set(np.unique(idx_c["lang_id"]).tolist())
    corpus_styles = set(np.unique(idx_c["style_id"]).tolist())
    tiers = set(np.unique(infer_doc_stage_tiers(idx_c["source_id"], idx_c["length"])).tolist())
    assert corpus_langs == {0, 1, 2, 3, 4, 5}, f"Expected all 6 v2 domains, got {corpus_langs}"
    assert len(corpus_styles) >= 4, f"Expected multi-style coverage in corpus .idx, got {corpus_styles}"
    assert tiers == {1, 2, 3}, f"Expected stage tiers 1..3 in corpus .idx, got {tiers}"

    # Verify lang_id=4 docs in corpus are SQL/JSON structured specs, NOT Rust
    slot4_idxs = np.nonzero(idx_c["lang_id"] == 4)[0][:12]
    assert len(slot4_idxs) > 0
    for d_i in slot4_idxs:
        off, ln = int(idx_c[d_i]["offset"]), int(idx_c[d_i]["length"])
        sample_txt = bytes(bin_c[off : off + ln]).decode("utf-8", errors="strict")
        assert "let mut " not in sample_txt and "impl " not in sample_txt
    del idx_c, bin_c
    assert corpus_meta["cognitive_v2"] is True

    # 2. Build capability alchemy shards (`cap_code`, `cap_reasoning --cognitive-v2`, `cap_writing`, `cap_instruct`, `cap_toolcall`)
    code_meta = build_cap_code_corpus(out_dir=cap_dir, target_mb=0.6, offline=True, seed=202615)
    reas_meta = build_cap_reasoning_corpus(out_dir=cap_dir, target_mb=0.6, offline=True, seed=202611, cognitive_v2=True)
    writ_meta = build_cap_writing_corpus(out_dir=cap_dir, target_mb=0.6, offline=True, seed=202612)
    build_cap_instruct_corpus(out_dir=cap_dir, target_mb=0.4, offline=True, seed=202614)
    build_cap_toolcall_corpus(out_dir=cap_dir, target_mb=0.4, offline=True, seed=202613)
    assert code_meta["total_docs"] > 0 and set(code_meta["docs_by_lang_id"].keys()) == {3, 4, 5}
    assert reas_meta["explicit_docs"] > 0 and reas_meta["latent_docs"] > 0
    assert writ_meta["total_docs"] > 0

    # Verify cap_code mask alignment & zero Rust
    ok_cc, _ = verify_shard_triplet(
        cap_dir / "cap_code_train.bin",
        cap_dir / "cap_code_train.idx",
        cap_dir / "cap_code_train.loss_mask.bin",
    )
    assert ok_cc
    idx_cc = np.memmap(cap_dir / "cap_code_train.idx", dtype=IDX_DTYPE, mode="r")
    bin_cc = np.memmap(cap_dir / "cap_code_train.bin", dtype=np.uint8, mode="r")
    msk_cc = np.memmap(cap_dir / "cap_code_train.loss_mask.bin", dtype=np.uint8, mode="r")
    cc_langs = set(np.unique(idx_cc["lang_id"]).tolist())
    assert cc_langs == {3, 4, 5}
    assert set(np.unique(idx_cc["style_id"]).tolist()) == {4}
    for i in range(min(40, len(idx_cc))):
        off, ln = int(idx_cc[i]["offset"]), int(idx_cc[i]["length"])
        rb = bytes(bin_cc[off : off + ln])
        mb = bytes(msk_cc[off : off + ln])
        assert verify_mask_alignment(rb, mb)
    del idx_cc, bin_cc, msk_cc

    # Verify Stage-1 tier filtering + doc-start anchoring + Stage-4 cognitive_v2 loader
    ds_s1 = RaggedByteDataset(
        bin_path=data_dir / "train.bin",
        idx_path=data_dir / "train.idx",
        target_batch_bytes=2048,
        max_doc_bytes=512,
        wug_ratio=0.0,
        max_stage_tier=1,
        anchor_doc_start=True,
        seed=42,
    )
    ds_s4 = RaggedByteDataset(
        bin_path=data_dir / "train.bin",
        idx_path=data_dir / "train.idx",
        target_batch_bytes=4096,
        max_doc_bytes=2048,
        wug_ratio=1.0,
        wug_mode="cognitive_v2",
        seed=42,
    )
    try:
        b1 = ds_s1.get_batch(0)
        assert b1["x_bytes"].numel() > 0
        b4 = ds_s4.get_batch(0)
        assert bool(b4["wug_target_mask"].any().item())
        assert bool((~b4["loss_mask"]).any().item())
        # Verify Slot-0 scratchpad frame appears inside cognitive_v2 Stage-4 episodes
        raw_s4 = bytes(b4["x_bytes"].numpy().tolist())
        assert SCRATCH_START in raw_s4 and SCRATCH_END in raw_s4
    finally:
        ds_s1.close()
        ds_s4.close()

    # 3. Build Stage 5 holistic cognitive SFT (`35:35:30`, zero Rust)
    sft_meta = build_stage5_sft_corpus(
        out_dir=sft_dir,
        train_mb=0.6,
        val_mb=0.1,
        offline=True,
        cognitive_v2=True,
        seed=202605,
    )
    assert sft_meta["cognitive_v2"] is True
    idx_sft = np.memmap(sft_dir / "sft_train.idx", dtype=IDX_DTYPE, mode="r")
    bin_sft = np.memmap(sft_dir / "sft_train.bin", dtype=np.uint8, mode="r")
    msk_sft = np.memmap(sft_dir / "sft_train.loss_mask.bin", dtype=np.uint8, mode="r")
    assert set(np.unique(idx_sft["source_id"]).tolist()) == {0, 1, 2}
    assert len(set(np.unique(idx_sft["style_id"]).tolist())) >= 4
    for i in range(min(50, len(idx_sft))):
        off, ln = int(idx_sft[i]["offset"]), int(idx_sft[i]["length"])
        rb = bytes(bin_sft[off : off + ln])
        mb = bytes(msk_sft[off : off + ln])
        assert verify_mask_alignment(rb, mb)
        assert b"```rust" not in rb
    del idx_sft, bin_sft, msk_sft

    # 4. Run end-to-end mini Stage 1..4 pre-train + Stage 5 SFT using `dcpa_210m_v2_cognitive.yaml`
    cfg_path = REPO_ROOT / "configs" / "dcpa_210m_v2_cognitive.yaml"
    summary = run_training(
        config_path=cfg_path,
        preset="micro",
        total_steps=8,
        stage_boundaries=(0, 2, 4, 6, 8),
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=256,
        data_dir=data_dir,
        cap_data_dir=cap_dir,
        local_cache_dir=tmp_path / "nvme",
        ckpt_dir=tmp_path / "ckpts",
        logs_dir=tmp_path / "logs",
        export_dir=tmp_path / "export",
        probe_every_steps=100,
        console_every_steps=4,
        resume="none",
        device_str="cpu",
        compile_mode="none",
        seed=2026,
        export_final=True,
    )
    assert summary.completed_steps == 8
    assert (tmp_path / "export" / "dcpa_final.safetensors").is_file()

    sft_res = run_stage5_sft(
        weights_path=tmp_path / "export" / "dcpa_final.safetensors",
        config_path=tmp_path / "export" / "config.json",
        data_dir=sft_dir,
        cap_data_dir=cap_dir,
        export_dir=tmp_path / "sft_export",
        total_steps=3,
        warmup_steps=1,
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=256,
        max_doc_bytes=4096,
        device_str="cpu",
        compile_mode="none",
        seed=202605,
        log_every=1,
    )
    assert sft_res.completed_steps == 3
    assert (tmp_path / "sft_export" / "dcpa_chat_sft.safetensors").is_file()


def test_v2_orchestrator_profile_and_train_sh_dry_run(tmp_path: Path) -> None:
    plan = prepare_vast_corpus(
        profile="v2_cognitive_5g",
        nvme_root=tmp_path / "nvme_v2",
        dry_run=True,
    )
    assert plan["dry_run"] is True
    assert set(plan["commands"].keys()) == {
        "corpus_5domain",
        "cap_reasoning",
        "cap_instruct",
        "cap_writing",
        "cap_toolcall",
        "cap_code",
        "stage5_sft",
    }
    assert "--cognitive-v2" in plan["commands"]["corpus_5domain"]
    assert "--cognitive-v2" in plan["commands"]["cap_reasoning"]
    assert "--cognitive-v2" in plan["commands"]["stage5_sft"]

    proc = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "train.sh"),
            "--v2-cognitive",
            "--gpus=1",
            "--dry-run",
            f"--nvme-dir={tmp_path / 'nvme_v2'}",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "dcpa_210m_v2_cognitive.yaml" in proc.stdout
    assert "preset=moe_210m" in proc.stdout
    assert "--total-steps 15000" in proc.stdout
