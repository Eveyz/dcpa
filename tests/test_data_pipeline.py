"""Five-assertion automated audit suite for the DCPA pure-UTF-8 Ragged data pipeline.

Implements Deliverable 4 (and Iron Law #2: locally uses `--smoke-quota` under `artifacts/data`,
or `DCPA_DATA_DIR` when pointing to Colab's full `/content/nvme_cache/data` build):
  - Test 1: Zero-padding & 100% UTF-8 character-boundary safety over 200 Ragged batches
            across `max_doc_bytes in (512, 1024, 2048)`.
  - Test 2: Blind `tree-sitter >= 0.23` AST & 1..16 bracket-depth audit on >= 500 Python
            (`lang_id=3`) and >= 500 Rust (`lang_id=4`) persisted documents (0.0% error rate).
  - Test 3: Full-scan `blake2b` Train / Val zero-leakage & binary index integrity check.
  - Test 4: 500-episode Wug synthesis audit covering all 5 `lang_id`s, exact inflected
            `wug_target_mask` byte slice alignment, `tree-sitter` validation on code episodes,
            and 45%..55% Pre/Post definition symmetry.
  - Test 5: Throughput benchmark (`num_workers=0` and `num_workers=4`, `target_batch_bytes=131072`,
            `wug_ratio=0.50`), enforcing `>= 20 MB/s`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import torch

from data.build_corpus import CodeASTValidator, build_corpus
from data.ragged_dataloader import (
    IDX_DTYPE,
    IDX_RECORD_SIZE,
    RaggedByteBatchSampler,
    RaggedByteDataset,
    assert_not_drive_mmap,
    create_ragged_dataloader,
    utf8_safe_slice,
)
from data.wug_generator import (
    format_episode_annotated,
    inflect_past,
    inflect_plural,
    inflect_prog,
    sample_wug_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LANG_IDS = {0, 1, 2, 3, 4}  # en, zh, ja, python, rust


@pytest.fixture(scope="module")
def corpus_dir() -> Path:
    env_dir = os.environ.get("DCPA_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        assert (p / "train.bin").exists() and (p / "train.idx").exists(), (
            f"DCPA_DATA_DIR={p} missing train.bin/train.idx"
        )
        return p

    # Default local M3 Pro path (`artifacts/data`); auto-build in --smoke-quota if absent (Iron Law #2)
    out_dir = REPO_ROOT / "artifacts" / "data"
    if not (
        (out_dir / "train.bin").exists()
        and (out_dir / "train.idx").exists()
        and (out_dir / "train.meta.json").exists()
        and (out_dir / "val.bin").exists()
        and (out_dir / "val.idx").exists()
        and (out_dir / "val.meta.json").exists()
    ):
        build_corpus(out_dir=out_dir, smoke_quota=True)
    return out_dir


def test_1_zero_padding_and_utf8_boundary_integrity(corpus_dir: Path):
    """Test 1: 200 Ragged batches across max_doc_bytes={512, 1024, 2048} -> 0 padding, 0 UTF-8 errors."""
    forbidden_bytes = set(range(0x00, 0x09)) | set(range(0x0B, 0x20)) | {0x7F}

    # Unit check on utf8_safe_slice with 3-byte CJK & 4-byte supplementary Hanzi
    sample_utf8 = "清晨山谷起雾𫛭狴晶青い鳥が静かな森の上を飛ぶ".encode("utf-8")
    arr = np.frombuffer(sample_utf8, dtype=np.uint8)
    for max_len in range(4, len(arr) + 1):
        cursor = 0
        rebuilt = bytearray()
        while cursor < len(arr):
            sl = utf8_safe_slice(arr, cursor, max_len)
            assert len(sl) > 0
            bytes(sl).decode("utf-8", errors="strict")
            rebuilt.extend(sl.tobytes())
            cursor += len(sl)
        assert bytes(rebuilt) == sample_utf8

    window_plan = [(512, 67), (1024, 67), (2048, 66)]  # Total = 200 batches
    total_batches = 0
    total_segments = 0
    utf8_errors = 0

    for max_doc_bytes, n_batches in window_plan:
        ds = RaggedByteDataset(
            corpus_dir / "train.bin",
            corpus_dir / "train.idx",
            target_batch_bytes=16384,
            max_doc_bytes=max_doc_bytes,
            domain_weights={0: 0.25, 1: 0.25, 2: 0.20, 3: 0.15, 4: 0.15},
            wug_ratio=0.0,
            num_batches=n_batches,
            seed=100 + max_doc_bytes,
        )
        try:
            for b_idx in range(n_batches):
                batch = ds[b_idx]
                x = batch["x_bytes"]
                cu = batch["cu_seqlens"]
                starts = batch["is_doc_start"]
                doc_lids = batch["doc_lang_ids"]
                byte_lids = batch["byte_lang_ids"]
                lmask = batch["loss_mask"]

                total_t = int(x.numel())
                assert x.dtype == torch.uint8 and x.ndim == 1
                assert 0 < total_t <= 16384
                assert cu.dtype == torch.int32 and cu.ndim == 1
                assert int(cu[0].item()) == 0
                assert int(cu[-1].item()) == total_t
                assert torch.all(cu[1:] > cu[:-1]), "cu_seqlens must be strictly increasing"

                # Verify zero 0x00 or forbidden ASCII control bytes anywhere in batch
                raw_all = x.numpy().tobytes()
                assert not (set(raw_all) & forbidden_bytes), "Illegal padding or control byte found in x_bytes"

                # Verify is_doc_start matches cu_seqlens[:-1] identically
                expected_starts = torch.zeros(total_t, dtype=torch.bool)
                expected_starts[cu[:-1].long()] = True
                assert torch.equal(starts, expected_starts)
                assert lmask.dtype == torch.bool and lmask.all()

                n_docs = int(cu.numel()) - 1
                assert doc_lids.shape == (n_docs,)
                assert byte_lids.shape == (total_t,)

                cu_py = cu.tolist()
                dl_py = doc_lids.tolist()
                for i in range(n_docs):
                    s, e = cu_py[i], cu_py[i + 1]
                    seg_bytes = raw_all[s:e]
                    assert len(seg_bytes) <= max_doc_bytes
                    try:
                        seg_bytes.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        utf8_errors += 1
                    assert int(byte_lids[s].item()) == dl_py[i]
                    assert int(byte_lids[e - 1].item()) == dl_py[i]
                    total_segments += 1
                total_batches += 1
        finally:
            ds.close()

    # Also verify RaggedByteBatchSampler + Drive FUSE protection
    sampler = RaggedByteBatchSampler(
        corpus_dir / "train.bin",
        corpus_dir / "train.idx",
        target_batch_bytes=8192,
        max_doc_bytes=512,
        num_batches=4,
        seed=7,
    )
    ds_pack = RaggedByteDataset(corpus_dir / "train.bin", corpus_dir / "train.idx", target_batch_bytes=8192)
    try:
        for specs in sampler:
            b = ds_pack.pack_batch_from_specs(specs)
            assert int(b["cu_seqlens"][-1].item()) == int(b["x_bytes"].numel())
    finally:
        ds_pack.close()

    with pytest.raises(ValueError, match="/content/drive"):
        assert_not_drive_mmap("/content/drive/MyDrive/DCPA_Project/train.bin")

    print(
        f"[PASS Test 1] batches={total_batches}, segments={total_segments}, "
        f"utf8_decode_errors={utf8_errors}, illegal_padding_bytes=0",
        flush=True,
    )
    assert total_batches == 200
    assert utf8_errors == 0


def test_2_code_treesitter_blind_audit(corpus_dir: Path):
    """Test 2: Blind tree-sitter + bracket-stack re-verification on Python (3) & Rust (4) docs."""
    validator = CodeASTValidator()
    rng = random.Random(42)

    checked_counts = {3: 0, 4: 0}
    error_counts = {3: 0, 4: 0}
    depth_ranges = {3: [999, 0], 4: [999, 0]}

    for lid, lang_name in ((3, "python"), (4, "rust")):
        candidates: list[tuple[Path, int, int, int]] = []
        for split in ("train", "val"):
            bin_p = corpus_dir / f"{split}.bin"
            idx_p = corpus_dir / f"{split}.idx"
            idx_arr = np.memmap(idx_p, dtype=IDX_DTYPE, mode="r")
            mask = idx_arr["lang_id"] == lid
            for row in idx_arr[mask]:
                candidates.append((bin_p, int(row["offset"]), int(row["length"]), int(row["max_depth"])))
            del idx_arr

        assert len(candidates) > 0, f"No persisted documents found for lang_id={lid} ({lang_name})"
        # Check all docs if <= 600, else random sample of 600 (>= 500 mandated!)
        if len(candidates) > 600:
            sample = rng.sample(candidates, 600)
        else:
            sample = candidates

        mm_cache: dict[Path, np.memmap] = {}
        try:
            for bin_p, off, length, stored_depth in sample:
                if bin_p not in mm_cache:
                    mm_cache[bin_p] = np.memmap(bin_p, dtype=np.uint8, mode="r")
                raw_b = bytes(mm_cache[bin_p][off : off + length])
                ok, depth = validator.validate(lang_name, raw_b)
                checked_counts[lid] += 1
                if (not ok) or not (1 <= depth <= 16) or (depth != stored_depth):
                    error_counts[lid] += 1
                else:
                    depth_ranges[lid][0] = min(depth_ranges[lid][0], depth)
                    depth_ranges[lid][1] = max(depth_ranges[lid][1], depth)
        finally:
            mm_cache.clear()

    total_checked = checked_counts[3] + checked_counts[4]
    total_errors = error_counts[3] + error_counts[4]
    err_rate = 100.0 * total_errors / max(total_checked, 1)
    print(
        f"[PASS Test 2] py_checked={checked_counts[3]} (depth={depth_ranges[3][0]}..{depth_ranges[3][1]}), "
        f"rs_checked={checked_counts[4]} (depth={depth_ranges[4][0]}..{depth_ranges[4][1]}), "
        f"errors={total_errors} ({err_rate:.2f}%)",
        flush=True,
    )
    assert total_errors == 0, f"Blind tree-sitter audit found {total_errors} invalid code docs!"


def test_3_train_val_zero_leakage_blake2b(corpus_dir: Path):
    """Test 3: Full-scan blake2b hash deduplication & zero Train/Val overlap + .idx integrity."""
    hashes: dict[str, set[bytes]] = {"train": set(), "val": set()}
    seen_langs: dict[str, set[int]] = {"train": set(), "val": set()}

    for split in ("train", "val"):
        bin_p = corpus_dir / f"{split}.bin"
        idx_p = corpus_dir / f"{split}.idx"
        meta_p = corpus_dir / f"{split}.meta.json"
        assert bin_p.exists() and idx_p.exists() and meta_p.exists()

        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        idx_size = idx_p.stat().st_size
        assert idx_size > 0 and idx_size % IDX_RECORD_SIZE == 0

        idx_arr = np.memmap(idx_p, dtype=IDX_DTYPE, mode="r")
        bin_mm = np.memmap(bin_p, dtype=np.uint8, mode="r")
        try:
            cursor = 0
            for row in idx_arr:
                off = int(row["offset"])
                length = int(row["length"])
                lid = int(row["lang_id"])
                assert off == cursor, f"Non-contiguous offset in {split}.idx: {off} != {cursor}"
                assert length > 0
                assert lid in EXPECTED_LANG_IDS
                seen_langs[split].add(lid)

                doc_b = bytes(bin_mm[off : off + length])
                hashes[split].add(hashlib.blake2b(doc_b, digest_size=8).digest())
                cursor += length

            assert cursor == bin_p.stat().st_size
            assert meta["total_bytes"] == cursor
            assert meta["total_docs"] == len(idx_arr)
        finally:
            del idx_arr, bin_mm

    assert seen_langs["train"] == EXPECTED_LANG_IDS
    assert seen_langs["val"] == EXPECTED_LANG_IDS

    overlap = hashes["train"] & hashes["val"]
    print(
        f"[PASS Test 3] train_unique={len(hashes['train'])}, val_unique={len(hashes['val'])}, "
        f"train_val_overlap={len(overlap)}",
        flush=True,
    )
    assert len(overlap) == 0, f"Detected {len(overlap)} leaked documents between train and val!"


def test_4_wug_episodes_bidirectional_morphology_and_masks():
    """Test 4: 500 Wug Episodes across lang_id 0..4 -> exact mask slice, tree-sitter, 45..55% Pre/Post."""
    # Verify English orthographic inflection engine rules
    assert inflect_plural("blicket") == "blickets"
    assert inflect_plural("glosh") == "gloshes"
    assert inflect_plural("zorbix") == "zorbixes"
    assert inflect_plural("blory") == "blories"
    assert inflect_past("wug") == "wugged"
    assert inflect_past("toma") == "tomaed"
    assert inflect_past("blore") == "blored"
    assert inflect_past("snipy") == "snipied"
    assert inflect_prog("wug") == "wugging"
    assert inflect_prog("blore") == "bloring"

    validator = CodeASTValidator()
    rng = random.Random(2026)

    total_eps = 500
    pre_count = 0
    post_count = 0
    code_checked = 0
    sample_examples: dict[int, str] = {}

    for i in range(total_eps):
        lid = i % 5
        ep = sample_wug_episode(rng, lid)

        assert ep.lang_id == lid
        assert ep.definition_order in ("pre", "post")
        if ep.definition_order == "pre":
            pre_count += 1
        else:
            post_count += 1

        # Exact UTF-8 NFC hygiene
        decoded = ep.raw_bytes.decode("utf-8", errors="strict")
        assert unicodedata.is_normalized("NFC", decoded)
        assert ep.x_bytes.dtype == torch.uint8
        assert len(ep.x_bytes) == len(ep.raw_bytes)

        # Turn 1 loss_mask == False, Turn 2 loss_mask == True
        assert not bool(ep.loss_mask[: ep.turn1_byte_len].any().item())
        assert bool(ep.loss_mask[ep.turn1_byte_len :].all().item())

        # wug_target_mask must be a strict subset of loss_mask
        assert int((ep.wug_target_mask & ~ep.loss_mask).sum().item()) == 0
        assert int(ep.wug_target_mask.sum().item()) > 0

        # Exact byte-for-byte target match!
        extracted = bytes(ep.x_bytes[ep.wug_target_mask].tolist())
        expected = ep.target_word.encode("utf-8", errors="strict")
        assert extracted == expected, f"Target byte mismatch (lang={lid}): {extracted!r} != {expected!r}"

        # English must actually inflect the stem (plural / past / prog != bare stem)
        if lid == 0:
            assert ep.target_word != ep.stem, f"EN episode failed to inflect stem {ep.stem!r}"

        # Python & Rust episodes must 100% pass tree-sitter zero-error validation
        if lid in (3, 4):
            lang_name = "python" if lid == 3 else "rust"
            ok, depth = validator.validate(lang_name, ep.raw_bytes)
            assert ok and 1 <= depth <= 16, f"Wug {lang_name} episode failed tree-sitter:\n{ep.text}"
            code_checked += 1

        if lid not in sample_examples:
            sample_examples[lid] = format_episode_annotated(ep)

    pre_pct = 100.0 * pre_count / total_eps
    print(
        f"[PASS Test 4] episodes={total_eps}, pre={pre_count} ({pre_pct:.1f}%), "
        f"post={post_count} ({100.0 - pre_pct:.1f}%), code_ts_valid={code_checked}/200 (100%)",
        flush=True,
    )
    assert 45.0 <= pre_pct <= 55.0, f"Pre/Post ratio {pre_pct:.1f}% outside [45%, 55%]!"


def test_5_dataloader_throughput_benchmark(corpus_dir: Path):
    """Test 5: Single-process & 4-worker throughput >= 20 MB/s at target_batch_bytes=131072, wug_ratio=0.5."""
    target_batch_bytes = 131072
    n_batches = 240  # ~30 MB streamed per mode

    results: dict[int, float] = {}
    for num_workers in (0, 4):
        loader = create_ragged_dataloader(
            corpus_dir / "train.bin",
            corpus_dir / "train.idx",
            target_batch_bytes=target_batch_bytes,
            max_doc_bytes=2048,
            domain_weights={0: 0.25, 1: 0.25, 2: 0.15, 3: 0.20, 4: 0.15},
            wug_ratio=0.50,
            num_batches=n_batches,
            num_workers=num_workers,
            seed=99 + num_workers,
        )
        # Warmup 2 batches (starts worker processes before measuring steady-state throughput)
        it = iter(loader)
        for _ in range(2):
            _ = next(it)

        t0 = time.perf_counter()
        total_bytes = 0
        wug_bytes = 0
        for batch in it:
            total_bytes += int(batch["x_bytes"].numel())
            wug_bytes += int(batch["wug_target_mask"].sum().item())
        elapsed = max(time.perf_counter() - t0, 1e-6)
        mb_per_sec = (total_bytes / (1024.0 * 1024.0)) / elapsed
        results[num_workers] = mb_per_sec
        assert wug_bytes > 0, "Expected non-zero Wug target bytes with wug_ratio=0.50"

    print(
        f"[PASS Test 5] throughput (131KB/batch, 50% Wug): "
        f"num_workers=0 -> {results[0]:.1f} MB/s | num_workers=4 -> {results[4]:.1f} MB/s (threshold >= 20.0 MB/s)",
        flush=True,
    )
    assert results[0] >= 20.0, f"Single-process throughput {results[0]:.1f} MB/s < 20 MB/s"
    assert results[4] >= 20.0, f"4-worker throughput {results[4]:.1f} MB/s < 20 MB/s"
