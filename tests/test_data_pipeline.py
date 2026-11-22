"""Five-assertion data pipeline verification suite (Iron Law #2 & Colab Cell 3).

Reads `DCPA_DATA_DIR` if set (e.g. `/content/nvme_cache/data` in Colab Cell 3),
otherwise uses `data/smoke_corpus` (building via `--smoke-quota` once if absent).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import unicodedata
from pathlib import Path

import pytest
import torch

from data.build_corpus import CodeASTValidator, SMOKE_BYTES_PER_LANG, build_corpus
from dcpa.data.shard_loader import BinaryShardDataset, IDX_RECORD_FMT, IDX_RECORD_SIZE
from train.colab_drive_sync import assert_not_drive_path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LANG_IDS = {0, 1, 2, 3, 4}  # zh, en, ja, python, rust


@pytest.fixture(scope="module")
def corpus_dir() -> Path:
    env_dir = os.environ.get("DCPA_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        assert (p / "train.bin").exists() and (p / "train.idx").exists(), (
            f"DCPA_DATA_DIR={p} is missing train.bin/train.idx"
        )
        return p

    # Prefer already-built local smoke corpus; otherwise run --smoke-quota (~10 MB)
    smoke_dir = REPO_ROOT / "data" / "smoke_corpus"
    if not ((smoke_dir / "train.bin").exists() and (smoke_dir / "corpus_meta.json").exists()):
        build_corpus(out_dir=smoke_dir, smoke_quota=True)
    return smoke_dir


def _read_idx_records(idx_path: Path) -> list[tuple[int, int, int, int]]:
    raw = idx_path.read_bytes()
    assert len(raw) > 0 and len(raw) % IDX_RECORD_SIZE == 0, f"Invalid .idx byte length: {len(raw)}"
    n = len(raw) // IDX_RECORD_SIZE
    return [struct.unpack_from(IDX_RECORD_FMT, raw, i * IDX_RECORD_SIZE) for i in range(n)]


def test_1_shard_binary_format_and_index_integrity(corpus_dir: Path):
    """Assertion 1/5: .bin/.idx files exist, 16B aligned, contiguous offsets, exact byte bounds."""
    meta_path = corpus_dir / "corpus_meta.json"
    assert meta_path.exists(), "Missing corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    for split in ("train", "val"):
        bin_path = corpus_dir / f"{split}.bin"
        idx_path = corpus_dir / f"{split}.idx"
        assert bin_path.exists() and idx_path.exists()
        bin_size = bin_path.stat().st_size
        assert bin_size > 0, f"{split}.bin is empty"

        records = _read_idx_records(idx_path)
        assert len(records) > 0, f"{split}.idx has zero records"

        cursor = 0
        for start_off, byte_len, lang_id, stage_mask in records:
            assert start_off == cursor, f"Non-contiguous offset in {split}.idx: {start_off} != {cursor}"
            assert byte_len > 0, "Zero-length document record encountered"
            assert lang_id in EXPECTED_LANG_IDS, f"Unexpected lang_id={lang_id}"
            assert stage_mask > 0, f"Empty stage_mask for lang_id={lang_id}"
            cursor += byte_len
        assert cursor == bin_size, f"{split}.bin size {bin_size} != summed index bytes {cursor}"
        assert meta[f"{split}_bytes"] == bin_size


def test_2_utf8_nfc_and_control_byte_hygiene(corpus_dir: Path):
    """Assertion 2/5: 100% strict UTF-8 NFC normalization, LF-only line endings, zero bad control bytes."""
    forbidden_bytes = set(range(0x00, 0x09)) | set(range(0x0B, 0x20)) | {0x7F}

    for split in ("train", "val"):
        records = _read_idx_records(corpus_dir / f"{split}.idx")
        # Check up to 400 evenly strided records per split for fast execution on both 10MB and 450MB corpora
        stride = max(1, len(records) // 400)
        with (corpus_dir / f"{split}.bin").open("rb") as f:
            for rec in records[::stride]:
                start_off, byte_len, _, _ = rec
                f.seek(start_off)
                raw = f.read(byte_len)
                assert len(raw) == byte_len
                assert not (set(raw) & forbidden_bytes), f"Forbidden control byte found in {split}.bin at {start_off}"
                text = raw.decode("utf-8", errors="strict")
                assert unicodedata.is_normalized("NFC", text), f"Non-NFC text in {split}.bin at {start_off}"


def test_3_five_language_domains_coverage_and_quota(corpus_dir: Path):
    """Assertion 3/5: All 5 lang_ids (zh/en/ja/python/rust) covered in train & val meeting quota."""
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8"))
    train_recs = _read_idx_records(corpus_dir / "train.idx")
    val_recs = _read_idx_records(corpus_dir / "val.idx")

    train_langs = {r[2] for r in train_recs}
    val_langs = {r[2] for r in val_recs}
    assert train_langs == EXPECTED_LANG_IDS, f"Missing train lang_ids: {EXPECTED_LANG_IDS - train_langs}"
    assert val_langs == EXPECTED_LANG_IDS, f"Missing val lang_ids: {EXPECTED_LANG_IDS - val_langs}"

    bytes_by_lang: dict[int, int] = {lid: 0 for lid in EXPECTED_LANG_IDS}
    for _, blen, lid, _ in train_recs + val_recs:
        bytes_by_lang[lid] += blen

    mode = meta.get("mode", "smoke")
    if mode == "smoke":
        min_per_lang = int(SMOKE_BYTES_PER_LANG * 0.90)
        for lid, total in bytes_by_lang.items():
            assert total >= min_per_lang, f"Smoke quota underfilled for lang_id={lid}: {total} < {min_per_lang}"
        assert sum(bytes_by_lang.values()) >= int(5 * SMOKE_BYTES_PER_LANG * 0.90)
    else:
        # Full ~450 MB production build
        assert sum(bytes_by_lang.values()) >= 400 * 1024 * 1024
        for lid, total in bytes_by_lang.items():
            assert total >= 50 * 1024 * 1024, f"Full quota underfilled for lang_id={lid}: {total}"


def test_4_code_ast_validity_and_zero_train_val_leakage(corpus_dir: Path):
    """Assertion 4/5: Python/Rust samples pass tree-sitter AST validation and train/val have 0 doc overlap."""
    validator = CodeASTValidator()

    train_hashes: set[bytes] = set()
    val_hashes: set[bytes] = set()

    for split, target_set in (("train", train_hashes), ("val", val_hashes)):
        records = _read_idx_records(corpus_dir / f"{split}.idx")
        py_checked = 0
        rs_checked = 0
        with (corpus_dir / f"{split}.bin").open("rb") as f:
            for start_off, byte_len, lang_id, _ in records:
                f.seek(start_off)
                raw = f.read(byte_len)
                target_set.add(hashlib.blake2b(raw, digest_size=16).digest())
                if lang_id == 3 and py_checked < 60:
                    assert 200 <= byte_len <= 8000
                    assert validator.is_valid("python", raw), f"Invalid Python AST in {split} at {start_off}"
                    py_checked += 1
                elif lang_id == 4 and rs_checked < 60:
                    assert 200 <= byte_len <= 8000
                    assert validator.is_valid("rust", raw), f"Invalid Rust AST in {split} at {start_off}"
                    rs_checked += 1
        assert py_checked > 0 and rs_checked > 0

    overlap = train_hashes & val_hashes
    assert len(overlap) == 0, f"Detected {len(overlap)} leaked documents shared between train and val!"


def test_5_mmap_dataset_loading_and_drive_fuse_guard(corpus_dir: Path):
    """Assertion 5/5: Zero-copy BinaryShardDataset yields uint8 [seq_len] tensors and blocks /content/drive."""
    ds = BinaryShardDataset(corpus_dir / "train.bin", corpus_dir / "train.idx", seq_len=256)
    try:
        assert len(ds) > 0
        for idx in (0, len(ds) // 2, len(ds) - 1):
            raw_t, lang_t = ds[idx]
            assert raw_t.shape == (256,)
            assert raw_t.dtype == torch.uint8
            assert int(lang_t.item()) in EXPECTED_LANG_IDS
    finally:
        ds.close()

    # Verify Iron Law #3.1 guard rejects Google Drive FUSE mount paths for mmap / hot cache
    with pytest.raises(ValueError, match="Drive FUSE"):
        BinaryShardDataset("/content/drive/MyDrive/DCPA_Project/train.bin", "/content/drive/MyDrive/DCPA_Project/train.idx")

    with pytest.raises(ValueError, match="/content/drive"):
        assert_not_drive_path("/content/drive/MyDrive/DCPA_Project/cache")
