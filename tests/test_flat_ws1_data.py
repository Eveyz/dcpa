"""WS-1 Acceptance Test Suite for DCPA-Flat v3.3 Unified Data & Static-Grid Pipeline.

Verifies:
  1. Unified 6-domain / 5-style protocol constants and `<QIBBBB` layout across
     `dcpa.interfaces`, `dcpa.config`, `utils.chat_protocol`, and `data.cognitive_v2_sources`,
     plus Stage-2 `validate_framed_record` protocol hygiene.
  2. Syntax- and paragraph-aligned lossless pre-chunking of long EN/ZH prose and
     long Python/SQL/JS source files (100% valid UTF-8, 100% balanced brackets,
     `max_depth <= 16`, compact continuation headers `<= 256 B` with `0` loss mask).
  3. Fixed-point x10 (`0 / 10 / 15 / 30`) `.loss_mask.bin` encoding & decoding
     (including legacy `1 -> 1.0` compatibility) and end-to-end `build_corpus(..., v3_flat=True)`.
  4. `StaticGridByteDataset` / `build_static_grid_dataloader` over 200+ batches:
     passes `StaticGridBatch.validate()`, zero UTF-8 splits, zero truncated unbalanced
     code docs, zero lane-tail leakage, and average lane-tail pad waste `< 6%`.
  5. Registration of all 12 2025-2026 high-purity source descriptors and dry-run
     verification of `prepare_vast_corpus.py` profiles `v3_flat_45g` and `v3_flat_90g`.
"""

from __future__ import annotations

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

from data.build_corpus import CodeASTValidator, SOURCE_IDS, build_corpus
from data.cognitive_v2_sources import (
    COGNITIVE_V2_DOMAIN_NAMES,
    LOSS_MASK_ANCHOR_X10,
    LOSS_MASK_IGNORE,
    LOSS_MASK_LEGACY_ONE,
    LOSS_MASK_NORMAL_X10,
    LOSS_MASK_SYNTAX_X10,
    SYNTAX_DELIM_BYTES,
    V3_FLAT_SOURCE_DESCRIPTORS,
    chunk_code_lossless,
    chunk_prose_lossless,
    encode_fixed_point_loss_mask,
    prechunk_document,
    scan_code_lexical,
)
from data.prepare_vast_corpus import PROFILES, prepare_vast_corpus, verify_shard_triplet
from data.ragged_dataloader import (
    IDX_DTYPE,
    IDX_RECORD_FMT,
    IDX_RECORD_SIZE,
    StaticGridByteDataset,
    _LOSS_MASK_DECODE_LUT,
    build_static_grid_dataloader,
    static_grid_batch_from_dict,
)
from dcpa import config as dcpa_config
from dcpa.interfaces import (
    ALLOWED_PROTOCOL_CTRL_BYTES as IF_ALLOWED_CTRL,
    IDX_RECORD_FMT as IF_IDX_FMT,
    IDX_RECORD_SIZE as IF_IDX_SIZE,
    LANG_DOMAINS_V3,
    LANG_ID_TO_NAME_V3,
    STYLE_DOMAINS_V3,
    STYLE_ID_TO_NAME_V3,
    StaticGridBatch,
    decode_fixed_point_loss_mask,
)
from utils import chat_protocol


# =============================================================================
# 1. Unified Protocol Constants & Stage-2 Framed Validation
# =============================================================================


def test_unified_protocol_constants_ssot() -> None:
    expected_langs = {
        "en": 0,
        "zh": 1,
        "formal": 2,
        "python": 3,
        "sql_struct": 4,
        "javascript": 5,
    }
    expected_lang_names = {v: k for k, v in expected_langs.items()}
    expected_styles = {
        "rigorous": 0,
        "concise_memo": 1,
        "expository": 2,
        "literary": 3,
        "strict_eng": 4,
    }
    expected_style_names = {v: k for k, v in expected_styles.items()}

    # SSOT parity across interfaces.py, config.py, chat_protocol.py, cognitive_v2_sources.py, ragged_dataloader.py
    assert IF_IDX_FMT == IDX_RECORD_FMT == dcpa_config.IDX_RECORD_FMT == "<QIBBBB"
    assert IF_IDX_SIZE == IDX_RECORD_SIZE == dcpa_config.IDX_RECORD_SIZE == struct.calcsize("<QIBBBB") == 16

    assert LANG_DOMAINS_V3 == expected_langs
    assert dcpa_config.LANG_DOMAINS == expected_langs
    assert chat_protocol.LANG_DOMAINS == expected_langs

    assert LANG_ID_TO_NAME_V3 == expected_lang_names
    assert dcpa_config.LANG_ID_TO_NAME == expected_lang_names
    assert chat_protocol.LANG_ID_TO_NAME == expected_lang_names
    assert COGNITIVE_V2_DOMAIN_NAMES == expected_lang_names

    assert STYLE_DOMAINS_V3 == expected_styles
    assert dcpa_config.STYLE_DOMAINS == expected_styles
    assert chat_protocol.STYLE_DOMAINS == expected_styles

    assert STYLE_ID_TO_NAME_V3 == expected_style_names
    assert dcpa_config.STYLE_ID_TO_NAME == expected_style_names
    assert chat_protocol.STYLE_ID_TO_NAME == expected_style_names

    # Backwards-compatible aliases preserved in lookup tables
    for alias, lid in (("py", 3), ("js", 5), ("sql", 4), ("math", 2), ("ja", 2), ("rs", 4)):
        assert chat_protocol.LANG_NAME_TO_ID[alias] == lid
    for alias, sid in (("neutral", 0), ("memo", 1), ("narrative", 3), ("strict_code_tool", 4)):
        assert chat_protocol.STYLE_NAME_TO_ID[alias] == sid

    # Allowed protocol control bytes parity & Stage-2 framed validation
    assert chat_protocol.ALLOWED_PROTOCOL_CTRL_BYTES == IF_ALLOWED_CTRL
    assert chat_protocol.ALLOWED_PROTOCOL_CTRL_BYTES == frozenset({0x04, 0x11, 0x12, 0x13, 0x14, 0x1D, 0x1E, 0x1F})

    good_frame = (
        b"\x14page\n\x1eU\nSolve x+1=3\n\x1fA\n"
        b"\x1dS\n" + (b"x" * 260) + b"\x1dE\n"
        b"\x11C\ncalc('3-1')\x13\n\x12O\n2\n\x1fA\n2\x04\n"
    )
    ok, msg = chat_protocol.validate_framed_record(good_frame)
    assert ok, f"Valid framed record rejected: {msg}"

    # Over-260B scratchpad, unclosed scratchpad, and disallowed controls rejected
    bad_over = b"\x1dS\n" + (b"a" * 261) + b"\x1dE\n"
    assert not chat_protocol.validate_framed_record(bad_over)[0]
    assert not chat_protocol.validate_framed_record(b"\x1dS\nunclosed")[0]
    assert not chat_protocol.validate_framed_record(b"oops\x1dE\n")[0]
    assert not chat_protocol.validate_framed_record(b"bad\x07bell")[0]
    assert not chat_protocol.validate_framed_record(b"bad\x7fdel")[0]


# =============================================================================
# 2. Syntax- and Paragraph-Aligned Lossless Pre-Chunkers
# =============================================================================


def test_lossless_prechunking_en_zh_prose() -> None:
    # Long multi-paragraph & long single-paragraph English document
    en_paras = [
        " ".join(
            f"Sentence {p}_{s} verifies that decimal values like 3.14159 remain intact and boundaries respect punctuation."
            for s in range(12)
        )
        for p in range(6)
    ]
    en_doc = "\n\n".join(en_paras)
    assert len(en_doc.encode("utf-8")) > 6000

    en_chunks = chunk_prose_lossless(en_doc, max_chunk_bytes=900)
    assert len(en_chunks) >= 6
    reconstructed_en = ""
    for ch in en_chunks:
        assert 0 < len(ch) <= 900
        txt = ch.decode("utf-8", errors="strict")
        stripped = txt.rstrip()
        # Every chunk ends strictly on sentence-ending punctuation (never mid-sentence)
        assert stripped[-1] in ".!?;", f"EN chunk ended mid-sentence: {stripped[-40:]!r}"
        assert "3." not in stripped[-3:], f"Decimal split detected: {stripped[-20:]!r}"
        reconstructed_en += stripped + " "
    # Lossless: every single sentence marker survives across emitted chunks
    for p in range(6):
        for s in range(12):
            assert f"Sentence {p}_{s}" in reconstructed_en

    # Long Chinese document with multibyte UTF-8 characters (3B Hanzi + 4B supplementary) and CJK punctuation
    zh_paras = [
        "".join(
            f"第{p}段第{s}句讨论了结构守恒与长程推演的边界条件，包含扩展汉字𫛭并确保绝不在多字节字符中间切断。"
            if s % 3 == 0
            else (
                f"第{p}段第{s}句进一步检验感叹号切分是否完整！"
                if s % 3 == 1
                else f"第{p}段第{s}句追问分号与问号是否保持语段连贯？"
            )
            for s in range(10)
        )
        for p in range(8)
    ]
    zh_doc = "\n\n".join(zh_paras)
    assert len(zh_doc.encode("utf-8")) > 5000

    zh_chunks = chunk_prose_lossless(zh_doc, max_chunk_bytes=800)
    assert len(zh_chunks) >= 6
    reconstructed_zh = ""
    for ch in zh_chunks:
        assert 0 < len(ch) <= 800
        txt = ch.decode("utf-8", errors="strict")
        stripped = txt.rstrip()
        assert stripped[-1] in "。！？；.!?;", f"ZH chunk ended mid-sentence: {stripped[-24:]!r}"
        reconstructed_zh += stripped
    for p in range(8):
        for s in range(10):
            assert f"第{p}段第{s}句" in reconstructed_zh


def test_lossless_prechunking_python_js_sql_balanced_with_zero_mask_headers() -> None:
    validator = CodeASTValidator()

    # --- Long Python file (~6.5 KB) with imports, decorators, nested defs, docstrings with fake brackets ---
    py_header = "import math\nfrom dataclasses import dataclass\nfrom typing import List, Dict\n\n"
    py_blocks = []
    for i in range(18):
        py_blocks.append(
            f"@dataclass\n"
            f"class MetricNode_{i}:\n"
            f'    """Unbalanced docstring ((( [ {{ ignored safely for node {i}."""\n'
            f"    value: int = {i}\n\n"
            f"def compute_metric_{i}(items: List[int]) -> Dict[str, int]:\n"
            f"    # comment with ] ) }} ignored\n"
            f"    def _inner(x: int) -> int:\n"
            f"        return (x * x) + {i}\n"
            f"    vals = [_inner(v) for v in items if (v > 0)]\n"
            f"    return {{'idx': {i}, 'total': sum(vals), 'count': len(vals)}}\n\n"
        )
    py_code = py_header + "".join(py_blocks)
    assert len(py_code.encode("utf-8")) > 5000

    py_chunks = chunk_code_lossless(py_code, lang_id=3, max_chunk_bytes=1600, max_header_bytes=256)
    assert len(py_chunks) >= 4
    for idx, pc in enumerate(py_chunks):
        assert 0 < len(pc.raw_bytes) <= 1600
        assert len(pc.loss_mask_bytes) == len(pc.raw_bytes)
        txt = pc.raw_bytes.decode("utf-8", errors="strict")
        lex_ok, lex_depth, _ = scan_code_lexical(pc.raw_bytes, 3)
        ast_ok, ast_depth = validator.validate("python", pc.raw_bytes)
        assert lex_ok and ast_ok, f"Python chunk {idx} failed AST/lexical validation:\n{txt}"
        assert 1 <= lex_depth <= 16 and 1 <= ast_depth <= 16
        assert pc.max_depth == lex_depth
        if idx == 0:
            assert pc.header_byte_len == 0
        else:
            assert 0 < pc.header_byte_len <= 256
            assert pc.raw_bytes[: pc.header_byte_len].startswith(b"import math\n")
            assert all(b == 0 for b in pc.loss_mask_bytes[: pc.header_byte_len])
            assert any(b > 0 for b in pc.loss_mask_bytes[pc.header_byte_len :])
    # Verify all 18 functions survived losslessly
    joined_py = "\n".join(pc.raw_bytes.decode("utf-8") for pc in py_chunks)
    for i in range(18):
        assert f"def compute_metric_{i}(" in joined_py

    # --- Long JavaScript file (~5 KB) with imports, template literals `${...}`, block comments ---
    js_header = "import { format } from 'node:util';\nconst assert = require('node:assert');\n\n"
    js_blocks = []
    for i in range(24):
        js_blocks.append(
            f"export function transformBatch_{i}(rows) {{\n"
            f"  /* block comment with unmatched (( [ ignored */\n"
            f"  // line comment with ] }} ignored\n"
            f"  return rows.map((r) => `row_{i}=${{String([r.id, r.val + {i}])}}`);\n"
            f"}}\n\n"
        )
    js_code = js_header + "".join(js_blocks)
    assert len(js_code.encode("utf-8")) > 4000

    js_chunks = chunk_code_lossless(js_code, lang_id=5, max_chunk_bytes=1400, max_header_bytes=256)
    assert len(js_chunks) >= 3
    for idx, pc in enumerate(js_chunks):
        assert 0 < len(pc.raw_bytes) <= 1400
        assert len(pc.loss_mask_bytes) == len(pc.raw_bytes)
        pc.raw_bytes.decode("utf-8", errors="strict")
        lex_ok, lex_depth, _ = scan_code_lexical(pc.raw_bytes, 5)
        ast_ok, ast_depth = validator.validate("javascript", pc.raw_bytes)
        assert lex_ok and ast_ok, f"JS chunk {idx} failed AST/lexical validation"
        assert 1 <= lex_depth <= 16 and 1 <= ast_depth <= 16
        if idx > 0:
            assert 0 < pc.header_byte_len <= 256
            assert all(b == 0 for b in pc.loss_mask_bytes[: pc.header_byte_len])
            assert any(b > 0 for b in pc.loss_mask_bytes[pc.header_byte_len :])
    joined_js = "\n".join(pc.raw_bytes.decode("utf-8") for pc in js_chunks)
    for i in range(24):
        assert f"export function transformBatch_{i}(" in joined_js

    # --- Long SQL script (~4.5 KB) with CREATE TABLE schema header + CTEs + escaped quotes ---
    sql_header = "CREATE TABLE ledger (id BIGINT PRIMARY KEY, region TEXT, cents BIGINT);\n\n"
    sql_blocks = []
    for i in range(24):
        sql_blocks.append(
            f"-- Query {i} with unmatched ( in comment\n"
            f"WITH bucket_{i} AS (\n"
            f"    SELECT id, cents FROM ledger WHERE region <> 'it''s (skip)' AND cents >= {i * 100}\n"
            f")\n"
            f"SELECT COUNT(*), SUM(cents) FROM bucket_{i};\n\n"
        )
    sql_code = sql_header + "".join(sql_blocks)
    assert len(sql_code.encode("utf-8")) > 3500

    sql_chunks = chunk_code_lossless(sql_code, lang_id=4, max_chunk_bytes=1200, max_header_bytes=256)
    assert len(sql_chunks) >= 3
    for idx, pc in enumerate(sql_chunks):
        assert 0 < len(pc.raw_bytes) <= 1200
        assert len(pc.loss_mask_bytes) == len(pc.raw_bytes)
        pc.raw_bytes.decode("utf-8", errors="strict")
        lex_ok, lex_depth, _ = scan_code_lexical(pc.raw_bytes, 4)
        assert lex_ok and lex_depth <= 16
        if idx > 0:
            assert 0 < pc.header_byte_len <= 256
            assert b"CREATE TABLE ledger" in pc.raw_bytes[: pc.header_byte_len]
            assert all(b == 0 for b in pc.loss_mask_bytes[: pc.header_byte_len])
            assert any(b > 0 for b in pc.loss_mask_bytes[pc.header_byte_len :])


# =============================================================================
# 3. Fixed-Point x10 `.loss_mask.bin` Encoding & Decoding
# =============================================================================


def test_fixed_point_loss_mask_encode_decode_and_v3_corpus(tmp_path: Path) -> None:
    sample = b"import os\ndef f(x): return [x];\n# anchor_target\n"
    hdr_end = len(b"import os\n")
    anc_start = sample.index(b"anchor_target")
    anc_end = anc_start + len(b"anchor_target")

    encoded = encode_fixed_point_loss_mask(
        sample,
        zero_spans=[(0, hdr_end)],
        anchor_spans=[(anc_start, anc_end)],
        highlight_syntax_delims=True,
    )
    assert len(encoded) == len(sample)
    arr = np.frombuffer(encoded, dtype=np.uint8)
    assert np.all(arr[:hdr_end] == LOSS_MASK_IGNORE)
    assert np.all(arr[anc_start:anc_end] == LOSS_MASK_ANCHOR_X10)

    # Check supervised body bytes: structural delimiters -> 15, normal bytes -> 10
    for pos in range(hdr_end, anc_start):
        expected_v = LOSS_MASK_SYNTAX_X10 if sample[pos] in SYNTAX_DELIM_BYTES else LOSS_MASK_NORMAL_X10
        assert int(arr[pos]) == expected_v

    # Legacy `1` compatibility & exact parity between torch `decode_fixed_point_loss_mask` and numpy LUT
    all_u8 = torch.arange(256, dtype=torch.uint8)
    dec_torch = decode_fixed_point_loss_mask(all_u8)
    assert torch.allclose(dec_torch, torch.from_numpy(_LOSS_MASK_DECODE_LUT), atol=0.0)
    probe = torch.tensor(
        [LOSS_MASK_IGNORE, LOSS_MASK_LEGACY_ONE, LOSS_MASK_NORMAL_X10, LOSS_MASK_SYNTAX_X10, LOSS_MASK_ANCHOR_X10],
        dtype=torch.uint8,
    )
    assert torch.allclose(decode_fixed_point_loss_mask(probe), torch.tensor([0.0, 1.0, 1.0, 1.5, 3.0]))

    # Upgrading legacy `base_mask` of 0/1 produces 0/10/15/30 cleanly
    legacy_base = bytes([0] * hdr_end + [1] * (len(sample) - hdr_end))
    upgraded = encode_fixed_point_loss_mask(sample, base_mask=legacy_base, anchor_spans=[(anc_start, anc_end)])
    assert upgraded == encoded

    # End-to-end offline v3_flat build writes `.bin`, `.idx`, and `.loss_mask.bin` in lockstep
    out_dir = tmp_path / "v3_flat_smoke"
    meta = build_corpus(out_dir=out_dir, smoke_quota=True, v3_flat=True, offline=True, workers=2, seed=2026)
    assert meta["v3_flat"] is True
    assert meta["loss_mask_encoding"] == "fixed_point_x10"
    ok_tr, msg_tr = verify_shard_triplet(
        out_dir / "train.bin", out_dir / "train.idx", out_dir / "train.loss_mask.bin"
    )
    ok_va, msg_va = verify_shard_triplet(
        out_dir / "val.bin", out_dir / "val.idx", out_dir / "val.loss_mask.bin"
    )
    assert ok_tr, msg_tr
    assert ok_va, msg_va
    msk_probe = np.memmap(out_dir / "train.loss_mask.bin", dtype=np.uint8, mode="r")[:4096]
    uniq = set(np.unique(msk_probe).tolist())
    assert uniq.issubset({0, 10, 15, 30})
    assert 10 in uniq and 15 in uniq
    del msk_probe


# =============================================================================
# 4. StaticGridByteDataset over 200+ Batches (FFD Packing & < 6% Pad Waste)
# =============================================================================


def _build_mixed_v3_test_shard(shard_dir: Path) -> tuple[Path, Path]:
    """Create a realistic 6-domain mixed-length shard (100 B .. 3200 B) including pre-chunked code & anchor masks."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    bin_path = shard_dir / "mix.bin"
    idx_path = shard_dir / "mix.idx"
    msk_path = shard_dir / "mix.loss_mask.bin"

    docs: list[tuple[bytes, bytes, int, int, int, int]] = []
    # Mix of short, medium, and long pre-chunked docs across all 6 domains
    for i in range(60):
        # 0: EN prose (varied lengths 150B .. 2400B)
        n_rep = (i % 9) + 1
        en_txt = "\n\n".join(
            f"Paragraph {k} of essay {i}: Bounded working memory maintains invariants across {k * 17 + i} steps."
            for k in range(n_rep * 2)
        )
        for pc in prechunk_document(en_txt, lang_id=0, max_chunk_bytes=3600):
            docs.append((pc.raw_bytes, pc.loss_mask_bytes, 0, 41, 0, i % 4))

        # 1: ZH prose (multibyte UTF-8)
        zh_txt = "\n\n".join(
            f"样本{i}第{k}段：无损切块与静态网格装箱确保中文多字节字符𫛭绝不被截断，并保持语段连贯。"
            for k in range((i % 7) + 1)
        )
        for pc in prechunk_document(zh_txt, lang_id=1, max_chunk_bytes=3600):
            docs.append((pc.raw_bytes, pc.loss_mask_bytes, 1, 43, 0, (i + 1) % 4))

        # 2: Formal math with an explicit anchor span (30 -> 3.0x)
        fm_txt = f"Lemma {i}: ∀x (P(x) -> Q(x)). Proof: direct deduction yields Q(a_{i}) with bound {i * 3 + 1}.\n"
        fm_b = fm_txt.encode("utf-8")
        anc_s = fm_b.index(b"Proof:")
        fm_m = encode_fixed_point_loss_mask(fm_b, anchor_spans=[(anc_s, anc_s + 6)])
        docs.append((fm_b, fm_m, 2, 48, 0, 0))

        # 3: Long Python file pre-chunked into continuation sub-chunks with 0-masked header
        py_src = "import math\nfrom typing import List\n\n" + "".join(
            f"def py_fn_{i}_{j}(xs: List[int]) -> int:\n    return sum([(v + {j}) for v in xs if (v >= 0)])\n\n"
            for j in range((i % 10) + 2)
        )
        for pc in prechunk_document(py_src, lang_id=3, max_chunk_bytes=1800):
            docs.append((pc.raw_bytes, pc.loss_mask_bytes, 3, 49, pc.max_depth, 4))

        # 4: SQL / JSON structured
        sql_src = (
            f"CREATE TABLE t_{i} (id INT PRIMARY KEY, v INT);\n"
            + "".join(f"SELECT id, (v + {j}) FROM t_{i} WHERE v > {j};\n" for j in range((i % 6) + 1))
        )
        for pc in prechunk_document(sql_src, lang_id=4, max_chunk_bytes=1800):
            docs.append((pc.raw_bytes, pc.loss_mask_bytes, 4, 33, pc.max_depth, 4))

        # 5: JavaScript ES6+
        js_src = "import assert from 'node:assert';\n\n" + "".join(
            f"export function jsFn_{i}_{j}(arr) {{ return arr.map((x) => [x, x + {j}]); }}\n\n"
            for j in range((i % 8) + 2)
        )
        for pc in prechunk_document(js_src, lang_id=5, max_chunk_bytes=1800):
            docs.append((pc.raw_bytes, pc.loss_mask_bytes, 5, 50, pc.max_depth, 4))

    # Also append one intentionally oversized (> 4096 B) unchunked Python doc to test on-the-fly lossless splitting
    huge_py = "import sys\n\n" + "".join(
        f"def huge_func_{k}(a: int, b: int) -> int:\n    return (a * {k + 1}) + (b - {k})\n\n"
        for k in range(80)
    )
    huge_b = huge_py.encode("utf-8")
    assert len(huge_b) > 5000
    docs.append((huge_b, encode_fixed_point_loss_mask(huge_b), 3, 50, 2, 4))

    offset = 0
    with bin_path.open("wb") as fb, idx_path.open("wb") as fi, msk_path.open("wb") as fm:
        for raw_b, msk_b, lid, sid, depth, sty in docs:
            fb.write(raw_b)
            fm.write(msk_b)
            fi.write(struct.pack(IDX_RECORD_FMT, offset, len(raw_b), lid, sid, depth, sty))
            offset += len(raw_b)

    ok, msg = verify_shard_triplet(bin_path, idx_path, msk_path)
    assert ok, msg
    return bin_path, idx_path


def test_static_grid_dataloader_200_batches_and_pad_waste(tmp_path: Path) -> None:
    bin_path, idx_path = _build_mixed_v3_test_shard(tmp_path / "mixed_shard")

    P, T = 32, 4096
    n_batches = 210
    loader = build_static_grid_dataloader(
        bin_path=bin_path,
        idx_path=idx_path,
        grid_lanes=P,
        grid_bytes=T,
        doc_meta_cap_per_lane=64,
        num_batches=n_batches,
        num_workers=0,
        seed=2026,
    )

    batches_seen = 0
    total_slots = 0
    total_pad_bytes = 0
    total_docs_checked = 0
    code_docs_checked = 0
    seen_weights: set[float] = set()

    for batch in loader:
        assert isinstance(batch, StaticGridBatch)
        batch.validate()
        assert batch.x_bytes.shape == (P, T)
        assert batch.doc_meta.shape == (P * 64, 3)

        x_np = batch.x_bytes.numpy()
        did_np = batch.doc_id_byte.numpy()
        ds_np = batch.doc_start_byte.numpy()
        de_np = batch.doc_end_byte.numpy()
        lw_np = batch.loss_weight.numpy()
        meta_np = batch.doc_meta.numpy()

        pad_mask = did_np < 0
        total_slots += P * T
        total_pad_bytes += int(pad_mask.sum())

        # Lane-tail padding invariants
        assert np.all(x_np[pad_mask] == 0), "Non-zero byte in padded lane tail"
        assert np.all(lw_np[pad_mask] == 0.0), "Non-zero loss_weight in padded lane tail"
        assert not np.any(ds_np[pad_mask]), "doc_start_byte True in padded lane tail"
        assert not np.any(de_np[pad_mask]), "doc_end_byte True in padded lane tail"

        if batches_seen < 15:
            for v in np.unique(np.round(lw_np, 2)).tolist():
                seen_weights.add(float(v))

        # Deep per-document verification on first 20 batches + strided spot checks across all 210 batches
        do_deep_check = (batches_seen < 20) or (batches_seen % 15 == 0)
        if do_deep_check:
            prev_max_id = -1
            for p in range(P):
                lane_ids = did_np[p]
                valid_len = int(np.count_nonzero(lane_ids >= 0))
                # Valid bytes must form a single contiguous prefix [0 : valid_len]
                assert np.all(lane_ids[:valid_len] >= 0)
                assert np.all(lane_ids[valid_len:] == -1)
                if valid_len == 0:
                    continue
                # doc_id_byte monotonically non-decreasing within & across lanes
                assert int(lane_ids[0]) == prev_max_id + 1
                assert np.all(np.diff(lane_ids[:valid_len]) >= 0)
                prev_max_id = int(lane_ids[valid_len - 1])

                starts = np.nonzero(ds_np[p, :valid_len])[0]
                ends = np.nonzero(de_np[p, :valid_len])[0]
                assert len(starts) == len(ends) and len(starts) > 0
                assert int(starts[0]) == 0 and int(ends[-1]) == valid_len - 1
                for s_i, e_i in zip(starts.tolist(), ends.tolist()):
                    assert s_i <= e_i
                    doc_id = int(lane_ids[s_i])
                    assert np.all(lane_ids[s_i : e_i + 1] == doc_id)
                    doc_bytes = x_np[p, s_i : e_i + 1].tobytes()
                    # Zero UTF-8 boundary splits
                    doc_bytes.decode("utf-8", errors="strict")

                    lid = int(meta_np[doc_id, 0])
                    sty = int(meta_np[doc_id, 1])
                    assert 0 <= lid <= 5
                    assert 0 <= sty <= 4
                    if lid in (3, 4, 5):
                        lex_ok, depth, _ = scan_code_lexical(doc_bytes, lid)
                        assert lex_ok and depth <= 16, (
                            f"Batch {batches_seen} lane {p} doc {doc_id} (lang={lid}) truncated/unbalanced!"
                        )
                        code_docs_checked += 1
                    total_docs_checked += 1

        batches_seen += 1

    avg_pad_waste = total_pad_bytes / float(total_slots)
    assert batches_seen == n_batches
    assert total_docs_checked > 500
    assert code_docs_checked > 200
    assert {0.0, 1.0, 1.5, 3.0}.issubset(seen_weights), f"Expected 0.0/1.0/1.5/3.0 weights, saw {seen_weights}"
    assert avg_pad_waste < 0.06, f"Average lane-tail pad waste {avg_pad_waste * 100:.2f}% >= 6.0%"

    # Also verify `return_dict=True` roundtrip to StaticGridBatch
    ds_dict = StaticGridByteDataset(bin_path, idx_path, grid_lanes=4, grid_bytes=1024, num_batches=2, return_dict=True)
    raw_d = next(iter(ds_dict))
    assert isinstance(raw_d, dict)
    static_grid_batch_from_dict(raw_d).validate()

    print(
        f"\n[WS-1 StaticGrid] batches={batches_seen} | avg_pad_waste={avg_pad_waste * 100:.2f}% (<6%) | "
        f"docs_audited={total_docs_checked} (code={code_docs_checked}, 0 UTF-8/bracket errors)"
    )


# =============================================================================
# 5. 2025-2026 High-Purity Sources & `v3_flat_45g` / `v3_flat_90g` Profiles
# =============================================================================


def test_v3_flat_source_registry_and_prepare_vast_profiles(tmp_path: Path) -> None:
    expected_repos = {
        "nvidia/Nemotron-CC-v2.1",
        "epfml/FineWeb2-HQ",
        "opencsg/chinese-fineweb-edu-v2",
        "opencsg/Fineweb-Edu-Chinese-V2.2",
        "opencsg/chinese-cosmopedia",
        "BAAI/CCI3-HQ",
        "BAAI/Infinity-Instruct",
        "nvidia/Nemotron-CC-Math-v1",
        "tokyotech-llm/swallow-code-v2",
        "nvidia/Nemotron-CC-Code-v1",
        "nvidia/OpenCodeReasoning",
        "nvidia/OpenCodeReasoning-2",
    }
    assert expected_repos.issubset(set(V3_FLAT_SOURCE_DESCRIPTORS.keys()))
    assert expected_repos.issubset(set(SOURCE_IDS.keys()))
    for repo in expected_repos:
        desc = V3_FLAT_SOURCE_DESCRIPTORS[repo]
        assert 0 <= desc.lang_id <= 5
        assert desc.stage_tier in (1, 2, 3)

    # All existing profiles preserved alongside new v3_flat_45g and v3_flat_90g
    for legacy_prof in ("smoke", "v1_fast", "v2_stem_10g", "v2_full_40g", "v2_cognitive_3g", "v2_cognitive_5g"):
        assert legacy_prof in PROFILES

    for prof_name, exp_scale in (("v3_flat_45g", 90.0), ("v3_flat_90g", 180.0)):
        assert prof_name in PROFILES
        plan = prepare_vast_corpus(profile=prof_name, nvme_root=tmp_path / prof_name, dry_run=True)
        assert plan["dry_run"] is True
        assert plan["profile"] == prof_name
        assert plan["v3_flat"] is True
        assert expected_repos.issubset(set(plan["v3_sources"]))
        corpus_cmd = plan["commands"]["corpus_5domain"]
        assert "--v3-flat" in corpus_cmd
        assert "--write-loss-mask" in corpus_cmd
        assert "--cognitive-v2" in corpus_cmd
        assert str(exp_scale) in corpus_cmd
        assert "cap_code" in plan["commands"]
        assert "stage5_sft" in plan["commands"]
