"""Five-domain raw UTF-8 byte corpus builder for DCPA (`en`, `zh`, `ja`, `python`, `rust`).

Implements Deliverable 1 & Global Override Iron Law #2:
  - Streams natively from HuggingFace Parquet / HTTP tar.gz sources (`streaming=True`,
    zero legacy `.py` scripts, zero `trust_remote_code=True`, local cache << 2 GB).
  - Strictly zero tokenizers, zero `0x00` padding, zero pseudo `<BOS>`/`<EOS>` bytes.
  - Uniform NFC normalization, CRLF/CR -> LF, rejects samples containing disallowed
    ASCII control characters (< 32 except `\\n`, `\\t`, or 0x7F) or `\\ufffd`.
  - Enforces 4-part `tree-sitter >= 0.23` AST zero-error & 1..16 bracket-depth hard gate
    on Python (`lang_id=3`) and Rust (`lang_id=4`) code (with automatic shallow git-clone
    fallback to `serde`/`tokio`/`rayon` if streaming Rust falls short of quota).
  - Global 64-bit `blake2b` deduplication & guaranteed leak-free Train / Val split.
  - Writes zero-copy contiguous `.bin` + 16-byte little-endian `.idx` (`<QIBBH`:
    `offset:<u8, length:<u4, lang_id:<u1, source_id:<u1, max_depth:<u2`) +
    `train.meta.json`, `val.meta.json`, and `corpus_meta.json`.
  - Default production build (~430 MB Train + 8 MB Val = 438 MB) is for Colab;
    local M3 Pro verification MUST use `--smoke-quota` (2 MB per lang_id = ~10 MB total).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.ragged_dataloader import IDX_RECORD_FMT, IDX_RECORD_SIZE, utf8_safe_end

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

# Domain mapping across the 6 embedding slots (`num_lang_domains = 6`):
#   Active v2 4-domain core: 0=en, 1=zh, 3=python, 5=javascript
#   Legacy slots retained for enum stability (0% active quota): 2=ja, 4=rust
DOMAIN_NAMES: dict[int, str] = {
    0: "en",
    1: "zh",
    2: "ja",
    3: "python",
    4: "rust",
    5: "javascript",
}
ACTIVE_DOMAIN_IDS: tuple[int, ...] = (0, 1, 3, 5)

# Full production base quotas (~440 MB Train + 8 MB Val = 448 MB @ scale_factor=1.0; ~10.6 GB @ 24x)
FULL_TRAIN_QUOTA_BYTES: dict[int, int] = {
    0: 145 * 1024 * 1024,  # en: 145 MB (STEM + FineWeb-Edu + OpenWebMath + BabyLM)
    1: 185 * 1024 * 1024,  # zh: 185 MB (Fineweb-Edu-Chinese V2.1 4-5 + V3)
    3: 55 * 1024 * 1024,   # python: 55 MB (AST-gated tiny-codes + idiomatic git repos)
    5: 55 * 1024 * 1024,   # javascript: 55 MB (AST-gated tiny-codes + idiomatic ES6+ git repos)
}
FULL_VAL_QUOTA_BYTES: dict[int, int] = {
    0: 2 * 1024 * 1024,    # en: 2 MB
    1: 2 * 1024 * 1024,    # zh: 2 MB
    3: 2 * 1024 * 1024,    # python: 2 MB
    5: 2 * 1024 * 1024,    # javascript: 2 MB
}

# Local M3 Pro smoke quotas (Iron Law #2: 2 MB per active lang_id -> ~8 MB total)
SMOKE_BYTES_PER_LANG: int = 2 * 1024 * 1024
SMOKE_TRAIN_QUOTA_BYTES: dict[int, int] = {lid: int(1.8 * 1024 * 1024) for lid in ACTIVE_DOMAIN_IDS}
SMOKE_VAL_QUOTA_BYTES: dict[int, int] = {lid: int(0.2 * 1024 * 1024) for lid in ACTIVE_DOMAIN_IDS}

# Sub-source numeric IDs stored in `.idx` `source_id` (<u1)
SOURCE_IDS: dict[str, int] = {
    "nilq/babylm-10M": 1,
    "roneneldan/TinyStories": 2,
    "opencsg/Fineweb-Edu-Chinese-V2.1:4_5": 3,
    "opencsg/Fineweb-Edu-Chinese-V3": 4,
    "Helsinki-NLP/opus-100:en-ja(mono)": 5,
    "Helsinki-NLP/opus-100:en-ja(pair)": 6,
    "Helsinki-NLP/opus-100:en-zh(pair)": 7,
    "Helsinki-NLP/tatoeba_mt": 8,
    "nampdn-ai/tiny-codes": 9,
    "TinyPixel/tiny-codes": 10,
    "git:serde-tokio-rayon": 11,
    "HuggingFaceFW/fineweb-edu:sample-10BT(>=4)": 12,
    "open-web-math/open-web-math": 13,
    "git:rust-ecosystem-extended": 14,
    "git:python-ecosystem": 15,
    "wikimedia/wikipedia:20231101.ja": 16,
    "git:javascript-ecosystem": 17,
}

# Regexes for text hygiene & code fence extraction
_RE_HTML = re.compile(r"<[a-zA-Z/!][^>]{0,120}>")
_RE_STAGE_NOISE = re.compile(
    r"^\s*(\[[^\]]{1,80}\]|\([^)]{1,80}\)|\*[^*]{1,60}\*|=+[^=]{1,60}=+|-{3,}|SCENE\s+\w+.*|ACT\s+\w+.*)\s*$"
)
_RE_DIGITS_ONLY = re.compile(r"^[\d\s\.,:;!?\-+/=%()\[\]{}]+$")
_RE_KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
_RE_PY_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RE_JS_FENCE = re.compile(r"```(?:javascript|js|mjs)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RE_RS_FENCE = re.compile(r"```(?:rust|rs)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RE_ANY_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
_RE_SFT_TAG = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|^\s*(?:USER|ASSISTANT|SYSTEM)\s*:", re.MULTILINE)

# Valid Japanese sentence-ending characters (verbs/auxiliaries/punctuation)
_JA_VALID_ENDINGS = (
    "。", "！", "？", "!", "?", "た", "だ", "る", "す", "い", "く", "ぐ", "ぶ", "む", "ぬ",
    "う", "つ", "ね", "よ", "か", "な", "わ", "ぞ", "さ", "ん", "」", "』",
)

# Tree-sitter node types excluded from bracket balance & depth checking
_STRING_AND_COMMENT_NODE_TYPES: frozenset[str] = frozenset({
    # Python & JavaScript
    "string", "string_content", "string_fragment", "template_string", "regex",
    "comment", "interpolated_string",
    # Legacy Rust
    "string_literal", "raw_string_literal", "byte_string_literal",
    "raw_byte_string_literal", "char_literal", "byte_literal",
    "line_comment", "block_comment",
})


def _has_forbidden_chars(text: str) -> bool:
    """Return True if `text` contains U+FFFD or ASCII control chars (<32 except \\n,\\t) or 0x7F."""
    if "\ufffd" in text:
        return True
    for ch in text:
        o = ord(ch)
        if (o < 32 and o not in (9, 10)) or o == 127:
            return True
    return False


def normalize_and_validate_text(text: str) -> Optional[str]:
    """Apply NFC normalization, CRLF/CR -> LF, and reject dirty control/replacement chars."""
    if not text:
        return None
    nfc = unicodedata.normalize("NFC", text)
    unified = nfc.replace("\r\n", "\n").replace("\r", "\n")
    if _has_forbidden_chars(unified):
        return None
    cleaned = unified.strip()
    if not cleaned:
        return None
    return cleaned + "\n"


def _hanzi_ratio(text: str) -> float:
    non_ws = 0
    hanzi = 0
    for ch in text:
        if ch.isspace():
            continue
        non_ws += 1
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF):
            hanzi += 1
    return (hanzi / non_ws) if non_ws > 0 else 0.0


def _split_utf8_paragraphs(text: str, min_bytes: int, max_bytes: int) -> list[bytes]:
    """Chunk coherent paragraphs into valid UTF-8 byte spans within `[min_bytes, max_bytes]`."""
    raw = text.encode("utf-8", errors="strict")
    if len(raw) < min_bytes:
        return []
    if len(raw) <= max_bytes:
        return [raw]

    import numpy as np

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[bytes] = []
    acc_parts: list[str] = []
    acc_bytes = 0

    for para in paragraphs:
        p_b = (para + "\n").encode("utf-8")
        if len(p_b) > max_bytes:
            if acc_bytes >= min_bytes:
                out.append(("\n\n".join(acc_parts) + "\n").encode("utf-8"))
            acc_parts, acc_bytes = [], 0

            # Split oversized paragraph cleanly on UTF-8 boundaries
            arr = np.frombuffer(p_b, dtype=np.uint8)
            start = 0
            while start < len(arr):
                end = utf8_safe_end(arr, start, max_bytes - 1)
                if end <= start:
                    break
                piece = bytes(arr[start:end]).strip(b"\n") + b"\n"
                if min_bytes <= len(piece) <= max_bytes:
                    out.append(piece)
                start = end
            continue

        sep = 2 if acc_parts else 0
        if acc_bytes + sep + len(p_b) > max_bytes and acc_parts:
            chunk = ("\n\n".join(acc_parts) + "\n").encode("utf-8")
            if len(chunk) >= min_bytes:
                out.append(chunk)
            acc_parts = [para]
            acc_bytes = len(p_b)
        else:
            acc_parts.append(para)
            acc_bytes += sep + len(p_b)

    if acc_parts:
        chunk = ("\n\n".join(acc_parts) + "\n").encode("utf-8")
        if len(chunk) >= min_bytes:
            out.append(chunk)
    return out


def validate_js_lexical_brackets(code_bytes: bytes, *, min_bytes: int = 12, max_bytes: int = 8192) -> tuple[bool, int]:
    """Strict ES6+ JavaScript lexical/quote/comment & 1..16 bracket-depth fallback validator."""
    n = len(code_bytes)
    if n < min_bytes or n > max_bytes:
        return False, 0
    try:
        text = code_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, 0
    low = text.lower()
    if "```" in text or "<html" in low or "<!doctype" in low:
        return False, 0
    if not any(kw in text for kw in ("function", "class ", "const ", "let ", "var ", "return", "=>", "export ")):
        return False, 0
    # Reject minified single-line bundles (> 280 bytes/line average)
    lines = text.splitlines()
    if lines and (n / max(len(lines), 1)) > 280:
        return False, 0

    b_stack: list[str] = []
    max_depth = 0
    i = 0
    m = len(text)
    pairs = {")": "(", "]": "[", "}": "{"}
    while i < m:
        ch = text[i]
        nxt = text[i + 1] if (i + 1) < m else ""
        if ch == "/" and nxt == "/":
            nl = text.find("\n", i + 2)
            i = m if nl < 0 else nl + 1
            continue
        if ch == "/" and nxt == "*":
            end_c = text.find("*/", i + 2)
            if end_c < 0:
                return False, 0
            i = end_c + 2
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            closed = False
            while i < m:
                c2 = text[i]
                if c2 == "\\":
                    i += 2
                    continue
                if c2 == "\n":
                    return False, 0
                if c2 == quote:
                    closed = True
                    i += 1
                    break
                i += 1
            if not closed:
                return False, 0
            continue
        if ch == "`":
            i += 1
            closed_t = False
            while i < m:
                c2 = text[i]
                if c2 == "\\":
                    i += 2
                    continue
                if c2 == "`":
                    closed_t = True
                    i += 1
                    break
                i += 1
            if not closed_t:
                return False, 0
            continue
        if ch in "([{":
            b_stack.append(ch)
            if len(b_stack) > max_depth:
                max_depth = len(b_stack)
                if max_depth > 16:
                    return False, max_depth
        elif ch in pairs:
            if not b_stack or b_stack.pop() != pairs[ch]:
                return False, max_depth
        i += 1

    if b_stack or max_depth < 1 or max_depth > 16:
        return False, max_depth
    return True, max_depth


class CodeASTValidator:
    """Strict 4-part `tree-sitter >= 0.23` AST & 1..16 bracket-depth hard gate for Python & JavaScript."""

    def __init__(self) -> None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        self._parsers: dict[str, Any] = {
            "python": Parser(Language(tspython.language())),
        }
        try:
            import tree_sitter_javascript as tsjs

            self._parsers["javascript"] = Parser(Language(tsjs.language()))
            self._parsers["js"] = self._parsers["javascript"]
        except ImportError:
            pass
        try:
            import tree_sitter_rust as tsrust

            self._parsers["rust"] = Parser(Language(tsrust.language()))
        except ImportError:
            pass
        self.tree_sitter_active = True

    def validate(self, lang: str, code_bytes: bytes) -> tuple[bool, int]:
        """Validate all 4 hard gates simultaneously; returns `(is_valid, max_bracket_depth)`."""
        # Gate 1: Byte length in [64, 8192]
        n = len(code_bytes)
        if n < 64 or n > 8192:
            return False, 0

        lang_key = "javascript" if lang == "js" else lang
        parser = self._parsers.get(lang_key)
        if parser is None:
            if lang_key == "javascript":
                return validate_js_lexical_brackets(code_bytes, min_bytes=64, max_bytes=8192)
            raise ValueError(f"Unsupported code language: {lang}")

        # Gate 2: Zero AST errors, zero missing nodes, zero ERROR nodes
        tree = parser.parse(code_bytes)
        root = tree.root_node
        if root.has_error:
            return False, 0

        # Single iterative DFS: verify zero missing/ERROR nodes AND collect string/comment spans
        excluded_spans: list[tuple[int, int]] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_missing or node.type == "ERROR":
                return False, 0
            if node.type in _STRING_AND_COMMENT_NODE_TYPES:
                excluded_spans.append((node.start_byte, node.end_byte))
                # Do not descend into string/comment children
                continue
            children = node.children
            if children:
                stack.extend(children)

        # Merge sorted excluded spans for O(N) bracket scan outside strings/comments
        excluded_spans.sort()
        merged: list[tuple[int, int]] = []
        for s, e in excluded_spans:
            if merged and s <= merged[-1][1]:
                if e > merged[-1][1]:
                    merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Gate 3 & 4: Strict {} () [] bracket balance & 1 <= max_bracket_depth <= 16
        b_stack: list[int] = []
        max_depth = 0
        span_idx = 0
        num_spans = len(merged)

        # ASCII byte codes: {=123, }=125, (=40, )=41, [=91, ]=93
        pos = 0
        while pos < n:
            if span_idx < num_spans:
                sp_s, sp_e = merged[span_idx]
                if pos >= sp_e:
                    span_idx += 1
                    continue
                if pos >= sp_s:
                    pos = sp_e
                    span_idx += 1
                    continue

            b = code_bytes[pos]
            if b == 123:      # {
                b_stack.append(125)
                if len(b_stack) > max_depth:
                    max_depth = len(b_stack)
                    if max_depth > 16:
                        return False, max_depth
            elif b == 40:     # (
                b_stack.append(41)
                if len(b_stack) > max_depth:
                    max_depth = len(b_stack)
                    if max_depth > 16:
                        return False, max_depth
            elif b == 91:     # [
                b_stack.append(93)
                if len(b_stack) > max_depth:
                    max_depth = len(b_stack)
                    if max_depth > 16:
                        return False, max_depth
            elif b in (125, 41, 93):
                if not b_stack or b_stack.pop() != b:
                    return False, max_depth
            pos += 1

        if b_stack or max_depth < 1 or max_depth > 16:
            return False, max_depth
        return True, max_depth

    def is_valid(self, lang: str, code_bytes: bytes) -> bool:
        ok, _ = self.validate(lang, code_bytes)
        return ok


def extract_code_from_tiny_codes(raw_text: str, lang: str) -> Optional[str]:
    """Extract pure Python/JavaScript/Rust code body from Markdown fences (stripping prose/SFT wrappers)."""
    if not raw_text:
        return None
    if lang == "python":
        fence_re = _RE_PY_FENCE
    elif lang in ("javascript", "js"):
        fence_re = _RE_JS_FENCE
    else:
        fence_re = _RE_RS_FENCE
    matches = fence_re.findall(raw_text)
    if matches:
        # Pick the longest fenced block matching the requested language
        best = max(matches, key=len).strip()
        return best if best else None

    # If the row comes from raw `response` without language tag on fence, check generic fence
    if "<|im_start|>" not in raw_text:
        generic = _RE_ANY_FENCE.findall(raw_text)
        if generic:
            best = max(generic, key=len).strip()
            return best if best else None
        # No fence at all: accept candidate directly and let tree-sitter reject non-code prose
        return raw_text.strip()
    return None


def _strip_rust_license_header(code: str) -> str:
    """Strip leading license/copyright block comments from `.rs` source files."""
    lines = code.splitlines()
    i = 0
    n = len(lines)
    in_block = False
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if in_block:
            if "*/" in s:
                in_block = False
            i += 1
            continue
        if s.startswith("/*"):
            if "*/" not in s[2:]:
                in_block = True
            i += 1
            continue
        if s.startswith("//") and not s.startswith("///"):
            low = s.lower()
            if any(k in low for k in ("copyright", "license", "apache", "mit", "spdx", "authors")):
                i += 1
                continue
            break
        break
    return "\n".join(lines[i:]).strip()


def _is_noisy_rust_table(code: str) -> bool:
    """Reject huge auto-generated lookup tables or repetitive macro expansion dumps."""
    lines = code.splitlines()
    if len(lines) > 260:
        return True
    # High density of hex/numeric array entries
    hex_tokens = len(re.findall(r"\b0x[0-9a-fA-F]+\b", code))
    if hex_tokens > 48:
        return True
    return False


# =============================================================================
# Streaming Generators per Domain (Zero Full-Dataset Download!)
# =============================================================================

def _stream_en_babylm() -> Iterator[tuple[bytes, int]]:
    """Stream `nilq/babylm-10M` (Parquet), stripping stage noise & merging short lines >= 64 B."""
    from datasets import load_dataset

    sid = SOURCE_IDS["nilq/babylm-10M"]
    ds = load_dataset("nilq/babylm-10M", split="train", streaming=True)
    it = iter(ds)
    buf_lines: list[str] = []
    buf_bytes = 0
    try:
        for row in it:
            raw = row.get("text")
            if not isinstance(raw, str):
                continue
            for line in raw.replace("\r\n", "\n").replace("\r", "\n").splitlines():
                s = unicodedata.normalize("NFC", line).strip()
                if not s or _RE_STAGE_NOISE.match(s) or _has_forbidden_chars(s):
                    if buf_bytes >= 64:
                        doc = ("\n".join(buf_lines) + "\n").encode("utf-8", errors="strict")
                        yield doc, sid
                        buf_lines, buf_bytes = [], 0
                    continue
                s_len = len(s.encode("utf-8")) + 1
                if buf_bytes + s_len > 2048 and buf_bytes >= 64:
                    doc = ("\n".join(buf_lines) + "\n").encode("utf-8", errors="strict")
                    yield doc, sid
                    buf_lines, buf_bytes = [s], s_len
                else:
                    buf_lines.append(s)
                    buf_bytes += s_len
                    if buf_bytes >= 512:
                        doc = ("\n".join(buf_lines) + "\n").encode("utf-8", errors="strict")
                        yield doc, sid
                        buf_lines, buf_bytes = [], 0
        if buf_bytes >= 64:
            doc = ("\n".join(buf_lines) + "\n").encode("utf-8", errors="strict")
            yield doc, sid
    finally:
        del it, ds
        gc.collect()


def _stream_en_tinystories() -> Iterator[tuple[bytes, int]]:
    """Stream `roneneldan/TinyStories` (Parquet) narrative paragraphs."""
    from datasets import load_dataset

    sid = SOURCE_IDS["roneneldan/TinyStories"]
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    it = iter(ds)
    try:
        for row in it:
            raw = row.get("text")
            if not isinstance(raw, str):
                continue
            cleaned = normalize_and_validate_text(raw)
            if cleaned is None:
                continue
            for chunk in _split_utf8_paragraphs(cleaned, min_bytes=64, max_bytes=4096):
                yield chunk, sid
    finally:
        del it, ds
        gc.collect()


def _stream_zh_fineweb_v21(max_shards: int) -> Iterator[tuple[bytes, int]]:
    """Stream `opencsg/Fineweb-Edu-Chinese-V2.1` high-score `4_5/*.parquet` (or `4-5/*.parquet`)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sid = SOURCE_IDS["opencsg/Fineweb-Edu-Chinese-V2.1:4_5"]
    api = HfApi()
    all_files = api.list_repo_files("opencsg/Fineweb-Edu-Chinese-V2.1", repo_type="dataset")
    parquet_45 = sorted(
        f for f in all_files if f.startswith(("4_5/", "4-5/")) and f.endswith(".parquet")
    )
    if not parquet_45:
        raise RuntimeError("Could not locate 4_5/*.parquet or 4-5/*.parquet in Fineweb-Edu-Chinese-V2.1")

    selected = parquet_45[:max_shards]
    ds = load_dataset(
        "opencsg/Fineweb-Edu-Chinese-V2.1",
        data_files=selected,
        split="train",
        streaming=True,
    )
    it = iter(ds)
    try:
        for row in it:
            raw = row.get("text")
            if not isinstance(raw, str):
                continue
            cleaned = normalize_and_validate_text(raw)
            if cleaned is None:
                continue
            for chunk in _split_utf8_paragraphs(cleaned, min_bytes=120, max_bytes=6000):
                txt = chunk.decode("utf-8", errors="strict")
                if _hanzi_ratio(txt) >= 0.50:
                    yield chunk, sid
    finally:
        del it, ds
        gc.collect()


def _stream_zh_fineweb_v3() -> Iterator[tuple[bytes, int]]:
    """Stream `opencsg/Fineweb-Edu-Chinese-V3` `.tar.gz` archives over HTTP (`r|gz`, zero full download)."""
    from huggingface_hub import HfApi, hf_hub_url

    sid = SOURCE_IDS["opencsg/Fineweb-Edu-Chinese-V3"]
    api = HfApi()
    files = api.list_repo_files("opencsg/Fineweb-Edu-Chinese-V3", repo_type="dataset")
    # Prioritize Chinese-dense technical/textbook tarballs first
    preferred = [
        "csdn_article_sft.tar.gz",
        "高质量文档结构化数据_第一批_CS_sft.tar.gz",
        "高质量文档结构化数据_第一批_NA_sft.tar.gz",
        "高质量文档结构化数据_第一批_SO_sft.tar.gz",
        "人民邮电出版社_sft.tar.gz",
        "机械工业出版社_sft.tar.gz",
        "电子工业出版社_sft.tar.gz",
        "科学出版社_sft.tar.gz",
        "清华大学出版社_sft.tar.gz",
        "北京大学出版社电子教材_sft.tar.gz",
    ]
    tar_files = [f for f in preferred if f in files] + [
        f for f in sorted(files) if f.endswith(".tar.gz") and f not in preferred
    ]

    for tar_name in tar_files:
        url = hf_hub_url("opencsg/Fineweb-Edu-Chinese-V3", tar_name, repo_type="dataset")
        req = urllib.request.Request(url, headers={"User-Agent": "dcpa-corpus-builder/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                with tarfile.open(fileobj=resp, mode="r|gz") as tf:
                    for member in tf:
                        if not member.isfile() or not member.name.endswith((".jsonl", ".json")):
                            continue
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        for raw_line in fobj:
                            try:
                                obj = json.loads(raw_line.decode("utf-8", errors="strict"))
                            except Exception:
                                continue
                            if not isinstance(obj, dict):
                                continue
                            # Concatenate QA / reasoning prose without SFT template tags
                            parts: list[str] = []
                            for key in ("input", "question", "text", "content", "output", "answer"):
                                val = obj.get(key)
                                if isinstance(val, str) and val.strip():
                                    s = _RE_SFT_TAG.sub("", val).strip()
                                    if s:
                                        parts.append(s)
                            if not parts:
                                continue
                            joined = "\n\n".join(parts)
                            cleaned = normalize_and_validate_text(joined)
                            if cleaned is None:
                                continue
                            for chunk in _split_utf8_paragraphs(cleaned, min_bytes=120, max_bytes=6000):
                                txt = chunk.decode("utf-8", errors="strict")
                                if _hanzi_ratio(txt) >= 0.50:
                                    yield chunk, sid
        except Exception as exc:
            print(f"  [WARN] V3 stream {tar_name} interrupted ({type(exc).__name__}: {exc}); advancing.", flush=True)
            continue


def _is_clean_opus_pair(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if _RE_HTML.search(a) or _RE_HTML.search(b):
        return False
    if _RE_DIGITS_ONLY.match(a) or _RE_DIGITS_ONLY.match(b):
        return False
    if _has_forbidden_chars(a) or _has_forbidden_chars(b):
        return False
    la, lb = len(a), len(b)
    if min(la, lb) < 4:
        return False
    if max(la, lb) / max(min(la, lb), 1) > 4.0:
        return False
    return True


def _is_complete_ja_sentence(ja: str) -> bool:
    if len(ja) < 8:
        return False
    if not _RE_KANA.search(ja):
        return False
    return ja.endswith(_JA_VALID_ENDINGS)


def _stream_ja_opus(mode: str) -> Iterator[tuple[bytes, int]]:
    """Stream `Helsinki-NLP/opus-100` (`en-ja` mono, `en-ja` pair, or `en-zh` pair)."""
    from datasets import load_dataset

    if mode in ("ja_mono", "en_ja_pair"):
        cfg_name = "en-ja"
        src_k, tgt_k = "en", "ja"
        sid = SOURCE_IDS[
            "Helsinki-NLP/opus-100:en-ja(mono)" if mode == "ja_mono" else "Helsinki-NLP/opus-100:en-ja(pair)"
        ]
    else:
        cfg_name = "en-zh"
        src_k, tgt_k = "en", "zh"
        sid = SOURCE_IDS["Helsinki-NLP/opus-100:en-zh(pair)"]

    ds = load_dataset("Helsinki-NLP/opus-100", name=cfg_name, split="train", streaming=True)
    it = iter(ds)
    buf: list[str] = []
    buf_bytes = 0
    try:
        for row in it:
            tr = row.get("translation")
            if not isinstance(tr, dict):
                continue
            s_raw = tr.get(src_k, "")
            t_raw = tr.get(tgt_k, "")
            if not isinstance(s_raw, str) or not isinstance(t_raw, str):
                continue
            s_txt = unicodedata.normalize("NFC", s_raw).replace("\r\n", "\n").replace("\r", "\n").strip()
            t_txt = unicodedata.normalize("NFC", t_raw).replace("\r\n", "\n").replace("\r", "\n").strip()
            if "\n" in s_txt or "\n" in t_txt:
                continue
            if not _is_clean_opus_pair(s_txt, t_txt):
                continue

            if mode == "ja_mono":
                if not _is_complete_ja_sentence(t_txt):
                    continue
                unit = t_txt
            elif mode == "en_ja_pair":
                if not _RE_KANA.search(t_txt):
                    continue
                unit = f"{s_txt}\n{t_txt}"
            else:
                if _hanzi_ratio(t_txt) < 0.40:
                    continue
                unit = f"{s_txt}\n{t_txt}"

            u_len = len(unit.encode("utf-8")) + 1
            if buf_bytes + u_len > 1024 and buf_bytes >= 64:
                sep = "\n\n" if mode != "ja_mono" else "\n"
                yield (sep.join(buf) + "\n").encode("utf-8", errors="strict"), sid
                buf, buf_bytes = [unit], u_len
            else:
                buf.append(unit)
                buf_bytes += u_len
                if buf_bytes >= 256:
                    sep = "\n\n" if mode != "ja_mono" else "\n"
                    yield (sep.join(buf) + "\n").encode("utf-8", errors="strict"), sid
                    buf, buf_bytes = [], 0
        if buf_bytes >= 64:
            sep = "\n\n" if mode != "ja_mono" else "\n"
            yield (sep.join(buf) + "\n").encode("utf-8", errors="strict"), sid
    finally:
        del it, ds
        gc.collect()


def _stream_tiny_codes(lang: str) -> Iterator[tuple[str, int]]:
    """Stream Python/JavaScript code candidates from `nampdn-ai/tiny-codes` with fallback `TinyPixel/tiny-codes`."""
    from datasets import load_dataset

    lang_map_cap = {"python": "Python", "javascript": "JavaScript", "js": "JavaScript", "rust": "Rust"}
    target_lang_cap = lang_map_cap.get(lang, "Python")
    repos = ("nampdn-ai/tiny-codes", "TinyPixel/tiny-codes")

    for repo in repos:
        sid = SOURCE_IDS[repo]
        ds = None
        it = None
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            it = iter(ds)
            first = next(it)
        except Exception as exc:
            print(
                f"  [INFO] Code source {repo} unavailable ({type(exc).__name__}); falling back.",
                flush=True,
            )
            del it, ds
            gc.collect()
            continue

        print(f"  -> [{lang}] streaming {repo}...", flush=True)
        try:
            # Yield from first + remaining rows
            def _rows():
                yield first
                yield from it

            for row in _rows():
                if "programming_language" in row:
                    if row.get("programming_language") != target_lang_cap:
                        continue
                    raw = row.get("response") or row.get("code") or ""
                else:
                    raw = row.get("text") or ""
                    low_raw = raw.lower()
                    if lang == "python":
                        if "```python" not in low_raw and "```py\n" not in low_raw:
                            continue
                    elif lang in ("javascript", "js"):
                        if "```javascript" not in low_raw and "```js\n" not in low_raw:
                            continue
                    else:
                        if "```rust" not in low_raw and "```rs\n" not in low_raw:
                            continue
                if not isinstance(raw, str):
                    continue
                code = extract_code_from_tiny_codes(raw, lang)
                if code:
                    yield code, sid
            # Successfully exhausted primary repo
            return
        finally:
            del it, ds
            gc.collect()


def _stream_en_fineweb_edu_stem() -> Iterator[tuple[bytes, int]]:
    """Stream high-score `HuggingFaceFW/fineweb-edu` (`sample-10BT`, score >= 3.8 / int_score >= 4)."""
    from datasets import load_dataset

    sid = SOURCE_IDS["HuggingFaceFW/fineweb-edu:sample-10BT(>=4)"]
    ds = None
    it = None
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        it = iter(ds)
        for row in it:
            int_sc = row.get("int_score")
            sc = row.get("score")
            if isinstance(int_sc, (int, float)):
                if int_sc < 4:
                    continue
            elif isinstance(sc, (int, float)):
                if sc < 3.8:
                    continue
            raw = row.get("text")
            if not isinstance(raw, str):
                continue
            cleaned = normalize_and_validate_text(raw)
            if cleaned is None:
                continue
            for chunk in _split_utf8_paragraphs(cleaned, min_bytes=120, max_bytes=4096):
                yield chunk, sid
    except Exception as exc:
        print(f"  [WARN] fineweb-edu stream interrupted ({type(exc).__name__}: {exc}); falling back.", flush=True)
    finally:
        del it, ds
        gc.collect()


def _stream_en_open_web_math() -> Iterator[tuple[bytes, int]]:
    """Stream STEM mathematical proofs & expositions from `open-web-math/open-web-math`."""
    from datasets import load_dataset

    sid = SOURCE_IDS["open-web-math/open-web-math"]
    ds = None
    it = None
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        it = iter(ds)
        for row in it:
            raw = row.get("text")
            if not isinstance(raw, str):
                continue
            cleaned = normalize_and_validate_text(raw)
            if cleaned is None:
                continue
            for chunk in _split_utf8_paragraphs(cleaned, min_bytes=120, max_bytes=4096):
                yield chunk, sid
    except Exception as exc:
        print(f"  [WARN] open-web-math stream interrupted ({type(exc).__name__}: {exc}); falling back.", flush=True)
    finally:
        del it, ds
        gc.collect()


def _stream_ja_wikipedia() -> Iterator[tuple[bytes, int]]:
    """Extended Japanese prose stream (`wikimedia/wikipedia` 20231101.ja) for multi-GB quotas."""
    from datasets import load_dataset

    sid = SOURCE_IDS["wikimedia/wikipedia:20231101.ja"]
    ds = None
    it = None
    try:
        ds = load_dataset("wikimedia/wikipedia", name="20231101.ja", split="train", streaming=True)
        it = iter(ds)
        for row in it:
            raw = row.get("text")
            if not isinstance(raw, str):
                continue
            cleaned = normalize_and_validate_text(raw)
            if cleaned is None or not _RE_KANA.search(cleaned):
                continue
            for chunk in _split_utf8_paragraphs(cleaned, min_bytes=120, max_bytes=4096):
                txt = chunk.decode("utf-8", errors="strict").strip()
                if _RE_KANA.search(txt) and txt.endswith(_JA_VALID_ENDINGS):
                    yield chunk, sid
    except Exception as exc:
        print(f"  [WARN] ja-wikipedia stream interrupted ({type(exc).__name__}: {exc}); advancing.", flush=True)
    finally:
        del it, ds
        gc.collect()


def _stream_python_git_repos() -> Iterator[tuple[str, int]]:
    """Extended shallow git-clone extractor for idiomatic pure-Python ecosystems."""
    sid = SOURCE_IDS["git:python-ecosystem"]
    repos = (
        "https://github.com/pallets/click.git",
        "https://github.com/pallets/flask.git",
        "https://github.com/psf/requests.git",
        "https://github.com/encode/httpx.git",
        "https://github.com/Textualize/rich.git",
        "https://github.com/pytest-dev/pytest.git",
        "https://github.com/pydantic/pydantic.git",
        "https://github.com/fastapi/fastapi.git",
    )
    tmp_root = Path(tempfile.mkdtemp(prefix="dcpa_py_repos_"))
    try:
        for url in repos:
            name = url.rsplit("/", 1)[-1].removesuffix(".git")
            dest = tmp_root / name
            print(f"  -> [python] shallow cloning {url}...", flush=True)
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                print(f"  [WARN] git clone {url} failed: {res.stderr.strip()}", flush=True)
                continue
            for py_file in sorted(dest.rglob("*.py")):
                try:
                    raw = py_file.read_text(encoding="utf-8", errors="strict")
                except Exception:
                    continue
                stripped = raw.strip()
                if len(stripped) < 64:
                    continue
                yield stripped, sid
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _stream_rust_git_repos(extended: bool = False) -> Iterator[tuple[str, int]]:
    """Shallow git-clone extractor for official `serde`, `tokio`, `rayon` (plus extended crates)."""
    core_repos = (
        ("https://github.com/serde-rs/serde.git", SOURCE_IDS["git:serde-tokio-rayon"]),
        ("https://github.com/tokio-rs/tokio.git", SOURCE_IDS["git:serde-tokio-rayon"]),
        ("https://github.com/rayon-rs/rayon.git", SOURCE_IDS["git:serde-tokio-rayon"]),
    )
    ext_sid = SOURCE_IDS["git:rust-ecosystem-extended"]
    ext_repos = (
        ("https://github.com/BurntSushi/ripgrep.git", ext_sid),
        ("https://github.com/dtolnay/syn.git", ext_sid),
        ("https://github.com/dtolnay/anyhow.git", ext_sid),
        ("https://github.com/clap-rs/clap.git", ext_sid),
        ("https://github.com/crossbeam-rs/crossbeam.git", ext_sid),
        ("https://github.com/hyperium/http.git", ext_sid),
        ("https://github.com/tokio-rs/axum.git", ext_sid),
        ("https://github.com/rust-lang/regex.git", ext_sid),
    )
    repos = core_repos + ext_repos if extended else core_repos
    tmp_root = Path(tempfile.mkdtemp(prefix="dcpa_rust_repos_"))
    try:
        for url, sid in repos:
            name = url.rsplit("/", 1)[-1].removesuffix(".git")
            dest = tmp_root / name
            print(f"  -> [rust] shallow cloning {url}...", flush=True)
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                print(f"  [WARN] git clone {url} failed: {res.stderr.strip()}", flush=True)
                continue
            for rs_file in sorted(dest.rglob("*.rs")):
                try:
                    raw = rs_file.read_text(encoding="utf-8", errors="strict")
                except Exception:
                    continue
                stripped = _strip_rust_license_header(raw)
                if not stripped or _is_noisy_rust_table(stripped):
                    continue
                yield stripped, sid
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _stream_js_git_repos(extended: bool = False) -> Iterator[tuple[str, int]]:
    """Shallow git-clone extractor for idiomatic pure-JavaScript (ES6+) ecosystems."""
    sid = SOURCE_IDS["git:javascript-ecosystem"]
    core_repos = (
        "https://github.com/expressjs/express.git",
        "https://github.com/axios/axios.git",
        "https://github.com/tj/commander.js.git",
        "https://github.com/lodash/lodash.git",
    )
    ext_repos = (
        "https://github.com/chalk/chalk.git",
        "https://github.com/sindresorhus/got.git",
        "https://github.com/fastify/fastify.git",
        "https://github.com/pinojs/pino.git",
        "https://github.com/markedjs/marked.git",
        "https://github.com/date-fns/date-fns.git",
    )
    repos = core_repos + ext_repos if extended else core_repos
    skip_parts = {"node_modules", "dist", "build", "vendor", "coverage", ".git"}
    tmp_root = Path(tempfile.mkdtemp(prefix="dcpa_js_repos_"))
    try:
        for url in repos:
            name = url.rsplit("/", 1)[-1].removesuffix(".git")
            dest = tmp_root / name
            print(f"  -> [javascript] shallow cloning {url}...", flush=True)
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                print(f"  [WARN] git clone {url} failed: {res.stderr.strip()}", flush=True)
                continue
            for js_file in sorted(list(dest.rglob("*.js")) + list(dest.rglob("*.mjs"))):
                if js_file.name.endswith(".min.js") or any(p in skip_parts for p in js_file.parts):
                    continue
                try:
                    raw = js_file.read_text(encoding="utf-8", errors="strict")
                except Exception:
                    continue
                stripped = _strip_rust_license_header(raw)
                if len(stripped) < 64:
                    continue
                yield stripped, sid
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _assert_local_output_dir(out_dir: Path) -> None:
    resolved = str(out_dir.resolve())
    if resolved.startswith(("/content/drive", "/mnt/gdrive")):
        raise ValueError(
            f"Refusing to build binary shards directly on network/FUSE mount ({resolved}). "
            "Build on local NVMe (e.g. /workspace/nvme_cache/data) and archive afterwards."
        )


@dataclass
class _DomainAccum:
    lang_id: int
    name: str
    train_target: int
    val_target: int
    train_bytes: int = 0
    val_bytes: int = 0
    train_docs: int = 0
    val_docs: int = 0
    code_candidates: int = 0
    code_rejected: int = 0
    dup_rejected: int = 0
    sources_used: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sources_used is None:
            self.sources_used = []

    @property
    def done(self) -> bool:
        return self.train_bytes >= self.train_target and self.val_bytes >= self.val_target


def _populate_single_domain(
    lang_id: int,
    acc: _DomainAccum,
    emit_fn: Any,
    validator: CodeASTValidator,
    *,
    smoke_quota: bool,
    scale_factor: float,
    stem_upgrade: bool,
) -> dict[str, Any]:
    """Populate a single `lang_id` (`0..4`) via `emit_fn(acc, doc_bytes, source_id, max_depth)`."""
    t0 = time.perf_counter()
    use_stem = stem_upgrade or (not smoke_quota and scale_factor > 1.5)
    total_target = acc.train_target + acc.val_target

    if lang_id == 0:
        if use_stem:
            babylm_cap = int(0.15 * total_target)
            fw_edu_cap = int(0.80 * total_target)
            print("  -> [en | lang_id=0] streaming nilq/babylm-10M (15% syntax anchor)...", flush=True)
            for doc_b, sid in _stream_en_babylm():
                emit_fn(acc, doc_b, sid, 0)
                if (acc.train_bytes + acc.val_bytes) >= babylm_cap or acc.done:
                    break
            if not acc.done:
                print("  -> [en | lang_id=0] streaming HuggingFaceFW/fineweb-edu (score>=4 STEM 65%)...", flush=True)
                for doc_b, sid in _stream_en_fineweb_edu_stem():
                    emit_fn(acc, doc_b, sid, 0)
                    if (acc.train_bytes + acc.val_bytes) >= fw_edu_cap or acc.done:
                        break
            if not acc.done:
                print("  -> [en | lang_id=0] streaming open-web-math/open-web-math (20% proofs)...", flush=True)
                for doc_b, sid in _stream_en_open_web_math():
                    emit_fn(acc, doc_b, sid, 0)
                    if acc.done:
                        break
        else:
            babylm_cap = int(0.50 * total_target) if smoke_quota else total_target
            print("  -> [en | lang_id=0] streaming nilq/babylm-10M...", flush=True)
            for doc_b, sid in _stream_en_babylm():
                emit_fn(acc, doc_b, sid, 0)
                if (acc.train_bytes + acc.val_bytes) >= babylm_cap or acc.done:
                    break
        if not acc.done:
            print("  -> [en | lang_id=0] topping up via roneneldan/TinyStories...", flush=True)
            for doc_b, sid in _stream_en_tinystories():
                emit_fn(acc, doc_b, sid, 0)
                if acc.done:
                    break
        return _finalize_domain_stat(acc, time.perf_counter() - t0)

    if lang_id == 1:
        v21_cap = int(total_target * (130.0 / 180.0))
        v21_shards = 3 if smoke_quota else min(400, max(40, int(40 * scale_factor)))
        print("  -> [zh | lang_id=1] streaming opencsg/Fineweb-Edu-Chinese-V2.1 (4_5/*.parquet)...", flush=True)
        for doc_b, sid in _stream_zh_fineweb_v21(max_shards=v21_shards):
            emit_fn(acc, doc_b, sid, 0)
            if (acc.train_bytes + acc.val_bytes) >= v21_cap or acc.done:
                break
        if not acc.done:
            print("  -> [zh | lang_id=1] streaming opencsg/Fineweb-Edu-Chinese-V3 (HTTP tar.gz)...", flush=True)
            for doc_b, sid in _stream_zh_fineweb_v3():
                emit_fn(acc, doc_b, sid, 0)
                if acc.done:
                    break
        if not acc.done:
            for doc_b, sid in _stream_zh_fineweb_v21(max_shards=min(500, v21_shards + 40)):
                emit_fn(acc, doc_b, sid, 0)
                if acc.done:
                    break
        return _finalize_domain_stat(acc, time.perf_counter() - t0)

    if lang_id == 2:
        ja_mono_cap = int(total_target * 0.50)
        ja_enja_cap = int(total_target * 0.85)
        print("  -> [ja | lang_id=2] streaming Helsinki-NLP/opus-100 (en-ja monolingual 50%)...", flush=True)
        for doc_b, sid in _stream_ja_opus("ja_mono"):
            emit_fn(acc, doc_b, sid, 0)
            if (acc.train_bytes + acc.val_bytes) >= ja_mono_cap or acc.done:
                break
        if not acc.done:
            print("  -> [ja | lang_id=2] streaming Helsinki-NLP/opus-100 (en-ja bilingual pairs)...", flush=True)
            for doc_b, sid in _stream_ja_opus("en_ja_pair"):
                emit_fn(acc, doc_b, sid, 0)
                if (acc.train_bytes + acc.val_bytes) >= ja_enja_cap or acc.done:
                    break
        if not acc.done and scale_factor > 1.5:
            print("  -> [ja | lang_id=2] topping up via wikimedia/wikipedia (20231101.ja)...", flush=True)
            for doc_b, sid in _stream_ja_wikipedia():
                emit_fn(acc, doc_b, sid, 0)
                if acc.done:
                    break
        if not acc.done:
            print("  -> [ja | lang_id=2] streaming Helsinki-NLP/opus-100 (en-zh cross-order pairs)...", flush=True)
            for doc_b, sid in _stream_ja_opus("en_zh_pair"):
                emit_fn(acc, doc_b, sid, 0)
                if acc.done:
                    break
        return _finalize_domain_stat(acc, time.perf_counter() - t0)

    if lang_id == 3:
        for raw_code, sid in _stream_tiny_codes("python"):
            acc.code_candidates += 1
            cleaned = normalize_and_validate_text(raw_code)
            if cleaned is None:
                acc.code_rejected += 1
                continue
            code_b = cleaned.encode("utf-8", errors="strict")
            ok, depth = validator.validate("python", code_b)
            if not ok:
                acc.code_rejected += 1
                continue
            emit_fn(acc, code_b, sid, depth)
            if acc.done:
                break
        if not acc.done and scale_factor > 1.5:
            for raw_code, sid in _stream_python_git_repos():
                acc.code_candidates += 1
                cleaned = normalize_and_validate_text(raw_code)
                if cleaned is None:
                    acc.code_rejected += 1
                    continue
                code_b = cleaned.encode("utf-8", errors="strict")
                ok, depth = validator.validate("python", code_b)
                if not ok:
                    acc.code_rejected += 1
                    continue
                emit_fn(acc, code_b, sid, depth)
                if acc.done:
                    break
        return _finalize_domain_stat(acc, time.perf_counter() - t0)

    if lang_id == 4:
        for raw_code, sid in _stream_tiny_codes("rust"):
            acc.code_candidates += 1
            cleaned = normalize_and_validate_text(raw_code)
            if cleaned is None:
                acc.code_rejected += 1
                continue
            code_b = cleaned.encode("utf-8", errors="strict")
            ok, depth = validator.validate("rust", code_b)
            if not ok:
                acc.code_rejected += 1
                continue
            emit_fn(acc, code_b, sid, depth)
            if acc.done:
                break
        if not acc.done:
            print("  -> [rust | lang_id=4] topping up from official Rust git crates...", flush=True)
            for raw_code, sid in _stream_rust_git_repos(extended=(scale_factor > 1.2 or stem_upgrade)):
                acc.code_candidates += 1
                cleaned = normalize_and_validate_text(raw_code)
                if cleaned is None:
                    acc.code_rejected += 1
                    continue
                code_b = cleaned.encode("utf-8", errors="strict")
                ok, depth = validator.validate("rust", code_b)
                if not ok:
                    acc.code_rejected += 1
                    continue
                emit_fn(acc, code_b, sid, depth)
                if acc.done:
                    break
        return _finalize_domain_stat(acc, time.perf_counter() - t0)

    if lang_id == 5:
        for raw_code, sid in _stream_tiny_codes("javascript"):
            acc.code_candidates += 1
            cleaned = normalize_and_validate_text(raw_code)
            if cleaned is None:
                acc.code_rejected += 1
                continue
            code_b = cleaned.encode("utf-8", errors="strict")
            ok, depth = validator.validate("javascript", code_b)
            if not ok:
                acc.code_rejected += 1
                continue
            emit_fn(acc, code_b, sid, depth)
            if acc.done:
                break
        if not acc.done:
            print("  -> [javascript | lang_id=5] topping up from idiomatic ES6+ JavaScript git repos...", flush=True)
            for raw_code, sid in _stream_js_git_repos(extended=(scale_factor > 1.2 or stem_upgrade)):
                acc.code_candidates += 1
                cleaned = normalize_and_validate_text(raw_code)
                if cleaned is None:
                    acc.code_rejected += 1
                    continue
                code_b = cleaned.encode("utf-8", errors="strict")
                ok, depth = validator.validate("javascript", code_b)
                if not ok:
                    acc.code_rejected += 1
                    continue
                emit_fn(acc, code_b, sid, depth)
                if acc.done:
                    break
        return _finalize_domain_stat(acc, time.perf_counter() - t0)

    raise ValueError(f"Unknown lang_id={lang_id}")


def _worker_spool_domain(args_tuple: tuple[Any, ...]) -> dict[str, Any]:
    """Subprocess worker that streams & AST-gates a single domain into a temporary spool file."""
    lang_id, train_target, val_target, spool_dir_str, seed, smoke_quota, scale_factor, stem_upgrade = args_tuple
    spool_dir = Path(spool_dir_str)
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool_bin = spool_dir / f"dom_{lang_id}.bin"
    spool_rec = spool_dir / f"dom_{lang_id}.rec"

    validator = CodeASTValidator()
    acc = _DomainAccum(lang_id, DOMAIN_NAMES[lang_id], train_target, val_target)
    local_hashes: set[bytes] = set()
    hash_person = f"dcpa{seed}".encode("ascii")[:16]
    val_mod = 10 if smoke_quota else 20
    source_id_to_name = {v: k for k, v in SOURCE_IDS.items()}
    spool_offset = 0

    with spool_bin.open("wb") as f_bin, spool_rec.open("wb") as f_rec:
        def _spool_emit(a: _DomainAccum, doc_bytes: bytes, source_id: int, max_depth: int) -> bool:
            nonlocal spool_offset
            if a.done or not doc_bytes:
                return False
            digest = hashlib.blake2b(doc_bytes, digest_size=8, person=hash_person).digest()
            if digest in local_hashes:
                a.dup_rejected += 1
                return False
            local_hashes.add(digest)
            rec_len = len(doc_bytes)
            bucket_hit = (int.from_bytes(digest[:4], "little") % val_mod) == 0
            if a.val_bytes < a.val_target and (bucket_hit or a.train_bytes >= a.train_target):
                a.val_bytes += rec_len
                a.val_docs += 1
            elif a.train_bytes < a.train_target:
                a.train_bytes += rec_len
                a.train_docs += 1
            else:
                return False
            f_bin.write(doc_bytes)
            f_rec.write(struct.pack("<QIBB", spool_offset, rec_len, source_id, min(max_depth, 255)))
            spool_offset += rec_len
            sname = source_id_to_name.get(source_id, f"src_{source_id}")
            if sname not in a.sources_used:
                a.sources_used.append(sname)
            return True

        stat = _populate_single_domain(
            lang_id,
            acc,
            _spool_emit,
            validator,
            smoke_quota=smoke_quota,
            scale_factor=scale_factor,
            stem_upgrade=stem_upgrade,
        )
    return {
        "lang_id": lang_id,
        "code_candidates": acc.code_candidates,
        "code_rejected": acc.code_rejected,
        "dup_rejected": acc.dup_rejected,
        "elapsed_sec": stat["elapsed_sec"],
        "spool_bin": str(spool_bin),
        "spool_rec": str(spool_rec),
    }


def build_corpus(
    out_dir: str | Path = "artifacts/data",
    smoke_quota: bool = False,
    seed: int = 2026,
    allow_drive_output: bool = False,
    scale_factor: float = 1.0,
    stem_upgrade: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Stream, filter, deduplicate, and write 5-domain DCPA `.bin/.idx/.meta.json` shards."""
    out_path = Path(out_dir)
    if not allow_drive_output:
        _assert_local_output_dir(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    if scale_factor <= 0.0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}")

    validator = CodeASTValidator()
    mode = "smoke" if smoke_quota else ("full_stem" if (stem_upgrade or scale_factor > 1.5) else "full")
    if smoke_quota:
        train_quotas = dict(SMOKE_TRAIN_QUOTA_BYTES)
        val_quotas = dict(SMOKE_VAL_QUOTA_BYTES)
    else:
        val_scale = min(float(scale_factor), 8.0)
        train_quotas = {lid: max(64 * 1024, int(round(q * scale_factor))) for lid, q in FULL_TRAIN_QUOTA_BYTES.items()}
        val_quotas = {lid: max(16 * 1024, int(round(q * val_scale))) for lid, q in FULL_VAL_QUOTA_BYTES.items()}

    train_bin_path = out_path / "train.bin"
    train_idx_path = out_path / "train.idx"
    val_bin_path = out_path / "val.bin"
    val_idx_path = out_path / "val.idx"

    seen_hashes: set[bytes] = set()
    train_offset = 0
    val_offset = 0
    train_docs_total = 0
    val_docs_total = 0
    hash_person = f"dcpa{seed}".encode("ascii")[:16]

    t_global = time.perf_counter()
    print(
        f"[build_corpus] mode={mode} | scale={scale_factor:.2f}x | workers={max(1, int(workers))} | "
        f"out_dir={out_path} | tree_sitter={validator.tree_sitter_active} | "
        f"target_train={sum(train_quotas.values()) / (1024*1024):.1f} MB, "
        f"target_val={sum(val_quotas.values()) / (1024*1024):.1f} MB",
        flush=True,
    )

    domain_stats: dict[str, dict[str, Any]] = {}
    source_id_to_name = {v: k for k, v in SOURCE_IDS.items()}

    with (
        train_bin_path.open("wb") as train_bin,
        train_idx_path.open("wb") as train_idx,
        val_bin_path.open("wb") as val_bin,
        val_idx_path.open("wb") as val_idx,
    ):

        def _emit_doc(acc: _DomainAccum, doc_bytes: bytes, source_id: int, max_depth: int) -> bool:
            """Deduplicate via 64-bit blake2b and route strictly without Train/Val leakage."""
            nonlocal train_offset, val_offset, train_docs_total, val_docs_total
            if acc.done:
                return False
            rec_len = len(doc_bytes)
            if rec_len == 0:
                return False

            digest = hashlib.blake2b(doc_bytes, digest_size=8, person=hash_person).digest()
            if digest in seen_hashes:
                acc.dup_rejected += 1
                return False
            seen_hashes.add(digest)

            # Deterministic val bucket (~5% in full, ~10% in smoke); prioritize filling val
            # cleanly if train is full, and never let the same hash enter both splits.
            val_mod = 10 if smoke_quota else 20
            bucket_hit = (int.from_bytes(digest[:4], "little") % val_mod) == 0
            if acc.val_bytes < acc.val_target and (bucket_hit or acc.train_bytes >= acc.train_target):
                val_bin.write(doc_bytes)
                val_idx.write(
                    struct.pack(IDX_RECORD_FMT, val_offset, rec_len, acc.lang_id, source_id, min(max_depth, 255), 0)
                )
                val_offset += rec_len
                val_docs_total += 1
                acc.val_bytes += rec_len
                acc.val_docs += 1
            elif acc.train_bytes < acc.train_target:
                train_bin.write(doc_bytes)
                train_idx.write(
                    struct.pack(IDX_RECORD_FMT, train_offset, rec_len, acc.lang_id, source_id, min(max_depth, 255), 0)
                )
                train_offset += rec_len
                train_docs_total += 1
                acc.train_bytes += rec_len
                acc.train_docs += 1
            else:
                return False

            sname = source_id_to_name.get(source_id, f"src_{source_id}")
            if sname not in acc.sources_used:
                acc.sources_used.append(sname)
            return True

        if int(workers) > 1:
            from concurrent.futures import ProcessPoolExecutor

            spool_dir = out_path / f".tmp_spool_{os.getpid()}"
            spool_dir.mkdir(parents=True, exist_ok=True)
            try:
                tasks = [
                    (
                        lid,
                        train_quotas[lid],
                        val_quotas[lid],
                        str(spool_dir),
                        seed,
                        smoke_quota,
                        float(scale_factor),
                        bool(stem_upgrade),
                    )
                    for lid in ACTIVE_DOMAIN_IDS
                ]
                with ProcessPoolExecutor(max_workers=min(len(ACTIVE_DOMAIN_IDS), int(workers))) as pool:
                    worker_results = list(pool.map(_worker_spool_domain, tasks))

                # Deterministic merge in lang_id = 0..4 order with global leak-free seen_hashes check
                rec_struct = struct.Struct("<QIBB")
                for res in sorted(worker_results, key=lambda r: int(r["lang_id"])):
                    lid = int(res["lang_id"])
                    dname = DOMAIN_NAMES[lid]
                    acc = _DomainAccum(
                        lid,
                        dname,
                        train_quotas[lid],
                        val_quotas[lid],
                        code_candidates=int(res["code_candidates"]),
                        code_rejected=int(res["code_rejected"]),
                        dup_rejected=int(res["dup_rejected"]),
                    )
                    sp_bin = Path(res["spool_bin"])
                    sp_rec = Path(res["spool_rec"])
                    with sp_bin.open("rb") as fb, sp_rec.open("rb") as fr:
                        while buf := fr.read(rec_struct.size):
                            off, rlen, sid, depth = rec_struct.unpack(buf)
                            fb.seek(off)
                            doc_b = fb.read(rlen)
                            _emit_doc(acc, doc_b, sid, depth)
                    total_b = acc.train_bytes + acc.val_bytes
                    total_d = acc.train_docs + acc.val_docs
                    avg_len = (total_b / total_d) if total_d > 0 else 0.0
                    rej_pct = (
                        (100.0 * acc.code_rejected / acc.code_candidates) if acc.code_candidates > 0 else 0.0
                    )
                    domain_stats[dname] = {
                        "lang_id": lid,
                        "name": dname,
                        "train_docs": acc.train_docs,
                        "train_bytes": acc.train_bytes,
                        "val_docs": acc.val_docs,
                        "val_bytes": acc.val_bytes,
                        "total_bytes": total_b,
                        "avg_doc_bytes": round(avg_len, 2),
                        "code_candidates": acc.code_candidates,
                        "code_rejected": acc.code_rejected,
                        "code_rejection_rate_pct": round(rej_pct, 2) if acc.code_candidates > 0 else None,
                        "dup_rejected": acc.dup_rejected,
                        "sources_used": acc.sources_used,
                        "elapsed_sec": res["elapsed_sec"],
                    }
            finally:
                shutil.rmtree(spool_dir, ignore_errors=True)
        else:
            for lid in ACTIVE_DOMAIN_IDS:
                dname = DOMAIN_NAMES[lid]
                acc = _DomainAccum(lid, dname, train_quotas[lid], val_quotas[lid])
                domain_stats[dname] = _populate_single_domain(
                    lid,
                    acc,
                    _emit_doc,
                    validator,
                    smoke_quota=smoke_quota,
                    scale_factor=float(scale_factor),
                    stem_upgrade=bool(stem_upgrade),
                )

    def _file_blake2b(path: Path) -> str:
        h = hashlib.blake2b()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    total_elapsed = round(time.perf_counter() - t_global, 2)
    train_meta = {
        "split": "train",
        "mode": mode,
        "scale_factor": float(scale_factor),
        "stem_upgrade": bool(stem_upgrade),
        "seed": seed,
        "idx_format": IDX_RECORD_FMT,
        "idx_record_bytes": IDX_RECORD_SIZE,
        "total_bytes": train_offset,
        "total_docs": train_docs_total,
        "bin_blake2b": _file_blake2b(train_bin_path),
        "idx_blake2b": _file_blake2b(train_idx_path),
        "domains": {
            k: {
                "lang_id": v["lang_id"],
                "docs": v["train_docs"],
                "bytes": v["train_bytes"],
                "sources": v["sources_used"],
            }
            for k, v in domain_stats.items()
        },
    }
    val_meta = {
        "split": "val",
        "mode": mode,
        "scale_factor": float(scale_factor),
        "stem_upgrade": bool(stem_upgrade),
        "seed": seed,
        "idx_format": IDX_RECORD_FMT,
        "idx_record_bytes": IDX_RECORD_SIZE,
        "total_bytes": val_offset,
        "total_docs": val_docs_total,
        "bin_blake2b": _file_blake2b(val_bin_path),
        "idx_blake2b": _file_blake2b(val_idx_path),
        "domains": {
            k: {
                "lang_id": v["lang_id"],
                "docs": v["val_docs"],
                "bytes": v["val_bytes"],
                "sources": v["sources_used"],
            }
            for k, v in domain_stats.items()
        },
    }
    combined_meta = {
        "mode": mode,
        "scale_factor": float(scale_factor),
        "stem_upgrade": bool(stem_upgrade),
        "seed": seed,
        "tree_sitter_active": validator.tree_sitter_active,
        "idx_format": IDX_RECORD_FMT,
        "total_bytes": train_offset + val_offset,
        "train_bytes": train_offset,
        "val_bytes": val_offset,
        "train_docs": train_docs_total,
        "val_docs": val_docs_total,
        "domains": domain_stats,
        "checksums_blake2b": {
            "train.bin": train_meta["bin_blake2b"],
            "train.idx": train_meta["idx_blake2b"],
            "val.bin": val_meta["bin_blake2b"],
            "val.idx": val_meta["idx_blake2b"],
        },
        "elapsed_sec": total_elapsed,
    }

    (out_path / "train.meta.json").write_text(json.dumps(train_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_path / "val.meta.json").write_text(json.dumps(val_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_path / "corpus_meta.json").write_text(json.dumps(combined_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[build_corpus] DONE: {(train_offset + val_offset) / (1024*1024):.2f} MB "
        f"(Train={train_docs_total} docs / {train_offset/(1024*1024):.2f} MB, "
        f"Val={val_docs_total} docs / {val_offset/(1024*1024):.2f} MB) "
        f"in {total_elapsed:.1f}s -> {out_path}",
        flush=True,
    )
    return combined_meta


def _finalize_domain_stat(acc: _DomainAccum, elapsed: float) -> dict[str, Any]:
    total_b = acc.train_bytes + acc.val_bytes
    total_d = acc.train_docs + acc.val_docs
    avg_len = (total_b / total_d) if total_d > 0 else 0.0
    rej_pct = (100.0 * acc.code_rejected / acc.code_candidates) if acc.code_candidates > 0 else 0.0
    print(
        f"  [OK] {acc.name} (id={acc.lang_id}): Train={acc.train_bytes/(1024*1024):.2f} MB ({acc.train_docs} docs), "
        f"Val={acc.val_bytes/(1024*1024):.2f} MB ({acc.val_docs} docs), avg={avg_len:.1f} B"
        + (f", code_rej={rej_pct:.1f}% ({acc.code_rejected}/{acc.code_candidates})" if acc.code_candidates > 0 else "")
        + f" in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "lang_id": acc.lang_id,
        "name": acc.name,
        "train_docs": acc.train_docs,
        "train_bytes": acc.train_bytes,
        "val_docs": acc.val_docs,
        "val_bytes": acc.val_bytes,
        "total_bytes": total_b,
        "avg_doc_bytes": round(avg_len, 2),
        "code_candidates": acc.code_candidates,
        "code_rejected": acc.code_rejected,
        "code_rejection_rate_pct": round(rej_pct, 2) if acc.code_candidates > 0 else None,
        "dup_rejected": acc.dup_rejected,
        "sources_used": acc.sources_used,
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream and build DCPA 5-domain raw UTF-8 .bin/.idx/.meta.json corpus."
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/data",
        help="Local NVMe output directory (must NOT be on /content/drive).",
    )
    ap.add_argument(
        "--smoke-quota",
        action="store_true",
        help="Local M3 Pro smoke mode: stream 2 MB per lang_id (~10 MB total).",
    )
    ap.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Quota multiplier over default 430 MB train build (e.g. 24.0 -> ~10.3 GB).",
    )
    ap.add_argument(
        "--stem-upgrade",
        action="store_true",
        help="Upgrade English stream to FineWeb-Edu (score>=4) + OpenWebMath + extended Git code.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel per-domain workers (up to 5) for multi-core Vast.ai NVMe builds.",
    )
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    build_corpus(
        out_dir=args.out_dir,
        smoke_quota=args.smoke_quota,
        seed=args.seed,
        scale_factor=args.scale_factor,
        stem_upgrade=args.stem_upgrade,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
