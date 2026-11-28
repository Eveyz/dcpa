"""DCPA v2 Holistic 6-Domain Staged Corpus Streamers, Style Classifier & Offline Top-Up Generators.

Implements `docs/DCPA_ROADMAP.md` Section 2.3 & 2.4 for `build_corpus.py --cognitive-v2`:
  - 6 Orthogonal `lang_id` slots (ZERO Rust):
      * 0 = `en`         : English logic/expository + classic/modern literature (`finephrase`, `US-PD-Books`, `cosmopedia-v2`, `fineweb-edu`)
      * 1 = `zh`         : Chinese logic/criticism + literary prose (`Fineweb-Edu-Chinese V2.1/V3`, `BAAI/CCI3-HQ` w/ ungated fallback)
      * 2 = `formal`     : Formal first-order logic, deductive reading & math proofs (`tasksource/folio`, `proofwriter`, `logiqa`, `finemath`)
      * 3 = `python`     : Stage-1 clean stdlib (`smollm-corpus/python-edu`) -> Stage-2/3 rewritten modular Python (`tokyotech-llm/swallow-code-v2`)
      * 4 = `sql_struct` : Schema-grounded SQL queries & structured JSON/Markdown specs (`b-mc2/sql-create-context` + analytic CTE/JSON schemas)
      * 5 = `javascript` : Stage-1 pure ES6+ short primitives -> Stage-2/3 modular JSDoc JavaScript/TypeScript
  - Enforces `<= 3 epochs` hard repeat cap on small curated datasets (`FOLIO`).
  - Classifies every document into `style_id` in `0..4`:
      `0=rigorous`, `1=concise_memo`, `2=expository`, `3=literary`, `4=strict_eng`.
"""

from __future__ import annotations

import gc
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

# Fixed-point x10 `.loss_mask.bin` constants (SSOT: dcpa/interfaces.py)
LOSS_MASK_IGNORE: int = 0
LOSS_MASK_LEGACY_ONE: int = 1
LOSS_MASK_NORMAL_X10: int = 10
LOSS_MASK_SYNTAX_X10: int = 15
LOSS_MASK_ANCHOR_X10: int = 30
SYNTAX_DELIM_BYTES: frozenset[int] = frozenset(b"{}()[]:;\n")
_SYNTAX_DELIM_LUT: np.ndarray = np.array(
    [i in SYNTAX_DELIM_BYTES for i in range(256)], dtype=np.bool_
)

# Stage-aware `.idx` `source_id` constants matching `data/ragged_dataloader.py::SOURCE_STAGE_TIERS`
SID_FINEPHRASE_S1 = 21
SID_US_PD_BOOKS_S1 = 22
SID_ZH_PROSE_SHORT_S1 = 23
SID_PY_EDU_SHORT_S1 = 24
SID_JS_ES6_SHORT_S1 = 25
SID_SQL_JSON_BASIC_S1 = 26

SID_FOLIO_LOGIC_S2 = 27
SID_LOGIQA_RECLOR_S2 = 28
SID_FINEMATH_S2 = 29
SID_COSMOPEDIA_S2 = 30
SID_CCI3_HQ_S2 = 31
SID_SWALLOW_PY_S2 = 32
SID_SQL_CONTEXT_S2 = 33
SID_JS_MODULAR_S2 = 34

SID_LONG_ESSAYS_S3 = 35
SID_COMPLEX_PROOFS_S3 = 36
SID_SQL_CTE_SCHEMA_S3 = 37

# DCPA-Flat v3.3 2025-2026 High-Purity Source IDs (41..52)
SID_NEMOTRON_CC_V21 = 41
SID_FINEWEB2_HQ = 42
SID_ZH_FINEWEB_EDU_V2 = 43
SID_ZH_FINEWEB_EDU_V22 = 44
SID_ZH_COSMOPEDIA = 45
SID_BAAI_CCI3_HQ_V3 = 46
SID_BAAI_INFINITY_INSTRUCT = 47
SID_NEMOTRON_CC_MATH_V1 = 48
SID_SWALLOW_CODE_V2_V3 = 49
SID_NEMOTRON_CC_CODE_V1 = 50
SID_OPENCODEREASONING = 51
SID_OPENCODEREASONING_2 = 52
SID_FINEWEB_EDU_100BT_V3 = 53
SID_STACK_SMOL_XL_JS_V3 = 54


@dataclass(frozen=True)
class SourceDescriptor:
    """Declarative registry entry for a 2025-2026 high-purity streaming corpus source."""

    repo_id: str
    source_id: int
    lang_id: int
    stage_tier: int
    text_fields: tuple[str, ...]
    description: str
    config_name: Optional[str] = None
    split: str = "train"
    data_dir: Optional[str] = None
    parquet_glob: Optional[str] = None


V3_FLAT_SOURCE_DESCRIPTORS: dict[str, SourceDescriptor] = {
    "nvidia/Nemotron-CC-v2.1": SourceDescriptor(
        repo_id="nvidia/Nemotron-CC-v2.1",
        source_id=SID_NEMOTRON_CC_V21,
        lang_id=0,
        stage_tier=2,
        text_fields=("text", "content"),
        description="High-purity English rephrased/filtered web prose & STEM (Nemotron-CC v2.1; gated=manual).",
        config_name="High-Quality",
        split="train",
    ),
    "HuggingFaceFW/fineweb-edu": SourceDescriptor(
        repo_id="HuggingFaceFW/fineweb-edu",
        source_id=SID_FINEWEB_EDU_100BT_V3,
        lang_id=0,
        stage_tier=2,
        text_fields=("text", "content"),
        description="Ungated high-purity English educational web prose (FineWeb-Edu sample-100BT).",
        config_name="sample-100BT",
        split="train",
    ),
    "epfml/FineWeb2-HQ": SourceDescriptor(
        repo_id="epfml/FineWeb2-HQ",
        source_id=SID_FINEWEB2_HQ,
        lang_id=1,
        stage_tier=2,
        text_fields=("text", "content"),
        description="Classifier-filtered high-quality Chinese web corpus (EPFL FineWeb2-HQ cmn_Hani).",
        config_name="cmn_Hani",
        split="train",
    ),
    "opencsg/chinese-fineweb-edu-v2": SourceDescriptor(
        repo_id="opencsg/chinese-fineweb-edu-v2",
        source_id=SID_ZH_FINEWEB_EDU_V2,
        lang_id=1,
        stage_tier=2,
        text_fields=("text", "content"),
        description="High-score Chinese educational web prose (OpenCSG Chinese FineWeb-Edu v2).",
        config_name=None,
        split="train",
    ),
    "opencsg/Fineweb-Edu-Chinese-V2.2": SourceDescriptor(
        repo_id="opencsg/Fineweb-Edu-Chinese-V2.2",
        source_id=SID_ZH_FINEWEB_EDU_V22,
        lang_id=1,
        stage_tier=2,
        text_fields=("text", "content"),
        description="Updated Chinese educational score 4-5 prose (OpenCSG Fineweb-Edu-Chinese V2.2 parquet).",
        config_name=None,
        split="train",
        parquet_glob="hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.2/4_5/*.parquet",
    ),
    "opencsg/chinese-cosmopedia": SourceDescriptor(
        repo_id="opencsg/chinese-cosmopedia",
        source_id=SID_ZH_COSMOPEDIA,
        lang_id=1,
        stage_tier=2,
        text_fields=("text", "content"),
        description="Synthetic Chinese textbooks, tutorials, and conceptual essays (Chinese Cosmopedia).",
        config_name=None,
        split="train",
    ),
    "BAAI/CCI3-HQ": SourceDescriptor(
        repo_id="BAAI/CCI3-HQ",
        source_id=SID_BAAI_CCI3_HQ_V3,
        lang_id=1,
        stage_tier=2,
        text_fields=("text", "content"),
        description="BAAI CCI3-HQ high-purity Chinese commentary & knowledge prose (gated=auto).",
        config_name=None,
        split="train",
    ),
    "BAAI/Infinity-Instruct": SourceDescriptor(
        repo_id="BAAI/Infinity-Instruct",
        source_id=SID_BAAI_INFINITY_INSTRUCT,
        lang_id=1,
        stage_tier=3,
        text_fields=("conversations", "text", "content"),
        description="BAAI Infinity-Instruct bilingual reasoning & instructional dialogues (7M_core; gated=auto).",
        config_name="7M_core",
        split="train",
    ),
    "nvidia/Nemotron-CC-Math-v1": SourceDescriptor(
        repo_id="nvidia/Nemotron-CC-Math-v1",
        source_id=SID_NEMOTRON_CC_MATH_V1,
        lang_id=2,
        stage_tier=3,
        text_fields=("text", "content"),
        description="Formal math derivations, symbolic equations & proofs (Nemotron-CC-Math v1 4plus; gated=auto).",
        config_name="4plus",
        split="train",
    ),
    "tokyotech-llm/swallow-code-v2": SourceDescriptor(
        repo_id="tokyotech-llm/swallow-code-v2",
        source_id=SID_SWALLOW_CODE_V2_V3,
        lang_id=3,
        stage_tier=2,
        text_fields=("text", "improved_text", "code", "content"),
        description="Self-contained rewritten high-clarity Python modules (Swallow-Code v2 stage4-llm-rewrite).",
        config_name="stage4-llm-rewrite",
        split="train",
    ),
    "nvidia/Nemotron-CC-Code-v1": SourceDescriptor(
        repo_id="nvidia/Nemotron-CC-Code-v1",
        source_id=SID_NEMOTRON_CC_CODE_V1,
        lang_id=3,
        stage_tier=3,
        text_fields=("text", "code", "content"),
        description="High-purity web-extracted Python & JavaScript code (Nemotron-CC-Code v1; gated=manual).",
        config_name="data",
        split="train",
    ),
    "nvidia/OpenCodeReasoning": SourceDescriptor(
        repo_id="nvidia/OpenCodeReasoning",
        source_id=SID_OPENCODEREASONING,
        lang_id=3,
        stage_tier=3,
        text_fields=("solution", "output", "text", "code"),
        description="Verified algorithmic code reasoning traces & solutions (OpenCodeReasoning split_0).",
        config_name="split_0",
        split="split_0",
    ),
    "nvidia/OpenCodeReasoning-2": SourceDescriptor(
        repo_id="nvidia/OpenCodeReasoning-2",
        source_id=SID_OPENCODEREASONING_2,
        lang_id=3,
        stage_tier=3,
        text_fields=("solution", "r1_generation", "output", "text", "code"),
        description="Expanded verified Python code reasoning solutions (OpenCodeReasoning-2 train/python).",
        config_name="train",
        split="python",
    ),
    "bigcode/the-stack-smol-xl": SourceDescriptor(
        repo_id="bigcode/the-stack-smol-xl",
        source_id=SID_STACK_SMOL_XL_JS_V3,
        lang_id=5,
        stage_tier=2,
        text_fields=("content", "text", "code"),
        description="Ungated permissive JavaScript source modules (The Stack Smol XL data/javascript).",
        config_name=None,
        split="train",
        data_dir="data/javascript",
    ),
}

V3_FLAT_SOURCE_IDS: dict[str, int] = {
    repo: desc.source_id for repo, desc in V3_FLAT_SOURCE_DESCRIPTORS.items()
}
V3_FLAT_SOURCE_STAGE_TIERS: dict[int, int] = {
    desc.source_id: desc.stage_tier for desc in V3_FLAT_SOURCE_DESCRIPTORS.values()
}

COGNITIVE_V2_SOURCE_IDS: dict[str, int] = {
    "HuggingFaceFW/finephrase:s1": SID_FINEPHRASE_S1,
    "storytracer/US-PD-Books:s1": SID_US_PD_BOOKS_S1,
    "zh_literary_expository:s1": SID_ZH_PROSE_SHORT_S1,
    "HuggingFaceTB/smollm-corpus:python-edu-s1": SID_PY_EDU_SHORT_S1,
    "es6_clean_primitives:s1": SID_JS_ES6_SHORT_S1,
    "sql_json_primitives:s1": SID_SQL_JSON_BASIC_S1,
    "tasksource/folio+proofwriter:s2": SID_FOLIO_LOGIC_S2,
    "lucasmccabe/logiqa+reclor:s2": SID_LOGIQA_RECLOR_S2,
    "HuggingFaceTB/finemath-4plus:s2": SID_FINEMATH_S2,
    "HuggingFaceTB/cosmopedia-v2:s2": SID_COSMOPEDIA_S2,
    "BAAI/CCI3-HQ_or_FinewebZH5:s2": SID_CCI3_HQ_S2,
    "tokyotech-llm/swallow-code-v2:s2": SID_SWALLOW_PY_S2,
    "b-mc2/sql-create-context:s2": SID_SQL_CONTEXT_S2,
    "js_modular_jsdoc:s2": SID_JS_MODULAR_S2,
    "long_essays_criticism:s3": SID_LONG_ESSAYS_S3,
    "complex_formal_proofs:s3": SID_COMPLEX_PROOFS_S3,
    "sql_cte_analytic_schemas:s3": SID_SQL_CTE_SCHEMA_S3,
    **V3_FLAT_SOURCE_IDS,
}

# Base quotas at scale_factor=1.0 (~440 MB Train -> ~5.28 GB at scale_factor=12.0 for v2_cognitive_5g)
COGNITIVE_V2_DOMAIN_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
COGNITIVE_V2_DOMAIN_NAMES: dict[int, str] = {
    0: "en",
    1: "zh",
    2: "formal",
    3: "python",
    4: "sql_struct",
    5: "javascript",
}
COGNITIVE_V2_TRAIN_QUOTAS: dict[int, int] = {
    0: 130 * 1024 * 1024,  # en: 130 MB
    1: 140 * 1024 * 1024,  # zh: 140 MB
    2: 45 * 1024 * 1024,   # formal (FOL + deductive logic + math proofs): 45 MB
    3: 65 * 1024 * 1024,   # python (stdlib + swallow-code-v2 + modular): 65 MB
    4: 20 * 1024 * 1024,   # sql_struct (SQL + JSON/Markdown schemas): 20 MB
    5: 40 * 1024 * 1024,   # javascript (ES6+ primitives + modular JS): 40 MB
}
COGNITIVE_V2_VAL_QUOTAS: dict[int, int] = {
    0: 2 * 1024 * 1024,
    1: 2 * 1024 * 1024,
    2: 1 * 1024 * 1024,
    3: 2 * 1024 * 1024,
    4: 1 * 1024 * 1024,
    5: 2 * 1024 * 1024,
}
COGNITIVE_V2_SMOKE_TRAIN_QUOTAS: dict[int, int] = {
    lid: int(1.4 * 1024 * 1024) for lid in COGNITIVE_V2_DOMAIN_IDS
}
COGNITIVE_V2_SMOKE_VAL_QUOTAS: dict[int, int] = {
    lid: int(0.2 * 1024 * 1024) for lid in COGNITIVE_V2_DOMAIN_IDS
}

_CTRL_OK = {9, 10}
_CLICHE_BLACKLIST_RE = re.compile(
    r"(?i)\b(delve into|testament to|rich tapestry|it is worth noting|in conclusion,)\b"
    r"|总而言之|综上所述|值得注意的是|是一把双刃剑"
)
_LITERARY_RE = re.compile(
    r'[“”「」『』—…]|\b(whispered|shadow|lantern|dusk|footsteps|silence|glanced|breeze|autumn|hearth)\b'
    r"|暮色|微光|灯影|脚步|沉默|晚风|庭院|青石|炊烟|月色|屋檐"
)
_RIGOROUS_RE = re.compile(
    r"(?i)\b(therefore|consequently|premise|contradiction|invariant|hypothesis|theorem|lemma|implies|necessary)\b"
    r"|因此|前提|结论|矛盾|充要|守恒|不变量|命题|推论|因果|机制"
)
_MEMO_RE = re.compile(
    r"(?i)\b(summary:|decision:|kpi|latency|throughput|action item|bottleneck|trade-off)\b"
    r"|摘要：|结论：|核心指标|瓶颈|取舍|建议："
)


def _clean_nfc(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    nfc = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    if "\ufffd" in nfc or _CLICHE_BLACKLIST_RE.search(nfc):
        return None
    for ch in nfc:
        o = ord(ch)
        if (o < 32 and o not in _CTRL_OK) or o == 127:
            return None
    s = nfc.strip()
    return (s + "\n") if s else None


def infer_corpus_style_id(lang_id: int, doc_bytes: bytes, source_id: int) -> int:
    """Classify document into `0..4` orthogonal `style_id` subspace:
    `0=rigorous`, `1=concise_memo`, `2=expository`, `3=literary`, `4=strict_eng`.
    """
    if source_id == SID_US_PD_BOOKS_S1:
        return 3
    if source_id in (
        SID_FOLIO_LOGIC_S2,
        SID_LOGIQA_RECLOR_S2,
        SID_FINEMATH_S2,
        SID_COMPLEX_PROOFS_S3,
        SID_NEMOTRON_CC_MATH_V1,
    ):
        return 0
    if source_id in (SID_COSMOPEDIA_S2, SID_ZH_COSMOPEDIA):
        return 2
    if source_id in (
        SID_SWALLOW_CODE_V2_V3,
        SID_NEMOTRON_CC_CODE_V1,
        SID_OPENCODEREASONING,
        SID_OPENCODEREASONING_2,
    ):
        return 4
    txt = doc_bytes[:1536].decode("utf-8", errors="ignore")
    if lang_id in (3, 5):
        if '"""' in txt or "'''" in txt or "/**" in txt:
            if "Example" in txt or "Tutorial" in txt:
                return 2
        return 4
    if lang_id == 4:
        return 1 if txt.lstrip().startswith(("{", "#", "|")) else 4
    if lang_id == 2:
        return 1 if ("->" in txt or "⊢" in txt or "∀" in txt) and len(doc_bytes) < 420 else 0
    # Natural language (0=en, 1=zh)
    if _MEMO_RE.search(txt):
        return 1
    lit_hits = len(_LITERARY_RE.findall(txt))
    rig_hits = len(_RIGOROUS_RE.findall(txt))
    if lit_hits >= 2 and lit_hits > rig_hits:
        return 3
    if rig_hits >= 2:
        return 0
    return 2


# =============================================================================
# Fixed-Point x10 `.loss_mask.bin` Writer Helper & Lossless Pre-Chunkers (WS-1)
# =============================================================================

_CTRL_STRIP_TABLE_V3: dict[int, None] = {
    **{c: None for c in range(32) if c not in (0x09, 0x0A)},
    0x7F: None,
}
_SENT_SPLIT_RE = re.compile(
    r'(?:[。！？；]+[”’」』）】]*\s*|[.!?;]+["\'”’\)\]]*(?:\s+|$))'
)


def _utf8_truncate(raw: bytes, max_bytes: int) -> bytes:
    """Truncate `raw` to `<= max_bytes` without ever splitting a multi-byte UTF-8 sequence."""
    if len(raw) <= max_bytes:
        return raw
    end = max(0, max_bytes)
    steps = 0
    while end > 0 and steps < 4 and (raw[end] & 0xC0) == 0x80:
        end -= 1
        steps += 1
    return raw[:end]


def encode_fixed_point_loss_mask(
    raw_bytes: bytes,
    *,
    zero_spans: Sequence[tuple[int, int]] = (),
    anchor_spans: Sequence[tuple[int, int]] = (),
    base_mask: Optional[bytes | np.ndarray] = None,
    highlight_syntax_delims: bool = True,
) -> bytes:
    """Encode byte-aligned fixed-point x10 `.loss_mask.bin` weights (`uint8`):
      - `0`  (0.0x): ignored bytes (prepended headers, prompts, observations, `zero_spans`)
      - `10` (1.0x): standard supervised bytes (upgrades legacy `1` -> `10`)
      - `15` (1.5x): structural delimiters `{ } ( ) [ ] : ; \\n` on supervised bytes
      - `30` (3.0x): high-salience anchor spans (`anchor_spans`)
    """
    n = len(raw_bytes)
    if n == 0:
        return b""

    if base_mask is not None:
        base_arr = (
            np.frombuffer(base_mask, dtype=np.uint8)
            if isinstance(base_mask, (bytes, bytearray))
            else np.asarray(base_mask, dtype=np.uint8)
        )
        if len(base_arr) != n:
            raise ValueError(f"base_mask length {len(base_arr)} != raw_bytes length {n}")
        mask = np.where(
            base_arr == LOSS_MASK_LEGACY_ONE,
            np.uint8(LOSS_MASK_NORMAL_X10),
            base_arr,
        ).astype(np.uint8, copy=True)
    else:
        mask = np.full(n, LOSS_MASK_NORMAL_X10, dtype=np.uint8)

    if highlight_syntax_delims:
        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        is_delim = _SYNTAX_DELIM_LUT[raw_arr]
        mask[(mask == LOSS_MASK_NORMAL_X10) & is_delim] = np.uint8(LOSS_MASK_SYNTAX_X10)

    for s, e in anchor_spans:
        cs = max(0, min(int(s), n))
        ce = max(cs, min(int(e), n))
        if ce > cs:
            mask[cs:ce] = np.uint8(LOSS_MASK_ANCHOR_X10)

    for s, e in zero_spans:
        cs = max(0, min(int(s), n))
        ce = max(cs, min(int(e), n))
        if ce > cs:
            mask[cs:ce] = np.uint8(LOSS_MASK_IGNORE)

    return mask.tobytes()


@dataclass(frozen=True)
class PreChunk:
    """Lossless pre-chunk emitted for static-grid packing (zero mid-doc truncation)."""

    raw_bytes: bytes
    loss_mask_bytes: bytes
    lang_id: int
    max_depth: int
    header_byte_len: int = 0
    chunk_index: int = 0


def _normalize_lossless_text(text: str) -> str:
    """Stage-1 NFC + LF normalization stripping disallowed ASCII control chars without dropping prose."""
    nfc = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return nfc.translate(_CTRL_STRIP_TABLE_V3)


def _split_sentences_preserve(para: str) -> list[str]:
    """Split a paragraph strictly at sentence-ending punctuation (. ! ? 。 ！ ？ ; ；)."""
    pieces: list[str] = []
    prev = 0
    for m in _SENT_SPLIT_RE.finditer(para):
        seg = para[prev : m.end()]
        if seg.strip():
            pieces.append(seg)
        prev = m.end()
    if prev < len(para):
        tail = para[prev:]
        if tail.strip():
            pieces.append(tail)
    return pieces


def chunk_prose_lossless(
    text: str,
    max_chunk_bytes: int = 3800,
    min_chunk_bytes: int = 1,
) -> list[bytes]:
    """Lossless paragraph- & sentence-aligned pre-chunker for natural language & formal math (`lang_id in {0,1,2}`).

    Splits long documents exceeding `max_chunk_bytes` ONLY at `\\n\\n` paragraph breaks or
    sentence-ending punctuation (`. ! ? 。 ！ ？ ; ；` followed by whitespace/newline or CJK continuation),
    never mid-sentence or mid-UTF-8 character, and preserves all content without dropping tails.
    """
    if max_chunk_bytes < 16:
        raise ValueError(f"max_chunk_bytes must be >= 16, got {max_chunk_bytes}")
    norm = _normalize_lossless_text(text).strip()
    if not norm:
        return []
    full_b = (norm + "\n").encode("utf-8", errors="strict")
    if len(full_b) <= max_chunk_bytes:
        return [full_b] if len(full_b) >= min_chunk_bytes else []

    paras = [p.strip() for p in re.split(r"\n{2,}", norm) if p.strip()]
    out: list[bytes] = []
    acc: list[str] = []
    acc_bytes = 0

    def _flush_acc() -> None:
        nonlocal acc, acc_bytes
        if not acc:
            return
        chunk = ("\n\n".join(acc) + "\n").encode("utf-8", errors="strict")
        if len(chunk) >= min_chunk_bytes:
            out.append(chunk)
        acc = []
        acc_bytes = 0

    for para in paras:
        p_b = (para + "\n").encode("utf-8", errors="strict")
        if len(p_b) > max_chunk_bytes:
            _flush_acc()
            sents = _split_sentences_preserve(para)
            s_acc: list[str] = []
            s_bytes = 0
            for sent in sents:
                s_clean = sent if s_acc else sent.lstrip()
                sb = s_clean.encode("utf-8", errors="strict")
                if len(sb) + 1 > max_chunk_bytes:
                    if s_acc:
                        chunk_s = ("".join(s_acc).strip() + "\n").encode("utf-8", errors="strict")
                        if len(chunk_s) >= min_chunk_bytes:
                            out.append(chunk_s)
                        s_acc, s_bytes = [], 0
                    # Fallback only if a single sentence has zero sentence terminators and exceeds max_chunk_bytes:
                    # prefer newline boundaries, then whitespace boundaries, never splitting a UTF-8 char.
                    rem_str = sent.strip()
                    while rem_str:
                        rem_b = (rem_str + "\n").encode("utf-8", errors="strict")
                        if len(rem_b) <= max_chunk_bytes:
                            if len(rem_b) >= min_chunk_bytes:
                                out.append(rem_b)
                            break
                        cut_b = _utf8_truncate(rem_b, max_chunk_bytes - 1)
                        cut_s = cut_b.decode("utf-8", errors="strict")
                        split_at = cut_s.rfind("\n")
                        if split_at <= 0:
                            split_at = cut_s.rfind(" ")
                        if split_at > 0:
                            piece_s = cut_s[:split_at].rstrip()
                            rem_str = rem_str[split_at:].lstrip()
                        else:
                            piece_s = cut_s
                            rem_str = rem_str[len(cut_s) :]
                        if piece_s:
                            pb_out = (piece_s + "\n").encode("utf-8", errors="strict")
                            if len(pb_out) >= min_chunk_bytes:
                                out.append(pb_out)
                    continue

                if s_acc and (s_bytes + len(sb) + 1) > max_chunk_bytes:
                    chunk_s = ("".join(s_acc).strip() + "\n").encode("utf-8", errors="strict")
                    if len(chunk_s) >= min_chunk_bytes:
                        out.append(chunk_s)
                    s_clean = sent.lstrip()
                    sb = s_clean.encode("utf-8", errors="strict")
                    s_acc = [s_clean]
                    s_bytes = len(sb)
                else:
                    s_acc.append(s_clean)
                    s_bytes += len(sb)
            if s_acc:
                chunk_s = ("".join(s_acc).strip() + "\n").encode("utf-8", errors="strict")
                if len(chunk_s) >= min_chunk_bytes:
                    out.append(chunk_s)
            continue

        sep = 2 if acc else 0
        if acc and (acc_bytes + sep + len(p_b)) > max_chunk_bytes:
            _flush_acc()
            acc = [para]
            acc_bytes = len(p_b)
        else:
            acc.append(para)
            acc_bytes += sep + len(p_b)

    _flush_acc()
    return out


def scan_code_lexical(code: str | bytes, lang_id: int) -> tuple[bool, int, list[int]]:
    """Lightweight lexical state machine for Python (`3`), SQL/JSON (`4`), and JavaScript (`5`).

    Tracks strings (incl. Python triple quotes, SQL `''` escapes, JS template literals w/ `${...}`),
    line/block comments, and `{}()[]` bracket stack.

    Returns:
      `(is_balanced_and_valid, max_depth, depth0_split_byte_offsets)`
    Rejects unbalanced brackets, unclosed strings/block comments, invalid UTF-8, or `max_depth > 16`.
    """
    if lang_id not in (3, 4, 5):
        return False, 0, []
    if isinstance(code, str):
        raw = code.encode("utf-8", errors="strict")
    else:
        raw = bytes(code)
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False, 0, []
    n = len(raw)
    if n == 0:
        return False, 0, []

    # Lexer modes
    M_CODE = 0
    M_LINE_COMMENT = 1
    M_BLOCK_COMMENT = 2
    M_SINGLE = 3
    M_DOUBLE = 4
    M_TRIPLE_SINGLE = 5
    M_TRIPLE_DOUBLE = 6
    M_TEMPLATE = 7

    mode = M_CODE
    stack: list[int] = []
    template_expr_depths: list[int] = []
    max_depth = 0
    splits: list[int] = []
    json_splits: list[int] = []

    at_line_start = True
    prev_py_decorator = False
    i = 0

    while i < n:
        c = raw[i]
        nxt = raw[i + 1] if (i + 1) < n else 0
        nxt2 = raw[i + 2] if (i + 2) < n else 0

        if mode == M_LINE_COMMENT:
            if c == 0x0A:
                mode = M_CODE
                at_line_start = True
            i += 1
            continue

        if mode == M_BLOCK_COMMENT:
            if c == 0x2A and nxt == 0x2F:  # */
                mode = M_CODE
                i += 2
            else:
                i += 1
            continue

        if mode == M_SINGLE:
            if lang_id == 4:
                if c == 0x27 and nxt == 0x27:  # ''
                    i += 2
                    continue
                if c == 0x5C:  # \
                    if i + 1 >= n:
                        return False, max_depth, []
                    i += 2
                    continue
                if c == 0x27:
                    mode = M_CODE
                i += 1
                continue
            if c == 0x5C:
                if i + 1 >= n:
                    return False, max_depth, []
                i += 2
                continue
            if c == 0x0A:
                return False, max_depth, []
            if c == 0x27:
                mode = M_CODE
            i += 1
            continue

        if mode == M_DOUBLE:
            if c == 0x5C:
                if i + 1 >= n:
                    return False, max_depth, []
                i += 2
                continue
            if c == 0x0A and lang_id != 4:
                return False, max_depth, []
            if c == 0x22:
                mode = M_CODE
            i += 1
            continue

        if mode == M_TRIPLE_SINGLE:
            if c == 0x5C:
                if i + 1 >= n:
                    return False, max_depth, []
                i += 2
                continue
            if c == 0x27 and nxt == 0x27 and nxt2 == 0x27:
                mode = M_CODE
                i += 3
                continue
            i += 1
            continue

        if mode == M_TRIPLE_DOUBLE:
            if c == 0x5C:
                if i + 1 >= n:
                    return False, max_depth, []
                i += 2
                continue
            if c == 0x22 and nxt == 0x22 and nxt2 == 0x22:
                mode = M_CODE
                i += 3
                continue
            i += 1
            continue

        if mode == M_TEMPLATE:
            if c == 0x5C:
                if i + 1 >= n:
                    return False, max_depth, []
                i += 2
                continue
            if c == 0x24 and nxt == 0x7B:  # ${
                template_expr_depths.append(len(stack))
                stack.append(0x7D)
                if len(stack) > max_depth:
                    max_depth = len(stack)
                    if max_depth > 16:
                        return False, max_depth, []
                mode = M_CODE
                at_line_start = False
                i += 2
                continue
            if c == 0x60:  # `
                mode = M_CODE
            i += 1
            continue

        # M_CODE
        if at_line_start:
            if not stack:
                if lang_id == 3:
                    if raw.startswith((b"def ", b"async def ", b"class "), i):
                        if i > 0 and not prev_py_decorator:
                            splits.append(i)
                        prev_py_decorator = False
                    elif c == 0x40:  # @decorator at top-level column 0
                        if i > 0 and not prev_py_decorator:
                            splits.append(i)
                        prev_py_decorator = True
                    elif c not in (0x0A, 0x0D, 0x20, 0x09, 0x23):  # non-blank, non-comment
                        prev_py_decorator = False
                elif lang_id == 5:
                    if raw.startswith((b"export ", b"function ", b"async function ", b"class "), i):
                        if i > 0:
                            splits.append(i)
            if c != 0x0A:
                at_line_start = False

        if c == 0x0A:
            at_line_start = True
            i += 1
            continue

        # Comments
        if lang_id == 3 and c == 0x23:  # #
            mode = M_LINE_COMMENT
            i += 1
            continue
        if lang_id == 4 and c == 0x2D and nxt == 0x2D:  # --
            mode = M_LINE_COMMENT
            i += 2
            continue
        if lang_id == 5 and c == 0x2F and nxt == 0x2F:  # //
            mode = M_LINE_COMMENT
            i += 2
            continue
        if lang_id in (4, 5) and c == 0x2F and nxt == 0x2A:  # /*
            mode = M_BLOCK_COMMENT
            i += 2
            continue

        # String / template openers
        if lang_id == 3 and c == 0x22 and nxt == 0x22 and nxt2 == 0x22:
            mode = M_TRIPLE_DOUBLE
            i += 3
            continue
        if lang_id == 3 and c == 0x27 and nxt == 0x27 and nxt2 == 0x27:
            mode = M_TRIPLE_SINGLE
            i += 3
            continue
        if c == 0x22:
            mode = M_DOUBLE
            i += 1
            continue
        if c == 0x27:
            mode = M_SINGLE
            i += 1
            continue
        if lang_id == 5 and c == 0x60:
            mode = M_TEMPLATE
            i += 1
            continue

        # SQL statement terminator `;\n` at depth == 0
        if lang_id == 4 and c == 0x3B and not stack:
            j = i + 1
            while j < n and raw[j] in (0x20, 0x09):
                j += 1
            if j < n and raw[j] == 0x0A:
                splits.append(j + 1)
            i += 1
            continue

        # Brackets
        if c == 0x7B:  # {
            stack.append(0x7D)
            if len(stack) > max_depth:
                max_depth = len(stack)
                if max_depth > 16:
                    return False, max_depth, []
        elif c == 0x28:  # (
            stack.append(0x29)
            if len(stack) > max_depth:
                max_depth = len(stack)
                if max_depth > 16:
                    return False, max_depth, []
        elif c == 0x5B:  # [
            stack.append(0x5D)
            if len(stack) > max_depth:
                max_depth = len(stack)
                if max_depth > 16:
                    return False, max_depth, []
        elif c == 0x7D:  # }
            if not stack or stack.pop() != 0x7D:
                return False, max_depth, []
            if lang_id == 5 and template_expr_depths and template_expr_depths[-1] == len(stack):
                template_expr_depths.pop()
                mode = M_TEMPLATE
            elif lang_id == 4 and not stack and (i + 1) < n and raw[i + 1] == 0x0A:
                json_splits.append(i + 2)
        elif c in (0x29, 0x5D):  # ) or ]
            if not stack or stack.pop() != c:
                return False, max_depth, []
            if lang_id == 4 and c == 0x5D and not stack and (i + 1) < n and raw[i + 1] == 0x0A:
                json_splits.append(i + 2)

        i += 1

    if mode not in (M_CODE, M_LINE_COMMENT):
        return False, max_depth, []
    if stack or template_expr_depths or max_depth > 16:
        return False, max_depth, []

    final_splits = splits if splits else json_splits
    return True, max_depth, final_splits


def extract_compact_code_header(
    raw_bytes: bytes,
    lang_id: int,
    max_header_bytes: int = 256,
) -> bytes:
    """Extract a compact, bracket-balanced file header (imports / schema / signatures) `<= max_header_bytes`."""
    if max_header_bytes <= 0 or not raw_bytes:
        return b""
    text = raw_bytes.decode("utf-8", errors="strict")
    lines = text.splitlines()

    if lang_id == 3:
        header_lines: list[str] = []
        used = 0
        # 1. Collect balanced top-level import lines
        for ln in lines:
            if ln.startswith(("import ", "from ")):
                cand = ln.rstrip() + "\n"
                cb = cand.encode("utf-8", errors="strict")
                ok, depth, _ = scan_code_lexical(cb, 3)
                if ok and depth <= 16 and (used + len(cb)) <= max_header_bytes:
                    header_lines.append(cand)
                    used += len(cb)
        if header_lines:
            return "".join(header_lines).encode("utf-8", errors="strict")

        # 2. Fallback: compact top-level signature context comment (always balanced & valid Python AST)
        sigs: list[str] = []
        for ln in lines:
            if ln.startswith(("def ", "async def ", "class ")):
                sigs.append(ln.split(":", 1)[0].strip())
                if len(sigs) >= 4:
                    break
        if sigs:
            comment = "# context: " + "; ".join(sigs)
            cb = _utf8_truncate(comment.encode("utf-8", errors="strict"), max_header_bytes - 1) + b"\n"
            return cb
        return b""

    if lang_id == 5:
        header_lines = []
        used = 0
        for ln in lines:
            if ln.startswith("import ") or (ln.startswith(("const ", "let ", "var ")) and "require(" in ln):
                cand = ln.rstrip() + "\n"
                cb = cand.encode("utf-8", errors="strict")
                ok, depth, _ = scan_code_lexical(cb, 5)
                if ok and depth <= 16 and (used + len(cb)) <= max_header_bytes:
                    header_lines.append(cand)
                    used += len(cb)
        if header_lines:
            return "".join(header_lines).encode("utf-8", errors="strict")

        sigs = []
        for ln in lines:
            if ln.startswith(("export ", "function ", "async function ", "class ")):
                sigs.append(ln.split("{", 1)[0].strip())
                if len(sigs) >= 4:
                    break
        if sigs:
            comment = "// context: " + "; ".join(sigs)
            cb = _utf8_truncate(comment.encode("utf-8", errors="strict"), max_header_bytes - 1) + b"\n"
            return cb
        return b""

    if lang_id == 4:
        # Collect leading balanced CREATE TABLE statements up to max_header_bytes
        ok, _, splits = scan_code_lexical(raw_bytes, 4)
        if ok:
            bounds = [0] + sorted(set(splits)) + [len(raw_bytes)]
            stmts: list[bytes] = []
            used = 0
            for s, e in zip(bounds[:-1], bounds[1:]):
                stmt = raw_bytes[s:e].lstrip(b"\n")
                if not stmt:
                    continue
                upper = stmt.decode("utf-8", errors="ignore").lstrip().upper()
                if upper.startswith("CREATE TABLE") or upper.startswith("-- SCHEMA"):
                    if not stmt.endswith(b"\n"):
                        stmt += b"\n"
                    s_ok, s_d, _ = scan_code_lexical(stmt, 4)
                    if s_ok and s_d <= 16 and (used + len(stmt)) <= max_header_bytes:
                        stmts.append(stmt)
                        used += len(stmt)
            if stmts:
                return b"".join(stmts)
        # Fallback: compact schema/table comment
        table_names = re.findall(r"(?i)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)", text)
        if table_names:
            comment = "-- schema tables: " + ", ".join(dict.fromkeys(table_names))
            return _utf8_truncate(comment.encode("utf-8", errors="strict"), max_header_bytes - 1) + b"\n"
        return b""

    return b""


def chunk_code_lossless(
    code: str,
    lang_id: int,
    max_chunk_bytes: int = 3800,
    max_header_bytes: int = 256,
    highlight_syntax_delims: bool = True,
) -> list[PreChunk]:
    """Syntax-aligned lossless pre-chunker for Python (`3`), SQL/JSON (`4`), and JavaScript (`5`).

    - Splits ONLY at bracket-depth == 0 top-level boundaries (`\\ndef `, `\\nasync def `, `\\nclass `,
      `\\nexport `, `\\nfunction `, SQL `;\\n`).
    - Automatically prepends compact file header imports/signatures (`<= max_header_bytes`, default 256 B)
      to continuation sub-chunks (`chunk_index > 0`) with corresponding `.loss_mask.bin` bytes set to `0`.
    - Guarantees every emitted chunk is 100% valid UTF-8, 100% bracket-balanced, and `max_depth <= 16`.
    """
    if lang_id not in (3, 4, 5):
        raise ValueError(f"chunk_code_lossless requires lang_id in {{3, 4, 5}}, got {lang_id}")
    if max_chunk_bytes < 64:
        raise ValueError(f"max_chunk_bytes must be >= 64, got {max_chunk_bytes}")

    norm = _normalize_lossless_text(code).strip()
    if not norm:
        return []
    raw = (norm + "\n").encode("utf-8", errors="strict")

    ok, full_depth, splits = scan_code_lexical(raw, lang_id)
    if not ok or full_depth > 16:
        return []

    if len(raw) <= max_chunk_bytes:
        if lang_id in (3, 5) and full_depth < 1:
            return []
        mask = encode_fixed_point_loss_mask(raw, highlight_syntax_delims=highlight_syntax_delims)
        return [
            PreChunk(
                raw_bytes=raw,
                loss_mask_bytes=mask,
                lang_id=lang_id,
                max_depth=full_depth,
                header_byte_len=0,
                chunk_index=0,
            )
        ]

    # Partition into depth-0 atomic top-level blocks
    bounds = sorted({0, len(raw), *(p for p in splits if 0 < p < len(raw))})
    blocks: list[bytes] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        blk = raw[s:e]
        if blk.strip():
            blocks.append(blk)
    if not blocks:
        return []

    eff_max_hdr = min(max_header_bytes, max(0, max_chunk_bytes // 4))
    header = extract_compact_code_header(raw, lang_id, max_header_bytes=eff_max_hdr)

    chunks: list[PreChunk] = []
    acc: list[bytes] = []
    acc_len = 0

    def _emit_current() -> None:
        nonlocal acc, acc_len
        if not acc:
            return
        body = b"".join(acc).lstrip(b"\n")
        acc = []
        acc_len = 0
        if not body.strip():
            return
        if not body.endswith(b"\n"):
            body += b"\n"

        chunk_idx = len(chunks)
        use_hdr = (
            chunk_idx > 0
            and len(header) > 0
            and not body.startswith(header)
            and (len(header) + len(body)) <= max_chunk_bytes
        )
        hdr_len = len(header) if use_hdr else 0
        chunk_bytes = (header + body) if use_hdr else body

        c_ok, c_depth, _ = scan_code_lexical(chunk_bytes, lang_id)
        if not c_ok or c_depth > 16:
            return
        if lang_id in (3, 5) and c_depth < 1:
            return
        mask = encode_fixed_point_loss_mask(
            chunk_bytes,
            zero_spans=((0, hdr_len),) if hdr_len > 0 else (),
            highlight_syntax_delims=highlight_syntax_delims,
        )
        chunks.append(
            PreChunk(
                raw_bytes=chunk_bytes,
                loss_mask_bytes=mask,
                lang_id=lang_id,
                max_depth=c_depth,
                header_byte_len=hdr_len,
                chunk_index=chunk_idx,
            )
        )

    for blk in blocks:
        next_idx = len(chunks)
        reserved_hdr = len(header) if next_idx > 0 else 0
        budget = max_chunk_bytes - reserved_hdr
        blk_clean = blk if acc else blk.lstrip(b"\n")
        if len(blk_clean) > budget:
            _emit_current()
            # Re-evaluate against fresh chunk budget after flushing
            next_idx = len(chunks)
            reserved_hdr = len(header) if next_idx > 0 else 0
            budget = max_chunk_bytes - reserved_hdr
            blk_clean = blk.lstrip(b"\n")
            if len(blk_clean) > budget:
                # Single unsplittable block > budget: never break brackets mid-block
                continue
            acc = [blk_clean]
            acc_len = len(blk_clean)
            continue

        if acc and (acc_len + len(blk_clean)) > budget:
            _emit_current()
            next_idx = len(chunks)
            reserved_hdr = len(header) if next_idx > 0 else 0
            budget = max_chunk_bytes - reserved_hdr
            blk_clean = blk.lstrip(b"\n")
            if len(blk_clean) > budget:
                continue
            acc = [blk_clean]
            acc_len = len(blk_clean)
        else:
            acc.append(blk_clean)
            acc_len += len(blk_clean)

    _emit_current()
    return chunks


def prechunk_document(
    text: str,
    lang_id: int,
    max_chunk_bytes: int = 3800,
    max_header_bytes: int = 256,
    highlight_syntax_delims: bool = True,
) -> list[PreChunk]:
    """Unified lossless pre-chunker routing `lang_id in {0,1,2}` to prose and `{3,4,5}` to lexical code."""
    if lang_id in (3, 4, 5):
        return chunk_code_lossless(
            text,
            lang_id=lang_id,
            max_chunk_bytes=max_chunk_bytes,
            max_header_bytes=max_header_bytes,
            highlight_syntax_delims=highlight_syntax_delims,
        )
    prose_chunks = chunk_prose_lossless(text, max_chunk_bytes=max_chunk_bytes, min_chunk_bytes=1)
    return [
        PreChunk(
            raw_bytes=b,
            loss_mask_bytes=encode_fixed_point_loss_mask(b, highlight_syntax_delims=highlight_syntax_delims),
            lang_id=lang_id,
            max_depth=0,
            header_byte_len=0,
            chunk_index=idx,
        )
        for idx, b in enumerate(prose_chunks)
    ]


def stream_v3_flat_source(
    repo_id: str,
    *,
    max_chunk_bytes: int = 3800,
    min_chunk_bytes: int = 96,
) -> Iterator[tuple[bytes, bytes, int, int]]:
    """Stream any registered 2025-2026 high-purity source from `V3_FLAT_SOURCE_DESCRIPTORS`.

    Yields `(raw_bytes, fixed_point_x10_loss_mask_bytes, source_id, max_depth)`.
    Gracefully returns if offline/gated/unavailable.
    """
    desc = V3_FLAT_SOURCE_DESCRIPTORS.get(repo_id)
    if desc is None:
        raise KeyError(f"Unregistered v3 source descriptor: {repo_id!r}")
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        if desc.parquet_glob:
            ds = load_dataset(
                "parquet",
                data_files=desc.parquet_glob,
                split=desc.split,
                streaming=True,
                **hf_token_kwargs(),
            )
        else:
            load_args = [desc.repo_id] + ([desc.config_name] if desc.config_name else [])
            extra_kw = {"data_dir": desc.data_dir} if desc.data_dir else {}
            ds = load_dataset(*load_args, split=desc.split, streaming=True, **extra_kw, **hf_token_kwargs())
        for row in ds:
            text_val = ""
            for field in desc.text_fields:
                v = row.get(field)
                if isinstance(v, str) and v.strip():
                    text_val = v
                    break
                if field == "conversations" and isinstance(v, list):
                    parts = [
                        str(turn.get("value") or turn.get("content") or "").strip()
                        for turn in v
                        if isinstance(turn, dict)
                    ]
                    joined = "\n\n".join(p for p in parts if p)
                    if joined:
                        text_val = joined
                        break
            if not text_val:
                continue
            for pc in prechunk_document(text_val, lang_id=desc.lang_id, max_chunk_bytes=max_chunk_bytes):
                if len(pc.raw_bytes) >= min_chunk_bytes:
                    yield pc.raw_bytes, pc.loss_mask_bytes, desc.source_id, pc.max_depth
    except Exception as exc:
        print(f"  [INFO] v3 source {repo_id} skipped ({type(exc).__name__}); falling back.", flush=True)
    finally:
        gc.collect()


def _chunk_bounded_paragraphs(text: str, min_b: int, max_b: int) -> Iterator[bytes]:
    cleaned = _clean_nfc(text)
    if cleaned is None:
        return
    paras = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    acc: list[str] = []
    acc_len = 0
    for p in paras:
        pb = (p + "\n").encode("utf-8")
        if len(pb) > max_b:
            if acc_len >= min_b:
                yield ("\n\n".join(acc) + "\n").encode("utf-8")
            acc, acc_len = [], 0
            # Split cleanly on sentence boundaries only (never mid-sentence!)
            sents = re.split(r"(?<=[.!?。！？；;])\s*", p)
            s_acc: list[str] = []
            s_len = 0
            for st in sents:
                st = st.strip()
                if not st:
                    continue
                sb = (st + " ").encode("utf-8")
                if len(sb) > max_b:
                    continue
                if s_len + len(sb) > max_b and s_len >= min_b:
                    out_s = (" ".join(s_acc).strip() + "\n").encode("utf-8")
                    if min_b <= len(out_s) <= max_b:
                        yield out_s
                    s_acc, s_len = [st], len(sb)
                else:
                    s_acc.append(st)
                    s_len += len(sb)
            if s_acc and s_len >= min_b:
                out_s = (" ".join(s_acc).strip() + "\n").encode("utf-8")
                if min_b <= len(out_s) <= max_b:
                    yield out_s
            continue
        sep = 2 if acc else 0
        if acc_len + sep + len(pb) > max_b and acc:
            if acc_len >= min_b:
                yield ("\n\n".join(acc) + "\n").encode("utf-8")
            acc, acc_len = [p], len(pb)
        else:
            acc.append(p)
            acc_len += sep + len(pb)
    if acc and acc_len >= min_b:
        out_b = ("\n\n".join(acc) + "\n").encode("utf-8")
        if len(out_b) <= max_b:
            yield out_b


# =============================================================================
# Online Hugging Face Parquet Streamers (Graceful Fallback on Offline/Gated)
# =============================================================================


def stream_en_finephrase_s1() -> Iterator[tuple[bytes, int]]:
    """Stage 1 (`64..512 B`): `HuggingFaceFW/finephrase` high-clarity rephrased paragraphs."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        ds = load_dataset("HuggingFaceFW/finephrase", "all", split="train", streaming=True, **hf_token_kwargs())
        for row in ds:
            raw = row.get("text") or row.get("rephrased_text") or row.get("content") or ""
            for chunk in _chunk_bounded_paragraphs(str(raw), min_b=80, max_b=500):
                yield chunk, SID_FINEPHRASE_S1
    except Exception as exc:
        print(f"  [INFO] finephrase stream skipped ({type(exc).__name__}); using fallback.", flush=True)
    finally:
        gc.collect()


def stream_en_us_pd_books_s1() -> Iterator[tuple[bytes, int]]:
    """Stage 1 (`90..512 B`): `storytracer/US-PD-Books` literary & classic non-fiction paragraphs."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        ds = load_dataset("storytracer/US-PD-Books", split="train", streaming=True, **hf_token_kwargs())
        for row in ds:
            raw = row.get("text") or ""
            if not isinstance(raw, str) or len(raw) < 200:
                continue
            # Sample clean middle excerpt to skip OCR title/TOC pages
            mid = raw[min(800, len(raw) // 5) : min(len(raw), 16000)]
            for chunk in _chunk_bounded_paragraphs(mid, min_b=96, max_b=500):
                yield chunk, SID_US_PD_BOOKS_S1
    except Exception as exc:
        print(f"  [INFO] US-PD-Books stream skipped ({type(exc).__name__}); using fallback.", flush=True)
    finally:
        gc.collect()


def stream_en_cosmopedia_s2() -> Iterator[tuple[bytes, int]]:
    """Stage 2/3 (`600..3600 B`): `HuggingFaceTB/cosmopedia-v2` rhetoric, philosophy, history & STEM."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        ds = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True, **hf_token_kwargs())
        for row in ds:
            raw = row.get("text") or ""
            for chunk in _chunk_bounded_paragraphs(str(raw), min_b=560, max_b=3600):
                yield chunk, SID_COSMOPEDIA_S2
    except Exception as exc:
        print(f"  [INFO] cosmopedia-v2 stream skipped ({type(exc).__name__}); using fallback.", flush=True)
    finally:
        gc.collect()


def stream_zh_cci3_hq_s2() -> Iterator[tuple[bytes, int]]:
    """Stage 2/3 (`600..3800 B`): `BAAI/CCI3-HQ` when HF_TOKEN authorized (silent fallback if gated)."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        ds = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True, **hf_token_kwargs())
        for row in ds:
            raw = row.get("text") or row.get("content") or ""
            for chunk in _chunk_bounded_paragraphs(str(raw), min_b=560, max_b=3800):
                yield chunk, SID_CCI3_HQ_S2
    except Exception as exc:
        print(
            f"  [INFO] BAAI/CCI3-HQ gated/unavailable ({type(exc).__name__}); falling back to Fineweb-Edu-ZH 4_5+V3.",
            flush=True,
        )
    finally:
        gc.collect()


def stream_formal_folio_and_logic_s2(max_folio_epochs: int = 3) -> Iterator[tuple[bytes, int]]:
    """Stage 2 (`lang_id=2`): `tasksource/folio` (`<= 3 epochs`), `tasksource/proofwriter`, `lucasmccabe/logiqa`."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        tok_kw = hf_token_kwargs()

        # 1. FOLIO (strictly capped at <= 3 epochs to prevent memorization)
        folio_rows: list[bytes] = []
        try:
            ds_f = load_dataset("tasksource/folio", split="train", streaming=True, **tok_kw)
            for row in ds_f:
                prem = str(row.get("premises") or "").strip()
                fol = str(row.get("premises-FOL") or "").strip()
                conc = str(row.get("conclusion") or "").strip()
                lbl = str(row.get("label") or "").strip()
                if not prem or not conc:
                    continue
                doc = f"Premises:\n{prem}\n"
                if fol:
                    doc += f"FOL:\n{fol}\n"
                doc += f"Hypothesis: {conc}\nVerdict: {lbl}\n"
                cleaned = _clean_nfc(doc)
                if cleaned:
                    b = cleaned.encode("utf-8")
                    if 96 <= len(b) <= 2048:
                        folio_rows.append(b)
                        yield b, SID_FOLIO_LOGIC_S2
        except Exception:
            pass

        for _ in range(max(0, min(max_folio_epochs, 3) - 1)):
            for b in folio_rows:
                yield b, SID_FOLIO_LOGIC_S2

        # 2. ProofWriter deductive chains
        try:
            ds_pw = load_dataset("tasksource/proofwriter", split="train", streaming=True, **tok_kw)
            for row in ds_pw:
                theory = str(row.get("theory") or row.get("context") or "").strip()
                q = str(row.get("question") or "").strip()
                ans = str(row.get("answer") or "").strip()
                proof = str(row.get("proof") or "").strip()
                if not theory or not q:
                    continue
                doc = f"Axioms: {theory}\nQuery: {q}\n"
                if proof:
                    doc += f"Derivation: {proof}\n"
                doc += f"Conclusion: {ans}\n"
                cleaned = _clean_nfc(doc)
                if cleaned:
                    b = cleaned.encode("utf-8")
                    if 100 <= len(b) <= 2048:
                        yield b, SID_FOLIO_LOGIC_S2
        except Exception:
            pass

        # 3. LogiQA deductive arguments (direct parquet branch compatible with datasets>=4.0)
        try:
            ds_lq = load_dataset(
                "parquet",
                data_files="hf://datasets/lucasmccabe/logiqa@refs/convert/parquet/default/train/*.parquet",
                split="train",
                streaming=True,
                **tok_kw,
            )
            for row in ds_lq:
                ctx = str(row.get("context") or "").strip()
                query = str(row.get("query") or "").strip()
                opts = row.get("options") or []
                corr = row.get("correct_option")
                if not ctx or not query or not isinstance(opts, list) or not isinstance(corr, int):
                    continue
                if 0 <= corr < len(opts):
                    doc = f"Context: {ctx}\nQuestion: {query}\nValid Inference: {opts[corr]}\n"
                    cleaned = _clean_nfc(doc)
                    if cleaned:
                        b = cleaned.encode("utf-8")
                        if 100 <= len(b) <= 2048:
                            yield b, SID_LOGIQA_RECLOR_S2
        except Exception:
            pass
    except ImportError:
        return
    finally:
        gc.collect()


def stream_formal_finemath_s2_s3() -> Iterator[tuple[bytes, int]]:
    """Stage 2/3 (`lang_id=2`): `HuggingFaceTB/finemath` (`finemath-4plus`) mathematical proofs."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True, **hf_token_kwargs())
        for row in ds:
            raw = str(row.get("text") or "")
            for chunk in _chunk_bounded_paragraphs(raw, min_b=160, max_b=3600):
                sid = SID_FINEMATH_S2 if len(chunk) <= 1800 else SID_COMPLEX_PROOFS_S3
                yield chunk, sid
    except Exception as exc:
        print(f"  [INFO] finemath-4plus skipped ({type(exc).__name__}); using OpenWebMath/offline proofs.", flush=True)
    finally:
        gc.collect()


def stream_python_edu_and_swallow() -> Iterator[tuple[str, int]]:
    """Stage 1 `smollm-corpus (python-edu)` (`sid=24`) + Stage 2 `tokyotech-llm/swallow-code-v2` (`sid=32`)."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        tok_kw = hf_token_kwargs()
        try:
            ds_edu = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True, **tok_kw)
            for row in ds_edu:
                code = str(row.get("text") or row.get("content") or "").strip()
                if 64 <= len(code.encode("utf-8", errors="ignore")) <= 512:
                    yield code, SID_PY_EDU_SHORT_S1
        except Exception as exc:
            print(f"  [INFO] smollm-corpus/python-edu skipped ({type(exc).__name__}).", flush=True)

        for repo, cfg_name in (("tokyotech-llm/swallow-code-v2", "stage4-llm-rewrite"), ("tokyotech-llm/swallow-code", None)):
            try:
                args_sw = [repo] + ([cfg_name] if cfg_name else [])
                ds_sw = load_dataset(*args_sw, split="train", streaming=True, **tok_kw)
                for row in ds_sw:
                    code = str(row.get("text") or row.get("improved_text") or row.get("code") or row.get("content") or "").strip()
                    if 160 <= len(code.encode("utf-8", errors="ignore")) <= 4096:
                        yield code, SID_SWALLOW_PY_S2
                break
            except Exception:
                continue
    except ImportError:
        return
    finally:
        gc.collect()


def stream_sql_create_context_s2() -> Iterator[tuple[bytes, int]]:
    """Stage 1/2 (`lang_id=4`): `b-mc2/sql-create-context` schema-grounded SQL queries."""
    try:
        from datasets import load_dataset
        from data.hf_auth import hf_token_kwargs

        ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True, **hf_token_kwargs())
        for row in ds:
            ctx = str(row.get("context") or "").strip()
            q = str(row.get("question") or "").strip()
            ans = str(row.get("answer") or "").strip()
            if not ctx or not ans:
                continue
            doc = f"-- Schema\n{ctx};\n-- Intent: {q}\n{ans};\n"
            cleaned = _clean_nfc(doc)
            if cleaned:
                b = cleaned.encode("utf-8")
                if 64 <= len(b) <= 2048:
                    sid = SID_SQL_JSON_BASIC_S1 if len(b) <= 450 and "JOIN" not in ans.upper() else SID_SQL_CONTEXT_S2
                    yield b, sid
    except Exception as exc:
        print(f"  [INFO] sql-create-context skipped ({type(exc).__name__}); using procedural SQL+Schema.", flush=True)
    finally:
        gc.collect()


# =============================================================================
# Deterministic Multi-Tier Offline Generators (Smoke & Network-Free Fallback)
# =============================================================================


def stream_offline_cognitive_domain(lang_id: int, seed: int = 2026) -> Iterator[tuple[bytes, int]]:
    """Infinite deterministic multi-stage generator for `lang_id` in `0..5` (100% zero-Rust, AST-clean)."""
    rng = random.Random(seed + lang_id * 1009)
    idx = 0

    if lang_id == 0:
        topics = ("thermodynamic entropy", "distributed consensus", "urban water viaducts", "monetary velocity", "optical diffraction")
        places = ("autumn harbor", "cedar library", "slate courtyard", "foggy ridge", "lantern-lit pier")
        while True:
            idx += 1
            if idx % 3 == 1:
                # Tier 1 literary rhythm & nested punctuation (< 500B)
                pl = rng.choice(places)
                hr = rng.randint(4, 11)
                txt = (
                    f'At {hr} o\'clock beside the {pl}, Margaret closed her leather ledger—its brass clasp clicking softly—and asked, '
                    f'"Did the courier leave \'Batch-{idx % 97}\' on the oak bench, or beside the rain barrel?" '
                    f"Only the wind answered across the wet flagstones.\n"
                )
                yield txt.encode("utf-8"), SID_US_PD_BOOKS_S1
            elif idx % 3 == 2:
                # Tier 1 crisp definition (< 500B)
                tp = rng.choice(topics)
                k = rng.randint(2, 19)
                txt = (
                    f"Principle #{idx}: When analyzing {tp} across {k} coupled subsystems, conservation constraints dictate "
                    f"that any local flux deficit must be balanced either by boundary inflow or internal state depletion; "
                    f"therefore, steady-state equilibrium holds if and only if net divergence vanishes.\n"
                )
                yield txt.encode("utf-8"), SID_FINEPHRASE_S1
            else:
                # Tier 2/3 multi-paragraph discourse (700..1500B)
                tp = rng.choice(topics)
                n_obs = rng.randint(12, 95)
                txt = (
                    f"Comparative Analysis of {tp.title()} (Case {idx})\n\n"
                    f"First, consider why naive linear extrapolation fails across {n_obs} empirical trials. "
                    f"Whenever feedback delay exceeds half the natural oscillation period, corrective control amplifies rather than dampens phase error.\n\n"
                    f"However, introducing a bounded damping reserve alters the regime entirely. "
                    f"Instead of oscillating wildly under peak load, each node sheds non-essential work early, preserving invariant throughput across the bottleneck.\n\n"
                    f"Consequently, structural resilience emerges not from infinite capacity, but from explicit backpressure boundaries.\n"
                )
                yield txt.encode("utf-8"), SID_COSMOPEDIA_S2

    if lang_id == 1:
        scenes = ("青石老街", "江畔渡口", "松风书院", "冬夜驿站", "竹林山房")
        concepts = ("供应链反脆弱机制", "货币流动性传导", "城市微气候热岛效应", "分布式账本状态收敛", "历史制度路径依赖")
        while True:
            idx += 1
            if idx % 3 == 1:
                sc = rng.choice(scenes)
                txt = (
                    f"暮色漫过{sc}的屋檐时，老匠人放下手中的铜尺——尺背还留着第{idx}道细密的刻痕——轻声问道："
                    f"“昨日送来的那批‘云纹青砖’，是先砌东侧风火墙，还是留着修补廊下水渠？”窗外细雨无声，灯花微微一晃。\n"
                )
                yield txt.encode("utf-8"), SID_ZH_PROSE_SHORT_S1
            elif idx % 3 == 2:
                cp = rng.choice(concepts)
                m_val = rng.randint(3, 28)
                txt = (
                    f"定义辨析（第{idx}则）：在评估{cp}时，若观测窗口短于{m_val}个完整周期，局部均值往往掩盖尾部风险；"
                    f"因此，唯有先确立边界守恒条件，再追踪异常扰动的传导方向，才能区分真实因果与统计巧合。\n"
                )
                yield txt.encode("utf-8"), SID_ZH_PROSE_SHORT_S1
            else:
                cp = rng.choice(concepts)
                ratio = rng.randint(15, 85)
                txt = (
                    f"论{cp}中的结构张力（样本编号 {idx}）\n\n"
                    f"表面上看，将系统冗余压缩{ratio}%似乎能在短期内显著压低账面成本。"
                    f"然而，一旦外部冲击突破稳态阈值，缺乏缓冲层的单链条结构便会迅速发生级联阻塞。\n\n"
                    f"反观具备分层隔离机制的体系，其关键不在算力或库存的堆砌，而在于明确划定局部失效的防火墙。"
                    f"当局部节点超载时，上游主动降级分流，核心账本与关键契约依然保持严密自洽。\n\n"
                    f"真正的工程与治理智慧，往往体现在对失控边界的敬畏与前置约束之中。\n"
                )
                yield txt.encode("utf-8"), SID_CCI3_HQ_S2

    if lang_id == 2:
        entities = ("Scholar", "Auditor", "Validator", "NodeA", "Curator", "Archivist")
        preds = ("verified", "deterministic", "monotonic", "auditable", "consistent", "bounded")
        while True:
            idx += 1
            e_name = rng.choice(entities) + str(idx % 50)
            p1, p2, p3 = rng.sample(preds, 3)
            a = rng.randint(3, 40)
            b = rng.randint(2, 25)
            if idx % 2 == 0:
                doc = (
                    f"Premises:\n"
                    f"1. Every {p1} system is {p2}.\n"
                    f"2. No {p2} system violates invariant I_{idx % 19} when load <= {a * b}.\n"
                    f"3. {e_name} is {p1} and operates at load {a * (b - 1)}.\n"
                    f"FOL:\n"
                    f"∀x ({p1}(x) -> {p2}(x)); ∀x ({p2}(x) ∧ Load_ok(x) -> Safe(x))\n"
                    f"Derivation: {p1}({e_name}) -> {p2}({e_name}) -> Safe({e_name}) since {a * (b - 1)} <= {a * b}.\n"
                    f"Verdict: True ({p3} check not required).\n"
                )
                yield doc.encode("utf-8"), SID_FOLIO_LOGIC_S2
            else:
                n = rng.randint(4, 30)
                s_val = n * (n + 1) // 2
                doc = (
                    f"Lemma {idx}: Arithmetic Series Parity Bound.\n"
                    f"Let S(n) = 1 + 2 + ... + n = n(n + 1)/2. For n = {n}, S({n}) = {s_val}.\n"
                    f"Proof: Since n and n + 1 are consecutive integers, exactly one is even; "
                    f"hence n(n + 1) is divisible by 2, and S({n}) + {a} = {s_val + a} strictly exceeds {s_val}.\n"
                )
                yield doc.encode("utf-8"), SID_FINEMATH_S2

    if lang_id == 3:
        while True:
            idx += 1
            fn_suf = f"_{idx}"
            k_const = (idx % 17) + 2
            if idx % 2 == 1:
                # Stage 1 clean stdlib primitive (<= 512B, depth <= 6)
                code = (
                    f"def clamp_delta{fn_suf}(values: list[int], floor: int = 0, cap: int = {k_const * 10}) -> list[int]:\n"
                    f'    """Clamp each integer into [floor, cap] without mutating input."""\n'
                    f"    if floor > cap:\n"
                    f"        raise ValueError('floor must not exceed cap')\n"
                    f"    return [min(cap, max(floor, v)) for v in values]\n"
                )
                yield code.encode("utf-8"), SID_PY_EDU_SHORT_S1
            else:
                # Stage 2 modular self-contained class + function
                code = (
                    f"class WindowAggregator{idx}:\n"
                    f'    """Bounded O(1)-step rolling sum tracker with strict capacity invariant."""\n\n'
                    f"    def __init__(self, capacity: int = {k_const}) -> None:\n"
                    f"        if capacity <= 0:\n"
                    f"            raise ValueError('capacity must be positive')\n"
                    f"        self.capacity = capacity\n"
                    f"        self._ring: list[int] = []\n"
                    f"        self._total = 0\n\n"
                    f"    def push(self, sample: int) -> float:\n"
                    f"        if len(self._ring) == self.capacity:\n"
                    f"            self._total -= self._ring.pop(0)\n"
                    f"        self._ring.append(sample)\n"
                    f"        self._total += sample\n"
                    f"        return self._total / len(self._ring)\n"
                )
                yield code.encode("utf-8"), SID_SWALLOW_PY_S2

    if lang_id == 4:
        tables = ("orders", "shipments", "invoices", "telemetry_events", "subscriptions")
        regions = ("APAC", "EMEA", "NA", "LATAM")
        while True:
            idx += 1
            tbl = rng.choice(tables)
            reg = rng.choice(regions)
            thr = (idx % 400 + 1) * 50
            if idx % 3 == 1:
                doc = (
                    f"-- Schema #{idx}\n"
                    f"CREATE TABLE {tbl}_{idx} (id BIGINT PRIMARY KEY, region TEXT NOT NULL, cents BIGINT NOT NULL);\n"
                    f"-- Intent: list active rows in {reg} above {thr}\n"
                    f"SELECT id, cents FROM {tbl}_{idx} WHERE region = '{reg}' AND cents >= {thr} ORDER BY cents DESC LIMIT 25;\n"
                )
                yield doc.encode("utf-8"), SID_SQL_JSON_BASIC_S1
            elif idx % 3 == 2:
                payload = {
                    "schema_version": 2,
                    "task_id": f"spec-{idx}",
                    "region": reg,
                    "constraints": {"min_cents": thr, "allow_partial": False, "max_retries": 3},
                }
                doc = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
                yield doc.encode("utf-8"), SID_SQL_JSON_BASIC_S1
            else:
                doc = (
                    f"-- Schema #{idx}\n"
                    f"CREATE TABLE accounts_{idx} (acct_id BIGINT PRIMARY KEY, tier INT NOT NULL, region TEXT NOT NULL);\n"
                    f"CREATE TABLE {tbl}_{idx} (tx_id BIGINT PRIMARY KEY, acct_id BIGINT NOT NULL, cents BIGINT NOT NULL, ok INT NOT NULL);\n"
                    f"-- Intent: per-tier settled volume in {reg} exceeding {thr * 10}\n"
                    f"WITH settled AS (\n"
                    f"    SELECT a.tier, t.cents\n"
                    f"    FROM {tbl}_{idx} AS t\n"
                    f"    INNER JOIN accounts_{idx} AS a ON a.acct_id = t.acct_id\n"
                    f"    WHERE a.region = '{reg}' AND t.ok = 1\n"
                    f")\n"
                    f"SELECT tier, COUNT(*) AS tx_cnt, SUM(cents) AS total_cents\n"
                    f"FROM settled\n"
                    f"GROUP BY tier\n"
                    f"HAVING SUM(cents) > {thr * 10}\n"
                    f"ORDER BY total_cents DESC;\n"
                )
                yield doc.encode("utf-8"), SID_SQL_CTE_SCHEMA_S3

    if lang_id == 5:
        while True:
            idx += 1
            suf = idx
            cap = (idx % 15 + 2) * 8
            if idx % 2 == 1:
                code = (
                    f"function normalizeScores{suf}(items, ceiling = {cap}) {{\n"
                    f"  if (!Array.isArray(items) || ceiling <= 0) {{\n"
                    f"    return [];\n"
                    f"  }}\n"
                    f"  return items.map((x) => Math.min(1, Math.max(0, x / ceiling)));\n"
                    f"}}\n"
                )
                yield code.encode("utf-8"), SID_JS_ES6_SHORT_S1
            else:
                code = (
                    f"/**\n"
                    f" * Deduplicate records by string key while preserving first-seen order in O(n).\n"
                    f" */\n"
                    f"export function dedupByKey{suf}(records, keyField = 'id') {{\n"
                    f"  if (!Array.isArray(records)) {{\n"
                    f"    throw new TypeError('records must be an array');\n"
                    f"  }}\n"
                    f"  const seen = new Set();\n"
                    f"  const result = [];\n"
                    f"  for (const row of records) {{\n"
                    f"    if (row !== null && typeof row === 'object' && Object.hasOwn(row, keyField)) {{\n"
                    f"      const k = row[keyField];\n"
                    f"      if (!seen.has(k)) {{\n"
                    f"        seen.add(k);\n"
                    f"        result.push(row);\n"
                    f"      }}\n"
                    f"    }}\n"
                    f"  }}\n"
                    f"  return result;\n"
                    f"}}\n"
                )
                yield code.encode("utf-8"), SID_JS_MODULAR_S2
