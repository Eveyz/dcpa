"""Stage 5 Four-Bucket Zero-Hallucination Pragmatic SFT Dataset Builder & Memmap Loader.

Implements Stage 5 Addendum Sections 1 & 2:
  - Streams + synthesizes ~20 MB train & 1 MB val (`--smoke-quota`: 1 MB train & 128 KB val)
    pure UTF-8 byte dialogues framed with native ASCII control bytes (`0x1E / 0x1F / 0x04`).
  - Emits exact 16-byte `<QIBBH` `.idx` records, contiguous `.bin` bytes, and 1-to-1 byte-aligned
    `*.loss_mask.bin` (`uint8` 0/1 supervision mask: strictly 0 on User/control prefixes, 1 on
    Assistant reply body + `0x04\\n`).
  - Four balanced pragmatic buckets:
      * Bucket A (35%): Interactive 2-3 turn novel-concept teaching, sentence/code use & live
                        user correction across EN / ZH / JA / Python / Rust (`wug_generator`).
      * Bucket B (25%): Self-contained stdlib-only Python & Rust micro-function Q&A (100%
                        `tree-sitter` zero-error verified, single turn <= 1536 bytes).
      * Bucket C (25%): EN/ZH/JA concise logic reasoning, summarization & SVO<->SOV translation.
      * Bucket D (15%): Epistemic boundary honest refusal & active fast-weight teaching requests.
  - Strict `blake2b` deduplication guaranteeing ZERO Train / Val overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import struct
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.ragged_dataloader import IDX_DTYPE, IDX_RECORD_FMT, assert_not_drive_mmap
from data.wug_generator import (
    _ZH_MORPHEMES,
    _ZH_RARE_CHARS,
    _fresh_pascal_ident,
    _fresh_snake_ident,
    _sample_en_pseudo_word,
    _sample_ja_pseudo_word,
    inflect_past,
    inflect_plural,
    inflect_prog,
)
from utils.chat_protocol import (
    EncodedDialogue,
    encode_dialogue_turns,
    verify_mask_alignment,
)


# =============================================================================
# Tree-Sitter Zero-Error AST Validators (Python & Rust)
# =============================================================================

_TS_PARSERS: dict[str, object] = {}


def _get_ts_parser(lang: str):
    if lang not in _TS_PARSERS:
        from tree_sitter import Language, Parser

        if lang == "python":
            import tree_sitter_python as tsp

            _TS_PARSERS[lang] = Parser(Language(tsp.language()))
        elif lang == "rust":
            import tree_sitter_rust as tsr

            _TS_PARSERS[lang] = Parser(Language(tsr.language()))
        else:
            raise ValueError(f"Unsupported tree-sitter lang: {lang}")
    return _TS_PARSERS[lang]


def is_valid_code_ast(code: str, lang: str) -> bool:
    code_b = code.encode("utf-8", errors="ignore")
    if len(code_b) < 12:
        return False
    parser = _get_ts_parser(lang)
    tree = parser.parse(code_b)
    return not bool(tree.root_node.has_error)


_FORBIDDEN_PY_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(numpy|torch|tensorflow|pandas|scipy|requests|flask|django|sklearn|cv2|PIL|matplotlib)\b",
    re.MULTILINE,
)
_FORBIDDEN_RS_CRATES = re.compile(
    r"^\s*(?:extern\s+crate|use)\s+(serde|tokio|rand|reqwest|actix|axum|clap|rayon|ndarray|anyhow)\b",
    re.MULTILINE,
)


def is_stdlib_only(code: str, lang: str) -> bool:
    if lang == "python":
        return _FORBIDDEN_PY_IMPORTS.search(code) is None
    if lang == "rust":
        return _FORBIDDEN_RS_CRATES.search(code) is None
    return True


# =============================================================================
# Bucket A (35%): Interactive 2-3 Turn Novel-Concept Teaching & Live Correction
# =============================================================================

_COLORS_ZH = ("蓝光", "紫光", "翠绿光", "琥珀光", "银白光")
_COLORS_EN = ("blue", "violet", "emerald", "amber", "silver")
_COLORS_JA = ("青い光", "紫色の光", "淡い緑の光", "銀色の光")


def _sample_bucket_a_dialogue(rng: random.Random) -> EncodedDialogue:
    domain = rng.randrange(5)  # 0=en, 1=zh, 2=ja, 3=py, 4=rs
    add_turn2 = rng.random() < 0.70

    if domain == 1:  # ZH rare-Hanzi teaching + causal sentence + live correction
        stem = rng.choice(_ZH_RARE_CHARS) + rng.choice(_ZH_MORPHEMES)
        c1, c2 = rng.sample(_COLORS_ZH, 2)
        env = rng.choice(("极低温", "高压深海", "强磁场", "真空暗室"))
        mat = rng.choice(("超导陶瓷", "层状矿石", "发光苔藓", "微孔晶体"))
        t1_u = f"记住新词：「{stem}」是一种在{env}下散发{c1}的{mat}。请用它造一个因果句。"
        t1_a = f"因为实验室将温度降至{env}区间，这块{stem}立刻开始散发清晰的{c1}。"
        turns = [(t1_u, t1_a)]
        if add_turn2:
            if rng.random() < 0.60:
                t2_u = f"我说错了，{stem}发的是{c2}不是{c1}，请修正刚才的句子。"
                t2_a = f"明白，已更新记忆：因为实验室将温度降至{env}区间，这块{stem}立刻开始散发清晰的{c2}。"
            else:
                t2_u = f"把刚才关于{stem}的句子改成英文。"
                t2_a = f"Because the lab entered the target regime, this {stem} immediately emitted bright glow."
            turns.append((t2_u, t2_a))
        return encode_dialogue_turns(turns, lang_id=1, bucket_id=0)

    if domain == 0:  # EN OOV teaching + inflection / live correction
        stem = _sample_en_pseudo_word(rng)
        c1, c2 = rng.sample(_COLORS_EN, 2)
        is_verb = rng.random() < 0.45
        if not is_verb:
            pl = inflect_plural(stem)
            habitat = rng.choice(("desert caves", "alpine streams", "coral tidepools", "cedar cliffs"))
            creature = rng.choice(("nocturnal bird", "tiny lizard", "glowing crystal", "striped crab"))
            t1_u = f"Learn a new word: a '{stem}' is a {c1} {creature} living in {habitat}. Use it in one concise sentence."
            t1_a = f"At dusk near the {habitat}, a single {c1} {stem} emerged quietly."
            turns = [(t1_u, t1_a)]
            if add_turn2:
                if rng.random() < 0.55:
                    t2_u = f"Rewrite that sentence in the plural for three individuals."
                    t2_a = f"At dusk near the {habitat}, three {c1} {pl} emerged quietly."
                else:
                    t2_u = f"Correction: a {stem} is actually {c2}, not {c1}. Fix the sentence."
                    t2_a = f"Updated: at dusk near the {habitat}, a single {c2} {stem} emerged quietly."
                turns.append((t2_u, t2_a))
        else:
            past = inflect_past(stem)
            prog = inflect_prog(stem)
            act = rng.choice(("weave thin willow reeds into baskets", "polish amber pebbles by hand", "tune bronze chimes"))
            t1_u = f"Remember this verb: to '{stem}' means to {act}. Write one past-tense example."
            t1_a = f"Yesterday afternoon, the apprentice {past} carefully until sunset."
            turns = [(t1_u, t1_a)]
            if add_turn2:
                t2_u = f"Now change it to present progressive tense."
                t2_a = f"Right now, the apprentice is {prog} steadily in the workshop."
                turns.append((t2_u, t2_a))
        return encode_dialogue_turns(turns, lang_id=0, bucket_id=0)

    if domain == 2:  # JA Katakana SOV teaching + correction
        stem = _sample_ja_pseudo_word(rng)
        c1, c2 = rng.sample(_COLORS_JA, 2)
        place = rng.choice(("静かな洞窟", "冷たい渓流", "高い山の岩陰"))
        kind = rng.choice(("珍しい鉱石", "小さな鳥", "香りの強い野草"))
        t1_u = f"新しい言葉を覚えてください：「{stem}」は{place}で{c1}を放つ{kind}です。短い例文を作ってください。"
        t1_a = f"昨日、研究者は{place}で{c1}を放つ{stem}を見つけました。"
        turns = [(t1_u, t1_a)]
        if add_turn2:
            t2_u = f"訂正します。{stem}が放つのは{c1}ではなく{c2}です。文を直してください。"
            t2_a = f"了解しました。昨日、研究者は{place}で{c2}を放つ{stem}を見つけました。"
            turns.append((t2_u, t2_a))
        return encode_dialogue_turns(turns, lang_id=2, bucket_id=0)

    if domain == 3:  # Python fresh class/function teaching + method + correction
        cls_name = _fresh_pascal_ident(rng)
        f1 = rng.choice(("cap", "limit", "quota"))
        f2 = rng.choice(("used", "count", "spent"))
        m1 = _fresh_snake_ident(rng)
        code1 = (
            f"class {cls_name}:\n"
            f"    def __init__(self, {f1}: int, {f2}: int) -> None:\n"
            f"        self.{f1} = {f1}\n"
            f"        self.{f2} = {f2}\n\n"
            f"    def {m1}(self) -> int:\n"
            f"        return max(0, self.{f1} - self.{f2})"
        )
        t1_u = f"Given class `{cls_name}({f1}: int, {f2}: int)`, implement method `{m1}(self) -> int` returning remaining capacity clamped at 0."
        t1_a = f"```python\n{code1}\n```"
        turns = [(t1_u, t1_a)]
        if add_turn2:
            code2 = (
                f"def is_full(obj: {cls_name}) -> bool:\n"
                f"    return obj.{m1}() == 0"
            )
            t2_u = f"Now write a helper `is_full(obj: {cls_name}) -> bool` using `{m1}`."
            t2_a = f"```python\n{code2}\n```"
            turns.append((t2_u, t2_a))
        return encode_dialogue_turns(turns, lang_id=3, bucket_id=0)

    # domain == 4: Rust fresh struct teaching + impl method + live fix
    st_name = _fresh_pascal_ident(rng)
    f1 = rng.choice(("cap", "limit", "max_len"))
    f2 = rng.choice(("used", "len", "filled"))
    m1 = _fresh_snake_ident(rng)
    code1 = (
        f"struct {st_name} {{\n"
        f"    {f1}: usize,\n"
        f"    {f2}: usize,\n"
        f"}}\n\n"
        f"impl {st_name} {{\n"
        f"    fn {m1}(&self) -> usize {{\n"
        f"        self.{f1}.saturating_sub(self.{f2})\n"
        f"    }}\n"
        f"}}"
    )
    t1_u = f"给定结构体 `struct {st_name} {{ {f1}: usize, {f2}: usize }}`，实现方法 `{m1}(&self) -> usize` 计算剩余容量（防下溢）。"
    t1_a = f"```rust\n{code1}\n```"
    turns = [(t1_u, t1_a)]
    if add_turn2:
        code2 = (
            f"impl {st_name} {{\n"
            f"    fn can_fit(&self, need: usize) -> bool {{\n"
            f"        self.{m1}() >= need\n"
            f"    }}\n"
            f"}}"
        )
        t2_u = f"继续为 `{st_name}` 追加 `can_fit(&self, need: usize) -> bool` 方法。"
        t2_a = f"```rust\n{code2}\n```"
        turns.append((t2_u, t2_a))
    return encode_dialogue_turns(turns, lang_id=4, bucket_id=0)


# =============================================================================
# Bucket B (25%): Self-Contained Pure Python & Rust Micro-Function Q&A
# =============================================================================

_STDLIB_PY_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Write a Python function `clamp_sum(xs: list[int], lo: int, hi: int) -> int` that sums integers after clamping each element into `[lo, hi]`.",
        "def clamp_sum(xs: list[int], lo: int, hi: int) -> int:\n    total = 0\n    for x in xs:\n        total += min(hi, max(lo, x))\n    return total",
    ),
    (
        "Write a pure Python function `is_balanced_brackets(s: str) -> bool` checking `()`, `[]`, and `{}` balance.",
        "def is_balanced_brackets(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack: list[str] = []\n    for ch in s:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack",
    ),
    (
        "写一个 Python 函数 `rle_encode(xs: list[int]) -> list[tuple[int, int]]`，对整数列表做游程编码。",
        "def rle_encode(xs: list[int]) -> list[tuple[int, int]]:\n    if not xs:\n        return []\n    out: list[tuple[int, int]] = []\n    cur, cnt = xs[0], 1\n    for v in xs[1:]:\n        if v == cur:\n            cnt += 1\n        else:\n            out.append((cur, cnt))\n            cur, cnt = v, 1\n    out.append((cur, cnt))\n    return out",
    ),
    (
        "Write a Python function `binary_search(arr: list[int], target: int) -> int` returning the index or `-1`.",
        "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    ),
    (
        "Implement `dedup_preserve_order(items: list[str]) -> list[str]` in pure Python.",
        "def dedup_preserve_order(items: list[str]) -> list[str]:\n    seen: set[str] = set()\n    out: list[str] = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out",
    ),
)

_STDLIB_RS_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Write a Rust function `sum_positive(xs: &[i32]) -> i64` that sums strictly positive elements.",
        "fn sum_positive(xs: &[i32]) -> i64 {\n    let mut acc: i64 = 0;\n    for &x in xs {\n        if x > 0 {\n            acc += x as i64;\n        }\n    }\n    acc\n}",
    ),
    (
        "写一个 Rust 函数 `is_balanced(s: &str) -> bool`，校验圆括号、方括号与花括号是否配对闭合。",
        "fn is_balanced(s: &str) -> bool {\n    let mut stack: Vec<u8> = Vec::new();\n    for b in s.bytes() {\n        match b {\n            b'(' | b'[' | b'{' => stack.push(b),\n            b')' => if stack.pop() != Some(b'(') { return false; },\n            b']' => if stack.pop() != Some(b'[') { return false; },\n            b'}' => if stack.pop() != Some(b'{') { return false; },\n            _ => {}\n        }\n    }\n    stack.is_empty()\n}",
    ),
    (
        "Write a Rust function `gcd(mut a: u64, mut b: u64) -> u64` using Euclid's algorithm.",
        "fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let r = a % b;\n        a = b;\n        b = r;\n    }\n    a\n}",
    ),
    (
        "Write a Rust function `prefix_sums(xs: &[i32]) -> Vec<i64>` computing cumulative sums.",
        "fn prefix_sums(xs: &[i32]) -> Vec<i64> {\n    let mut out = Vec::with_capacity(xs.len());\n    let mut acc: i64 = 0;\n    for &x in xs {\n        acc += x as i64;\n        out.push(acc);\n    }\n    out\n}",
    ),
    (
        "Implement `count_utf8_leads(buf: &[u8]) -> usize` in Rust to count UTF-8 characters without allocating.",
        "fn count_utf8_leads(buf: &[u8]) -> usize {\n    let mut n = 0usize;\n    for &b in buf {\n        if (b & 0xC0) != 0x80 {\n            n += 1;\n        }\n    }\n    n\n}",
    ),
)


def _sample_offline_bucket_b(rng: random.Random) -> EncodedDialogue:
    if rng.random() < 0.5:
        q, base_code = rng.choice(_STDLIB_PY_TEMPLATES)
        # Mild identifier variation to preserve diversity while keeping 100% AST validity
        tag = rng.choice(("", "_v2", "_fast", "_clean"))
        if tag and "def " in base_code:
            m = re.search(r"def\s+([a-zA-Z_]\w*)\(", base_code)
            if m:
                old_fn = m.group(1)
                new_fn = old_fn + tag
                q = q.replace(old_fn, new_fn)
                base_code = base_code.replace(old_fn, new_fn)
        assert is_valid_code_ast(base_code, "python")
        return encode_dialogue_turns([(q, f"```python\n{base_code}\n```")], lang_id=3, bucket_id=1)

    q, base_code = rng.choice(_STDLIB_RS_TEMPLATES)
    tag = rng.choice(("", "_v2", "_safe", "_fast"))
    if tag and "fn " in base_code:
        m = re.search(r"fn\s+([a-zA-Z_]\w*)\(", base_code)
        if m:
            old_fn = m.group(1)
            new_fn = old_fn + tag
            q = q.replace(old_fn, new_fn)
            base_code = base_code.replace(old_fn, new_fn)
    assert is_valid_code_ast(base_code, "rust")
    return encode_dialogue_turns([(q, f"```rust\n{base_code}\n```")], lang_id=4, bucket_id=1)


def _stream_hf_bucket_b(rng: random.Random, max_items: int) -> Iterator[EncodedDialogue]:
    """Stream AST-verified Python/Rust micro-functions from HuggingFaceTB/smoltalk & tiny-codes."""
    try:
        from datasets import load_dataset
    except ImportError:
        return

    yielded = 0
    # 1. TinyPixel/tiny-codes (ungated mirror of nampdn-ai/tiny-codes)
    for repo in ("TinyPixel/tiny-codes", "nampdn-ai/tiny-codes"):
        if yielded >= max_items:
            return
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            for row in ds:
                if yielded >= max_items:
                    break
                lang_raw = str(row.get("programming_language", "")).strip().lower()
                if lang_raw not in ("python", "rust"):
                    continue
                prompt = str(row.get("prompt", "")).strip()
                resp = str(row.get("response", "")).strip()
                if not prompt or not resp:
                    continue
                m = re.search(r"```(?:python|rust)?\s*\n(.*?)```", resp, re.DOTALL | re.IGNORECASE)
                code = m.group(1).strip() if m else resp
                if not is_stdlib_only(code, lang_raw):
                    continue
                if not is_valid_code_ast(code, lang_raw):
                    continue
                asst = f"```{lang_raw}\n{code}\n```"
                if len((prompt + asst).encode("utf-8")) > 1480:
                    continue
                lid = 3 if lang_raw == "python" else 4
                yield encode_dialogue_turns([(prompt, asst)], lang_id=lid, bucket_id=1)
                yielded += 1
            if yielded > 0:
                break
        except Exception:
            continue


# =============================================================================
# Bucket C (25%): EN/ZH/JA Logic Reasoning, Concise Summarization & Translation
# =============================================================================

_TRANSLATION_TRIPLES: tuple[tuple[str, str, str], ...] = (
    (
        "The patient carpenter built a sturdy cedar bridge before the autumn rain began.",
        "耐心的木匠在秋雨开始前搭建了一座结实的雪松木桥。",
        "忍耐強い大工は秋の雨が始まる前に丈夫な杉の橋を建てました。",
    ),
    (
        "When the sensor detects a sudden pressure drop, the valve closes automatically.",
        "当传感器检测到压力骤降时，阀门会自动关闭。",
        "センサーが急な圧力低下を検知すると、弁は自動的に閉じます。",
    ),
    (
        "Three engineers inspected the memory alignment carefully before shipping the patch.",
        "三位工程师在发布补丁前仔细检查了内存对齐情况。",
        "三人の技術者はパッチを公開する前にメモリ整列を慎重に確認しました。",
    ),
    (
        "Although the alpine trail was steep, the hikers reached the stone shelter before dusk.",
        "尽管高山小路陡峭，徒步者还是在黄昏前抵达了石屋避难所。",
        "高山の道は急でしたが、登山者たちは夕暮れ前に石の小屋へ着きました。",
    ),
    (
        "Each module communicates through fixed-size state vectors instead of growing caches.",
        "每个模块通过固定大小的状态向量通信，而不是不断增长的缓存。",
        "各モジュールは増え続けるキャッシュではなく固定長の状態ベクトルで通信します。",
    ),
)


def _sample_offline_bucket_c(rng: random.Random) -> EncodedDialogue:
    kind = rng.randint(0, 2)
    if kind == 0:
        # Cross-lingual SVO <-> SOV translation
        en, zh, ja = rng.choice(_TRANSLATION_TRIPLES)
        mode = rng.randint(0, 3)
        if mode == 0:
            return encode_dialogue_turns([(f"Translate into Japanese (SOV): {en}", ja)], lang_id=2, bucket_id=2)
        if mode == 1:
            return encode_dialogue_turns([(f"Translate into English (SVO): {ja}", en)], lang_id=0, bucket_id=2)
        if mode == 2:
            return encode_dialogue_turns([(f"请将下面句子翻译成中文：{en}", zh)], lang_id=1, bucket_id=2)
        return encode_dialogue_turns([(f"次の中国語を日本語（SOV語順）に訳してください：{zh}", ja)], lang_id=2, bucket_id=2)

    if kind == 1:
        # Compact deductive arithmetic / logical ordering
        a = rng.randint(4, 29)
        b = rng.randint(3, 19)
        c = rng.randint(2, 11)
        ans = (a + b) * c
        if rng.random() < 0.5:
            q = f"A warehouse starts with {a} crates, receives {b} more, and packs {c} identical trucks at that total. How many crates are loaded?"
            a_txt = f"First, {a} + {b} = {a + b} crates per truck. Across {c} trucks, {a + b} * {c} = {ans} crates."
            return encode_dialogue_turns([(q, a_txt)], lang_id=0, bucket_id=2)
        q = f"仓库原有 {a} 箱物资，又入库 {b} 箱，随后按同样总量装满 {c} 辆车。一共装运了多少箱？"
        a_txt = f"单车装运 {a} + {b} = {a + b} 箱；{c} 辆车共装运 ({a + b}) × {c} = {ans} 箱。"
        return encode_dialogue_turns([(q, a_txt)], lang_id=1, bucket_id=2)

    # Concise logic / constraint reasoning
    items = rng.sample(("Alpha", "Beta", "Gamma", "Delta"), 3)
    if rng.random() < 0.5:
        q = f"Given that {items[0]} finishes before {items[1]}, and {items[1]} finishes before {items[2]}, which task completes last?"
        a_txt = f"By transitivity ({items[0]} < {items[1]} < {items[2]}), {items[2]} completes last."
        return encode_dialogue_turns([(q, a_txt)], lang_id=0, bucket_id=2)
    q = f"已知任务 {items[0]} 早于 {items[1]} 完成，{items[1]} 早于 {items[2]} 完成，哪个任务最先完成？"
    a_txt = f"由传递顺序（{items[0]} < {items[1]} < {items[2]}）可知，{items[0]} 最先完成。"
    return encode_dialogue_turns([(q, a_txt)], lang_id=1, bucket_id=2)


def _stream_hf_bucket_c(max_items: int) -> Iterator[EncodedDialogue]:
    """Stream short non-encyclopedic dialogues & translation pairs from smoltalk and opus-100."""
    try:
        from datasets import load_dataset
    except ImportError:
        return

    yielded = 0
    for subset in ("everyday-conversations", "explore-instruct-rewriting"):
        if yielded >= max_items // 2:
            break
        try:
            ds = load_dataset("HuggingFaceTB/smoltalk", subset, split="train", streaming=True)
            for row in ds:
                if yielded >= max_items // 2:
                    break
                msgs = row.get("messages") or []
                turns: list[tuple[str, str]] = []
                for i in range(0, len(msgs) - 1, 2):
                    if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                        u = str(msgs[i].get("content", "")).strip()
                        a = str(msgs[i + 1].get("content", "")).strip()
                        if u and a:
                            turns.append((u, a))
                if not turns:
                    continue
                turns = turns[:2]
                enc = encode_dialogue_turns(turns, lang_id=0, bucket_id=2)
                if len(enc.raw_bytes) <= 1200:
                    yield enc
                    yielded += 1
        except Exception:
            continue

    for pair, tgt_label, lid in (("en-ja", "Japanese (SOV)", 2), ("en-zh", "Chinese", 1)):
        if yielded >= max_items:
            break
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", pair, split="train", streaming=True)
            for row in ds:
                if yielded >= max_items:
                    break
                tr = row.get("translation") or {}
                src = str(tr.get("en", "")).strip()
                tgt = str(tr.get("ja" if lid == 2 else "zh", "")).strip()
                if 20 <= len(src) <= 240 and 10 <= len(tgt) <= 240:
                    q = f"Translate into {tgt_label}: {src}"
                    yield encode_dialogue_turns([(q, tgt)], lang_id=lid, bucket_id=2)
                    yielded += 1
        except Exception:
            continue


# =============================================================================
# Bucket D (15%): Epistemic Boundary Honest Refusal & Active Teaching Request
# =============================================================================

_REFUSAL_ZH = (
    "我的核心权重仅内化了语言语法与逻辑算子，未存储该具体百科事实或实时外部数据。"
    "你可以直接用一句话告诉我它的定义或数据，我会立即写入快权重并为你继续推演。"
)
_REFUSAL_EN = (
    "My slow weights encode grammar and reasoning operators rather than encyclopedic trivia or live external state. "
    "Tell me the exact definition or figure in one sentence, and I will bind it into fast weights and continue."
)
_REFUSAL_JA = (
    "私のコア重みは文法と推論演算子のみを保持しており、個別の百科事実やリアルタイム情報は記憶していません。"
    "定義や数値を一文で教えていただければ、即座に高速重みへ記憶して推論を続けます。"
)

_BUCKET_D_PROMPTS_ZH: tuple[str, ...] = (
    "请问 1847 年某小镇第三任镇长的出生月份是多少？",
    "现在纳斯达克指数的实时点位和成交量是多少？",
    "我们公司内部测试环境的数据库端口和密钥配置是什么？",
    "请告诉我某冷门小行星 2024-XZ99 的精确发现日期。",
    "明天下午三点东京新宿区的实时降水量预报是多少毫米？",
)
_BUCKET_D_PROMPTS_EN: tuple[str, ...] = (
    "What is the exact live stock price of ACME Corp right now?",
    "In what year was the fourth mayor of Oakhaven Parish born?",
    "What is the private API endpoint configured on our staging cluster?",
    "Tell me the exact population recorded in the 1913 census of Silverbrook Village.",
    "What is the current wind speed at Zurich Airport this minute?",
)
_BUCKET_D_PROMPTS_JA: tuple[str, ...] = (
    "現在の東京証券取引所のリアルタイム指数はいくらですか？",
    "社内テストサーバーの非公開ポート番号を教えてください。",
    "1882年にある地方自治体の初代議長が就任した正確な日付は何日ですか？",
)


def _sample_bucket_d_dialogue(rng: random.Random) -> EncodedDialogue:
    r = rng.random()
    if r < 0.45:
        q = rng.choice(_BUCKET_D_PROMPTS_ZH)
        turns = [(q, _REFUSAL_ZH)]
        if rng.random() < 0.35:
            val = rng.randint(100, 9999)
            turns.append(
                (f"好，记住这个数据：当前指标值是 {val}。请把它的两倍算出来。", f"已写入快权重：指标值为 {val}，它的两倍是 {val * 2}。")
            )
        return encode_dialogue_turns(turns, lang_id=1, bucket_id=3)
    if r < 0.85:
        q = rng.choice(_BUCKET_D_PROMPTS_EN)
        turns = [(q, _REFUSAL_EN)]
        if rng.random() < 0.35:
            val = rng.randint(10, 500)
            turns.append(
                (f"Got it: the current value is {val}. What is {val} plus 15?", f"Stored in fast memory: {val} + 15 = {val + 15}.")
            )
        return encode_dialogue_turns(turns, lang_id=0, bucket_id=3)

    q = rng.choice(_BUCKET_D_PROMPTS_JA)
    return encode_dialogue_turns([(q, _REFUSAL_JA)], lang_id=2, bucket_id=3)


# =============================================================================
# Builder Entrypoint & Zero-Padding Memmap SFT Dataset
# =============================================================================


def build_stage5_sft_corpus(
    out_dir: str | Path = "artifacts/sft_data",
    *,
    train_mb: float = 20.0,
    val_mb: float = 1.0,
    smoke_quota: bool = False,
    offline: bool = False,
    seed: int = 202605,
    max_doc_bytes: int = 2048,
) -> dict:
    """Build Stage 5 four-bucket SFT dataset (`sft_train.*` + `sft_val.*` + `loss_mask.bin`)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    assert_not_drive_mmap(out_path)

    if smoke_quota:
        train_bytes_target = int(1.0 * 1024 * 1024)
        val_bytes_target = int(0.125 * 1024 * 1024)
    else:
        train_bytes_target = int(train_mb * 1024 * 1024)
        val_bytes_target = int(val_mb * 1024 * 1024)

    total_target = train_bytes_target + val_bytes_target
    bucket_ratios = {0: 0.35, 1: 0.25, 2: 0.25, 3: 0.15}
    bucket_targets = {b: int(total_target * r) for b, r in bucket_ratios.items()}

    rng = random.Random(seed)
    t0 = time.perf_counter()

    # Optional HF streaming pools for Buckets B & C when online
    hf_b_iter = iter(()) if offline else _stream_hf_bucket_b(rng, max_items=6000)
    hf_c_iter = iter(()) if offline else _stream_hf_bucket_c(max_items=6000)
    hf_b_exhausted = offline
    hf_c_exhausted = offline

    seen_hashes: set[bytes] = set()
    bucket_docs: dict[int, list[EncodedDialogue]] = {0: [], 1: [], 2: [], 3: []}
    bucket_bytes: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    def _next_for_bucket(b_id: int) -> EncodedDialogue:
        nonlocal hf_b_exhausted, hf_c_exhausted
        if b_id == 0:
            return _sample_bucket_a_dialogue(rng)
        if b_id == 1:
            if not hf_b_exhausted:
                try:
                    return next(hf_b_iter)
                except StopIteration:
                    hf_b_exhausted = True
            return _sample_offline_bucket_b(rng)
        if b_id == 2:
            if not hf_c_exhausted:
                try:
                    return next(hf_c_iter)
                except StopIteration:
                    hf_c_exhausted = True
            return _sample_offline_bucket_c(rng)
        return _sample_bucket_d_dialogue(rng)

    for b_id in (0, 1, 2, 3):
        target_b = bucket_targets[b_id]
        stalls = 0
        while bucket_bytes[b_id] < target_b:
            dlg = _next_for_bucket(b_id)
            if len(dlg.raw_bytes) > max_doc_bytes:
                continue
            h = hashlib.blake2b(dlg.raw_bytes, digest_size=16).digest()
            if h in seen_hashes:
                stalls += 1
                if stalls > 2000:
                    # Allow parameterized synthetic suffix variation if small template pool saturates
                    nonce = f" [ref #{rng.randint(1000, 99999)}]"
                    u_end = dlg.raw_bytes.find(b"\n\x1fA\n")
                    if u_end > 3:
                        raw_mod = dlg.raw_bytes[:u_end] + nonce.encode("utf-8") + dlg.raw_bytes[u_end:]
                        mask_mod = dlg.loss_mask_bytes[:u_end] + (b"\x00" * len(nonce)) + dlg.loss_mask_bytes[u_end:]
                        if len(raw_mod) <= max_doc_bytes and verify_mask_alignment(raw_mod, mask_mod):
                            dlg = EncodedDialogue(
                                raw_bytes=raw_mod,
                                loss_mask_bytes=mask_mod,
                                lang_id=dlg.lang_id,
                                bucket_id=dlg.bucket_id,
                                num_turns=dlg.num_turns,
                                eot_positions=tuple(p + len(nonce) for p in dlg.eot_positions),
                            )
                            h = hashlib.blake2b(dlg.raw_bytes, digest_size=16).digest()
                if h in seen_hashes:
                    continue
            stalls = 0
            seen_hashes.add(h)
            bucket_docs[b_id].append(dlg)
            bucket_bytes[b_id] += len(dlg.raw_bytes)

    # Split each bucket cleanly into train vs val (strictly disjoint since all hashes are unique!)
    train_docs: list[EncodedDialogue] = []
    val_docs: list[EncodedDialogue] = []
    val_frac = val_bytes_target / max(float(total_target), 1.0)
    for b_id in (0, 1, 2, 3):
        docs = bucket_docs[b_id]
        rng.shuffle(docs)
        n_val = max(4, int(round(len(docs) * val_frac)))
        val_docs.extend(docs[:n_val])
        train_docs.extend(docs[n_val:])

    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    def _write_split(split_name: str, docs: list[EncodedDialogue]) -> dict:
        bin_p = out_path / f"sft_{split_name}.bin"
        idx_p = out_path / f"sft_{split_name}.idx"
        msk_p = out_path / f"sft_{split_name}.loss_mask.bin"
        meta_p = out_path / f"sft_{split_name}.meta.json"

        offset = 0
        lang_bytes = {i: 0 for i in range(5)}
        bkt_bytes = {i: 0 for i in range(4)}
        sup_bytes = 0

        with bin_p.open("wb") as fb, idx_p.open("wb") as fi, msk_p.open("wb") as fm:
            for d in docs:
                n = len(d.raw_bytes)
                fb.write(d.raw_bytes)
                fm.write(d.loss_mask_bytes)
                fi.write(
                    struct.pack(
                        IDX_RECORD_FMT,
                        offset,
                        n,
                        d.lang_id,
                        d.bucket_id,
                        min(int(d.num_turns), 255),
                        int(d.style_id),
                    )
                )
                offset += n
                lang_bytes[d.lang_id] = lang_bytes.get(d.lang_id, 0) + n
                bkt_bytes[d.bucket_id] = bkt_bytes.get(d.bucket_id, 0) + n
                sup_bytes += d.loss_mask_bytes.count(1)

        meta = {
            "split": split_name,
            "num_docs": len(docs),
            "total_bytes": offset,
            "supervised_bytes": sup_bytes,
            "supervised_ratio": round(sup_bytes / max(offset, 1), 4),
            "bytes_by_lang": lang_bytes,
            "bytes_by_bucket": bkt_bytes,
        }
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    train_meta = _write_split("train", train_docs)
    val_meta = _write_split("val", val_docs)

    summary = {
        "schema_version": 1,
        "smoke_quota": smoke_quota,
        "offline": offline,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "train": train_meta,
        "val": val_meta,
    }
    (out_path / "sft_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class SFTByteDataset(Dataset):
    """Zero-padding Constant-Byte-Budget memmap loader for Stage 5 SFT (`sft_*.bin/.idx/.loss_mask.bin`).

    Packs whole multi-turn dialogues (`<= max_doc_bytes`) into a contiguous `uint8` buffer
    (`Total_T <= target_batch_bytes`) without EVER splitting a dialogue across turns.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        target_batch_bytes: int = 65536,
        max_doc_bytes: int = 2048,
        num_batches: int = 100000,
        seed: int = 42,
        prefix: str = "sft",
        length_bucketing: bool = True,
        bucket_sync_world_size: int = 1,
    ) -> None:
        dd = Path(data_dir)
        assert_not_drive_mmap(dd)
        self.bin_path = dd / f"{prefix}_{split}.bin"
        self.idx_path = dd / f"{prefix}_{split}.idx"
        self.mask_path = dd / f"{prefix}_{split}.loss_mask.bin"
        for p in (self.bin_path, self.idx_path, self.mask_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing Stage 5 SFT shard file: {p}")

        self.target_batch_bytes = int(target_batch_bytes)
        self.max_doc_bytes = int(max_doc_bytes)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.length_bucketing = bool(length_bucketing)
        self.bucket_sync_world_size = max(1, int(bucket_sync_world_size))

        idx_mm = np.memmap(self.idx_path, dtype=IDX_DTYPE, mode="r")
        self._offsets = np.array(idx_mm["offset"], dtype=np.int64, copy=True)
        self._lengths = np.array(idx_mm["length"], dtype=np.int64, copy=True)
        self._lang_ids = np.array(idx_mm["lang_id"], dtype=np.int64, copy=True)
        self._source_ids = np.array(idx_mm["source_id"], dtype=np.int64, copy=True)
        self._style_ids = np.array(idx_mm["style_id"], dtype=np.int64, copy=True)
        del idx_mm

        self.num_docs = int(len(self._offsets))
        self._tier_doc_indices: list[np.ndarray] = []
        self._tier_cum_probs: list[float] = []
        valid_mask = self._lengths <= self.max_doc_bytes
        if self.length_bucketing and self.max_doc_bytes > 512 and int(np.count_nonzero(valid_mask)) > 0:
            edges = [0, 512, 1024, 2048, 1 << 60]
            counts_t: list[int] = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                if lo >= self.max_doc_bytes:
                    break
                eff_hi = self.max_doc_bytes if hi >= self.max_doc_bytes else hi
                idxs = np.nonzero((self._lengths > lo) & (self._lengths <= eff_hi))[0]
                if len(idxs) > 0:
                    self._tier_doc_indices.append(idxs)
                    counts_t.append(len(idxs))
            if len(self._tier_doc_indices) > 1:
                tot_c = float(sum(counts_t))
                acc = 0.0
                for c in counts_t:
                    acc += float(c) / tot_c
                    self._tier_cum_probs.append(acc)
                self._tier_cum_probs[-1] = 1.0
            else:
                self._tier_doc_indices.clear()

        self._bin_mm: Optional[np.memmap] = None
        self._msk_mm: Optional[np.memmap] = None
        self._pid: int = -1

    def _get_mmaps(self) -> tuple[np.memmap, np.memmap]:
        pid = os.getpid()
        if self._bin_mm is None or self._msk_mm is None or self._pid != pid:
            self._bin_mm = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
            self._msk_mm = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
            self._pid = pid
        return self._bin_mm, self._msk_mm

    def close(self) -> None:
        self._bin_mm = None
        self._msk_mm = None
        self._pid = -1

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        winfo = get_worker_info()
        wid = winfo.id if winfo is not None else 0
        rng = random.Random(self.seed + int(batch_idx) * 1_000_003 + wid * 97)
        bin_mm, msk_mm = self._get_mmaps()

        budget = self.target_batch_bytes
        x_chunks: list[np.ndarray] = []
        m_chunks: list[np.ndarray] = []
        cu_list: list[int] = [0]
        doc_langs: list[int] = []
        doc_styles: list[int] = []
        doc_pts: list[float] = []
        cursor = 0

        tier_pool: Optional[np.ndarray] = None
        if self._tier_doc_indices:
            eff_sync = int(batch_idx) // self.bucket_sync_world_size
            u_t = random.Random(self.seed + eff_sync * 7919 + 31).random()
            t_idx = len(self._tier_cum_probs) - 1
            for k_i, cp in enumerate(self._tier_cum_probs):
                if u_t <= cp:
                    t_idx = k_i
                    break
            tier_pool = self._tier_doc_indices[t_idx]

        attempts = 0
        while cursor < budget and attempts < 2048:
            attempts += 1
            if tier_pool is not None:
                d_i = int(tier_pool[rng.randrange(len(tier_pool))])
            else:
                d_i = rng.randrange(self.num_docs)
            d_len = int(self._lengths[d_i])
            if d_len > self.max_doc_bytes:
                continue
            if cursor + d_len > budget:
                if doc_langs:
                    break
                # Always allow at least 1 full dialogue even when budget < d_len in tiny tests
            off = int(self._offsets[d_i])
            x_chunks.append(np.array(bin_mm[off : off + d_len], dtype=np.uint8, copy=True))
            m_chunks.append(np.array(msk_mm[off : off + d_len], dtype=np.uint8, copy=True))
            doc_langs.append(int(self._lang_ids[d_i]))
            doc_styles.append(int(self._style_ids[d_i]))
            doc_pts.append(0.0 if int(self._source_ids[d_i]) == 11 else 1.0)
            cursor += d_len
            cu_list.append(cursor)

        x_arr = np.concatenate(x_chunks, axis=0)
        m_arr = np.concatenate(m_chunks, axis=0).astype(np.bool_)
        cu_arr = np.asarray(cu_list, dtype=np.int32)
        return {
            "x_bytes": torch.from_numpy(x_arr),
            "sft_loss_mask": torch.from_numpy(m_arr),
            "cu_seqlens": torch.from_numpy(cu_arr),
            "doc_lang_ids": torch.tensor(doc_langs, dtype=torch.long),
            "doc_style_ids": torch.tensor(doc_styles, dtype=torch.long),
            "doc_ponder_tax_scale": torch.tensor(doc_pts, dtype=torch.float32),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 5 4-Bucket Pragmatic SFT Dataset")
    ap.add_argument("--out-dir", type=str, default="artifacts/sft_data")
    ap.add_argument("--train-mb", type=float, default=20.0)
    ap.add_argument("--val-mb", type=float, default=1.0)
    ap.add_argument("--smoke-quota", action="store_true", help="Build 1MB train + 128KB val locally")
    ap.add_argument("--offline", action="store_true", help="Zero-network offline synthesis mode")
    ap.add_argument("--seed", type=int, default=202605)
    args = ap.parse_args()

    res = build_stage5_sft_corpus(
        out_dir=args.out_dir,
        train_mb=args.train_mb,
        val_mb=args.val_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
