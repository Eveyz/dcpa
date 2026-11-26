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

from data.build_corpus import validate_js_lexical_brackets
from data.ragged_dataloader import IDX_DTYPE, IDX_RECORD_FMT, assert_not_drive_mmap
from data.wug_generator import (
    _ZH_MORPHEMES,
    _ZH_RARE_CHARS,
    _fresh_camel_ident,
    _fresh_pascal_ident,
    _fresh_snake_ident,
    _sample_en_pseudo_word,
    inflect_past,
    inflect_plural,
    inflect_prog,
)
from utils.chat_protocol import (
    SCRATCH_END,
    SCRATCH_START,
    EncodedDialogue,
    encode_dialogue_turns,
    sanitize_raw_text,
    verify_mask_alignment,
)


# =============================================================================
# Tree-Sitter Zero-Error AST Validators (Python & Rust)
# =============================================================================

_TS_PARSERS: dict[str, object] = {}


def _get_ts_parser(lang: str):
    lang_key = "javascript" if lang == "js" else lang
    if lang_key not in _TS_PARSERS:
        from tree_sitter import Language, Parser

        if lang_key == "python":
            import tree_sitter_python as tsp

            _TS_PARSERS[lang_key] = Parser(Language(tsp.language()))
        elif lang_key == "javascript":
            try:
                import tree_sitter_javascript as tsjs

                _TS_PARSERS[lang_key] = Parser(Language(tsjs.language()))
            except ImportError:
                _TS_PARSERS[lang_key] = None
        elif lang_key == "rust":
            import tree_sitter_rust as tsr

            _TS_PARSERS[lang_key] = Parser(Language(tsr.language()))
        else:
            raise ValueError(f"Unsupported tree-sitter lang: {lang}")
    return _TS_PARSERS[lang_key]


def is_valid_code_ast(code: str, lang: str) -> bool:
    code_b = code.encode("utf-8", errors="ignore")
    if len(code_b) < 12:
        return False
    lang_key = "javascript" if lang == "js" else lang
    parser = _get_ts_parser(lang_key)
    if parser is None:
        if lang_key == "javascript":
            ok, _ = validate_js_lexical_brackets(code_b, min_bytes=12, max_bytes=8192)
            return ok
        return False
    tree = parser.parse(code_b)
    return not bool(tree.root_node.has_error)


_FORBIDDEN_PY_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(numpy|torch|tensorflow|pandas|scipy|requests|flask|django|sklearn|cv2|PIL|matplotlib)\b",
    re.MULTILINE,
)
_FORBIDDEN_JS_IMPORTS = re.compile(
    r"(?:require\s*\(\s*['\"](?:lodash|axios|express|react|vue|moment|jquery|chalk|fs-extra)['\"]\s*\))"
    r"|(?:^\s*import\s+.*?\s+from\s+['\"](?:lodash|axios|express|react|vue|moment|jquery|chalk)['\"])",
    re.MULTILINE,
)
_FORBIDDEN_RS_CRATES = re.compile(
    r"^\s*(?:extern\s+crate|use)\s+(serde|tokio|rand|reqwest|actix|axum|clap|rayon|ndarray|anyhow)\b",
    re.MULTILINE,
)


def is_stdlib_only(code: str, lang: str) -> bool:
    if lang == "python":
        return _FORBIDDEN_PY_IMPORTS.search(code) is None
    if lang in ("javascript", "js"):
        return _FORBIDDEN_JS_IMPORTS.search(code) is None
    if lang == "rust":
        return _FORBIDDEN_RS_CRATES.search(code) is None
    return True


# =============================================================================
# Bucket A (35%): Interactive 2-3 Turn Novel-Concept Teaching & Live Correction
# =============================================================================

_COLORS_ZH = ("蓝光", "紫光", "翠绿光", "琥珀光", "银白光")
_COLORS_EN = ("blue", "violet", "emerald", "amber", "silver")


def _sample_bucket_a_dialogue(rng: random.Random) -> EncodedDialogue:
    domain = rng.choice((0, 1, 3, 5))  # 0=en, 1=zh, 3=python, 5=javascript
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

    # domain == 5: JavaScript ES6 class teaching + method + live follow-up
    st_name = _fresh_pascal_ident(rng)
    f1 = rng.choice(("cap", "limit", "maxLen"))
    f2 = rng.choice(("used", "count", "filled"))
    m1 = _fresh_camel_ident(rng)
    code1 = (
        f"class {st_name} {{\n"
        f"  constructor({f1}, {f2}) {{\n"
        f"    this.{f1} = {f1};\n"
        f"    this.{f2} = {f2};\n"
        f"  }}\n"
        f"  {m1}() {{\n"
        f"    return Math.max(0, this.{f1} - this.{f2});\n"
        f"  }}\n"
        f"}}"
    )
    t1_u = f"在 JavaScript 中定义类 `{st_name}({f1}, {f2})`，并实现方法 `{m1}()` 返回不小于 0 的剩余容量。"
    t1_a = f"```javascript\n{code1}\n```"
    turns = [(t1_u, t1_a)]
    if add_turn2:
        code2 = (
            f"function canFit(box, need) {{\n"
            f"  return box.{m1}() >= need;\n"
            f"}}"
        )
        t2_u = f"基于 `{st_name}` 的 `{m1}()`，再写一个辅助函数 `canFit(box, need)`。"
        t2_a = f"```javascript\n{code2}\n```"
        turns.append((t2_u, t2_a))
    return encode_dialogue_turns(turns, lang_id=5, bucket_id=0)


# =============================================================================
# Bucket B (25%): Self-Contained Pure Python & JavaScript Micro-Function Q&A
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

_STDLIB_JS_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Write a JavaScript function `sumPositive(xs)` that sums strictly positive numbers in an array.",
        "function sumPositive(xs) {\n  let acc = 0;\n  for (const x of xs) {\n    if (x > 0) {\n      acc += x;\n    }\n  }\n  return acc;\n}",
    ),
    (
        "写一个 JavaScript 函数 `isBalanced(s)`，校验字符串中的圆括号、方括号与花括号是否严格闭合。",
        "function isBalanced(s) {\n  const pairs = { ')': '(', ']': '[', '}': '{' };\n  const stack = [];\n  for (const ch of s) {\n    if (ch === '(' || ch === '[' || ch === '{') {\n      stack.push(ch);\n    } else if (ch in pairs) {\n      if (stack.pop() !== pairs[ch]) {\n        return false;\n      }\n    }\n  }\n  return stack.length === 0;\n}",
    ),
    (
        "Write an ES6 JavaScript function `groupByKey(items, key)` grouping an array of objects by property `key`.",
        "function groupByKey(items, key) {\n  const out = {};\n  for (const item of items) {\n    const k = item[key];\n    if (!out[k]) {\n      out[k] = [];\n    }\n    out[k].push(item);\n  }\n  return out;\n}",
    ),
    (
        "Write a JavaScript function `prefixSums(xs)` returning cumulative sums of `xs`.",
        "function prefixSums(xs) {\n  const out = [];\n  let acc = 0;\n  for (const x of xs) {\n    acc += x;\n    out.push(acc);\n  }\n  return out;\n}",
    ),
    (
        "用原生 JavaScript 实现 `chunkArray(arr, size)`，将数组按固定长度 `size` 切分为二维数组。",
        "function chunkArray(arr, size) {\n  const step = Math.max(1, size);\n  const out = [];\n  for (let i = 0; i < arr.length; i += step) {\n    out.push(arr.slice(i, i + step));\n  }\n  return out;\n}",
    ),
)


def _sample_offline_bucket_b(rng: random.Random) -> EncodedDialogue:
    if rng.random() < 0.5:
        q, base_code = rng.choice(_STDLIB_PY_TEMPLATES)
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

    q, base_code = rng.choice(_STDLIB_JS_TEMPLATES)
    tag = rng.choice(("", "V2", "Safe", "Fast"))
    if tag and "function " in base_code:
        m = re.search(r"function\s+([a-zA-Z_]\w*)\(", base_code)
        if m:
            old_fn = m.group(1)
            new_fn = old_fn + tag
            q = q.replace(old_fn, new_fn)
            base_code = base_code.replace(old_fn, new_fn)
    assert is_valid_code_ast(base_code, "javascript")
    return encode_dialogue_turns([(q, f"```javascript\n{base_code}\n```")], lang_id=5, bucket_id=1)


def _stream_hf_bucket_b(rng: random.Random, max_items: int) -> Iterator[EncodedDialogue]:
    """Stream AST-verified Python/JavaScript micro-functions from tiny-codes."""
    try:
        from datasets import load_dataset
    except ImportError:
        return

    yielded = 0
    for repo in ("TinyPixel/tiny-codes", "nampdn-ai/tiny-codes"):
        if yielded >= max_items:
            return
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            for row in ds:
                if yielded >= max_items:
                    break
                lang_raw = str(row.get("programming_language", "")).strip().lower()
                if lang_raw not in ("python", "javascript"):
                    continue
                prompt = str(row.get("prompt", "")).strip()
                resp = str(row.get("response", "")).strip()
                if not prompt or not resp:
                    continue
                m = re.search(r"```(?:python|javascript|js)?\s*\n(.*?)```", resp, re.DOTALL | re.IGNORECASE)
                code = m.group(1).strip() if m else resp
                if not is_stdlib_only(code, lang_raw):
                    continue
                if not is_valid_code_ast(code, lang_raw):
                    continue
                asst = f"```{lang_raw}\n{code}\n```"
                if len((prompt + asst).encode("utf-8")) > 1480:
                    continue
                lid = 3 if lang_raw == "python" else 5
                yield encode_dialogue_turns([(prompt, asst)], lang_id=lid, bucket_id=1)
                yielded += 1
            if yielded > 0:
                break
        except Exception:
            continue


# =============================================================================
# Bucket C (25%): EN/ZH Logic Reasoning, Instruction Following & Translation
# =============================================================================

_TRANSLATION_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "The patient carpenter built a sturdy cedar bridge before the autumn rain began.",
        "耐心的木匠在秋雨开始前搭建了一座结实的雪松木桥。",
    ),
    (
        "When the sensor detects a sudden pressure drop, the valve closes automatically.",
        "当传感器检测到压力骤降时，阀门会自动关闭。",
    ),
    (
        "Three engineers inspected the memory alignment carefully before shipping the patch.",
        "三位工程师在发布补丁前仔细检查了内存对齐情况。",
    ),
    (
        "Although the alpine trail was steep, the hikers reached the stone shelter before dusk.",
        "尽管高山小路陡峭，徒步者还是在黄昏前抵达了石屋避难所。",
    ),
    (
        "Each module communicates through fixed-size state vectors instead of growing caches.",
        "每个模块通过固定大小的状态向量通信，而不是不断增长的缓存。",
    ),
)


def _sample_offline_bucket_c(rng: random.Random) -> EncodedDialogue:
    kind = rng.randint(0, 3)
    if kind == 0:
        # Cross-lingual EN <-> ZH translation
        en, zh = rng.choice(_TRANSLATION_PAIRS)
        if rng.random() < 0.5:
            return encode_dialogue_turns([(f"请将下面英文准确翻译成中文：{en}", zh)], lang_id=1, bucket_id=2)
        return encode_dialogue_turns([(f"Translate the following Chinese sentence into natural English: {zh}", en)], lang_id=0, bucket_id=2)

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

    if kind == 2:
        # Strict JSON / constraint instruction following
        svc = rng.choice(("auth-gw", "search-api", "ingest-worker", "billing-core"))
        port = rng.choice((8080, 8443, 9000, 9092))
        retries = rng.randint(2, 5)
        payload = json.dumps({"service": svc, "port": port, "retries": retries}, ensure_ascii=False)
        if rng.random() < 0.5:
            q = f"Extract a compact JSON object with keys `service`, `port`, and `retries` (no extra prose): service {svc} listens on port {port} with {retries} retries."
            return encode_dialogue_turns([(q, payload)], lang_id=0, bucket_id=2)
        q = f"请仅输出包含 `service`、`port`、`retries` 三个字段的紧凑 JSON（不要任何多余解释）：服务 {svc} 监听端口 {port}，最大重试次数为 {retries}。"
        return encode_dialogue_turns([(q, payload)], lang_id=1, bucket_id=2)

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
    """Stream short non-encyclopedic EN/ZH instruction dialogues & translation pairs."""
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

    try:
        ds_zh = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train", streaming=True)
        for row in ds_zh:
            if yielded >= max_items:
                break
            tr = row.get("translation") or {}
            src = str(tr.get("en", "")).strip()
            tgt = str(tr.get("zh", "")).strip()
            if 20 <= len(src) <= 240 and 10 <= len(tgt) <= 240:
                if yielded % 2 == 0:
                    yield encode_dialogue_turns([(f"请将下面句子翻译成中文：{src}", tgt)], lang_id=1, bucket_id=2)
                else:
                    yield encode_dialogue_turns([(f"Translate into English: {tgt}", src)], lang_id=0, bucket_id=2)
                yielded += 1
    except Exception:
        pass


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

_BUCKET_D_PROMPTS_ZH: tuple[str, ...] = (
    "请问 1847 年某小镇第三任镇长的出生月份是多少？",
    "现在纳斯达克指数的实时点位和成交量是多少？",
    "我们公司内部测试环境的数据库端口和密钥配置是什么？",
    "请告诉我某冷门小行星 2024-XZ99 的精确发现日期。",
    "明天下午三点深圳市南山区的实时降水量预报是多少毫米？",
)
_BUCKET_D_PROMPTS_EN: tuple[str, ...] = (
    "What is the exact live stock price of ACME Corp right now?",
    "In what year was the fourth mayor of Oakhaven Parish born?",
    "What is the private API endpoint configured on our staging cluster?",
    "Tell me the exact population recorded in the 1913 census of Silverbrook Village.",
    "What is the current wind speed at Zurich Airport this minute?",
)


def _sample_bucket_d_dialogue(rng: random.Random) -> EncodedDialogue:
    if rng.random() < 0.50:
        q = rng.choice(_BUCKET_D_PROMPTS_ZH)
        turns = [(q, _REFUSAL_ZH)]
        if rng.random() < 0.35:
            val = rng.randint(100, 9999)
            turns.append(
                (f"好，记住这个数据：当前指标值是 {val}。请把它的两倍算出来。", f"已写入快权重：指标值为 {val}，它的两倍是 {val * 2}。")
            )
        return encode_dialogue_turns(turns, lang_id=1, bucket_id=3)
    q = rng.choice(_BUCKET_D_PROMPTS_EN)
    turns = [(q, _REFUSAL_EN)]
    if rng.random() < 0.35:
        val = rng.randint(10, 500)
        turns.append(
            (f"Got it: the current value is {val}. What is {val} plus 15?", f"Stored in fast memory: {val} + 15 = {val + 15}.")
        )
    return encode_dialogue_turns(turns, lang_id=0, bucket_id=3)


# =============================================================================
# DCPA v2 Holistic Cognitive Stage-5 Builders (35% Logic : 35% Writing : 30% Code)
# Zero Rust: 0=en, 1=zh, 2=formal, 3=python, 4=sql_struct, 5=javascript
# =============================================================================


def _wrap_scratch(plan: str, body: str) -> str:
    p_clean = sanitize_raw_text(plan).strip()
    b_clean = sanitize_raw_text(body).strip()
    return f"\x1dS\n{p_clean}\n\x1dE\n{b_clean}"


def _sample_v2_logic_socratic_dialogue(rng: random.Random) -> EncodedDialogue:
    """Bucket 0 (35%): Socratic premise questioning, fallacy pushback & epistemic boundary self-awareness."""
    mode = rng.random()
    if mode < 0.45:
        # Socratic missing-premise detection -> Turn-2 binding & deduction
        rate_a = rng.randint(62, 88)
        rate_b = rate_a + rng.randint(5, 11)
        sample_b = rng.choice((12, 15, 18))
        sample_a = rng.choice((800, 1200, 2500))
        if rng.random() < 0.52:
            u1 = (
                f"方案 B 的转化率达到 {rate_b}%，高于方案 A 的 {rate_a}%。"
                f"我们是否应该立刻全量切换到方案 B？"
            )
            a1 = _wrap_scratch(
                f"◐ missing_premise: sample_size & traffic_split unknown -> guard against base-rate/survivorship fallacy",
                "目前不能直接下结论：缺少两组的样本量、流量分层是否同质以及统计显著性边界。"
                "如果方案 B 仅在小样本或高意向子群上测得高转化，直接全量切换可能触发辛普森悖论或均值回归。"
                "请补充两组的样本规模与分流方式。",
            )
            u2 = f"补充数据：方案 A 测试了 {sample_a} 人（全量随机流量），方案 B 只在老客群测试了 {sample_b} 人。"
            a2 = _wrap_scratch(
                f"● bound: A={sample_a}(random) vs B={sample_b}(loyal_bias) -> selection bias confirmed -> reject full rollout",
                f"不应立即全量切换。方案 B 不仅样本量极小（仅 {sample_b} 人，置信区间过宽），"
                f"而且局限于高留存老客群，与方案 A 的 {sample_a} 人随机流量不具可比性。"
                f"建议先在同质随机流量下补做等权 A/B 实验，再决定是否放量。",
            )
            return encode_dialogue_turns([(u1, a1), (u2, a2)], lang_id=1, bucket_id=0, style_id=0, allow_assistant_controls=True)

        u1 = (
            f"Vendor B reports {rate_b}% uptime resolution within 10 minutes versus Vendor A's {rate_a}%. "
            f"Does that prove Vendor B has strictly stronger engineering reliability?"
        )
        a1 = _wrap_scratch(
            "◐ missing_premise: incident severity mix & sample denominator unstated -> ask before concluding",
            "Not necessarily. That comparison omits two critical premises: total incident count and severity distribution. "
            "A vendor handling mostly trivial tickets can post a higher 10-minute resolution percentage while failing on severe outages. "
            "Please share the ticket volume and P1/P2 breakdown for both vendors.",
        )
        u2 = f"Vendor A handled {sample_a} mixed P1/P2 incidents; Vendor B handled {sample_b} minor P4 tickets only."
        a2 = _wrap_scratch(
            f"● Simpson/selection bias: A={sample_a}(P1/P2) vs B={sample_b}(P4) -> premise invalidates claim",
            f"With that breakdown, Vendor B's {rate_b}% figure reflects {sample_b} minor P4 requests rather than comparable outage load, "
            f"whereas Vendor A sustained {rate_a}% across {sample_a} severe P1/P2 incidents. "
            f"Compare both vendors within matching severity tiers before switching.",
        )
        return encode_dialogue_turns([(u1, a1), (u2, a2)], lang_id=0, bucket_id=0, style_id=0, allow_assistant_controls=True)

    if mode < 0.78:
        # Epistemic boundary self-awareness + live fast-weight rule/metric binding
        metric_v = rng.randint(120, 960)
        delta_v = rng.randint(15, 85)
        if rng.random() < 0.5:
            q = rng.choice(_BUCKET_D_PROMPTS_ZH)
            a1 = _wrap_scratch(
                "○ epistemic_boundary: unverified external/private fact -> refuse fabrication & invite fast-weight input",
                _REFUSAL_ZH,
            )
            u2 = f"给你现场数据：基准读数为 {metric_v}，安全余量需要扣除 {delta_v}。请给出可用上限。"
            a2 = _wrap_scratch(
                f"● fast_weight_bound: base={metric_v}, margin={delta_v} -> net={metric_v - delta_v}",
                f"已锁定你提供的基准数据：{metric_v} - {delta_v} = {metric_v - delta_v}，当前可用安全上限为 {metric_v - delta_v}。",
            )
            return encode_dialogue_turns([(q, a1), (u2, a2)], lang_id=1, bucket_id=0, style_id=0, allow_assistant_controls=True)

        q = rng.choice(_BUCKET_D_PROMPTS_EN)
        a1 = _wrap_scratch(
            "○ epistemic_boundary: live/private fact absent from context -> decline guesswork & request input",
            _REFUSAL_EN,
        )
        u2 = f"Here is the live figure: baseline capacity is {metric_v} units and reserve holdback is {delta_v} units. Compute net dispatchable capacity."
        a2 = _wrap_scratch(
            f"● fast_weight_bound: {metric_v} - {delta_v} = {metric_v - delta_v}",
            f"Binding your live figures: {metric_v} - {delta_v} = {metric_v - delta_v} dispatchable units.",
        )
        return encode_dialogue_turns([(q, a1), (u2, a2)], lang_id=0, bucket_id=0, style_id=0, allow_assistant_controls=True)

    # Formal logic consistency & countermodel check (lang_id = 2, style_id = 0)
    k_mod = rng.choice((6, 8, 10, 12))
    odd_c = rng.choice((1, 3, 5))
    u1 = (
        f"Check consistency of this formal constraint set over integers:\n"
        f"  C1: x ≡ 0 (mod 2)\n"
        f"  C2: y = {k_mod} * q + x for some integer q\n"
        f"  C3: y ≡ {odd_c} (mod 2)\n"
        f"Does a valid integer solution (x, y, q) exist?"
    )
    a1 = _wrap_scratch(
        f"● C1: x even & {k_mod} even -> C2 implies y even -> contradicts C3 (y odd={odd_c}) -> ∴ UNSAT",
        f"No solution exists (the system is unsatisfiable).\n"
        f"1. By C1, `x` is even (`x ≡ 0 mod 2`).\n"
        f"2. Since `{k_mod}` is even, `{k_mod} * q` is also even for every integer `q`; hence `y = {k_mod} * q + x` must satisfy `y ≡ 0 (mod 2)`.\n"
        f"3. This directly contradicts C3 (`y ≡ {odd_c} ≡ 1 mod 2`).\n"
        f"∴ `{{C1, C2, C3}}` is inconsistent.",
    )
    u2 = f"If we relax C3 to `y ≡ 0 (mod 2)` and add `x = 4, q = 2`, what is `y`?"
    y_ans = k_mod * 2 + 4
    a2 = _wrap_scratch(
        f"● C3' consistent -> substitute q=2, x=4 -> y={k_mod}*2+4={y_ans}",
        f"With C3 relaxed to `y ≡ 0 (mod 2)`, the parity constraint is consistent. Substituting `x = 4` and `q = 2` into C2 gives `y = {k_mod} * 2 + 4 = {y_ans}`.",
    )
    return encode_dialogue_turns([(u1, a1), (u2, a2)], lang_id=2, bucket_id=0, style_id=0, allow_assistant_controls=True)


_ZH_POLISH_SEEDS: tuple[tuple[str, str, str, str, int], ...] = (
    (
        "请以“技术系统中的冗余设计”为题写一段短评，先给初稿，再接受审校打磨。",
        "立意: 冗余不是浪费而是容错边界 | 结构: 误区 -> 工程机制 -> 结论",
        "在当今快速发展的技术时代，值得注意的是，冗余设计往往被很多人误解为一种资源浪费。不仅如此，许多团队为了追求极致的短期效率，会砍掉备用链路。总而言之，一旦发生突发故障，没有冗余的系统就会陷入瘫痪，因此冗余其实非常重要。",
        "诊断: 删去“当今时代/值得注意的是/总而言之”等套话；补入热备切换与尾部风险的具体因果锚点",
        2,
    ),
    (
        "写一段关于“老旧船坞清晨停航检修”的叙事速写（约 150 字）。",
        "意象锚点: 铁锈、退潮刻度、扳手敲击声 | 情绪: 克制沉静",
        "清晨的海港非常安静，令人印象深刻。老工匠走进船坞，看到大船停在那里，心里有很多感慨。他拿起工具开始认真检查船底的阀门，展现出了极高的敬业精神，晨光洒在他身上显得十分温暖。",
        "诊断: 剔除“令人印象深刻/极高敬业精神”等空洞总结词，改用退潮水痕与金属冷感动作细节呈现",
        3,
    ),
    (
        "起草一封向研发团队通报“周三晚间索引库只读维护 20 分钟”的内部备忘录。",
        "要素: 窗口 22:00-22:20 | 影响: 写入排队、读取正常 | 回滚预案",
        "各位同事大家好，为了进一步全面提升我们的系统稳定性和综合服务体验，经过慎重讨论，我们计划在本周三晚上对索引库开展一次非常重要的维护工作。届时可能会对部分写入操作产生一定程度的影响，希望大家提前做好准备并予以配合。",
        "诊断: 砍掉寒暄与虚词；精确列出起止时间、只读表现、上游重试策略与超时回滚阈值",
        1,
    ),
)

_EN_POLISH_SEEDS: tuple[tuple[str, str, str, str, int], ...] = (
    (
        "Draft a short analytical paragraph on why small engineering teams should prefer boring protocols.",
        "thesis: predictable failure modes beat novelty | arc: hidden maintenance tax -> debuggability -> takeaway",
        "In today's fast-paced software landscape, it is important to note that shiny new protocols often attract teams. Moreover, cutting-edge tools look impressive in benchmarks. All in all, boring protocols are actually much better because they save time when things break.",
        "diagnosis: strip AI throat-clearing ('In today's...', 'important to note', 'All in all'); replace vague praise with concrete failure-mode visibility and on-call recovery cost",
        2,
    ),
    (
        "Write a brief scene of a watchmaker testing a repaired marine chronometer before dawn.",
        "anchor: brass balance wheel, frost on window, tick interval | tone: quiet precision",
        "The workshop was very quiet and atmospheric before sunrise. The dedicated watchmaker looked closely at the antique chronometer with deep focus. After hours of hard work, the gears finally moved smoothly, proving that patience always pays off.",
        "diagnosis: cut telling adjectives ('atmospheric', 'dedicated') and moralizing ending; ground the paragraph in physical sound, loupe magnification, and beat error",
        3,
    ),
    (
        "Write a crisp executive memo recommending a two-week freeze on schema migrations before peak quarter-end close.",
        "decision: freeze DDL Sep 20 - Oct 4 | exception path: P0 hotfix sign-off | rationale: settlement stability",
        "It goes without saying that quarter-end financial close is a critical period for our organization. Furthermore, database changes can sometimes introduce unexpected risks. Therefore, we would like to suggest pausing schema updates for two weeks so everything runs smoothly.",
        "diagnosis: delete filler transitions; state exact freeze window, operational blast radius, and single-owner P0 exception gate upfront",
        1,
    ),
)


def _sample_v2_writing_polish_dialogue(rng: random.Random) -> EncodedDialogue:
    """Bucket 1 (35%): Multi-draft author loop (`Brief -> Slot-0 Outline + Draft 1 -> Self-Critique -> Polished Final`)."""
    batch_tag = rng.randint(12, 48)
    if rng.random() < 0.52:
        idx = rng.randrange(len(_ZH_POLISH_SEEDS))
        prompt, outline, draft1, diag, sid = _ZH_POLISH_SEEDS[idx]
        if idx == 0:
            polished = (
                f"冗余设计的本质不是堆叠闲置算力，而是为极端尾部故障预留隔离带。"
                f"日常负载下，双路校验与旁路缓存看似拉高了约 {batch_tag}% 的静态成本；"
                f"但当主链路遭遇瞬时拥塞或硬件静默损坏时，正是这层缓冲决定了系统是平滑降级，还是触发雪崩式停摆。"
            )
            crit_u = "初稿有明显的套话（如“值得注意的是”“总而言之”），且论证偏空泛。请自我审校并精修定稿。"
        elif idx == 1:
            polished = (
                f"潮水退至三号石阶下方，露出船底暗褐色的藤壶与湿冷铁锈。"
                f"老师傅蹲在龙骨阴影里，用铜锤轻叩止回阀法兰，侧耳辨认回音是否发闷；"
                f"直到第 {batch_tag} 颗螺栓渗出的盐水被棉纱拭净，坞门外才透进第一缕灰白晨光。"
            )
            crit_u = "初稿用了太多抽象概括词，缺少触觉和听觉细节。请删掉空洞抒情，重写成克制凝练的定稿。"
        else:
            polished = (
                f"【维护窗口】周三 22:00–22:{batch_tag}（共 {batch_tag} 分钟），主索引库切换为只读模式。\n"
                f"【业务影响】在线查询不受影响；异步写入请求将暂存本地队列，恢复后自动重放。\n"
                f"【回滚边界】若切换后只读延迟 P99 超过 80ms，22:08 前无损切回原主库。"
            )
            crit_u = "请删掉客套话和模糊表述，改成条理分明、可直接发给研发值班群的精炼工程备忘录。"
        a1 = _wrap_scratch(outline, draft1)
        a2 = _wrap_scratch(diag, polished)
        return encode_dialogue_turns([(prompt, a1), (crit_u, a2)], lang_id=1, bucket_id=1, style_id=sid, allow_assistant_controls=True)

    idx = rng.randrange(len(_EN_POLISH_SEEDS))
    prompt, outline, draft1, diag, sid = _EN_POLISH_SEEDS[idx]
    if idx == 0:
        polished = (
            f"Small teams pay for protocol novelty twice: first during integration, and again at 3 a.m. when an undocumented edge case stalls traffic. "
            f"A battle-tested protocol with {batch_tag} years of field postmortems trades marginal benchmark throughput for known failure modes, "
            f"plain-text wire inspection, and fast recovery under pressure."
        )
        crit_u = "Cut the AI clichés ('In today's fast-paced...', 'All in all') and tighten this into sharp, concrete prose."
    elif idx == 1:
        polished = (
            f"Frost rimmed the north pane while the brass balance wheel swung under his loupe. "
            f"He nudged the regulator screw a hairbreadth, counted {batch_tag} crisp beats against the wall pendulum, "
            f"and closed the mahogany lid only when the beat error settled to zero."
        )
        crit_u = "Strip the abstract adjectives and moralizing conclusion; revise using concrete sensory action."
    else:
        polished = (
            f"Effective Day 1 through Day {batch_tag} of close, all non-emergency production DDL migrations are frozen.\n"
            f"- Scope: Ledger, settlement, and reporting schemas (read/write DML unaffected).\n"
            f"- Exception Gate: P0 data-integrity fixes require joint sign-off from On-Call Lead and Finance Eng."
        )
        crit_u = "Remove the throat-clearing filler and rewrite as a direct, high-signal executive memo."
    a1 = _wrap_scratch(outline, draft1)
    a2 = _wrap_scratch(diag, polished)
    return encode_dialogue_turns([(prompt, a1), (crit_u, a2)], lang_id=0, bucket_id=1, style_id=sid, allow_assistant_controls=True)


def _sample_v2_code_bugfix_dialogue(rng: random.Random) -> EncodedDialogue:
    """Bucket 2 (30%): Python/SQL/JS root-cause bugfix + Turn-2 live spec evolution (`style_id=4`, zero Rust)."""
    lang_choice = rng.random()
    if lang_choice < 0.50:
        # Python AST-verified bugfix + Turn-2 extension (lang_id = 3)
        fn_name = f"window_max_{_fresh_snake_ident(rng)}"
        cap_hi = rng.choice((256, 512, 1024))
        buggy_py = (
            f"def {fn_name}(xs: list[int], k: int, seen: list[int] = []) -> list[int]:\n"
            f"    for i in range(len(xs) - k):\n"
            f"        seen.append(max(xs[i : i + k]))\n"
            f"    return seen"
        )
        fixed_py_1 = (
            f"def {fn_name}(xs: list[int], k: int) -> list[int]:\n"
            f"    if k <= 0 or k > len(xs):\n"
            f"        return []\n"
            f"    return [max(xs[i : i + k]) for i in range(len(xs) - k + 1)]"
        )
        fixed_py_2 = (
            f"def {fn_name}(xs: list[int], k: int, ceiling: int = {cap_hi}) -> list[int]:\n"
            f"    if k <= 0 or k > len(xs):\n"
            f"        return []\n"
            f"    return [min(ceiling, max(xs[i : i + k])) for i in range(len(xs) - k + 1)]"
        )
        assert is_valid_code_ast(fixed_py_1, "python") and is_valid_code_ast(fixed_py_2, "python")
        u1 = f"Diagnose both bugs in this Python sliding-window helper and provide a clean fix:\n```python\n{buggy_py}\n```"
        a1 = _wrap_scratch(
            "bugs: 1) mutable default `seen=[]` leaks state across calls; 2) off-by-one `len(xs)-k` drops final window & lacks `k<=0` guard",
            f"Two root causes:\n"
            f"1. `seen: list[int] = []` mutates a shared default list across invocations.\n"
            f"2. `range(len(xs) - k)` misses the last valid window (`len(xs) - k + 1`) and fails when `k <= 0` or `k > len(xs)`.\n\n"
            f"```python\n{fixed_py_1}\n```",
        )
        u2 = f"Now extend `{fn_name}` with an optional `ceiling: int = {cap_hi}` parameter that clamps each window maximum."
        a2 = _wrap_scratch(
            f"extend signature -> clamp each window max via min(ceiling, ...)",
            f"```python\n{fixed_py_2}\n```",
        )
        return encode_dialogue_turns([(u1, a1), (u2, a2)], lang_id=3, bucket_id=2, style_id=4, allow_assistant_controls=True)

    if lang_choice < 0.75:
        # SQL LEFT JOIN filter trap bugfix (lang_id = 4, style_id = 4)
        tbl_u = f"orgs_{_fresh_snake_ident(rng)}"
        tbl_o = f"invoices_{_fresh_snake_ident(rng)}"
        min_amt = rng.choice((50, 100, 200))
        u1 = (
            f"Why does this SQL query drop organizations with zero paid invoices, and how do we fix it?\n"
            f"```sql\n"
            f"SELECT o.org_id, COUNT(i.invoice_id) AS paid_cnt\n"
            f"FROM {tbl_u} o\n"
            f"LEFT JOIN {tbl_o} i ON o.org_id = i.org_id\n"
            f"WHERE i.status = 'paid' AND i.amount_cents >= {min_amt}\n"
            f"GROUP BY o.org_id;\n"
            f"```"
        )
        fixed_sql_1 = (
            f"SELECT\n"
            f"    o.org_id,\n"
            f"    COUNT(i.invoice_id) AS paid_cnt\n"
            f"FROM {tbl_u} AS o\n"
            f"LEFT JOIN {tbl_o} AS i\n"
            f"    ON o.org_id = i.org_id\n"
            f"    AND i.status = 'paid'\n"
            f"    AND i.amount_cents >= {min_amt}\n"
            f"GROUP BY o.org_id;"
        )
        fixed_sql_2 = (
            f"SELECT\n"
            f"    o.org_id,\n"
            f"    COUNT(i.invoice_id) AS paid_cnt,\n"
            f"    COALESCE(SUM(i.amount_cents), 0) AS paid_total_cents\n"
            f"FROM {tbl_u} AS o\n"
            f"LEFT JOIN {tbl_o} AS i\n"
            f"    ON o.org_id = i.org_id\n"
            f"    AND i.status = 'paid'\n"
            f"    AND i.amount_cents >= {min_amt}\n"
            f"GROUP BY o.org_id;"
        )
        a1 = _wrap_scratch(
            "root_cause: filtering right-table columns `i.status` in WHERE turns LEFT JOIN into INNER JOIN on NULL rows -> move predicates into ON",
            f"Filtering `i.status = 'paid'` inside `WHERE` rejects unmatched rows where `i.*` is `NULL`, silently converting the `LEFT JOIN` into an `INNER JOIN`. Move right-table predicates into the `ON` clause:\n\n```sql\n{fixed_sql_1}\n```",
        )
        u2 = "Also return `paid_total_cents` defaulting to `0` instead of `NULL` for zero-invoice organizations."
        a2 = _wrap_scratch(
            "wrap SUM(i.amount_cents) in COALESCE(..., 0)",
            f"```sql\n{fixed_sql_2}\n```",
        )
        return encode_dialogue_turns([(u1, a1), (u2, a2)], lang_id=4, bucket_id=2, style_id=4, allow_assistant_controls=True)

    # JavaScript ES6 mutation & numeric sort bugfix (lang_id = 5, style_id = 4)
    fn_js = _fresh_camel_ident(rng)
    top_n = rng.choice((3, 5, 10))
    buggy_js = (
        f"function {fn_js}(scores) {{\n"
        f"  return scores.sort().slice(0, {top_n});\n"
        f"}}"
    )
    fixed_js_1 = (
        f"function {fn_js}(scores) {{\n"
        f"  if (!Array.isArray(scores)) return [];\n"
        f"  return [...scores].sort((a, b) => b - a).slice(0, {top_n});\n"
        f"}}"
    )
    fixed_js_2 = (
        f"function {fn_js}(scores, minScore = 0) {{\n"
        f"  if (!Array.isArray(scores)) return [];\n"
        f"  return scores.filter((v) => v >= minScore).sort((a, b) => b - a).slice(0, {top_n});\n"
        f"}}"
    )
    assert is_valid_code_ast(fixed_js_1, "javascript") and is_valid_code_ast(fixed_js_2, "javascript")
    u1 = f"Fix the two bugs in `{fn_js}` (intended to return the top {top_n} numeric scores descending without mutating the caller's array):\n```javascript\n{buggy_js}\n```"
    a1 = _wrap_scratch(
        "bugs: 1) Array.prototype.sort() mutates in place; 2) default comparator sorts lexicographically ('10' < '2')",
        f"1. `scores.sort()` mutates `scores` in place.\n"
        f"2. Without `(a, b) => b - a`, JavaScript coerces numbers to UTF-16 strings and sorts lexicographically.\n\n"
        f"```javascript\n{fixed_js_1}\n```",
    )
    u2 = f"Add an optional `minScore = 0` threshold filter before sorting."
    a2 = _wrap_scratch(
        "filter(v => v >= minScore) creates fresh copy -> numeric sort descending -> slice",
        f"```javascript\n{fixed_js_2}\n```",
    )
    return encode_dialogue_turns([(u1, a1), (u2, a2)], lang_id=5, bucket_id=2, style_id=4, allow_assistant_controls=True)


def _stream_hf_v2_logic(max_items: int, max_doc_bytes: int) -> Iterator[EncodedDialogue]:
    """Stream analytical logic & Socratic dialogues from OpenHermes-2.5 and COIG-CQIA when online."""
    try:
        from datasets import load_dataset
    except ImportError:
        return

    yielded = 0
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        for row in ds:
            if yielded >= max_items // 2:
                break
            cat = str(row.get("category") or "").lower()
            if cat and cat not in ("logic", "reasoning", "philosophy", "analytical", "math", "general"):
                continue
            convs = row.get("conversations") or []
            turns: list[tuple[str, str]] = []
            for i in range(len(convs) - 1):
                r0 = str(convs[i].get("from", "")).lower()
                r1 = str(convs[i + 1].get("from", "")).lower()
                if r0 in ("human", "user") and r1 in ("gpt", "assistant"):
                    u = sanitize_raw_text(str(convs[i].get("value", ""))).strip()
                    a = sanitize_raw_text(str(convs[i + 1].get("value", ""))).strip()
                    if u and a and "```rust" not in a.lower():
                        turns.append((u, a))
            if not turns:
                continue
            enc = encode_dialogue_turns(turns[:2], lang_id=0, bucket_id=0, style_id=0)
            if 160 <= len(enc.raw_bytes) <= max_doc_bytes:
                yield enc
                yielded += 1
    except Exception:
        pass

    for cfg_name in ("ruozhiba", "zhihu"):
        if yielded >= max_items:
            break
        try:
            ds_zh = load_dataset("m-a-p/COIG-CQIA", cfg_name, split="train", streaming=True)
            for row in ds_zh:
                if yielded >= max_items:
                    break
                inst = sanitize_raw_text(str(row.get("instruction", ""))).strip()
                inp = sanitize_raw_text(str(row.get("input", ""))).strip()
                out = sanitize_raw_text(str(row.get("output", ""))).strip()
                u = f"{inst}\n{inp}".strip() if inp else inst
                if not u or len(out) < 40:
                    continue
                enc = encode_dialogue_turns([(u, out)], lang_id=1, bucket_id=0, style_id=2 if cfg_name == "zhihu" else 0)
                if 160 <= len(enc.raw_bytes) <= max_doc_bytes:
                    yield enc
                    yielded += 1
        except Exception:
            continue


def _stream_hf_v2_writing(max_items: int, max_doc_bytes: int) -> Iterator[EncodedDialogue]:
    """Stream multi-turn writing & rewriting dialogues when online."""
    try:
        from datasets import load_dataset
    except ImportError:
        return

    yielded = 0
    for repo, subset in (
        ("Crownelius/Creative-Writing-Multiturn_Enhanced", None),
        ("HuggingFaceTB/smoltalk", "explore-instruct-rewriting"),
    ):
        if yielded >= max_items:
            break
        try:
            ds = (
                load_dataset(repo, split="train", streaming=True)
                if subset is None
                else load_dataset(repo, subset, split="train", streaming=True)
            )
            for row in ds:
                if yielded >= max_items:
                    break
                msgs = row.get("messages") or row.get("conversations") or []
                turns: list[tuple[str, str]] = []
                for i in range(len(msgs) - 1):
                    r0 = str(msgs[i].get("role") or msgs[i].get("from") or "").lower()
                    r1 = str(msgs[i + 1].get("role") or msgs[i + 1].get("from") or "").lower()
                    if r0 in ("user", "human") and r1 in ("assistant", "gpt"):
                        u = sanitize_raw_text(str(msgs[i].get("content") or msgs[i].get("value") or "")).strip()
                        a = sanitize_raw_text(str(msgs[i + 1].get("content") or msgs[i + 1].get("value") or "")).strip()
                        if u and a:
                            turns.append((u, a))
                if not turns:
                    continue
                sid = 3 if "Creative-Writing" in repo else 2
                enc = encode_dialogue_turns(turns[:2], lang_id=0, bucket_id=1, style_id=sid)
                if 200 <= len(enc.raw_bytes) <= max_doc_bytes:
                    yield enc
                    yielded += 1
        except Exception:
            continue


def _stream_hf_v2_code_bugfix(rng: random.Random, max_items: int, max_doc_bytes: int) -> Iterator[EncodedDialogue]:
    """Stream compact Python/JS bugfix & commit-repair dialogues (`commitpackft` / `SWE-smith-py`) when online."""
    try:
        from datasets import load_dataset
    except ImportError:
        return

    yielded = 0
    for lang_cfg, lid in (("python", 3), ("javascript", 5)):
        if yielded >= max_items:
            break
        try:
            ds = load_dataset("bigcode/commitpackft", lang_cfg, split="train", streaming=True)
            for row in ds:
                if yielded >= max_items:
                    break
                old_c = sanitize_raw_text(str(row.get("old_contents", ""))).strip()
                new_c = sanitize_raw_text(str(row.get("new_contents", ""))).strip()
                subj = sanitize_raw_text(str(row.get("subject", "") or row.get("message", ""))).strip()
                if not old_c or not new_c or old_c == new_c or len(old_c) > 1400 or len(new_c) > 1400:
                    continue
                if not is_stdlib_only(new_c, lang_cfg) or not is_valid_code_ast(new_c, lang_cfg):
                    continue
                u = f"Fix the issue ({subj or 'bugfix'}) in the following {lang_cfg} snippet:\n```{lang_cfg}\n{old_c}\n```"
                a = _wrap_scratch(
                    f"fix: {subj[:120] or 'apply minimal AST-valid patch'}",
                    f"```{lang_cfg}\n{new_c}\n```",
                )
                enc = encode_dialogue_turns([(u, a)], lang_id=lid, bucket_id=2, style_id=4, allow_assistant_controls=True)
                if len(enc.raw_bytes) <= max_doc_bytes:
                    yield enc
                    yielded += 1
        except Exception:
            continue


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
    max_doc_bytes: Optional[int] = None,
    cognitive_v2: bool = False,
) -> dict:
    """Build Stage 5 SFT dataset (`sft_train.*` + `sft_val.*` + `loss_mask.bin`)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    assert_not_drive_mmap(out_path)
    eff_max_doc_bytes = int(max_doc_bytes) if max_doc_bytes is not None else (4096 if cognitive_v2 else 2048)

    if smoke_quota:
        train_bytes_target = int(1.0 * 1024 * 1024)
        val_bytes_target = int(0.125 * 1024 * 1024)
    else:
        train_bytes_target = int(train_mb * 1024 * 1024)
        val_bytes_target = int(val_mb * 1024 * 1024)

    total_target = train_bytes_target + val_bytes_target
    if cognitive_v2:
        # Exact 35% Logic : 35% Multi-Draft Writing : 30% Engineering Bugfix/SDK (Zero Rust)
        bucket_ids: tuple[int, ...] = (0, 1, 2)
        bucket_ratios = {0: 0.35, 1: 0.35, 2: 0.30}
    else:
        bucket_ids = (0, 1, 2, 3)
        bucket_ratios = {0: 0.35, 1: 0.25, 2: 0.25, 3: 0.15}
    bucket_targets = {b: int(total_target * r) for b, r in bucket_ratios.items()}

    rng = random.Random(seed)
    t0 = time.perf_counter()

    if cognitive_v2:
        hf_v2_0 = iter(()) if offline else _stream_hf_v2_logic(max_items=6000, max_doc_bytes=eff_max_doc_bytes)
        hf_v2_1 = iter(()) if offline else _stream_hf_v2_writing(max_items=6000, max_doc_bytes=eff_max_doc_bytes)
        hf_v2_2 = iter(()) if offline else _stream_hf_v2_code_bugfix(rng, max_items=6000, max_doc_bytes=eff_max_doc_bytes)
        v2_exhausted = {0: offline, 1: offline, 2: offline}
        hf_b_iter = iter(())
        hf_c_iter = iter(())
        hf_b_exhausted = True
        hf_c_exhausted = True
    else:
        v2_exhausted = {}
        hf_v2_0 = iter(())
        hf_v2_1 = iter(())
        hf_v2_2 = iter(())
        hf_b_iter = iter(()) if offline else _stream_hf_bucket_b(rng, max_items=6000)
        hf_c_iter = iter(()) if offline else _stream_hf_bucket_c(max_items=6000)
        hf_b_exhausted = offline
        hf_c_exhausted = offline

    seen_hashes: set[bytes] = set()
    bucket_docs: dict[int, list[EncodedDialogue]] = {b: [] for b in bucket_ids}
    bucket_bytes: dict[int, int] = {b: 0 for b in bucket_ids}

    def _next_for_bucket(b_id: int) -> EncodedDialogue:
        nonlocal hf_b_exhausted, hf_c_exhausted
        if cognitive_v2:
            if b_id == 0:
                if not v2_exhausted[0] and rng.random() < 0.45:
                    try:
                        return next(hf_v2_0)
                    except StopIteration:
                        v2_exhausted[0] = True
                return _sample_v2_logic_socratic_dialogue(rng)
            if b_id == 1:
                if not v2_exhausted[1] and rng.random() < 0.40:
                    try:
                        return next(hf_v2_1)
                    except StopIteration:
                        v2_exhausted[1] = True
                return _sample_v2_writing_polish_dialogue(rng)
            if not v2_exhausted[2] and rng.random() < 0.45:
                try:
                    return next(hf_v2_2)
                except StopIteration:
                    v2_exhausted[2] = True
            return _sample_v2_code_bugfix_dialogue(rng)

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

    for b_id in bucket_ids:
        target_b = bucket_targets[b_id]
        stalls = 0
        while bucket_bytes[b_id] < target_b:
            dlg = _next_for_bucket(b_id)
            if len(dlg.raw_bytes) > eff_max_doc_bytes:
                continue
            h = hashlib.blake2b(dlg.raw_bytes, digest_size=16).digest()
            if h in seen_hashes:
                stalls += 1
                if stalls > 200:
                    # Allow parameterized synthetic suffix variation if small template pool saturates
                    nonce = f" [case #{rng.randint(1000, 999999)}]"
                    u_end = dlg.raw_bytes.find(b"\n\x1fA\n")
                    if u_end > 3:
                        raw_mod = dlg.raw_bytes[:u_end] + nonce.encode("utf-8") + dlg.raw_bytes[u_end:]
                        mask_mod = dlg.loss_mask_bytes[:u_end] + (b"\x00" * len(nonce)) + dlg.loss_mask_bytes[u_end:]
                        if len(raw_mod) <= eff_max_doc_bytes and verify_mask_alignment(raw_mod, mask_mod):
                            dlg = EncodedDialogue(
                                raw_bytes=raw_mod,
                                loss_mask_bytes=mask_mod,
                                lang_id=dlg.lang_id,
                                bucket_id=dlg.bucket_id,
                                num_turns=dlg.num_turns,
                                eot_positions=tuple(p + len(nonce) for p in dlg.eot_positions),
                                style_id=dlg.style_id,
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
    for b_id in bucket_ids:
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
        lang_bytes = {i: 0 for i in range(6)}
        bkt_bytes = {b: 0 for b in bucket_ids}
        style_bytes = {i: 0 for i in range(5)}
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
                style_bytes[int(d.style_id)] = style_bytes.get(int(d.style_id), 0) + n
                sup_bytes += d.loss_mask_bytes.count(1)

        meta = {
            "split": split_name,
            "num_docs": len(docs),
            "total_bytes": offset,
            "supervised_bytes": sup_bytes,
            "supervised_ratio": round(sup_bytes / max(offset, 1), 4),
            "bytes_by_lang": lang_bytes,
            "bytes_by_bucket": bkt_bytes,
            "bytes_by_style": style_bytes,
        }
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    train_meta = _write_split("train", train_docs)
    val_meta = _write_split("val", val_docs)

    summary = {
        "schema_version": 2 if cognitive_v2 else 1,
        "cognitive_v2": bool(cognitive_v2),
        "max_doc_bytes": eff_max_doc_bytes,
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
            edges = [0, 512, 1024, 2048, 4096, 1 << 60]
            byte_masses_t: list[float] = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                if lo >= self.max_doc_bytes:
                    break
                eff_hi = self.max_doc_bytes if hi >= self.max_doc_bytes else hi
                idxs = np.nonzero((self._lengths > lo) & (self._lengths <= eff_hi))[0]
                if len(idxs) > 0:
                    self._tier_doc_indices.append(idxs)
                    # Weight outer tier selection by byte mass (cancels B / mu_k packing count bias!)
                    byte_masses_t.append(float(self._lengths[idxs].astype(np.float64).sum()))
            if len(self._tier_doc_indices) > 1:
                tot_b = sum(byte_masses_t)
                acc = 0.0
                for bm in byte_masses_t:
                    acc += bm / tot_b
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

    def get_batch(self, batch_idx: int, sync_group: Optional[int] = None) -> dict[str, torch.Tensor]:
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
            eff_sync = (int(batch_idx) // self.bucket_sync_world_size) if sync_group is None else int(sync_group)
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

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.get_batch(batch_idx, sync_group=None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 5 Pragmatic / Holistic Cognitive SFT Dataset")
    ap.add_argument("--out-dir", type=str, default="artifacts/sft_data")
    ap.add_argument("--train-mb", type=float, default=20.0)
    ap.add_argument("--val-mb", type=float, default=1.0)
    ap.add_argument("--smoke-quota", action="store_true", help="Build 1MB train + 128KB val locally")
    ap.add_argument("--offline", action="store_true", help="Zero-network offline synthesis mode")
    ap.add_argument(
        "--cognitive-v2",
        action="store_true",
        help="Build DCPA v2 35:35:30 holistic cognitive SFT (Socratic logic + multi-draft polish + Py/SQL/JS bugfix, zero Rust)",
    )
    ap.add_argument("--max-doc-bytes", type=int, default=None)
    ap.add_argument("--hf-token", type=str, default=None, help="Optional HF Hub token (overrides HF_TOKEN/local.env/Colab Secrets)")
    ap.add_argument("--seed", type=int, default=202605)
    args = ap.parse_args()
    if not args.offline:
        from data.hf_auth import configure_hf_auth

        configure_hf_auth(args.hf_token)

    res = build_stage5_sft_corpus(
        out_dir=args.out_dir,
        train_mb=args.train_mb,
        val_mb=args.val_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
        max_doc_bytes=args.max_doc_bytes,
        cognitive_v2=args.cognitive_v2,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
