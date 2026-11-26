"""Bilingual (EN/ZH + Python/JS) Multi-Constraint Instruction-Following Builder (`cap_instruct`).

Builds pure UTF-8 byte instruction-following shards (`cap_instruct_{train,val}.{bin,idx,loss_mask.bin}`):
  - Covers ONLY active domains `0=en`, `1=zh`, `3=python`, `5=javascript` (strictly zero Japanese/Rust).
  - Streams ungated HuggingFace Parquet instruction corpora when online (`HuggingFaceTB/smoltalk`
    `self-oss-instruct`, `systemchats-30k`, `everyday-conversations`, `explore-instruct-rewriting`,
    plus `OpenAssistant/oasst2` / `CohereForAI/aya_datasets` EN & ZH subsets).
  - Synthesizes deterministic high-precision multi-constraint instruction samples offline & as top-up:
      1. Strict JSON / Markdown table extraction & schema adherence (EN & ZH)
      2. Constrained rewriting (exact bullet counts, forbidden words, length caps, EN<->ZH translation)
      3. Cross-language `Python <-> JavaScript` function translation & bug repair (100% AST-gated)
      4. Multi-turn follow-up constraint modification with compact `\x1dS\n...\x1dE\n` planning anchors
         (triggering Slot-0 Thesis Pinning on complex instructions).
  - Verifies 1-to-1 `loss_mask.bin` byte alignment (`0` on user turns, `1` on assistant turns + `\x04\n`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import struct
import sys
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_sft_stage5 import is_stdlib_only, is_valid_code_ast
from data.ragged_dataloader import IDX_RECORD_FMT, assert_not_drive_mmap
from utils.chat_protocol import (
    SCRATCH_END,
    SCRATCH_START,
    EncodedDialogue,
    detect_lang_id,
    encode_dialogue_turns,
    sanitize_raw_text,
    verify_mask_alignment,
)

SOURCE_ID_INSTRUCT: int = 16
_KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
_RUST_FENCE_RE = re.compile(r"```(?:rust|rs)\b", re.IGNORECASE)

_PY_JS_TRANSLATION_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    (
        "clampRange",
        "def clamp_range(xs: list[int], lo: int, hi: int) -> list[int]:\n    return [min(hi, max(lo, x)) for x in xs]",
        "function clampRange(xs, lo, hi) {\n  return xs.map((x) => Math.min(hi, Math.max(lo, x)));\n}",
        "clamp each element into [lo, hi]",
    ),
    (
        "uniqueSorted",
        "def unique_sorted(xs: list[int]) -> list[int]:\n    return sorted(set(xs))",
        "function uniqueSorted(xs) {\n  return Array.from(new Set(xs)).sort((a, b) => a - b);\n}",
        "deduplicate integers and sort ascending",
    ),
    (
        "countFreq",
        "def count_freq(words: list[str]) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts",
        "function countFreq(words) {\n  const counts = {};\n  for (const w of words) {\n    counts[w] = (counts[w] || 0) + 1;\n  }\n  return counts;\n}",
        "count string frequencies into a dictionary/object",
    ),
    (
        "twoSumIndices",
        "def two_sum_indices(nums: list[int], target: int) -> list[int]:\n    seen: dict[int, int] = {}\n    for i, x in enumerate(nums):\n        need = target - x\n        if need in seen:\n            return [seen[need], i]\n        seen[x] = i\n    return []",
        "function twoSumIndices(nums, target) {\n  const seen = new Map();\n  for (let i = 0; i < nums.length; i += 1) {\n    const need = target - nums[i];\n    if (seen.has(need)) {\n      return [seen.get(need), i];\n    }\n    seen.set(nums[i], i);\n  }\n  return [];\n}",
        "find two indices summing to target in O(n)",
    ),
    (
        "flattenOnce",
        "def flatten_once(rows: list[list[int]]) -> list[int]:\n    out: list[int] = []\n    for r in rows:\n        out.extend(r)\n    return out",
        "function flattenOnce(rows) {\n  const out = [];\n  for (const r of rows) {\n    out.push(...r);\n  }\n  return out;\n}",
        "flatten a 2D array by one level",
    ),
)


def _wrap_plan(plan: str, body: str) -> str:
    p_b = sanitize_raw_text(plan).strip().encode("utf-8")[:160].decode("utf-8", errors="ignore")
    return (
        SCRATCH_START.decode("ascii")
        + p_b
        + SCRATCH_END.decode("ascii")
        + sanitize_raw_text(body).strip()
    )


def _sample_procedural_instruct(rng: random.Random, idx: int) -> EncodedDialogue:
    kind = idx % 5
    is_zh = (idx % 2) == 1

    if kind == 0:
        # Strict JSON schema extraction + Turn-2 constraint update
        svc = rng.choice(("gateway", "indexer", "auth", "scheduler", "telemetry", "router"))
        shard_id = rng.randint(1, 64)
        host = f"{svc}-{shard_id:02d}.{rng.choice(('core.local', 'internal.net', 'edge.io'))}"
        port = rng.choice((443, 8080, 8443, 9000, 9092))
        timeout_ms = rng.randint(12, 180) * 25
        retries = rng.randint(1, 6)
        tls = rng.choice((True, False))
        region = rng.choice(("ap-east-1", "ap-northeast-1", "us-west-2", "eu-central-1", "us-east-1"))
        obj1 = {"host": host, "port": port, "timeout_ms": timeout_ms, "retries": retries, "tls": tls}
        obj2 = {
            "host": host,
            "port": port,
            "timeout_ms": timeout_ms * 2,
            "retries": max(0, retries - 1),
            "tls": True,
            "region": region,
        }
        j1 = json.dumps(obj1, ensure_ascii=False)
        j2 = json.dumps(obj2, ensure_ascii=False)
        if is_zh:
            u1 = (
                f"请严格按 JSON 格式提取配置（仅包含键 `host`、`port`、`timeout_ms`、`retries`、`tls`，禁止输出代码块或解释）："
                f"节点 {host} 端口 {port}，超时 {timeout_ms} 毫秒，重试 {retries} 次，TLS {'开启' if tls else '关闭'}。"
            )
            a1 = _wrap_plan("keys=[host,port,timeout_ms,retries,tls]; raw_json", j1)
            u2 = f"把 `timeout_ms` 翻倍，`retries` 减 1，强制开启 `tls`，并新增字段 `region` 为 \"{region}\"，只输出纯 JSON。"
            return encode_dialogue_turns(
                [(u1, a1), (u2, j2)], lang_id=1, bucket_id=SOURCE_ID_INSTRUCT, style_id=1, allow_assistant_controls=True
            )
        u1 = (
            f"Output ONLY a valid minified JSON object with keys `host`, `port`, `timeout_ms`, `retries`, and `tls` (no markdown fences): "
            f"connect to {host}:{port} with {timeout_ms}ms timeout, {retries} retries, and TLS {'enabled' if tls else 'disabled'}."
        )
        a1 = _wrap_plan("keys=[host,port,timeout_ms,retries,tls]; no_fence", j1)
        u2 = (
            f"Update the JSON: double `timeout_ms`, decrement `retries` by 1, force `tls` to true, "
            f"and add `region` set to \"{region}\". Output JSON only."
        )
        return encode_dialogue_turns(
            [(u1, a1), (u2, j2)], lang_id=0, bucket_id=SOURCE_ID_INSTRUCT, style_id=1, allow_assistant_controls=True
        )

    if kind == 1:
        # Exact N-bullet + metric grounding + forbidden-word constraint following
        qps = rng.randint(12, 95) * 100
        p99_ms = rng.randint(180, 960)
        pool_cap = rng.choice((32, 48, 64, 96, 128))
        new_timeout = rng.choice((150, 200, 250, 300, 400))
        bullet_marker = rng.choice(("- ", "* "))
        if is_zh:
            forbid_zh = rng.choice(("建议", "或许", "大概"))
            pts_zh = [
                f"将连接池上限从 {pool_cap} 扩容至 {pool_cap * 2} 以承接 {qps} QPS",
                f"设置单次慢查询硬超时为 {new_timeout}ms 压降 {p99_ms}ms 尾延迟",
                f"开启热点只读键本地缓存并隔离慢请求队列",
            ]
            u1 = (
                f"当前服务流量 {qps} QPS、P99 延迟 {p99_ms}ms、连接池上限 {pool_cap}。"
                f"请给出恰好 3 条整改动作：每行以 `{bullet_marker}` 开头，包含具体数值，且全文禁止出现「{forbid_zh}」。"
            )
            body = "\n".join(f"{bullet_marker}{p}" for p in pts_zh)
            assert forbid_zh not in body
            a1 = _wrap_plan(f"bullets=3; prefix={bullet_marker!r}; forbid={forbid_zh}", body)
            return encode_dialogue_turns(
                [(u1, a1)], lang_id=1, bucket_id=SOURCE_ID_INSTRUCT, style_id=1, allow_assistant_controls=True
            )
        forbid_en = rng.choice(("simply", "just", "maybe"))
        pts_en = [
            f"Raise pool capacity from {pool_cap} to {pool_cap * 2} connections for {qps} QPS load",
            f"Enforce a {new_timeout}ms query deadline to cut the {p99_ms}ms P99 tail",
            "Enable bounded read-through caching and isolate slow-tenant queues",
        ]
        u1 = (
            f"Given {qps} QPS traffic, {p99_ms}ms P99 latency, and a {pool_cap}-connection pool, "
            f"list exactly 3 concrete mitigations starting with `{bullet_marker}` without using the word '{forbid_en}'."
        )
        body = "\n".join(f"{bullet_marker}{p}" for p in pts_en)
        assert forbid_en not in body.lower()
        a1 = _wrap_plan(f"bullets=3; prefix={bullet_marker!r}; forbid={forbid_en}", body)
        return encode_dialogue_turns(
            [(u1, a1)], lang_id=0, bucket_id=SOURCE_ID_INSTRUCT, style_id=1, allow_assistant_controls=True
        )

    if kind == 2:
        # Parameterized cross-language Python <-> JavaScript translation (100% AST-verified)
        sub_k = rng.randrange(4)
        lo_v = rng.randint(-20, 15)
        hi_v = lo_v + rng.randint(10, 90)
        scale_v = rng.randint(2, 9)
        mod_v = rng.choice((3, 5, 7, 11))
        suf = rng.choice(("", "_v2", "_fast", "_clean"))
        js_suf = {"": "", "_v2": "V2", "_fast": "Fast", "_clean": "Clean"}[suf]

        if sub_k == 0:
            js_name = f"clampScale{js_suf}"
            py_code = (
                f"def clamp_scale{suf}(xs: list[int]) -> list[int]:\n"
                f"    return [min({hi_v}, max({lo_v}, x * {scale_v})) for x in xs]"
            )
            js_code = (
                f"function {js_name}(xs) {{\n"
                f"  return xs.map((x) => Math.min({hi_v}, Math.max({lo_v}, x * {scale_v})));\n"
                f"}}"
            )
            desc = f"multiply each integer by {scale_v} and clamp into [{lo_v}, {hi_v}]"
        elif sub_k == 1:
            js_name = f"sumFilteredMod{js_suf}"
            py_code = (
                f"def sum_filtered_mod{suf}(xs: list[int]) -> int:\n"
                f"    return sum(x * {scale_v} for x in xs if x > {lo_v} and x % {mod_v} != 0)"
            )
            js_code = (
                f"function {js_name}(xs) {{\n"
                f"  return xs.filter((x) => x > {lo_v} && x % {mod_v} !== 0).reduce((acc, x) => acc + x * {scale_v}, 0);\n"
                f"}}"
            )
            desc = f"sum x * {scale_v} for elements > {lo_v} not divisible by {mod_v}"
        elif sub_k == 2:
            js_name = f"groupByLenCap{js_suf}"
            cap_len = rng.randint(3, 12)
            py_code = (
                f"def group_by_len_cap{suf}(words: list[str]) -> dict[int, int]:\n"
                f"    counts: dict[int, int] = {{}}\n"
                f"    for w in words:\n"
                f"        k = min({cap_len}, len(w))\n"
                f"        counts[k] = counts.get(k, 0) + 1\n"
                f"    return counts"
            )
            js_code = (
                f"function {js_name}(words) {{\n"
                f"  const counts = {{}};\n"
                f"  for (const w of words) {{\n"
                f"    const k = Math.min({cap_len}, w.length);\n"
                f"    counts[k] = (counts[k] || 0) + 1;\n"
                f"  }}\n"
                f"  return counts;\n"
                f"}}"
            )
            desc = f"count strings grouped by min({cap_len}, length)"
        else:
            js_name, py_code, js_code, desc = rng.choice(_PY_JS_TRANSLATION_PAIRS)
            if suf:
                js_name = f"{js_name}{js_suf}"
                py_code = py_code.replace("(", f"{suf}(", 1)
                js_code = js_code.replace("(", f"{js_suf}(", 1)

        assert is_valid_code_ast(py_code, "python")
        assert is_valid_code_ast(js_code, "javascript")
        if rng.random() < 0.5:
            u1 = (
                f"将下面的 Python 函数等价改写为现代 ES6+ JavaScript 函数 `{js_name}`（仅输出代码块）：\n```python\n{py_code}\n```"
                if is_zh
                else f"Translate this Python function into idiomatic ES6+ JavaScript named `{js_name}` ({desc}). Output only the code block:\n```python\n{py_code}\n```"
            )
            return encode_dialogue_turns(
                [(u1, f"```javascript\n{js_code}\n```")], lang_id=5, bucket_id=SOURCE_ID_INSTRUCT, style_id=4
            )
        u1 = (
            f"将下面的 JavaScript 函数等价改写为带类型注解的纯 Python 函数（仅输出代码块）：\n```javascript\n{js_code}\n```"
            if is_zh
            else f"Convert this JavaScript function into type-annotated pure Python ({desc}). Output only the code block:\n```javascript\n{js_code}\n```"
        )
        return encode_dialogue_turns(
            [(u1, f"```python\n{py_code}\n```")], lang_id=3, bucket_id=SOURCE_ID_INSTRUCT, style_id=4
        )

    if kind == 3:
        # Parameterized Python / JavaScript bug diagnosis + minimal fix instruction
        bias_v = rng.randint(1, 50)
        fn_tag = rng.choice(("Scores", "Metrics", "Weights", "Deltas", "Latencies"))
        if rng.random() < 0.5:
            if rng.random() < 0.5:
                bug_js = (
                    f"function mean{fn_tag}(xs) {{\n"
                    f"  let sum = {bias_v};\n"
                    f"  for (let i = 0; i <= xs.length; i += 1) {{\n"
                    f"    sum += xs[i];\n"
                    f"  }}\n"
                    f"  return xs.length ? sum / xs.length : 0;\n"
                    f"}}"
                )
                fix_js = bug_js.replace("i <= xs.length", "i < xs.length")
                expl = (
                    "循环条件 `i <= xs.length` 越界读取了 `xs[xs.length]`（`undefined`），导致数值累加变为 `NaN`。"
                    if is_zh
                    else "The loop condition `i <= xs.length` reads `xs[xs.length]` (`undefined`), turning `sum` into `NaN`."
                )
                plan_s = "bug: i<=xs.length OOB undefined -> NaN; fix: i<xs.length"
            else:
                top_k = rng.randint(2, 6)
                bug_js = (
                    f"function top{top_k}{fn_tag}(xs) {{\n"
                    f"  return [...xs].sort().slice(-{top_k});\n"
                    f"}}"
                )
                fix_js = (
                    f"function top{top_k}{fn_tag}(xs) {{\n"
                    f"  return [...xs].sort((a, b) => a - b).slice(-{top_k});\n"
                    f"}}"
                )
                expl = (
                    "JavaScript 默认 `Array.prototype.sort()` 按 UTF-16 字符串字典序排序（如 `10 < 2`），数值升序必须传入 `(a, b) => a - b`。"
                    if is_zh
                    else "Default `Array.prototype.sort()` sorts numbers lexicographically as strings; pass `(a, b) => a - b` for numeric order."
                )
                plan_s = "bug: lexicographic .sort() on numbers; fix: (a, b) => a - b"
            assert is_valid_code_ast(fix_js, "javascript")
            u1 = (
                f"定位下面 JavaScript 函数的逻辑缺陷并给出修复后的完整代码：\n```javascript\n{bug_js}\n```"
                if is_zh
                else f"Diagnose the bug in this JavaScript function and output the fixed implementation:\n```javascript\n{bug_js}\n```"
            )
            a1 = _wrap_plan(plan_s, f"{expl}\n```javascript\n{fix_js}\n```")
            return encode_dialogue_turns(
                [(u1, a1)], lang_id=5, bucket_id=SOURCE_ID_INSTRUCT, style_id=4, allow_assistant_controls=True
            )

        py_tag = fn_tag.lower()
        bug_py = (
            f"def record_{py_tag}(val: int, offset: int = {bias_v}, history: list[int] = []) -> list[int]:\n"
            f"    history.append(val + offset)\n"
            f"    return history"
        )
        fix_py = (
            f"def record_{py_tag}(val: int, offset: int = {bias_v}, history: list[int] | None = None) -> list[int]:\n"
            f"    if history is None:\n"
            f"        history = []\n"
            f"    history.append(val + offset)\n"
            f"    return history"
        )
        assert is_valid_code_ast(fix_py, "python")
        u1 = (
            f"修复下面 Python 函数中可变默认参数跨调用共享状态的 Bug，并用一句话说明原因：\n```python\n{bug_py}\n```"
            if is_zh
            else f"Fix the mutable default argument bug in this Python function and explain the cause in one sentence:\n```python\n{bug_py}\n```"
        )
        expl = (
            "默认列表 `[]` 在函数定义时只创建一次，缺省调用会持续修改同一个共享列表。"
            if is_zh
            else "The default list `[]` is created once at function definition time, mutating shared state across calls."
        )
        a1 = _wrap_plan("mutable default [] -> None sentinel", f"{expl}\n```python\n{fix_py}\n```")
        return encode_dialogue_turns(
            [(u1, a1)], lang_id=3, bucket_id=SOURCE_ID_INSTRUCT, style_id=4, allow_assistant_controls=True
        )

    # kind == 4: Parameterized Markdown table structured extraction + Turn-2 numeric filter
    names_pool = ("BatchSync", "StreamPipe", "EdgeQueue", "VectorIdx", "LogRouter", "KvShard", "RpcMesh")
    picked = rng.sample(names_pool, 3)
    lats = [rng.randint(3, 12), rng.randint(14, 26), rng.randint(28, 65)]
    rng.shuffle(lats)
    slas = [rng.choice(("99.99%", "99.95%", "99.90%", "99.50%")) for _ in range(3)]
    thr = rng.randint(13, 27)
    rows = list(zip(picked, lats, slas))
    tbl_all = "| Name | Latency_ms | SLA |\n| :--- | ---: | :--- |\n" + "\n".join(
        f"| {n} | {lat} | {sla} |" for n, lat, sla in rows
    )
    kept = [(n, lat, sla) for n, lat, sla in rows if lat < thr]
    tbl_fast = "| Name | Latency_ms | SLA |\n| :--- | ---: | :--- |\n" + "\n".join(
        f"| {n} | {lat} | {sla} |" for n, lat, sla in kept
    )
    if is_zh:
        spec_zh = "；".join(f"{n} 延迟 {lat}ms SLA {sla}" for n, lat, sla in rows)
        u1 = f"将以下服务指标整理为包含 `Name`、`Latency_ms`、`SLA` 三列的 Markdown 表格（禁止多余说明）：{spec_zh}。"
        u2 = f"仅保留 `Latency_ms < {thr}` 的行，重新输出该 Markdown 表格。"
        return encode_dialogue_turns([(u1, tbl_all), (u2, tbl_fast)], lang_id=1, bucket_id=SOURCE_ID_INSTRUCT, style_id=1)
    spec_en = ", ".join(f"{n} ({lat}ms, {sla})" for n, lat, sla in rows)
    u1 = f"Format these services as a Markdown table with columns `Name`, `Latency_ms`, and `SLA` (table only): {spec_en}."
    u2 = f"Filter the table to keep only rows with `Latency_ms < {thr}`."
    return encode_dialogue_turns([(u1, tbl_all), (u2, tbl_fast)], lang_id=0, bucket_id=SOURCE_ID_INSTRUCT, style_id=1)


def _stream_hf_instruct(max_items: int) -> Iterator[EncodedDialogue]:
    """Stream clean EN/ZH/Python/JS instruction-following dialogues from ungated HF Parquet repos."""
    try:
        from datasets import load_dataset
    except Exception:
        return

    yielded = 0
    smol_subsets = ("self-oss-instruct", "explore-instruct-rewriting", "everyday-conversations", "systemchats-30k")
    per_sub = max(1, max_items // (len(smol_subsets) + 1))

    for sub in smol_subsets:
        if yielded >= max_items:
            break
        got_sub = 0
        try:
            ds = load_dataset("HuggingFaceTB/smoltalk", sub, split="train", streaming=True)
            for row in ds:
                if got_sub >= per_sub or yielded >= max_items:
                    break
                msgs = row.get("messages") or []
                sys_prompt = ""
                pairs: list[tuple[str, str]] = []
                idx_m = 0
                if msgs and msgs[0].get("role") == "system":
                    sys_prompt = sanitize_raw_text(str(msgs[0].get("content", ""))).strip()
                    idx_m = 1
                while idx_m + 1 < len(msgs):
                    if msgs[idx_m].get("role") == "user" and msgs[idx_m + 1].get("role") == "assistant":
                        u = sanitize_raw_text(str(msgs[idx_m].get("content", ""))).strip()
                        a = sanitize_raw_text(str(msgs[idx_m + 1].get("content", ""))).strip()
                        if u and a:
                            if not pairs and sys_prompt and len(sys_prompt) <= 280:
                                u = f"[Constraint: {sys_prompt}]\n{u}"
                            pairs.append((u, a))
                        idx_m += 2
                    else:
                        break
                if not pairs:
                    continue
                pairs = pairs[:3]
                joined = "\n".join(u + "\n" + a for u, a in pairs)
                # Reject Japanese Kana or Rust code fences to keep corpus strictly EN/ZH/Python/JS
                if _KANA_RE.search(joined) or _RUST_FENCE_RE.search(joined):
                    continue
                lid = detect_lang_id(joined)
                if lid in (2, 4):
                    continue
                if lid in (3, 5):
                    lang_tag = "python" if lid == 3 else "javascript"
                    m_code = re.search(r"```(?:python|py|javascript|js)\s*\n(.*?)```", joined, re.DOTALL | re.IGNORECASE)
                    if m_code and (not is_stdlib_only(m_code.group(1), lang_tag) or not is_valid_code_ast(m_code.group(1), lang_tag)):
                        continue
                enc = encode_dialogue_turns(pairs, lang_id=lid, bucket_id=SOURCE_ID_INSTRUCT, style_id=4 if lid in (3, 5) else 1)
                if 80 <= len(enc.raw_bytes) <= 3200:
                    yield enc
                    got_sub += 1
                    yielded += 1
        except Exception:
            continue

    # Stream Chinese + English human-rated multi-turn instruction trees from OpenAssistant/oasst2
    if yielded < max_items:
        try:
            ds_oa = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
            pending_prompt: dict[str, tuple[str, int]] = {}
            for row in ds_oa:
                if yielded >= max_items:
                    break
                lang_code = str(row.get("lang", "")).lower()
                if not (lang_code.startswith("zh") or lang_code.startswith("en")):
                    continue
                role = str(row.get("role", ""))
                msg_id = str(row.get("message_id", ""))
                parent_id = str(row.get("parent_id") or "")
                txt = sanitize_raw_text(str(row.get("text", ""))).strip()
                if not txt or _KANA_RE.search(txt) or _RUST_FENCE_RE.search(txt):
                    continue
                if role == "prompter" and not parent_id:
                    if len(pending_prompt) > 4096:
                        pending_prompt.clear()
                    pending_prompt[msg_id] = (txt, 1 if lang_code.startswith("zh") else 0)
                elif role == "assistant" and parent_id in pending_prompt:
                    u_txt, base_lid = pending_prompt.pop(parent_id)
                    lid = detect_lang_id(u_txt + "\n" + txt)
                    if lid in (2, 4):
                        lid = base_lid
                    enc = encode_dialogue_turns([(u_txt, txt)], lang_id=lid, bucket_id=SOURCE_ID_INSTRUCT, style_id=1)
                    if 80 <= len(enc.raw_bytes) <= 3200:
                        yield enc
                        yielded += 1
        except Exception:
            pass


def build_cap_instruct_corpus(
    out_dir: str | Path = "artifacts/cap_data",
    target_mb: float = 100.0,
    smoke_quota: bool = False,
    offline: bool = False,
    seed: int = 202614,
) -> dict:
    """Build `cap_instruct_{train,val}.{bin,idx,loss_mask.bin}` instruction-following shards."""
    t0 = time.perf_counter()
    out_path = Path(out_dir)
    assert_not_drive_mmap(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if smoke_quota:
        train_target = int(1.85 * 1024 * 1024)
        val_target = int(0.15 * 1024 * 1024)
    else:
        total_b = int(target_mb * 1024 * 1024)
        val_target = max(int(total_b * 0.05), 256 * 1024)
        train_target = total_b - val_target

    total_target = train_target + val_target
    rng = random.Random(seed)
    seen_hashes: set[bytes] = set()
    docs: list[EncodedDialogue] = []
    acc_bytes = 0

    def _try_add(enc: EncodedDialogue) -> bool:
        nonlocal acc_bytes
        if acc_bytes >= total_target:
            return False
        if len(enc.raw_bytes) > 4096:
            return False
        h = hashlib.blake2b(enc.raw_bytes, digest_size=16).digest()
        if h in seen_hashes:
            return False
        assert verify_mask_alignment(enc.raw_bytes, enc.loss_mask_bytes)
        seen_hashes.add(h)
        docs.append(enc)
        acc_bytes += len(enc.raw_bytes)
        return True

    if not offline:
        est_items = max(500, int((total_target * 0.65) // 480))
        for enc in _stream_hf_instruct(max_items=est_items):
            _try_add(enc)
            if acc_bytes >= int(total_target * 0.65):
                break

    proc_idx = 0
    while acc_bytes < total_target and proc_idx < 500_000:
        enc = _sample_procedural_instruct(rng, proc_idx)
        proc_idx += 1
        _try_add(enc)

    rng.shuffle(docs)
    val_docs: list[EncodedDialogue] = []
    train_docs: list[EncodedDialogue] = []
    v_acc = 0
    for d in docs:
        if v_acc < val_target:
            val_docs.append(d)
            v_acc += len(d.raw_bytes)
        else:
            train_docs.append(d)

    def _write_split(split_name: str, items: list[EncodedDialogue]) -> dict:
        bin_p = out_path / f"cap_instruct_{split_name}.bin"
        msk_p = out_path / f"cap_instruct_{split_name}.loss_mask.bin"
        idx_p = out_path / f"cap_instruct_{split_name}.idx"
        offset = 0
        lang_counts: dict[int, int] = {}
        with bin_p.open("wb") as fb, msk_p.open("wb") as fm, idx_p.open("wb") as fi:
            for d in items:
                n = len(d.raw_bytes)
                fb.write(d.raw_bytes)
                fm.write(d.loss_mask_bytes)
                fi.write(struct.pack(IDX_RECORD_FMT, offset, n, d.lang_id, d.bucket_id, 0, d.style_id))
                offset += n
                lang_counts[d.lang_id] = lang_counts.get(d.lang_id, 0) + 1
        return {
            "docs": len(items),
            "total_bytes": offset,
            "lang_counts": lang_counts,
        }

    tr_stat = _write_split("train", train_docs)
    va_stat = _write_split("val", val_docs)
    meta = {
        "dataset": "cap_instruct",
        "capability": "instruct_following_bilingual_code",
        "smoke_quota": smoke_quota,
        "offline": offline,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "total_bytes": tr_stat["total_bytes"] + va_stat["total_bytes"],
        "total_docs": tr_stat["docs"] + va_stat["docs"],
        "train": tr_stat,
        "val": va_stat,
    }
    (out_path / "cap_instruct_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DCPA Bilingual + Python/JS Instruction-Following Shards")
    ap.add_argument("--out-dir", type=str, default="artifacts/cap_data")
    ap.add_argument("--target-mb", type=float, default=100.0)
    ap.add_argument("--smoke-quota", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--hf-token", type=str, default=None, help="Optional HF Hub token (overrides HF_TOKEN/local.env/Colab Secrets)")
    ap.add_argument("--seed", type=int, default=202614)
    args = ap.parse_args()
    if not args.offline:
        from data.hf_auth import configure_hf_auth

        configure_hf_auth(args.hf_token)

    res = build_cap_instruct_corpus(
        out_dir=args.out_dir,
        target_mb=args.target_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
