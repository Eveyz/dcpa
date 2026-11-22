"""Hippocampus Zero-Context Tool-Calling Pure-Byte Dataset Builder (Capability Spec 4.3).

Builds ~60 MB full quota (or 2 MB under `--smoke-quota` locally on M3 Pro) of raw UTF-8 byte
tool-calling shards (`cap_toolcall_{train,val}.{bin,idx,loss_mask.bin}`):
  - Streams ungated Parquet sources first: `HuggingFaceTB/smoltalk` (`name="apigen-80k"`),
    `NousResearch/hermes-function-calling-v1`, optional `Salesforce/xlam-function-calling-60k`
    when `HF_TOKEN` is configured, plus a rich offline multi-domain procedural generator.
  - Zero JSON Schema bloat: compresses all available tools (target + 2..4 distractors) into
    compact single-line Hippocampus signatures at the top of the document (`loss_mask = False`):
      `tool calc(expr: str) -> float : evaluate exact math expression\\n`
  - 30%+ OOV synthetic tool aliasing via `data/wug_generator.py`: generates fresh non-keyword
    Python identifiers (e.g. `zorb_calc`, `glorp_fx_v2`) and synchronously renames BOTH the
    top signature header AND every `\\x11C\\n...\\x13\\n` call site.
  - Exact 60% / 20% / 20% distribution:
      * 60% single-step & multi-hop tool chains
      * 20% tool error self-healing retry (`\\x12O\\nError: ...` -> `\\x1fA\\n\\x1dS\\nfix...\\x1dE\\n\\x11C\\n...`)
      * 20% zero-call restraint negative samples (tools registered, model answers directly without 0x11)
  - 100% AST-validated (`ast.parse(..., mode='eval')`, root `ast.Call`, bracket depth <= 16).
  - Red Line #2: Uses `format_tool_observation_after_suspend(res)` so training observation byte
    streams are 100% bit-for-bit identical to `cli/chat_m3.py` runtime injection!
  - Strict `loss_mask`: header = False, user turn = False, ALL `\\x12O\\n...\\n\\x1fA\\n`
    observations = False; assistant draft, `\\x11C\\n...\\x13\\n`, and final answer + `\\x04\\n` = True.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import os
import random
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.ragged_dataloader import IDX_RECORD_FMT, assert_not_drive_mmap
from data.wug_generator import _sample_en_pseudo_word
from utils.chat_protocol import (
    SCRATCH_END,
    SCRATCH_START,
    SOURCE_ID_TOOLCALL_CHAIN,
    SOURCE_ID_TOOLCALL_NEGATIVE,
    SOURCE_ID_TOOLCALL_RETRY,
    TOOL_CALL_PREFIX,
    TOOL_SUSPEND_END,
    TURN_ASST_PREFIX,
    TURN_EOT_SUFFIX,
    TURN_USER_PREFIX,
    format_tool_observation_after_suspend,
    sanitize_raw_text,
    verify_mask_alignment,
)

STYLE_STRICT_CODE_TOOL = 4


@dataclass(frozen=True)
class ToolSpec:
    name: str
    params: tuple[tuple[str, str], ...]  # (param_name, type_str)
    ret_type: str
    desc: str

    def format_line(self, alias: Optional[str] = None) -> str:
        fn = alias or self.name
        p_str = ", ".join(f"{k}: {t}" for k, t in self.params)
        return f"tool {fn}({p_str}) -> {self.ret_type} : {sanitize_raw_text(self.desc).strip()}"


# Canonical standard tool catalog (also includes all 5 cli/chat_m3.py stdlib tools!)
STANDARD_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("calc", (("expr", "str"),), "float", "evaluate exact math expression"),
    ToolSpec("now_hk", (), "str", "return current Hong Kong ISO timestamp"),
    ToolSpec("list_files", (("path", "str"),), "list", "list files in workspace directory"),
    ToolSpec("read_file_span", (("path", "str"), ("start", "int"), ("end", "int")), "str", "read line range from file"),
    ToolSpec("word_char_count", (("text", "str"),), "dict", "count words chars and utf8 bytes"),
    ToolSpec("get_fx", (("pair", "str"),), "float", "query spot currency exchange rate"),
    ToolSpec("unit_convert", (("value", "float"), ("src", "str"), ("dst", "str")), "float", "convert physical unit value"),
    ToolSpec("lookup_stock", (("ticker", "str"),), "dict", "fetch latest equity quote and volume"),
    ToolSpec("get_weather", (("city", "str"), ("unit", "str")), "dict", "fetch temperature and humidity"),
    ToolSpec("date_diff_days", (("start", "str"), ("end", "str")), "int", "compute calendar days between ISO dates"),
    ToolSpec("query_metric", (("service", "str"), ("window", "str")), "dict", "fetch service p99 latency and error rate"),
    ToolSpec("hash_digest", (("algo", "str"), ("payload", "str")), "str", "compute hex cryptographic digest"),
)
STANDARD_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in STANDARD_TOOLS)


def _is_safe_ast_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str, bool, type(None)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_ast_node(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        return _is_safe_ast_node(node.left) and _is_safe_ast_node(node.right)
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_safe_ast_node(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and _is_safe_ast_node(k) and _is_safe_ast_node(v)
            for k, v in zip(node.keys, node.values)
        )
    return False


def validate_tool_call_ast(call_str: str) -> tuple[bool, str, int]:
    """Verify `call_str` strictly passes `ast.parse(..., mode='eval')`, safe literals, and depth <= 16."""
    s = sanitize_raw_text(call_str).strip()
    if not s or "\n" in s or len(s.encode("utf-8")) > 360:
        return False, "", 0
    # Check bracket nesting depth <= 16
    depth = 0
    max_d = 0
    in_str: Optional[str] = None
    esc = False
    for ch in s:
        if in_str is not None:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch in "([{":
            depth += 1
            if depth > max_d:
                max_d = depth
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return False, "", max_d
    if depth != 0 or max_d < 1 or max_d > 16:
        return False, "", max_d

    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return False, "", max_d
    if not isinstance(tree.body, ast.Call):
        return False, "", max_d
    if not isinstance(tree.body.func, ast.Name):
        return False, "", max_d
    fn_name = tree.body.func.id
    if not fn_name.isidentifier() or keyword.iskeyword(fn_name):
        return False, "", max_d
    for a in tree.body.args:
        if not _is_safe_ast_node(a):
            return False, "", max_d
    for kw in tree.body.keywords:
        if kw.arg is None or not _is_safe_ast_node(kw.value):
            return False, "", max_d
    return True, fn_name, max_d


def make_oov_tool_alias(rng: random.Random, base_name: str, used: set[str]) -> str:
    """Synthesize a fresh OOV valid Python identifier via `wug_generator` (e.g. `zorb_calc`)."""
    tail = base_name.split("_")[-1][:8] or "op"
    for _ in range(32):
        wug = _sample_en_pseudo_word(rng).lower()
        ver = rng.choice(["", "_v2", "_x", "_fn", ""])
        cand = f"{wug}_{tail}{ver}"
        if (
            cand.isidentifier()
            and not keyword.iskeyword(cand)
            and cand not in STANDARD_TOOL_NAMES
            and cand not in used
        ):
            used.add(cand)
            return cand
    cand = f"zorb_{tail}_{rng.randint(10, 99)}"
    used.add(cand)
    return cand


def _format_kw_call(fn_name: str, kwargs: dict[str, Any]) -> str:
    parts = [f"{k}={repr(v)}" for k, v in kwargs.items()]
    return f"{fn_name}({', '.join(parts)})"


def encode_toolcall_episode(
    header_lines: list[str],
    user_query: str,
    steps: list[tuple[Optional[str], Optional[str], Optional[str]]],
    final_answer: str,
    lang_id: int,
    source_id: int,
) -> tuple[bytes, bytes, int, int, int]:
    """Assemble exact byte stream & 1-to-1 `loss_mask` (Red Line #2: uses `format_tool_observation_after_suspend`).

    Each entry in `steps` is `(optional_scratch, call_str, obs_result_str)`.
    For zero-call negative samples, `steps == []`.
    """
    hdr_text = "\n".join(sanitize_raw_text(ln).strip() for ln in header_lines) + "\n"
    hdr_b = hdr_text.encode("utf-8", errors="strict")

    q_clean = sanitize_raw_text(user_query).strip()
    u_frame = TURN_USER_PREFIX + q_clean.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX

    raw_parts: list[bytes] = [hdr_b, u_frame]
    mask_parts: list[bytes] = [b"\x00" * len(hdr_b), b"\x00" * len(u_frame)]

    max_call_depth = 0
    for scratch_txt, call_str, obs_str in steps:
        if scratch_txt:
            sc_b = SCRATCH_START + sanitize_raw_text(scratch_txt).strip().encode("utf-8") + SCRATCH_END
            raw_parts.append(sc_b)
            mask_parts.append(b"\x01" * len(sc_b))

        assert call_str is not None and obs_str is not None
        ok, _, d = validate_tool_call_ast(call_str)
        if not ok:
            raise ValueError(f"Tool call failed AST/depth validation: {call_str!r}")
        max_call_depth = max(max_call_depth, d)

        # Exact byte sequence: `\x11C\n{call}\x13` (supervised) + `\n\x12O\n{res}\n\x1fA\n` (unsupervised!)
        # Note: `\x13` is supervised (2.5x weight); the `\n` right after `\x13` starts the observation handoff
        # matching `cli/chat_m3.py` halting immediately upon sampling byte 0x13 and injecting
        # `format_tool_observation_after_suspend(res)` == `b"\n\x12O\n" + res + b"\n\x1fA\n"`!
        call_b = TOOL_CALL_PREFIX + sanitize_raw_text(call_str).strip().encode("utf-8") + b"\x13"
        raw_parts.append(call_b)
        mask_parts.append(b"\x01" * len(call_b))

        obs_b = format_tool_observation_after_suspend(obs_str)
        # Note: `verify_mask_alignment` expects bytes up to `\x12O\n` to be supervised (including `\n` after `\x13`),
        # so keep `\x13\n` (`TOOL_SUSPEND_END`) in the supervised call span and `\x12O\n{res}\n\x1fA\n` masked 0!
        # Verify bit-identity: `call + b"\x13" + b"\n\x12O\n..."` == `call + b"\x13\n" + b"\x12O\n..."`!
        assert obs_b.startswith(b"\n")
        raw_parts.append(b"\n")
        mask_parts.append(b"\x01")
        obs_tail = obs_b[1:]  # starts right at `b"\x12O\n"`
        raw_parts.append(obs_tail)
        mask_parts.append(b"\x00" * len(obs_tail))

    ans_b = sanitize_raw_text(final_answer).strip().encode("utf-8", errors="strict") + TURN_EOT_SUFFIX
    raw_parts.append(ans_b)
    mask_parts.append(b"\x01" * len(ans_b))

    raw_bytes = b"".join(raw_parts)
    mask_bytes = b"".join(mask_parts)
    assert verify_mask_alignment(raw_bytes, mask_bytes), "Tool-call loss_mask alignment failed!"
    raw_bytes.decode("utf-8", errors="strict")
    return raw_bytes, mask_bytes, int(lang_id), int(source_id), max_call_depth


def _build_header_with_distractors(
    rng: random.Random,
    active_tools: list[tuple[ToolSpec, str]],
) -> list[str]:
    """Compose top single-line Hippocampus tool table with 2..4 shuffled distractors."""
    used_names = {t.name for t, _ in active_tools} | {alias for _, alias in active_tools}
    pool = [t for t in STANDARD_TOOLS if t.name not in used_names]
    n_dist = rng.randint(2, min(4, len(pool)))
    distractors = rng.sample(pool, n_dist)
    lines = [t.format_line(alias) for t, alias in active_tools]
    lines.extend(d.format_line() for d in distractors)
    rng.shuffle(lines)
    return lines


def _gen_procedural_toolcall(
    rng: random.Random,
    idx: int,
    category: str,
    use_oov: bool,
) -> tuple[list[str], str, list[tuple[Optional[str], Optional[str], Optional[str]]], str, int, int]:
    """Generate clean procedural tool-call chains (60%), error self-healing retries (20%), or negatives (20%)."""
    is_zh = (idx % 3) == 1
    lang_id = 1 if is_zh else 0
    used_aliases: set[str] = set()

    if category == "negative":
        # 20% Zero-call restraint: tools registered in Hippocampus, but query needs NO tool call!
        sample_tools = rng.sample(STANDARD_TOOLS, rng.randint(3, 5))
        lines = []
        for t in sample_tools:
            al = make_oov_tool_alias(rng, t.name, used_aliases) if (use_oov and not lines) else t.name
            lines.append(t.format_line(al))
        rng.shuffle(lines)

        a, b = rng.randint(3, 19), rng.randint(2, 14)
        if is_zh:
            pair = rng.choice([
                (f"用一句话解释为什么常量槽位图（K=8）能避免 KV Cache 随长度线性膨胀？",
                 "固定 K=8 个槽位在每一步都将 K+1 个候选归并回 K 个节点，因此动态显存严格保持 O(1) 常数，不随序列长度 T 增长。"),
                (f"如果所有 A 都是 B，且没有任何 B 是 C，那么是否存在既是 A 又是 C 的元素？",
                 "不存在。因为 A 是 B 的子集而 B 与 C 不相交，所以 A 与 C 必然互斥。"),
                (f"把下面句子改写得更简洁有力：'这个系统在很多情况下都能够比较快地完成任务。'",
                 "该系统在多数场景下都能高速完成任务。"),
                (f"不用调用外部工具，口算 {a} + {b} 等于多少？",
                 f"{a} + {b} = {a + b}。"),
            ])
        else:
            pair = rng.choice([
                ("In two concise sentences, explain why a fixed K=8 slot graph avoids O(T) KV-cache growth.",
                 "Every concept step contracts K+1 candidates back into exactly K=8 slots, keeping working memory strictly O(1) regardless of sequence length T."),
                ("If every P is Q and no Q is R, can any P also be R?",
                 "No. Since P is contained in Q and Q is disjoint from R, no element of P can belong to R."),
                ("Rewrite concisely: 'The deployment pipeline was able to finish the build in a relatively short period of time.'",
                 "The deployment pipeline completed the build quickly."),
                (f"Without calling any tool, what is {a} + {b}?",
                 f"{a} + {b} = {a + b}."),
            ])
        return lines, pair[0], [], pair[1], lang_id, SOURCE_ID_TOOLCALL_NEGATIVE

    if category == "retry":
        # 20% Error self-healing retry loop
        kind = idx % 3
        if kind == 0:
            t_spec = STANDARD_TOOLS[5]  # get_fx
            fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
            lines = _build_header_with_distractors(rng, [(t_spec, fn)])
            pair_ok, bad_pair, rate = rng.choice([
                ("USD/HKD", "USDHKD", 7.812),
                ("EUR/USD", "EUR-USD", 1.084),
                ("USD/JPY", "USD_JPY", 154.25),
                ("GBP/USD", "GBPUSD", 1.271),
            ])
            bad_call = _format_kw_call(fn, {"pair": bad_pair})
            ok_call = _format_kw_call(fn, {"pair": pair_ok})
            err_obs = f"Error: ValueError: pair must use 'AAA/BBB' format, got {bad_pair!r}"
            scratch = f"fix pair format: {bad_pair} -> {pair_ok}"
            q = f"查询当前 {pair_ok} 汇率。" if is_zh else f"Check the current {pair_ok} spot rate."
            ans = f"当前 {pair_ok} 汇率为 {rate}。" if is_zh else f"The current {pair_ok} spot rate is {rate}."
            return lines, q, [(None, bad_call, err_obs), (scratch, ok_call, str(rate))], ans, lang_id, SOURCE_ID_TOOLCALL_RETRY
        if kind == 1:
            t_spec = STANDARD_TOOLS[0]  # calc
            fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
            lines = _build_header_with_distractors(rng, [(t_spec, fn)])
            x, y, z = rng.randint(12, 85), rng.randint(3, 19), rng.randint(2, 9)
            bad_call = _format_kw_call(fn, {"expr": f"({x} + {y} * {z}"})  # missing ')'
            val = (x + y) * z
            ok_call = _format_kw_call(fn, {"expr": f"({x} + {y}) * {z}"})
            err_obs = "Error: SyntaxError: '(' was never closed"
            scratch = "fix unbalanced paren -> close ')'"
            q = f"精确计算 ({x} + {y}) * {z}。" if is_zh else f"Compute ({x} + {y}) * {z} exactly."
            ans = f"结果为 {val}。" if is_zh else f"The exact result is {val}."
            return lines, q, [(None, bad_call, err_obs), (scratch, ok_call, str(val))], ans, lang_id, SOURCE_ID_TOOLCALL_RETRY
        t_spec = STANDARD_TOOLS[3]  # read_file_span
        fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
        lines = _build_header_with_distractors(rng, [(t_spec, fn)])
        fname = rng.choice(["configs/dcpa_48m_train.yaml", "README.md", "utils/device.py"])
        s_line, e_line = rng.randint(1, 15), rng.randint(20, 35)
        bad_call = _format_kw_call(fn, {"path": fname, "start": e_line, "end": s_line})
        ok_call = _format_kw_call(fn, {"path": fname, "start": s_line, "end": e_line})
        err_obs = f"Error: ValueError: start ({e_line}) must be <= end ({s_line})"
        scratch = f"swap start/end -> start={s_line}, end={e_line}"
        obs_ok = f"L{s_line}: # section header ({e_line - s_line + 1} lines)"
        q = f"读取 {fname} 第 {s_line} 到 {e_line} 行。" if is_zh else f"Read lines {s_line} to {e_line} of {fname}."
        ans = f"已读取 {fname} 第 {s_line}-{e_line} 行：{obs_ok}" if is_zh else f"Read lines {s_line}-{e_line} of {fname}: {obs_ok}"
        return lines, q, [(None, bad_call, err_obs), (scratch, ok_call, obs_ok)], ans, lang_id, SOURCE_ID_TOOLCALL_RETRY

    # category == "chain": 60% single-step (65%) & 2-hop multi-tool chains (35%)
    multi_hop = (idx % 3) == 0
    if multi_hop:
        t_fx = STANDARD_TOOLS[5]    # get_fx
        t_calc = STANDARD_TOOLS[0]  # calc
        fn_fx = make_oov_tool_alias(rng, t_fx.name, used_aliases) if use_oov else t_fx.name
        fn_calc = make_oov_tool_alias(rng, t_calc.name, used_aliases) if (use_oov and rng.random() < 0.5) else t_calc.name
        lines = _build_header_with_distractors(rng, [(t_fx, fn_fx), (t_calc, fn_calc)])
        pair, rate = rng.choice([("USD/HKD", 7.815), ("EUR/USD", 1.086), ("GBP/HKD", 9.92)])
        amt = rng.randint(50, 2500)
        fee = rng.choice([15, 25, 40])
        c1 = _format_kw_call(fn_fx, {"pair": pair})
        total = round(amt * rate - fee, 2)
        c2 = _format_kw_call(fn_calc, {"expr": f"{amt} * {rate} - {fee}"})
        if is_zh:
            q = f"按实时 {pair} 汇率换算 {amt} 单位，并扣除 {fee} 手续费，最终到手多少？"
            ans = f"按 {pair}={rate} 计算，扣除手续费 {fee} 后最终到手 {total}。"
        else:
            q = f"Convert {amt} at live {pair} spot rate and subtract a fee of {fee}."
            ans = f"At {pair}={rate}, after subtracting the {fee} fee, the net total is {total}."
        return lines, q, [(None, c1, str(rate)), (None, c2, str(total))], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN

    # Single-step tool call rotating across all 5 stdlib tools + extended catalog
    slot = idx % 6
    if slot == 0:
        t_spec = STANDARD_TOOLS[0]  # calc
        fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
        lines = _build_header_with_distractors(rng, [(t_spec, fn)])
        a, b, c = rng.randint(11, 299), rng.randint(7, 99), rng.randint(2, 13)
        expr = f"({a} * {b}) / {c}"
        val = round((a * b) / c, 4)
        call = _format_kw_call(fn, {"expr": expr})
        q = f"请计算 {expr}，保留 4 位小数。" if is_zh else f"Evaluate {expr} rounded to 4 decimals."
        ans = f"{expr} = {val}。" if is_zh else f"{expr} = {val}."
        return lines, q, [(None, call, str(val))], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN
    if slot == 1:
        t_spec = STANDARD_TOOLS[1]  # now_hk
        fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
        lines = _build_header_with_distractors(rng, [(t_spec, fn)])
        ts = f"2026-11-{rng.randint(10, 28):02d}T{rng.randint(8, 22):02d}:{rng.randint(0, 59):02d}:00+08:00"
        call = _format_kw_call(fn, {})
        q = "现在香港时间是几点？" if is_zh else "What is the current Hong Kong time?"
        ans = f"当前香港时间为 {ts}。" if is_zh else f"Current Hong Kong time is {ts}."
        return lines, q, [(None, call, ts)], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN
    if slot == 2:
        t_spec = STANDARD_TOOLS[2]  # list_files
        fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
        lines = _build_header_with_distractors(rng, [(t_spec, fn)])
        sub = rng.choice(["configs", "utils", "data", "cli"])
        files = [f"{sub}/a_{rng.randint(1,9)}.py", f"{sub}/b_{rng.randint(1,9)}.yaml"]
        call = _format_kw_call(fn, {"path": sub})
        q = f"列出 {sub} 目录下的文件。" if is_zh else f"List the files under {sub}."
        ans = f"{sub} 目录包含：{', '.join(files)}。" if is_zh else f"Files in {sub}: {', '.join(files)}."
        return lines, q, [(None, call, json.dumps(files))], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN
    if slot == 3:
        t_spec = STANDARD_TOOLS[4]  # word_char_count
        fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
        lines = _build_header_with_distractors(rng, [(t_spec, fn)])
        sample_s = rng.choice(["Fast weights adapt online.", "DCPA pure byte engine", "Zero KV cache design"])
        res_d = {"words": len(sample_s.split()), "chars": len(sample_s), "bytes": len(sample_s.encode("utf-8"))}
        call = _format_kw_call(fn, {"text": sample_s})
        q = f"统计文本 '{sample_s}' 的词数与字节数。" if is_zh else f"Count words and UTF-8 bytes in '{sample_s}'."
        ans = (
            f"共 {res_d['words']} 个词、{res_d['chars']} 字符、{res_d['bytes']} 字节。"
            if is_zh
            else f"{res_d['words']} words, {res_d['chars']} chars, {res_d['bytes']} UTF-8 bytes."
        )
        return lines, q, [(None, call, json.dumps(res_d))], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN
    if slot == 4:
        t_spec = STANDARD_TOOLS[6]  # unit_convert
        fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
        lines = _build_header_with_distractors(rng, [(t_spec, fn)])
        v = round(rng.uniform(5.0, 250.0), 1)
        km_to_mi = round(v * 0.621371, 3)
        call = _format_kw_call(fn, {"value": v, "src": "km", "dst": "mi"})
        q = f"把 {v} 公里换算成英里。" if is_zh else f"Convert {v} km to miles."
        ans = f"{v} 公里约等于 {km_to_mi} 英里。" if is_zh else f"{v} km equals {km_to_mi} miles."
        return lines, q, [(None, call, str(km_to_mi))], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN

    t_spec = STANDARD_TOOLS[10]  # query_metric
    fn = make_oov_tool_alias(rng, t_spec.name, used_aliases) if use_oov else t_spec.name
    lines = _build_header_with_distractors(rng, [(t_spec, fn)])
    svc = rng.choice(["checkout-api", "search-edge", "ingest-worker", "auth-gw"])
    win = rng.choice(["5m", "15m", "1h"])
    obs_d = {"p99_ms": rng.randint(14, 180), "err_rate": round(rng.uniform(0.0, 0.02), 4)}
    call = _format_kw_call(fn, {"service": svc, "window": win})
    q = f"查询 {svc} 最近 {win} 的 P99 延迟和错误率。" if is_zh else f"Check {svc} p99 latency and error rate over the last {win}."
    ans = (
        f"{svc} 最近 {win} P99 延迟为 {obs_d['p99_ms']}ms，错误率 {obs_d['err_rate']}。"
        if is_zh
        else f"{svc} ({win}): p99 latency {obs_d['p99_ms']}ms, error rate {obs_d['err_rate']}."
    )
    return lines, q, [(None, call, json.dumps(obs_d))], ans, lang_id, SOURCE_ID_TOOLCALL_CHAIN


def _parse_json_tools_to_specs(tools_obj: Any) -> list[ToolSpec]:
    """Convert OpenAI/xLAM JSON tool schema objects into compact single-line `ToolSpec`s."""
    specs: list[ToolSpec] = []
    if isinstance(tools_obj, str):
        try:
            tools_obj = json.loads(tools_obj)
        except Exception:
            return []
    if not isinstance(tools_obj, list):
        return []
    for item in tools_obj[:6]:
        if not isinstance(item, dict):
            continue
        fn_obj = item.get("function", item)
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(fn_obj.get("name", ""))).strip("_")
        if not name or not name[0].isalpha() or keyword.iskeyword(name):
            continue
        desc = sanitize_raw_text(str(fn_obj.get("description", "tool action"))).strip().replace("\n", " ")[:64]
        params_raw = fn_obj.get("parameters") or {}
        if isinstance(params_raw, dict) and "properties" in params_raw:
            props = params_raw.get("properties") or {}
        elif isinstance(params_raw, dict):
            props = params_raw
        else:
            props = {}
        p_list: list[tuple[str, str]] = []
        for pk, pv in list(props.items())[:5]:
            pk_clean = re.sub(r"[^A-Za-z0-9_]", "_", str(pk)).strip("_")
            if not pk_clean or not pk_clean.isidentifier() or keyword.iskeyword(pk_clean):
                continue
            pt = str(pv.get("type", "any"))[:12] if isinstance(pv, dict) else "any"
            p_list.append((pk_clean, pt))
        specs.append(ToolSpec(name=name, params=tuple(p_list), ret_type="any", desc=desc or "execute tool"))
    return specs


def _stream_hf_toolcalls(max_items: int, rng: random.Random) -> Iterator[tuple[list[str], str, list[tuple[Optional[str], Optional[str], Optional[str]]], str, int, int]]:
    """Stream ungated `HuggingFaceTB/smoltalk` (`apigen-80k`) + `hermes-function-calling-v1` (+ optional xLAM)."""
    try:
        from datasets import load_dataset
    except Exception:
        return

    yielded = 0
    # 1. Ungated HuggingFaceTB/smoltalk apigen-80k
    try:
        ds = load_dataset("HuggingFaceTB/smoltalk", "apigen-80k", split="train", streaming=True)
        for row in ds:
            if yielded >= max_items:
                break
            msgs = row.get("messages") or []
            sys_txt = next((str(m.get("content", "")) for m in msgs if m.get("role") == "system"), "")
            u_txt = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "")
            a_txt = next((str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"), "")
            if not u_txt or not a_txt or len(u_txt.encode("utf-8")) > 500:
                continue
            # Extract JSON tool definitions from system prompt & JSON call list from assistant
            j_match = re.search(r"\[.*\]", sys_txt, re.DOTALL)
            a_match = re.search(r"\[.*\]", a_txt, re.DOTALL)
            if not j_match or not a_match:
                continue
            specs = _parse_json_tools_to_specs(j_match.group(0))
            try:
                calls_raw = json.loads(a_match.group(0))
            except Exception:
                continue
            if not specs or not isinstance(calls_raw, list) or not calls_raw:
                continue
            c0 = calls_raw[0]
            if not isinstance(c0, dict) or "name" not in c0:
                continue
            orig_fn = re.sub(r"[^A-Za-z0-9_]", "_", str(c0["name"])).strip("_")
            args_dict = c0.get("arguments") or {}
            if not isinstance(args_dict, dict):
                continue
            spec_map = {s.name: s for s in specs}
            if orig_fn not in spec_map:
                continue
            use_oov = rng.random() < 0.36
            used_al: set[str] = set()
            eff_fn = make_oov_tool_alias(rng, orig_fn, used_al) if use_oov else orig_fn
            clean_kwargs = {}
            for k, v in list(args_dict.items())[:5]:
                kc = re.sub(r"[^A-Za-z0-9_]", "_", str(k)).strip("_")
                if kc and kc.isidentifier() and not keyword.iskeyword(kc) and isinstance(v, (int, float, str, bool, list, dict)):
                    clean_kwargs[kc] = v
            call_str = _format_kw_call(eff_fn, clean_kwargs)
            ok, _, _ = validate_tool_call_ast(call_str)
            if not ok:
                continue
            lines = [s.format_line(eff_fn if s.name == orig_fn else None) for s in specs[:4]]
            if len(lines) < 3:
                lines.extend(d.format_line() for d in rng.sample(STANDARD_TOOLS, 3 - len(lines)))
            rng.shuffle(lines)
            obs_str = json.dumps({"status": "ok", "tool": eff_fn, "result": "completed"})
            ans = f"Executed `{eff_fn}` with the requested parameters; result confirmed."
            yield lines, u_txt, [(None, call_str, obs_str)], ans, 0, SOURCE_ID_TOOLCALL_CHAIN
            yielded += 1
    except Exception:
        pass

    # 2. Optional gated Salesforce/xlam-function-calling-60k only if HF_TOKEN is explicitly provided
    if yielded < max_items and os.environ.get("HF_TOKEN"):
        try:
            ds_xl = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
            for row in ds_xl:
                if yielded >= max_items:
                    break
                q = str(row.get("query") or "").strip()
                specs = _parse_json_tools_to_specs(row.get("tools"))
                try:
                    ans_calls = json.loads(str(row.get("answers") or "[]"))
                except Exception:
                    continue
                if not q or not specs or not isinstance(ans_calls, list) or not ans_calls:
                    continue
                c0 = ans_calls[0]
                orig_fn = re.sub(r"[^A-Za-z0-9_]", "_", str(c0.get("name", ""))).strip("_")
                args_d = c0.get("arguments") or {}
                if orig_fn not in {s.name for s in specs} or not isinstance(args_d, dict):
                    continue
                use_oov = rng.random() < 0.36
                eff_fn = make_oov_tool_alias(rng, orig_fn, set()) if use_oov else orig_fn
                clean_kw = {
                    re.sub(r"[^A-Za-z0-9_]", "_", str(k)).strip("_"): v
                    for k, v in list(args_d.items())[:5]
                    if isinstance(v, (int, float, str, bool))
                }
                clean_kw = {k: v for k, v in clean_kw.items() if k.isidentifier() and not keyword.iskeyword(k)}
                call_str = _format_kw_call(eff_fn, clean_kw)
                ok, _, _ = validate_tool_call_ast(call_str)
                if not ok:
                    continue
                lines = [s.format_line(eff_fn if s.name == orig_fn else None) for s in specs[:4]]
                rng.shuffle(lines)
                obs_str = json.dumps({"ok": True, "fn": eff_fn})
                yield lines, q, [(None, call_str, obs_str)], f"Completed `{eff_fn}` call successfully.", 0, SOURCE_ID_TOOLCALL_CHAIN
                yielded += 1
        except Exception:
            pass


def build_cap_toolcall_corpus(
    out_dir: str | Path = "artifacts/cap_data",
    target_mb: float = 60.0,
    smoke_quota: bool = False,
    offline: bool = False,
    seed: int = 202613,
) -> dict:
    """Build `cap_toolcall_{train,val}.{bin,idx,loss_mask.bin}` Hippocampus tool-bus shards."""
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
    docs: list[tuple[bytes, bytes, int, int, int]] = []
    acc_bytes = 0
    cat_counts = {SOURCE_ID_TOOLCALL_CHAIN: 0, SOURCE_ID_TOOLCALL_RETRY: 0, SOURCE_ID_TOOLCALL_NEGATIVE: 0}
    oov_docs = 0

    def _try_add(
        header_lines: list[str],
        query: str,
        steps: list[tuple[Optional[str], Optional[str], Optional[str]]],
        ans: str,
        lid: int,
        src_i: int,
    ) -> bool:
        nonlocal acc_bytes, oov_docs
        if acc_bytes >= total_target:
            return False
        try:
            raw_b, msk_b, l_i, s_i, max_d = encode_toolcall_episode(header_lines, query, steps, ans, lid, src_i)
        except Exception:
            return False
        if len(raw_b) < 64 or len(raw_b) > 2048:
            return False
        h = hashlib.blake2b(raw_b, digest_size=12).digest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        # Track whether any called function is an OOV synthetic alias
        has_oov = False
        for _, cs, _ in steps:
            if cs:
                ok, fn_n, _ = validate_tool_call_ast(cs)
                if ok and fn_n not in STANDARD_TOOL_NAMES:
                    has_oov = True
        if has_oov:
            oov_docs += 1
        docs.append((raw_b, msk_b, l_i, s_i, max_d))
        cat_counts[s_i] = cat_counts.get(s_i, 0) + 1
        acc_bytes += len(raw_b)
        return True

    # 1. Stream HF apigen/xlam chains up to 35% of total target (leaving room for retries, negatives, & stdlib tools)
    if not offline:
        hf_cap = int(total_target * 0.35)
        est_items = max(300, int(hf_cap / 450))
        for hdr, q, steps, ans, lid, src_i in _stream_hf_toolcalls(est_items, rng):
            _try_add(hdr, q, steps, ans, lid, src_i)
            if acc_bytes >= hf_cap:
                break

    # 2. Fill remaining quota preserving exact 60% chains / 20% retries / 20% restraint negatives & ~32% OOV rate
    idx = 0
    while acc_bytes < total_target:
        tot_n = max(len(docs), 1)
        f_retry = cat_counts[SOURCE_ID_TOOLCALL_RETRY] / tot_n
        f_neg = cat_counts[SOURCE_ID_TOOLCALL_NEGATIVE] / tot_n
        if f_retry < 0.20:
            cat = "retry"
        elif f_neg < 0.20:
            cat = "negative"
        else:
            cat = "chain"
        # Drive ~38% OOV on call-bearing samples so >= 30% of ALL docs have verified OOV header+call match
        use_oov = rng.random() < 0.42
        hdr, q, steps, ans, lid, src_i = _gen_procedural_toolcall(rng, idx, cat, use_oov)
        _try_add(hdr, q, steps, ans, lid, src_i)
        idx += 1

    rng.shuffle(docs)
    n_val = max(16, int(len(docs) * (val_target / max(float(total_target), 1.0))))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    def _write_split(split: str, split_docs: list[tuple[bytes, bytes, int, int, int]]) -> dict:
        bin_p = out_path / f"cap_toolcall_{split}.bin"
        idx_p = out_path / f"cap_toolcall_{split}.idx"
        msk_p = out_path / f"cap_toolcall_{split}.loss_mask.bin"
        off = 0
        sup_b = 0
        with bin_p.open("wb") as fb, idx_p.open("wb") as fi, msk_p.open("wb") as fm:
            for raw_b, msk_b, lid, src_i, max_d in split_docs:
                n = len(raw_b)
                fb.write(raw_b)
                fm.write(msk_b)
                fi.write(struct.pack(IDX_RECORD_FMT, off, n, lid, src_i, min(max_d, 16), STYLE_STRICT_CODE_TOOL))
                off += n
                sup_b += msk_b.count(1)
        return {
            "split": split,
            "num_docs": len(split_docs),
            "total_bytes": off,
            "supervised_bytes": sup_b,
            "supervised_ratio": round(sup_b / max(off, 1), 4),
        }

    train_meta = _write_split("train", train_docs)
    val_meta = _write_split("val", val_docs)
    total_docs = len(train_docs) + len(val_docs)
    summary = {
        "dataset": "cap_toolcall",
        "smoke_quota": smoke_quota,
        "offline": offline,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "total_bytes": train_meta["total_bytes"] + val_meta["total_bytes"],
        "total_docs": total_docs,
        "oov_called_docs": oov_docs,
        "oov_called_ratio": round(oov_docs / max(total_docs, 1), 4),
        "docs_by_category": {
            "chain": cat_counts[SOURCE_ID_TOOLCALL_CHAIN],
            "retry": cat_counts[SOURCE_ID_TOOLCALL_RETRY],
            "negative": cat_counts[SOURCE_ID_TOOLCALL_NEGATIVE],
        },
        "train": train_meta,
        "val": val_meta,
    }
    (out_path / "cap_toolcall_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DCPA Hippocampus Tool-Calling Pure-Byte Dataset")
    ap.add_argument("--out-dir", type=str, default="artifacts/cap_data")
    ap.add_argument("--target-mb", type=float, default=60.0)
    ap.add_argument("--smoke-quota", action="store_true", help="Build 2 MB local smoke quota")
    ap.add_argument("--offline", action="store_true", help="Zero-network deterministic synthesis")
    ap.add_argument("--seed", type=int, default=202613)
    args = ap.parse_args()
    res = build_cap_toolcall_corpus(
        out_dir=args.out_dir,
        target_mb=args.target_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
