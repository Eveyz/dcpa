"""Stage-3/5 Code Best-Practice Refactoring & Skeleton-First Pure-Byte Builder (`cap_code`).

Implements `docs/DCPA_ROADMAP.md` Section 2.3 & 2.5 Data Alchemy for Engineering Code (ZERO Rust):
  - Covers ONLY `3=python`, `4=sql_struct`, `5=javascript` with `style_id=4` (`strict_eng`).
  - Two synergistic high-leverage training trajectories:
      1. **Skeleton-First Spec -> Idiomatic Code (with Curriculum iCoT)**:
         * Multi-branch algorithms & state machines emit compact `<= 220 B` `\x1dS\nguard:... | bp:... | flow:...\x1dE\n`
           locking invariants into `Slot-0` (`source_id=10`, `ponder_tax_scale=1.0`).
         * Simple `<= 5`-line helpers strip `\x1dS...\x1dE` (`80%`, `source_id=11`, `ponder_tax_scale=0.0`)
           forcing `BCG-PC` implicit Pondering.
      2. **Anti-Pattern -> Best-Practice Contrastive Refactoring Pairs (`mask=0` Iron Law)**:
         * Working-but-flawed code (mutable default args, `O(n^2)` list lookup, bare `except:`, JS `==` coercion,
           in-place caller array mutation, SQL non-sargable full-scan) sits STRICTLY inside `\x1eU\n...\n\x1fA\n`
           (`loss_mask == 0`, NEVER trained to generate!).
         * Assistant diagnoses the code smell inside `\x1dS...\x1dE` and emits clean, idiomatic code (`loss_mask == 1`).
  - Online streams when network is enabled: `nvidia/OpenCodeReasoning-2` & `bigcode/commitpackft` (Python/JS/SQL).
  - 100% of Python & JavaScript assistant outputs are validated by `tree-sitter` (zero syntax errors, depth <= 16),
    and 100% of byte streams pass `verify_mask_alignment`.
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
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_corpus import CodeASTValidator
from data.ragged_dataloader import IDX_RECORD_FMT, assert_not_drive_mmap
from utils.chat_protocol import (
    SCRATCH_END,
    SCRATCH_START,
    SOURCE_ID_REASONING_EXPLICIT,
    SOURCE_ID_REASONING_LATENT,
    TURN_ASST_PREFIX,
    TURN_EOT_SUFFIX,
    TURN_USER_PREFIX,
    sanitize_raw_text,
    verify_mask_alignment,
)

MAX_CODE_SCRATCH_BYTES = 216  # strictly < 240 UTF-8 bytes
STYLE_STRICT_ENG = 4
_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JS_FENCE_RE = re.compile(r"```(?:javascript|js|ts|typescript)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _truncate_utf8(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return s
    b = b[:max_bytes]
    while b and (b[-1] & 0xC0) == 0x80:
        b = b[:-1]
    return b.decode("utf-8", errors="ignore").rstrip(" ;,|->")


def encode_code_alchemy_sample(
    user_prompt: str,
    skeleton: str,
    fenced_code: str,
    lang_id: int,
    *,
    is_latent: bool = False,
    style_id: int = STYLE_STRICT_ENG,
) -> tuple[bytes, bytes, int, int, int]:
    """Encode `(user_prompt [mask=0], optional \x1dS skeleton \x1dE + fenced_code [mask=1])`."""
    u_clean = sanitize_raw_text(user_prompt).strip()
    c_clean = sanitize_raw_text(fenced_code).strip()
    u_frame = TURN_USER_PREFIX + u_clean.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX

    if is_latent:
        a_frame = c_clean.encode("utf-8", errors="strict") + TURN_EOT_SUFFIX
        src_id = SOURCE_ID_REASONING_LATENT
    else:
        sk_clean = _truncate_utf8(sanitize_raw_text(skeleton).strip(), MAX_CODE_SCRATCH_BYTES)
        sk_b = sk_clean.encode("utf-8", errors="strict")
        assert len(sk_b) < 240
        a_frame = SCRATCH_START + sk_b + SCRATCH_END + c_clean.encode("utf-8", errors="strict") + TURN_EOT_SUFFIX
        src_id = SOURCE_ID_REASONING_EXPLICIT

    raw_bytes = u_frame + a_frame
    mask_bytes = (b"\x00" * len(u_frame)) + (b"\x01" * len(a_frame))
    assert verify_mask_alignment(raw_bytes, mask_bytes), "Code alchemy loss_mask alignment violation!"
    raw_bytes.decode("utf-8", errors="strict")
    return raw_bytes, mask_bytes, int(lang_id), int(src_id), int(style_id)


# =============================================================================
# Procedural Anti-Pattern -> Best-Practice & Skeleton-First Generators
# =============================================================================


def _gen_procedural_code_sample(rng: random.Random, idx: int) -> tuple[str, str, str, int, str, bool]:
    """Return `(prompt, skeleton, raw_code, lang_id, fence_tag, allow_latent)`."""
    kind = idx % 10
    suf = f"_{idx}" if idx >= 20 else ""
    is_zh = (idx % 2) == 1

    if kind == 0:
        # Python Anti-Pattern 1: Mutable default list + O(n^2) membership check
        fn = f"collect_active_ids{suf}"
        bad = (
            f"def {fn}(events, blocked=[], acc=[]):\n"
            f"    for ev in events:\n"
            f"        if ev['id'] not in blocked and ev.get('ok') == True:\n"
            f"            acc.append(ev['id'])\n"
            f"    return acc"
        )
        good = (
            f"def {fn}(\n"
            f"    events: list[dict[str, object]],\n"
            f"    blocked: list[str] | None = None,\n"
            f"    acc: list[str] | None = None,\n"
            f") -> list[str]:\n"
            f"    blocked_set = set(blocked) if blocked else set()\n"
            f"    out = list(acc) if acc is not None else []\n"
            f"    for ev in events:\n"
            f"        ev_id = ev.get('id')\n"
            f"        if isinstance(ev_id, str) and ev.get('ok') is True and ev_id not in blocked_set:\n"
            f"            out.append(ev_id)\n"
            f"    return out"
        )
        prompt = (
            f"请审查并重构以下存在坏味道的 Python 函数，消除跨调用状态污染与性能隐患：\n```python\n{bad}\n```"
            if is_zh
            else f"Refactor this flawed Python function to eliminate state bugs and quadratic lookup:\n```python\n{bad}\n```"
        )
        skel = "诊断:可变默认参数跨调用串状态 + list线性查找O(n*m) | bp:None守卫新列表 + set预索引O(1) + is True严格判等"
        return prompt, skel, good, 3, "python", False

    if kind == 1:
        # Python Anti-Pattern 2: Bare except + unchecked division + quadratic string concat
        fn = f"format_ratio_report{suf}"
        bad = (
            f"def {fn}(rows):\n"
            f"    s = ''\n"
            f"    for r in rows:\n"
            f"        try:\n"
            f"            pct = round(r['passed'] / r['total'] * 100, 1)\n"
            f"            s = s + r['name'] + ':' + str(pct) + '%\\n'\n"
            f"        except:\n"
            f"            pass\n"
            f"    return s"
        )
        good = (
            f"def {fn}(rows: list[dict[str, int | str]]) -> str:\n"
            f"    lines: list[str] = []\n"
            f"    for row in rows:\n"
            f"        total = row.get('total')\n"
            f"        passed = row.get('passed')\n"
            f"        name = row.get('name')\n"
            f"        if not isinstance(name, str) or not isinstance(total, int) or not isinstance(passed, int):\n"
            f"            continue\n"
            f"        if total <= 0:\n"
            f"            continue\n"
            f"        pct = (passed * 100.0) / total\n"
            f"        lines.append(f'{{name}}:{{pct:.1f}}%')\n"
            f"    return '\\n'.join(lines)"
        )
        prompt = (
            f"重构以下包含裸 `except:` 与循环字符串拼接坏味道的 Python 代码：\n```python\n{bad}\n```"
            if is_zh
            else f"Refactor this Python snippet to remove bare `except:`, division traps, and `O(n^2)` string concatenation:\n```python\n{bad}\n```"
        )
        skel = "诊断:裸except吞KeyboardInterrupt + 除零风险 + 循环+=字符串O(n^2) | bp:显式total<=0守卫 + list收集join O(n)"
        return prompt, skel, good, 3, "python", False

    if kind == 2:
        # Python Algorithm Skeleton-First: Overflow-safe binary search lower_bound + boundary guards
        fn = f"lower_bound_threshold{suf}"
        target_mul = rng.choice((1, 2, 5))
        good = (
            f"def {fn}(sorted_vals: list[int], target: int) -> int:\n"
            f"    if not sorted_vals:\n"
            f"        return -1\n"
            f"    need = target * {target_mul}\n"
            f"    lo, hi = 0, len(sorted_vals)\n"
            f"    while lo < hi:\n"
            f"        mid = lo + (hi - lo) // 2\n"
            f"        if sorted_vals[mid] < need:\n"
            f"            lo = mid + 1\n"
            f"        else:\n"
            f"            hi = mid\n"
            f"    return lo if lo < len(sorted_vals) else -1"
        )
        prompt = (
            f"实现 Python 函数 `{fn}(sorted_vals: list[int], target: int) -> int`，在升序列表中查找首个 `>= target * {target_mul}` 的下标，不存在或为空返回 `-1`。"
            if is_zh
            else f"Implement `{fn}(sorted_vals: list[int], target: int) -> int` returning the first index `>= target * {target_mul}` in a sorted list, or `-1`."
        )
        skel = f"guard:empty->-1 | bp:半开区间[0,n)不变量, O(log n)/O(1) | flow:need=target*{target_mul}->while lo<hi mid收缩->越界返-1"
        return prompt, skel, good, 3, "python", False

    if kind == 3:
        # Python Simple Helper (eligible for Curriculum iCoT latent pondering)
        fn = f"clamp_int{suf}"
        lo_v, hi_v = rng.randint(0, 10), rng.randint(50, 200)
        good = (
            f"def {fn}(value: int, lo: int = {lo_v}, hi: int = {hi_v}) -> int:\n"
            f"    return min(hi, max(lo, value))"
        )
        prompt = f"Write a concise Python helper `{fn}(value: int, lo: int = {lo_v}, hi: int = {hi_v}) -> int` clamping `value` into `[lo, hi]`."
        skel = "guard:none | bp:O(1) pure | flow:min(hi,max(lo,value))"
        return prompt, skel, good, 3, "python", True

    if kind == 4:
        # JS Anti-Pattern 1: Caller array mutation (.sort) + `==` loose coercion + `var`
        fn = f"topValidScores{suf}"
        k_lim = rng.choice((3, 5, 10))
        bad = (
            f"function {fn}(scores) {{\n"
            f"  var cleaned = [];\n"
            f"  for (var i = 0; i < scores.length; i++) {{\n"
            f"    if (scores[i] != null && scores[i] != '') cleaned.push(scores[i]);\n"
            f"  }}\n"
            f"  scores.sort((a, b) => b - a);\n"
            f"  return scores.slice(0, {k_lim});\n"
            f"}}"
        )
        good = (
            f"function {fn}(scores) {{\n"
            f"  if (!Array.isArray(scores)) {{\n"
            f"    return [];\n"
            f"  }}\n"
            f"  const valid = scores.filter((v) => typeof v === 'number' && Number.isFinite(v));\n"
            f"  return [...valid].sort((a, b) => b - a).slice(0, {k_lim});\n"
            f"}}"
        )
        prompt = (
            f"审查并重构以下存在原地修改入参与隐式类型转换漏洞的 JavaScript 函数：\n```javascript\n{bad}\n```"
            if is_zh
            else f"Refactor this JavaScript function to fix input mutation, loose coercion, and NaN bugs:\n```javascript\n{bad}\n```"
        )
        skel = f"诊断:scores.sort原地污染入参 + !=隐式转换漏过NaN + 返回未过滤数组 | bp:Array.isArray守卫 + Number.isFinite + 浅拷贝排序取前{k_lim}"
        return prompt, skel, good, 5, "javascript", False

    if kind == 5:
        # JS Anti-Pattern 2: O(n^2) Array.includes join -> O(n) Map join
        fn = f"attachUserNames{suf}"
        bad = (
            f"function {fn}(orders, users) {{\n"
            f"  const out = [];\n"
            f"  for (let i = 0; i < orders.length; i++) {{\n"
            f"    const u = users.find((x) => x.id == orders[i].userId);\n"
            f"    out.push({{ ...orders[i], userName: u ? u.name : 'unknown' }});\n"
            f"  }}\n"
            f"  return out;\n"
            f"}}"
        )
        good = (
            f"function {fn}(orders, users) {{\n"
            f"  if (!Array.isArray(orders)) {{\n"
            f"    return [];\n"
            f"  }}\n"
            f"  const nameById = new Map();\n"
            f"  if (Array.isArray(users)) {{\n"
            f"    for (const user of users) {{\n"
            f"      if (user && user.id !== undefined) {{\n"
            f"        nameById.set(user.id, String(user.name ?? 'unknown'));\n"
            f"      }}\n"
            f"    }}\n"
            f"  }}\n"
            f"  return orders.map((ord) => ({{\n"
            f"    ...ord,\n"
            f"    userName: nameById.get(ord.userId) ?? 'unknown',\n"
            f"  }}));\n"
            f"}}"
        )
        prompt = (
            f"将下面 `O(N*M)` 嵌套 `users.find` 的 JavaScript 关联函数重构为生产级 `O(N+M)` 实现：\n```javascript\n{bad}\n```"
            if is_zh
            else f"Refactor this `O(N*M)` JavaScript join function into an idiomatic `O(N+M)` implementation:\n```javascript\n{bad}\n```"
        )
        skel = "诊断:循环内users.find导致O(n*m)退化 + ==松散匹配 | bp:Map预建哈希索引O(1)查找 + ??空值合并"
        return prompt, skel, good, 5, "javascript", False

    if kind == 6:
        # JS Simple Helper (eligible for Curriculum iCoT latent pondering)
        fn = f"sumSafeNumbers{suf}"
        good = (
            f"function {fn}(items) {{\n"
            f"  if (!Array.isArray(items)) return 0;\n"
            f"  return items.reduce((acc, x) => (Number.isFinite(x) ? acc + x : acc), 0);\n"
            f"}}"
        )
        prompt = f"Write an ES6 helper `{fn}(items)` summing finite numbers in an array safely."
        skel = "guard:!isArray->0 | bp:Number.isFinite | flow:reduce"
        return prompt, skel, good, 5, "javascript", True

    if kind == 7:
        # SQL Anti-Pattern -> Best Practice: Non-sargable YEAR() + SELECT * + division by zero
        yr = rng.choice((2024, 2025, 2026))
        min_cnt = rng.choice((5, 10, 20))
        tbl_p = f"payment_events{suf}"
        bad_sql = (
            f"SELECT * FROM {tbl_p}\n"
            f"WHERE YEAR(created_at) = {yr}\n"
            f"GROUP BY merchant_id\n"
            f"HAVING SUM(refunded) / COUNT(*) > 0.1;"
        )
        good_sql = (
            f"SELECT\n"
            f"    merchant_id,\n"
            f"    COUNT(*) AS total_tx,\n"
            f"    COALESCE(SUM(refunded) * 1.0 / NULLIF(COUNT(*), 0), 0.0) AS refund_rate\n"
            f"FROM {tbl_p}\n"
            f"WHERE created_at >= '{yr}-01-01' AND created_at < '{yr + 1}-01-01'\n"
            f"GROUP BY merchant_id\n"
            f"HAVING COUNT(*) >= {min_cnt}\n"
            f"   AND (SUM(refunded) * 1.0 / NULLIF(COUNT(*), 0)) > 0.10\n"
            f"ORDER BY refund_rate DESC;"
        )
        prompt = (
            f"优化以下违反 SQL 索引与聚合规范的慢查询（要求命中 `created_at` 索引、消除整除截断与 SELECT *，并要求样本数 >= {min_cnt}）：\n```sql\n{bad_sql}\n```"
            if is_zh
            else f"Refactor this anti-pattern SQL query to be index-sargable on `created_at`, safe against integer division/zero, and require `COUNT(*) >= {min_cnt}`:\n```sql\n{bad_sql}\n```"
        )
        skel = f"诊断:YEAR(created_at)破坏B-Tree索引 + SELECT*未对齐GROUP BY + 整数相除截断 | bp:半开时间区间 + 显式列 + NULLIF防除零 + 样本门槛>={min_cnt}"
        return prompt, skel, good_sql, 4, "sql", False

    if kind == 8:
        # Python Sliding Window Monotonic Deque / Two-Pointer Algorithm
        fn = f"max_subarray_within_budget{suf}"
        good = (
            f"def {fn}(costs: list[int], budget: int) -> int:\n"
            f"    if budget < 0 or not costs:\n"
            f"        return 0\n"
            f"    left = 0\n"
            f"    window_sum = 0\n"
            f"    best = 0\n"
            f"    for right, c in enumerate(costs):\n"
            f"        window_sum += c\n"
            f"        while left <= right and window_sum > budget:\n"
            f"            window_sum -= costs[left]\n"
            f"            left += 1\n"
            f"        best = max(best, right - left + 1)\n"
            f"    return best"
        )
        prompt = (
            f"实现 Python 函数 `{fn}(costs: list[int], budget: int) -> int`，假设 `costs` 非负，以 `O(n)` 时间求总和 `<= budget` 的最长连续子数组长度。"
            if is_zh
            else f"Implement `{fn}(costs: list[int], budget: int) -> int` finding the maximum contiguous subarray length with sum `<= budget` (non-negative `costs`) in `O(n)` time."
        )
        skel = "guard:budget<0 or empty->0 | bp:双指针滑动窗口单调性 O(n)/O(1) | flow:右扩累加->while超预算左缩->更新最大跨度"
        return prompt, skel, good, 3, "python", False

    # kind == 9: JS Retry / Rate-Window Guard Class
    cls_n = f"TokenBucketLimiter{idx}"
    good = (
        f"export class {cls_n} {{\n"
        f"  constructor(capacity, refillPerSec) {{\n"
        f"    if (capacity <= 0 || refillPerSec <= 0) {{\n"
        f"      throw new RangeError('capacity and refillPerSec must be positive');\n"
        f"    }}\n"
        f"    this.capacity = capacity;\n"
        f"    this.refillPerSec = refillPerSec;\n"
        f"    this.tokens = capacity;\n"
        f"    this.lastSec = 0;\n"
        f"  }}\n\n"
        f"  tryAcquire(cost, nowSec) {{\n"
        f"    if (cost <= 0 || cost > this.capacity) return false;\n"
        f"    const elapsed = Math.max(0, nowSec - this.lastSec);\n"
        f"    this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.refillPerSec);\n"
        f"    this.lastSec = Math.max(this.lastSec, nowSec);\n"
        f"    if (this.tokens < cost) return false;\n"
        f"    this.tokens -= cost;\n"
        f"    return true;\n"
        f"  }}\n"
        f"}}"
    )
    prompt = (
        f"用现代 ES6 实现令牌桶限流器 `class {cls_n}`，包含正数参数校验、时钟回拨保护与 `tryAcquire(cost, nowSec)`。"
        if is_zh
        else f"Implement an idiomatic ES6 `class {cls_n}` token-bucket rate limiter with clock-skew protection and `tryAcquire(cost, nowSec)`."
    )
    skel = "guard:cap/rate<=0抛RangeError; cost越界->false | bp:Math.max防时钟倒退 | flow:按elapsed补令牌封顶->扣减cost返回布尔"
    return prompt, skel, good, 5, "javascript", False


def _stream_hf_code_alchemy(max_items: int, validator: CodeASTValidator) -> Iterator[tuple[str, str, str, int, str]]:
    """Stream `nvidia/OpenCodeReasoning-2` and `bigcode/commitpackft` (Python/JS/SQL only, zero Rust)."""
    try:
        from datasets import load_dataset
    except Exception:
        return

    got = 0
    ocr_cap = max(1, int(max_items * 0.55))
    try:
        ds_ocr = load_dataset("nvidia/OpenCodeReasoning-2", split="train", streaming=True)
        for row in ds_ocr:
            if got >= ocr_cap:
                break
            q = str(row.get("input") or row.get("question") or row.get("prompt") or "").strip()
            out = str(row.get("output") or row.get("solution") or row.get("response") or "").strip()
            if not q or not out or len(q.encode("utf-8", errors="ignore")) > 1200:
                continue
            m_py = _PY_FENCE_RE.search(out)
            m_js = _JS_FENCE_RE.search(out)
            if m_py:
                code = m_py.group(1).strip()
                lid, ftag = 3, "python"
            elif m_js:
                code = m_js.group(1).strip()
                lid, ftag = 5, "javascript"
            else:
                continue
            code_b = code.encode("utf-8", errors="ignore")
            if not (64 <= len(code_b) <= 2400):
                continue
            ok, _ = validator.validate(ftag, code_b)
            if not ok:
                continue
            skel = "guard:validate_bounds | bp:idiomatic_clean_ast | flow:parse_constraints->step_transform->return"
            yield q, skel, code, lid, ftag
            got += 1
    except Exception:
        pass

    for subset, lid, ftag in (("python", 3, "python"), ("javascript", 5, "javascript")):
        if got >= max_items:
            break
        try:
            ds_cp = load_dataset("bigcode/commitpackft", subset, split="train", streaming=True)
            for row in ds_cp:
                if got >= max_items:
                    break
                old_c = str(row.get("old_contents") or "").strip()
                new_c = str(row.get("new_contents") or "").strip()
                msg = sanitize_raw_text(str(row.get("message") or row.get("subject") or "fix bug")).strip()[:140]
                if not old_c or not new_c or old_c == new_c:
                    continue
                if len(old_c.encode("utf-8", errors="ignore")) > 1100 or len(new_c.encode("utf-8", errors="ignore")) > 1400:
                    continue
                ok_new, _ = validator.validate(ftag, new_c.encode("utf-8", errors="ignore"))
                if not ok_new:
                    continue
                prompt = f"Diagnose and fix the defect ({msg}) in this {ftag} code:\n```{ftag}\n{old_c}\n```"
                skel = f"诊断:{msg} | bp:preserve_contract_and_edge_guards"
                yield prompt, skel, new_c, lid, ftag
                got += 1
        except Exception:
            continue


def build_cap_code_corpus(
    out_dir: str | Path = "artifacts/cap_data",
    target_mb: float = 80.0,
    smoke_quota: bool = False,
    offline: bool = False,
    seed: int = 202615,
) -> dict:
    """Build `cap_code_{train,val}.{bin,idx,loss_mask.bin}` (zero Rust, 100% AST-validated Python/JS/SQL)."""
    t0 = time.perf_counter()
    out_path = Path(out_dir)
    assert_not_drive_mmap(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    validator = CodeASTValidator()
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
    docs: list[tuple[bytes, bytes, int, int, int, int]] = []
    acc_bytes = 0
    domain_counts: dict[int, int] = {3: 0, 4: 0, 5: 0}

    def _try_add(prompt: str, skel: str, code: str, lid: int, ftag: str, allow_latent: bool) -> bool:
        nonlocal acc_bytes
        if acc_bytes >= total_target or lid not in (3, 4, 5):
            return False
        code_b = code.encode("utf-8", errors="ignore")
        depth = 0
        if lid in (3, 5):
            ok, depth = validator.validate("python" if lid == 3 else "javascript", code_b)
            if not ok:
                return False
        is_latent = bool(allow_latent and rng.random() < 0.80)
        fenced = f"```{ftag}\n{code.strip()}\n```"
        try:
            raw_b, msk_b, l_i, src_i, sty_i = encode_code_alchemy_sample(
                prompt, skel, fenced, lid, is_latent=is_latent
            )
        except Exception:
            return False
        if len(raw_b) < 64 or len(raw_b) > 4096:
            return False
        h = hashlib.blake2b(raw_b, digest_size=12).digest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        docs.append((raw_b, msk_b, l_i, src_i, min(depth, 16), sty_i))
        domain_counts[l_i] = domain_counts.get(l_i, 0) + 1
        acc_bytes += len(raw_b)
        return True

    if not offline:
        hf_cap = int(total_target * 0.55)
        est_items = max(300, int(hf_cap / 700))
        for prompt, skel, code, lid, ftag in _stream_hf_code_alchemy(est_items, validator):
            _try_add(prompt, skel, code, lid, ftag, allow_latent=False)
            if acc_bytes >= hf_cap:
                break

    idx = 0
    while acc_bytes < total_target:
        prompt, skel, code, lid, ftag, allow_lat = _gen_procedural_code_sample(rng, idx)
        _try_add(prompt, skel, code, lid, ftag, allow_latent=allow_lat)
        idx += 1

    rng.shuffle(docs)
    n_val = max(12, int(len(docs) * (val_target / max(float(total_target), 1.0))))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    def _write_split(split: str, split_docs: list[tuple[bytes, bytes, int, int, int, int]]) -> dict:
        bin_p = out_path / f"cap_code_{split}.bin"
        idx_p = out_path / f"cap_code_{split}.idx"
        msk_p = out_path / f"cap_code_{split}.loss_mask.bin"
        off = 0
        with bin_p.open("wb") as fb, idx_p.open("wb") as fi, msk_p.open("wb") as fm:
            for raw_b, msk_b, lid, src_i, depth, sty_i in split_docs:
                n = len(raw_b)
                fb.write(raw_b)
                fm.write(msk_b)
                fi.write(struct.pack(IDX_RECORD_FMT, off, n, lid, src_i, depth, sty_i))
                off += n
        return {"split": split, "num_docs": len(split_docs), "total_bytes": off}

    train_meta = _write_split("train", train_docs)
    val_meta = _write_split("val", val_docs)
    summary = {
        "dataset": "cap_code",
        "smoke_quota": smoke_quota,
        "offline": offline,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "total_bytes": train_meta["total_bytes"] + val_meta["total_bytes"],
        "total_docs": train_meta["num_docs"] + val_meta["num_docs"],
        "docs_by_lang_id": domain_counts,
        "train": train_meta,
        "val": val_meta,
    }
    (out_path / "cap_code_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DCPA Stage-3/5 Code Best-Practice & Refactoring Shards")
    ap.add_argument("--out-dir", type=str, default="artifacts/cap_data")
    ap.add_argument("--target-mb", type=float, default=80.0)
    ap.add_argument("--smoke-quota", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=202615)
    args = ap.parse_args()
    res = build_cap_code_corpus(
        out_dir=args.out_dir,
        target_mb=args.target_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
