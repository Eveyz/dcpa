"""Compact Symbolic & Latent Implicit Reasoning Pure-Byte Dataset Builder (Capability Spec 4.1).

Builds ~100 MB full quota (or 2 MB under `--smoke-quota` locally on M3 Pro) of raw UTF-8 byte
reasoning shards (`cap_reasoning_{train,val}.{bin,idx,loss_mask.bin}`):
  - Streams `nvidia/OpenMathInstruct-2` (auto-fallback `meta-math/MetaMathQA`, `openai/gsm8k`),
    `allenai/ai2_arc` (`ARC-Challenge`), `cruxeval-org/cruxeval`, plus a deterministic bilingual
    (EN/ZH) 4..9-variable first-order logic / algebra / inequality transitivity state machine.
  - Red Line #1: Strips all ASCII < 32 control bytes except `\\t` and `\\n` before framing.
  - Compresses intermediate derivation into `< 240 B` high-density symbolic chains wrapped in
    `\\x1dS\\n...\\x1dE\\n` (stripping verbose CoT fluff like "Let's think step by step", "Wait,").
  - 60% Explicit Scratchpad (`source_id=10`, `ponder_tax_scale=1.0`) vs.
    40% Latent Implicit Reasoning (`source_id=11`, `ponder_tax_scale=0.0`, draft stripped to push
    BCG-PC `L_max=8` multi-step latent refinement).
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

# Verbose CoT filler patterns stripped before symbolic compression
_FLUFF_RE = re.compile(
    r"(?i)\b("
    r"let'?s think step by step|let me think|wait a second|wait,|hmm,|well,|"
    r"to solve this problem|first,?\s+let'?s analyze|step\s*\d+\s*:|"
    r"therefore,?\s+the final answer is|so the answer is"
    r")\b|首先我们来分析|让我们一步一步来|让我们仔细思考|由此可得最终答案是|综上所述"
)
_EQ_RE = re.compile(r"[A-Za-z0-9_()+\-*/=<>.,%^\[\] ]{3,80}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

MAX_SCRATCH_BYTES = 236  # strictly < 240 UTF-8 bytes


def _truncate_utf8(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return s
    b = b[:max_bytes]
    while b and (b[-1] & 0xC0) == 0x80:
        b = b[:-1]
    return b.decode("utf-8", errors="ignore").rstrip(" ;,->")


def compress_to_symbolic_chain(raw_cot: str, answer: str, max_bytes: int = MAX_SCRATCH_BYTES) -> str:
    """Distill verbose derivation text into a `< 240 B` compact symbolic step chain."""
    cleaned = sanitize_raw_text(raw_cot)
    cleaned = _FLUFF_RE.sub("", cleaned)
    # Remove LaTeX wrapper noise (`\\boxed{}`, `$`, `\\times`, `\\frac`)
    cleaned = cleaned.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    cleaned = re.sub(r"\\boxed\{([^}]+)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", cleaned)
    cleaned = cleaned.replace("$", "").replace("\\(", "").replace("\\)", "")

    steps: list[str] = []
    for line in re.split(r"[\n。.;]+", cleaned):
        ln = re.sub(r"\s+", " ", line).strip(" -*>:")
        if not ln:
            continue
        if "=" in ln or any(op in ln for op in ("+", "-", "*", "/", ">", "<", "->")):
            # Extract compact equation clause
            m = _EQ_RE.search(ln)
            clause = m.group(0).strip() if m else ln[:48]
            clause = re.sub(r"\s+", " ", clause).strip()
            if len(clause) >= 3 and clause not in steps:
                steps.append(clause)
        if len(steps) >= 5:
            break

    if not steps:
        nums = _NUM_RE.findall(cleaned)[:4]
        if nums:
            steps = [f"v{i}={n}" for i, n in enumerate(nums)]
        else:
            steps = [f"ans={sanitize_raw_text(answer).strip()[:40]}"]

    chain = "; ".join(steps)
    return _truncate_utf8(chain, max_bytes)


def _gen_state_machine_sample(rng: random.Random, idx: int) -> tuple[str, str, str, int, int]:
    """Deterministic 4..9-variable first-order logic / algebra / inequality state machine (EN/ZH).

    Returns `(question, symbolic_chain, answer, lang_id, style_id)`.
    """
    n_vars = rng.randint(4, 9)
    kind = idx % 6
    is_zh = (idx % 3) == 1
    lang_id = 1 if is_zh else 0

    if kind == 0:
        # Multi-step affine / modular integer assignment chain
        vals = [rng.randint(2, 35)]
        exprs_en = [f"v0 = {vals[0]}"]
        exprs_zh = [f"令 v0 = {vals[0]}"]
        chain = [f"v0={vals[0]}"]
        for k in range(1, n_vars):
            op = rng.choice(["+", "-", "*"])
            prev = rng.randint(max(0, k - 2), k - 1)
            c = rng.randint(2, 6) if op == "*" else rng.randint(1, 25)
            if op == "+":
                v = vals[prev] + c
            elif op == "-":
                v = vals[prev] - c
            else:
                v = vals[prev] * c
                if abs(v) > 5000:
                    op = "+"
                    v = vals[prev] + c
            vals.append(v)
            exprs_en.append(f"v{k} = v{prev} {op} {c}")
            exprs_zh.append(f"v{k} = v{prev} {op} {c}")
            chain.append(f"v{k}=v{prev}{op}{c}={v}")
        thresh = vals[-1] + rng.choice([-7, -3, 0, 5, 11])
        cond = vals[-1] > thresh
        chain.append(f"cond=(v{n_vars-1}>{thresh})={cond}")
        if is_zh:
            q = f"已知状态转移：{'；'.join(exprs_zh)}。求 v{n_vars-1} 的精确值，并判断 v{n_vars-1} > {thresh} 是否成立。"
            ans = f"v{n_vars-1} = {vals[-1]}，条件 v{n_vars-1} > {thresh} 为 {'真 (True)' if cond else '假 (False)'}。"
        else:
            q = f"Given assignments: {'; '.join(exprs_en)}. Compute v{n_vars-1} and evaluate whether v{n_vars-1} > {thresh}."
            ans = f"v{n_vars-1} = {vals[-1]}; condition (v{n_vars-1} > {thresh}) is {cond}."
        return q, _truncate_utf8("; ".join(chain), MAX_SCRATCH_BYTES), ans, lang_id, 0

    if kind == 1:
        # Strict inequality transitivity & rank ordering over 4..9 entities
        names = [f"X{i}" for i in range(n_vars)]
        order = list(names)
        rng.shuffle(order)  # order[0] > order[1] > ... > order[n-1]
        rels = [f"{order[i]} > {order[i+1]}" for i in range(n_vars - 1)]
        rng.shuffle(rels)
        a_pos, b_pos = sorted(rng.sample(range(n_vars), 2))
        ask_rev = rng.random() < 0.5
        lhs, rhs = (order[b_pos], order[a_pos]) if ask_rev else (order[a_pos], order[b_pos])
        truth = not ask_rev
        chain_str = " > ".join(order) + f"; check({lhs}>{rhs})={truth}"
        if is_zh:
            q = f"已知偏序关系：{'，'.join(rels)}。最高者是谁？且 {lhs} > {rhs} 是否必然成立？"
            ans = f"最高者为 {order[0]}，{lhs} > {rhs} 必然为{'真' if truth else '假'}。"
        else:
            q = f"Given strict order constraints: {', '.join(rels)}. Identify the maximum element and determine if {lhs} > {rhs}."
            ans = f"Maximum is {order[0]}; {lhs} > {rhs} is {truth}."
        return q, _truncate_utf8(chain_str, MAX_SCRATCH_BYTES), ans, lang_id, 0

    if kind == 2:
        # First-order Horn-clause implication propagation (4..9 boolean atoms)
        atoms = [f"P{i}" for i in range(n_vars)]
        true_set = {atoms[0]}
        rules: list[str] = []
        chain = [f"{atoms[0]}=1"]
        for k in range(1, n_vars):
            if rng.random() < 0.72:
                src = rng.choice(list(true_set))
                rules.append(f"{src} -> {atoms[k]}")
                true_set.add(atoms[k])
                chain.append(f"{src}->{atoms[k]}=1")
            else:
                # Guarded rule whose antecedent is false
                false_pool = [a for a in atoms[:k] if a not in true_set]
                if false_pool:
                    src = rng.choice(false_pool)
                    rules.append(f"{src} -> {atoms[k]}")
                    chain.append(f"{src}=0=>{atoms[k]}=0")
                else:
                    rules.append(f"{atoms[k-1]} & ~{atoms[0]} -> {atoms[k]}")
                    chain.append(f"~{atoms[0]}=0=>{atoms[k]}=0")
        target = atoms[-1]
        derived = target in true_set
        rng.shuffle(rules)
        if is_zh:
            q = f"已知事实 {atoms[0]}=真，蕴含规则：{'；'.join(rules)}。在最小模型下 {target} 是否可推导为真？"
            ans = f"{target} 可推导为真 (True)。" if derived else f"{target} 不可推导为真 (False)。"
        else:
            q = f"Given fact {atoms[0]}=True and Horn rules: {'; '.join(rules)}. Is {target} entailed in the minimal model?"
            ans = f"{target} is {'True (entailed)' if derived else 'False (not entailed)'}."
        return q, _truncate_utf8("; ".join(chain), MAX_SCRATCH_BYTES), ans, lang_id, 0

    if kind == 3:
        # Python function execution / inverse-input trace deduction (cruxeval style)
        a = rng.randint(2, 9)
        b = rng.randint(1, 15)
        xs = [rng.randint(1, 12) for _ in range(rng.randint(3, 5))]
        mode = rng.choice(["sum_even", "acc_mul", "filter_shift"])
        if mode == "sum_even":
            fn_src = f"def f(xs):\n    return sum(x * {a} + {b} for x in xs if x % 2 == 0)"
            evens = [x for x in xs if x % 2 == 0]
            terms = [x * a + b for x in evens]
            res = sum(terms)
            chain = f"evens={evens}; terms={terms}; sum={res}"
        elif mode == "acc_mul":
            fn_src = f"def f(xs):\n    acc = {b}\n    for x in xs:\n        acc = (acc + x) % {a * 7 + 3}\n    return acc"
            mod = a * 7 + 3
            acc = b
            st = [f"a0={acc}"]
            for i, x in enumerate(xs, 1):
                acc = (acc + x) % mod
                st.append(f"a{i}={acc}")
            res = acc
            chain = "; ".join(st)
        else:
            fn_src = f"def f(xs):\n    return [x + {b} for x in xs if x >= {a}]"
            res = [x + b for x in xs if x >= a]
            chain = f"xs={xs}; ge_{a}={[x for x in xs if x >= a]}; out={res}"
        q = (
            f"请推导调用 Python 函数 `f({xs})` 的精确返回值：\n{fn_src}"
            if is_zh
            else f"Trace the exact Python return value of `f({xs})`:\n{fn_src}"
        )
        ans = repr(res)
        return q, _truncate_utf8(chain, MAX_SCRATCH_BYTES), ans, 3, 4

    if kind == 4:
        # JavaScript ES6 filter/map/reduce & closure trace deduction (lang_id=5)
        a = rng.randint(2, 8)
        b = rng.randint(1, 12)
        xs = [rng.randint(1, 11) for _ in range(rng.randint(3, 5))]
        js_arg = json.dumps(xs)
        if rng.random() < 0.5:
            fn_src = f"function f(xs) {{\n  return xs.filter((x) => x % 2 === 1).map((x) => x * {a} - {b});\n}}"
            odds = [x for x in xs if x % 2 == 1]
            out_js = [x * a - b for x in odds]
            chain = f"odds={odds}; map=*{a}-{b}; out={out_js}"
            ans = json.dumps(out_js)
        else:
            fn_src = f"function f(xs) {{\n  return xs.reduce((acc, x) => (x > {a} ? acc + x * 2 : acc + {b}), 0);\n}}"
            acc = 0
            steps_js: list[str] = []
            for i, x in enumerate(xs, 1):
                acc = acc + x * 2 if x > a else acc + b
                steps_js.append(f"s{i}={acc}")
            chain = "; ".join(steps_js)
            ans = str(acc)
        q = (
            f"推导下面 JavaScript 函数执行 `f({js_arg})` 的精确返回值：\n{fn_src}"
            if is_zh
            else f"Predict the exact JavaScript return value of `f({js_arg})`:\n{fn_src}"
        )
        return q, _truncate_utf8(chain, MAX_SCRATCH_BYTES), ans, 5, 4

    # kind == 5: In-scratchpad constraint satisfaction with hypothesis backtracking (`try -> contra; fix`)
    x_true = rng.randint(3, 9)
    y_true = rng.randint(2, 8)
    while y_true == x_true:
        y_true = rng.randint(2, 8)
    s_xy = x_true + y_true
    p_xy = x_true * y_true
    wrong_x = max(1, s_xy - 1) if (s_xy - 1) != x_true else max(1, x_true - 1)
    wrong_y = s_xy - wrong_x
    chain = (
        f"x+y={s_xy}, x*y={p_xy}; try ({wrong_x},{wrong_y}) -> prod={wrong_x*wrong_y}!={p_xy} contra; "
        f"fix ({max(x_true,y_true)},{min(x_true,y_true)}) -> prod={p_xy} ok"
    )
    hi_v, lo_v = max(x_true, y_true), min(x_true, y_true)
    if is_zh:
        q = f"已知两个正整数满足 x >= y、x + y = {s_xy} 且 x * y = {p_xy}，求 (x, y)。"
        ans = f"({hi_v}, {lo_v})"
    else:
        q = f"Two positive integers satisfy x >= y, x + y = {s_xy}, and x * y = {p_xy}. Find (x, y)."
        ans = f"({hi_v}, {lo_v})"
    return q, _truncate_utf8(chain, MAX_SCRATCH_BYTES), ans, lang_id, 0


def _stream_hf_reasoning(max_items: int, seed: int) -> Iterator[tuple[str, str, str, int, int]]:
    """Stream external open reasoning datasets cleanly with `streaming=True` and graceful fallbacks."""
    try:
        from datasets import load_dataset
    except Exception:
        return

    yielded = 0

    # 1. Math derivation: OpenMathInstruct-2 -> MetaMathQA -> gsm8k
    math_sources = [
        ("nvidia/OpenMathInstruct-2", None, "train"),
        ("meta-math/MetaMathQA", None, "train"),
        ("openai/gsm8k", "main", "train"),
    ]
    math_budget = max(1, int(max_items * 0.55))
    got_math = 0
    for repo, cfg_name, split in math_sources:
        if got_math >= math_budget:
            break
        try:
            ds = (
                load_dataset(repo, cfg_name, split=split, streaming=True)
                if cfg_name
                else load_dataset(repo, split=split, streaming=True)
            )
            for row in ds:
                if got_math >= math_budget:
                    break
                q = str(row.get("problem") or row.get("query") or row.get("question") or "").strip()
                sol = str(row.get("generated_solution") or row.get("response") or row.get("answer") or "").strip()
                ans = str(row.get("expected_answer") or "").strip()
                if not q or not sol or len(q.encode("utf-8")) > 900:
                    continue
                if not ans:
                    if "####" in sol:
                        sol, ans = sol.split("####", 1)
                    else:
                        nums = _NUM_RE.findall(sol)
                        ans = nums[-1] if nums else sol.splitlines()[-1][:80]
                ans = sanitize_raw_text(ans).strip()
                if not ans:
                    continue
                chain = compress_to_symbolic_chain(sol, ans)
                yield sanitize_raw_text(q), chain, ans, 0, 0
                got_math += 1
                yielded += 1
            if got_math > 0:
                break
        except Exception:
            continue

    # 2. ARC-Challenge scientific logic
    arc_budget = max(1, int(max_items * 0.15))
    got_arc = 0
    try:
        ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)
        for row in ds_arc:
            if got_arc >= arc_budget:
                break
            q_txt = sanitize_raw_text(str(row.get("question", ""))).strip()
            choices = row.get("choices") or {}
            labels = choices.get("label") or []
            texts = choices.get("text") or []
            ak = str(row.get("answerKey", "")).strip()
            if not q_txt or not labels or ak not in labels:
                continue
            opts = " | ".join(f"{l}) {sanitize_raw_text(str(t)).strip()}" for l, t in zip(labels, texts))
            idx_a = labels.index(ak)
            ans_txt = sanitize_raw_text(str(texts[idx_a])).strip()
            q_full = f"{q_txt}\nChoices: {opts}"
            chain = _truncate_utf8(f"eval({len(labels)} opts) -> match={ak}: {ans_txt[:120]}", MAX_SCRATCH_BYTES)
            yield q_full, chain, f"{ak}: {ans_txt}", 0, 0
            got_arc += 1
            yielded += 1
    except Exception:
        pass

    # 3. CRUXEval Python execution deduction
    crux_budget = max(1, int(max_items * 0.15))
    got_crux = 0
    try:
        ds_crux = load_dataset("cruxeval-org/cruxeval", split="test", streaming=True)
        for row in ds_crux:
            if got_crux >= crux_budget:
                break
            code = sanitize_raw_text(str(row.get("code", ""))).strip()
            inp = sanitize_raw_text(str(row.get("input", ""))).strip()
            out = sanitize_raw_text(str(row.get("output", ""))).strip()
            if not code or not inp or not out or len(code.encode("utf-8")) > 900:
                continue
            q_full = f"Predict exact return value for input ({inp}):\n{code}"
            chain = _truncate_utf8(f"args=({inp[:80]}); trace -> ret={out[:100]}", MAX_SCRATCH_BYTES)
            yield q_full, chain, out, 3, 4
            got_crux += 1
            yielded += 1
    except Exception:
        pass


def encode_reasoning_sample(
    question: str,
    symbolic_chain: str,
    answer: str,
    lang_id: int,
    style_id: int,
    is_latent: bool,
) -> tuple[bytes, bytes, int, int, int]:
    """Encode one reasoning sample into raw UTF-8 bytes & exact byte-aligned `loss_mask`.

    Returns `(raw_bytes, mask_bytes, lang_id, source_id, style_id)`.
    """
    q_clean = sanitize_raw_text(question).strip()
    a_clean = sanitize_raw_text(answer).strip()
    u_frame = TURN_USER_PREFIX + q_clean.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX

    if is_latent:
        # 40% Latent implicit reasoning: strip explicit scratchpad body; tag source_id=11 -> ponder_tax_scale=0.0
        a_frame = a_clean.encode("utf-8", errors="strict") + TURN_EOT_SUFFIX
        source_id = SOURCE_ID_REASONING_LATENT
    else:
        c_clean = _truncate_utf8(sanitize_raw_text(symbolic_chain).strip(), MAX_SCRATCH_BYTES)
        c_bytes = c_clean.encode("utf-8", errors="strict")
        assert len(c_bytes) < 240, f"Scratch chain exceeded 240B: {len(c_bytes)}"
        a_frame = SCRATCH_START + c_bytes + SCRATCH_END + a_clean.encode("utf-8", errors="strict") + TURN_EOT_SUFFIX
        source_id = SOURCE_ID_REASONING_EXPLICIT

    raw_bytes = u_frame + a_frame
    mask_bytes = (b"\x00" * len(u_frame)) + (b"\x01" * len(a_frame))
    assert verify_mask_alignment(raw_bytes, mask_bytes), "Reasoning loss_mask alignment violation!"
    raw_bytes.decode("utf-8", errors="strict")
    return raw_bytes, mask_bytes, int(lang_id), int(source_id), int(style_id)


def build_cap_reasoning_corpus(
    out_dir: str | Path = "artifacts/cap_data",
    target_mb: float = 100.0,
    smoke_quota: bool = False,
    offline: bool = False,
    seed: int = 202611,
) -> dict:
    """Build `cap_reasoning_{train,val}.{bin,idx,loss_mask.bin}` pure-byte reasoning corpus."""
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
    explicit_count = 0
    latent_count = 0
    max_scratch_len = 0

    def _try_add(q: str, chain: str, ans: str, lid: int, sid: int) -> bool:
        nonlocal acc_bytes, explicit_count, latent_count, max_scratch_len
        if acc_bytes >= total_target:
            return False
        # Exact 60% explicit / 40% latent implicit ratio
        is_latent = (rng.random() < 0.40)
        try:
            raw_b, msk_b, l_i, src_i, sty_i = encode_reasoning_sample(q, chain, ans, lid, sid, is_latent)
        except Exception:
            return False
        if len(raw_b) < 32 or len(raw_b) > 2048:
            return False
        h = hashlib.blake2b(raw_b, digest_size=12).digest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        if not is_latent:
            s0 = raw_b.find(SCRATCH_START)
            s1 = raw_b.find(SCRATCH_END)
            if s0 >= 0 and s1 > s0:
                max_scratch_len = max(max_scratch_len, s1 - (s0 + len(SCRATCH_START)))
            explicit_count += 1
        else:
            latent_count += 1
        docs.append((raw_b, msk_b, l_i, src_i, sty_i))
        acc_bytes += len(raw_b)
        return True

    # 1. Stream HF reasoning corpora (~65% of target if online)
    if not offline:
        hf_cap = int(total_target * 0.65)
        est_items = max(500, int(hf_cap / 280))
        for q, chain, ans, lid, sid in _stream_hf_reasoning(est_items, seed):
            _try_add(q, chain, ans, lid, sid)
            if acc_bytes >= hf_cap:
                break

    # 2. Fill remaining quota with deterministic 4..9-var logic / algebra / code-trace state machine
    idx = 0
    while acc_bytes < total_target:
        q, chain, ans, lid, sid = _gen_state_machine_sample(rng, idx)
        _try_add(q, chain, ans, lid, sid)
        idx += 1

    rng.shuffle(docs)
    n_val = max(16, int(len(docs) * (val_target / max(float(total_target), 1.0))))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    def _write_split(split: str, split_docs: list[tuple[bytes, bytes, int, int, int]]) -> dict:
        bin_p = out_path / f"cap_reasoning_{split}.bin"
        idx_p = out_path / f"cap_reasoning_{split}.idx"
        msk_p = out_path / f"cap_reasoning_{split}.loss_mask.bin"
        off = 0
        n_exp = 0
        n_lat = 0
        with bin_p.open("wb") as fb, idx_p.open("wb") as fi, msk_p.open("wb") as fm:
            for raw_b, msk_b, lid, src_i, sty_i in split_docs:
                n = len(raw_b)
                fb.write(raw_b)
                fm.write(msk_b)
                fi.write(struct.pack(IDX_RECORD_FMT, off, n, lid, src_i, 0, sty_i))
                off += n
                if src_i == SOURCE_ID_REASONING_LATENT:
                    n_lat += 1
                else:
                    n_exp += 1
        return {
            "split": split,
            "num_docs": len(split_docs),
            "total_bytes": off,
            "explicit_docs": n_exp,
            "latent_docs": n_lat,
            "latent_ratio": round(n_lat / max(len(split_docs), 1), 4),
        }

    train_meta = _write_split("train", train_docs)
    val_meta = _write_split("val", val_docs)
    summary = {
        "dataset": "cap_reasoning",
        "smoke_quota": smoke_quota,
        "offline": offline,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "total_bytes": train_meta["total_bytes"] + val_meta["total_bytes"],
        "total_docs": train_meta["num_docs"] + val_meta["num_docs"],
        "explicit_docs": explicit_count,
        "latent_docs": latent_count,
        "latent_ratio": round(latent_count / max(explicit_count + latent_count, 1), 4),
        "max_scratch_bytes": max_scratch_len,
        "train": train_meta,
        "val": val_meta,
    }
    (out_path / "cap_reasoning_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DCPA Capability Reasoning Pure-Byte Dataset")
    ap.add_argument("--out-dir", type=str, default="artifacts/cap_data")
    ap.add_argument("--target-mb", type=float, default=100.0)
    ap.add_argument("--smoke-quota", action="store_true", help="Build 2 MB local smoke quota")
    ap.add_argument("--offline", action="store_true", help="Zero-network deterministic synthesis")
    ap.add_argument("--seed", type=int, default=202611)
    args = ap.parse_args()
    res = build_cap_reasoning_corpus(
        out_dir=args.out_dir,
        target_mb=args.target_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
