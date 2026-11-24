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
from typing import Iterator, Optional

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
    if source_id in (SID_FOLIO_LOGIC_S2, SID_LOGIQA_RECLOR_S2, SID_FINEMATH_S2, SID_COMPLEX_PROOFS_S3):
        return 0
    if source_id == SID_COSMOPEDIA_S2:
        return 2
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

        ds = load_dataset("HuggingFaceFW/finephrase", split="train", streaming=True)
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

        ds = load_dataset("storytracer/US-PD-Books", split="train", streaming=True)
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

        ds = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True)
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

        ds = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True)
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

        # 1. FOLIO (strictly capped at <= 3 epochs to prevent memorization)
        folio_rows: list[bytes] = []
        try:
            ds_f = load_dataset("tasksource/folio", split="train", streaming=True)
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
            ds_pw = load_dataset("tasksource/proofwriter", split="train", streaming=True)
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

        # 3. LogiQA deductive arguments
        try:
            ds_lq = load_dataset("lucasmccabe/logiqa", split="train", streaming=True)
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

        ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
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

        try:
            ds_edu = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
            for row in ds_edu:
                code = str(row.get("text") or row.get("content") or "").strip()
                if 64 <= len(code.encode("utf-8", errors="ignore")) <= 512:
                    yield code, SID_PY_EDU_SHORT_S1
        except Exception as exc:
            print(f"  [INFO] smollm-corpus/python-edu skipped ({type(exc).__name__}).", flush=True)

        for repo in ("tokyotech-llm/swallow-code-v2", "tokyotech-llm/swallow-code"):
            try:
                ds_sw = load_dataset(repo, split="train", streaming=True)
                for row in ds_sw:
                    code = str(row.get("text") or row.get("code") or row.get("content") or "").strip()
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

        ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
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
