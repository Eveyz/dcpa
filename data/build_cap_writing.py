"""Anti-Cliché Long-Form Writing + Reverse Outline + Style-ID Pure-Byte Builder (Capability Spec 4.2).

Builds ~100 MB full quota (or 2 MB under `--smoke-quota` locally on M3 Pro) of raw UTF-8 writing
shards (`cap_writing_{train,val}.{bin,idx,loss_mask.bin}`):
  - Streams `opencsg/Fineweb-Edu-Chinese-V2.1` (`4_5/*.parquet` / `4-5/*.parquet`, 900..4096 B),
    `HuggingFaceTB/cosmopedia-v2`, and `HuggingFaceTB/smoltalk` (`explore-instruct-rewriting`),
    with a zero-cliché bilingual (EN/ZH) procedural generator fallback across all 3 target styles.
  - Red Line #1: Strips all ASCII < 32 control bytes except `\\t` and `\\n` before framing.
  - Anti-cliché blacklist filter: rejects/strips cliché templates ("总而言之", "综上所述", "值得注意的是",
    "是一把双刃剑", "In conclusion,", "It is worth noting", "delve into", etc.).
  - Reverse outline synthesis: extracts 3..4 paragraph thesis nodes (`< 180 B` total) prepended as
    `\\x1dS\\n1.{node1} -> 2.{node2} -> 3.{node3}\\x1dE\\n` (automatically triggers Slot-0 Thesis Pinning!).
  - Heuristic feature classifier writes `style_id` (`1=memo`, `2=expository`, `3=narrative`) into
    byte 15 of the 16-byte `<QIBBBB` `.idx` record.
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

from data.ragged_dataloader import IDX_RECORD_FMT, assert_not_drive_mmap
from utils.chat_protocol import (
    SCRATCH_END,
    SCRATCH_START,
    SOURCE_ID_WRITING,
    STYLE_ID_TO_NAME,
    TURN_ASST_PREFIX,
    TURN_EOT_SUFFIX,
    TURN_USER_PREFIX,
    sanitize_raw_text,
)

MAX_OUTLINE_BYTES = 176  # strictly < 180 UTF-8 bytes

CLICHE_PATTERNS: tuple[str, ...] = (
    "总而言之",
    "综上所述",
    "值得注意的是",
    "是一把双刃剑",
    "由此可见",
    "归根结底",
    "in conclusion,",
    "it is worth noting",
    "delve into",
    "double-edged sword",
    "in summary,",
    "needless to say",
)
_CLICHE_RE = re.compile("|".join(re.escape(p) for p in CLICHE_PATTERNS), re.IGNORECASE)

_CAUSAL_RE = re.compile(
    r"(?i)\b(because|therefore|consequently|mechanism|causes|results in|implies|drives|analogous)\b"
    r"|因为|因此|导致|机制|本质上|从而|类比|传导|原理"
)
_SENSORY_RE = re.compile(
    r"(?i)\b(whispered|flickered|shadow|rain|footsteps|cold|warmth|glanced|quietly|breeze|lantern|horizon)\b"
    r"|微光|雨点|脚步|呼吸|暮色|潮湿|回声|灯影|沉默|风声|轮廓|窗沿"
)
_QUANT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|ms|s|MB|GB|TB|KB|x|倍|亿|万|\$|USD|HKD|bps)")
_MEMO_RE = re.compile(
    r"(?i)\b(decision|recommendation|action|bottleneck|risk|kpi|latency|throughput|budget|cut|ship)\b"
    r"|结论|决策|瓶颈|风险|执行|成本|交付|指标|优先级|建议"
)


def _truncate_utf8(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return s
    b = b[:max_bytes]
    while b and (b[-1] & 0xC0) == 0x80:
        b = b[:-1]
    return b.decode("utf-8", errors="ignore").rstrip(" ,;:-")


def filter_cliches(text: str) -> str | None:
    """Strip cliché closing paragraphs or reject articles saturated with cliché phrases."""
    clean = sanitize_raw_text(text).strip()
    paras = [p.strip() for p in re.split(r"\n{2,}", clean) if p.strip()]
    if len(paras) < 2:
        # Try single-newline paragraph split
        paras = [p.strip() for p in clean.splitlines() if len(p.strip()) >= 40]
    if len(paras) < 3:
        return None

    kept: list[str] = []
    for i, p in enumerate(paras):
        if _CLICHE_RE.search(p):
            # Drop cliché closing paragraph; reject article if cliché pollutes body paragraphs
            if i == len(paras) - 1 and len(kept) >= 3:
                continue
            return None
        kept.append(p)
    if len(kept) < 3:
        return None
    out = "\n\n".join(kept)
    if _CLICHE_RE.search(out):
        return None
    return out


def classify_writing_style(text: str) -> int:
    """Heuristic classifier assigning `style_id`: 1 (memo), 2 (expository), or 3 (narrative)."""
    sents = [s.strip() for s in re.split(r"[.!?。！？\n]+", text) if len(s.strip()) >= 8]
    avg_sent_bytes = sum(len(s.encode("utf-8")) for s in sents) / max(len(sents), 1)
    n_kb = max(len(text.encode("utf-8")) / 1024.0, 0.25)

    quant_density = len(_QUANT_RE.findall(text)) / n_kb
    memo_density = len(_MEMO_RE.findall(text)) / n_kb
    causal_density = len(_CAUSAL_RE.findall(text)) / n_kb
    sensory_density = len(_SENSORY_RE.findall(text)) / n_kb

    memo_score = quant_density * 1.4 + memo_density * 1.2 + (1.2 if avg_sent_bytes < 78 else 0.0)
    expos_score = causal_density * 1.6 + (0.6 if 70 <= avg_sent_bytes <= 155 else 0.0)
    narr_score = sensory_density * 2.2 + (0.5 if '"' in text or "“" in text else 0.0)

    if narr_score >= memo_score and narr_score >= expos_score and sensory_density >= 1.5:
        return 3  # narrative
    if memo_score >= expos_score and (quant_density >= 1.2 or memo_density >= 1.8):
        return 1  # memo
    return 2  # expository


def extract_reverse_outline(article: str, max_bytes: int = MAX_OUTLINE_BYTES) -> str:
    """Extract 3..4 paragraph thesis nodes (`< 180` UTF-8 bytes) for Slot-0 Thesis Pinning."""
    paras = [p.strip() for p in re.split(r"\n{2,}", article) if p.strip()]
    if len(paras) > 4:
        # Pick first, two middle anchors, and final synthesis paragraph
        idxs = [0, len(paras) // 3, (2 * len(paras)) // 3, len(paras) - 1]
        selected = [paras[i] for i in dict.fromkeys(idxs)][:4]
    else:
        selected = paras[:4]

    n_nodes = len(selected)
    per_node_budget = max(24, (max_bytes - 8 * n_nodes) // max(n_nodes, 1))
    nodes: list[str] = []
    for i, p in enumerate(selected, 1):
        first_clause = re.split(r"[。！？.!?;；\n]", p)[0].strip()
        first_clause = re.sub(r"\s+", " ", first_clause)
        node = _truncate_utf8(first_clause, per_node_budget)
        if node:
            nodes.append(f"{i}.{node}")
    outline = " -> ".join(nodes)
    return _truncate_utf8(outline, max_bytes)


def _gen_offline_article(rng: random.Random, idx: int) -> tuple[str, int, int, str]:
    """Generate clean, zero-cliché multi-paragraph EN/ZH writing samples spanning styles 1, 2, 3."""
    style_id = (idx % 3) + 1  # 1=memo, 2=expository, 3=narrative
    is_zh = (idx % 2) == 1
    lang_id = 1 if is_zh else 0

    if style_id == 1:  # memo
        proj = rng.choice(["EdgeCache-v4", "LedgerSync", "ByteNet-R3", "GridPulse", "QuarryDB", "HarborOps"])
        lat_old = rng.randint(42, 180)
        lat_new = max(6, lat_old // rng.randint(3, 6))
        save_pct = rng.randint(28, 64)
        budget_k = rng.randint(120, 950)
        if is_zh:
            prompt = f"以决策备忘录风格撰写 {proj} 架构切换方案，给出量化瓶颈与执行结论。"
            body = (
                f"结论先行：本季度将 {proj} 核心查询路径迁至零拷贝内存映射方案，P99 延迟从 {lat_old}ms 压降至 {lat_new}ms，算力成本下降 {save_pct}%。\n\n"
                f"瓶颈定位：线上压测显示 {rng.randint(55, 78)}% 的 CPU 时间耗散在反序列化与临时堆分配上，单节点吞吐被锁死在 {rng.randint(12, 38)}k QPS。现行缓存层每小时触发 {rng.randint(4, 15)} 次全局锁抖动，直接拖慢下游结算指标。\n\n"
                f"执行方案：拆分为三周滚动交付。第一周冻结旧接口并切换只读索引分片，预算上限 ${budget_k}K；第二周在 15% 灰度流量下校验一致性错误率为 0%；第三周切全量并回收冗余集群，释放 {rng.randint(8, 32)} TB 内存。\n\n"
                f"风险兜底：若任一分片校验延迟连续 30s 超过 {lat_new * 2}ms，自动回切本地只读副本并在 5s 内触发告警，优先级高于新特性上线。"
            )
        else:
            prompt = f"Write an executive decision memo for migrating {proj} with hard latency and budget numbers."
            body = (
                f"Decision: Migrate {proj} to zero-copy mapped shards this sprint. P99 latency drops from {lat_old}ms to {lat_new}ms while cutting compute spend by {save_pct}%.\n\n"
                f"Bottleneck: Production profiles show {rng.randint(55, 78)}% of CPU cycles vanish into heap copies and serialization overhead. Lock contention caps throughput at {rng.randint(12, 38)}k QPS during peak windows.\n\n"
                f"Execution Plan: Ship in three gated milestones within a ${budget_k}K ceiling. Week 1 freezes schema contracts and stages read-only indexes; Week 2 validates 0% divergence on 15% shadow traffic; Week 3 cuts over and retires {rng.randint(8, 32)} TB of idle RAM.\n\n"
                f"Risk Controls: Any shard exceeding {lat_new * 2}ms for 30s rolls back automatically to the local replica in under 5s. Reliability gates override all feature launches."
            )
    elif style_id == 2:  # expository
        topic_en, topic_zh = rng.choice([
            ("heat-exchange topology in deep-sea hydrothermal vents", "深海热液喷口的热交换拓扑机制"),
            ("phase locking in coupled pendulum clocks", "耦合摆钟的相位锁定原理"),
            ("charge depletion gates in biological spiking neurons", "生物脉冲神经元的电荷耗竭门控机制"),
            ("hydraulic feedback inside alpine glacier melt channels", "高山冰川融水通道的水力反馈机制"),
            ("resonance damping in suspension bridge cables", "悬索桥拉索的共振阻尼机制"),
            ("eddy diffusion across stratified estuary layers", "河口分层水体的涡旋扩散机制"),
            ("acoustic impedance matching in mammalian middle ears", "哺乳动物中耳的声阻抗匹配原理"),
            ("capillary transport in tall conifer xylem networks", "高大针叶树木质部的毛细输运机制"),
        ])
        k_val = rng.randint(2, 29)
        t_ms = rng.randint(15, 480)
        scale_n = rng.randint(4, 16)
        analogy_en, analogy_zh = rng.choice([
            ("a narrow canyon accelerating meltwater until the current carves deeper walls", "狭窄河道迫使水流加速并反向冲刷河床"),
            ("a tuned organ pipe reinforcing only harmonics aligned with its length", "定长管风琴只放大与其声波波长匹配的谐振频率"),
            ("a spring-loaded spillway valve releasing pressure before the reservoir overflows", "带阻尼的溢洪闸门在库容见顶前定向泄压"),
            ("a thermostat relay switching power off once thermal inertia overshoots the band", "温控继电器在热惯性越界瞬间切断输入回路"),
        ])
        if is_zh:
            prompt = f"以深度科普语体剖析{topic_zh}的因果链条与底层物理类比（尺度因子 {scale_n}）。"
            body = (
                f"理解{topic_zh}的关键，在于先拆开局部能量输入与宏观结构反馈之间的因果链条。当系统处于低于 {t_ms} 单位的低扰动区间时，表层介质仅通过缓慢扩散消耗梯度，因此整体状态保持平稳。\n\n"
                f"一旦外部通量越过临界阈值（约为稳态基线的 {k_val} 倍），局部非线性放大效应立刻接管主导权。正如{analogy_zh}一样，能量流会反向重塑边界几何形状，从而在 {scale_n} 个耦合层级间形成自维持循环。\n\n"
                f"这种机制之所以稳定存在，是因为负反馈边界在更高尺度上重新建立了平衡。过快的脉冲消耗会迅速拉低驱动势能，迫使系统短暂休止并重置初始条件，避免整个结构因无限正反馈而解体。\n\n"
                f"将这一模型放回工程设计中，我们看到类似原理反复出现：通过硬约束局部工作容量并引入精确耗竭阈值，复杂系统可以在有限资源内维持长程秩序。"
            )
        else:
            prompt = f"Explain the causal mechanism behind {topic_en} (scale factor {scale_n}) using concrete physical analogies."
            body = (
                f"Explaining {topic_en} begins by separating local energy injection from global structural feedback. Below {t_ms} units of activation flux, gradients dissipate smoothly through linear diffusion, so the system remains quiet.\n\n"
                f"Once flux crosses a critical barrier—roughly {k_val} times baseline—nonlinear amplification drives rapid reorganization. Much like {analogy_en}, the flow reshapes its own boundary conditions across {scale_n} coupled layers, which implies a self-reinforcing cycle.\n\n"
                f"Stability emerges because negative feedback operates on a slower timescale. Rapid discharge depletes the local driving potential, forcing a recovery interval that resets the channel before runaway oscillation can fracture the structure.\n\n"
                f"The same causal architecture reappears in engineered controllers: pairing a tight working-state bottleneck with explicit depletion gates keeps multi-scale dynamics coherent without unbounded memory."
            )
    else:  # narrative
        place_en, place_zh = rng.choice([
            ("the ferry pier at Sheung Wan", "上环码头的木质栈桥"),
            ("the quiet bookshop behind the station", "车站后巷的旧书铺"),
            ("the observatory steps above the harbor", "俯瞰海港的天文台石阶"),
            ("the salt marsh east of the lighthouse", "灯塔东侧的盐沼浅滩"),
            ("the timber footbridge beside the canal", "运河边的老木人行桥"),
            ("the limestone arcade near the dry dock", "干船坞旁的灰石骑楼"),
        ])
        person_en, person_zh = rng.choice([
            ("Lin", "林默"), ("Sora", "沈岚"), ("Hao", "陈砚"), ("Mei", "周宁"), ("Kai", "陆衡"), ("An", "许安"),
        ])
        obj_en, obj_zh = rng.choice([
            ("cold brass key", "微凉的黄铜钥匙"),
            ("worn compass case", "磨白边角的指南针盒"),
            ("folded timetable", "折成方块的旧时刻表"),
            ("silver fountain pen", "冰凉的银色钢笔"),
        ])
        hour = rng.randint(5, 22)
        mins = rng.randint(4, 38)
        if is_zh:
            prompt = f"以克制的文学叙事笔触，描写 {hour} 点前后在{place_zh}发生的短景。"
            body = (
                f"接近 {hour} 点时，潮湿的海风沿着{place_zh}吹过，把铁栏杆上残留的雨点抖落在石板上。远处渡轮的汽笛声低沉而短促，像被雾气捂住的一口呼吸。\n\n"
                f"{person_zh}停下脚步，抬头看着屋檐下摇晃的橘色灯影。他把收好的旧票根塞进外衣口袋，指尖触到那枚{obj_zh}。水面反照出零碎的微光，一阵风声掠过，轮廓立刻散成细小的波纹。\n\n"
                f"“末班班次还要 {mins} 分钟，”售票窗后的老人头也没抬，只把一杯冒着热气的茶往窗沿推了半寸，“今天起风早，站里面些。”\n\n"
                f"{person_zh}轻声道谢，退进廊柱的阴影里。廊外雨丝重新密了起来，敲打着空荡的木椅，而港口尽头的航标灯依旧按固定的节拍亮起、暗下，在暮色里留下一道清晰的回声。"
            )
        else:
            prompt = f"Write a restrained literary scene set near {place_en} around {hour}:00 with crisp sensory detail."
            body = (
                f"Near {hour}:00, a damp breeze swept across {place_en}, shaking leftover rain from the iron rail onto dark stone. Across the channel, a ferry horn sounded once and vanished into the low haze.\n\n"
                f"{person_en} paused at the corner where a single lantern flickered under the eaves, tucking a damp ticket beside the {obj_en} inside the coat pocket and watching broken ribbons of warmth scatter across the tide.\n\n"
                f"\"{mins} minutes until the last crossing,\" the clerk said quietly without looking up, nudging a chipped porcelain cup across the counter. \"Wind turned early tonight.\"\n\n"
                f"{person_en} nodded and stepped back into the narrow shadow of the colonnade. Footsteps faded down the alley behind as the harbor beacon kept its patient rhythm along the horizon."
            )
    return body, lang_id, style_id, prompt


def _stream_hf_writing(max_items: int) -> Iterator[tuple[str, int, str]]:
    """Stream Fineweb-Edu-Chinese-V2.1, cosmopedia-v2, and smoltalk rewriting cleanly."""
    try:
        from datasets import load_dataset
    except Exception:
        return

    # 1. Fineweb-Edu-Chinese-V2.1 (try 4_5/*.parquet first, then 4-5/*.parquet)
    zh_budget = max(1, int(max_items * 0.45))
    got_zh = 0
    for pattern in ("4_5/*.parquet", "4-5/*.parquet"):
        if got_zh >= zh_budget:
            break
        try:
            ds = load_dataset(
                "opencsg/Fineweb-Edu-Chinese-V2.1",
                data_files=pattern,
                split="train",
                streaming=True,
            )
            for row in ds:
                if got_zh >= zh_budget:
                    break
                txt = str(row.get("text") or "").strip()
                b_len = len(txt.encode("utf-8", errors="ignore"))
                if not (900 <= b_len <= 4096):
                    continue
                cleaned = filter_cliches(txt)
                if cleaned is None:
                    continue
                yield cleaned, 1, "请按清晰结构撰写一篇深度中文长文。"
                got_zh += 1
            if got_zh > 0:
                break
        except Exception:
            continue

    # 2. HuggingFaceTB/cosmopedia-v2
    en_budget = max(1, int(max_items * 0.35))
    got_en = 0
    try:
        ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True)
        for row in ds_cosmo:
            if got_en >= en_budget:
                break
            txt = str(row.get("text") or "").strip()
            b_len = len(txt.encode("utf-8", errors="ignore"))
            if not (900 <= b_len <= 4096):
                continue
            cleaned = filter_cliches(txt)
            if cleaned is None:
                continue
            prompt = sanitize_raw_text(str(row.get("prompt") or "Write a structured analytical article.")).strip()[:200]
            yield cleaned, 0, prompt
            got_en += 1
    except Exception:
        pass

    # 3. HuggingFaceTB/smoltalk (explore-instruct-rewriting) & Crownelius high-aesthetic subsets (<= 3 epochs)
    rw_budget = max(1, int(max_items * 0.25))
    got_rw = 0
    for crown_repo in ("Crownelius/Creative-Writing-High-Quality-1300x", "Crownelius/Novel-Crafting-250x"):
        if got_rw >= rw_budget // 2:
            break
        try:
            ds_cr = load_dataset(crown_repo, split="train", streaming=True)
            for row in ds_cr:
                if got_rw >= rw_budget // 2:
                    break
                msgs = row.get("messages") or row.get("conversations") or []
                a_txt = ""
                u_txt = ""
                if isinstance(msgs, list) and msgs:
                    u_txt = next((str(m.get("content") or m.get("value") or "") for m in msgs if m.get("role") in ("user", "human") or m.get("from") in ("user", "human")), "")
                    a_txt = next((str(m.get("content") or m.get("value") or "") for m in msgs if m.get("role") in ("assistant", "gpt") or m.get("from") in ("assistant", "gpt")), "")
                else:
                    a_txt = str(row.get("text") or row.get("output") or row.get("response") or "")
                    u_txt = str(row.get("prompt") or row.get("instruction") or "Write a grounded literary scene.")
                if not a_txt or not (600 <= len(a_txt.encode("utf-8", errors="ignore")) <= 4096):
                    continue
                cleaned = filter_cliches(a_txt)
                if cleaned is None:
                    continue
                yield cleaned, 0, sanitize_raw_text(u_txt).strip()[:220] or "Write a grounded literary scene."
                got_rw += 1
        except Exception:
            continue

    try:
        ds_rw = load_dataset("HuggingFaceTB/smoltalk", "explore-instruct-rewriting", split="train", streaming=True)
        for row in ds_rw:
            if got_rw >= rw_budget:
                break
            msgs = row.get("messages") or []
            u_txt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            a_txt = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            if not a_txt or not (600 <= len(a_txt.encode("utf-8", errors="ignore")) <= 4096):
                continue
            cleaned = filter_cliches(str(a_txt))
            if cleaned is None:
                continue
            yield cleaned, 0, sanitize_raw_text(str(u_txt)).strip()[:240]
            got_rw += 1
    except Exception:
        pass


def synthesize_slop_draft_from_clean(clean_article: str, lang_id: int) -> tuple[str, str]:
    """Corrupt a clean essay/scene into an AI-slop draft (`mask=0` input ONLY!) and return `(slop, diagnosis)`."""
    paras = [p.strip() for p in clean_article.split("\n\n") if p.strip()]
    if len(paras) >= 3:
        # Swap middle two paragraphs to simulate broken causal/narrative flow
        mut = list(paras)
        mut[1], mut[2] = mut[2], mut[1]
    else:
        mut = list(paras)

    if lang_id == 1:
        slop = (
            "值得注意的是，这无疑是一把双刃剑。\n\n"
            + "\n\n".join(mut)
            + "\n\n总而言之，综上所述，我们必须全面看待这一深刻议题。"
        )
        diag = "诊断:删去首尾AI塑料套话 + 修复第2/3段倒置脉络 -> 凝练重写"
    else:
        slop = (
            "It is worth noting that we must delve into this double-edged sword.\n\n"
            + "\n\n".join(mut)
            + "\n\nIn conclusion, needless to say, these dynamics remain crucial."
        )
        diag = "诊断:purge AI clichés + restore paragraph 2/3 causal progression -> tighten prose"
    return slop, diag


def encode_anti_slop_rewrite_sample(
    clean_article: str,
    lang_id: int,
    style_id: int,
) -> tuple[bytes, bytes, int, int]:
    """Data Alchemy #2: `[Slop Draft in User Turn, mask=0] -> [\x1dS diagnosis \x1dE + Clean Final, mask=1]`."""
    clean_body = sanitize_raw_text(clean_article).strip()
    assert not _CLICHE_RE.search(clean_body), "Target clean article must be 100% cliché-free!"
    slop_draft, diag = synthesize_slop_draft_from_clean(clean_body, lang_id)
    instruction = (
        "请删去以下初稿中的空洞 AI 套话、修复错乱的段落因果顺序，并润色为凝练定稿：\n"
        if lang_id == 1
        else "Strip all hollow AI clichés, restore logical paragraph order, and tighten this rough draft:\n"
    )
    u_text = sanitize_raw_text(instruction + slop_draft).strip()
    diag_b = _truncate_utf8(sanitize_raw_text(diag).strip(), MAX_OUTLINE_BYTES).encode("utf-8", errors="strict")
    body_b = clean_body.encode("utf-8", errors="strict")

    u_frame = TURN_USER_PREFIX + u_text.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX
    a_frame = SCRATCH_START + diag_b + SCRATCH_END + body_b + TURN_EOT_SUFFIX
    raw_bytes = u_frame + a_frame
    mask_bytes = (b"\x00" * len(u_frame)) + (b"\x01" * len(a_frame))
    assert len(raw_bytes) == len(mask_bytes)
    raw_bytes.decode("utf-8", errors="strict")
    return raw_bytes, mask_bytes, int(lang_id), int(style_id)


def encode_writing_sample(
    article: str,
    lang_id: int,
    style_id: int,
    prompt: str = "",
    as_dialogue: bool = True,
) -> tuple[bytes, bytes, int, int]:
    """Wrap writing article with prepended `< 180 B` reverse outline (`\\x1dS\\n...\\x1dE\\n`).

    Returns `(raw_bytes, mask_bytes, lang_id, style_id)`.
    """
    clean_body = sanitize_raw_text(article).strip()
    outline = extract_reverse_outline(clean_body, MAX_OUTLINE_BYTES)
    out_b = outline.encode("utf-8", errors="strict")
    assert len(out_b) < 180, f"Outline exceeded 180B: {len(out_b)}"

    body_b = clean_body.encode("utf-8", errors="strict")
    if as_dialogue and prompt:
        style_tag = STYLE_ID_TO_NAME.get(style_id, "expository")
        p_clean = sanitize_raw_text(prompt).strip()
        u_frame = (
            TURN_USER_PREFIX
            + f"[{style_tag}] {p_clean}".encode("utf-8", errors="strict")
            + b"\n"
            + TURN_ASST_PREFIX
        )
        a_frame = SCRATCH_START + out_b + SCRATCH_END + body_b + TURN_EOT_SUFFIX
        raw_bytes = u_frame + a_frame
        mask_bytes = (b"\x00" * len(u_frame)) + (b"\x01" * len(a_frame))
    else:
        raw_bytes = SCRATCH_START + out_b + SCRATCH_END + body_b + b"\n"
        mask_bytes = b"\x01" * len(raw_bytes)

    raw_bytes.decode("utf-8", errors="strict")
    return raw_bytes, mask_bytes, int(lang_id), int(style_id)


def build_cap_writing_corpus(
    out_dir: str | Path = "artifacts/cap_data",
    target_mb: float = 100.0,
    smoke_quota: bool = False,
    offline: bool = False,
    seed: int = 202612,
) -> dict:
    """Build `cap_writing_{train,val}.{bin,idx,loss_mask.bin}` anti-cliché styled writing shards."""
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
    docs: list[tuple[bytes, bytes, int, int]] = []
    acc_bytes = 0
    style_counts = {1: 0, 2: 0, 3: 0}
    max_outline_len = 0

    def _try_add(body: str, lid: int, sid: int, prompt: str, as_dialogue: bool, as_antislop: bool = False) -> bool:
        nonlocal acc_bytes, max_outline_len
        if acc_bytes >= total_target:
            return False
        if _CLICHE_RE.search(body):
            return False
        try:
            if as_antislop and len(body.encode("utf-8", errors="ignore")) <= 1800:
                raw_b, msk_b, l_i, s_i = encode_anti_slop_rewrite_sample(body, lid, sid)
            else:
                raw_b, msk_b, l_i, s_i = encode_writing_sample(body, lid, sid, prompt, as_dialogue=as_dialogue)
        except Exception:
            return False
        if len(raw_b) < 256 or len(raw_b) > 4608:
            return False
        h = hashlib.blake2b(raw_b, digest_size=12).digest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        s0 = raw_b.find(SCRATCH_START)
        s1 = raw_b.find(SCRATCH_END)
        if s0 >= 0 and s1 > s0:
            max_outline_len = max(max_outline_len, s1 - (s0 + len(SCRATCH_START)))
        docs.append((raw_b, msk_b, l_i, s_i))
        style_counts[s_i] = style_counts.get(s_i, 0) + 1
        acc_bytes += len(raw_b)
        return True

    if not offline:
        hf_cap = int(total_target * 0.70)
        est_items = max(300, int(hf_cap / 1400))
        for i, (body, lid, prompt) in enumerate(_stream_hf_writing(est_items)):
            sid = classify_writing_style(body)
            _try_add(
                body,
                lid,
                sid,
                prompt,
                as_dialogue=(i % 2 == 0),
                as_antislop=((i % 4) == 3),
            )
            if acc_bytes >= hf_cap:
                break

    idx = 0
    while acc_bytes < total_target:
        body, lid, sid, prompt = _gen_offline_article(rng, idx)
        # Mix 25% Anti-Slop Rewriting Pairs (`slop [mask=0] -> diagnosis + clean [mask=1]`)
        _try_add(
            body,
            lid,
            sid,
            prompt,
            as_dialogue=(idx % 2 == 0),
            as_antislop=((idx % 4) == 3),
        )
        idx += 1

    rng.shuffle(docs)
    n_val = max(12, int(len(docs) * (val_target / max(float(total_target), 1.0))))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    def _write_split(split: str, split_docs: list[tuple[bytes, bytes, int, int]]) -> dict:
        bin_p = out_path / f"cap_writing_{split}.bin"
        idx_p = out_path / f"cap_writing_{split}.idx"
        msk_p = out_path / f"cap_writing_{split}.loss_mask.bin"
        off = 0
        s_dist = {1: 0, 2: 0, 3: 0}
        with bin_p.open("wb") as fb, idx_p.open("wb") as fi, msk_p.open("wb") as fm:
            for raw_b, msk_b, lid, sid in split_docs:
                n = len(raw_b)
                fb.write(raw_b)
                fm.write(msk_b)
                fi.write(struct.pack(IDX_RECORD_FMT, off, n, lid, SOURCE_ID_WRITING, 0, sid))
                off += n
                s_dist[sid] = s_dist.get(sid, 0) + 1
        return {
            "split": split,
            "num_docs": len(split_docs),
            "total_bytes": off,
            "docs_by_style": s_dist,
        }

    train_meta = _write_split("train", train_docs)
    val_meta = _write_split("val", val_docs)
    summary = {
        "dataset": "cap_writing",
        "smoke_quota": smoke_quota,
        "offline": offline,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "total_bytes": train_meta["total_bytes"] + val_meta["total_bytes"],
        "total_docs": train_meta["num_docs"] + val_meta["num_docs"],
        "docs_by_style": style_counts,
        "max_outline_bytes": max_outline_len,
        "train": train_meta,
        "val": val_meta,
    }
    (out_path / "cap_writing_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DCPA Capability Writing Pure-Byte Dataset")
    ap.add_argument("--out-dir", type=str, default="artifacts/cap_data")
    ap.add_argument("--target-mb", type=float, default=100.0)
    ap.add_argument("--smoke-quota", action="store_true", help="Build 2 MB local smoke quota")
    ap.add_argument("--offline", action="store_true", help="Zero-network deterministic synthesis")
    ap.add_argument("--seed", type=int, default=202612)
    args = ap.parse_args()
    res = build_cap_writing_corpus(
        out_dir=args.out_dir,
        target_mb=args.target_mb,
        smoke_quota=args.smoke_quota,
        offline=args.offline,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
