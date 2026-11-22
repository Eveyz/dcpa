"""64-Sample Multilingual & Nested-Code Ragged Overfit Smoke Test (500 Steps).

Implements Deliverable 2 of the DCPA Chief Verification & System Audit:
  - 64 deterministic built-in samples (80..240 UTF-8 bytes each, zero external network):
      * 16 English SVO dialogues & morphological inflections (lang_id=0)
      * 16 Chinese multi-byte causal narratives with full-width punctuation (lang_id=1)
      * 16 Japanese SOV sentences with particles は/を/が & final verbs (lang_id=2)
      * 16 syntactically valid Python (lang_id=3) & Rust (lang_id=4) functions with
        >= 3 levels of nested { ( [ ... ] ) } brackets and indented newlines
  - Zero-padding Ragged `doc_cu_seqlens` packing (strictly zero 0x00 tail padding).
  - Prototype config: d_s=192, D=512, K=8, d_r=32, dropout=0.0, target_rho=9.5 (r0=1/9.5).
  - Optimizer: AdamW(lr=8e-4, weight_decay=0.0, betas=(0.9, 0.95)), grad clip max_norm=1.0.
  - Linear temperature annealing over 500 steps: tau_s: 1.0 -> 0.35, tau_m: 1.2 -> 0.35.
  - Hard Step-500 Pass Gates (enforced via assert):
      1. bpc = L_lang / ln(2) < 0.25
      2. 8.0 <= compression_ratio (T / M) <= 11.0
      3. concept_std_mean >= 0.85 AND concept_std_min >= 0.60
      4. L_halt < 0.05
      5. Autoregressive greedy decoding on EN/ZH/JA/Rust probes:
         100% valid UTF-8 (zero UnicodeDecodeError) AND 100% {}()[] bracket closure!
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from utils.device import (
    amp_autocast,
    describe_hardware_profile,
    detect_hardware_profile,
    synchronize,
)


# =============================================================================
# Fixed Deterministic 64-Sample Multilingual & Nested-Code Corpus (80..240 B)
# =============================================================================
EN_SAMPLES: list[str] = [
    "Maya asked Liam: 'Did the two foxes cross the wooden bridge?' He replied: 'Yes, they crossed quickly and carried three berries.'",
    "Clara told Noah: 'Yesterday our little birds built warm nests.' Noah smiled: 'Today those birds are singing louder than before.'",
    "Ethan asked Zoe: 'Have you repaired the broken lanterns yet?' She answered: 'I repaired two lanterns and polished three glass lenses.'",
    "Lucas said to Ava: 'The tall baker baked fresh loaves before sunrise.' Ava replied: 'His apprentices stacked twenty warm loaves neatly.'",
    "Hannah asked Owen: 'Why did the silver trains stop near the canyon?' Owen answered: 'Heavy rocks rolled across the iron tracks.'",
    "Leo asked Mia: 'Do the playful puppies chase yellow butterflies?' Mia laughed: 'They chased five butterflies and barked happily.'",
    "Nora told Eli: 'Our chemistry teacher explained those crystal reactions clearly.' Eli replied: 'Her diagrams helped every student understand.'",
    "Caleb asked Ruby: 'Will the sailboats reach the northern harbor tonight?' Ruby answered: 'Strong winds pushed all six boats forward.'",
    "Grace told Isaac: 'Last autumn we planted forty apple saplings here.' Isaac smiled: 'Now their branches carry bright green leaves.'",
    "Julian asked Chloe: 'Did the clockmaker tighten those brass springs?' Chloe replied: 'He tightened every spring and tested the chimes.'",
    "Stella said to Henry: 'Three curious otters splashed beneath the willow roots.' Henry replied: 'One otter balanced a smooth pebble.'",
    "Oliver asked Lily: 'Are the librarians sorting those ancient atlases today?' Lily answered: 'They sorted twelve boxes and labeled shelves.'",
    "Hazel told Samuel: 'Our violinists practiced those difficult passages twice.' Samuel replied: 'Their melodies sounded crisp and balanced.'",
    "Mason asked Aria: 'Did the mountain hikers pack extra woolen blankets?' Aria answered: 'They packed blankets, maps, and copper flasks.'",
    "Evelyn said to Jack: 'Two patient carpenters carved oak benches for the square.' Jack replied: 'Neighbors admired their sturdy joints.'",
    "Daniel asked Sofia: 'Have the gardeners watered those blooming orchids?' Sofia replied: 'They watered every pot before noon arrived.'",
]

ZH_SAMPLES: list[str] = [
    "清晨山谷起雾，老石桥下水流湍急；因为昨晚下过暴雨，所以木船暂时停靠在东岸码头。",
    "小满在书房整理旧地图时发现一枚铜钥匙，于是她顺着纸页上的红线找到了阁楼暗格。",
    "秋风吹落银杏叶，铺满了安静的青石小巷；扫街的大叔推着竹筐，把金黄的叶片仔细收拢。",
    "因为炉火太旺，铁壶里的山泉很快就沸腾了；茶师立刻提壶冲泡，清香瞬间溢满整间木屋。",
    "天文台的圆顶缓缓打开，镜筒对准明亮的猎户座；由于今晚云层稀薄，观测记录格外清晰。",
    "老木匠先用墨斗弹好直线，再用刨子推平松木板；经过三次打磨，书架的榫卯严丝合缝。",
    "暴雨停歇之后，梯田蓄满清澈的雨水；青蛙在田埂间齐声鸣叫，远处的灯火映在水面上。",
    "因为海风突然转向，渔村的灯塔提前亮起暖光；三艘返航的木帆船顺着航标稳稳进港。",
    "林医生翻阅完泛黄的药草图谱，决定在山坡北侧试种紫苏；果然半个月后幼苗破土而出。",
    "地铁列车穿过江底隧道，车窗外闪过一排蓝色信号灯；乘客合上书本，准备在下一站换乘。",
    "由于齿轮缺少润滑油，钟楼的报时机构慢了五秒；技师拆下铜轴清洗后，钟声重新准时响起。",
    "春雷响过第一声，竹林里的春笋接连顶开湿润的泥土；村里的孩子们提着小篮上山采挖。",
    "画家在宣纸上先淡墨勾勒远山轮廓，再蘸青绿点染松林；待墨色干透，整幅画卷灵动自然。",
    "因为电路保险丝熔断，温室的恒温灯突然熄灭；值班员迅速更换备件，兰花没有受到冻害。",
    "邮差骑着绿色自行车穿过长长的梧桐大道；每当铃声响起，院子里的橘猫总会跳上矮墙张望。",
    "雪夜里火车站台灯光明亮，蒸汽从机车底部缓缓升腾；旅人握紧热茶，等待发车广播响起。",
]

JA_SAMPLES: list[str] = [
    "青い鳥が静かな森の上をゆっくりと飛び、古い杉の枝にそっと止まりました。",
    "少年は赤い自転車で坂道を下り、川辺の小さなパン屋で焼きたてのパンを買いました。",
    "冷たい雨が石畳を濡らし、軒先の提灯が柔らかな橙色の光を道に落としました。",
    "図書館の司書は分厚い辞書を棚に戻し、新しい地図帳を机の端に丁寧に並べました。",
    "白い猫が日当たりの良い縁側で丸くなり、庭の風鈴が涼しい音を静かに鳴らしました。",
    "職人は細かい鉋で檜の板を削り、木目の美しい小箱を夕方までに仕上げました。",
    "朝の特急列車が長い鉄橋を渡り、霧に包まれた湖の水面を窓の向こうに映しました。",
    "料理人は新鮮な大根を薄く切り、昆布の出汁が香る温かい鍋へ静かに入れました。",
    "子どもたちは広い公園で紙飛行機を飛ばし、強い秋風がそれを高く運びました。",
    "天文学者が望遠鏡の角度を微調整し、冬の夜空に輝く淡い星雲を正確に記録しました。",
    "古い時計屋は小さな歯車を一つずつ磨き、止まっていた掛け時計を再び動かしました。",
    "漁師は夜明け前の港で網を点検し、穏やかな波の上へ小さな船をゆっくり出しました。",
    "学生は白いノートへ数式を丁寧に書き写し、難しい証明の手順を最後まで理解しました。",
    "庭師は伸びた松の枝を鋏で整え、苔むした石灯籠の周りをきれいに掃き清めました。",
    "音楽家は静かな講堂でチェロの弦を調律し、深い響きの旋律をゆったりと奏でました。",
    "旅人は雪の積もる駅で熱いお茶を飲み、次の普通列車が到着する合図を待ちました。",
]

PYTHON_SAMPLES: list[str] = [
    "def sum_pos(rows):\n    acc = 0\n    for r in rows:\n        if len(r) > 0:\n            acc += sum([int(x) for x in (r[0], r[-1]) if x > 0])\n    return acc\n",
    "def map_pairs(items):\n    out = {}\n    for k, v in items:\n        if k not in out:\n            out[k] = [(v, (k, [v]))]\n    return len(out)\n",
    "def clip_norm(vals, hi):\n    res = []\n    for v in vals:\n        if v >= 0:\n            res.append(min((hi, max([0, v]))))\n    return tuple(res)\n",
    "def build_idx(keys):\n    tbl = {k: [i, (k, i + 1)] for i, k in enumerate(keys)}\n    if len(tbl) > 0:\n        return sorted(list(tbl.keys()))\n    return []\n",
    "def dot_grid(mat, vec):\n    total = 0\n    for row in mat:\n        if len(row) == len(vec):\n            total += sum([a * b for a, b in zip(row, (vec))])\n    return total\n",
    "def filter_tags(tags):\n    kept = {}\n    for idx, t in enumerate(tags):\n        if len(t) > 1:\n            kept[t] = (idx, [t[0], t[-1]])\n    return kept\n",
    "def pack_spans(spans):\n    buf = []\n    for lo, hi in spans:\n        if hi > lo:\n            buf.append({'range': (lo, [hi, hi - lo])})\n    return buf\n",
    "def merge_bins(a, b):\n    out = {0: [(a[0], [b[0]])]}\n    for i in range(1, min(len(a), len(b))):\n        out[i] = [(a[i], [b[i]])]\n    return out\n",
]

RUST_SAMPLES: list[str] = [
    "fn sum_nested(xs: &[i32]) -> i32 {\n    let mut s = 0;\n    for &v in xs {\n        if v > 0 {\n            s += ([v, (v + 1)][0]);\n        }\n    }\n    s\n}\n",
    "fn first_pair(a: &[u8]) -> (u8, usize) {\n    let mut n = 0usize;\n    if !a.is_empty() {\n        n = ([a[0], (a.len() as u8)][1]) as usize;\n    }\n    (a[0], n)\n}\n",
    "fn clamp_step(x: i32, lim: i32) -> i32 {\n    let mut y = x;\n    if lim > 0 {\n        let arr = [(y, [lim, -lim])];\n        y = arr[0].0;\n    }\n    y\n}\n",
    "fn count_even(nums: &[u32]) -> usize {\n    let mut c = 0;\n    for &n in nums {\n        if (n % 2) == 0 {\n            let t = ([n, (n + 2)][0]);\n            if t >= 0 { c += 1; }\n        }\n    }\n    c\n}\n",
    "fn fold_bytes(buf: &[u8]) -> u32 {\n    let mut acc: u32 = 0;\n    for &b in buf {\n        if b >= 32 {\n            acc += ([(b as u32), (acc & 7)][0]);\n        }\n    }\n    acc\n}\n",
    "fn swap_tag(a: i32, b: i32) -> [i32; 2] {\n    let mut out = [a, b];\n    if a > b {\n        let tmp = ([b, (a)][0]);\n        out = [tmp, a];\n    }\n    out\n}\n",
    "fn span_len(lo: usize, hi: usize) -> usize {\n    if hi > lo {\n        let d = ([(hi - lo), (lo + 1)][0]);\n        return d;\n    }\n    0\n}\n",
    "fn pick_max(xs: &[i32]) -> i32 {\n    let mut best = 0;\n    for &x in xs {\n        if x > best {\n            best = ([x, (best + 1)][0]);\n        }\n    }\n    best\n}\n",
]


def _check_bracket_balance(text: str) -> tuple[bool, int, int]:
    """Verify 100% {}()[] bracket pairing balance and report (balanced, total_pairs, max_depth)."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = {"}", ")", "]"}
    stack: list[str] = []
    total_pairs = 0
    max_depth = 0
    in_str: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch in pairs:
            stack.append(pairs[ch])
            max_depth = max(max_depth, len(stack))
        elif ch in closing:
            if not stack or stack.pop() != ch:
                return False, total_pairs, max_depth
            total_pairs += 1
        i += 1
    return (len(stack) == 0 and total_pairs > 0), total_pairs, max_depth


def build_64_fixture() -> tuple[list[tuple[int, str]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble 64 multi-domain UTF-8 samples into zero-padding Ragged `doc_cu_seqlens`."""
    docs: list[tuple[int, str]] = []
    docs.extend((0, s) for s in EN_SAMPLES)
    docs.extend((1, s) for s in ZH_SAMPLES)
    docs.extend((2, s) for s in JA_SAMPLES)
    docs.extend((3, s) for s in PYTHON_SAMPLES)
    docs.extend((4, s) for s in RUST_SAMPLES)
    assert len(docs) == 64, f"Expected exactly 64 samples, got {len(docs)}"

    flat_bytes: list[int] = []
    offsets: list[int] = [0]
    lang_ids: list[int] = []

    for idx, (lid, txt) in enumerate(docs):
        enc = txt.encode("utf-8", errors="strict")
        assert 80 <= len(enc) <= 240, f"Sample {idx} (lang={lid}) length {len(enc)} outside [80, 240]"
        if lid in (3, 4):
            ok, n_pairs, max_d = _check_bracket_balance(txt)
            assert ok and max_d >= 3, (
                f"Code sample {idx} (lang={lid}) failed >=3-level bracket check: ok={ok}, depth={max_d}"
            )
        flat_bytes.extend(enc)
        offsets.append(len(flat_bytes))
        lang_ids.append(lid)

    raw_stream = torch.tensor(flat_bytes, dtype=torch.uint8).unsqueeze(0)  # [1, N_total] zero padding!
    doc_cu = torch.tensor(offsets, dtype=torch.int32)
    lang_t = torch.tensor(lang_ids, dtype=torch.long)
    return docs, raw_stream, doc_cu, lang_t


@torch.no_grad()
def _greedy_decode_single_doc(
    model: DCPAModel,
    text: str,
    lang_id: int,
    device: torch.device,
    tau_s: float = 0.35,
    tau_m: float = 0.35,
) -> dict[str, Any]:
    """Run SGSC segmentation + BCG-PC + true autoregressive greedy PMD byte decoding."""
    model.eval()
    raw_py = text.encode("utf-8")
    x = torch.tensor(list(raw_py), dtype=torch.uint8, device=device).unsqueeze(0)  # [1, L]
    lang_t = torch.tensor([lang_id], dtype=torch.long, device=device)

    out = model(x, lang_t, stage="B", tau_s=tau_s, tau_m=tau_m, use_broca_cq=False)
    cu = out.sgsc.cu_seqlens.detach().cpu().tolist()
    segments = [raw_py[cu[i] : cu[i + 1]].decode("utf-8", errors="replace") for i in range(len(cu) - 1)]
    seg_vis = "|".join(segments)

    # Recompute segment-level (f_out, delta_syn) identically to forward streaming path
    l_reg = model.pmd.get_lang_vec(lang_t)  # [1, d_l]
    m_total = out.sgsc.f_stem.shape[0]
    g_state = model.bcg_pc.init_state(1, device=device, dtype=out.sgsc.c_slow.dtype)

    f_outs = []
    delta_syns = []
    for m in range(m_total):
        hip_r = model.hippocampus.read_comprehension(
            f_stem=out.sgsc.f_stem[m : m + 1],
            c_slow=out.sgsc.c_slow[m : m + 1],
            sem_norm=out.sgsc.sem_norm[m : m + 1],
            mean_info=out.sgsc.mean_info[m : m + 1],
            fast_state=None,
            enabled=False,
        )
        bcg_o = model.bcg_pc.step(
            c_fused=hip_r.c_fused,
            prec_bu=hip_r.prec_bu,
            ach=hip_r.ach_signal,
            state=g_state,
            tau_m=tau_m,
            use_gumbel=False,
            update_c0_ema=False,
        )
        g_state = bcg_o.state
        f_out_m, _ = model.hippocampus.read_expression(
            z_out=bcg_o.z_star, l_reg=l_reg, fast_state=None, enabled=False
        )
        delta_syn_m = model.pmd.compute_streaming_delta_syn(
            g_ctx=bcg_o.g_ctx, z_star=bcg_o.z_star, l_reg=l_reg
        )
        f_outs.append(f_out_m)
        delta_syns.append(delta_syn_m)

    # True autoregressive greedy decoding using motor charge depletion (zero ground-truth length hints!)
    m_cur = model.pmd.m0.unsqueeze(0).to(device=device, dtype=out.sgsc.f_stem.dtype).clone()
    xi_cur = torch.zeros((1, model.cfg.stack_depth, model.cfg.stack_tag_dim), device=device, dtype=torch.float32)
    prev_b = torch.zeros((1,), device=device, dtype=torch.long)

    gen_bytes: list[int] = []
    for m in range(m_total):
        f_m = f_outs[m]
        d_m = delta_syns[m]
        charge_cur = model.pmd.init_segment_charge(f_m, d_m, l_reg)
        for tau_step in range(1, model.cfg.max_seg_len + 1):
            logits, m_cur, xi_cur, charge_cur, p_end = model.pmd.step_byte_streaming(
                prev_byte=prev_b,
                f_out=f_m,
                delta_syn=d_m,
                l_reg=l_reg,
                tau_step=tau_step,
                m_prev=m_cur,
                xi_prev=xi_cur,
                charge_prev=charge_cur,
            )
            next_b = logits.argmax(dim=-1)
            gen_bytes.append(int(next_b.item()))
            prev_b = next_b
            if float(p_end[0].item()) >= 0.5:
                break

    gen_raw = bytes(gen_bytes)
    utf8_ok = True
    try:
        decoded_str = gen_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        utf8_ok = False
        decoded_str = gen_raw.decode("utf-8", errors="replace")

    max_len = max(len(text), len(decoded_str), 1)
    char_matches = sum(1 for a, b in zip(text, decoded_str) if a == b)
    exact_match_pct = 100.0 * char_matches / max_len

    bracket_ok, bracket_pairs, bracket_depth = _check_bracket_balance(decoded_str)
    return {
        "lang_id": lang_id,
        "original": text,
        "decoded": decoded_str,
        "seg_vis": seg_vis,
        "num_segments": m_total,
        "utf8_ok": utf8_ok,
        "exact_match_pct": exact_match_pct,
        "bracket_ok": bracket_ok,
        "bracket_pairs": bracket_pairs,
        "bracket_depth": bracket_depth,
    }


def run_smoke_overfit_64(
    total_steps: int = 500,
    device_str: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    profile = detect_hardware_profile(preferred=device_str, deterministic=False)
    device = profile.device
    print(describe_hardware_profile(profile), flush=True)

    docs, raw_stream, doc_cu, lang_t = build_64_fixture()
    raw_stream = raw_stream.to(device)
    doc_cu = doc_cu.to(device)
    lang_t = lang_t.to(device)
    total_bytes = int(raw_stream.shape[1])

    # Prototype configuration strictly per audit spec: d_s=192, D=512, K=8, d_r=32, rho_target=9.5
    cfg = DCPAConfig(
        d_s=192,
        d_concept=512,
        k_slots=8,
        d_r=32,
        d_merge=192,
        d_h=2048,
        s_active=24,
        num_experts=8,
        top_k_experts=2,
        expert_hidden=768,
        l_max=4,
        target_spike_rate=1.0 / 9.5,
    )
    model = DCPAModel(cfg).to(device=device, dtype=torch.float32)
    param_counts = model.count_parameters()
    print(
        f"[smoke_overfit_64] 64 packed docs (zero padding, total_bytes={total_bytes}) | "
        f"cfg=(d_s={cfg.d_s}, D={cfg.d_concept}, K={cfg.k_slots}, d_r={cfg.d_r}) | "
        f"params={param_counts['total'] / 1e6:.2f}M | target_rho=9.5 (r0={cfg.target_spike_rate:.4f})",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    t0 = time.perf_counter()
    trajectory: list[dict[str, float]] = []

    def _record(step: int, out) -> dict[str, float]:
        m = out.metrics
        bpc = m["loss_lang"] / math.log(2.0)
        row = {
            "step": step,
            "bpc": bpc,
            "loss_lang": m["loss_lang"],
            "loss_pc": m["loss_pc"],
            "loss_halt": m["loss_halt"],
            "loss_rate": m["loss_rate"],
            "comp_ratio": m["compression_ratio"],
            "std_mean": m["c_slow_std_mean"],
            "std_min": m["c_slow_std_min"],
            "merge_entropy": m["merge_entropy_nats"],
            "theta": m["theta"],
            "num_segs": int(out.sgsc.f_stem.shape[0]),
        }
        trajectory.append(row)
        print(
            f"[Step {step:03d}/{total_steps}] "
            f"BPC={bpc:.4f} | L_PC={row['loss_pc']:.4f} | L_halt={row['loss_halt']:.4f} | "
            f"Comp={row['comp_ratio']:.2f}x (M={row['num_segs']}) | "
            f"Std={row['std_mean']:.3f}/{row['std_min']:.3f} | "
            f"MergeH={row['merge_entropy']:.3f} | theta={row['theta']:.3f} | "
            f"t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return row

    # Step 0 baseline measurement (before parameter updates)
    model.eval()
    with torch.no_grad():
        out0 = model(
            raw_stream, lang_t, stage="B", tau_s=1.0, tau_m=1.2,
            use_broca_cq=False, doc_cu_seqlens=doc_cu,
        )
    _record(0, out0)

    # 500-step overfit loop with linear annealing tau_s: 1.0 -> 0.35, tau_m: 1.2 -> 0.35
    model.train()
    model.set_stage_freeze("B")
    last_row: dict[str, float] = trajectory[-1]

    for step in range(1, total_steps + 1):
        prog = float(step - 1) / float(max(total_steps - 1, 1))
        tau_s = 1.0 + (0.35 - 1.0) * prog
        tau_m = 1.2 + (0.35 - 1.2) * prog

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, dtype=torch.bfloat16, enabled=profile.use_bf16):
            out = model(
                raw_stream,
                lang_t,
                stage="B",
                tau_s=tau_s,
                tau_m=tau_m,
                use_broca_cq=False,
                doc_cu_seqlens=doc_cu,
            )

        assert torch.isfinite(out.loss_total), f"Non-finite loss at step {step}"
        out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == total_steps:
            synchronize(device)
            last_row = _record(step, out)

    # Final deterministic evaluation at Step 500
    model.eval()
    with torch.no_grad():
        out_final = model(
            raw_stream,
            lang_t,
            stage="B",
            tau_s=0.35,
            tau_m=0.35,
            use_broca_cq=False,
            doc_cu_seqlens=doc_cu,
        )
    final_m = out_final.metrics
    final_bpc = final_m["loss_lang"] / math.log(2.0)
    final_comp = final_m["compression_ratio"]
    final_std_mean = final_m["c_slow_std_mean"]
    final_std_min = final_m["c_slow_std_min"]
    final_l_halt = final_m["loss_halt"]

    print("\n=== STEP 500 EVAL SUMMARY ===", flush=True)
    print(
        f"BPC={final_bpc:.4f} (<0.25) | Comp={final_comp:.2f}x ([8.0, 11.0]) | "
        f"Std_Mean={final_std_mean:.4f} (>=0.85) | Std_Min={final_std_min:.4f} (>=0.60) | "
        f"L_halt={final_l_halt:.4f} (<0.05) | MergeH={final_m['merge_entropy_nats']:.4f}",
        flush=True,
    )

    # Probe 4 representative samples (EN, ZH, JA, Rust) via true greedy autoregressive decoding
    probe_indices = [0, 16, 32, 56]  # EN #0, ZH #0, JA #0, Rust #0
    probe_names = ["English (lang_id=0)", "Chinese (lang_id=1)", "Japanese (lang_id=2)", "Rust (lang_id=4)"]
    probe_results: list[dict[str, Any]] = []

    print("\n=== FOUR-DOMAIN AUTOREGRESSIVE GREEDY DECODING PROBES ===", flush=True)
    for name, d_idx in zip(probe_names, probe_indices):
        lid, txt = docs[d_idx]
        res = _greedy_decode_single_doc(model, txt, lid, device=device, tau_s=0.35, tau_m=0.35)
        res["domain_name"] = name
        probe_results.append(res)
        print(f"\n--- {name} (segments={res['num_segments']}, UTF-8={res['utf8_ok']}, ExactMatch={res['exact_match_pct']:.2f}%) ---")
        print(f"SGSC Split: {res['seg_vis']}")
        print(f"Original  : {res['original']!r}")
        print(f"Decoded   : {res['decoded']!r}")
        if lid in (3, 4):
            print(f"Brackets  : balanced={res['bracket_ok']}, pairs={res['bracket_pairs']}, depth={res['bracket_depth']}")

    # Hard Step-500 Pass Gates (STRICT — NEVER RELAX!)
    assert final_bpc < 0.25, f"GATE FAIL: BPC={final_bpc:.4f} >= 0.25"
    assert 8.0 <= final_comp <= 11.0, f"GATE FAIL: compression_ratio={final_comp:.2f} outside [8.0, 11.0]"
    assert final_std_mean >= 0.85, f"GATE FAIL: concept_std_mean={final_std_mean:.4f} < 0.85"
    assert final_std_min >= 0.60, f"GATE FAIL: concept_std_min={final_std_min:.4f} < 0.60"
    assert final_l_halt < 0.05, f"GATE FAIL: L_halt={final_l_halt:.4f} >= 0.05"

    for res in probe_results:
        assert res["utf8_ok"], f"GATE FAIL: Invalid UTF-8 decoded on {res['domain_name']}"
        if res["lang_id"] in (3, 4):
            assert res["bracket_ok"], f"GATE FAIL: Unbalanced brackets decoded on {res['domain_name']}"

    print("\n[ALL SMOKE_OVERFIT_64 HARD GATES PASSED!]", flush=True)
    return {
        "trajectory": trajectory,
        "final": {
            "bpc": final_bpc,
            "comp_ratio": final_comp,
            "std_mean": final_std_mean,
            "std_min": final_std_min,
            "loss_halt": final_l_halt,
            "loss_pc": final_m["loss_pc"],
            "merge_entropy": final_m["merge_entropy_nats"],
        },
        "probes": probe_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA 64-sample multilingual Ragged 500-step smoke overfit.")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_smoke_overfit_64(total_steps=args.steps, device_str=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
