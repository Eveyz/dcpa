"""Zero-gradient 500-step Visual Diagnostic Probe & Concise Console Telemetry for DCPA.

Implements Deliverable 4 of the DCPA Training System:
  - Compact single-line console progress row every 50 steps (no terminal spam).
  - Every 500 steps under `model.eval()` + `torch.no_grad()`:
      * Appends structured JSON telemetry to `artifacts/logs/metrics.jsonl`.
      * Atomically overwrites `artifacts/logs/latest_probe.md` and appends to `probe_history.md`.
      * Emits 4 fixed observation sections:
          1. Core Vital-Signs Dashboard (single Markdown table incl. per-domain Val_BPC en/zh/ja/py/rs).
          2. 4-Domain SGSC `|` Spike Segmentation (with explicit `[UTF8-SPLIT!]` alarm), PMD decoding,
             efference-copy cosine error `eps_mon`, and Rust `tree-sitter` AST + bracket closure check.
          3. Restricted `K=8` Working-Memory Syntactic Merge Trajectory `(i*, j*) -> u` (first 12 steps).
          4. Zero-grad Live Wug Test Fast-Mapping Probe (EN plural + ZH rare-Hanzi) active from
             Step 500 through Step 100,000, reporting peak `ACh_m`, peak `beta_{m,u}`, exact spelling
             hit, and `wug_target_bpc`.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from data.ragged_dataloader import RaggedByteDataset
from dcpa.model import DCPAModel
from utils.device import amp_autocast, get_peak_memory_gb


LN2 = math.log(2.0)

DOMAIN_ID_TO_NAME: dict[int, str] = {
    0: "en",
    1: "zh",
    2: "ja",
    3: "py",
    4: "rs",
}

FIXED_DOMAIN_PROBES: tuple[tuple[str, int, str], ...] = (
    ("EN", 0, "The quick silver falcon glides silently over the alpine valley before dusk."),
    ("ZH", 1, "深度学习系统通过字节流感知与受限概念图实现高效的跨语言泛化。"),
    ("JA", 2, "静かな森の奥で青い鳥が澄んだ声で鳴き、朝の光が木々を照らしました。"),
    ("Rust", 4, "fn sum_nested(xs: &[i32]) -> i32 { let mut acc = 0; for &v in xs { if v > 0 { acc += v; } } acc }"),
)

FIXED_WUG_PROBES: tuple[dict[str, Any], ...] = (
    {
        "label": "EN-Plural-Wug",
        "lang_id": 0,
        "stem": "zorb",
        "target": "zorbs",
        "turn1": "A zorb is a tiny nocturnal desert bird.\n",
        "turn2_prefix": "Now looking under the cactus, there are two ",
        "turn2_suffix": ".\n",
    },
    {
        "label": "ZH-RareHanzi-Wug",
        "lang_id": 1,
        "stem": "鹀晶",
        "target": "鹀晶",
        "turn1": "记录显示，鹀晶是一种在极低温下呈现深紫色光泽的超导矿石。\n",
        "turn2_prefix": "实验室里，研究员将三块",
        "turn2_suffix": "缓缓放进液氮杜瓦瓶。\n",
    },
)


@dataclass
class ProbeResult:
    step: int
    stage_id: int
    train_bpc: float
    val_bpc_overall: Optional[float]
    val_bpc_by_domain: dict[str, Optional[float]]
    wug_target_bpc: float
    wug_exact_hits: int
    wug_total: int
    markdown: str
    record: dict[str, Any]


def format_console_header() -> str:
    return (
        f"{'Step':>8} | {'Stg':>3} | {'Loss':>7} | {'BPC':>6} | {'PC':>6} | {'rho':>5} | "
        f"{'Std(m/mn)':>11} | {'Ent':>5} | {'Skip%':>6} | {'L_m':>4} | {'GN':>6} | {'LR':>8} | {'MemGB':>5} | {'KB/s':>7}"
    )


def format_console_row(
    *,
    step: int,
    stage_id: int,
    metrics: Mapping[str, float],
    lr: float,
    mem_gb: float,
    kb_per_sec: float,
) -> str:
    loss_tot = float(metrics.get("loss_total", 0.0))
    bpc = float(metrics.get("loss_lang", 0.0)) / LN2
    pc_raw = float(metrics.get("loss_pc_raw", metrics.get("loss_pc", 0.0)))
    rho = float(metrics.get("compression_ratio", 0.0))
    std_m = float(metrics.get("c_slow_std_mean", 0.0))
    std_mn = float(metrics.get("c_slow_std_min", 0.0))
    ent = float(metrics.get("merge_entropy_nats", 0.0))
    skip_pct = 100.0 * float(metrics.get("ponder_skip_rate", 0.0))
    avg_l = float(metrics.get("avg_ponder_steps", 0.0))
    gn = float(metrics.get("grad_norm", 0.0))
    gn_str = f"{gn:6.2f}" if gn < 1000.0 else f"{gn:6.1e}"
    return (
        f"{step:8d} | S{stage_id:<2d} | {loss_tot:7.4f} | {bpc:6.3f} | {pc_raw:6.3f} | {rho:5.2f} | "
        f"{std_m:5.2f}/{std_mn:5.2f} | {ent:5.2f} | {skip_pct:5.1f}% | {avg_l:4.2f} | "
        f"{gn_str} | {lr:8.2e} | {mem_gb:5.2f} | {kb_per_sec:7.1f}"
    )


def format_sgsc_segmentation(raw_bytes: bytes, cu_seqlens: list[int]) -> tuple[str, int]:
    """Render SGSC segment boundaries with `|` and flag any invalid UTF-8 split with `[UTF8-SPLIT!]`."""
    parts: list[str] = []
    utf8_splits = 0
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        chunk = raw_bytes[s:e]
        try:
            txt = chunk.decode("utf-8", errors="strict")
            parts.append(txt.replace("\n", "\\n"))
        except UnicodeDecodeError:
            utf8_splits += 1
            safe_txt = chunk.decode("utf-8", errors="replace").replace("\n", "\\n")
            parts.append(f"[UTF8-SPLIT!]{safe_txt}")
    return " | ".join(parts), utf8_splits


def _check_brackets_balanced(text: str) -> tuple[bool, int]:
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
    stack: list[str] = []
    max_depth = 0
    for ch in text:
        if ch in opens:
            stack.append(ch)
            max_depth = max(max_depth, len(stack))
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False, max_depth
            stack.pop()
    return len(stack) == 0, max_depth


_RUST_PARSER_CACHE: Any = None


def _check_rust_treesitter(code_bytes: bytes) -> tuple[bool, str]:
    """Check Rust AST syntax validity via `tree-sitter-rust` (cached parser)."""
    global _RUST_PARSER_CACHE
    try:
        if _RUST_PARSER_CACHE is None:
            import tree_sitter_rust as tsrust
            from tree_sitter import Language, Parser

            lang = Language(tsrust.language())
            parser = Parser(lang)
            _RUST_PARSER_CACHE = parser
        tree = _RUST_PARSER_CACHE.parse(code_bytes)
        root = tree.root_node
        has_err = bool(root.has_error)
        return (not has_err), ("AST_OK" if not has_err else "AST_ERR")
    except Exception as exc:
        return False, f"AST_UNAVAILABLE({exc.__class__.__name__})"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


@torch.no_grad()
def evaluate_validation_bpc_by_domain(
    model: DCPAModel,
    val_bin_path: Optional[Path],
    val_idx_path: Optional[Path],
    *,
    device: torch.device,
    use_bf16: bool,
    model_stage: str,
    tau_s: float,
    tau_m: float,
    max_doc_bytes: int,
    val_budget_bytes: int = 2048,
    val_batches_per_domain: int = 2,
) -> tuple[Optional[float], dict[str, Optional[float]]]:
    """Evaluate per-domain `Val_BPC` across `en/zh/ja/py/rs` on `val.bin`/`val.idx`."""
    per_domain: dict[str, Optional[float]] = {name: None for name in DOMAIN_ID_TO_NAME.values()}
    if val_bin_path is None or val_idx_path is None or not val_bin_path.exists() or not val_idx_path.exists():
        return None, per_domain

    vals_collected: list[float] = []
    for lid, dname in DOMAIN_ID_TO_NAME.items():
        try:
            ds = RaggedByteDataset(
                bin_path=val_bin_path,
                idx_path=val_idx_path,
                target_batch_bytes=max(256, int(val_budget_bytes)),
                max_doc_bytes=min(int(max_doc_bytes), max(256, int(val_budget_bytes))),
                domain_weights={lid: 1.0},
                wug_ratio=0.0,
                num_batches=val_batches_per_domain,
                seed=777 + lid * 31,
            )
        except ValueError:
            continue

        total_nll = 0.0
        total_bytes = 0
        try:
            for b_idx in range(val_batches_per_domain):
                batch = ds[b_idx]
                x_b = batch["x_bytes"].unsqueeze(0).to(device)
                cu = batch["cu_seqlens"].to(device)
                lids = batch["doc_lang_ids"].to(device)
                if x_b.numel() == 0:
                    continue
                with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(
                        raw_bytes=x_b,
                        lang_ids=lids,
                        stage=model_stage,
                        tau_s=tau_s,
                        tau_m=tau_m,
                        use_broca_cq=False,
                        doc_cu_seqlens=cu,
                    )
                n_bytes = int(x_b.numel())
                total_nll += float(out.loss_lang.detach().item()) * n_bytes
                total_bytes += n_bytes
        finally:
            ds.close()

        if total_bytes > 0:
            bpc = (total_nll / float(total_bytes)) / LN2
            per_domain[dname] = bpc
            vals_collected.append(bpc)

    overall = (sum(vals_collected) / len(vals_collected)) if vals_collected else None
    return overall, per_domain


@torch.no_grad()
def _run_domain_probes(
    model: DCPAModel,
    *,
    device: torch.device,
    use_bf16: bool,
    model_stage: str,
    tau_s: float,
    tau_m: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run 4 fixed domain probes: SGSC `|` split, PMD decode, `eps_mon`, Rust AST, & K=8 merge trace."""
    domain_rows: list[dict[str, Any]] = []
    merge_lines: list[str] = []

    for label, lid, text in FIXED_DOMAIN_PROBES:
        raw_b = text.encode("utf-8")
        x_t = torch.tensor(list(raw_b), dtype=torch.uint8, device=device).unsqueeze(0)
        lang_t = torch.tensor([lid], dtype=torch.long, device=device)

        with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                raw_bytes=x_t,
                lang_ids=lang_t,
                stage=model_stage,
                tau_s=tau_s,
                tau_m=tau_m,
                use_broca_cq=False,
            )

        cu = out.sgsc.cu_seqlens.detach().cpu().tolist()
        seg_str, utf8_splits = format_sgsc_segmentation(raw_b, cu)

        pred_ids = out.pmd.byte_logits[0].float().argmax(dim=-1).to(torch.uint8)
        pred_bytes = bytes(pred_ids.detach().cpu().tolist())
        decoded_str = pred_bytes.decode("utf-8", errors="replace").replace("\n", "\\n")

        # Efference-copy self-monitoring cosine error eps_mon between original and re-encoded concepts
        with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
            re_sgsc = model.sgsc(pred_ids.unsqueeze(0), tau_s=tau_s)
        c_orig = out.sgsc.c_slow.float().mean(dim=0, keepdim=True)
        c_re = re_sgsc.c_slow.float().mean(dim=0, keepdim=True)
        cos_sim = float(F.cosine_similarity(c_orig, c_re, dim=-1, eps=1e-8).item())
        eps_mon = max(0.0, 1.0 - cos_sim)

        rust_note = ""
        if label == "Rust":
            ast_ok, ast_status = _check_rust_treesitter(pred_bytes)
            bal_ok, max_d = _check_brackets_balanced(decoded_str)
            rust_note = f" | `{ast_status}` | Brackets=`{'BALANCED' if bal_ok else 'UNBALANCED'}` (depth={max_d})"

        domain_rows.append(
            {
                "label": label,
                "num_bytes": len(raw_b),
                "num_segments": len(cu) - 1,
                "utf8_splits": utf8_splits,
                "segmented": seg_str,
                "decoded": decoded_str,
                "eps_mon": eps_mon,
                "rust_note": rust_note,
            }
        )

        # Collect K=8 syntactic merge snapshot (first 12 concept steps) on EN & ZH
        if label in ("EN", "ZH"):
            if out.win_pair_trace is not None and out.win_pair_trace.numel() > 0:
                pairs = out.win_pair_trace.detach().cpu().tolist()[:12]
                seg_labels = [
                    raw_b[cu[k] : cu[k + 1]].decode("utf-8", errors="replace").replace("\n", "\\n")
                    for k in range(min(len(pairs), len(cu) - 1))
                ]
                steps_txt = ", ".join(
                    f"m{m}[{seg_labels[m]!r}]:({i},{j})->u{u}"
                    for m, (i, j, u) in enumerate(pairs)
                )
                merge_lines.append(f"- **{label}** (`M={len(cu) - 1}`, showing first {len(pairs)}): {steps_txt}")
            else:
                merge_lines.append(
                    f"- **{label}**: BCG-PC bypassed in Stage 1 (`z_m* = c_m^slow`, `g_{{m-1}} = EMA(c_<m^slow)`)."
                )

    return domain_rows, merge_lines


@torch.no_grad()
def _run_live_wug_probes(
    model: DCPAModel,
    *,
    device: torch.device,
    use_bf16: bool,
    tau_s: float,
    tau_m: float,
) -> tuple[list[dict[str, Any]], float, int]:
    """Run zero-grad Wug Test fast-mapping probes (active across Step 500..100,000!)."""
    results: list[dict[str, Any]] = []
    bpc_vals: list[float] = []
    exact_hits = 0

    for spec in FIXED_WUG_PROBES:
        t1_b = spec["turn1"].encode("utf-8")
        pre_b = spec["turn2_prefix"].encode("utf-8")
        tgt_b = spec["target"].encode("utf-8")
        suf_b = spec["turn2_suffix"].encode("utf-8")
        full_b = t1_b + pre_b + tgt_b + suf_b

        tgt_start = len(t1_b) + len(pre_b)
        tgt_end = tgt_start + len(tgt_b)
        x_t = torch.tensor(list(full_b), dtype=torch.uint8, device=device).unsqueeze(0)
        lang_t = torch.tensor([int(spec["lang_id"])], dtype=torch.long, device=device)

        # Isolated zero-initialized temporary fast weights (NEVER mutates model buffers!)
        temp_fast = model.hippocampus.init_fast_state(1, mode="column_major", storage_dtype=torch.float32, device=device)

        with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                raw_bytes=x_t,
                lang_ids=lang_t,
                stage="D",
                tau_s=tau_s,
                tau_m=tau_m,
                use_broca_cq=False,
                fast_state=temp_fast,
                write_threshold_override=0.0,
            )

        # Peak ACh_m & peak beta_{m,u} on Turn 1 support segments
        cu = out.sgsc.cu_seqlens.detach().cpu().tolist()
        t1_seg_count = sum(1 for k in range(len(cu) - 1) if cu[k] < len(t1_b))
        if out.ach_trace is not None and t1_seg_count > 0:
            peak_ach = float(out.ach_trace[:t1_seg_count].max().item())
        else:
            peak_ach = float(out.metrics.get("mean_ach", 0.0))
        if out.beta_max_trace is not None and t1_seg_count > 0:
            peak_beta = float(out.beta_max_trace[:t1_seg_count].max().item())
        else:
            peak_beta = float(out.metrics.get("mean_beta", 0.0))

        # Evaluate target pseudo-word BPC & greedy exact spelling match in Turn 2
        tgt_logits = out.pmd.byte_logits[0, tgt_start:tgt_end, :].float()
        tgt_gold = x_t[0, tgt_start:tgt_end].long()
        tgt_ce = float(F.cross_entropy(tgt_logits, tgt_gold).item())
        tgt_bpc = tgt_ce / LN2
        bpc_vals.append(tgt_bpc)

        pred_bytes = bytes(tgt_logits.argmax(dim=-1).to(torch.uint8).detach().cpu().tolist())
        pred_text = pred_bytes.decode("utf-8", errors="replace").replace("\n", "\\n")
        hit = pred_bytes == tgt_b
        if hit:
            exact_hits += 1

        results.append(
            {
                "label": spec["label"],
                "stem": spec["stem"],
                "expected": spec["target"],
                "predicted": pred_text,
                "exact_hit": hit,
                "peak_ach": peak_ach,
                "peak_beta": peak_beta,
                "wug_target_bpc": tgt_bpc,
            }
        )

    mean_wug_bpc = sum(bpc_vals) / max(len(bpc_vals), 1)
    return results, mean_wug_bpc, exact_hits


@torch.no_grad()
def run_visual_probe(
    model: DCPAModel,
    *,
    step: int,
    total_steps: int,
    stage_id: int,
    model_stage: str,
    tau_s: float,
    tau_m: float,
    lr: float,
    train_metrics: Mapping[str, float],
    device: torch.device,
    use_bf16: bool,
    kb_per_sec: float,
    max_doc_bytes: int,
    val_bin_path: Optional[Path] = None,
    val_idx_path: Optional[Path] = None,
    logs_dir: str | Path = "artifacts/logs",
    val_budget_bytes: int = 2048,
    val_batches_per_domain: int = 2,
    watchdog_alerts: Optional[list[str]] = None,
) -> ProbeResult:
    """Execute full 4-section zero-grad diagnostic probe and persist JSONL + Markdown artifacts."""
    was_training = model.training
    model.eval()
    try:
        train_bpc = float(train_metrics.get("loss_lang", 0.0)) / LN2
        val_overall, val_by_dom = evaluate_validation_bpc_by_domain(
            model,
            val_bin_path=val_bin_path,
            val_idx_path=val_idx_path,
            device=device,
            use_bf16=use_bf16,
            model_stage=model_stage,
            tau_s=tau_s,
            tau_m=tau_m,
            max_doc_bytes=max_doc_bytes,
            val_budget_bytes=val_budget_bytes,
            val_batches_per_domain=val_batches_per_domain,
        )
        domain_rows, merge_lines = _run_domain_probes(
            model,
            device=device,
            use_bf16=use_bf16,
            model_stage=model_stage,
            tau_s=tau_s,
            tau_m=tau_m,
        )
        wug_rows, wug_bpc, wug_hits = _run_live_wug_probes(
            model,
            device=device,
            use_bf16=use_bf16,
            tau_s=tau_s,
            tau_m=tau_m,
        )
    finally:
        if was_training:
            model.train()

    mem_gb = get_peak_memory_gb(device)
    raw_pc_nll = float(train_metrics.get("loss_pc_raw", train_metrics.get("loss_pc", 0.0)))
    rho = float(train_metrics.get("compression_ratio", 0.0))
    std_mean = float(train_metrics.get("c_slow_std_mean", 0.0))
    std_min = float(train_metrics.get("c_slow_std_min", 0.0))
    merge_ent = float(train_metrics.get("merge_entropy_nats", 0.0))
    skip_pct = 100.0 * float(train_metrics.get("ponder_skip_rate", 0.0))
    avg_l = float(train_metrics.get("avg_ponder_steps", 0.0))

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    alerts_txt = ", ".join(watchdog_alerts) if watchdog_alerts else "None"
    decode_mode_note = (
        "SGSC + PMD Autoencoder Efference Copy"
        if model_stage.upper() == "A"
        else "SGSC + PMD Strictly Causal Next-Byte Prediction"
    )

    lines: list[str] = [
        f"# DCPA Visual Diagnostic Probe — Step {step:,} / {total_steps:,} (Stage {stage_id} / {model_stage})",
        "",
        f"- **UTC**: `{ts_utc}` | **LR**: `{lr:.2e}` | **tau_s**: `{tau_s:.3f}` | **tau_m**: `{tau_m:.3f}` | **Watchdog Alerts**: `{alerts_txt}`",
        "",
        "## 1. 核心生命体征仪表盘 (Vital Signs)",
        "",
        "| Step / Stage | Train_BPC | Val_BPC (en / zh / ja / py / rs) | Raw_PC_NLL | ρ = T/M | Concept_Std (Mean / Min) | Merge_Entropy | Ponder_Skip_% | Avg_L_m | VRAM_GB | KB/s |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{step:,} / S{stage_id}({model_stage})` | `{train_bpc:.3f}` | "
            f"`{_fmt(val_by_dom['en'])} / {_fmt(val_by_dom['zh'])} / {_fmt(val_by_dom['ja'])} / "
            f"{_fmt(val_by_dom['py'])} / {_fmt(val_by_dom['rs'])}` (avg `{_fmt(val_overall)}`) | "
            f"`{raw_pc_nll:.3f}` | `{rho:.2f}x` | `{std_mean:.3f} / {std_min:.3f}` | `{merge_ent:.3f}` | "
            f"`{skip_pct:.1f}%` | `{avg_l:.2f}` | `{mem_gb:.2f}` | `{kb_per_sec:.1f}` |"
        ),
        "",
        f"## 2. 四语域脉冲切分 `|` 与解码对比 ({decode_mode_note})",
        "",
    ]

    for row in domain_rows:
        split_flag = "OK (0 UTF8 splits)" if row["utf8_splits"] == 0 else f"ALARM ({row['utf8_splits']} UTF8 splits!)"
        lines.extend(
            [
                f"- **{row['label']}** (`T={row['num_bytes']}B -> M={row['num_segments']}`, UTF-8=`{split_flag}`, `eps_mon={row['eps_mon']:.4f}`{row['rust_note']}):",
                f"  - SGSC `|`: `{row['segmented']}`",
                f"  - PMD Decoded: `{row['decoded']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 受限工作记忆 K=8 句法归并快照 (Merge Trajectory, First 12 Steps)",
            "",
            *merge_lines,
            "",
            "## 4. 全程零梯度 Wug Test 实时快映射探针 (Zero-Grad Hippocampal Fast-Mapping)",
            "",
            "| Probe | Expected | Predicted | Exact Hit | Peak ACh_m | Peak beta_{m,u} | wug_target_bpc |",
            "| :--- | :--- | :--- | :---: | ---: | ---: | ---: |",
        ]
    )

    for wr in wug_rows:
        hit_s = "HIT" if wr["exact_hit"] else "MISS"
        lines.append(
            f"| `{wr['label']}` | `{wr['expected']}` | `{wr['predicted']}` | `{hit_s}` | "
            f"`{wr['peak_ach']:.4f}` | `{wr['peak_beta']:.4f}` | `{wr['wug_target_bpc']:.3f}` |"
        )
    lines.append("")

    markdown = "\n".join(lines)

    record: dict[str, Any] = {
        "timestamp_utc": ts_utc,
        "step": int(step),
        "stage_id": int(stage_id),
        "model_stage": model_stage,
        "lr": float(lr),
        "tau_s": float(tau_s),
        "tau_m": float(tau_m),
        "train_bpc": round(train_bpc, 5),
        "val_bpc_overall": None if val_overall is None else round(val_overall, 5),
        "val_bpc_by_domain": {
            k: (None if v is None else round(v, 5)) for k, v in val_by_dom.items()
        },
        "wug_target_bpc": round(wug_bpc, 5),
        "wug_exact_hits": int(wug_hits),
        "wug_total": len(wug_rows),
        "vram_gb": round(mem_gb, 3),
        "kb_per_sec": round(kb_per_sec, 2),
        "utf8_split_alarms": sum(int(r["utf8_splits"]) for r in domain_rows),
        "eps_mon_by_domain": {r["label"]: round(float(r["eps_mon"]), 5) for r in domain_rows},
        **{k: float(v) for k, v in train_metrics.items()},
    }

    out_dir = Path(logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _append_text_fsync(out_dir / "metrics.jsonl", json.dumps(record, ensure_ascii=False) + "\n")
    _atomic_write_text(out_dir / "latest_probe.md", markdown)
    _append_text_fsync(out_dir / "probe_history.md", f"\n---\n\n{markdown}\n")

    return ProbeResult(
        step=step,
        stage_id=stage_id,
        train_bpc=train_bpc,
        val_bpc_overall=val_overall,
        val_bpc_by_domain=val_by_dom,
        wug_target_bpc=wug_bpc,
        wug_exact_hits=wug_hits,
        wug_total=len(wug_rows),
        markdown=markdown,
        record=record,
    )
