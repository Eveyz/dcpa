#!/usr/bin/env python3
"""Lightweight GPU throughput autotuner for DCPA-Turbo (`train/train_dcpa.py`).

Benchmarks mathematically equivalent hardware execution configurations (fixed
`--target-tokens-bytes-per-step`) in isolated subprocesses so OOM candidates fail
cleanly without crashing the sweep, then ranks candidates by steady-state median
`KB/s` (excluding initial TorchInductor compile/warmup steps) and emits ready-to-paste
CLI flags + Colab `EXTRA_TRAIN_ARGS` + JSON report.

Example (Colab L4 / A100 / G4):
  python3 scripts/autotune_gpu.py \
    --config configs/v2/dcpa_210m_v2_cognitive.yaml \
    --preset moe_210m \
    --data-dir /content/nvme_cache/data_v2_cognitive_12x \
    --cap-data-dir /content/nvme_cache/cap_v2_cognitive \
    --stage 2 --steps 14 --warmup-steps 5

Preview candidate grid & commands without executing:
  python3 scripts/autotune_gpu.py --config configs/v2/dcpa_210m_v2_cognitive.yaml --preset moe_210m --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.device import detect_hardware_profile


_CONSOLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(S\d)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*\|"
    r".*?\|\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class AutotuneCandidate:
    """Single hardware execution configuration to benchmark."""

    micro_batch_bytes: int
    bcg_chunk_steps: int
    bcg_ckpt_watermark_gb: float
    bcg_cudagraphs: bool
    stream_chunk_bytes: int = 512
    compile_mode: str = "auto"

    def label(self) -> str:
        cg = "cg1" if self.bcg_cudagraphs else "cg0"
        return (
            f"mb={self.micro_batch_bytes // 1024}K_"
            f"chk={self.bcg_chunk_steps}_"
            f"wm={self.bcg_ckpt_watermark_gb:.1f}G_"
            f"{cg}_sc={self.stream_chunk_bytes}"
        )

    def to_cli_args(self, target_step_bytes: int) -> list[str]:
        return [
            "--target-tokens-bytes-per-step",
            str(int(target_step_bytes)),
            "--micro-batch-bytes",
            str(int(self.micro_batch_bytes)),
            "--bcg-chunk-steps",
            str(int(self.bcg_chunk_steps)),
            "--bcg-ckpt-watermark-gb",
            f"{float(self.bcg_ckpt_watermark_gb):.1f}",
            "--bcg-cudagraphs" if self.bcg_cudagraphs else "--no-bcg-cudagraphs",
            "--stream-chunk-bytes",
            str(int(self.stream_chunk_bytes)),
            "--compile-mode",
            str(self.compile_mode),
        ]

    def to_extra_train_args(self, target_step_bytes: int) -> str:
        return shlex.join(self.to_cli_args(target_step_bytes))


@dataclass
class AutotuneTrialResult:
    candidate: AutotuneCandidate
    status: str  # "ok" | "oom" | "timeout" | "error" | "dry_run"
    steady_kb_per_sec: float = 0.0
    mean_kb_per_sec: float = 0.0
    peak_mem_gb: float = 0.0
    wall_seconds: float = 0.0
    steady_steps_measured: int = 0
    exit_code: int = 0
    error_summary: str = ""
    command: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.command is None:
            self.command = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "label": self.candidate.label(),
            "status": self.status,
            "steady_kb_per_sec": round(self.steady_kb_per_sec, 2),
            "mean_kb_per_sec": round(self.mean_kb_per_sec, 2),
            "peak_mem_gb": round(self.peak_mem_gb, 2),
            "wall_seconds": round(self.wall_seconds, 2),
            "steady_steps_measured": int(self.steady_steps_measured),
            "exit_code": int(self.exit_code),
            "error_summary": self.error_summary,
            "command": list(self.command),
        }


def _parse_int_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_candidate_grid(
    *,
    vram_gb: float,
    target_step_bytes: int = 131_072,
    micro_bytes_list: Optional[Sequence[int]] = None,
    bcg_chunks: Sequence[int] = (8, 16),
    stream_chunks: Sequence[int] = (512,),
    cudagraphs_mode: str = "auto",
    watermarks: Optional[Sequence[float]] = None,
    compile_mode: str = "auto",
    max_trials: int = 12,
) -> list[AutotuneCandidate]:
    """Build a pruned, VRAM-safe hardware candidate grid ordered by expected throughput."""
    if micro_bytes_list is None:
        if vram_gb <= 0.0:
            micros = [min(8192, target_step_bytes)]
        elif vram_gb < 18.0:
            micros = [16_384, 32_768]
        elif vram_gb < 36.0:
            micros = [65_536, 32_768, 131_072]
        elif vram_gb < 72.0:
            micros = [65_536, 131_072]
        else:
            micros = [131_072, 65_536]
    else:
        micros = list(micro_bytes_list)
    micros = [m for m in micros if 64 <= m <= target_step_bytes]
    if not micros:
        micros = [min(32_768, target_step_bytes)]

    if watermarks is None:
        if vram_gb < 18.0:
            wms = [0.0]
        elif vram_gb < 36.0:
            wms = [round(0.65 * vram_gb, 1), 0.0]
        else:
            wms = [round(0.75 * vram_gb, 1), round(0.62 * vram_gb, 1)]
    else:
        wms = [max(0.0, float(w)) for w in watermarks]
    if vram_gb > 0.0:
        wms = [w for w in wms if w <= round(0.88 * vram_gb, 1)]
        if not wms:
            wms = [0.0]

    cg_mode = cudagraphs_mode.strip().lower()
    if cg_mode == "true":
        cg_opts = [True]
    elif cg_mode == "false":
        cg_opts = [False]
    else:
        cg_opts = [True, False] if vram_gb >= 18.0 else [False]

    seen: set[AutotuneCandidate] = set()
    candidates: list[AutotuneCandidate] = []
    for mb in micros:
        for wm in wms:
            for cg in cg_opts:
                # When watermark == 0.0, checkpointing always engages and bypasses CUDA Graphs;
                # skip duplicate cg=True trials to save sweep time.
                if wm <= 0.0 and cg:
                    continue
                for bc in bcg_chunks:
                    for sc in stream_chunks:
                        cand = AutotuneCandidate(
                            micro_batch_bytes=int(mb),
                            bcg_chunk_steps=int(bc),
                            bcg_ckpt_watermark_gb=float(wm),
                            bcg_cudagraphs=bool(cg),
                            stream_chunk_bytes=int(sc),
                            compile_mode=str(compile_mode),
                        )
                        if cand not in seen:
                            seen.add(cand)
                            candidates.append(cand)
    return candidates[: max(1, int(max_trials))]


def stage_boundaries_for_probe(target_stage: int, total_steps: int) -> tuple[int, int, int, int, int]:
    """Return 5 curriculum boundaries so probe steps land cleanly in `target_stage` (2 or 3)."""
    st = int(target_stage)
    n = max(4, int(total_steps))
    if st == 1:
        return (0, n, n + 1, n + 2, n + 3)
    if st == 2:
        # Step 1 runs Stage 1, steps 2..n run Stage 2 (B)
        return (0, 1, n + 1, n + 2, n + 3)
    if st == 3:
        # Step 1 = S1, Step 2 = S2, Steps 3..n = Stage 3 (C)
        return (0, 1, 2, n + 1, n + 2)
    raise ValueError(f"Unsupported probe --stage={target_stage}; expected 1, 2, or 3")


def parse_trial_log(
    log_text: str,
    *,
    target_stage: int,
    warmup_steps: int,
) -> tuple[float, float, float, int]:
    """Extract `(steady_median_kb_s, steady_mean_kb_s, peak_mem_gb, n_steady)` from trainer stdout."""
    want_stg = f"S{int(target_stage)}"
    stage_kbs: list[float] = []
    peak_mem = 0.0
    for line in log_text.splitlines():
        m = _CONSOLE_ROW_RE.match(line)
        if not m:
            continue
        stg = m.group(2)
        mem_gb = float(m.group(5))
        kb_s = float(m.group(6))
        peak_mem = max(peak_mem, mem_gb)
        if stg == want_stg:
            stage_kbs.append(kb_s)

    steady = stage_kbs[int(warmup_steps) :] if len(stage_kbs) > int(warmup_steps) else stage_kbs[-1:]
    if not steady:
        return 0.0, 0.0, peak_mem, 0
    return float(statistics.median(steady)), float(statistics.mean(steady)), peak_mem, len(steady)


def build_trial_command(
    candidate: AutotuneCandidate,
    *,
    python_bin: str,
    config_path: str,
    preset: str,
    data_dir: str,
    cap_data_dir: Optional[str],
    local_cache_dir: str,
    trial_dir: Path,
    target_step_bytes: int,
    target_stage: int,
    steps: int,
    device_str: str,
    extra_args: Sequence[str] = (),
) -> list[str]:
    bounds = stage_boundaries_for_probe(target_stage, steps)
    cmd = [
        python_bin,
        "-u",
        str(_REPO_ROOT / "train" / "train_dcpa.py"),
        "--gpus",
        "1",
        "--device",
        device_str,
        "--config",
        str(config_path),
        "--preset",
        str(preset),
        "--total-steps",
        str(int(steps)),
        "--stage-boundaries",
        ",".join(str(b) for b in bounds),
        "--data-dir",
        str(data_dir),
        "--local-cache-dir",
        str(local_cache_dir),
        "--ckpt-dir",
        str(trial_dir / "ckpts"),
        "--logs-dir",
        str(trial_dir / "logs"),
        "--export-dir",
        str(trial_dir / "export"),
        "--resume",
        "none",
        "--no-export",
        "--no-wandb",
        "--save-every-steps",
        "999999",
        "--save-every-minutes",
        "9999",
        "--probe-every-steps",
        "999999",
        "--console-every-steps",
        "1",
        "--max-consecutive-meltdowns",
        "3",
        *candidate.to_cli_args(target_step_bytes),
        *list(extra_args),
    ]
    if cap_data_dir:
        cmd.extend(["--cap-data-dir", str(cap_data_dir)])
    return cmd


def _detect_oom(output_text: str) -> bool:
    low = output_text.lower()
    return (
        "cuda out of memory" in low
        or "outofmemoryerror" in low
        or "hip out of memory" in low
        or "cublas_status_alloc_failed" in low
    )


def format_leaderboard(results: Sequence[AutotuneTrialResult]) -> str:
    header = (
        f"{'Rank':>4} | {'Candidate':<36} | {'Status':<7} | {'Steady KB/s':>11} | "
        f"{'Peak GB':>7} | {'Wall(s)':>7} | {'N':>3}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for rank, r in enumerate(results, start=1):
        lines.append(
            f"{rank:>4} | {r.candidate.label():<36} | {r.status:<7} | "
            f"{r.steady_kb_per_sec:>11.1f} | {r.peak_mem_gb:>7.2f} | "
            f"{r.wall_seconds:>7.1f} | {r.steady_steps_measured:>3d}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Lightweight DCPA-Turbo GPU throughput autotuner.")
    ap.add_argument("--config", type=str, default="configs/v2/dcpa_210m_v2_cognitive.yaml")
    ap.add_argument("--preset", type=str, default="moe_210m")
    ap.add_argument("--data-dir", type=str, default="artifacts/data")
    ap.add_argument("--cap-data-dir", type=str, default=None)
    ap.add_argument("--local-cache-dir", type=str, default="artifacts/nvme_cache")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=2, help="Target curriculum stage to benchmark (default: 2).")
    ap.add_argument("--steps", type=int, default=14, help="Total steps per trial (includes warmup).")
    ap.add_argument("--warmup-steps", type=int, default=5, help="Initial target-stage steps excluded from steady KB/s.")
    ap.add_argument("--target-step-bytes", type=int, default=131_072, help="Locked effective step bytes across candidates.")
    ap.add_argument("--micro-bytes", type=str, default=None, help="Comma-separated micro_batch_bytes candidates (e.g. 32768,65536,131072).")
    ap.add_argument("--bcg-chunks", type=str, default="8,16", help="Comma-separated bcg_chunk_steps candidates.")
    ap.add_argument("--stream-chunks", type=str, default="512", help="Comma-separated stream_chunk_bytes candidates.")
    ap.add_argument("--cudagraphs", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--watermarks", type=str, default=None, help="Comma-separated bcg_ckpt_watermark_gb values (default: auto from VRAM).")
    ap.add_argument("--compile-mode", choices=["none", "auto", "reduce-overhead"], default="auto")
    ap.add_argument("--max-trials", type=int, default=12)
    ap.add_argument("--timeout-seconds", type=int, default=600)
    ap.add_argument("--vram-gb-override", type=float, default=None, help="Simulate a GPU VRAM size (useful with --dry-run).")
    ap.add_argument("--out-json", type=str, default="artifacts/autotune_gpu_result.json")
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra flag forwarded to every trial.")
    ap.add_argument("--dry-run", action="store_true", help="Print candidate grid & commands without launching trials.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prof = detect_hardware_profile(preferred=args.device)
    eff_vram = float(args.vram_gb_override) if args.vram_gb_override is not None else float(prof.vram_gb)

    micro_list = _parse_int_csv(args.micro_bytes) if args.micro_bytes else None
    bcg_chunks = _parse_int_csv(args.bcg_chunks)
    stream_chunks = _parse_int_csv(args.stream_chunks)
    wm_list = _parse_float_csv(args.watermarks) if args.watermarks else None

    candidates = build_candidate_grid(
        vram_gb=eff_vram,
        target_step_bytes=args.target_step_bytes,
        micro_bytes_list=micro_list,
        bcg_chunks=bcg_chunks,
        stream_chunks=stream_chunks,
        cudagraphs_mode=args.cudagraphs,
        watermarks=wm_list,
        compile_mode=args.compile_mode,
        max_trials=args.max_trials,
    )
    print(
        f"[Autotune] device={prof.device_name} ({eff_vram:.1f} GB VRAM) | preset={args.preset} | "
        f"stage=S{args.stage} | target_step_bytes={args.target_step_bytes:,} | "
        f"trials={len(candidates)} (steps={args.steps}, warmup={args.warmup_steps})",
        flush=True,
    )

    results: list[AutotuneTrialResult] = []
    with tempfile.TemporaryDirectory(prefix="dcpa_autotune_") as scratch_root:
        scratch_p = Path(scratch_root)
        for idx, cand in enumerate(candidates, start=1):
            t_dir = scratch_p / f"trial_{idx:02d}"
            cmd = build_trial_command(
                cand,
                python_bin=sys.executable,
                config_path=args.config,
                preset=args.preset,
                data_dir=args.data_dir,
                cap_data_dir=args.cap_data_dir,
                local_cache_dir=args.local_cache_dir,
                trial_dir=t_dir,
                target_step_bytes=args.target_step_bytes,
                target_stage=args.stage,
                steps=args.steps,
                device_str=args.device,
                extra_args=args.extra_arg,
            )
            if args.dry_run:
                print(f"  [{idx:02d}/{len(candidates):02d}] {cand.label()} -> {shlex.join(cmd)}")
                results.append(AutotuneTrialResult(candidate=cand, status="dry_run", command=cmd))
                continue

            print(f"\n[Autotune][{idx:02d}/{len(candidates):02d}] Running {cand.label()} ...", flush=True)
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(_REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=int(args.timeout_seconds),
                    check=False,
                )
                wall = time.monotonic() - t0
                out_txt = proc.stdout or ""
                if proc.returncode == 0:
                    med_kb, mean_kb, peak_gb, n_st = parse_trial_log(
                        out_txt, target_stage=args.stage, warmup_steps=args.warmup_steps
                    )
                    res = AutotuneTrialResult(
                        candidate=cand,
                        status="ok" if n_st > 0 else "error",
                        steady_kb_per_sec=med_kb,
                        mean_kb_per_sec=mean_kb,
                        peak_mem_gb=peak_gb,
                        wall_seconds=wall,
                        steady_steps_measured=n_st,
                        exit_code=proc.returncode,
                        error_summary="" if n_st > 0 else "no target-stage console rows parsed",
                        command=cmd,
                    )
                elif _detect_oom(out_txt):
                    res = AutotuneTrialResult(
                        candidate=cand,
                        status="oom",
                        wall_seconds=wall,
                        exit_code=proc.returncode,
                        error_summary="CUDA Out Of Memory",
                        command=cmd,
                    )
                else:
                    tail = " | ".join(line.strip() for line in out_txt.splitlines()[-4:] if line.strip())
                    res = AutotuneTrialResult(
                        candidate=cand,
                        status="error",
                        wall_seconds=wall,
                        exit_code=proc.returncode,
                        error_summary=tail[:240],
                        command=cmd,
                    )
            except subprocess.TimeoutExpired:
                wall = time.monotonic() - t0
                res = AutotuneTrialResult(
                    candidate=cand,
                    status="timeout",
                    wall_seconds=wall,
                    exit_code=-1,
                    error_summary=f"Timed out after {args.timeout_seconds}s",
                    command=cmd,
                )
            print(
                f"  -> status={res.status} | steady={res.steady_kb_per_sec:.1f} KB/s | "
                f"peak={res.peak_mem_gb:.2f} GB | wall={res.wall_seconds:.1f}s"
                + (f" | {res.error_summary}" if res.error_summary else ""),
                flush=True,
            )
            results.append(res)

    ranked = sorted(
        results,
        key=lambda r: (r.status == "ok", r.steady_kb_per_sec, -r.peak_mem_gb),
        reverse=True,
    )
    print("\n" + format_leaderboard(ranked), flush=True)

    best = ranked[0] if ranked and ranked[0].status == "ok" else None
    payload: dict[str, Any] = {
        "device_name": prof.device_name,
        "vram_gb": eff_vram,
        "preset": args.preset,
        "config": args.config,
        "stage": int(args.stage),
        "target_step_bytes": int(args.target_step_bytes),
        "dry_run": bool(args.dry_run),
        "best": best.to_dict() if best is not None else None,
        "recommended_cli_args": best.candidate.to_cli_args(args.target_step_bytes) if best is not None else [],
        "recommended_extra_train_args": (
            best.candidate.to_extra_train_args(args.target_step_bytes) if best is not None else ""
        ),
        "trials": [r.to_dict() for r in ranked],
    }

    if not args.dry_run and args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[Autotune] Saved report: {out_p}", flush=True)

    if best is not None:
        extra_str = best.candidate.to_extra_train_args(args.target_step_bytes)
        print(f"\n[Autotune][WINNER] {best.candidate.label()} ({best.steady_kb_per_sec:.1f} KB/s, {best.peak_mem_gb:.2f} GB)")
        print(f"  Colab Cell 1 -> EXTRA_TRAIN_ARGS = \"{extra_str}\"")
        print(f"  CLI flags    -> {extra_str}", flush=True)
        return 0
    return 0 if args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
