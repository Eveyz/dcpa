"""High-speed parallel corpus & capability shard orchestrator for Vast.ai / multi-GPU nodes.

Features:
  1. Concurrent execution: runs `build_corpus.py` (with 5 per-domain workers),
     `build_cap_reasoning.py`, `build_cap_writing.py`, `build_cap_toolcall.py`, and
     `build_sft_stage5.py` simultaneously to saturate 10 Gbps NICs and multi-core CPUs.
  2. High-speed HuggingFace Rust transport (`HF_HUB_ENABLE_HF_TRANSFER=1`,
     `HF_XET_HIGH_PERFORMANCE=1`) + NVMe FUSE/network-filesystem guardrails.
  3. Idempotent shard verification: checks exact 16-byte `<QIBBH` `.idx` alignment,
     `offset + length == .bin size`, and 1-to-1 `.loss_mask.bin` byte parity; skips
     already-valid targets automatically unless `--force` is passed.
  4. Fast `.tar.zst` snapshot restore (`--from-archive <url_or_path>`) via `aria2c`/`curl`
     + `zstd -T0`, and snapshot creation (`--pack-archive <path.tar.zst>`) for <30s cold
     starts across Vast.ai rentals.
  5. `--dry-run` mode for offline plan inspection without network traffic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.ragged_dataloader import IDX_RECORD_FMT, IDX_RECORD_SIZE


@dataclass(frozen=True)
class DataProfileSpec:
    name: str
    description: str
    smoke_quota: bool
    corpus_scale_factor: float
    stem_upgrade: bool
    corpus_workers: int
    cap_reasoning_mb: float
    cap_instruct_mb: float
    cap_writing_mb: float
    cap_toolcall_mb: float
    sft_train_mb: float
    sft_val_mb: float
    min_free_gb: float
    cognitive_v2: bool = False
    cap_code_mb: float = 0.0


PROFILES: dict[str, DataProfileSpec] = {
    "smoke": DataProfileSpec(
        name="smoke",
        description="Fast ~18 MB smoke shards across all 6 builders (EN/ZH/Python/JS + Capabilities).",
        smoke_quota=True,
        corpus_scale_factor=1.0,
        stem_upgrade=False,
        corpus_workers=2,
        cap_reasoning_mb=2.0,
        cap_instruct_mb=2.0,
        cap_writing_mb=2.0,
        cap_toolcall_mb=2.0,
        sft_train_mb=1.0,
        sft_val_mb=0.125,
        min_free_gb=2.0,
    ),
    "v1_fast": DataProfileSpec(
        name="v1_fast",
        description="Standard ~820 MB EN/ZH/Python/JS corpus + Reasoning/Instruct/Toolcall/Writing + SFT.",
        smoke_quota=False,
        corpus_scale_factor=1.0,
        stem_upgrade=False,
        corpus_workers=4,
        cap_reasoning_mb=100.0,
        cap_instruct_mb=100.0,
        cap_writing_mb=80.0,
        cap_toolcall_mb=80.0,
        sft_train_mb=24.0,
        sft_val_mb=1.0,
        min_free_gb=5.0,
    ),
    "v2_stem_10g": DataProfileSpec(
        name="v2_stem_10g",
        description="Recommended 4x A100 80GB 1-week corpus (~12.6 GB EN/ZH STEM + Python/JS AST + Reasoning/Instruct/Toolcall).",
        smoke_quota=False,
        corpus_scale_factor=24.0,
        stem_upgrade=True,
        corpus_workers=4,
        cap_reasoning_mb=700.0,
        cap_instruct_mb=600.0,
        cap_writing_mb=350.0,
        cap_toolcall_mb=350.0,
        sft_train_mb=140.0,
        sft_val_mb=4.0,
        min_free_gb=32.0,
    ),
    "v2_full_40g": DataProfileSpec(
        name="v2_full_40g",
        description="Large multi-node / multi-week corpus (~44 GB EN/ZH/Python/JS + full capabilities).",
        smoke_quota=False,
        corpus_scale_factor=90.0,
        stem_upgrade=True,
        corpus_workers=4,
        cap_reasoning_mb=1600.0,
        cap_instruct_mb=1400.0,
        cap_writing_mb=800.0,
        cap_toolcall_mb=800.0,
        sft_train_mb=280.0,
        sft_val_mb=8.0,
        min_free_gb=95.0,
    ),
    "v2_cognitive_smoke": DataProfileSpec(
        name="v2_cognitive_smoke",
        description="DCPA v2 (moe_210m) ~22 MB holistic 35:35:30 smoke shards (6-domain staged + Logic/Writing/Code alchemy, zero Rust).",
        smoke_quota=True,
        corpus_scale_factor=1.0,
        stem_upgrade=False,
        corpus_workers=2,
        cap_reasoning_mb=2.0,
        cap_instruct_mb=2.0,
        cap_writing_mb=2.0,
        cap_toolcall_mb=2.0,
        sft_train_mb=1.0,
        sft_val_mb=0.125,
        min_free_gb=2.0,
        cognitive_v2=True,
        cap_code_mb=2.0,
    ),
    "v2_cognitive_5g": DataProfileSpec(
        name="v2_cognitive_5g",
        description="Recommended single-GPU DCPA v2 (moe_210m) ~5.9 GB holistic 35:35:30 corpus (6-domain staged + SYNTH/Writing/Code alchemy + Stage 5 multi-draft, zero Rust).",
        smoke_quota=False,
        corpus_scale_factor=12.0,
        stem_upgrade=False,
        corpus_workers=6,
        cap_reasoning_mb=160.0,
        cap_instruct_mb=40.0,
        cap_writing_mb=160.0,
        cap_toolcall_mb=40.0,
        sft_train_mb=80.0,
        sft_val_mb=4.0,
        min_free_gb=14.0,
        cognitive_v2=True,
        cap_code_mb=120.0,
    ),
}


def detect_default_nvme_root() -> Path:
    """Select the fastest local NVMe mount on Vast.ai / RunPod / Lambda / local workstations."""
    env_override = os.environ.get("DCPA_NVME_ROOT", "").strip()
    if env_override:
        return Path(env_override)
    for candidate in ("/workspace/nvme_cache", "/raid/dcpa_nvme", "/nvme/dcpa_cache"):
        parent = Path(candidate).parent
        if parent.exists() and os.access(parent, os.W_OK):
            return Path(candidate)
    return _REPO_ROOT / "artifacts"


def verify_shard_triplet(
    bin_path: Path,
    idx_path: Path,
    mask_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """Verify `.bin`, `.idx` (`<QIBBH`), and optional `.loss_mask.bin` zero-copy invariants."""
    if not bin_path.is_file() or not idx_path.is_file():
        return False, f"missing {bin_path.name} or {idx_path.name}"
    bin_size = bin_path.stat().st_size
    idx_size = idx_path.stat().st_size
    if bin_size <= 0 or idx_size <= 0:
        return False, f"empty shard ({bin_path.name}={bin_size}B, {idx_path.name}={idx_size}B)"
    if idx_size % IDX_RECORD_SIZE != 0:
        return False, f"{idx_path.name} size {idx_size} not divisible by {IDX_RECORD_SIZE}"

    with idx_path.open("rb") as f:
        f.seek(idx_size - IDX_RECORD_SIZE)
        last_rec = f.read(IDX_RECORD_SIZE)
    last_off, last_len, _, _, _, _ = struct.unpack(IDX_RECORD_FMT, last_rec)
    if last_off + last_len != bin_size:
        return (
            False,
            f"tail mismatch in {idx_path.name}: offset({last_off})+len({last_len}) != bin({bin_size})",
        )

    if mask_path is not None:
        if not mask_path.is_file():
            return False, f"missing {mask_path.name}"
        mask_size = mask_path.stat().st_size
        if mask_size != bin_size:
            return False, f"mask size mismatch: {mask_path.name}({mask_size}) != {bin_path.name}({bin_size})"

    return True, f"ok ({idx_size // IDX_RECORD_SIZE} docs, {bin_size / (1024 * 1024):.2f} MB)"


def verify_all_shards(nvme_root: Path, *, include_cap_code: bool = False) -> dict[str, dict[str, Any]]:
    """Verify shard groups (`data`, `cap_reasoning`, `cap_instruct`, `cap_writing`, `cap_toolcall`, optional `cap_code`, `sft`)."""
    data_dir = nvme_root / "data"
    cap_dir = nvme_root / "cap_data"
    sft_dir = nvme_root / "sft_data"

    checks: dict[str, list[tuple[Path, Path, Optional[Path]]]] = {
        "corpus_5domain": [
            (data_dir / "train.bin", data_dir / "train.idx", None),
            (data_dir / "val.bin", data_dir / "val.idx", None),
        ],
        "cap_reasoning": [
            (
                cap_dir / "cap_reasoning_train.bin",
                cap_dir / "cap_reasoning_train.idx",
                cap_dir / "cap_reasoning_train.loss_mask.bin",
            ),
            (
                cap_dir / "cap_reasoning_val.bin",
                cap_dir / "cap_reasoning_val.idx",
                cap_dir / "cap_reasoning_val.loss_mask.bin",
            ),
        ],
        "cap_instruct": [
            (
                cap_dir / "cap_instruct_train.bin",
                cap_dir / "cap_instruct_train.idx",
                cap_dir / "cap_instruct_train.loss_mask.bin",
            ),
            (
                cap_dir / "cap_instruct_val.bin",
                cap_dir / "cap_instruct_val.idx",
                cap_dir / "cap_instruct_val.loss_mask.bin",
            ),
        ],
        "cap_writing": [
            (
                cap_dir / "cap_writing_train.bin",
                cap_dir / "cap_writing_train.idx",
                cap_dir / "cap_writing_train.loss_mask.bin",
            ),
            (
                cap_dir / "cap_writing_val.bin",
                cap_dir / "cap_writing_val.idx",
                cap_dir / "cap_writing_val.loss_mask.bin",
            ),
        ],
        "cap_toolcall": [
            (
                cap_dir / "cap_toolcall_train.bin",
                cap_dir / "cap_toolcall_train.idx",
                cap_dir / "cap_toolcall_train.loss_mask.bin",
            ),
            (
                cap_dir / "cap_toolcall_val.bin",
                cap_dir / "cap_toolcall_val.idx",
                cap_dir / "cap_toolcall_val.loss_mask.bin",
            ),
        ],
    }
    if include_cap_code:
        checks["cap_code"] = [
            (
                cap_dir / "cap_code_train.bin",
                cap_dir / "cap_code_train.idx",
                cap_dir / "cap_code_train.loss_mask.bin",
            ),
            (
                cap_dir / "cap_code_val.bin",
                cap_dir / "cap_code_val.idx",
                cap_dir / "cap_code_val.loss_mask.bin",
            ),
        ]
    checks["stage5_sft"] = [
        (
            sft_dir / "sft_train.bin",
            sft_dir / "sft_train.idx",
            sft_dir / "sft_train.loss_mask.bin",
        ),
        (
            sft_dir / "sft_val.bin",
            sft_dir / "sft_val.idx",
            sft_dir / "sft_val.loss_mask.bin",
        ),
    ]

    report: dict[str, dict[str, Any]] = {}
    for group_name, triplets in checks.items():
        group_ok = True
        details: list[str] = []
        total_bytes = 0
        for b_p, i_p, m_p in triplets:
            ok, msg = verify_shard_triplet(b_p, i_p, m_p)
            details.append(f"{b_p.name}: {msg}")
            if not ok:
                group_ok = False
            else:
                total_bytes += b_p.stat().st_size
        report[group_name] = {
            "valid": group_ok,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "details": details,
        }
    return report


def build_job_plan(
    nvme_root: Path,
    spec: DataProfileSpec,
    *,
    python_bin: str = sys.executable,
    seed: int = 2026,
    offline_synthetic: bool = False,
) -> dict[str, list[str]]:
    """Construct the 5 concurrent builder command lines."""
    data_dir = nvme_root / "data"
    cap_dir = nvme_root / "cap_data"
    sft_dir = nvme_root / "sft_data"

    corpus_cmd = [
        python_bin,
        str(_REPO_ROOT / "data" / "build_corpus.py"),
        "--out-dir",
        str(data_dir),
        "--scale-factor",
        str(spec.corpus_scale_factor),
        "--workers",
        str(spec.corpus_workers),
        "--seed",
        str(seed),
    ]
    if spec.smoke_quota:
        corpus_cmd.append("--smoke-quota")
    if spec.stem_upgrade:
        corpus_cmd.append("--stem-upgrade")
    if spec.cognitive_v2:
        corpus_cmd.append("--cognitive-v2")
    if offline_synthetic:
        corpus_cmd.append("--offline")

    def _cap_cmd(script_name: str, target_mb: float, sub_seed: int, *, extra_flags: tuple[str, ...] = ()) -> list[str]:
        cmd = [
            python_bin,
            str(_REPO_ROOT / "data" / script_name),
            "--out-dir",
            str(cap_dir),
            "--target-mb",
            str(target_mb),
            "--seed",
            str(sub_seed),
        ]
        if spec.smoke_quota:
            cmd.append("--smoke-quota")
        if offline_synthetic:
            cmd.append("--offline")
        cmd.extend(extra_flags)
        return cmd

    sft_cmd = [
        python_bin,
        str(_REPO_ROOT / "data" / "build_sft_stage5.py"),
        "--out-dir",
        str(sft_dir),
        "--train-mb",
        str(spec.sft_train_mb),
        "--val-mb",
        str(spec.sft_val_mb),
        "--seed",
        str(seed + 5),
    ]
    if spec.smoke_quota:
        sft_cmd.append("--smoke-quota")
    if spec.cognitive_v2:
        sft_cmd.append("--cognitive-v2")
    if offline_synthetic:
        sft_cmd.append("--offline")

    reas_flags = ("--cognitive-v2",) if spec.cognitive_v2 else ()
    jobs: dict[str, list[str]] = {
        "corpus_5domain": corpus_cmd,
        "cap_reasoning": _cap_cmd("build_cap_reasoning.py", spec.cap_reasoning_mb, seed + 11, extra_flags=reas_flags),
        "cap_instruct": _cap_cmd("build_cap_instruct.py", spec.cap_instruct_mb, seed + 14),
        "cap_writing": _cap_cmd("build_cap_writing.py", spec.cap_writing_mb, seed + 12),
        "cap_toolcall": _cap_cmd("build_cap_toolcall.py", spec.cap_toolcall_mb, seed + 13),
    }
    if spec.cap_code_mb > 0.0:
        jobs["cap_code"] = _cap_cmd("build_cap_code_v2.py", spec.cap_code_mb, seed + 15)
    jobs["stage5_sft"] = sft_cmd
    return jobs


def _restore_from_archive(archive_src: str, nvme_root: Path) -> None:
    """Download (if URL) and extract `.tar.zst` / `.tar.gz` directly onto local NVMe."""
    nvme_root.mkdir(parents=True, exist_ok=True)
    local_archive = Path(archive_src)
    cleanup_after = False

    if archive_src.startswith(("http://", "https://")):
        local_archive = nvme_root / ("_download_" + archive_src.rsplit("/", 1)[-1].split("?")[0])
        cleanup_after = True
        print(f"[prepare_vast] Downloading archive {archive_src} -> {local_archive} ...", flush=True)
        if shutil.which("aria2c"):
            subprocess.run(
                [
                    "aria2c",
                    "-x",
                    "16",
                    "-s",
                    "16",
                    "-k",
                    "4M",
                    "--dir",
                    str(local_archive.parent),
                    "--out",
                    local_archive.name,
                    archive_src,
                ],
                check=True,
            )
        else:
            subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(local_archive), archive_src], check=True)

    if not local_archive.is_file():
        raise FileNotFoundError(f"Archive not found: {local_archive}")

    print(f"[prepare_vast] Extracting {local_archive} into {nvme_root} ...", flush=True)
    if local_archive.name.endswith(".zst"):
        subprocess.run(
            ["tar", "-I", "zstd -d -T0", "-xf", str(local_archive), "-C", str(nvme_root)],
            check=True,
        )
    else:
        subprocess.run(["tar", "-xzf", str(local_archive), "-C", str(nvme_root)], check=True)

    if cleanup_after:
        local_archive.unlink(missing_ok=True)


def _pack_archive(nvme_root: Path, out_archive: Path) -> None:
    """Compress `data/`, `cap_data/`, and `sft_data/` into a fast `.tar.zst` snapshot."""
    out_archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_vast] Packing shards into {out_archive} ...", flush=True)
    if out_archive.name.endswith(".zst") and shutil.which("zstd"):
        subprocess.run(
            ["tar", "-I", "zstd -T0 -3", "-cf", str(out_archive), "-C", str(nvme_root), "data", "cap_data", "sft_data"],
            check=True,
        )
    else:
        subprocess.run(
            ["tar", "-czf", str(out_archive), "-C", str(nvme_root), "data", "cap_data", "sft_data"],
            check=True,
        )
    sz_mb = out_archive.stat().st_size / (1024 * 1024)
    print(f"[prepare_vast] Archive ready: {out_archive} ({sz_mb:.1f} MB)", flush=True)


def prepare_vast_corpus(
    *,
    profile: str = "v2_stem_10g",
    nvme_root: Optional[str | Path] = None,
    from_archive: Optional[str] = None,
    pack_archive: Optional[str | Path] = None,
    force: bool = False,
    dry_run: bool = False,
    max_parallel_jobs: int = 6,
    seed: int = 2026,
    offline_synthetic: bool = False,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    spec = PROFILES[profile]
    root = Path(nvme_root) if nvme_root is not None else detect_default_nvme_root()

    resolved_str = str(root.resolve()) if root.exists() else str(root)
    if resolved_str.startswith(("/content/drive", "/mnt/gdrive")):
        raise ValueError(f"Refusing to place Vast.ai NVMe shards on FUSE mount: {resolved_str}")

    plan = build_job_plan(root, spec, seed=seed, offline_synthetic=offline_synthetic)
    env_overrides = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": str(root / "hf_cache"),
        "TOKENIZERS_PARALLELISM": "false",
    }

    has_cap_code = spec.cap_code_mb > 0.0
    existing_status = verify_all_shards(root, include_cap_code=has_cap_code) if root.exists() else {}

    if dry_run:
        free_gb = None
        probe_dir = root if root.exists() else root.parent
        if probe_dir.exists():
            free_gb = round(shutil.disk_usage(probe_dir).free / (1024**3), 2)
        summary = {
            "dry_run": True,
            "profile": spec.name,
            "description": spec.description,
            "nvme_root": str(root),
            "data_dir": str(root / "data"),
            "cap_data_dir": str(root / "cap_data"),
            "sft_data_dir": str(root / "sft_data"),
            "free_disk_gb": free_gb,
            "required_free_gb": spec.min_free_gb,
            "env": env_overrides,
            "existing_shards": existing_status,
            "commands": plan,
        }
        return summary

    for sub in ("data", "cap_data", "sft_data", "hf_cache", "build_logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(root).free / (1024**3)
    if free_gb < spec.min_free_gb:
        raise RuntimeError(
            f"Insufficient free NVMe space in {root}: {free_gb:.1f} GB available, "
            f"profile '{spec.name}' recommends >= {spec.min_free_gb:.1f} GB."
        )

    t0 = time.perf_counter()
    if from_archive:
        _restore_from_archive(from_archive, root)

    status = verify_all_shards(root, include_cap_code=has_cap_code)
    pending_jobs: list[tuple[str, list[str]]] = []
    for job_name, cmd in plan.items():
        if not force and status.get(job_name, {}).get("valid", False):
            print(
                f"[prepare_vast] [SKIP] {job_name} already valid ({status[job_name]['total_mb']:.2f} MB)",
                flush=True,
            )
        else:
            pending_jobs.append((job_name, cmd))

    if pending_jobs:
        print(
            f"[prepare_vast] Launching {len(pending_jobs)} builder(s) (max_parallel={max_parallel_jobs}) "
            f"for profile='{spec.name}' into {root} ...",
            flush=True,
        )
        child_env = os.environ.copy()
        child_env.update(env_overrides)
        existing_py = child_env.get("PYTHONPATH", "")
        py_parts = [str(_REPO_ROOT), str(_TRAIN_PY)] + ([existing_py] if existing_py else [])
        child_env["PYTHONPATH"] = os.pathsep.join(py_parts)

        active: list[tuple[str, subprocess.Popen[bytes], Any, Path, float]] = []
        queue = list(pending_jobs)
        failed: list[tuple[str, int, Path]] = []
        last_heartbeat = time.perf_counter()

        while queue or active:
            while queue and len(active) < max(1, int(max_parallel_jobs)):
                j_name, j_cmd = queue.pop(0)
                log_path = root / "build_logs" / f"{j_name}.log"
                log_f = log_path.open("wb")
                print(f"  -> [START] {j_name} (log: {log_path})", flush=True)
                proc = subprocess.Popen(
                    j_cmd,
                    cwd=str(_REPO_ROOT),
                    env=child_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
                active.append((j_name, proc, log_f, log_path, time.perf_counter()))

            time.sleep(0.5)
            still_running = []
            for j_name, proc, log_f, log_path, j_t0 in active:
                rc = proc.poll()
                if rc is None:
                    still_running.append((j_name, proc, log_f, log_path, j_t0))
                else:
                    log_f.close()
                    dt = time.perf_counter() - j_t0
                    if rc == 0:
                        print(f"  <- [DONE]  {j_name} finished in {dt:.1f}s", flush=True)
                    else:
                        print(f"  <- [FAIL]  {j_name} exited with code {rc} after {dt:.1f}s!", flush=True)
                        failed.append((j_name, rc, log_path))
            active = still_running

            now = time.perf_counter()
            if active and (now - last_heartbeat) >= 15.0:
                names = ", ".join(f"{n}({now - jt:.0f}s)" for n, _, _, _, jt in active)
                print(f"  .. [PROGRESS] running: {names}", flush=True)
                last_heartbeat = now

        if failed:
            msg_lines = ["One or more dataset builders failed:"]
            for j_name, rc, log_path in failed:
                tail = ""
                if log_path.is_file():
                    tail_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                    tail = "\n    ".join(tail_lines)
                msg_lines.append(f"  * {j_name} (exit={rc}, log={log_path}):\n    {tail}")
            raise RuntimeError("\n".join(msg_lines))

    final_status = verify_all_shards(root, include_cap_code=has_cap_code)
    invalid = [k for k, v in final_status.items() if not v["valid"]]
    if invalid:
        raise RuntimeError(f"Shard integrity verification failed for {invalid}: {json.dumps(final_status, indent=2)}")

    # Clean temporary HuggingFace streaming datasets/xet cache without deleting auth token files
    for cache_sub in ("datasets", "xet", "downloads"):
        shutil.rmtree(root / "hf_cache" / cache_sub, ignore_errors=True)

    if pack_archive:
        _pack_archive(root, Path(pack_archive))

    total_mb = round(sum(v["total_mb"] for v in final_status.values()), 2)
    elapsed = round(time.perf_counter() - t0, 2)
    result = {
        "dry_run": False,
        "profile": spec.name,
        "nvme_root": str(root),
        "data_dir": str(root / "data"),
        "cap_data_dir": str(root / "cap_data"),
        "sft_data_dir": str(root / "sft_data"),
        "total_verified_mb": total_mb,
        "elapsed_sec": elapsed,
        "shards": final_status,
    }
    (root / "vast_corpus_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[prepare_vast] ALL SHARDS VERIFIED: {total_mb:.2f} MB ready in {elapsed:.1f}s at {root}",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel NVMe corpus & capability shard orchestrator for Vast.ai")
    ap.add_argument(
        "--profile",
        type=str,
        default="v2_stem_10g",
        choices=sorted(PROFILES.keys()),
        help="Dataset scale profile (default: v2_stem_10g for 4x A100 80GB).",
    )
    ap.add_argument(
        "--nvme-root",
        type=str,
        default=None,
        help="Local NVMe root directory (auto-detects /workspace/nvme_cache if available).",
    )
    ap.add_argument(
        "--from-archive",
        type=str,
        default=None,
        help="Optional prebuilt .tar.zst / .tar.gz URL or local path for <30s cold start.",
    )
    ap.add_argument(
        "--pack-archive",
        type=str,
        default=None,
        help="Optional output .tar.zst path to snapshot all built shards.",
    )
    ap.add_argument("--max-parallel-jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true", help="Rebuild even if valid shards already exist.")
    ap.add_argument("--offline-synthetic", action="store_true", help="Pass --offline to capability/SFT builders.")
    ap.add_argument("--dry-run", action="store_true", help="Print execution plan JSON without downloading data.")
    args = ap.parse_args()

    res = prepare_vast_corpus(
        profile=args.profile,
        nvme_root=args.nvme_root,
        from_archive=args.from_archive,
        pack_archive=args.pack_archive,
        force=args.force,
        dry_run=args.dry_run,
        max_parallel_jobs=args.max_parallel_jobs,
        seed=args.seed,
        offline_synthetic=args.offline_synthetic,
    )
    if args.dry_run:
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
