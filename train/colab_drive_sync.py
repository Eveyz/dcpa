"""Two-tier NVMe + Google Drive A/B Ping-Pong Anti-Disconnect Checkpoint Engine.

Implements Iron Law #3:
  1. Strict two-tier storage isolation: all hot mmap reads and checkpoint staging occur ONLY
     under `--local-cache-dir` (e.g. `/content/nvme_cache/`). Direct `mmap` / hot I/O under
     `/content/drive` is explicitly rejected.
  2. Dual-trigger checkpoint policy: saves whenever EITHER `step % save_every_steps == 0`
     (default 1,000) OR wall-clock elapsed >= `save_every_minutes * 60` (default 10 minutes).
  3. A/B Ping-Pong in-place overwrite protocol inside `--drive-sync-dir`:
       - Maintains ONLY two fixed rolling files (`slot_A.pt`, `slot_B.pt`) plus `manifest.json`
         so Google Drive Trash never fills up from create/delete churn.
       - Step 1: Atomically stage full checkpoint to `<local_cache_dir>/ckpts/staging.pt`,
                 compute full `blake2b` digest and `file_size`.
       - Step 2: Select inactive backup slot (`A` <-> `B`), overwrite it IN-PLACE in Drive
                 with chunked copy + `f.flush()` + `os.fsync(f.fileno())`.
       - Step 3: Verify exact file size on Drive, then update `manifest.json` with
                 `active_slot`, `backup_slot`, `step`, `stage_id`, `blake2b`, `file_size`,
                 `wall_time`, and per-slot verification records.
  4. Self-healing `--resume auto`: checks `active_slot` size + `blake2b` first; if truncated
     or corrupted by an abrupt Colab disconnect, automatically falls back to `backup_slot`
     and logs the recovery event.
  5. Milestone archival strictly bounded to <= 5 files:
     steps `15k / 65k / 85k / 100k` (`milestone_step_XXXXXX.pt`) + `best_model.pt`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

SLOT_FILES: dict[str, str] = {
    "A": "slot_A.pt",
    "B": "slot_B.pt",
}

MILESTONE_STEPS: tuple[int, ...] = (15_000, 65_000, 94_000, 100_000)
MAX_MILESTONE_FILES: int = 5  # 4 stage boundaries + best_model.pt
COPY_CHUNK_BYTES: int = 4 * 1024 * 1024  # 4 MB FUSE-friendly sequential buffer


def is_drive_fuse_path(path: str | Path) -> bool:
    """Return True if path resides inside a Google Drive FUSE mount (`/content/drive`)."""
    raw = str(path)
    if raw.startswith("/content/drive"):
        return True
    try:
        resolved = str(Path(path).resolve())
        if resolved.startswith("/content/drive"):
            return True
    except Exception:
        pass
    return False


def assert_not_drive_path(path: str | Path, role: str = "hot cache") -> Path:
    """Guard enforcing Iron Law #3.1: hot mmap / staging directories must live on local NVMe."""
    p = Path(path)
    if is_drive_fuse_path(p):
        raise ValueError(
            f"Iron Law #3 violation: {role} path ({p}) must NOT reside on /content/drive FUSE mount. "
            "Use local NVMe such as /content/nvme_cache/."
        )
    return p


def compute_file_blake2b(path: str | Path, chunk_size: int = COPY_CHUNK_BYTES) -> tuple[str, int]:
    """Compute full BLAKE2b hex digest and exact byte size of a file."""
    p = Path(path)
    h = hashlib.blake2b()
    total = 0
    with p.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
            total += len(buf)
    return h.hexdigest(), total


def copy_inplace_fsync(src: Path, dst: Path, chunk_size: int = COPY_CHUNK_BYTES) -> int:
    """Overwrite `dst` IN-PLACE (no unlink/create churn on Google Drive) and flush+fsync through FUSE."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    mode = "r+b" if dst.exists() else "wb"
    with src.open("rb") as fin, dst.open(mode) as fout:
        if mode == "r+b":
            fout.seek(0)
        while True:
            buf = fin.read(chunk_size)
            if not buf:
                break
            fout.write(buf)
            written += len(buf)
        fout.truncate(written)
        fout.flush()
        os.fsync(fout.fileno())
    return written


def write_text_fsync(dst: Path, content: str) -> None:
    """Write a small metadata/log text file in-place with flush + os.fsync."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    mode = "r+b" if dst.exists() else "wb"
    with dst.open(mode) as fout:
        if mode == "r+b":
            fout.seek(0)
        fout.write(data)
        fout.truncate(len(data))
        fout.flush()
        os.fsync(fout.fileno())


@dataclass
class ResumeResult:
    checkpoint: dict[str, Any]
    slot_used: str
    fallback_triggered: bool
    source_path: Path
    step: int
    stage_id: str
    reason: str


class CheckpointTrigger:
    """Dual-trigger checkpoint scheduler: step interval OR wall-clock timer (default 10 min)."""

    def __init__(
        self,
        save_every_steps: int = 1000,
        save_every_minutes: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.save_every_steps = max(0, int(save_every_steps))
        self.save_every_seconds = max(0.0, float(save_every_minutes) * 60.0)
        self._clock = clock
        self.last_save_time: float = self._clock()
        self.last_saved_step: int = -1

    def should_save(self, step: int) -> tuple[bool, str]:
        """Check whether either step interval or wall-clock interval has elapsed."""
        if step <= self.last_saved_step:
            return False, ""
        step_due = self.save_every_steps > 0 and step > 0 and (step % self.save_every_steps == 0)
        elapsed = self._clock() - self.last_save_time
        time_due = self.save_every_seconds > 0.0 and elapsed >= self.save_every_seconds
        if step_due and time_due:
            return True, "step_and_wall_clock"
        if step_due:
            return True, "step_interval"
        if time_due:
            return True, "wall_clock"
        return False, ""

    def mark_saved(self, step: int) -> None:
        self.last_saved_step = step
        self.last_save_time = self._clock()


class ABDriveSyncManager:
    """A/B Ping-Pong in-place checkpoint sync manager between local NVMe and Google Drive."""

    def __init__(
        self,
        local_cache_dir: str | Path,
        drive_sync_dir: Optional[str | Path] = None,
        milestone_steps: tuple[int, ...] = MILESTONE_STEPS,
    ) -> None:
        self.local_cache_dir = assert_not_drive_path(local_cache_dir, role="--local-cache-dir")
        self.local_ckpt_dir = self.local_cache_dir / "ckpts"
        self.local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.local_ckpt_dir / "staging.pt"

        # If drive_sync_dir is omitted (e.g. local M3 Pro test), sync to local_cache_dir/drive_mirror
        self.drive_sync_dir = (
            Path(drive_sync_dir) if drive_sync_dir is not None else (self.local_cache_dir / "drive_checkpoints")
        )
        self.drive_sync_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.drive_sync_dir / "manifest.json"
        self.manifest_backup_path = self.drive_sync_dir / "manifest.bak.json"
        self.milestone_steps = tuple(milestone_steps)

    def _read_manifest(self) -> Optional[dict[str, Any]]:
        for candidate in (self.manifest_path, self.manifest_backup_path):
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "active_slot" in data:
                    return data
            except Exception:
                continue
        return None

    def _next_target_slot(self, manifest: Optional[dict[str, Any]]) -> tuple[str, Optional[str]]:
        if manifest is None:
            return "A", None
        current_active = manifest.get("active_slot")
        if current_active == "A":
            return "B", "A"
        return "A", "B" if current_active == "B" else None

    def _stage_local_checkpoint(self, payload: dict[str, Any]) -> tuple[str, int]:
        """Step 1: Atomically serialize checkpoint to local NVMe staging.pt and hash it."""
        tmp_path = self.local_ckpt_dir / "staging.pt.tmp"
        torch.save(payload, str(tmp_path))
        with tmp_path.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, self.staging_path)
        return compute_file_blake2b(self.staging_path)

    def save_ping_pong(
        self,
        payload: dict[str, Any],
        step: int,
        stage_id: str,
        reason: str = "periodic",
        is_best: bool = False,
        verify_readback: bool = False,
    ) -> dict[str, Any]:
        """Execute the 3-step A/B Ping-Pong commit protocol.

        Step 1: Fast local NVMe staging write + BLAKE2b checksum + byte size.
        Step 2: In-place overwrite of inactive Drive slot (`slot_A.pt` <-> `slot_B.pt`) + flush + fsync.
        Step 3: Size verification on Drive + manifest.json commit.
        """
        # Step 1: Local NVMe staging
        digest, file_size = self._stage_local_checkpoint(payload)

        # Step 2: Determine target slot and overwrite in-place with fsync
        prev_manifest = self._read_manifest()
        target_slot, prev_active = self._next_target_slot(prev_manifest)
        target_file = self.drive_sync_dir / SLOT_FILES[target_slot]

        written = copy_inplace_fsync(self.staging_path, target_file)
        actual_size = target_file.stat().st_size
        if written != file_size or actual_size != file_size:
            raise IOError(
                f"Drive sync size mismatch on {target_file}: expected={file_size}, "
                f"written={written}, stat={actual_size}"
            )
        if verify_readback:
            rb_digest, rb_size = compute_file_blake2b(target_file)
            if rb_digest != digest or rb_size != file_size:
                raise IOError(f"Drive readback BLAKE2b verification failed on {target_file}")

        # Step 3: Commit manifest.json (preserve prior slot metadata for self-healing fallback!)
        wall_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        slots_meta: dict[str, Any] = {}
        if prev_manifest and isinstance(prev_manifest.get("slots"), dict):
            slots_meta.update(prev_manifest["slots"])

        slot_record = {
            "slot": target_slot,
            "filename": SLOT_FILES[target_slot],
            "step": int(step),
            "stage_id": str(stage_id),
            "blake2b": digest,
            "file_size": int(file_size),
            "wall_time": wall_time,
            "reason": reason,
        }
        slots_meta[target_slot] = slot_record

        backup_slot = prev_active if (prev_active in slots_meta) else None
        manifest = {
            "schema_version": 1,
            "active_slot": target_slot,
            "backup_slot": backup_slot,
            "step": int(step),
            "stage_id": str(stage_id),
            "blake2b": digest,
            "file_size": int(file_size),
            "wall_time": wall_time,
            "reason": reason,
            "slots": slots_meta,
        }
        manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
        # Keep backup manifest updated first, then commit primary manifest in-place
        if prev_manifest is not None:
            write_text_fsync(self.manifest_backup_path, json.dumps(prev_manifest, indent=2, ensure_ascii=False))
        write_text_fsync(self.manifest_path, manifest_text)

        # Optional milestone archival (strictly capped to 4 boundary steps + best_model.pt <= 5 files)
        if int(step) in self.milestone_steps:
            ms_name = f"milestone_step_{int(step):06d}.pt"
            ms_path = self.drive_sync_dir / ms_name
            copy_inplace_fsync(self.staging_path, ms_path)
            print(f"[DriveSync] Archived milestone checkpoint -> {ms_path}", flush=True)

        if is_best:
            best_path = self.drive_sync_dir / "best_model.pt"
            copy_inplace_fsync(self.staging_path, best_path)
            print(f"[DriveSync] Updated best_model.pt at step={step}", flush=True)

        return manifest

    def _verify_slot_file(self, slot: str, expected: dict[str, Any]) -> tuple[bool, str]:
        filename = expected.get("filename") or SLOT_FILES.get(slot)
        if not filename:
            return False, f"unknown slot {slot}"
        path = self.drive_sync_dir / filename
        if not path.exists():
            return False, f"file missing ({path})"
        exp_size = int(expected.get("file_size", -1))
        actual_size = path.stat().st_size
        if exp_size <= 0 or actual_size != exp_size:
            return False, f"size mismatch on {filename} (expected={exp_size}, actual={actual_size})"
        exp_hash = expected.get("blake2b")
        actual_hash, _ = compute_file_blake2b(path)
        if not exp_hash or actual_hash != exp_hash:
            return False, f"blake2b mismatch on {filename} (expected={str(exp_hash)[:12]}, actual={actual_hash[:12]})"
        return True, "ok"

    def load_latest_auto(self, map_location: str | torch.device = "cpu") -> Optional[ResumeResult]:
        """Self-healing `--resume auto` loader.

        Verifies `active_slot` size + `blake2b` first; on ANY corruption or truncation,
        automatically falls back to `backup_slot`, verifies it, stages it to local NVMe,
        and loads it cleanly.
        """
        manifest = self._read_manifest()
        if manifest is None:
            # If no active A/B manifest exists (e.g., after archiving stale slots to rewind cleanly),
            # automatically resume from the highest-step archived checkpoint among `best_model.pt`
            # and `milestone_step_*.pt`.
            cand_paths: list[Path] = sorted(self.drive_sync_dir.glob("milestone_step_*.pt"))
            best_p = self.drive_sync_dir / "best_model.pt"
            if best_p.exists():
                cand_paths.append(best_p)
            best_ckpt: Optional[dict[str, Any]] = None
            best_path: Optional[Path] = None
            best_step = -1
            for cp in cand_paths:
                local_restore = self.local_ckpt_dir / f"resume_{cp.name}"
                try:
                    shutil.copyfile(cp, local_restore)
                    ck = torch.load(str(local_restore), map_location=map_location, weights_only=False)
                    st = int(ck.get("step", -1))
                    if st > best_step:
                        best_step = st
                        best_ckpt = ck
                        best_path = cp
                except Exception as exc:
                    print(f"[DriveSync][WARN] Skipping unreadable fallback checkpoint {cp.name}: {exc}", flush=True)
            if best_ckpt is None or best_path is None:
                return None
            stage_id = str(best_ckpt.get("stage_id", "A"))
            print(
                f"[DriveSync][RESUME] No manifest.json; auto-resuming from {best_path.name} "
                f"| step={best_step} | stage={stage_id}",
                flush=True,
            )
            return ResumeResult(
                checkpoint=best_ckpt,
                slot_used=best_path.name,
                fallback_triggered=True,
                source_path=best_path,
                step=best_step,
                stage_id=stage_id,
                reason=f"Auto-resumed from {best_path.name} (no active manifest.json)",
            )

        slots_meta = manifest.get("slots", {})
        active = manifest.get("active_slot")
        backup = manifest.get("backup_slot")

        candidates: list[tuple[str, dict[str, Any], bool]] = []
        if active in SLOT_FILES:
            active_meta = slots_meta.get(active, {
                "filename": SLOT_FILES[active],
                "step": manifest.get("step", 0),
                "stage_id": manifest.get("stage_id", "A"),
                "blake2b": manifest.get("blake2b"),
                "file_size": manifest.get("file_size"),
            })
            candidates.append((active, active_meta, False))

        if backup in SLOT_FILES and backup != active:
            backup_meta = slots_meta.get(backup)
            if isinstance(backup_meta, dict):
                candidates.append((backup, backup_meta, True))

        # Also check the complementary slot if slots_meta recorded it even when backup was None
        other = "B" if active == "A" else "A"
        if other != backup and isinstance(slots_meta.get(other), dict):
            candidates.append((other, slots_meta[other], True))

        failure_reasons: list[str] = []
        for slot, meta, is_fallback in candidates:
            ok, msg = self._verify_slot_file(slot, meta)
            if not ok:
                failure_reasons.append(f"slot_{slot}: {msg}")
                print(f"[DriveSync][WARN] Slot {slot} failed integrity check: {msg}", flush=True)
                continue

            src_path = self.drive_sync_dir / (meta.get("filename") or SLOT_FILES[slot])
            # Stage back to local NVMe before torch.load so deserialization avoids Drive FUSE random seeks
            local_restore = self.local_ckpt_dir / f"resume_slot_{slot}.pt"
            shutil.copyfile(src_path, local_restore)
            try:
                ckpt = torch.load(str(local_restore), map_location=map_location, weights_only=False)
            except Exception as exc:
                failure_reasons.append(f"slot_{slot}: torch.load error ({exc})")
                print(f"[DriveSync][WARN] Slot {slot} torch.load failed: {exc}", flush=True)
                continue

            reason = (
                f"Self-healed fallback to backup slot_{slot} after active failure ({'; '.join(failure_reasons)})"
                if is_fallback
                else f"Verified active slot_{slot} (blake2b + size OK)"
            )
            print(
                f"[DriveSync][RESUME] {reason} | step={meta.get('step', ckpt.get('step', 0))} "
                f"| stage={meta.get('stage_id', ckpt.get('stage_id', 'A'))}",
                flush=True,
            )
            return ResumeResult(
                checkpoint=ckpt,
                slot_used=slot,
                fallback_triggered=is_fallback,
                source_path=src_path,
                step=int(ckpt.get("step", meta.get("step", 0))),
                stage_id=str(ckpt.get("stage_id", meta.get("stage_id", "A"))),
                reason=reason,
            )

        raise RuntimeError(
            "All checkpoint slots in Drive manifest failed integrity verification: "
            + "; ".join(failure_reasons)
        )

    def list_milestone_files(self) -> list[Path]:
        """List milestone files + best_model.pt currently in `drive_sync_dir`."""
        found: list[Path] = []
        for p in sorted(self.drive_sync_dir.glob("milestone_step_*.pt")):
            found.append(p)
        best = self.drive_sync_dir / "best_model.pt"
        if best.exists():
            found.append(best)
        return found


def sync_file_to_drive(local_path: str | Path, drive_dir: str | Path, dest_name: Optional[str] = None) -> Path:
    """In-place fsync copy of a log/probe/export artifact from local NVMe to Google Drive."""
    src = Path(local_path)
    dst_dir = Path(drive_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (dest_name or src.name)
    expected_size = src.stat().st_size
    written = copy_inplace_fsync(src, dst)
    if written != expected_size or dst.stat().st_size != expected_size:
        raise IOError(f"Failed syncing {src} -> {dst}: expected={expected_size}, written={written}")
    return dst


def restore_drive_log_file(
    drive_logs_dir: Optional[str | Path],
    local_logs_dir: str | Path,
    filename: str,
) -> Optional[Path]:
    """Restore an append-only log file (e.g. `metrics.jsonl`) from Drive onto fresh NVMe before appending."""
    if drive_logs_dir is None:
        return None
    src = Path(drive_logs_dir) / filename
    dst = Path(local_logs_dir) / filename
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(src, dst)
            return dst
        except Exception as exc:
            print(f"[DriveSync][WARN] Could not restore {src} -> {dst}: {exc}", flush=True)
    return None


class _TeeStream:
    """Transparent stdout/stderr tee writing immediately to local NVMe log file."""

    def __init__(self, orig_stream: Any, log_fp: Any) -> None:
        self._orig = orig_stream
        self._fp = log_fp

    def write(self, data: str) -> int:
        n = self._orig.write(data)
        if data and not self._fp.closed:
            try:
                self._fp.write(data)
                if "\n" in data:
                    self._fp.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        self._orig.flush()
        if not self._fp.closed:
            try:
                self._fp.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return bool(getattr(self._orig, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._orig.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._orig, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class RunLogTee:
    """Capture full console stdout+stderr to NVMe and mirror safely to Google Drive `logs/`.

    Writes all hot line-by-line output ONLY to local NVMe (`<local_logs_dir>/train_run_<ts>.log`)
    and periodically (plus on rollbacks, probes, checkpoints, and exit/crash) syncs in-place to:
      - `<drive_logs_dir>/train_run_<ts>.log` (per-session archive, never overwrites past runs)
      - `<drive_logs_dir>/train_latest.log`   (fixed pointer to the latest run's full console log)
      - `<drive_logs_dir>/watchdog_events.jsonl` (whenever present)
    """

    def __init__(
        self,
        local_logs_dir: str | Path,
        drive_logs_dir: Optional[str | Path] = None,
        *,
        sync_interval_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.local_logs_dir = Path(local_logs_dir)
        self.drive_logs_dir = Path(drive_logs_dir) if drive_logs_dir is not None else None
        self.sync_interval_seconds = max(0.0, float(sync_interval_seconds))
        self._clock = clock
        self.local_logs_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_logs_dir is not None:
            self.drive_logs_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        base_name = f"train_run_{ts}"
        candidate = f"{base_name}.log"
        suffix = 1
        while (self.local_logs_dir / candidate).exists() or (
            self.drive_logs_dir is not None and (self.drive_logs_dir / candidate).exists()
        ):
            candidate = f"{base_name}_{suffix}.log"
            suffix += 1
        self.log_filename = candidate
        self.local_log_path = self.local_logs_dir / self.log_filename
        self._fp: Optional[Any] = None
        self._orig_stdout: Optional[Any] = None
        self._orig_stderr: Optional[Any] = None
        self._last_sync_t: float = -1e9

    def __enter__(self) -> "RunLogTee":
        self._fp = self.local_log_path.open("a", encoding="utf-8", buffering=1)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._fp)
        sys.stderr = _TeeStream(self._orig_stderr, self._fp)
        return self

    def sync_to_drive(self, *, force: bool = False) -> bool:
        if self.drive_logs_dir is None or self._fp is None or self._fp.closed:
            return False
        now = self._clock()
        if not force and (now - self._last_sync_t) < self.sync_interval_seconds:
            return False
        try:
            self._fp.flush()
            sync_file_to_drive(self.local_log_path, self.drive_logs_dir, dest_name=self.log_filename)
            sync_file_to_drive(self.local_log_path, self.drive_logs_dir, dest_name="train_latest.log")
            wd_events = self.local_logs_dir / "watchdog_events.jsonl"
            if wd_events.exists():
                sync_file_to_drive(wd_events, self.drive_logs_dir, dest_name="watchdog_events.jsonl")
            self._last_sync_t = now
            return True
        except Exception as exc:
            err_stream = self._orig_stderr if self._orig_stderr is not None else sys.stderr
            try:
                err_stream.write(f"[DriveSync][WARN] Failed syncing training log to Drive: {exc}\n")
                err_stream.flush()
            except Exception:
                pass
            return False

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None and self._fp is not None and not self._fp.closed:
            try:
                tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                self._fp.write(f"\n[RunLogTee][EXIT] {tb_text}\n")
                self._fp.flush()
            except Exception:
                pass
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        try:
            self.sync_to_drive(force=True)
        finally:
            if self._fp is not None and not self._fp.closed:
                self._fp.close()
        return False
