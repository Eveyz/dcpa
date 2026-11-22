"""Five-domain raw UTF-8 byte corpus builder for DCPA (`zh`, `en`, `ja`, `python`, `rust`).

Implements Iron Law #2 & Engineering Blueprint Section 4.1:
  - Streams directly from ungated HuggingFace Parquet datasets (zero tokenizers / BPE).
  - Applies strict UTF-8 NFC normalization, unifies CRLF/CR -> LF (0x0A), preserves TAB/space,
    and strips disallowed ASCII control bytes (0x00..0x08, 0x0B..0x1F, 0x7F).
  - Validates Python (`lang_id=3`) and Rust (`lang_id=4`) code via `tree-sitter` AST parsing
    (`not root_node.has_error`) and length bounds `[200, 8000]` bytes.
  - Deduplicates documents via `blake2b` hash and performs deterministic leak-free train/val split.
  - Writes exact 16-byte `<QIHH` `.idx` + contiguous raw `.bin` shards bit-compatible with
    `dcpa-data-forge` and `dcpa.data.shard_loader.BinaryShardDataset`.
  - Local M3 Pro validation MUST use `--smoke-quota` (2 MB per lang_id -> ~10 MB total).
  - Full Colab production run (default, without `--smoke-quota`) builds ~450 MB across all 5 domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

# Ensure repo root and dcpa-train-py are importable regardless of working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.data.shard_loader import IDX_RECORD_FMT

# Enable high-speed HuggingFace downloads when running in Colab or local environments.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

# Disallowed control byte table (matches dcpa-data-forge/src/main.rs: keep 0x09, 0x0A, >=0x20 except 0x7F)
_DISALLOWED_CONTROL_CHARS = frozenset(
    [chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)] + [chr(0x7F)]
)
_CONTROL_TRANS = str.maketrans("", "", "".join(_DISALLOWED_CONTROL_CHARS))

# Stage bitmask: bit0=A (1), bit1=B (2), bit2=C (4), bit3=D (8)
STAGE_MASK_NATURAL: int = 0b1111  # Available across A/B/C (+ background context in D)
STAGE_MASK_CODE: int = 0b0111     # Stages A/B/C (syntax-heavy in C)

# Quotas in bytes
SMOKE_BYTES_PER_LANG: int = 2 * 1024 * 1024  # 2 MB per lang_id (~10 MB total)

FULL_QUOTA_BYTES: dict[str, int] = {
    "zh": 90 * 1024 * 1024,      # 90 MB
    "en": 120 * 1024 * 1024,     # 120 MB
    "ja": 70 * 1024 * 1024,      # 70 MB
    "python": 85 * 1024 * 1024,  # 85 MB
    "rust": 85 * 1024 * 1024,    # 85 MB
}  # Total = 450 MB


@dataclass(frozen=True)
class SourceSpec:
    repo_id: str
    config_name: Optional[str]
    data_dir: Optional[str]
    split: str
    text_column: str


@dataclass(frozen=True)
class DomainSpec:
    name: str
    lang_id: int
    stage_mask: int
    is_code: bool
    min_bytes: int
    max_bytes: int
    sources: tuple[SourceSpec, ...]


DOMAIN_SPECS: tuple[DomainSpec, ...] = (
    DomainSpec(
        name="zh",
        lang_id=0,
        stage_mask=STAGE_MASK_NATURAL,
        is_code=False,
        min_bytes=128,
        max_bytes=8192,
        sources=(
            SourceSpec("wikimedia/wikipedia", "20231101.zh", None, "train", "text"),
        ),
    ),
    DomainSpec(
        name="en",
        lang_id=1,
        stage_mask=STAGE_MASK_NATURAL,
        is_code=False,
        min_bytes=128,
        max_bytes=8192,
        sources=(
            SourceSpec("HuggingFaceFW/fineweb-edu", "sample-10BT", None, "train", "text"),
            SourceSpec("wikimedia/wikipedia", "20231101.en", None, "train", "text"),
        ),
    ),
    DomainSpec(
        name="ja",
        lang_id=2,
        stage_mask=STAGE_MASK_NATURAL,
        is_code=False,
        min_bytes=128,
        max_bytes=8192,
        sources=(
            SourceSpec("wikimedia/wikipedia", "20231101.ja", None, "train", "text"),
        ),
    ),
    DomainSpec(
        name="python",
        lang_id=3,
        stage_mask=STAGE_MASK_CODE,
        is_code=True,
        min_bytes=200,
        max_bytes=8000,
        sources=(
            SourceSpec("codeparrot/codeparrot-clean", None, None, "train", "content"),
            SourceSpec("bigcode/the-stack-smol-xl", None, "data/python", "train", "content"),
        ),
    ),
    DomainSpec(
        name="rust",
        lang_id=4,
        stage_mask=STAGE_MASK_CODE,
        is_code=True,
        min_bytes=200,
        max_bytes=8000,
        sources=(
            SourceSpec("ammarnasr/the-stack-rust-clean", None, None, "train", "content"),
            SourceSpec("bigcode/the-stack-smol-xl", None, "data/rust", "train", "content"),
            SourceSpec("Wholesomeisland/rust-the-stack-v2", None, None, "train", "content"),
        ),
    ),
)


class CodeASTValidator:
    """Syntax validator backed by tree-sitter-python and tree-sitter-rust (with safe fallback)."""

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}
        self.tree_sitter_active = False
        try:
            import tree_sitter
            import tree_sitter_python
            import tree_sitter_rust

            self._parsers["python"] = tree_sitter.Parser(
                tree_sitter.Language(tree_sitter_python.language())
            )
            self._parsers["rust"] = tree_sitter.Parser(
                tree_sitter.Language(tree_sitter_rust.language())
            )
            self.tree_sitter_active = True
        except Exception:
            self._parsers = {}
            self.tree_sitter_active = False

    def is_valid(self, lang: str, raw_bytes: bytes) -> bool:
        if len(raw_bytes) < 200 or len(raw_bytes) > 8000:
            return False
        parser = self._parsers.get(lang)
        if parser is not None:
            tree = parser.parse(raw_bytes)
            if tree.root_node.has_error:
                return False
            return True
        return _fallback_balanced_delimiters(raw_bytes, lang)


def _fallback_balanced_delimiters(raw_bytes: bytes, lang: str) -> bool:
    """Fallback bracket balance check if tree-sitter bindings are unavailable."""
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = {"}", ")", "]"}
    stack: list[str] = []
    in_str: Optional[str] = None
    in_comment = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
            i += 1
            continue
        if in_str is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if lang == "python" and ch == "#":
            in_comment = True
            i += 1
            continue
        if lang == "rust" and ch == "/" and i + 1 < n and text[i + 1] == "/":
            in_comment = True
            i += 2
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in closing:
            if not stack or stack.pop() != ch:
                return False
        i += 1
    return len(stack) == 0


def normalize_utf8_text(text: str) -> bytes:
    """Normalize text to strict NFC UTF-8 with LF newlines and zero control bytes."""
    if not text:
        return b""
    nfc = unicodedata.normalize("NFC", text)
    unified = nfc.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = unified.translate(_CONTROL_TRANS).strip()
    if not cleaned:
        return b""
    return (cleaned + "\n").encode("utf-8", errors="strict")


def chunk_natural_document(raw_bytes: bytes, min_bytes: int, max_bytes: int) -> list[bytes]:
    """Split long natural-language documents on paragraph boundaries into valid UTF-8 chunks."""
    if len(raw_bytes) < min_bytes:
        return []
    if len(raw_bytes) <= max_bytes:
        return [raw_bytes]

    text = raw_bytes.decode("utf-8", errors="strict")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[bytes] = []
    buf_parts: list[str] = []
    buf_len = 0

    for para in paragraphs:
        p_bytes = (para + "\n").encode("utf-8")
        if len(p_bytes) > max_bytes:
            # Flush existing buffer first
            if buf_len >= min_bytes:
                chunks.append("\n\n".join(buf_parts).encode("utf-8") + b"\n")
            buf_parts = []
            buf_len = 0
            # Split oversize paragraph cleanly on line or character boundary
            lines = para.splitlines()
            sub_buf: list[str] = []
            sub_len = 0
            for ln in lines:
                ln_b = (ln + "\n").encode("utf-8")
                if len(ln_b) > max_bytes:
                    # Character-level UTF-8 safe slice
                    enc = ln.encode("utf-8")
                    start = 0
                    while start < len(enc):
                        end = min(start + max_bytes - 1, len(enc))
                        # Step back over UTF-8 continuation bytes (10xxxxxx)
                        while end > start and end < len(enc) and (enc[end] & 0xC0) == 0x80:
                            end -= 1
                        piece = enc[start:end] + b"\n"
                        if len(piece) >= min_bytes:
                            chunks.append(piece)
                        start = end
                    continue
                if sub_len + len(ln_b) > max_bytes and sub_buf:
                    joined = ("\n".join(sub_buf) + "\n").encode("utf-8")
                    if len(joined) >= min_bytes:
                        chunks.append(joined)
                    sub_buf = [ln]
                    sub_len = len(ln_b)
                else:
                    sub_buf.append(ln)
                    sub_len += len(ln_b)
            if sub_buf:
                joined = ("\n".join(sub_buf) + "\n").encode("utf-8")
                if len(joined) >= min_bytes:
                    chunks.append(joined)
            continue

        sep_len = 2 if buf_parts else 0
        if buf_len + sep_len + len(p_bytes) > max_bytes and buf_parts:
            joined = ("\n\n".join(buf_parts) + "\n").encode("utf-8")
            if len(joined) >= min_bytes:
                chunks.append(joined)
            buf_parts = [para]
            buf_len = len(p_bytes)
        else:
            buf_parts.append(para)
            buf_len += sep_len + len(p_bytes)

    if buf_parts:
        joined = ("\n\n".join(buf_parts) + "\n").encode("utf-8")
        if len(joined) >= min_bytes:
            chunks.append(joined)
    return chunks


def _stream_source_texts(source: SourceSpec) -> Iterator[str]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": source.split,
        "streaming": True,
    }
    if source.config_name is not None:
        kwargs["name"] = source.config_name
    if source.data_dir is not None:
        kwargs["data_dir"] = source.data_dir

    ds = load_dataset(source.repo_id, **kwargs)
    col = source.text_column
    for row in ds:
        val = row.get(col)
        if isinstance(val, str) and val:
            yield val


def _assert_local_output_dir(out_dir: Path) -> None:
    """Prevent slow FUSE writes directly into Google Drive during build."""
    resolved = str(out_dir.resolve())
    if resolved.startswith("/content/drive"):
        raise ValueError(
            f"Refusing to build binary shards directly on Google Drive FUSE mount ({resolved}). "
            "Build into local NVMe (e.g. /content/nvme_cache/data) and tar/archive to Drive afterwards."
        )


def build_corpus(
    out_dir: str | Path = "data/corpus",
    smoke_quota: bool = False,
    quota_mb_per_lang: Optional[float] = None,
    val_ratio: float = 0.05,
    seed: int = 2026,
    allow_drive_output: bool = False,
) -> dict[str, Any]:
    """Stream, filter, normalize, deduplicate, and write 5-domain DCPA `.bin/.idx` shards."""
    out_path = Path(out_dir)
    if not allow_drive_output:
        _assert_local_output_dir(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    validator = CodeASTValidator()
    mode = "smoke" if smoke_quota else "full"
    quotas: dict[str, int] = {}
    for spec in DOMAIN_SPECS:
        if quota_mb_per_lang is not None:
            quotas[spec.name] = int(quota_mb_per_lang * 1024 * 1024)
        elif smoke_quota:
            quotas[spec.name] = SMOKE_BYTES_PER_LANG
        else:
            quotas[spec.name] = FULL_QUOTA_BYTES[spec.name]

    train_bin_path = out_path / "train.bin"
    train_idx_path = out_path / "train.idx"
    val_bin_path = out_path / "val.bin"
    val_idx_path = out_path / "val.idx"

    val_mod = max(2, int(round(1.0 / max(val_ratio, 1e-3))))
    seen_hashes: set[bytes] = set()
    train_offset = 0
    val_offset = 0
    train_docs = 0
    val_docs = 0
    t_global = time.perf_counter()

    domain_stats: dict[str, dict[str, Any]] = {}

    print(
        f"[build_corpus] mode={mode} | out_dir={out_path} | "
        f"tree_sitter={validator.tree_sitter_active} | "
        f"target_total={sum(quotas.values()) / (1024 * 1024):.1f} MB",
        flush=True,
    )

    with (
        train_bin_path.open("wb") as train_bin,
        train_idx_path.open("wb") as train_idx,
        val_bin_path.open("wb") as val_bin,
        val_idx_path.open("wb") as val_idx,
    ):
        for spec in DOMAIN_SPECS:
            target_bytes = quotas[spec.name]
            t_dom = time.perf_counter()
            collected_bytes = 0
            dom_train_bytes = 0
            dom_val_bytes = 0
            dom_train_docs = 0
            dom_val_docs = 0
            rejected_ast = 0
            rejected_len = 0
            rejected_dup = 0
            sources_used: list[str] = []

            for src in spec.sources:
                if collected_bytes >= target_bytes:
                    break
                src_label = f"{src.repo_id}" + (
                    f":{src.config_name or src.data_dir}" if (src.config_name or src.data_dir) else ""
                )
                print(
                    f"  -> [{spec.name} | lang_id={spec.lang_id}] streaming {src_label} "
                    f"(target={target_bytes / (1024 * 1024):.2f} MB)...",
                    flush=True,
                )
                stream = None
                try:
                    stream = _stream_source_texts(src)
                    sources_used.append(src_label)
                    for raw_text in stream:
                        if collected_bytes >= target_bytes:
                            break
                        norm = normalize_utf8_text(raw_text)
                        if not norm:
                            rejected_len += 1
                            continue

                        if spec.is_code:
                            if not (spec.min_bytes <= len(norm) <= spec.max_bytes):
                                rejected_len += 1
                                continue
                            if not validator.is_valid(spec.name, norm):
                                rejected_ast += 1
                                continue
                            candidates = [norm]
                        else:
                            candidates = chunk_natural_document(norm, spec.min_bytes, spec.max_bytes)
                            if not candidates:
                                rejected_len += 1
                                continue

                        for doc_bytes in candidates:
                            if collected_bytes >= target_bytes:
                                break
                            digest = hashlib.blake2b(
                                doc_bytes,
                                digest_size=16,
                                person=f"dcpa{seed}".encode("ascii")[:16],
                            ).digest()
                            if digest in seen_hashes:
                                rejected_dup += 1
                                continue
                            seen_hashes.add(digest)

                            # Guarantee at least 1 val doc and 1 train doc per domain, then deterministic hash split
                            bucket = int.from_bytes(digest[:4], "little") % val_mod
                            if dom_val_docs == 0 and dom_train_docs >= 1:
                                is_val = True
                            elif dom_train_docs == 0:
                                is_val = False
                            else:
                                is_val = bucket == 0

                            rec_len = len(doc_bytes)
                            if is_val:
                                val_bin.write(doc_bytes)
                                val_idx.write(
                                    struct.pack(
                                        IDX_RECORD_FMT,
                                        val_offset,
                                        rec_len,
                                        spec.lang_id,
                                        spec.stage_mask,
                                    )
                                )
                                val_offset += rec_len
                                val_docs += 1
                                dom_val_bytes += rec_len
                                dom_val_docs += 1
                            else:
                                train_bin.write(doc_bytes)
                                train_idx.write(
                                    struct.pack(
                                        IDX_RECORD_FMT,
                                        train_offset,
                                        rec_len,
                                        spec.lang_id,
                                        spec.stage_mask,
                                    )
                                )
                                train_offset += rec_len
                                train_docs += 1
                                dom_train_bytes += rec_len
                                dom_train_docs += 1

                            collected_bytes += rec_len
                except Exception as exc:
                    print(
                        f"  [WARN] Source {src_label} stopped early ({type(exc).__name__}: {exc}); "
                        "trying fallback source if available.",
                        flush=True,
                    )
                    continue
                finally:
                    if stream is not None and hasattr(stream, "close"):
                        try:
                            stream.close()
                        except Exception:
                            pass

            elapsed_dom = time.perf_counter() - t_dom
            if collected_bytes < int(target_bytes * 0.90):
                raise RuntimeError(
                    f"Domain {spec.name} (lang_id={spec.lang_id}) only collected "
                    f"{collected_bytes / (1024 * 1024):.2f} MB < 90% of target "
                    f"{target_bytes / (1024 * 1024):.2f} MB."
                )

            domain_stats[spec.name] = {
                "lang_id": spec.lang_id,
                "stage_mask": spec.stage_mask,
                "target_bytes": target_bytes,
                "total_bytes": collected_bytes,
                "train_bytes": dom_train_bytes,
                "val_bytes": dom_val_bytes,
                "train_docs": dom_train_docs,
                "val_docs": dom_val_docs,
                "rejected_ast": rejected_ast,
                "rejected_len": rejected_len,
                "rejected_dup": rejected_dup,
                "sources_used": sources_used,
                "elapsed_sec": round(elapsed_dom, 2),
            }
            print(
                f"  [OK] {spec.name} (id={spec.lang_id}): "
                f"{collected_bytes / (1024 * 1024):.2f} MB "
                f"(train={dom_train_docs}, val={dom_val_docs}, "
                f"rej_ast={rejected_ast}, rej_dup={rejected_dup}) in {elapsed_dom:.1f}s",
                flush=True,
            )

    def _file_blake2b(path: Path) -> str:
        h = hashlib.blake2b()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    total_elapsed = time.perf_counter() - t_global
    meta = {
        "mode": mode,
        "seed": seed,
        "val_ratio": val_ratio,
        "tree_sitter_active": validator.tree_sitter_active,
        "total_bytes": train_offset + val_offset,
        "train_bytes": train_offset,
        "val_bytes": val_offset,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "domains": domain_stats,
        "checksums_blake2b": {
            "train.bin": _file_blake2b(train_bin_path),
            "train.idx": _file_blake2b(train_idx_path),
            "val.bin": _file_blake2b(val_bin_path),
            "val.idx": _file_blake2b(val_idx_path),
        },
        "elapsed_sec": round(total_elapsed, 2),
    }

    meta_path = out_path / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[build_corpus] DONE: {(train_offset + val_offset) / (1024 * 1024):.2f} MB "
        f"({train_docs} train + {val_docs} val docs) in {total_elapsed:.1f}s -> {out_path}",
        flush=True,
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream and build DCPA 5-domain raw UTF-8 .bin/.idx corpus."
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="data/corpus",
        help="Local NVMe output directory (must NOT be on /content/drive).",
    )
    ap.add_argument(
        "--smoke-quota",
        action="store_true",
        help="Local M3 Pro smoke mode: stream 2 MB per lang_id (~10 MB total).",
    )
    ap.add_argument(
        "--quota-mb-per-lang",
        type=float,
        default=None,
        help="Optional explicit MB quota per language domain (overrides mode default).",
    )
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    build_corpus(
        out_dir=args.out_dir,
        smoke_quota=args.smoke_quota,
        quota_mb_per_lang=args.quota_mb_per_lang,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    # Flush streams and exit cleanly without waiting on orphaned hf-xet/pyarrow background threads
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
