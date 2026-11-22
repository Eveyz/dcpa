"""Zero-padding Constant-Byte-Budget Ragged UTF-8 Byte DataLoader for DCPA.

Implements Deliverable 2 of the DCPA Data Infrastructure & Curriculum Pipeline:
  - Exact 16-byte little-endian `.idx` schema (`offset:<u8, length:<u4, lang_id:<u1,
    source_id:<u1, max_depth:<u2`) read via zero-copy `np.memmap(..., mode="r")`.
  - Lazy per-worker memmap opening for zero RAM bloat across multiprocessing DataLoader workers.
  - Strictly enforces Iron Law #3.1: blocks direct mmap on `/content/drive` FUSE mounts.
  - `utf8_safe_slice(buf, start, max_len)`: guarantees zero split multi-byte UTF-8 characters
    (never stops on a UTF-8 continuation byte `0x80..0xBF`), allowing seamless contiguous windowing.
  - `RaggedByteDataset` & `RaggedByteBatchSampler`: constant-byte-budget dynamic varlen packing
    (zero `0x00` padding, zero `<BOS>`/`<EOS>` bytes) with optional online Stage-D Wug Episode mixing.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from data.wug_generator import sample_wug_episode_fast

# Exact 16-byte little-endian structured record layout for {split}.idx
IDX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),     # uint64 byte start offset in .bin
        ("length", "<u4"),     # uint32 exact UTF-8 byte length (> 0)
        ("lang_id", "<u1"),    # uint8 domain ID in 0..4 (0=en, 1=zh, 2=ja, 3=python, 4=rust)
        ("source_id", "<u1"),  # uint8 sub-source identifier
        ("max_depth", "<u2"),  # uint16 max bracket depth (0 for natural language, 1..16 for code)
    ]
)
IDX_RECORD_FMT = "<QIBBH"
IDX_RECORD_SIZE = int(IDX_DTYPE.itemsize)  # 16 bytes
assert IDX_RECORD_SIZE == 16


def assert_not_drive_mmap(path: str | Path) -> None:
    """Block direct mmap on Google Drive FUSE mount paths (Iron Law #3.1)."""
    p = Path(path)
    raw_s = str(p)
    resolved_s = str(p.resolve()) if p.exists() else raw_s
    if raw_s.startswith("/content/drive") or resolved_s.startswith("/content/drive"):
        raise ValueError(
            f"Refusing to np.memmap directly on Google Drive FUSE mount ({p}). "
            "Stage .bin/.idx shards under local NVMe (e.g. /content/nvme_cache/data) first."
        )


def utf8_safe_end(buf: np.ndarray, start: int, max_len: int) -> int:
    """Compute the largest valid UTF-8 character boundary `end <= min(start + max_len, len(buf))`.

    If `end < len(buf)` and `buf[end]` is a UTF-8 continuation byte (`(b & 0xC0) == 0x80`),
    steps backward (at most 3 bytes) to the start byte of that character so `buf[start:end]`
    is guaranteed 100% valid UTF-8 and the next window can resume seamlessly at `end`.
    """
    n = int(len(buf))
    if start >= n or max_len <= 0:
        return min(start, n)
    end = min(start + max_len, n)
    if end < n:
        steps = 0
        while end > start and steps < 4 and (int(buf[end]) & 0xC0) == 0x80:
            end -= 1
            steps += 1
    return end


def utf8_safe_slice(buf: np.ndarray, start: int, max_len: int) -> np.ndarray:
    """Return a UTF-8 character-boundary-aligned 1D `uint8` slice `buf[start:end]`."""
    end = utf8_safe_end(buf, start, max_len)
    return buf[start:end]


def _normalize_domain_weights(
    available_langs: Sequence[int],
    domain_weights: Optional[dict[int, float]],
) -> tuple[list[int], list[float]]:
    langs = sorted(int(lid) for lid in available_langs)
    if not langs:
        raise ValueError("Index contains zero documents across all domains.")
    if domain_weights is None:
        w = [1.0 / len(langs)] * len(langs)
        return langs, w
    raw = [max(0.0, float(domain_weights.get(lid, 0.0))) for lid in langs]
    total = sum(raw)
    if total <= 0.0:
        raise ValueError(f"domain_weights assigns zero total weight to available lang_ids {langs}")
    cum: list[float] = []
    acc = 0.0
    for val in raw:
        acc += val / total
        cum.append(acc)
    cum[-1] = 1.0
    return langs, cum


class RaggedByteBatchSampler(Sampler[list[tuple[int, int, int]]]):
    """Constant-Byte-Budget batch sampler yielding lists of `(doc_idx, win_start, win_len)` tuples.

    Groups variable-length UTF-8-safe document windows up to `target_batch_bytes` with zero padding.
    """

    def __init__(
        self,
        bin_path: str | Path,
        idx_path: str | Path,
        target_batch_bytes: int = 131072,
        max_doc_bytes: int = 2048,
        domain_weights: Optional[dict[int, float]] = None,
        num_batches: int = 1000,
        seed: int = 42,
        allow_drive_fuse: bool = False,
    ) -> None:
        if target_batch_bytes < 64:
            raise ValueError(f"target_batch_bytes must be >= 64, got {target_batch_bytes}")
        if max_doc_bytes < 16:
            raise ValueError(f"max_doc_bytes must be >= 16, got {max_doc_bytes}")

        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        if not allow_drive_fuse:
            assert_not_drive_mmap(self.bin_path)
            assert_not_drive_mmap(self.idx_path)

        self.target_batch_bytes = int(target_batch_bytes)
        self.max_doc_bytes = int(max_doc_bytes)
        self.num_batches = int(num_batches)
        self.seed = int(seed)

        idx_arr = np.memmap(self.idx_path, dtype=IDX_DTYPE, mode="r")
        self._offsets = np.array(idx_arr["offset"], dtype=np.int64, copy=True)
        self._lengths = np.array(idx_arr["length"], dtype=np.int64, copy=True)
        self._lang_ids = np.array(idx_arr["lang_id"], dtype=np.int64, copy=True)
        del idx_arr

        self._by_lang: dict[int, np.ndarray] = {}
        for lid in sorted(set(self._lang_ids.tolist())):
            self._by_lang[int(lid)] = np.nonzero(self._lang_ids == lid)[0]

        self._langs, self._cum_weights = _normalize_domain_weights(
            list(self._by_lang.keys()), domain_weights
        )

    def __len__(self) -> int:
        return self.num_batches

    def _pick_lang(self, rng: random.Random) -> int:
        u = rng.random()
        for lid, cw in zip(self._langs, self._cum_weights):
            if u <= cw:
                return lid
        return self._langs[-1]

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        rng = random.Random(self.seed)
        mm = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
        try:
            for _ in range(self.num_batches):
                batch_specs: list[tuple[int, int, int]] = []
                used = 0
                while used < self.target_batch_bytes:
                    rem = self.target_batch_bytes - used
                    if rem < 16:
                        break
                    lid = self._pick_lang(rng)
                    pool = self._by_lang[lid]
                    doc_idx = int(pool[rng.randrange(len(pool))])
                    off = int(self._offsets[doc_idx])
                    doc_len = int(self._lengths[doc_idx])
                    doc_view = mm[off : off + doc_len]

                    if doc_len <= self.max_doc_bytes:
                        w_start = 0
                        w_end = doc_len
                    else:
                        # Choose a random UTF-8-safe window start (align start forward over continuation bytes)
                        max_start = doc_len - min(self.max_doc_bytes, doc_len)
                        w_start = rng.randrange(0, max_start + 1)
                        steps = 0
                        while w_start < doc_len and steps < 4 and (int(doc_view[w_start]) & 0xC0) == 0x80:
                            w_start += 1
                            steps += 1
                        if w_start >= doc_len:
                            w_start = 0
                        w_end = utf8_safe_end(doc_view, w_start, self.max_doc_bytes)

                    seg_len = w_end - w_start
                    if seg_len <= 0:
                        continue
                    if seg_len > rem:
                        if rem < 64 and batch_specs:
                            break
                        w_end = utf8_safe_end(doc_view, w_start, rem)
                        seg_len = w_end - w_start
                        if seg_len <= 0:
                            break

                    batch_specs.append((doc_idx, w_start, seg_len))
                    used += seg_len
                if batch_specs:
                    yield batch_specs
        finally:
            del mm


class RaggedByteDataset(Dataset):
    """High-throughput lazy-mmap Ragged UTF-8 dataset producing Constant-Byte-Budget batches.

    Designed for both `num_workers=0` and `num_workers=4` (`DataLoader(ds, batch_size=None, ...)`):
      - `.bin` is memory-mapped lazily inside each worker process (zero memory duplication).
      - Packs variable-length documents (and optional Stage-D online Wug Episodes at `wug_ratio`)
        into a single contiguous `uint8` buffer `Total_T <= target_batch_bytes` with ZERO padding.
      - Every returned document segment is strictly UTF-8 character-boundary aligned.
    """

    def __init__(
        self,
        bin_path: str | Path,
        idx_path: str | Path,
        target_batch_bytes: int = 131072,
        max_doc_bytes: int = 2048,
        domain_weights: Optional[dict[int, float]] = None,
        wug_ratio: float = 0.0,
        num_batches: int = 10000,
        seed: int = 42,
        allow_drive_fuse: bool = False,
    ) -> None:
        if target_batch_bytes < 64:
            raise ValueError(f"target_batch_bytes must be >= 64, got {target_batch_bytes}")
        if max_doc_bytes < 16:
            raise ValueError(f"max_doc_bytes must be >= 16, got {max_doc_bytes}")
        if not (0.0 <= wug_ratio <= 1.0):
            raise ValueError(f"wug_ratio must be in [0.0, 1.0], got {wug_ratio}")

        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        if not allow_drive_fuse:
            assert_not_drive_mmap(self.bin_path)
            assert_not_drive_mmap(self.idx_path)

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Missing binary shard: {self.bin_path}")
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Missing index shard: {self.idx_path}")

        self.target_batch_bytes = int(target_batch_bytes)
        self.max_doc_bytes = int(max_doc_bytes)
        self.wug_ratio = float(wug_ratio)
        self.num_batches = int(num_batches)
        self.seed = int(seed)

        # Read lightweight index metadata once; .bin mmap is opened lazily per worker process.
        idx_mm = np.memmap(self.idx_path, dtype=IDX_DTYPE, mode="r")
        offsets = np.array(idx_mm["offset"], dtype=np.int64, copy=True)
        lengths = np.array(idx_mm["length"], dtype=np.int64, copy=True)
        lang_ids = np.array(idx_mm["lang_id"], dtype=np.int64, copy=True)
        del idx_mm

        self.num_docs = int(len(offsets))
        self._offsets = offsets
        self._lengths = lengths
        self._lang_ids = lang_ids

        self._lang_offsets: dict[int, np.ndarray] = {}
        self._lang_lengths: dict[int, np.ndarray] = {}
        for lid in sorted(set(lang_ids.tolist())):
            mask = lang_ids == lid
            self._lang_offsets[int(lid)] = offsets[mask]
            self._lang_lengths[int(lid)] = lengths[mask]

        self._langs, self._cum_weights = _normalize_domain_weights(
            list(self._lang_offsets.keys()), domain_weights
        )
        # Lazy per-process mmap handle (never pickled across DataLoader workers)
        self._bin_mmap: Optional[np.memmap] = None
        self._mmap_pid: int = -1

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_bin_mmap"] = None
        state["_mmap_pid"] = -1
        return state

    def _get_mmap(self) -> np.memmap:
        pid = os.getpid()
        if self._bin_mmap is None or self._mmap_pid != pid:
            self._bin_mmap = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
            self._mmap_pid = pid
        return self._bin_mmap

    def close(self) -> None:
        self._bin_mmap = None
        self._mmap_pid = -1

    def __len__(self) -> int:
        return self.num_batches

    def _pick_lang(self, rng: random.Random) -> int:
        u = rng.random()
        for lid, cw in zip(self._langs, self._cum_weights):
            if u <= cw:
                return lid
        return self._langs[-1]

    def pack_batch_from_specs(
        self,
        specs: Sequence[tuple[int, int, int]],
    ) -> dict[str, torch.Tensor]:
        """Materialize a zero-padding Ragged batch from `(doc_idx, win_start, win_len)` specs."""
        mm = self._get_mmap()
        total_t = sum(seg_len for _, _, seg_len in specs)
        n_docs = len(specs)

        x_buf = np.empty(total_t, dtype=np.uint8)
        byte_langs = np.empty(total_t, dtype=np.int64)
        cu_seqlens = np.empty(n_docs + 1, dtype=np.int32)
        doc_langs = np.empty(n_docs, dtype=np.int64)
        is_doc_start = np.zeros(total_t, dtype=np.bool_)

        cu_seqlens[0] = 0
        cursor = 0
        for i, (doc_idx, w_start, seg_len) in enumerate(specs):
            off = int(self._offsets[doc_idx]) + w_start
            lid = int(self._lang_ids[doc_idx])
            x_buf[cursor : cursor + seg_len] = mm[off : off + seg_len]
            byte_langs[cursor : cursor + seg_len] = lid
            is_doc_start[cursor] = True
            doc_langs[i] = lid
            cursor += seg_len
            cu_seqlens[i + 1] = cursor

        return {
            "x_bytes": torch.from_numpy(x_buf),
            "cu_seqlens": torch.from_numpy(cu_seqlens),
            "is_doc_start": torch.from_numpy(is_doc_start),
            "doc_lang_ids": torch.from_numpy(doc_langs),
            "byte_lang_ids": torch.from_numpy(byte_langs),
            "loss_mask": torch.ones(total_t, dtype=torch.bool),
            "wug_target_mask": torch.zeros(total_t, dtype=torch.bool),
        }

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        """Pack one Constant-Byte-Budget zero-padding Ragged batch (`Total_T <= target_batch_bytes`)."""
        winfo = get_worker_info()
        worker_id = winfo.id if winfo is not None else 0
        rng = random.Random(self.seed + batch_idx * 1000003 + worker_id * 97)
        mm = self._get_mmap()

        budget = self.target_batch_bytes
        max_doc = self.max_doc_bytes
        wug_p = self.wug_ratio

        x_buf = np.empty(budget, dtype=np.uint8)
        byte_langs = np.empty(budget, dtype=np.int64)
        loss_mask = np.ones(budget, dtype=np.bool_)
        wug_mask = np.zeros(budget, dtype=np.bool_)

        cu_list: list[int] = [0]
        doc_lang_list: list[int] = []

        cursor = 0
        # Pre-bind hot attributes to locals for maximum Python loop speed
        lang_offsets = self._lang_offsets
        lang_lengths = self._lang_lengths
        langs = self._langs
        cum_weights = self._cum_weights
        n_langs = len(langs)
        rand_fn = rng.random
        randrange_fn = rng.randrange

        while cursor < budget:
            rem = budget - cursor
            if rem < 16:
                break

            # Select language domain according to curriculum weights
            if n_langs == 1:
                lid = langs[0]
            else:
                u = rand_fn()
                lid = langs[-1]
                for i in range(n_langs):
                    if u <= cum_weights[i]:
                        lid = langs[i]
                        break

            # Optional Stage-D online Wug Episode injection
            if wug_p > 0.0 and rand_fn() < wug_p:
                raw_b, t1_len, tgt_s, tgt_e = sample_wug_episode_fast(rng, lid)
                ep_len = len(raw_b)
                if ep_len <= rem:
                    x_buf[cursor : cursor + ep_len] = np.frombuffer(raw_b, dtype=np.uint8)
                    byte_langs[cursor : cursor + ep_len] = lid
                    loss_mask[cursor : cursor + t1_len] = False
                    wug_mask[cursor + tgt_s : cursor + tgt_e] = True
                    doc_lang_list.append(lid)
                    cursor += ep_len
                    cu_list.append(cursor)
                    continue
                if doc_lang_list and rem < 256:
                    break
                # Fall through to a regular doc slice if Wug episode exceeded tiny tail budget

            offs = lang_offsets[lid]
            lens = lang_lengths[lid]
            d_i = randrange_fn(len(offs))
            off = int(offs[d_i])
            doc_len = int(lens[d_i])
            doc_view = mm[off : off + doc_len]

            if doc_len <= max_doc:
                w_start = 0
                w_end = doc_len
            else:
                # Pick a random UTF-8-safe window across the full document
                max_start = doc_len - max_doc
                w_start = randrange_fn(max_start + 1)
                steps = 0
                while w_start < doc_len and steps < 4 and (int(doc_view[w_start]) & 0xC0) == 0x80:
                    w_start += 1
                    steps += 1
                if w_start >= doc_len:
                    w_start = 0
                w_end = utf8_safe_end(doc_view, w_start, max_doc)

            seg_len = w_end - w_start
            if seg_len <= 0:
                continue

            if seg_len > rem:
                if rem < 64 and doc_lang_list:
                    break
                w_end = utf8_safe_end(doc_view, w_start, rem)
                seg_len = w_end - w_start
                if seg_len <= 0:
                    break

            x_buf[cursor : cursor + seg_len] = doc_view[w_start:w_end]
            byte_langs[cursor : cursor + seg_len] = lid
            doc_lang_list.append(lid)
            cursor += seg_len
            cu_list.append(cursor)

        total_t = cursor
        cu_arr = np.asarray(cu_list, dtype=np.int32)
        doc_lang_arr = np.asarray(doc_lang_list, dtype=np.int64)
        is_doc_start = np.zeros(total_t, dtype=np.bool_)
        if len(cu_arr) > 1:
            is_doc_start[cu_arr[:-1]] = True

        return {
            "x_bytes": torch.from_numpy(x_buf[:total_t].copy()),
            "cu_seqlens": torch.from_numpy(cu_arr),
            "is_doc_start": torch.from_numpy(is_doc_start),
            "doc_lang_ids": torch.from_numpy(doc_lang_arr),
            "byte_lang_ids": torch.from_numpy(byte_langs[:total_t].copy()),
            "loss_mask": torch.from_numpy(loss_mask[:total_t].copy()),
            "wug_target_mask": torch.from_numpy(wug_mask[:total_t].copy()),
        }


def create_ragged_dataloader(
    bin_path: str | Path,
    idx_path: str | Path,
    target_batch_bytes: int = 131072,
    max_doc_bytes: int = 2048,
    domain_weights: Optional[dict[int, float]] = None,
    wug_ratio: float = 0.0,
    num_batches: int = 10000,
    num_workers: int = 0,
    seed: int = 42,
    allow_drive_fuse: bool = False,
) -> DataLoader:
    """Create a multi-worker DataLoader yielding pre-packed Constant-Byte-Budget Ragged dicts."""
    ds = RaggedByteDataset(
        bin_path=bin_path,
        idx_path=idx_path,
        target_batch_bytes=target_batch_bytes,
        max_doc_bytes=max_doc_bytes,
        domain_weights=domain_weights,
        wug_ratio=wug_ratio,
        num_batches=num_batches,
        seed=seed,
        allow_drive_fuse=allow_drive_fuse,
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
