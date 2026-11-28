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

from collections import deque
from dataclasses import dataclass
import heapq
import os
import random
import sys
import threading
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
if _TRAIN_PY.is_dir() and str(_TRAIN_PY) not in sys.path:
    sys.path.insert(0, str(_TRAIN_PY))

from data import wug_generator
from data.cognitive_v2_sources import V3_FLAT_SOURCE_STAGE_TIERS, prechunk_document
from data.wug_generator import sample_wug_episode_fast
from dcpa.interfaces import StaticGridBatch, decode_fixed_point_loss_mask

# Exact 16-byte little-endian structured record layout for {split}.idx (Spec 3.2: 100% backward-compatible!)
# Old `<QIBBH` files with max_depth in 0..16 store 0x00 in byte 15, which reads cleanly as style_id=0 (neutral).
IDX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),     # uint64 byte start offset in .bin
        ("length", "<u4"),     # uint32 exact UTF-8 byte length (> 0)
        ("lang_id", "<u1"),    # uint8 domain ID in 0..5 (0=en, 1=zh, 2=formal, 3=python, 4=sql_struct, 5=javascript)
        ("source_id", "<u1"),  # uint8 sub-source identifier (11 = latent implicit reasoning -> ponder_tax_scale=0.0)
        ("max_depth", "<u1"),  # uint8 max bracket depth (0 for natural language, 1..16 for code)
        ("style_id", "<u1"),   # uint8 style subspace ID (0=rigorous, 1=concise_memo, 2=expository, 3=literary, 4=strict_eng)
    ]
)
IDX_RECORD_FMT = "<QIBBBB"
IDX_RECORD_SIZE = int(IDX_DTYPE.itemsize)  # 16 bytes
assert IDX_RECORD_SIZE == 16
LATENT_REASONING_SOURCE_ID = 11

# Deterministic developmental stage difficulty tier (1, 2, or 3) per `.idx` `source_id`:
#   - Tier 1: Stage-1 clean sensorimotor rhythm, short prose/literature & minimal stdlib/ES6/SQL primitives
#   - Tier 2: Stage-2 deductive logic edges (FOLIO/LogiQA/FineMath), multi-paragraph discourse & modular AST
#   - Tier 3: Stage-3 deep slow-thinking, long essays/criticism & complex multi-function engineering modules
SOURCE_STAGE_TIERS: dict[int, int] = {
    1: 1,   # nilq/babylm-10M
    2: 1,   # roneneldan/TinyStories
    3: 2,   # opencsg/Fineweb-Edu-Chinese-V2.1:4_5
    4: 2,   # opencsg/Fineweb-Edu-Chinese-V3
    9: 2,   # nampdn-ai/tiny-codes
    10: 2,  # TinyPixel/tiny-codes
    12: 2,  # HuggingFaceFW/fineweb-edu:sample-10BT(>=4)
    13: 3,  # open-web-math/open-web-math
    15: 3,  # git:python-ecosystem
    17: 3,  # git:javascript-ecosystem
    21: 1,  # HuggingFaceFW/finephrase (Stage 1 clean rephrased prose)
    22: 1,  # storytracer/US-PD-Books & short classic prose (Stage 1 punctuation/rhythm)
    23: 1,  # zh_prose_short (Stage 1 Chinese literary/expository short paragraphs)
    24: 1,  # smollm-corpus:python-edu & cpython stdlib short primitives (Stage 1)
    25: 1,  # es6_clean_short (Stage 1 JavaScript short primitives)
    26: 1,  # sql_json_basic (Stage 1 SQL/JSON basic syntax primitives)
    27: 2,  # tasksource/folio + proofwriter + FOL deductive chains (Stage 2)
    28: 2,  # logiqa + reclor deductive reading (Stage 2)
    29: 2,  # HuggingFaceTB/finemath-4plus proofs (Stage 2)
    30: 2,  # HuggingFaceTB/cosmopedia-v2 humanities/rhetoric/STEM chapters (Stage 2)
    31: 2,  # BAAI/CCI3-HQ & high-score Chinese deep commentary (Stage 2)
    32: 2,  # tokyotech-llm/swallow-code-v2 rewritten Python modules (Stage 2)
    33: 2,  # b-mc2/sql-create-context & JSON schema specs (Stage 2)
    34: 2,  # js_modular_jsdoc (Stage 2 modular JS/TS)
    35: 3,  # long_essays_and_criticism (Stage 3)
    36: 3,  # complex_formal_proofs (Stage 3)
    37: 3,  # sql_cte_analytic_and_complex_schemas (Stage 3)
    **V3_FLAT_SOURCE_STAGE_TIERS,
}


def infer_doc_stage_tiers(source_ids: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Compute stage tier in `{1, 2, 3}` for every document record in `.idx`."""
    tiers = np.where(lengths <= 640, 1, np.where(lengths <= 2048, 2, 3)).astype(np.int64)
    for sid, tier in SOURCE_STAGE_TIERS.items():
        mask = source_ids == sid
        if np.any(mask):
            tiers[mask] = int(tier)
    return tiers


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
        cand_mask = self.bin_path.with_suffix(".loss_mask.bin")
        self.mask_path: Optional[Path] = (
            cand_mask
            if (cand_mask.is_file() and self.bin_path.is_file() and cand_mask.stat().st_size == self.bin_path.stat().st_size)
            else None
        )
        if not allow_drive_fuse:
            assert_not_drive_mmap(self.bin_path)
            assert_not_drive_mmap(self.idx_path)
            if self.mask_path is not None:
                assert_not_drive_mmap(self.mask_path)

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
        length_bucketing: bool = True,
        bucket_sync_world_size: int = 1,
        max_stage_tier: int = 0,
        anchor_doc_start: bool = False,
        wug_mode: str = "classic",
    ) -> None:
        if target_batch_bytes < 64:
            raise ValueError(f"target_batch_bytes must be >= 64, got {target_batch_bytes}")
        if max_doc_bytes < 16:
            raise ValueError(f"max_doc_bytes must be >= 16, got {max_doc_bytes}")
        if not (0.0 <= wug_ratio <= 1.0):
            raise ValueError(f"wug_ratio must be in [0.0, 1.0], got {wug_ratio}")

        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        cand_mask = self.bin_path.with_suffix(".loss_mask.bin")
        self.mask_path: Optional[Path] = (
            cand_mask
            if (cand_mask.is_file() and self.bin_path.is_file() and cand_mask.stat().st_size == self.bin_path.stat().st_size)
            else None
        )
        if not allow_drive_fuse:
            assert_not_drive_mmap(self.bin_path)
            assert_not_drive_mmap(self.idx_path)
            if self.mask_path is not None:
                assert_not_drive_mmap(self.mask_path)

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Missing binary shard: {self.bin_path}")
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Missing index shard: {self.idx_path}")

        self.target_batch_bytes = int(target_batch_bytes)
        self.max_doc_bytes = int(max_doc_bytes)
        self.wug_ratio = float(wug_ratio)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.length_bucketing = bool(length_bucketing)
        self.bucket_sync_world_size = max(1, int(bucket_sync_world_size))
        self.max_stage_tier = max(0, int(max_stage_tier))
        self.anchor_doc_start = bool(anchor_doc_start)
        self.wug_mode = str(wug_mode or "classic")

        # Read lightweight index metadata once; .bin mmap is opened lazily per worker process.
        idx_mm = np.memmap(self.idx_path, dtype=IDX_DTYPE, mode="r")
        offsets = np.array(idx_mm["offset"], dtype=np.int64, copy=True)
        lengths = np.array(idx_mm["length"], dtype=np.int64, copy=True)
        lang_ids = np.array(idx_mm["lang_id"], dtype=np.int64, copy=True)
        source_ids = np.array(idx_mm["source_id"], dtype=np.int64, copy=True)
        style_ids = np.array(idx_mm["style_id"], dtype=np.int64, copy=True)
        del idx_mm

        self.num_docs = int(len(offsets))
        self._offsets = offsets
        self._lengths = lengths
        self._lang_ids = lang_ids
        self._source_ids = source_ids
        self._style_ids = style_ids
        doc_tiers = infer_doc_stage_tiers(source_ids, lengths) if self.max_stage_tier > 0 else None

        self._lang_offsets: dict[int, np.ndarray] = {}
        self._lang_lengths: dict[int, np.ndarray] = {}
        self._lang_sources: dict[int, np.ndarray] = {}
        self._lang_styles: dict[int, np.ndarray] = {}
        for lid in sorted(set(lang_ids.tolist())):
            mask = lang_ids == lid
            if doc_tiers is not None:
                tier_mask = mask & (doc_tiers <= self.max_stage_tier)
                if np.any(tier_mask):
                    mask = tier_mask
            self._lang_offsets[int(lid)] = offsets[mask]
            self._lang_lengths[int(lid)] = lengths[mask]
            self._lang_sources[int(lid)] = source_ids[mask]
            self._lang_styles[int(lid)] = style_ids[mask]

        self._langs, self._cum_weights = _normalize_domain_weights(
            list(self._lang_offsets.keys()), domain_weights
        )
        # Precompute Bayes-exact length-bucketed sub-pools so every micro-batch packs
        # homogeneous-length documents (eliminating straggler tails) AND synchronizes `m_max`
        # across all DDP ranks via `sync_group = batch_idx // bucket_sync_world_size`!
        self._use_tiered_buckets: bool = False
        self._tier_cum_probs: list[float] = []
        self._tier_langs: list[list[int]] = []
        self._tier_lang_cum: list[list[float]] = []
        self._tier_pools: list[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []
        if self.length_bucketing and self.wug_ratio == 0.0 and self.max_doc_bytes > 512 and len(self._langs) > 0:
            self._build_length_tier_tables()

        # Lazy per-process mmap handles (never pickled across DataLoader workers)
        self._bin_mmap: Optional[np.memmap] = None
        self._mask_mmap: Optional[np.memmap] = None
        self._mmap_pid: int = -1
        self._mmap_lock = threading.Lock()

    def _build_length_tier_tables(self) -> None:
        """Partition each domain into power-of-2 byte-length bands preserving exact marginal P(lid, doc).

        Crucial Bayes invariance under Constant-Byte-Budget batch packing:
          - Within band `k`, language `lid` is drawn with doc-count conditional mass
            `P(lid | band k) propto p_lang[lid] * (N_{lid,k} / N_{lid})`.
          - One draw from band `k` has expected clipped byte length `mu_k`, so a batch of fixed
            byte budget `B` packs `~ B / mu_k` documents.
          - Setting outer tier probability `P(band k) propto m_k * mu_k` (expected clipped byte mass
            in band `k`) cancels `B / mu_k` identically, guaranteeing 0.0% length or domain bias!
        """
        p_lang: dict[int, float] = {}
        prev_c = 0.0
        for lid, cw in zip(self._langs, self._cum_weights):
            p_lang[lid] = max(0.0, float(cw) - prev_c)
            prev_c = float(cw)

        # Length band edges up to max_doc_bytes: (0, 512], (512, 1024], (1024, 2048], (2048, inf)
        raw_edges = [0, 512, 1024, 2048, 1 << 60]
        bands: list[tuple[int, int]] = []
        for lo, hi in zip(raw_edges[:-1], raw_edges[1:]):
            if lo >= self.max_doc_bytes:
                break
            bands.append((lo, hi))
        if len(bands) <= 1:
            return

        tier_masses: list[float] = []
        tier_langs: list[list[int]] = []
        tier_lang_cum: list[list[float]] = []
        tier_pools: list[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []

        for lo, hi in bands:
            t_doc_mass = 0.0
            t_byte_mass = 0.0
            l_masses: list[tuple[int, float]] = []
            pools_k: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for lid in self._langs:
                lens_l = self._lang_lengths[lid]
                n_tot = len(lens_l)
                if n_tot == 0:
                    continue
                eff_lens = np.minimum(lens_l, self.max_doc_bytes)
                if hi >= self.max_doc_bytes:
                    m_band = eff_lens > lo
                else:
                    m_band = (eff_lens > lo) & (eff_lens <= hi)
                cnt = int(np.count_nonzero(m_band))
                if cnt == 0:
                    continue
                doc_p = p_lang[lid] * (float(cnt) / float(n_tot))
                byte_p = p_lang[lid] * (float(eff_lens[m_band].astype(np.float64).sum()) / float(n_tot))
                if doc_p <= 0.0 or byte_p <= 0.0:
                    continue
                t_doc_mass += doc_p
                t_byte_mass += byte_p
                l_masses.append((lid, doc_p))
                pools_k[lid] = (
                    self._lang_offsets[lid][m_band],
                    lens_l[m_band],
                    self._lang_sources[lid][m_band],
                    self._lang_styles[lid][m_band],
                )
            if t_doc_mass > 0.0 and t_byte_mass > 0.0 and l_masses:
                tier_masses.append(t_byte_mass)
                t_lids = [x[0] for x in l_masses]
                acc_l = 0.0
                t_cum: list[float] = []
                for _, jm in l_masses:
                    acc_l += jm / t_doc_mass
                    t_cum.append(acc_l)
                t_cum[-1] = 1.0
                tier_langs.append(t_lids)
                tier_lang_cum.append(t_cum)
                tier_pools.append(pools_k)

        if len(tier_masses) > 1:
            tot_m = sum(tier_masses)
            acc_t = 0.0
            cum_t: list[float] = []
            for tm in tier_masses:
                acc_t += tm / tot_m
                cum_t.append(acc_t)
            cum_t[-1] = 1.0
            self._tier_cum_probs = cum_t
            self._tier_langs = tier_langs
            self._tier_lang_cum = tier_lang_cum
            self._tier_pools = tier_pools
            self._use_tiered_buckets = True

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_bin_mmap"] = None
        state["_mask_mmap"] = None
        state["_mmap_pid"] = -1
        state.pop("_mmap_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._mmap_lock = threading.Lock()

    def _get_mmap(self) -> np.memmap:
        pid = os.getpid()
        if self._bin_mmap is None or self._mmap_pid != pid:
            with self._mmap_lock:
                if self._bin_mmap is None or self._mmap_pid != pid:
                    self._bin_mmap = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
                    self._mask_mmap = (
                        np.memmap(self.mask_path, dtype=np.uint8, mode="r")
                        if self.mask_path is not None
                        else None
                    )
                    self._mmap_pid = pid
        return self._bin_mmap

    def close(self) -> None:
        with self._mmap_lock:
            self._bin_mmap = None
            self._mask_mmap = None
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
        mask_mm = self._mask_mmap
        total_t = sum(seg_len for _, _, seg_len in specs)
        n_docs = len(specs)

        x_buf = np.empty(total_t, dtype=np.uint8)
        byte_langs = np.empty(total_t, dtype=np.int64)
        loss_mask_buf = np.ones(total_t, dtype=np.bool_)
        cu_seqlens = np.empty(n_docs + 1, dtype=np.int32)
        doc_langs = np.empty(n_docs, dtype=np.int64)
        doc_styles = np.empty(n_docs, dtype=np.int64)
        doc_pts = np.ones(n_docs, dtype=np.float32)
        is_doc_start = np.zeros(total_t, dtype=np.bool_)

        cu_seqlens[0] = 0
        cursor = 0
        for i, (doc_idx, w_start, seg_len) in enumerate(specs):
            off = int(self._offsets[doc_idx]) + w_start
            lid = int(self._lang_ids[doc_idx])
            sid = int(self._style_ids[doc_idx])
            src = int(self._source_ids[doc_idx])
            x_buf[cursor : cursor + seg_len] = mm[off : off + seg_len]
            if mask_mm is not None:
                loss_mask_buf[cursor : cursor + seg_len] = mask_mm[off : off + seg_len] > 0
            byte_langs[cursor : cursor + seg_len] = lid
            is_doc_start[cursor] = True
            doc_langs[i] = lid
            doc_styles[i] = sid
            doc_pts[i] = 0.0 if src == LATENT_REASONING_SOURCE_ID else 1.0
            cursor += seg_len
            cu_seqlens[i + 1] = cursor

        if total_t > 0 and not np.any(loss_mask_buf):
            loss_mask_buf[-1] = True

        return {
            "x_bytes": torch.from_numpy(x_buf),
            "cu_seqlens": torch.from_numpy(cu_seqlens),
            "is_doc_start": torch.from_numpy(is_doc_start),
            "doc_lang_ids": torch.from_numpy(doc_langs),
            "doc_style_ids": torch.from_numpy(doc_styles),
            "doc_ponder_tax_scale": torch.from_numpy(doc_pts),
            "byte_lang_ids": torch.from_numpy(byte_langs),
            "loss_mask": torch.from_numpy(loss_mask_buf),
            "wug_target_mask": torch.zeros(total_t, dtype=torch.bool),
        }

    def get_batch(self, batch_idx: int, sync_group: Optional[int] = None) -> dict[str, torch.Tensor]:
        """Pack one Constant-Byte-Budget zero-padding Ragged batch (`Total_T <= target_batch_bytes`).

        When `self._use_tiered_buckets` is active, `sync_group` (defaulting to
        `batch_idx // bucket_sync_world_size`) deterministically selects the same byte-length tier
        across all DDP ranks on the same step while `batch_idx` selects distinct documents per rank!
        """
        winfo = get_worker_info()
        worker_id = winfo.id if winfo is not None else 0
        rng = random.Random(self.seed + int(batch_idx) * 1000003 + worker_id * 97)
        mm = self._get_mmap()
        mask_mm = self._mask_mmap

        budget = self.target_batch_bytes
        max_doc = self.max_doc_bytes
        wug_p = self.wug_ratio

        x_buf = np.empty(budget, dtype=np.uint8)
        byte_langs = np.empty(budget, dtype=np.int64)
        loss_mask = np.ones(budget, dtype=np.bool_)
        wug_mask = np.zeros(budget, dtype=np.bool_)

        cu_list: list[int] = [0]
        doc_lang_list: list[int] = []
        doc_style_list: list[int] = []
        doc_pts_list: list[float] = []

        cursor = 0
        tier_pool: Optional[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
        if self._use_tiered_buckets:
            eff_sync = (int(batch_idx) // self.bucket_sync_world_size) if sync_group is None else int(sync_group)
            u_tier = random.Random(self.seed + eff_sync * 7919 + 31).random()
            t_idx = len(self._tier_cum_probs) - 1
            for k_i, cp in enumerate(self._tier_cum_probs):
                if u_tier <= cp:
                    t_idx = k_i
                    break
            langs = self._tier_langs[t_idx]
            cum_weights = self._tier_lang_cum[t_idx]
            tier_pool = self._tier_pools[t_idx]
        else:
            langs = self._langs
            cum_weights = self._cum_weights

        # Pre-bind hot attributes to locals for maximum Python loop speed
        lang_offsets = self._lang_offsets
        lang_lengths = self._lang_lengths
        lang_sources = self._lang_sources
        lang_styles = self._lang_styles
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

            # Optional Stage-D online Wug / Cognitive Stage-4 Episode injection
            if wug_p > 0.0 and rand_fn() < wug_p:
                cog_fn = getattr(wug_generator, "sample_cognitive_stage4_episode_fast", None)
                if self.wug_mode == "cognitive_v2" and cog_fn is not None and rem >= 320:
                    raw_b, t1_len, tgt_s, tgt_e, wug_style = cog_fn(rng, lid, max_bytes=min(rem, max_doc))
                else:
                    raw_b, t1_len, tgt_s, tgt_e = sample_wug_episode_fast(rng, lid)
                    wug_style = 0
                ep_len = len(raw_b)
                if ep_len <= rem:
                    x_buf[cursor : cursor + ep_len] = np.frombuffer(raw_b, dtype=np.uint8)
                    byte_langs[cursor : cursor + ep_len] = lid
                    loss_mask[cursor : cursor + t1_len] = False
                    wug_mask[cursor + tgt_s : cursor + tgt_e] = True
                    doc_lang_list.append(lid)
                    doc_style_list.append(int(wug_style))
                    doc_pts_list.append(1.0)
                    cursor += ep_len
                    cu_list.append(cursor)
                    continue
                if doc_lang_list and rem < 256:
                    break
                # Fall through to a regular doc slice if Wug episode exceeded tiny tail budget

            if tier_pool is not None:
                offs, lens, srcs, stys = tier_pool[lid]
            else:
                offs = lang_offsets[lid]
                lens = lang_lengths[lid]
                srcs = lang_sources[lid]
                stys = lang_styles[lid]
            d_i = randrange_fn(len(offs))
            off = int(offs[d_i])
            doc_len = int(lens[d_i])
            doc_view = mm[off : off + doc_len]

            if doc_len <= max_doc:
                w_start = 0
                w_end = doc_len
            elif mask_mm is not None or self.anchor_doc_start:
                # Preserve leading user prompt / tool header or Stage-1 sentence/function start
                w_start = 0
                w_end = utf8_safe_end(doc_view, 0, max_doc)
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
                if (rem < 64 or mask_mm is not None) and doc_lang_list:
                    break
                w_end = utf8_safe_end(doc_view, w_start, rem)
                seg_len = w_end - w_start
                if seg_len <= 0:
                    break

            x_buf[cursor : cursor + seg_len] = doc_view[w_start:w_end]
            if mask_mm is not None:
                loss_mask[cursor : cursor + seg_len] = mask_mm[off + w_start : off + w_end] > 0
            byte_langs[cursor : cursor + seg_len] = lid
            doc_lang_list.append(lid)
            doc_style_list.append(int(stys[d_i]))
            doc_pts_list.append(0.0 if int(srcs[d_i]) == LATENT_REASONING_SOURCE_ID else 1.0)
            cursor += seg_len
            cu_list.append(cursor)

        total_t = cursor
        if total_t > 0 and not np.any(loss_mask[:total_t]):
            loss_mask[total_t - 1] = True
        cu_arr = np.asarray(cu_list, dtype=np.int32)
        doc_lang_arr = np.asarray(doc_lang_list, dtype=np.int64)
        doc_style_arr = np.asarray(doc_style_list, dtype=np.int64)
        doc_pts_arr = np.asarray(doc_pts_list, dtype=np.float32)
        is_doc_start = np.zeros(total_t, dtype=np.bool_)
        if len(cu_arr) > 1:
            is_doc_start[cu_arr[:-1]] = True

        return {
            "x_bytes": torch.from_numpy(x_buf[:total_t].copy()),
            "cu_seqlens": torch.from_numpy(cu_arr),
            "is_doc_start": torch.from_numpy(is_doc_start),
            "doc_lang_ids": torch.from_numpy(doc_lang_arr),
            "doc_style_ids": torch.from_numpy(doc_style_arr),
            "doc_ponder_tax_scale": torch.from_numpy(doc_pts_arr),
            "byte_lang_ids": torch.from_numpy(byte_langs[:total_t].copy()),
            "loss_mask": torch.from_numpy(loss_mask[:total_t].copy()),
            "wug_target_mask": torch.from_numpy(wug_mask[:total_t].copy()),
        }

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.get_batch(batch_idx, sync_group=None)


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
    length_bucketing: bool = True,
    bucket_sync_world_size: int = 1,
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
        length_bucketing=length_bucketing,
        bucket_sync_world_size=bucket_sync_world_size,
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


# =============================================================================
# DCPA-Flat v3.3 Static-Grid FFD Minimum-Load Lane Packer (WS-1)
# =============================================================================

# Fast uint8 -> float32 lookup table identical to `dcpa.interfaces.decode_fixed_point_loss_mask`:
#   0 -> 0.0, 1 (legacy) -> 1.0, 10 -> 1.0, 15 -> 1.5, 30 -> 3.0
_LOSS_MASK_DECODE_LUT: np.ndarray = np.arange(256, dtype=np.float32) * np.float32(0.1)
_LOSS_MASK_DECODE_LUT[1] = np.float32(1.0)


@dataclass(slots=True)
class _GridDocItem:
    """Candidate document (either mmap-backed or losslessly pre-chunked in memory)."""

    offset: int
    length: int
    lang_id: int
    style_id: int
    source_id: int
    raw_bytes: Optional[bytes] = None
    mask_bytes: Optional[bytes] = None


def static_grid_batch_to_dict(batch: StaticGridBatch) -> dict[str, torch.Tensor]:
    """Convert a `StaticGridBatch` dataclass to a plain tensor dict."""
    return {
        "x_bytes": batch.x_bytes,
        "doc_id_byte": batch.doc_id_byte,
        "doc_start_byte": batch.doc_start_byte,
        "doc_end_byte": batch.doc_end_byte,
        "loss_weight": batch.loss_weight,
        "doc_meta": batch.doc_meta,
    }


def static_grid_batch_from_dict(d: dict[str, torch.Tensor]) -> StaticGridBatch:
    """Reconstruct a `StaticGridBatch` dataclass from a plain tensor dict."""
    return StaticGridBatch(
        x_bytes=d["x_bytes"],
        doc_id_byte=d["doc_id_byte"],
        doc_start_byte=d["doc_start_byte"],
        doc_end_byte=d["doc_end_byte"],
        loss_weight=d["loss_weight"],
        doc_meta=d["doc_meta"],
    )


class StaticGridByteDataset(IterableDataset):
    """DCPA-Flat v3.3 static `[P, T]` grid dataset with FFD minimum-load greedy lane packing.

    Guarantees:
      - ZERO mid-document truncation: documents `<= T` are packed whole; any unchunked shard
        document exceeding `T` is split on-the-fly with the lossless paragraph/code pre-chunker
        and queued into `carry_deque`.
      - ZERO dropped documents: candidates that do not fit into any lane overflow cleanly into
        cross-batch `carry_deque` and are prioritized on the very next batch.
      - Fixed-point x10 `.loss_mask.bin` decoding (`0/1/10/15/30 -> 0.0/1.0/1.0/1.5/3.0`).
      - Lane-tail padding strictly uses `x_bytes=0x00`, `doc_id_byte=-1`,
        `doc_start_byte=False`, `doc_end_byte=False`, `loss_weight=0.0`.
      - Emits `dcpa.interfaces.StaticGridBatch` (or dict when `return_dict=True`).
    """

    def __init__(
        self,
        bin_path: str | Path,
        idx_path: str | Path,
        *,
        grid_lanes: int = 32,
        grid_bytes: int = 4096,
        doc_meta_cap_per_lane: int = 64,
        domain_weights: Optional[dict[int, float]] = None,
        max_stage_tier: int = 0,
        num_batches: Optional[int] = 10000,
        seed: int = 42,
        allow_drive_fuse: bool = False,
        pool_fill_ratio: float = 0.95,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        if grid_lanes < 1:
            raise ValueError(f"grid_lanes (P) must be >= 1, got {grid_lanes}")
        if grid_bytes < 64:
            raise ValueError(f"grid_bytes (T) must be >= 64, got {grid_bytes}")
        if doc_meta_cap_per_lane < 1:
            raise ValueError(f"doc_meta_cap_per_lane must be >= 1, got {doc_meta_cap_per_lane}")
        if not (0.5 <= pool_fill_ratio <= 2.0):
            raise ValueError(f"pool_fill_ratio must be in [0.5, 2.0], got {pool_fill_ratio}")

        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        cand_mask = self.bin_path.with_suffix(".loss_mask.bin")
        self.mask_path: Optional[Path] = (
            cand_mask
            if (
                cand_mask.is_file()
                and self.bin_path.is_file()
                and cand_mask.stat().st_size == self.bin_path.stat().st_size
            )
            else None
        )
        if not allow_drive_fuse:
            assert_not_drive_mmap(self.bin_path)
            assert_not_drive_mmap(self.idx_path)
            if self.mask_path is not None:
                assert_not_drive_mmap(self.mask_path)

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Missing binary shard: {self.bin_path}")
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Missing index shard: {self.idx_path}")

        self.grid_lanes = int(grid_lanes)
        self.grid_bytes = int(grid_bytes)
        self.doc_meta_cap_per_lane = int(doc_meta_cap_per_lane)
        self.max_stage_tier = max(0, int(max_stage_tier))
        self.num_batches = int(num_batches) if (num_batches is not None and num_batches > 0) else None
        self.seed = int(seed)
        self.pool_fill_ratio = float(pool_fill_ratio)
        self.return_dict = bool(return_dict)
        self._explicit_domain_weights = domain_weights is not None

        idx_mm = np.memmap(self.idx_path, dtype=IDX_DTYPE, mode="r")
        offsets = np.array(idx_mm["offset"], dtype=np.int64, copy=True)
        lengths = np.array(idx_mm["length"], dtype=np.int64, copy=True)
        lang_ids = np.array(idx_mm["lang_id"], dtype=np.int64, copy=True)
        source_ids = np.array(idx_mm["source_id"], dtype=np.int64, copy=True)
        style_ids = np.array(idx_mm["style_id"], dtype=np.int64, copy=True)
        del idx_mm

        if len(offsets) == 0:
            raise ValueError(f"Index {self.idx_path} contains zero documents.")

        doc_tiers = infer_doc_stage_tiers(source_ids, lengths) if self.max_stage_tier > 0 else None
        valid_mask = lengths > 0
        if doc_tiers is not None:
            tier_ok = valid_mask & (doc_tiers <= self.max_stage_tier)
            if np.any(tier_ok):
                valid_mask = tier_ok

        self._offsets = offsets[valid_mask]
        self._lengths = lengths[valid_mask]
        self._lang_ids = lang_ids[valid_mask]
        self._source_ids = source_ids[valid_mask]
        self._style_ids = style_ids[valid_mask]
        self.num_docs = int(len(self._offsets))

        self._by_lang: dict[int, np.ndarray] = {}
        for lid in sorted(set(self._lang_ids.tolist())):
            self._by_lang[int(lid)] = np.nonzero(self._lang_ids == lid)[0]

        self._langs, self._cum_weights = _normalize_domain_weights(
            list(self._by_lang.keys()), domain_weights
        )

    def _draw_next_doc_idx(
        self,
        rng: random.Random,
        np_rng: np.random.Generator,
        all_perm: list[np.ndarray],
        all_pos: list[int],
        lang_perms: dict[int, np.ndarray],
        lang_pos: dict[int, int],
    ) -> int:
        if not self._explicit_domain_weights:
            if all_pos[0] >= len(all_perm[0]):
                all_perm[0] = np_rng.permutation(self.num_docs)
                all_pos[0] = 0
            idx = int(all_perm[0][all_pos[0]])
            all_pos[0] += 1
            return idx

        if len(self._langs) == 1:
            lid = self._langs[0]
        else:
            u = rng.random()
            lid = self._langs[-1]
            for cand_lid, cw in zip(self._langs, self._cum_weights):
                if u <= cw:
                    lid = cand_lid
                    break

        pool = self._by_lang[lid]
        pos = lang_pos[lid]
        perm = lang_perms[lid]
        if pos >= len(perm):
            perm = pool[np_rng.permutation(len(pool))]
            lang_perms[lid] = perm
            pos = 0
        idx = int(perm[pos])
        lang_pos[lid] = pos + 1
        return idx

    def _pull_doc_item(
        self,
        mm: np.memmap,
        mask_mm: Optional[np.memmap],
        carry_deque: deque[_GridDocItem],
        rng: random.Random,
        np_rng: np.random.Generator,
        all_perm: list[np.ndarray],
        all_pos: list[int],
        lang_perms: dict[int, np.ndarray],
        lang_pos: dict[int, int],
        *,
        allow_carry: bool = True,
    ) -> Optional[_GridDocItem]:
        if allow_carry and carry_deque:
            return carry_deque.popleft()

        T = self.grid_bytes
        for _ in range(64):
            doc_idx = self._draw_next_doc_idx(rng, np_rng, all_perm, all_pos, lang_perms, lang_pos)
            off = int(self._offsets[doc_idx])
            doc_len = int(self._lengths[doc_idx])
            lid = int(self._lang_ids[doc_idx])
            sid = int(self._style_ids[doc_idx])
            src = int(self._source_ids[doc_idx])

            if doc_len <= T:
                return _GridDocItem(
                    offset=off,
                    length=doc_len,
                    lang_id=lid,
                    style_id=sid,
                    source_id=src,
                )

            # Oversized unchunked doc (> T): split losslessly on-the-fly so ZERO mid-doc truncation occurs
            raw_b = bytes(mm[off : off + doc_len])
            try:
                text = raw_b.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                continue
            pieces = prechunk_document(text, lang_id=lid, max_chunk_bytes=T)
            if not pieces:
                continue
            first_item: Optional[_GridDocItem] = None
            for pc in pieces:
                if len(pc.raw_bytes) == 0 or len(pc.raw_bytes) > T:
                    continue
                if mask_mm is not None:
                    eff_mask: Optional[bytes] = pc.loss_mask_bytes
                elif pc.header_byte_len > 0:
                    eff_mask = (b"\x00" * pc.header_byte_len) + (b"\x01" * (len(pc.raw_bytes) - pc.header_byte_len))
                else:
                    eff_mask = None
                item = _GridDocItem(
                    offset=0,
                    length=len(pc.raw_bytes),
                    lang_id=lid,
                    style_id=sid,
                    source_id=src,
                    raw_bytes=pc.raw_bytes,
                    mask_bytes=eff_mask,
                )
                if first_item is None:
                    first_item = item
                else:
                    carry_deque.append(item)
            if first_item is not None:
                return first_item
        return None

    def _materialize_grid(
        self,
        mm: np.memmap,
        mask_mm: Optional[np.memmap],
        lane_docs: list[list[_GridDocItem]],
    ) -> StaticGridBatch | dict[str, torch.Tensor]:
        P = self.grid_lanes
        T = self.grid_bytes
        meta_cap = P * self.doc_meta_cap_per_lane

        x_grid = np.zeros((P, T), dtype=np.uint8)
        doc_id_grid = np.full((P, T), -1, dtype=np.int32)
        doc_start_grid = np.zeros((P, T), dtype=np.bool_)
        doc_end_grid = np.zeros((P, T), dtype=np.bool_)
        loss_w_grid = np.zeros((P, T), dtype=np.float32)
        doc_meta = np.zeros((meta_cap, 3), dtype=np.int32)

        doc_id = 0
        for p in range(P):
            pos = 0
            for item in lane_docs[p]:
                ln = item.length
                end = pos + ln
                if item.raw_bytes is not None:
                    x_grid[p, pos:end] = np.frombuffer(item.raw_bytes, dtype=np.uint8)
                    if item.mask_bytes is not None:
                        loss_w_grid[p, pos:end] = _LOSS_MASK_DECODE_LUT[
                            np.frombuffer(item.mask_bytes, dtype=np.uint8)
                        ]
                    else:
                        loss_w_grid[p, pos:end] = 1.0
                else:
                    off = item.offset
                    x_grid[p, pos:end] = mm[off : off + ln]
                    if mask_mm is not None:
                        loss_w_grid[p, pos:end] = _LOSS_MASK_DECODE_LUT[mask_mm[off : off + ln]]
                    else:
                        loss_w_grid[p, pos:end] = 1.0

                doc_id_grid[p, pos:end] = doc_id
                doc_start_grid[p, pos] = True
                doc_end_grid[p, end - 1] = True
                doc_meta[doc_id, 0] = item.lang_id
                doc_meta[doc_id, 1] = item.style_id
                doc_meta[doc_id, 2] = item.source_id
                doc_id += 1
                pos = end

        batch = StaticGridBatch(
            x_bytes=torch.from_numpy(x_grid),
            doc_id_byte=torch.from_numpy(doc_id_grid),
            doc_start_byte=torch.from_numpy(doc_start_grid),
            doc_end_byte=torch.from_numpy(doc_end_grid),
            loss_weight=torch.from_numpy(loss_w_grid),
            doc_meta=torch.from_numpy(doc_meta),
        )
        return static_grid_batch_to_dict(batch) if self.return_dict else batch

    def __iter__(self) -> Iterator[StaticGridBatch | dict[str, torch.Tensor]]:
        winfo = get_worker_info()
        worker_id = winfo.id if winfo is not None else 0
        num_workers = winfo.num_workers if winfo is not None else 1

        if self.num_batches is None:
            worker_batches: Optional[int] = None
        else:
            base_q, rem_q = divmod(self.num_batches, num_workers)
            worker_batches = base_q + (1 if worker_id < rem_q else 0)
            if worker_batches <= 0:
                return

        w_seed = self.seed + worker_id * 1000003
        rng = random.Random(w_seed)
        np_rng = np.random.default_rng(w_seed)

        all_perm = [np_rng.permutation(self.num_docs)]
        all_pos = [0]
        lang_perms = {
            lid: pool[np_rng.permutation(len(pool))] for lid, pool in self._by_lang.items()
        }
        lang_pos = {lid: 0 for lid in self._by_lang}

        P = self.grid_lanes
        T = self.grid_bytes
        grid_cap = P * T
        max_docs = P * self.doc_meta_cap_per_lane
        pool_target = max(T, int(round(self.pool_fill_ratio * grid_cap)))
        topup_target = int(round(0.985 * grid_cap))
        max_topup_misses = max(8, P // 2)

        mm = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
        mask_mm = np.memmap(self.mask_path, dtype=np.uint8, mode="r") if self.mask_path is not None else None
        carry_deque: deque[_GridDocItem] = deque()

        emitted = 0
        try:
            while worker_batches is None or emitted < worker_batches:
                # 1. Fill candidate pool (carry_deque first, then fresh corpus docs) to ~pool_fill_ratio capacity
                candidates: list[_GridDocItem] = []
                pool_bytes = 0
                while carry_deque and pool_bytes < pool_target and len(candidates) < max_docs:
                    item = carry_deque.popleft()
                    candidates.append(item)
                    pool_bytes += item.length

                while pool_bytes < pool_target and len(candidates) < max_docs:
                    item = self._pull_doc_item(
                        mm, mask_mm, carry_deque, rng, np_rng, all_perm, all_pos, lang_perms, lang_pos,
                        allow_carry=False,
                    )
                    if item is None:
                        break
                    candidates.append(item)
                    pool_bytes += item.length

                if not candidates:
                    break

                # 2. First-Fit-Decreasing (FFD) minimum-load greedy lane packing
                candidates.sort(key=lambda c: c.length, reverse=True)
                lanes_heap: list[tuple[int, int]] = [(0, p) for p in range(P)]
                lane_docs: list[list[_GridDocItem]] = [[] for _ in range(P)]
                overflow: list[_GridDocItem] = []
                total_placed = 0
                packed_bytes = 0

                for item in candidates:
                    min_used, best_lane = lanes_heap[0]
                    if total_placed < max_docs and (T - min_used) >= item.length:
                        heapq.heapreplace(lanes_heap, (min_used + item.length, best_lane))
                        lane_docs[best_lane].append(item)
                        total_placed += 1
                        packed_bytes += item.length
                    else:
                        overflow.append(item)

                # 3. Bounded gap-fill pass (places any fitting docs into remaining tails; non-fitting -> carry_deque)
                misses = 0
                while (
                    not carry_deque
                    and packed_bytes < topup_target
                    and total_placed < max_docs
                    and (T - lanes_heap[0][0]) >= 32
                    and misses < max_topup_misses
                ):
                    item = self._pull_doc_item(
                        mm, mask_mm, carry_deque, rng, np_rng, all_perm, all_pos, lang_perms, lang_pos,
                        allow_carry=False,
                    )
                    if item is None:
                        break
                    min_used, best_lane = lanes_heap[0]
                    if (T - min_used) >= item.length:
                        heapq.heapreplace(lanes_heap, (min_used + item.length, best_lane))
                        lane_docs[best_lane].append(item)
                        total_placed += 1
                        packed_bytes += item.length
                    else:
                        overflow.append(item)
                        misses += 1

                # Preserve ALL unplaced documents across batch boundaries (ZERO dropped docs)
                if overflow:
                    carry_deque.extend(overflow)

                yield self._materialize_grid(mm, mask_mm, lane_docs)
                emitted += 1
        finally:
            del mm
            if mask_mm is not None:
                del mask_mm


def build_static_grid_dataloader(
    bin_path: str | Path,
    idx_path: str | Path,
    *,
    grid_lanes: int = 32,
    grid_bytes: int = 4096,
    doc_meta_cap_per_lane: int = 64,
    domain_weights: Optional[dict[int, float]] = None,
    max_stage_tier: int = 0,
    num_batches: Optional[int] = 10000,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    allow_drive_fuse: bool = False,
    pool_fill_ratio: float = 0.95,
    return_dict: bool = False,
) -> DataLoader:
    """Create a DataLoader yielding static `[P, T]` `StaticGridBatch` grids packed with FFD."""
    ds = StaticGridByteDataset(
        bin_path=bin_path,
        idx_path=idx_path,
        grid_lanes=grid_lanes,
        grid_bytes=grid_bytes,
        doc_meta_cap_per_lane=doc_meta_cap_per_lane,
        domain_weights=domain_weights,
        max_stage_tier=max_stage_tier,
        num_batches=num_batches,
        seed=seed,
        allow_drive_fuse=allow_drive_fuse,
        pool_fill_ratio=pool_fill_ratio,
        return_dict=return_dict,
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
