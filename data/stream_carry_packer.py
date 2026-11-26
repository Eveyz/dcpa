"""Stream-carry batch packer (A1): turns document batches into `S` parallel byte streams of `chunk_bytes`.

Each optimizer step every stream row emits its next UTF-8-safe chunk (`<= chunk_bytes`). When a row's
document ends mid-chunk the row immediately continues with a fresh document in the same step (so the
byte budget stays ~constant). Every emitted piece is one packed entry (`cu_seqlens` document):

  - `carry_from[v] = s`  -> entry `v` continues stream `s` (its initial state comes from the carry bank)
  - `carry_from[v] = -1` -> fresh document start (learned initial state)
  - `carry_save[v] = s`  -> entry `v`'s end-of-chunk state is stored for stream `s` (document continues)
  - `carry_save[v] = -1` -> document finished inside this chunk (state discarded)

Documents are drawn from the regular `RaggedByteDataset` batches (`pull_fn`), so curriculum domain
weights, length tiers, capability mixes and loss masks are preserved exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np
import torch

from data.ragged_dataloader import utf8_safe_end

__all__ = ["StreamCarryPacker"]


@dataclass
class _Doc:
    x: np.ndarray
    byte_langs: np.ndarray
    loss_mask: np.ndarray
    wug_mask: np.ndarray
    lang: int
    style: int
    pts: float


@dataclass
class _Row:
    doc: Optional[_Doc] = None
    off: int = 0


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").numpy()


class StreamCarryPacker:
    def __init__(
        self,
        n_streams: int,
        chunk_bytes: int,
        min_piece_bytes: int = 64,
        max_entries: Optional[int] = None,
        max_pulls_per_step: int = 64,
    ) -> None:
        """`max_entries` caps packed entries per batch (rows + same-step refills) so the BCG batch width
        never exceeds a static bucket (e.g. 448 rows + <=64 refills -> always <= 512 entries)."""
        if chunk_bytes < 16:
            raise ValueError(f"chunk_bytes must be >= 16, got {chunk_bytes}")
        self.n_streams = max(1, int(n_streams))
        self.chunk_bytes = int(chunk_bytes)
        self.min_piece_bytes = max(1, min(int(min_piece_bytes), self.chunk_bytes))
        self.max_entries = max(self.n_streams, int(max_entries)) if max_entries is not None else None
        self.max_pulls_per_step = max(1, int(max_pulls_per_step))
        self._queue: Deque[_Doc] = deque()
        self._rows: list[_Row] = [_Row() for _ in range(self.n_streams)]
        self.docs_started = 0
        self.docs_finished = 0

    def reset(self) -> None:
        """Drop all in-flight documents (stage change, rollback, resume)."""
        self._queue.clear()
        self._rows = [_Row() for _ in range(self.n_streams)]

    @property
    def num_live_streams(self) -> int:
        return sum(1 for r in self._rows if r.doc is not None)

    def _enqueue_batch(self, batch: dict[str, torch.Tensor]) -> int:
        x = _np(batch["x_bytes"]).astype(np.uint8, copy=False)
        cu = _np(batch["cu_seqlens"]).astype(np.int64)
        langs = _np(batch["doc_lang_ids"]).astype(np.int64)
        n_docs = int(cu.shape[0]) - 1
        styles = (
            _np(batch["doc_style_ids"]).astype(np.int64)
            if batch.get("doc_style_ids") is not None
            else np.zeros((n_docs,), dtype=np.int64)
        )
        pts = (
            _np(batch["doc_ponder_tax_scale"]).astype(np.float32)
            if batch.get("doc_ponder_tax_scale") is not None
            else np.ones((n_docs,), dtype=np.float32)
        )
        byte_langs = (
            _np(batch["byte_lang_ids"]).astype(np.int64)
            if batch.get("byte_lang_ids") is not None
            else np.repeat(langs, np.diff(cu))
        )
        loss_mask = (
            _np(batch["loss_mask"]).astype(np.bool_)
            if batch.get("loss_mask") is not None
            else np.ones_like(x, dtype=np.bool_)
        )
        wug_mask = (
            _np(batch["wug_target_mask"]).astype(np.bool_)
            if batch.get("wug_target_mask") is not None
            else np.zeros_like(x, dtype=np.bool_)
        )
        added = 0
        for i in range(n_docs):
            s, e = int(cu[i]), int(cu[i + 1])
            if e <= s:
                continue
            self._queue.append(
                _Doc(
                    x=x[s:e].copy(),
                    byte_langs=byte_langs[s:e].copy(),
                    loss_mask=loss_mask[s:e].copy(),
                    wug_mask=wug_mask[s:e].copy(),
                    lang=int(langs[i]),
                    style=int(styles[i]),
                    pts=float(pts[i]),
                )
            )
            added += 1
        return added

    def _next_doc(self, pull_fn: Callable[[], dict[str, torch.Tensor]], pulls: list[int]) -> _Doc:
        while not self._queue:
            if pulls[0] >= self.max_pulls_per_step:
                raise RuntimeError("StreamCarryPacker: document source keeps returning empty batches")
            pulls[0] += 1
            self._enqueue_batch(pull_fn())
        self.docs_started += 1
        return self._queue.popleft()

    def next_batch(self, pull_fn: Callable[[], dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Emit one packed stream-carry batch (same keys as the regular packer + carry routing)."""
        L = self.chunk_bytes
        pulls = [0]
        xs: list[np.ndarray] = []
        bl: list[np.ndarray] = []
        lm: list[np.ndarray] = []
        wm: list[np.ndarray] = []
        cu: list[int] = [0]
        d_lang: list[int] = []
        d_style: list[int] = []
        d_pts: list[float] = []
        carry_from: list[int] = []
        carry_save: list[int] = []

        for r_idx, row in enumerate(self._rows):
            budget = L
            first = True
            while budget >= self.min_piece_bytes or first:
                if (
                    not first
                    and self.max_entries is not None
                    and (len(cu) - 1) + (self.n_streams - r_idx - 1) >= self.max_entries
                ):
                    break  # refill would push later rows past the static entry cap
                cont = False
                if row.doc is None:
                    row.doc = self._next_doc(pull_fn, pulls)
                    row.off = 0
                else:
                    cont = first and row.off > 0
                doc = row.doc
                end = utf8_safe_end(doc.x, row.off, budget)
                if end <= row.off:
                    if first and budget == L:
                        # Pathological: a single character longer than the chunk; force-advance 1 char.
                        end = min(len(doc.x), row.off + max(1, budget))
                    else:
                        break
                s, e = row.off, end
                xs.append(doc.x[s:e])
                bl.append(doc.byte_langs[s:e])
                lm.append(doc.loss_mask[s:e])
                wm.append(doc.wug_mask[s:e])
                cu.append(cu[-1] + (e - s))
                d_lang.append(doc.lang)
                d_style.append(doc.style)
                d_pts.append(doc.pts)
                carry_from.append(r_idx if cont else -1)
                row.off = e
                budget -= e - s
                first = False
                if row.off < len(doc.x):
                    carry_save.append(r_idx)
                    break
                carry_save.append(-1)
                row.doc = None
                row.off = 0
                self.docs_finished += 1

        x_all = np.concatenate(xs) if xs else np.zeros((0,), dtype=np.uint8)
        loss_all = np.concatenate(lm) if lm else np.zeros((0,), dtype=np.bool_)
        if x_all.size > 0 and not np.any(loss_all):
            loss_all[-1] = True
        cu_arr = np.asarray(cu, dtype=np.int32)
        is_doc_start = np.zeros((x_all.size,), dtype=np.bool_)
        if cu_arr.size > 1:
            is_doc_start[cu_arr[:-1]] = True
        return {
            "x_bytes": torch.from_numpy(x_all.astype(np.uint8, copy=False)),
            "cu_seqlens": torch.from_numpy(cu_arr),
            "is_doc_start": torch.from_numpy(is_doc_start),
            "doc_lang_ids": torch.from_numpy(np.asarray(d_lang, dtype=np.int64)),
            "doc_style_ids": torch.from_numpy(np.asarray(d_style, dtype=np.int64)),
            "doc_ponder_tax_scale": torch.from_numpy(np.asarray(d_pts, dtype=np.float32)),
            "byte_lang_ids": torch.from_numpy(np.concatenate(bl) if bl else np.zeros((0,), dtype=np.int64)),
            "loss_mask": torch.from_numpy(loss_all),
            "wug_target_mask": torch.from_numpy(np.concatenate(wm) if wm else np.zeros((0,), dtype=np.bool_)),
            "carry_from": torch.tensor(carry_from, dtype=torch.int64),
            "carry_save": torch.tensor(carry_save, dtype=torch.int64),
        }
