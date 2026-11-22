"""Zero-copy memory-mapped binary shard reader matching DCPA 16-byte `.idx` layout.

Binary index record layout (exact 16 bytes, little-endian `<QIBBH`):
  offset    : uint64 (8 bytes, start offset in .bin)
  length    : uint32 (4 bytes, exact UTF-8 byte length > 0)
  lang_id   : uint8  (1 byte: 0=en, 1=zh, 2=ja, 3=python, 4=rust)
  source_id : uint8  (1 byte: sub-source identifier)
  max_depth : uint16 (2 bytes: max bracket nesting depth for code; 0 for natural language)
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

IDX_RECORD_FMT = "<QIBBBB"
IDX_RECORD_SIZE = struct.calcsize(IDX_RECORD_FMT)  # 16 bytes


class BinaryShardDataset(Dataset):
    def __init__(
        self,
        bin_path: str | Path,
        idx_path: str | Path,
        seq_len: int = 2048,
        stage_bit: Optional[int] = None,
        allow_drive_fuse: bool = False,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        self.seq_len = seq_len

        if not allow_drive_fuse:
            for p in (self.bin_path, self.idx_path):
                raw_s = str(p)
                resolved_s = str(p.resolve()) if p.exists() else raw_s
                if raw_s.startswith("/content/drive") or resolved_s.startswith("/content/drive"):
                    raise ValueError(
                        f"Refusing to mmap binary shard directly on Google Drive FUSE mount ({p}). "
                        "Copy shards to local NVMe (e.g. /content/nvme_cache/data) first."
                    )

        raw_idx = self.idx_path.read_bytes()
        if len(raw_idx) % IDX_RECORD_SIZE != 0:
            raise ValueError(f"Corrupt .idx file: size {len(raw_idx)} not divisible by 16")
        num_records = len(raw_idx) // IDX_RECORD_SIZE

        records = [
            struct.unpack_from(IDX_RECORD_FMT, raw_idx, i * IDX_RECORD_SIZE)
            for i in range(num_records)
        ]
        if stage_bit is not None and stage_bit == 3:
            # Stage D uses natural language contexts + Wug episodes
            records = [r for r in records if r[2] in (0, 1, 2)]
        self.records = records

        self._bin_file = self.bin_path.open("rb")
        self._mmap = mmap.mmap(self._bin_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_off, byte_len, lang_id, _, _, _ = self.records[idx]
        take = min(byte_len, self.seq_len)
        # Step back over UTF-8 continuation bytes so sliced chunk stays valid UTF-8
        if take < byte_len:
            steps = 0
            while take > 0 and steps < 4 and (self._mmap[start_off + take] & 0xC0) == 0x80:
                take -= 1
                steps += 1
        buf = bytearray(self._mmap[start_off : start_off + take])
        if len(buf) < self.seq_len:
            # Pad with LF (0x0A) newline separator to fixed batch length T without corrupting UTF-8
            buf.extend(b"\n" * (self.seq_len - len(buf)))
        arr = np.frombuffer(buf, dtype=np.uint8).copy()
        return torch.from_numpy(arr), torch.tensor(lang_id, dtype=torch.long)

    def close(self) -> None:
        self._mmap.close()
        self._bin_file.close()
