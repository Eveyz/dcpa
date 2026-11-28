"""Unified V=256 raw ASCII control-byte bus for DCPA dialogues, scratchpads, and tool calls.

Implements Capability Upgrade Spec 1 & Red Lines #1-#2 (ZERO vocabulary expansion beyond 0..255):
  - `TURN_USER_PREFIX  = b"\\x1eU\\n"`  (0x1E RS: user turn start)
  - `TURN_ASST_PREFIX  = b"\\x1fA\\n"`  (0x1F US: assistant output / post-observation resume)
  - `TURN_EOT_SUFFIX   = b"\\x04\\n"`   (0x04 EOT: end of assistant turn)
  - `SCRATCH_START     = b"\\x1dS\\n"`  (0x1D GS + S: symbolic scratchpad / writing outline start)
  - `SCRATCH_END       = b"\\x1dE\\n"`  (0x1D GS + E: scratchpad / outline close -> Slot-0 Thesis Pin!)
  - `TOOL_CALL_PREFIX  = b"\\x11C\\n"`  (0x11 DC1 + C: start of tool call)
  - `TOOL_SUSPEND_END  = b"\\x13\\n"`   (0x13 DC3/XOFF: suspend decoding & run local AST sandbox)
  - `TOOL_OBS_PREFIX   = b"\\x12O\\n"`  (0x12 DC2 + O: sandbox execution observation injection)

Red Line #1 (Control-Byte Airspace Hygiene):
  `sanitize_raw_text` strips ALL ASCII < 32 control characters except `\\n` (0x0A) and `\\t` (0x09)
  from every untrusted prose/code input before framing.

PMD 128B Pushdown Stack Reset Contract:
  `compute_stack_reset_mask(raw_bytes, doc_reset_mask)` computes in O(1) vectorized ops:
    reset_t = is_doc_start[t] | (x_{t-1} in {0x04, 0x13, 0x1E}) | (x_{t-3:t} == b"\\x1dE\\n")
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


# Native ASCII control-byte framing constants (strictly within uint8 0..255)
TURN_USER_BYTE: int = 0x1E
TURN_ASST_BYTE: int = 0x1F
TURN_EOT_BYTE: int = 0x04
SCRATCH_CTRL_BYTE: int = 0x1D
TOOL_CALL_BYTE: int = 0x11
TOOL_OBS_BYTE: int = 0x12
TOOL_SUSPEND_BYTE: int = 0x13
TOOL_PAGING_BYTE: int = 0x14

ALLOWED_PROTOCOL_CTRL_BYTES: frozenset[int] = frozenset(
    {
        TURN_EOT_BYTE,
        TOOL_CALL_BYTE,
        TOOL_OBS_BYTE,
        TOOL_SUSPEND_BYTE,
        TOOL_PAGING_BYTE,
        SCRATCH_CTRL_BYTE,
        TURN_USER_BYTE,
        TURN_ASST_BYTE,
    }
)

TURN_USER_PREFIX: bytes = b"\x1eU\n"
TURN_ASST_PREFIX: bytes = b"\x1fA\n"
TURN_EOT_SUFFIX: bytes = b"\x04\n"
SCRATCH_START: bytes = b"\x1dS\n"
SCRATCH_END: bytes = b"\x1dE\n"
TOOL_CALL_PREFIX: bytes = b"\x11C\n"
TOOL_SUSPEND_END: bytes = b"\x13\n"
TOOL_OBS_PREFIX: bytes = b"\x12O\n"

# Exact byte sequence injected immediately after `0x13` (identical in training AND cli/chat_m3.py!)
def format_tool_observation_after_suspend(result_text: str) -> bytes:
    """Exact post-0x13 observation + resume bytes (Red Line #2: bit-identical train <-> REPL).

    In training streams:
      ... \\x11C\\n{call}\\x13\\n\\x12O\\n{result}\\n\\x1fA\\n ...
    When the REPL halts right upon emitting byte `0x13`, injecting `b"\\n\\x12O\\n" + clean + b"\\n\\x1fA\\n"`
    produces the EXACT same byte sequence bit-for-bit!
    """
    clean = sanitize_raw_text(str(result_text)).strip()
    return b"\n" + TOOL_OBS_PREFIX + clean.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX


# Canonical DCPA-Flat v3.3 6-domain and 5-style IDs (matching dcpa.interfaces SSOT)
LANG_DOMAINS: dict[str, int] = {
    "en": 0,
    "zh": 1,
    "formal": 2,
    "python": 3,
    "sql_struct": 4,
    "javascript": 5,
}
LANG_ID_TO_NAME: dict[int, str] = {v: k for k, v in LANG_DOMAINS.items()}

# Lookup table with backwards-compatible aliases
LANG_NAME_TO_ID: dict[str, int] = {
    **LANG_DOMAINS,
    "ja": 2,
    "fol": 2,
    "math": 2,
    "lean": 2,
    "py": 3,
    "sql": 4,
    "json": 4,
    "struct": 4,
    "rs": 4,
    "rust": 4,
    "js": 5,
    "ts": 5,
    "typescript": 5,
}

STYLE_DOMAINS: dict[str, int] = {
    "rigorous": 0,
    "concise_memo": 1,
    "expository": 2,
    "literary": 3,
    "strict_eng": 4,
}
STYLE_ID_TO_NAME: dict[int, str] = {v: k for k, v in STYLE_DOMAINS.items()}

# Lookup table with backwards-compatible aliases
STYLE_NAME_TO_ID: dict[str, int] = {
    **STYLE_DOMAINS,
    "neutral": 0,
    "memo": 1,
    "narrative": 3,
    "strict_code_tool": 4,
    "code": 4,
    "tool": 4,
}

# Source-ID tag marking 40% latent implicit reasoning samples (`ponder_tax_scale = 0.0`)
SOURCE_ID_REASONING_EXPLICIT: int = 10
SOURCE_ID_REASONING_LATENT: int = 11
SOURCE_ID_WRITING: int = 12
SOURCE_ID_TOOLCALL_CHAIN: int = 13
SOURCE_ID_TOOLCALL_RETRY: int = 14
SOURCE_ID_TOOLCALL_NEGATIVE: int = 15

# Precomputed translation table stripping all ASCII < 32 except \t (0x09) and \n (0x0A), plus 0x7F (DEL)
_CTRL_STRIP_TABLE: dict[int, None] = {
    **{c: None for c in range(32) if c not in (0x09, 0x0A)},
    0x7F: None,
}


def sanitize_raw_text(text: str) -> str:
    """Stage-1 pre-framing sanitization: NFC normalize, CRLF/CR -> LF, strip ASCII < 32 (except \\t,\\n) and 0x7F."""
    nfc = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return nfc.translate(_CTRL_STRIP_TABLE)


def sanitize_raw_bytes(raw: bytes) -> bytes:
    """Strip disallowed ASCII control bytes from raw UTF-8 bytes after NFC + LF normalization."""
    return sanitize_raw_text(raw.decode("utf-8", errors="ignore")).encode("utf-8", errors="strict")


def validate_framed_record(raw: bytes, max_scratch_inner_bytes: int = 260) -> tuple[bool, str]:
    """Stage-2 post-framing validator (parity with `dcpa-data-forge::validate_framed_record`).

    Allows the 8 protected protocol control bytes (`ALLOWED_PROTOCOL_CTRL_BYTES`) plus `\\t`/`\\n`,
    rejects any other ASCII `< 0x20` or `0x7F`, enforces strict UTF-8, and verifies that every
    `\\x1dS\\n` ... `\\x1dE\\n` scratchpad frame is properly paired/closed with inner span `<= 260` bytes.
    """
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, "invalid utf-8"

    n = len(raw)
    scratch_inner_start: Optional[int] = None
    i = 0
    while i < n:
        b = raw[i]
        if (b < 0x20 and b not in (0x09, 0x0A) and b not in ALLOWED_PROTOCOL_CTRL_BYTES) or b == 0x7F:
            return False, f"disallowed control byte 0x{b:02x} at offset {i}"
        if b == SCRATCH_CTRL_BYTE:
            if raw[i : i + 3] == SCRATCH_START:
                if scratch_inner_start is not None:
                    return False, "nested or unclosed scratchpad frame"
                scratch_inner_start = i + 3
                i += 3
                continue
            if raw[i : i + 3] == SCRATCH_END:
                if scratch_inner_start is None:
                    return False, "unmatched scratchpad end"
                inner_len = i - scratch_inner_start
                if inner_len > max_scratch_inner_bytes:
                    return False, f"scratchpad inner span {inner_len} exceeds {max_scratch_inner_bytes} bytes"
                scratch_inner_start = None
                i += 3
                continue
            return False, "malformed 0x1D scratchpad delimiter"
        if scratch_inner_start is not None and b in (TURN_EOT_BYTE, TURN_USER_BYTE):
            return False, "unclosed scratchpad frame before turn boundary"
        i += 1

    if scratch_inner_start is not None:
        return False, "unclosed scratchpad frame at end of record"
    return True, "ok"


def compute_stack_reset_mask(
    raw_bytes: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized O(1)-depth PMD 128B pushdown stack reset mask (Spec 1).

    reset_t = is_doc_start[t]
            | (x_{t-1} in {0x04, 0x13, 0x1E})
            | (x_{t-3:t} == b"\\x1dE\\n")

    Args:
      raw_bytes: [B, T] uint8/long byte tensor.
      doc_reset_mask: optional [B, T] bool marking document start positions.
    Returns:
      [B, T] bool tensor.
    """
    B, T = raw_bytes.shape
    device = raw_bytes.device
    if T == 0:
        return torch.zeros((B, 0), device=device, dtype=torch.bool)

    rb = raw_bytes.to(dtype=torch.long)
    z1 = torch.zeros((B, 1), device=device, dtype=torch.long)
    prev1 = torch.cat([z1, rb[:, :-1]], dim=1) if T > 1 else z1
    if doc_reset_mask is not None:
        prev1 = torch.where(doc_reset_mask, torch.zeros_like(prev1), prev1)

    ctrl_single = (prev1 == TURN_EOT_BYTE) | (prev1 == TOOL_SUSPEND_BYTE) | (prev1 == TURN_USER_BYTE)

    if T >= 3:
        z2 = torch.zeros((B, 2), device=device, dtype=torch.long)
        z3 = torch.zeros((B, 3), device=device, dtype=torch.long)
        prev2 = torch.cat([z2, rb[:, :-2]], dim=1)
        prev3 = torch.cat([z3, rb[:, :-3]], dim=1)
        scratch_end_hit = (prev3 == SCRATCH_CTRL_BYTE) & (prev2 == ord("E")) & (prev1 == 0x0A)
        if doc_reset_mask is not None:
            # Guard against 3-byte match crossing a document boundary
            drm = doc_reset_mask.bool()
            drm_p1 = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), drm[:, :-1]], dim=1)
            drm_p2 = torch.cat([torch.zeros((B, 2), device=device, dtype=torch.bool), drm[:, :-2]], dim=1)
            scratch_end_hit = scratch_end_hit & ~(drm | drm_p1 | drm_p2)
    else:
        scratch_end_hit = torch.zeros((B, T), device=device, dtype=torch.bool)

    frame_reset = ctrl_single | scratch_end_hit
    if doc_reset_mask is not None:
        return doc_reset_mask.bool() | frame_reset
    return frame_reset


def compute_pin_trigger_byte_mask(
    raw_bytes: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized [B, T] bool mask marking the closing byte of `\\x1dE\\n` or first `\\x1fA\\n` (Spec 2)."""
    B, T = raw_bytes.shape
    device = raw_bytes.device
    if T < 3:
        return torch.zeros((B, T), device=device, dtype=torch.bool)

    rb = raw_bytes.to(dtype=torch.long)
    b_m2 = rb[:, :-2]
    b_m1 = rb[:, 1:-1]
    b_0 = rb[:, 2:]

    is_scratch_end = (b_m2 == SCRATCH_CTRL_BYTE) & (b_m1 == ord("E")) & (b_0 == 0x0A)
    is_asst_start = (b_m2 == TURN_ASST_BYTE) & (b_m1 == ord("A")) & (b_0 == 0x0A)
    if T >= 6:
        next_is_scratch = (rb[:, 3:-2] == SCRATCH_CTRL_BYTE) & (rb[:, 4:-1] == ord("S")) & (rb[:, 5:] == 0x0A)
        pad_tail = torch.zeros((B, 3), device=device, dtype=torch.bool)
        is_asst_start = is_asst_start & ~torch.cat([next_is_scratch, pad_tail], dim=1)
    hit_tail = is_scratch_end | is_asst_start

    out = torch.zeros((B, T), device=device, dtype=torch.bool)
    out[:, 2:] = hit_tail
    if doc_reset_mask is not None:
        drm = doc_reset_mask.bool()
        drm_p1 = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), drm[:, :-1]], dim=1)
        out = out & ~(drm | drm_p1)
    return out


def compute_unpin_trigger_byte_mask(
    raw_bytes: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized [B, T] bool mask marking turn-boundary bytes `0x04` (EOT) or `0x1E` (user RS)."""
    rb = raw_bytes.to(dtype=torch.long)
    out = (rb == TURN_EOT_BYTE) | (rb == TURN_USER_BYTE)
    if doc_reset_mask is not None:
        out = out & ~doc_reset_mask.bool()
    return out


@dataclass(frozen=True)
class EncodedDialogue:
    raw_bytes: bytes
    loss_mask_bytes: bytes   # 1-to-1 byte-aligned uint8 (0=ignore, 1=supervise)
    lang_id: int
    bucket_id: int           # source_id / bucket identifier
    num_turns: int
    eot_positions: tuple[int, ...]
    style_id: int = 0


def format_user_turn_prompt(user_text: str) -> bytes:
    """Encode a live user turn prompt ready for incremental O(1) ingestion in REPL."""
    clean = sanitize_raw_text(user_text).strip()
    return TURN_USER_PREFIX + clean.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX


def encode_dialogue_turns(
    turns: Sequence[tuple[str, str]],
    lang_id: int,
    bucket_id: int = 0,
    style_id: int = 0,
    allow_assistant_controls: bool = False,
) -> EncodedDialogue:
    """Encode 1..N `(user_text, assistant_text)` turns into raw UTF-8 bytes & exact `sft_loss_mask`.

    User turns are ALWAYS sanitized (Red Line #1). Assistant turns are sanitized unless
    `allow_assistant_controls=True` (when callers explicitly embed `SCRATCH_START`/`SCRATCH_END`).
    """
    if not turns:
        raise ValueError("Dialogue must contain at least one (user, assistant) turn.")

    raw_chunks: list[bytes] = []
    mask_chunks: list[bytes] = []
    eot_positions: list[int] = []
    cursor = 0

    for user_text, asst_text in turns:
        u_clean = sanitize_raw_text(user_text).strip()
        a_clean = (
            unicodedata.normalize("NFC", asst_text).strip()
            if allow_assistant_controls
            else sanitize_raw_text(asst_text).strip()
        )
        if not u_clean or not a_clean:
            raise ValueError("Empty user or assistant turn is not allowed.")

        u_frame = TURN_USER_PREFIX + u_clean.encode("utf-8", errors="strict") + b"\n" + TURN_ASST_PREFIX
        a_body = a_clean.encode("utf-8", errors="strict")
        a_frame = a_body + TURN_EOT_SUFFIX

        raw_chunks.append(u_frame)
        mask_chunks.append(b"\x00" * len(u_frame))
        cursor += len(u_frame)

        eot_pos = cursor + len(a_body)
        eot_positions.append(eot_pos)

        raw_chunks.append(a_frame)
        mask_chunks.append(b"\x01" * len(a_frame))
        cursor += len(a_frame)

    raw_bytes = b"".join(raw_chunks)
    loss_mask_bytes = b"".join(mask_chunks)
    assert len(raw_bytes) == len(loss_mask_bytes)
    raw_bytes.decode("utf-8", errors="strict")
    for pos in eot_positions:
        assert raw_bytes[pos] == TURN_EOT_BYTE
        assert loss_mask_bytes[pos] == 1

    return EncodedDialogue(
        raw_bytes=raw_bytes,
        loss_mask_bytes=loss_mask_bytes,
        lang_id=int(lang_id),
        bucket_id=int(bucket_id),
        num_turns=len(turns),
        eot_positions=tuple(eot_positions),
        style_id=int(style_id),
    )


def build_sft_byte_loss_weights(
    x_bytes: torch.Tensor,
    sft_loss_mask: torch.Tensor,
    eot_weight: float = 2.0,
    tool_ctrl_weight: float = 2.5,
) -> torch.Tensor:
    """Build per-byte SFT CE weights:
      - `0.0` where `sft_loss_mask == False` (tool headers, user turns, `\\x12O\\n` observations)
      - `1.0` on standard supervised assistant bytes
      - `eot_weight` (`2.0x`) on `0x04` EOT bytes
      - `tool_ctrl_weight` (`2.5x`) on `0x11` (DC1 tool call start) & `0x13` (DC3 suspend) bytes
    """
    mask_bool = sft_loss_mask.bool()
    weights = mask_bool.to(dtype=torch.float32)
    xb = x_bytes.to(dtype=torch.int64)
    if eot_weight != 1.0:
        is_eot = (xb == TURN_EOT_BYTE) & mask_bool
        weights = torch.where(is_eot, torch.full_like(weights, float(eot_weight)), weights)
    if tool_ctrl_weight != 1.0:
        is_tool_ctrl = ((xb == TOOL_CALL_BYTE) | (xb == TOOL_SUSPEND_BYTE)) & mask_bool
        weights = torch.where(is_tool_ctrl, torch.full_like(weights, float(tool_ctrl_weight)), weights)
    return weights


def detect_lang_id(text: str) -> int:
    """Lightweight heuristic language/domain detector matching v2 domains (`4=Code/SQL`, `2=Formal/JA`, etc.)."""
    s = text.strip()
    low = s.lower()

    js_hits = sum(
        kw in s
        for kw in ("function ", "const ", "=>", "console.log", "export ", "async function", "=== ", "!== ")
    )
    rust_hits = sum(kw in s for kw in ("fn ", "let mut ", "impl ", "struct ", "-> ", "usize", "Vec<", "pub fn", "match "))
    sql_hits = sum(
        kw in low
        for kw in (
            "select ",
            " from ",
            " where ",
            "insert into ",
            "update ",
            "create table ",
            "join ",
            "group by ",
            "order by ",
            "inner join ",
            "left join ",
        )
    )
    py_hits = sum(kw in s for kw in ("def ", "class ", "import ", "return ", "elif ", "lambda ", "-> int", "-> list"))
    fol_hits = sum(
        kw in low
        for kw in (
            "forall ",
            "exists ",
            "|-",
            "|=",
            "<->",
            "==>",
            "\\forall",
            "\\exists",
            "\\implies",
            "\\vdash",
            "theorem ",
            "lemma ",
        )
    )
    if (
        "```javascript" in low
        or "```js" in low
        or js_hits >= 2
        or (("javascript" in low or "nodejs" in low) and ("function" in s or "const " in s or "=>" in s or "函数" in s))
    ):
        return 5
    if (
        rust_hits >= 2
        or sql_hits >= 2
        or "```rust" in low
        or "```sql" in low
        or ("rust" in low and ("struct " in s or "fn " in s))
    ):
        return 4
    if py_hits >= 2 or "```python" in low or ("python" in low and ("def " in s or "函数" in s or "function" in low)):
        return 3
    if (
        any(sym in s for sym in ("∀", "∃", "⊢", "⊨", "⟹", "⟺"))
        or "```lean" in low
        or "```coq" in low
        or "```fol" in low
        or fol_hits >= 2
    ):
        return 2

    ja_kana = sum(1 for ch in s if ("\u3040" <= ch <= "\u309f") or ("\u30a0" <= ch <= "\u30ff"))
    if ja_kana >= 2:
        return 2

    cjk = sum(1 for ch in s if ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf"))
    if cjk >= 2:
        return 1

    return 0


def verify_mask_alignment(raw_bytes: bytes, loss_mask: np.ndarray | bytes) -> bool:
    """Assert exact byte-level alignment of `sft_loss_mask` against `0x1E / 0x1F / 0x04 / 0x12` frames."""
    mask_arr = (
        np.frombuffer(loss_mask, dtype=np.uint8)
        if isinstance(loss_mask, (bytes, bytearray))
        else np.asarray(loss_mask, dtype=np.uint8)
    )
    if len(raw_bytes) != len(mask_arr):
        return False

    expected = np.zeros(len(raw_bytes), dtype=np.uint8)
    n = len(raw_bytes)
    # Optional top tool-signature block before first TURN_USER_PREFIX is strictly masked 0
    idx = raw_bytes.find(TURN_USER_PREFIX)
    if idx < 0:
        return False

    while idx < n:
        if not raw_bytes.startswith(TURN_USER_PREFIX, idx):
            return False
        asst_at = raw_bytes.find(TURN_ASST_PREFIX, idx + len(TURN_USER_PREFIX))
        if asst_at < 0:
            return False
        pos = asst_at + len(TURN_ASST_PREFIX)
        # Inside assistant turn, supervise everything EXCEPT `\\x12O\\n...\\n\\x1fA\\n` observation spans!
        while pos < n:
            eot_at = raw_bytes.find(bytes([TURN_EOT_BYTE]), pos)
            obs_at = raw_bytes.find(TOOL_OBS_PREFIX, pos)
            if obs_at >= 0 and (eot_at < 0 or obs_at < eot_at):
                expected[pos:obs_at] = 1
                resume_at = raw_bytes.find(TURN_ASST_PREFIX, obs_at + len(TOOL_OBS_PREFIX))
                if resume_at < 0:
                    return False
                # Observation span [obs_at : resume_at + len(TURN_ASST_PREFIX)] stays strictly 0!
                pos = resume_at + len(TURN_ASST_PREFIX)
            else:
                if eot_at < 0:
                    return False
                turn_end = eot_at + 2 if (eot_at + 1 < n and raw_bytes[eot_at + 1] == 0x0A) else eot_at + 1
                expected[pos:turn_end] = 1
                idx = turn_end
                break
        else:
            return False
    return bool(np.array_equal((mask_arr > 0).astype(np.uint8), expected))
