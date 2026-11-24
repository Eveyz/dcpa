"""Apple Silicon M3 Pro Zero-Dependency Stateful O(1) Interactive Streaming REPL & Safe AST Tool Sandbox.

Implements Stage 5 Addendum Section 4 + Capability Upgrade Spec 5.3 & Red Lines #2/#4:
  - Runs on macOS M3 Pro (`--device mps` / `cpu` / `auto`) via `utils/device.py` with zero external
    TUI dependencies (stdlib `readline` + minimal ANSI gray/white highlights).
  - True O(1) persistent multi-turn state: ONE fixed-size `DCPACarryState` (K=8 concept graph
    with Slot-0 Thesis Pinning + 7.52 MB column-major Hippocampus fast weights).
  - Startup Hippocampus Tool Pre-Registration: silently sweeps 5 built-in pure-stdlib safe tools
    (`calc`, `now_hk`, `list_files`, `read_file_span`, `word_char_count`) into fast weights and
    snapshots `M_tools_base` (`/reset` restores `M_tools_base` rather than zeroing; `/tool_add`
    dynamically registers new signatures and refreshes `M_tools_base`; `/wipe` zeros all).
  - `0x11` / `0x13` / `0x12` Hardware Interrupt Loop & Pure-AST Whitelist Sandbox (Red Line #4):
      * On `0x11`: automatically switches decoding to `temp=0.0` and `style_id=4` (`strict_code_tool`).
      * On `0x13`: immediately suspends decoding, validates call strictly via `ast.parse(..., mode='eval')`
        whitelist (STRICTLY NO `eval()` or `exec()`!), executes within a `2.0s` timeout, and injects
        exact `format_tool_observation_after_suspend(res)` (`b"\\n\\x12O\\n" + res + b"\\n\\x1fA\\n"`)
        into sensory input with `ach_override=0.0` (`pi_bu = pi_cort`).
      * Hard circuit breaker caps consecutive tool calls at `5` hops per turn.
  - Slash commands:
      `/teach <text>`, `/style neutral|memo|expository|narrative|strict_code_tool`,
      `/think show|dim|hide`, `/tools`, `/tool_add <sig>`,
      `/probe on|off|splits`, `/lang auto|zh|en|ja|py|rs`, `/sleep`, `/reset`, `/wipe`, `/quit`.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import concurrent.futures
import datetime
import json
import math
import operator
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

try:
    import readline  # noqa: F401
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.model import (
    DCPACarryState,
    DCPAModel,
    _compute_pin_trigger_byte_mask,
    _compute_unpin_trigger_byte_mask,
)
from dcpa.modules.bcg_pc import ConceptGraphState
from dcpa.modules.hippocampus import ColumnMajorFastWeights, EligibilityState
from dcpa.ops.ragged import ragged_local_offsets, ragged_segment_sum
from train.train_dcpa import export_final_safetensors
from train.train_stage5_sft import load_dcpa_model_from_weights
from train.visual_probe import format_sgsc_segmentation
from utils.chat_protocol import (
    LANG_ID_TO_NAME,
    LANG_NAME_TO_ID,
    SCRATCH_CTRL_BYTE,
    SCRATCH_END,
    SCRATCH_START,
    STYLE_ID_TO_NAME,
    STYLE_NAME_TO_ID,
    TOOL_CALL_BYTE,
    TOOL_CALL_PREFIX,
    TOOL_SUSPEND_BYTE,
    TURN_EOT_BYTE,
    detect_lang_id,
    format_tool_observation_after_suspend,
    format_user_turn_prompt,
    sanitize_raw_text,
)
from utils.device import get_device, synchronize


ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[90m"
ANSI_BOLD = "\033[1m"
ANSI_WHITE = "\033[97m"

MAX_TOOL_HOPS_PER_TURN: int = 5
TOOL_EXEC_TIMEOUT_SEC: float = 2.0


# =============================================================================
# Pure-AST Safe Mathematical Evaluator & Whitelist Tool Sandbox (Red Line #4)
# STRICTLY ZERO `eval()` OR `exec()` CALLS ANYWHERE IN THIS MODULE!
# =============================================================================

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MATH_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": lambda a, b: math.pow(a, b) if abs(b) <= 100 else (_ for _ in ()).throw(ValueError("exponent > 100")),
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
}
_MATH_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _eval_math_ast(node: ast.AST) -> Any:
    """Strictly pure-AST numeric/statistics expression evaluator (no attribute access, no builtins)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant in math expr: {type(node.value).__name__}")
    if isinstance(node, ast.Name):
        if node.id in _MATH_CONSTS:
            return _MATH_CONSTS[node.id]
        raise ValueError(f"Disallowed identifier in math expr: {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math_ast(node.operand))
    if isinstance(node, ast.BinOp):
        left = _eval_math_ast(node.left)
        right = _eval_math_ast(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > 100.0:
                raise ValueError("Exponent magnitude capped at 100")
            return left**right
        if type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](left, right)
        raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_math_ast(elt) for elt in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCS:
            raise ValueError("Disallowed function call inside math expression")
        if node.keywords:
            raise ValueError("Keyword arguments not supported in math helper call")
        fn = _MATH_FUNCS[node.func.id]
        args = [_eval_math_ast(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"Disallowed AST node in math expr: {type(node).__name__}")


def _eval_literal_ast(node: ast.AST) -> Any:
    """Whitelist AST argument evaluator: ONLY Constant, UnaryOp, BinOp, List, Tuple, Dict."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool, type(None))):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        val = _eval_literal_ast(node.operand)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError("UnaryOp requires numeric literal")
        return _UNARY_OPS[type(node.op)](val)
    if isinstance(node, ast.BinOp):
        left = _eval_literal_ast(node.left)
        right = _eval_literal_ast(node.right)
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise ValueError("BinOp requires numeric literals")
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > 100.0:
                raise ValueError("Exponent magnitude capped at 100")
            return left**right
        if type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](left, right)
        raise ValueError(f"Disallowed BinOp: {type(node.op).__name__}")
    if isinstance(node, ast.List):
        return [_eval_literal_ast(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_literal_ast(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                raise ValueError("Dict unpacking is disallowed")
            out[_eval_literal_ast(k)] = _eval_literal_ast(v)
        return out
    raise ValueError(f"SecurityPolicy: disallowed AST argument node `{type(node).__name__}`")


def build_builtin_tools(workspace_root: str | Path = ".") -> tuple[dict[str, Callable[..., Any]], list[str]]:
    """Construct the 5 pure-stdlib safe tools and their Hippocampus single-line signatures."""
    root = Path(workspace_root).resolve()

    def _safe_resolve(rel_path: str) -> Path:
        p = (root / str(rel_path)).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"Path escapes workspace root: {rel_path!r}")
        return p

    def tool_calc(expr: str) -> float:
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("expr must be a non-empty string")
        tree = ast.parse(expr.strip(), mode="eval")
        res = _eval_math_ast(tree.body)
        if isinstance(res, (int, float)):
            return round(float(res), 10) if isinstance(res, float) else float(res)
        raise ValueError(f"calc returned non-numeric type {type(res).__name__}")

    def tool_now_hk() -> str:
        hk_tz = datetime.timezone(datetime.timedelta(hours=8))
        return datetime.datetime.now(hk_tz).isoformat(timespec="seconds")

    def tool_list_files(path: str = ".") -> list[str]:
        target = _safe_resolve(path)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir() if not p.name.startswith("."))
        return entries[:64]

    def tool_read_file_span(path: str, start: int = 1, end: int = 40) -> str:
        s_i, e_i = int(start), int(end)
        if s_i < 1 or e_i < s_i:
            raise ValueError(f"start ({s_i}) must be >= 1 and <= end ({e_i})")
        if e_i - s_i > 200:
            raise ValueError("Maximum readable line span is 200 lines")
        target = _safe_resolve(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = lines[s_i - 1 : e_i]
        return "\n".join(f"L{s_i + i}: {ln}" for i, ln in enumerate(chunk))

    def tool_word_char_count(text: str) -> dict[str, int]:
        s = str(text)
        return {
            "words": len(s.split()),
            "chars": len(s),
            "bytes": len(s.encode("utf-8")),
        }

    registry: dict[str, Callable[..., Any]] = {
        "calc": tool_calc,
        "now_hk": tool_now_hk,
        "list_files": tool_list_files,
        "read_file_span": tool_read_file_span,
        "word_char_count": tool_word_char_count,
    }
    signatures: list[str] = [
        "tool calc(expr: str) -> float : evaluate exact math and statistics expression",
        "tool now_hk() -> str : return current Hong Kong UTC+8 ISO timestamp",
        "tool list_files(path: str) -> list : list workspace directory entries",
        "tool read_file_span(path: str, start: int, end: int) -> str : read line range from workspace file",
        "tool word_char_count(text: str) -> dict : count words chars and UTF-8 bytes",
    ]
    return registry, signatures


def execute_tool_call_safe(
    call_str: str,
    tool_registry: dict[str, Callable[..., Any]],
    *,
    timeout_s: float = TOOL_EXEC_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """Parse and execute a tool call strictly via `ast.parse(..., mode='eval')` whitelist.

    Red Line #4:
      - NEVER invokes `eval()` or `exec()`.
      - Rejects attribute access, nested calls, dunder names, unknown tools, and non-literal args.
      - Enforces `timeout_s` (default 2.0s) and returns `(ok, controlled_result_string)`.
    """
    raw = sanitize_raw_text(call_str).strip()
    if not raw:
        return False, "Error: ValueError: empty tool call"
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError as exc:
        return False, f"Error: SyntaxError: {exc.msg}"

    root = tree.body
    if not isinstance(root, ast.Call):
        return False, f"Error: SecurityPolicy: root AST node must be Call, got {type(root).__name__}"
    if not isinstance(root.func, ast.Name):
        return False, f"Error: SecurityPolicy: disallowed callable expression `{type(root.func).__name__}`"

    fn_name = root.func.id
    if fn_name.startswith("_") or fn_name not in tool_registry:
        return False, f"Error: SecurityPolicy: unregistered or forbidden tool `{fn_name}`"

    try:
        pos_args = [_eval_literal_ast(a) for a in root.args]
        kw_args: dict[str, Any] = {}
        for kw in root.keywords:
            if kw.arg is None or kw.arg.startswith("_"):
                return False, "Error: SecurityPolicy: **kwargs or private parameter name disallowed"
            kw_args[kw.arg] = _eval_literal_ast(kw.value)
    except ValueError as exc:
        return False, f"Error: {exc}"

    fn = tool_registry[fn_name]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn, *pos_args, **kw_args)
    try:
        result = fut.result(timeout=float(timeout_s))
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        return False, f"Error: TimeoutError: tool `{fn_name}` exceeded {timeout_s:.1f}s limit"
    except Exception as exc:
        pool.shutdown(wait=False)
        return False, f"Error: {type(exc).__name__}: {exc}"
    else:
        pool.shutdown(wait=False)

    if isinstance(result, (dict, list)):
        res_text = json.dumps(result, ensure_ascii=False)
    else:
        res_text = str(result)
    return True, sanitize_raw_text(res_text).strip()


# =============================================================================
# Stateful O(1) Interactive Streaming Session
# =============================================================================


@dataclass
class TurnTelemetry:
    lang_id: int
    style_id: int
    input_bytes: int
    output_bytes: int
    num_spikes: int
    compression_ratio: float
    max_ach: float
    hippo_writes: int
    ponder_skip_pct: float
    latency_ms: float
    stopped_on_eot: bool
    segmented_preview: str
    tool_calls_executed: int = 0
    tool_observations: list[bytes] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    circuit_breaker_tripped: bool = False

    def format_line(self) -> str:
        sty = STYLE_ID_TO_NAME.get(self.style_id, str(self.style_id))
        return (
            f"[Spikes: {self.num_spikes} (ρ={self.compression_ratio:.1f}x) | "
            f"Max ACh: {self.max_ach:.2f} | Hippo Writes: {self.hippo_writes} | "
            f"Tools: {self.tool_calls_executed} | Style: {sty} | "
            f"Ponder Skip: {self.ponder_skip_pct:.0f}% | Latency: {self.latency_ms:.0f}ms]"
        )


def _sample_next_byte(logits: torch.Tensor, temp: float, top_p: float) -> int:
    """Nucleus (`top_p`) + temperature byte sampler (greedy when `temp <= 0.0`)."""
    l_f = logits.float().view(-1)
    if temp <= 1e-5:
        return int(l_f.argmax().item())
    probs = F.softmax(l_f / float(temp), dim=-1)
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cum_p = torch.cumsum(sorted_p, dim=-1)
    keep = cum_p <= float(top_p)
    keep[0] = True
    filtered_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
    filtered_p = filtered_p / filtered_p.sum().clamp_min(1e-8)
    choice = int(torch.multinomial(filtered_p, num_samples=1).item())
    return int(sorted_idx[choice].item())


class DCPAChatSession:
    """Persistent O(1) stateful streaming multi-turn engine + Hippocampus Tool Bus + SWR `/sleep`."""

    def __init__(
        self,
        model: DCPAModel,
        device: torch.device,
        *,
        temp: float = 0.3,
        top_p: float = 0.9,
        max_new_bytes: int = 256,
        tau_s: float = 0.25,
        tau_m: float = 0.25,
        workspace_root: str | Path = ".",
        preload_builtin_tools: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.temp = float(temp)
        self.top_p = float(top_p)
        self.max_new_bytes = int(max_new_bytes)
        self.tau_s = float(tau_s)
        self.tau_m = float(tau_m)

        self.lang_mode: str = "auto"    # "auto" | en | zh | ja | py | rs
        self.style_id: int = 0          # 0=neutral, 1=memo, 2=expository, 3=narrative, 4=strict_code_tool
        self.think_mode: str = "dim"    # "show" | "dim" | "hide"
        self.probe_mode: str = "off"    # "off" | "on" | "splits"

        # Persistent O(1) carry state (SGSC h, K=8 graph, PMD m & 128B stack, Hippocampus fast weights)
        self.carry: DCPACarryState = self.model.init_carry_state(
            batch_size=1, device=self.device, dtype=torch.float32
        )
        # Fixed O(1) episodic prototype ring buffer (max 64 entries) feeding `/sleep` SWR replay
        self._swr_ring: deque[tuple[torch.Tensor, ...]] = deque(maxlen=64)
        self._slot_h_end: torch.Tensor = torch.zeros(
            (self.model.cfg.k_slots, self.model.cfg.d_s),
            device=self.device,
            dtype=torch.float32,
        )

        # Safe stdlib tool registry + Hippocampus startup pre-registration (`M_tools_base`)
        self.tool_registry, self.tool_signatures = build_builtin_tools(workspace_root=workspace_root)
        self.m_tools_base: ColumnMajorFastWeights = self.carry.fast_state.clone_snapshot()
        if preload_builtin_tools:
            self._register_tool_signatures_into_hippocampus(self.tool_signatures)

    def _register_tool_signatures_into_hippocampus(self, signatures: list[str]) -> None:
        """Silently sweep tool signatures into Hippocampus fast weights & update `M_tools_base`."""
        sig_block = ("\n".join(sanitize_raw_text(s).strip() for s in signatures) + "\n").encode("utf-8")
        self.ingest_bytes(sig_block, lang_id=0, style_id=4)
        self.m_tools_base = self.carry.fast_state.clone_snapshot()
        self._slot_h_end.zero_()
        # Clear transient 8-slot working memory after seeding tools so Turn 1 starts fresh
        fresh = self.model.init_carry_state(batch_size=1, device=self.device, dtype=torch.float32)
        self.carry = DCPACarryState(
            sgsc_h=fresh.sgsc_h,
            graph_state=fresh.graph_state,
            pmd_m=fresh.pmd_m,
            pmd_xi=fresh.pmd_xi,
            fast_state=self.carry.fast_state,
            elig_state=EligibilityState.zeros(
                1, self.model.cfg.k_slots, self.model.cfg.s_active, self.model.cfg.d_s, self.device, torch.float32
            ),
        )

    def add_tool_signature(self, sig_line: str, fn: Optional[Callable[..., Any]] = None) -> str:
        """Dynamically teach a new single-line tool signature into Hippocampus and update `M_tools_base`."""
        clean = sanitize_raw_text(sig_line).strip()
        if not clean.startswith("tool "):
            clean = "tool " + clean
        self.tool_signatures.append(clean)
        if fn is not None:
            # Parse function name from `tool name(...)`
            after = clean[5:].strip()
            fn_name = after.split("(", 1)[0].strip()
            if fn_name.isidentifier() and not fn_name.startswith("_"):
                self.tool_registry[fn_name] = fn
        self.ingest_bytes((clean + "\n").encode("utf-8"), lang_id=0, style_id=4)
        self.m_tools_base = self.carry.fast_state.clone_snapshot()
        return clean

    def reset_working_memory(self) -> None:
        """`/reset`: clear 8-slot WM and restore Hippocampus to `M_tools_base` (built-in tools stay active!)."""
        fresh = self.model.init_carry_state(batch_size=1, device=self.device, dtype=torch.float32)
        self.carry.fast_state.restore_from(self.m_tools_base)
        self._slot_h_end.zero_()
        self.carry = DCPACarryState(
            sgsc_h=fresh.sgsc_h,
            graph_state=fresh.graph_state,
            pmd_m=fresh.pmd_m,
            pmd_xi=fresh.pmd_xi,
            fast_state=self.carry.fast_state,
            elig_state=EligibilityState.zeros(
                1, self.model.cfg.k_slots, self.model.cfg.s_active, self.model.cfg.d_s, self.device, torch.float32
            ),
        )

    def wipe_all_memory(self) -> None:
        """`/wipe`: clear both 8-slot working memory AND Hippocampus fast weights back to factory zero."""
        self.carry = self.model.init_carry_state(batch_size=1, device=self.device, dtype=torch.float32)
        self._swr_ring.clear()
        self._slot_h_end.zero_()

    def resolve_lang_id(self, text: str) -> int:
        if self.lang_mode != "auto" and self.lang_mode in LANG_NAME_TO_ID:
            return LANG_NAME_TO_ID[self.lang_mode]
        return detect_lang_id(text)

    @torch.no_grad()
    def ingest_bytes(
        self,
        raw_bytes: bytes,
        lang_id: int,
        *,
        style_id: Optional[int] = None,
        ach_override: Optional[float] = None,
    ) -> dict:
        """Incrementally ingest ONLY new bytes into persistent `self.carry` (zero history replay!)."""
        if not raw_bytes:
            return {
                "num_spikes": 0,
                "compression_ratio": 0.0,
                "max_ach": 0.0,
                "hippo_writes": 0,
                "ponder_skip_pct": 100.0,
                "segmented_preview": "",
                "last_z_star": None,
                "last_g_ctx": None,
            }

        x_t = torch.tensor(list(raw_bytes), dtype=torch.uint8, device=self.device).unsqueeze(0)
        sgsc_out = self.model.sgsc(x_t, tau_s=self.tau_s, h_init=self.carry.sgsc_h)
        m_total = int(sgsc_out.f_stem.shape[0])
        cu = sgsc_out.cu_seqlens.detach().cpu().tolist()
        seg_preview, _ = format_sgsc_segmentation(raw_bytes, cu)

        # Compute per-segment Slot-0 Thesis Pinning & Turn-Boundary Unpinning triggers (`\x1dE\n` / `\x1fA\n`, `0x04` / `0x1E`)
        if m_total > 0 and x_t.shape[1] > 0:
            pin_bm = _compute_pin_trigger_byte_mask(x_t, None)
            unpin_bm = _compute_unpin_trigger_byte_mask(x_t)
            seg_id_flat, local_pos, _, _ = ragged_local_offsets(sgsc_out.cu_seqlens, x_t.shape[1])
            pos_pow = torch.pow(
                torch.tensor(2.0, device=self.device, dtype=torch.float32),
                local_pos.clamp_max(22).float(),
            ).unsqueeze(-1)
            pin_w = ragged_segment_sum(pin_bm.float().reshape(-1, 1) * pos_pow, seg_id_flat, m_total).squeeze(-1)
            unpin_w = ragged_segment_sum(unpin_bm.float().reshape(-1, 1) * pos_pow, seg_id_flat, m_total).squeeze(-1)
            seg_unpin_trig = (unpin_w > 0.5).float()
            seg_pin_trig = (pin_w > unpin_w).float()
        else:
            seg_unpin_trig = torch.zeros((m_total,), device=self.device, dtype=torch.float32)
            seg_pin_trig = torch.zeros((m_total,), device=self.device, dtype=torch.float32)

        g_state = self.carry.graph_state
        f_state = self.carry.fast_state
        e_state = self.carry.elig_state

        max_ach = 0.0
        hippo_writes = 0
        ponder_skips = 0
        last_z_star = None
        last_g_ctx = None

        for m in range(m_total):
            f_m = sgsc_out.f_stem[m : m + 1]
            c_m = sgsc_out.c_slow[m : m + 1]
            sn_m = sgsc_out.sem_norm[m : m + 1]
            mi_m = sgsc_out.mean_info[m : m + 1]
            trig_m = seg_pin_trig[m : m + 1]
            if float(seg_unpin_trig[m].item()) > 0.5 and g_state.slot0_pinned is not None:
                g_state = ConceptGraphState(
                    slots=g_state.slots,
                    edges=g_state.edges,
                    salience=g_state.salience,
                    slot0_pinned=torch.zeros_like(g_state.slot0_pinned),
                )

            hip_r = self.model.hippocampus.read_comprehension(
                f_stem=f_m,
                c_slow=c_m,
                sem_norm=sn_m,
                mean_info=mi_m,
                fast_state=f_state,
                enabled=True,
                ach_override=ach_override,
            )
            bcg_o = self.model.bcg_pc.step(
                c_fused=hip_r.c_fused,
                prec_bu=hip_r.prec_bu,
                ach=hip_r.ach_signal,
                state=g_state,
                tau_m=self.tau_m,
                use_gumbel=False,
                update_c0_ema=False,
                pin_trigger=trig_m,
            )
            g_state = bcg_o.state
            last_z_star = bcg_o.z_star
            last_g_ctx = bcg_o.g_ctx
            if float(bcg_o.ponder_steps[0].item()) < 0.5:
                ponder_skips += 1

            ach_val = float(hip_r.ach_signal[0].item())
            if ach_val > max_ach:
                max_ach = ach_val

            prev_eta = e_state.eta[0].detach().float().clone()
            wi = int(bcg_o.win_i[0].item())
            wj = int(bcg_o.win_j[0].item())
            k_slots = self.model.cfg.k_slots
            end_byte_idx = int(cu[m + 1]) - 1
            h_end_m = sgsc_out.h_seq[0, end_byte_idx, :].detach().float().clone()
            eta_i = float(prev_eta[wi].item()) if wi < k_slots else ach_val
            eta_j = float(prev_eta[wj].item()) if wj < k_slots else ach_val
            p_star = wi if eta_i >= eta_j else wj
            u_slot = min(wi, wj)
            v_slot = max(wi, wj)
            self._slot_h_end[u_slot] = h_end_m if p_star == k_slots else self._slot_h_end[p_star].clone()
            if v_slot < k_slots:
                self._slot_h_end[v_slot] = h_end_m

            e_state = self.model.hippocampus.inherit_eligibility_on_merge(
                elig=e_state,
                cand_idx=hip_r.psi_idx,
                cand_val=hip_r.psi_val,
                cand_f=f_m,
                cand_ach=hip_r.ach_signal,
                win_i=bcg_o.win_i,
                win_j=bcg_o.win_j,
            )
            e_state, _, beta_mat = self.model.hippocampus.three_factor_write(
                s_bar_post=bcg_o.s_bar_post,
                edges_post=g_state.edges,
                c0=self.model.bcg_pc.c0,
                elig=e_state,
                fast_state=f_state,
                write_threshold=0.0,
            )
            max_beta = float(beta_mat[0].max().item())
            if max_beta > 0.05:
                hippo_writes += 1
            if max_beta > 0.0:
                win_slot = int(beta_mat[0].argmax().item())
                self._swr_ring.append(
                    (
                        e_state.e_idx[0, win_slot].detach().clone(),
                        e_state.f_tilde[0, win_slot].detach().float().clone(),
                        bcg_o.s_bar_post[0, win_slot].detach().float().clone(),
                        self._slot_h_end[win_slot].clone(),
                    )
                )

        self.carry.sgsc_h = sgsc_out.h_seq[:, -1, :].detach().clone()
        self.carry.graph_state = g_state
        self.carry.elig_state = e_state

        return {
            "num_spikes": m_total,
            "compression_ratio": float(len(raw_bytes)) / max(float(m_total), 1.0),
            "max_ach": max_ach,
            "hippo_writes": hippo_writes,
            "ponder_skip_pct": 100.0 * float(ponder_skips) / max(float(m_total), 1.0),
            "segmented_preview": seg_preview,
            "last_z_star": last_z_star,
            "last_g_ctx": last_g_ctx,
        }

    def execute_tool_call_safe(self, call_str: str, timeout_s: float = TOOL_EXEC_TIMEOUT_SEC) -> tuple[bool, str]:
        return execute_tool_call_safe(call_str, self.tool_registry, timeout_s=timeout_s)

    @torch.no_grad()
    def chat_turn(
        self,
        user_text: str,
        *,
        on_chunk: Optional[Callable[[str], None]] = None,
        max_new_bytes: Optional[int] = None,
        force_eot_after: Optional[int] = None,
        byte_sampler: Optional[Callable[[torch.Tensor, int, float], int]] = None,
    ) -> tuple[str, TurnTelemetry]:
        """Execute one O(1) stateful turn with `0x11/0x13/0x12` AST tool interrupt loop & 5-hop fuse."""
        synchronize(self.device)
        t0 = time.perf_counter()

        lang_id = self.resolve_lang_id(user_text)
        active_style_id = int(self.style_id)
        prompt_bytes = format_user_turn_prompt(user_text)
        stats = self.ingest_bytes(prompt_bytes, lang_id=lang_id, style_id=active_style_id)

        def _get_l_reg(sid: int) -> torch.Tensor:
            return self.model.pmd.get_lang_vec(
                torch.tensor([lang_id], dtype=torch.long, device=self.device),
                torch.tensor([sid], dtype=torch.long, device=self.device),
            )

        l_reg = _get_l_reg(active_style_id)
        limit = self.max_new_bytes if max_new_bytes is None else int(max_new_bytes)

        # Closed-loop Mode-1 predictive motor drive aligned 1-to-1 with Stage 5 SFT (`use_broca_cq=False`):
        # derives `c_hat` from live `BCG-PC` graph readout `g_ctx`, updating `self.carry` after each segment!
        def _next_motor_drive():
            _, g_ctx = self.model.bcg_pc.message_pass(
                self.carry.graph_state.slots,
                self.carry.graph_state.edges,
                self.carry.graph_state.salience,
            )
            c_hat = self.model.bcg_pc.f_pred_2(F.silu(self.model.bcg_pc.f_pred_1(g_ctx)))
            f_out, _ = self.model.hippocampus.read_expression(
                z_out=c_hat,
                l_reg=l_reg,
                fast_state=self.carry.fast_state,
                enabled=True,
            )
            delta_syn = self.model.pmd.compute_streaming_delta_syn(
                g_ctx=g_ctx,
                z_star=c_hat,
                l_reg=l_reg,
            )
            charge0 = self.model.pmd.init_segment_charge(f_out, delta_syn, l_reg)
            return f_out, delta_syn, charge0

        f_out, delta_syn, charge_cur = _next_motor_drive()
        m_cur = self.carry.pmd_m
        xi_cur = self.carry.pmd_xi
        prev_b = torch.tensor([prompt_bytes[-1]], dtype=torch.long, device=self.device)

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        emitted_raw = bytearray()          # visible + control bytes since last sensory ingestion
        total_emitted_bytes = 0
        emitted_text_parts: list[str] = []
        recent_bytes = bytearray(prompt_bytes[-3:])
        utf8_rem = 0

        tau_step = 1
        stopped_on_eot = False
        in_code_fence = lang_id in (3, 4, 5)
        in_tool_call = False
        tool_call_buf = bytearray()
        in_scratch = False
        scratch_buf = bytearray()

        tool_hops = 0
        tool_observations: list[bytes] = []
        tool_events: list[dict[str, Any]] = []
        circuit_broken = False

        def _emit_visible(text_piece: str, *, is_think: bool = False) -> None:
            if not text_piece:
                return
            if is_think and self.think_mode == "hide":
                return
            emitted_text_parts.append(text_piece)
            if on_chunk is not None:
                if is_think and self.think_mode == "dim":
                    on_chunk(f"{ANSI_DIM}{text_piece}{ANSI_RESET}")
                else:
                    on_chunk(text_piece)

        for step_i in range(limit):
            # Spec 1: trigger extra stack reset immediately after `b"\x1dE\n"`
            extra_reset = (
                torch.tensor([True], device=self.device, dtype=torch.bool)
                if len(recent_bytes) >= 3 and bytes(recent_bytes[-3:]) == SCRATCH_END
                else None
            )
            logits, m_cur, xi_cur, charge_cur, p_end = self.model.pmd.step_single_byte(
                prev_byte=prev_b,
                f_out=f_out,
                delta_syn=delta_syn,
                l_reg=l_reg,
                tau_step=tau_step,
                m_prev=m_cur,
                xi_prev=xi_cur,
                charge_prev=charge_cur,
                extra_stack_reset=extra_reset,
            )

            eff_temp = 0.0 if (in_tool_call or in_code_fence or lang_id in (3, 4, 5)) else self.temp
            if byte_sampler is not None:
                next_byte_val = int(byte_sampler(logits[0], step_i, eff_temp))
            elif force_eot_after is not None and step_i >= force_eot_after:
                next_byte_val = TURN_EOT_BYTE
            else:
                next_byte_val = _sample_next_byte(logits[0], temp=eff_temp, top_p=self.top_p)

            if next_byte_val == TURN_EOT_BYTE:
                stopped_on_eot = True
                # Advance 1 reset tick on 0x04 so PMD's 128B stack Xi_t clears cleanly for next turn!
                _, m_cur, xi_cur, _, _ = self.model.pmd.step_single_byte(
                    prev_byte=torch.tensor([TURN_EOT_BYTE], dtype=torch.long, device=self.device),
                    f_out=f_out,
                    delta_syn=delta_syn,
                    l_reg=l_reg,
                    tau_step=1,
                    m_prev=m_cur,
                    xi_prev=xi_cur,
                    charge_prev=charge_cur,
                )
                break

            emitted_raw.append(next_byte_val)
            total_emitted_bytes += 1
            recent_bytes.append(next_byte_val)
            if len(recent_bytes) > 6:
                del recent_bytes[:-6]
            if next_byte_val < 0x80:
                utf8_rem = 0
            elif (next_byte_val & 0xE0) == 0xC0:
                utf8_rem = 1
            elif (next_byte_val & 0xF0) == 0xE0:
                utf8_rem = 2
            elif (next_byte_val & 0xF8) == 0xF0:
                utf8_rem = 3
            elif (next_byte_val & 0xC0) == 0x80:
                utf8_rem = max(utf8_rem - 1, 0)
            else:
                utf8_rem = 0

            # -----------------------------------------------------------------
            # Spec 5.3: `0x11` -> switch temp=0.0 & style_id=4 (strict_code_tool)
            # -----------------------------------------------------------------
            if next_byte_val == TOOL_CALL_BYTE:
                in_tool_call = True
                tool_call_buf.clear()
                active_style_id = 4
                l_reg = _get_l_reg(active_style_id)
                prev_b = torch.tensor([next_byte_val], dtype=torch.long, device=self.device)
                tau_step = 1
                continue

            # -----------------------------------------------------------------
            # Spec 5.3: `0x13` -> IMMEDIATE SUSPEND, AST sandbox exec, 0x12 inject!
            # -----------------------------------------------------------------
            if next_byte_val == TOOL_SUSPEND_BYTE:
                raw_call_bytes = bytes(tool_call_buf)
                # Strip optional leading `C\n` prefix from `\x11C\n{call}`
                if raw_call_bytes.startswith(b"C\n"):
                    raw_call_bytes = raw_call_bytes[2:]
                elif raw_call_bytes.startswith(b"C"):
                    raw_call_bytes = raw_call_bytes[1:]
                call_str = raw_call_bytes.decode("utf-8", errors="replace").strip()

                if tool_hops >= MAX_TOOL_HOPS_PER_TURN:
                    circuit_broken = True
                    ok = False
                    res_str = (
                        f"Error: CircuitBreaker: exceeded max {MAX_TOOL_HOPS_PER_TURN} consecutive "
                        "tool calls in a single turn"
                    )
                else:
                    tool_hops += 1
                    ok, res_str = self.execute_tool_call_safe(call_str, timeout_s=TOOL_EXEC_TIMEOUT_SEC)

                tool_events.append(
                    {"hop": tool_hops, "call": call_str, "ok": ok, "result": res_str, "circuit_broken": circuit_broken}
                )

                # Red Line #2: Exact bit-for-bit observation sequence matching training (`\n\x12O\n{res}\n\x1fA\n`)
                obs_bytes = format_tool_observation_after_suspend(res_str)
                tool_observations.append(obs_bytes)

                # Ingest emitted call bytes up to 0x13, then inject observation with ACh=0, pi_bu=pi_cort!
                if emitted_raw:
                    self.ingest_bytes(bytes(emitted_raw), lang_id=lang_id, style_id=4)
                    emitted_raw.clear()
                self.ingest_bytes(obs_bytes, lang_id=lang_id, style_id=self.style_id, ach_override=0.0)

                # Reset stack cleanly on 0x13 suspend boundary and restore user's selected style_id
                in_tool_call = False
                tool_call_buf.clear()
                utf8_rem = 0
                active_style_id = int(self.style_id)
                l_reg = _get_l_reg(active_style_id)
                xi_cur = torch.zeros_like(xi_cur)
                f_out, delta_syn, charge_cur = _next_motor_drive()
                prev_b = torch.tensor([obs_bytes[-1]], dtype=torch.long, device=self.device)
                recent_bytes = bytearray(obs_bytes[-3:])
                tau_step = 1
                if circuit_broken:
                    # Force controlled stop after injecting CircuitBreaker error observation
                    break
                continue

            if in_tool_call:
                tool_call_buf.append(next_byte_val)
            else:
                # Handle `\x1dS\n...\x1dE\n` scratchpad display modes (`/think show|dim|hide`)
                if next_byte_val == SCRATCH_CTRL_BYTE:
                    if not in_scratch:
                        in_scratch = True
                        scratch_buf.clear()
                    else:
                        # Closing `\x1dE\n`: flush scratch block per think_mode
                        s_txt = scratch_buf.decode("utf-8", errors="replace")
                        if s_txt.startswith("S\n"):
                            s_txt = s_txt[2:]
                        _emit_visible(s_txt, is_think=True)
                        scratch_buf.clear()
                        in_scratch = False
                elif in_scratch:
                    scratch_buf.append(next_byte_val)
                else:
                    # Skip trailing `E\n` immediately following closing `0x1d`
                    if len(recent_bytes) >= 2 and recent_bytes[-2] == SCRATCH_CTRL_BYTE and next_byte_val == ord("E"):
                        pass
                    elif len(recent_bytes) >= 3 and bytes(recent_bytes[-3:]) == SCRATCH_END:
                        pass
                    else:
                        piece = decoder.decode(bytes([next_byte_val]), final=False)
                        if piece:
                            _emit_visible(piece, is_think=False)
                            if "```" in "".join(emitted_text_parts[-4:]):
                                in_code_fence = not in_code_fence

            prev_b = torch.tensor([next_byte_val], dtype=torch.long, device=self.device)
            tau_step += 1
            hit_scratch_close = len(recent_bytes) >= 3 and bytes(recent_bytes[-3:]) == SCRATCH_END
            seg_done = float(p_end[0].item()) >= 0.5 or tau_step > self.model.cfg.max_seg_len
            if hit_scratch_close or seg_done:
                if (not in_tool_call) and emitted_raw and utf8_rem == 0:
                    self.ingest_bytes(bytes(emitted_raw), lang_id=lang_id, style_id=active_style_id)
                    emitted_raw.clear()
                if utf8_rem == 0 or tau_step > self.model.cfg.max_seg_len + 3:
                    f_out, delta_syn, charge_cur = _next_motor_drive()
                    tau_step = 1

        tail = decoder.decode(b"", final=True)
        if tail:
            _emit_visible(tail, is_think=False)

        self.carry.pmd_m = m_cur.detach().clone()
        self.carry.pmd_xi = xi_cur.detach().clone()

        # Efference-copy feedback: stream remaining assistant reply (+ EOT) into carry state in O(1)
        if emitted_raw or stopped_on_eot:
            self.ingest_bytes(
                bytes(emitted_raw) + bytes([TURN_EOT_BYTE, 0x0A]),
                lang_id=lang_id,
                style_id=self.style_id,
            )

        synchronize(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telem = TurnTelemetry(
            lang_id=lang_id,
            style_id=int(self.style_id),
            input_bytes=len(prompt_bytes),
            output_bytes=total_emitted_bytes,
            num_spikes=int(stats["num_spikes"]),
            compression_ratio=float(stats["compression_ratio"]),
            max_ach=float(stats["max_ach"]),
            hippo_writes=int(stats["hippo_writes"]),
            ponder_skip_pct=float(stats["ponder_skip_pct"]),
            latency_ms=latency_ms,
            stopped_on_eot=stopped_on_eot,
            segmented_preview=str(stats["segmented_preview"]),
            tool_calls_executed=tool_hops,
            tool_observations=tool_observations,
            tool_events=tool_events,
            circuit_breaker_tripped=circuit_broken,
        )
        return "".join(emitted_text_parts), telem

    def run_sleep_consolidation(
        self,
        *,
        steps: int = 50,
        lr: float = 1e-3,
        out_path: str | Path = "artifacts/export/dcpa_chat_consolidated.safetensors",
        familiarity_threshold: float = 0.5,
    ) -> dict:
        """`/sleep`: 50-step local Sharp-Wave Ripple (SWR) offline distillation on M3 Pro."""
        fast = self.carry.fast_state
        assert isinstance(fast, ColumnMajorFastWeights)

        high_fam_cols = torch.nonzero(fast.u_lex > float(familiarity_threshold), as_tuple=False).squeeze(-1)
        high_fam_count = int(high_fam_cols.numel())

        f_list: list[torch.Tensor] = []
        c_list: list[torch.Tensor] = []
        h_list: list[torch.Tensor] = []
        for entry in self._swr_ring:
            e_idx = entry[0]
            f_tilde = entry[1]
            s_bar = entry[2]
            h_end = (
                entry[3]
                if len(entry) >= 4
                else torch.zeros((self.model.cfg.d_s,), device=self.device, dtype=torch.float32)
            )
            col_u = float(fast.u_lex[e_idx].mean().item())
            if high_fam_count == 0 or col_u >= float(familiarity_threshold) * 0.5:
                f_list.append(f_tilde.to(self.device))
                c_list.append(s_bar.to(self.device))
                h_list.append(h_end.to(self.device))

        if not f_list:
            nz_cols = high_fam_cols if high_fam_count > 0 else torch.nonzero(fast.u_lex > 0.0, as_tuple=False).squeeze(-1)
            if nz_cols.numel() > 0:
                take = nz_cols[:32]
                c_cols = fast.m_c_from_l[take].float().to(self.device)
                f_cols = self.model.hippocampus.w_lex_slow(c_cols).detach()
                f_list.append(f_cols)
                c_list.append(c_cols)
                h_list.append(torch.zeros((take.shape[0], self.model.cfg.d_s), device=self.device, dtype=torch.float32))

        out_dst = Path(out_path)
        if not f_list:
            exported = export_final_safetensors(self.model, out_dst.parent, filename=out_dst.name)
            return {
                "steps": 0,
                "high_familiarity_cols": high_fam_count,
                "prototypes_replayed": 0,
                "loss_before": 0.0,
                "loss_after": 0.0,
                "export_path": str(exported["weights"]),
            }

        f_batch = torch.stack(f_list, dim=0) if f_list[0].ndim == 1 else torch.cat(f_list, dim=0)
        c_batch = torch.stack(c_list, dim=0) if c_list[0].ndim == 1 else torch.cat(c_list, dim=0)
        h_batch = torch.stack(h_list, dim=0) if h_list[0].ndim == 1 else torch.cat(h_list, dim=0)
        f_batch = f_batch.detach().float()
        c_batch = c_batch.detach().float()
        h_batch = h_batch.detach().float()

        prev_req_grad = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.model.sgsc.w_sem.weight.requires_grad_(True)
        self.model.hippocampus.w_lex_slow.weight.requires_grad_(True)

        swr_params = [self.model.sgsc.w_sem.weight, self.model.hippocampus.w_lex_slow.weight]
        opt = torch.optim.Adam(swr_params, lr=lr)

        loss_before = 0.0
        loss_after = 0.0
        try:
            self.model.train()
            for s in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                c_pred = self.model.sgsc.norm_c(self.model.sgsc.w_res(h_batch) + self.model.sgsc.w_sem(f_batch))
                f_pred = self.model.hippocampus.w_lex_slow(c_batch)
                loss = F.mse_loss(c_pred, c_batch) + F.mse_loss(f_pred, f_batch)
                if s == 0:
                    loss_before = float(loss.detach().item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(swr_params, 1.0)
                opt.step()
                loss_after = float(loss.detach().item())
        finally:
            self.model.eval()
            for n, p in self.model.named_parameters():
                p.requires_grad_(prev_req_grad[n])
                p.grad = None

        exported = export_final_safetensors(self.model, out_dst.parent, filename=out_dst.name)
        return {
            "steps": int(steps),
            "high_familiarity_cols": high_fam_count,
            "prototypes_replayed": int(f_batch.shape[0]),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "export_path": str(exported["weights"]),
        }

    def handle_command(self, line: str) -> tuple[bool, str]:
        """Handle slash commands (`/teach`, `/style`, `/think`, `/tools`, `/tool_add`, `/probe`, `/lang`, `/sleep`, `/reset`, `/wipe`, `/quit`)."""
        raw = line.strip()
        if raw in ("/quit", "/exit"):
            return True, "Bye."
        if raw == "/reset":
            self.reset_working_memory()
            return False, "[reset] Cleared K=8 working memory & restored Hippocampus to M_tools_base."
        if raw == "/wipe":
            self.wipe_all_memory()
            return False, "[wipe] Cleared K=8 working memory + zeroed Hippocampus fast weights."
        if raw.startswith("/style"):
            parts = raw.split()
            arg = parts[1].lower() if len(parts) > 1 else "neutral"
            if arg in STYLE_NAME_TO_ID:
                self.style_id = STYLE_NAME_TO_ID[arg]
                return False, f"[style] Posterior 32D style subspace set to `{STYLE_ID_TO_NAME[self.style_id]}` (id={self.style_id})."
            return False, "[style] Usage: /style neutral|memo|expository|narrative|strict_code_tool"
        if raw.startswith("/think"):
            parts = raw.split()
            arg = parts[1].lower() if len(parts) > 1 else "dim"
            if arg in ("show", "dim", "hide"):
                self.think_mode = arg
                return False, f"[think] Scratchpad display mode set to `{self.think_mode}`."
            return False, "[think] Usage: /think show|dim|hide"
        if raw == "/tools":
            lines = "\n".join(f"  - {s}" for s in self.tool_signatures)
            return False, f"[tools] {len(self.tool_signatures)} Hippocampus-resident tools:\n{lines}"
        if raw.startswith("/tool_add "):
            sig = raw[len("/tool_add ") :].strip()
            if not sig:
                return False, "[tool_add] Usage: /tool_add tool my_fn(x: str) -> str : description"
            added = self.add_tool_signature(sig)
            return False, f"[tool_add] Swept into Hippocampus & updated M_tools_base: {added}"
        if raw.startswith("/probe"):
            parts = raw.split()
            arg = parts[1].lower() if len(parts) > 1 else ("off" if self.probe_mode != "off" else "on")
            if arg in ("on", "off", "splits"):
                self.probe_mode = arg
                return False, f"[probe] telemetry mode set to `{self.probe_mode}`."
            return False, "[probe] Usage: /probe on|off|splits"
        if raw.startswith("/lang"):
            parts = raw.split()
            arg = parts[1].lower() if len(parts) > 1 else "auto"
            if arg == "auto" or arg in LANG_NAME_TO_ID:
                self.lang_mode = arg
                return False, f"[lang] Broca register mode set to `{self.lang_mode}`."
            return False, "[lang] Usage: /lang auto|en|zh|ja|formal|py|rs|sql|js"
        if raw.startswith("/sleep"):
            parts = raw.split()
            n_steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
            out_p = parts[2] if len(parts) > 2 else "artifacts/export/dcpa_chat_consolidated.safetensors"
            res = self.run_sleep_consolidation(steps=n_steps, out_path=out_p)
            return False, (
                f"[sleep] SWR distilled {res['prototypes_replayed']} prototypes "
                f"(u_lex>0.5 cols={res['high_familiarity_cols']}) over {res['steps']} steps | "
                f"MSE {res['loss_before']:.4f} -> {res['loss_after']:.4f} | saved -> {res['export_path']}"
            )
        return (
            False,
            "[help] Commands: /teach <text> | /style neutral|memo|expository|narrative | "
            "/think show|dim|hide | /tools | /tool_add <sig> | /probe on|off|splits | "
            "/lang auto|zh|en|ja|py|rs | /sleep | /reset | /wipe | /quit",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA M3 Pro Local O(1) Stateful Interactive CLI & AST Tool Sandbox")
    ap.add_argument("--weights", type=str, default="artifacts/export/dcpa_chat_sft.safetensors")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-bytes", type=int, default=256)
    args = ap.parse_args()

    device = get_device(preferred=args.device, deterministic=False)
    print(f"{ANSI_DIM}[DCPA M3 REPL] Loading {args.weights} on device={device.type} ...{ANSI_RESET}", flush=True)
    model = load_dcpa_model_from_weights(args.weights, config_path=args.config, device=device)
    session = DCPAChatSession(
        model,
        device=device,
        temp=args.temp,
        top_p=args.top_p,
        max_new_bytes=args.max_new_bytes,
    )
    print(
        f"{ANSI_BOLD}DCPA O(1) Stateful REPL + AST Tool Bus ready{ANSI_RESET} "
        f"{ANSI_DIM}(K={model.cfg.k_slots}, d_h={model.cfg.d_h}, L_max={model.cfg.l_max}, "
        f"tools={len(session.tool_signatures)}){ANSI_RESET}\n"
        f"{ANSI_DIM}Commands: /teach | /style | /think | /tools | /tool_add | /probe | /lang | /sleep | /reset | /wipe | /quit{ANSI_RESET}\n",
        flush=True,
    )

    while True:
        try:
            user_line = input(f"{ANSI_BOLD}you>{ANSI_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{ANSI_DIM}Bye.{ANSI_RESET}")
            break
        if not user_line:
            continue

        if user_line.startswith("/teach "):
            user_line = user_line[len("/teach ") :].strip()
        elif user_line.startswith("/"):
            should_exit, msg = session.handle_command(user_line)
            print(f"{ANSI_DIM}{msg}{ANSI_RESET}", flush=True)
            if should_exit:
                break
            continue

        sys.stdout.write(f"{ANSI_WHITE}dcpa>{ANSI_RESET} ")
        sys.stdout.flush()
        _, telem = session.chat_turn(
            user_line,
            on_chunk=lambda s: (sys.stdout.write(s), sys.stdout.flush()),
        )
        sys.stdout.write("\n")
        if session.probe_mode != "off":
            lang_lbl = LANG_ID_TO_NAME.get(telem.lang_id, str(telem.lang_id))
            print(f"{ANSI_DIM}{telem.format_line()} [lang={lang_lbl}]{ANSI_RESET}", flush=True)
            if session.probe_mode == "splits":
                print(f"{ANSI_DIM}  SGSC |: {telem.segmented_preview}{ANSI_RESET}", flush=True)


if __name__ == "__main__":
    main()
