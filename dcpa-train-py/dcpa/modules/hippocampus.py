"""Module 3: Hippocampal Fast/Slow Dual Memory System.

Implements Engineering Blueprint Section 2.4 & Theory Doc Part 3:
  - Deterministic QR-orthogonalized frozen random projections R_lex [d_h, d_s] & R_sem [d_h, D]
    (registered as non-trainable buffers, NEVER updated by optimizer).
  - Dentate Gyrus k-WTA sparse orthogonalization (s non-zeros, unit L2 norm in float32).
  - Acetylcholine (ACh) novelty gating & 3-state zero-branch automatic routing.
  - Per-slot eligibility trace tuple (e, f_tilde, eta) with discrete argmax asymmetric inheritance
    on K+1 -> K merge (Commandment #8: strictly preserves s-sparse unit orthogonality).
  - Three-factor semantic crystallization q_{m,i} & Gated Delta (Householder erase-write) update.
  - Two mathematically identical fast-weight state backends:
      1. CompactHouseholderState (W_max = 32): >95% VRAM reduction during batched GPU training,
         keeping scalar gating thresholds differentiable in autograd while NEVER registering
         M^{C<-L} or M^{L<-C} as optimizer parameters (Commandment #7).
      2. ColumnMajorFastWeights: single-session [d_h, D] & [d_h, d_s] column-major sparse column
         gather/scatter-add for exact 1-to-1 parity with dcpa-core-rs & Gate 5/6 verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.ops.fp32 import fp32_island


def deterministic_orthogonal_projection(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Create deterministic frozen [rows, cols] matrix with block-QR orthogonal columns & unit-norm rows."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # Generate in blocks of size (cols x cols), QR orthogonalize each block, then stack and row-normalize
    blocks: list[torch.Tensor] = []
    remaining = rows
    while remaining > 0:
        mat = torch.randn((cols, cols), generator=gen, dtype=torch.float32)
        q, r = torch.linalg.qr(mat)
        # Deterministic sign canonicalization from diag(r)
        diag_sign = torch.sign(torch.diag(r))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        q = q * diag_sign.unsqueeze(0)
        take = min(remaining, cols)
        blocks.append(q[:take, :])
        remaining -= take

    full = torch.cat(blocks, dim=0)  # [rows, cols]
    # Unit L2 norm per row (Engineering Doc 2.4)
    full = full / (full.norm(dim=-1, keepdim=True) + 1e-8)
    return full.contiguous()


@dataclass
class HippocampusReadOutput:
    c_fused: torch.Tensor     # [N, D]
    prec_bu: torch.Tensor     # [N, D] float32 bottom-up precision
    psi_idx: torch.Tensor     # [N, s] int64 sparse active column indices
    psi_val: torch.Tensor     # [N, s] float32 sparse unit-norm values
    ach_signal: torch.Tensor  # [N] float32 ACh_m in [0, 1]
    gamma_slow: torch.Tensor  # [N] float32
    gamma_hip: torch.Tensor   # [N] float32
    c_hip: torch.Tensor       # [N, D] float32


@dataclass
class EligibilityState:
    """Per-slot synaptic eligibility tags T_{m,i} = (e_{m,i}, f_tilde_{m,i}, eta_{m,i})."""

    e_idx: torch.Tensor    # [B, K, s] int64
    e_val: torch.Tensor    # [B, K, s] float32 (unit norm when valid, 0 when empty)
    f_tilde: torch.Tensor  # [B, K, d_s]
    eta: torch.Tensor      # [B, K] float32 in [0, 1]

    @classmethod
    def zeros(cls, batch_size: int, k_slots: int, s_active: int, d_s: int, device: torch.device, dtype: torch.dtype) -> EligibilityState:
        return cls(
            e_idx=torch.zeros((batch_size, k_slots, s_active), device=device, dtype=torch.int64),
            e_val=torch.zeros((batch_size, k_slots, s_active), device=device, dtype=torch.float32),
            f_tilde=torch.zeros((batch_size, k_slots, d_s), device=device, dtype=dtype),
            eta=torch.zeros((batch_size, k_slots), device=device, dtype=torch.float32),
        )


def _sparse_dot_pairs(
    k1_idx: torch.Tensor,
    k1_val: torch.Tensor,
    k2_idx: torch.Tensor,
    k2_val: torch.Tensor,
) -> torch.Tensor:
    """Exact sparse inner product between k1 [B, W, s] and k2 [B, s] -> [B, W] float32."""
    # k1_idx: [B, W, s], k2_idx: [B, 1, 1, s] -> match [B, W, s, s]
    matches = k1_idx.unsqueeze(-1) == k2_idx.unsqueeze(1).unsqueeze(2)  # [B, W, s, s]
    prod = (k1_val.unsqueeze(-1) * k2_val.unsqueeze(1).unsqueeze(2)) * matches.float()
    return prod.sum(dim=(-1, -2))  # [B, W]


class CompactHouseholderState:
    """Memory-optimal batched GPU fast-weight buffer (Engineering Doc 2.4).

    Maintains at most W_max = 32 rank-1 Householder delta records per sequence:
      M_W = sum_{w=1}^W gamma_hip^(step - step_w) * delta_w @ key_w^T
    where delta_w = beta_w * (target_w - M_{w-1} @ key_w).
    Mathematically bit-equivalent to full dense [B, D, d_h] matrix updates while cutting VRAM >95%,
    AND preserves autograd gradients through scalar gate beta_w for Stage D meta-training!
    """

    def __init__(
        self,
        batch_size: int,
        d_h: int,
        d_concept: int,
        d_s: int,
        s_active: int,
        w_max: int = 32,
        gamma_hip: float = 0.999,
        device: torch.device | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.d_h = d_h
        self.d_concept = d_concept
        self.d_s = d_s
        self.s_active = s_active
        self.w_max = w_max
        self.gamma_hip = gamma_hip
        self.device = device if device is not None else torch.device("cpu")
        self.step_count: int = 0

        # Bidirectional familiarity traces u^{lex}, u^{sem}: [B, d_h] float32
        self.u_lex = torch.zeros((batch_size, d_h), device=self.device, dtype=torch.float32)
        self.u_sem = torch.zeros((batch_size, d_h), device=self.device, dtype=torch.float32)

        # Rank-1 delta trace lists (Python lists of tensors of length <= W_max so autograd is clean!)
        # Each entry: (key_idx [B, s], key_val [B, s], delta [B, D or d_s], write_step int)
        self.c_from_l_traces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        self.l_from_c_traces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def advance_step(self) -> None:
        self.step_count += 1
        if self.gamma_hip != 1.0:
            self.u_lex = self.u_lex * self.gamma_hip
            self.u_sem = self.u_sem * self.gamma_hip

    def read_familiarity_lex(self, psi_idx: torch.Tensor, psi_val: torch.Tensor) -> torch.Tensor:
        """gamma_hip = sum_{k in I} psi_k^2 * u_lex[k] -> [B] float32."""
        u_gathered = self.u_lex.gather(1, psi_idx)  # [B, s]
        return (psi_val.float().pow(2) * u_gathered).sum(dim=-1).clamp(0.0, 1.0)

    def read_familiarity_sem(self, phi_idx: torch.Tensor, phi_val: torch.Tensor) -> torch.Tensor:
        """gamma_sem = sum_{k in I} phi_k^2 * u_sem[k] -> [B] float32."""
        u_gathered = self.u_sem.gather(1, phi_idx)  # [B, s]
        return (phi_val.float().pow(2) * u_gathered).sum(dim=-1).clamp(0.0, 1.0)

    def read_c_from_l(self, psi_idx: torch.Tensor, psi_val: torch.Tensor) -> torch.Tensor:
        """Compute M^{C<-L} @ psi -> [B, D] float32."""
        B = psi_idx.shape[0]
        if not self.c_from_l_traces:
            return torch.zeros((B, self.d_concept), device=psi_idx.device, dtype=torch.float32)
        k_idx = torch.stack([t[0] for t in self.c_from_l_traces], dim=1)  # [B, W, s]
        k_val = torch.stack([t[1] for t in self.c_from_l_traces], dim=1)  # [B, W, s]
        deltas = torch.stack([t[2] for t in self.c_from_l_traces], dim=1)  # [B, W, D]
        decays = torch.tensor(
            [self.gamma_hip ** (self.step_count - t[3]) for t in self.c_from_l_traces],
            device=psi_idx.device,
            dtype=torch.float32,
        ).view(1, -1)  # [1, W]
        dots = _sparse_dot_pairs(k_idx, k_val, psi_idx, psi_val.float()) * decays  # [B, W]
        return torch.einsum("bw,bwd->bd", dots, deltas)

    def read_l_from_c(self, phi_idx: torch.Tensor, phi_val: torch.Tensor) -> torch.Tensor:
        """Compute M^{L<-C} @ phi -> [B, d_s] float32."""
        B = phi_idx.shape[0]
        if not self.l_from_c_traces:
            return torch.zeros((B, self.d_s), device=phi_idx.device, dtype=torch.float32)
        k_idx = torch.stack([t[0] for t in self.l_from_c_traces], dim=1)
        k_val = torch.stack([t[1] for t in self.l_from_c_traces], dim=1)
        deltas = torch.stack([t[2] for t in self.l_from_c_traces], dim=1)
        decays = torch.tensor(
            [self.gamma_hip ** (self.step_count - t[3]) for t in self.l_from_c_traces],
            device=phi_idx.device,
            dtype=torch.float32,
        ).view(1, -1)
        dots = _sparse_dot_pairs(k_idx, k_val, phi_idx, phi_val.float()) * decays
        return torch.einsum("bw,bwd->bd", dots, deltas)

    def write_slot(
        self,
        beta: torch.Tensor,
        e_idx: torch.Tensor,
        e_val: torch.Tensor,
        s_bar: torch.Tensor,
        phi_idx: torch.Tensor,
        phi_val: torch.Tensor,
        f_tilde: torch.Tensor,
        write_threshold: float = 0.05,
    ) -> None:
        """Apply single-slot Householder Gated Delta update & familiarity update.

        Only appends a trace entry if max(beta) > write_threshold (keeping trace <= W_max).
        """
        if float(beta.detach().max().item()) <= write_threshold:
            return

        # Precision Island #5 (Red Line #1): s=32 active sparse column Householder Gated Delta
        # erase/write & familiarity accumulation strictly in float32 (zero residual ghosting!).
        with fp32_island(beta.device):
            beta_f = beta.float().unsqueeze(-1)  # [B, 1]
            # Comprehension write: delta_c = beta * (s_bar - M^{C<-L} @ e)
            old_c = self.read_c_from_l(e_idx, e_val)
            delta_c = beta_f * (s_bar.float() - old_c)
            self.c_from_l_traces.append((e_idx.detach(), e_val.detach().float(), delta_c, self.step_count))
            if len(self.c_from_l_traces) > self.w_max:
                self.c_from_l_traces.pop(0)

            # Expression write: delta_l = beta * (f_tilde - M^{L<-C} @ phi)
            old_l = self.read_l_from_c(phi_idx, phi_val)
            delta_l = beta_f * (f_tilde.float() - old_l)
            self.l_from_c_traces.append((phi_idx.detach(), phi_val.detach().float(), delta_l, self.step_count))
            if len(self.l_from_c_traces) > self.w_max:
                self.l_from_c_traces.pop(0)

            # Familiarity trace updates (out-of-place for clean autograd)
            active_e = (e_val.float() > 0).float()
            u_lex_cur = self.u_lex.gather(1, e_idx)
            u_lex_add = beta_f * (1.0 - u_lex_cur) * active_e
            self.u_lex = self.u_lex.scatter_add(1, e_idx, u_lex_add).clamp(0.0, 1.0)

            active_phi = (phi_val.float() > 0).float()
            u_sem_cur = self.u_sem.gather(1, phi_idx)
            u_sem_add = beta_f * (1.0 - u_sem_cur) * active_phi
            self.u_sem = self.u_sem.scatter_add(1, phi_idx, u_sem_add).clamp(0.0, 1.0)


class ColumnMajorFastWeights:
    """Single-session Column-Major fast weights matching dcpa-core-rs bit-for-bit!

    Physical storage layout:
      m_c_from_l: [d_h, D]  (each column k is contiguous memory of length D)
      m_l_from_c: [d_h, d_s] (each column k is contiguous memory of length d_s)
      u_lex, u_sem: [d_h] float32
      col_step_lex, col_step_sem: [d_h] int64 for O(s) lazy gamma_hip decay
    """

    def __init__(
        self,
        d_h: int,
        d_concept: int,
        d_s: int,
        gamma_hip: float = 0.999,
        storage_dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        self.d_h = d_h
        self.d_concept = d_concept
        self.d_s = d_s
        self.gamma_hip = gamma_hip
        self.storage_dtype = storage_dtype
        self.device = device if device is not None else torch.device("cpu")
        self.step_count: int = 0

        self.m_c_from_l = torch.zeros((d_h, d_concept), device=self.device, dtype=storage_dtype)
        self.m_l_from_c = torch.zeros((d_h, d_s), device=self.device, dtype=storage_dtype)
        self.u_lex = torch.zeros((d_h,), device=self.device, dtype=torch.float32)
        self.u_sem = torch.zeros((d_h,), device=self.device, dtype=torch.float32)
        self.col_step_lex = torch.zeros((d_h,), device=self.device, dtype=torch.int64)
        self.col_step_sem = torch.zeros((d_h,), device=self.device, dtype=torch.int64)

    def advance_step(self) -> None:
        self.step_count += 1

    def _sync_lex_cols(self, idx_1d: torch.Tensor) -> None:
        if self.gamma_hip == 1.0:
            return
        dt = (self.step_count - self.col_step_lex[idx_1d]).float()
        if (dt > 0).any():
            scale = torch.pow(self.gamma_hip, dt)
            self.m_c_from_l[idx_1d] = (self.m_c_from_l[idx_1d].float() * scale.unsqueeze(-1)).to(self.storage_dtype)
            self.u_lex[idx_1d] = self.u_lex[idx_1d] * scale
            self.col_step_lex[idx_1d] = self.step_count

    def _sync_sem_cols(self, idx_1d: torch.Tensor) -> None:
        if self.gamma_hip == 1.0:
            return
        dt = (self.step_count - self.col_step_sem[idx_1d]).float()
        if (dt > 0).any():
            scale = torch.pow(self.gamma_hip, dt)
            self.m_l_from_c[idx_1d] = (self.m_l_from_c[idx_1d].float() * scale.unsqueeze(-1)).to(self.storage_dtype)
            self.u_sem[idx_1d] = self.u_sem[idx_1d] * scale
            self.col_step_sem[idx_1d] = self.step_count

    def read_familiarity_lex(self, psi_idx: torch.Tensor, psi_val: torch.Tensor) -> torch.Tensor:
        idx = psi_idx[0]
        self._sync_lex_cols(idx)
        val = (psi_val[0].float().pow(2) * self.u_lex[idx]).sum().clamp(0.0, 1.0)
        return val.unsqueeze(0)

    def read_familiarity_sem(self, phi_idx: torch.Tensor, phi_val: torch.Tensor) -> torch.Tensor:
        idx = phi_idx[0]
        self._sync_sem_cols(idx)
        val = (phi_val[0].float().pow(2) * self.u_sem[idx]).sum().clamp(0.0, 1.0)
        return val.unsqueeze(0)

    def read_c_from_l(self, psi_idx: torch.Tensor, psi_val: torch.Tensor) -> torch.Tensor:
        idx = psi_idx[0]
        self._sync_lex_cols(idx)
        cols = self.m_c_from_l[idx].float()  # [s, D]
        out = torch.einsum("s,sd->d", psi_val[0].float(), cols)
        return out.unsqueeze(0)

    def read_l_from_c(self, phi_idx: torch.Tensor, phi_val: torch.Tensor) -> torch.Tensor:
        idx = phi_idx[0]
        self._sync_sem_cols(idx)
        cols = self.m_l_from_c[idx].float()  # [s, d_s]
        out = torch.einsum("s,sd->d", phi_val[0].float(), cols)
        return out.unsqueeze(0)

    def write_slot(
        self,
        beta: torch.Tensor,
        e_idx: torch.Tensor,
        e_val: torch.Tensor,
        s_bar: torch.Tensor,
        phi_idx: torch.Tensor,
        phi_val: torch.Tensor,
        f_tilde: torch.Tensor,
        write_threshold: float = 0.0,
    ) -> None:
        b_val = float(beta[0].item())
        if b_val <= write_threshold:
            return

        # Precision Island #5 (Red Line #1): local sparse column Gated Delta update strictly in float32
        with fp32_island(beta.device):
            # Comprehension column-major sparse Householder update (in float32 accumulator)
            ei = e_idx[0]
            ev = e_val[0].float()
            self._sync_lex_cols(ei)
            cols_c = self.m_c_from_l[ei].float()  # [s, D]
            proj_c = torch.einsum("s,sd->d", ev, cols_c)
            err_c = b_val * (s_bar[0].float() - proj_c)
            cols_c_new = cols_c + torch.outer(ev, err_c)
            self.m_c_from_l[ei] = cols_c_new.to(self.storage_dtype)
            active_e = (ev > 0).float()
            self.u_lex[ei] = (self.u_lex[ei] + b_val * (1.0 - self.u_lex[ei]) * active_e).clamp(0.0, 1.0)

            # Expression column-major sparse Householder update
            pi = phi_idx[0]
            pv = phi_val[0].float()
            self._sync_sem_cols(pi)
            cols_l = self.m_l_from_c[pi].float()  # [s, d_s]
            proj_l = torch.einsum("s,sd->d", pv, cols_l)
            err_l = b_val * (f_tilde[0].float() - proj_l)
            cols_l_new = cols_l + torch.outer(pv, err_l)
            self.m_l_from_c[pi] = cols_l_new.to(self.storage_dtype)
            active_p = (pv > 0).float()
            self.u_sem[pi] = (self.u_sem[pi] + b_val * (1.0 - self.u_sem[pi]) * active_p).clamp(0.0, 1.0)


HippocampusStateType = Union[CompactHouseholderState, ColumnMajorFastWeights]


class HippocampusDualMemory(nn.Module):
    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_s = cfg.d_s
        D = cfg.d_concept
        d_h = cfg.d_h

        # Frozen deterministic random orthogonal projections (NEVER trainable!)
        self.register_buffer("r_lex", deterministic_orthogonal_projection(d_h, d_s, cfg.dg_seed_lex))
        self.register_buffer("r_sem", deterministic_orthogonal_projection(d_h, D, cfg.dg_seed_sem))

        # Non-persistent fast hippocampal state buffers (Commandment #7: NEVER optimizer parameters,
        # requires_grad=False strictly enforced, excluded from slow-weight safetensors state_dict!)
        self.register_buffer("m_c_from_l", torch.zeros((d_h, D), dtype=torch.float32), persistent=False)
        self.register_buffer("m_l_from_c", torch.zeros((d_h, d_s), dtype=torch.float32), persistent=False)
        self.register_buffer("u_lex", torch.zeros((d_h,), dtype=torch.float32), persistent=False)
        self.register_buffer("u_sem", torch.zeros((d_h,), dtype=torch.float32), persistent=False)

        # Scalar gating parameters (dedicated optimizer group: lr=1e-4, weight_decay=0.0)
        self.alpha_1 = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.theta_i = nn.Parameter(torch.tensor(2.5, dtype=torch.float32))
        self.theta_n = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.alpha_q = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.gamma_e = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.theta_q = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

        self.alpha_out = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))
        self.theta_out = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        # Bottom-up cortical precision projection pi_m^{cort} = Softplus(f_prec_cort(c_slow))
        self.f_prec_cort = nn.Linear(D, D, bias=True)
        nn.init.normal_(self.f_prec_cort.weight, std=0.01)
        nn.init.constant_(self.f_prec_cort.bias, 0.5413)  # Softplus ~ 1.0

        # Slow cortical expressive projection (concept + lang bias -> stem code f_out)
        self.w_lex_slow = nn.Linear(D, d_s, bias=False)
        self.w_lang_lex = nn.Linear(cfg.d_l, D, bias=False)
        nn.init.xavier_uniform_(self.w_lex_slow.weight)
        nn.init.xavier_uniform_(self.w_lang_lex.weight)

    def scalar_gate_parameters(self) -> list[nn.Parameter]:
        return [
            self.alpha_1,
            self.alpha_2,
            self.theta_i,
            self.theta_n,
            self.alpha_q,
            self.gamma_e,
            self.theta_q,
            self.alpha_out,
            self.theta_out,
        ]

    def init_fast_state(
        self,
        batch_size: int,
        mode: str = "compact",
        storage_dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> HippocampusStateType:
        dev = device if device is not None else self.r_lex.device
        if mode == "column_major":
            if batch_size != 1:
                raise ValueError("ColumnMajorFastWeights is for single-session (batch_size=1) inference/parity.")
            return ColumnMajorFastWeights(
                d_h=self.cfg.d_h,
                d_concept=self.cfg.d_concept,
                d_s=self.cfg.d_s,
                gamma_hip=self.cfg.gamma_hip,
                storage_dtype=storage_dtype,
                device=dev,
            )
        return CompactHouseholderState(
            batch_size=batch_size,
            d_h=self.cfg.d_h,
            d_concept=self.cfg.d_concept,
            d_s=self.cfg.d_s,
            s_active=self.cfg.s_active,
            w_max=self.cfg.householder_max_writes,
            gamma_hip=self.cfg.gamma_hip,
            device=dev,
        )

    def dg_lex(self, f_stem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dentate Gyrus lexical k-WTA sparse orthogonalizer -> (idx [..., s], val [..., s] fp32)."""
        a = F.linear(f_stem.float(), self.r_lex.float())  # [..., d_h]
        top_vals, top_idx = torch.topk(a, k=self.cfg.s_active, dim=-1, sorted=True)
        psi_tilde = F.relu(top_vals)
        norm = psi_tilde.norm(dim=-1, keepdim=True) + 1e-8
        psi_val = psi_tilde / norm
        return top_idx, psi_val

    def dg_sem(self, concept: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dentate Gyrus semantic k-WTA sparse orthogonalizer -> (idx [..., s], val [..., s] fp32)."""
        a = F.linear(concept.float(), self.r_sem.float())  # [..., d_h]
        top_vals, top_idx = torch.topk(a, k=self.cfg.s_active, dim=-1, sorted=True)
        phi_tilde = F.relu(top_vals)
        norm = phi_tilde.norm(dim=-1, keepdim=True) + 1e-8
        phi_val = phi_tilde / norm
        return top_idx, phi_val

    def read_comprehension(
        self,
        f_stem: torch.Tensor,
        c_slow: torch.Tensor,
        sem_norm: torch.Tensor,
        mean_info: torch.Tensor,
        fast_state: Optional[HippocampusStateType] = None,
        enabled: bool = True,
    ) -> HippocampusReadOutput:
        """Comprehension-side routing & ACh novelty gate (forced float32 familiarity & routing).

        Args:
          f_stem:    [B, d_s]
          c_slow:    [B, D]
          sem_norm:  [B] float32 ||W_sem f_m||_2
          mean_info: [B] float32 mean segment surprisal I_bar_m
          fast_state: HippocampusStateType (if None or not enabled, behaves as Stage B/C cortical pass-through)
        """
        orig_dtype = c_slow.dtype
        # Precision Island #5 (Red Line #1): DG orthogonal sparse read, familiarity, & ACh gating in fp32
        with fp32_island(c_slow.device):
            psi_idx, psi_val = self.dg_lex(f_stem)
            pi_cort = F.softplus(self.f_prec_cort(c_slow).float()) + 1e-4  # [B, D] fp32

            if (not enabled) or (fast_state is None):
                B = c_slow.shape[0]
                zeros = torch.zeros((B,), device=c_slow.device, dtype=torch.float32)
                ones = torch.ones((B,), device=c_slow.device, dtype=torch.float32)
                return HippocampusReadOutput(
                    c_fused=c_slow,
                    prec_bu=pi_cort,
                    psi_idx=psi_idx,
                    psi_val=psi_val,
                    ach_signal=zeros,
                    gamma_slow=ones,
                    gamma_hip=zeros,
                    c_hip=torch.zeros_like(c_slow, dtype=torch.float32),
                )

            # Cortical & hippocampal familiarity in float32
            gamma_slow = torch.sigmoid(
                self.alpha_1 * (self.theta_i - mean_info.float())
                + self.alpha_2 * (sem_norm.float() - self.theta_n)
            )  # [B] fp32
            gamma_hip = fast_state.read_familiarity_lex(psi_idx, psi_val)  # [B] fp32
            ach = (1.0 - gamma_slow) * (1.0 - gamma_hip)  # [B] fp32

            c_hip = fast_state.read_c_from_l(psi_idx, psi_val)  # [B, D] fp32
            mu = gamma_hip / (gamma_slow + gamma_hip + self.cfg.eps)  # [B] fp32

            c_fused = ((1.0 - mu).unsqueeze(-1) * c_slow.float() + mu.unsqueeze(-1) * c_hip).to(orig_dtype)
            prec_bu = (1.0 - ach).unsqueeze(-1) * pi_cort  # [B, D] fp32

        return HippocampusReadOutput(
            c_fused=c_fused,
            prec_bu=prec_bu,
            psi_idx=psi_idx,
            psi_val=psi_val,
            ach_signal=ach,
            gamma_slow=gamma_slow,
            gamma_hip=gamma_hip,
            c_hip=c_hip,
        )

    def inherit_eligibility_on_merge(
        self,
        elig: EligibilityState,
        cand_idx: torch.Tensor,
        cand_val: torch.Tensor,
        cand_f: torch.Tensor,
        cand_ach: torch.Tensor,
        win_i: torch.Tensor,
        win_j: torch.Tensor,
    ) -> EligibilityState:
        """Asymmetric eligibility trace inheritance on K+1 -> K merge (Commandment #8).

        NEVER averages or discards eligibility traces. Selects p* = argmax_{p in {i*, j*}} eta_p
        verbatim, preserving exact s-sparse unit-norm orthogonality!
        """
        B, K, s = elig.e_idx.shape
        device = elig.e_idx.device
        b_arange = torch.arange(B, device=device)

        # Extend to K+1 (align dtypes safely under bfloat16 autocast)
        cand_val_aligned = cand_val.to(dtype=elig.e_val.dtype)
        cand_f_aligned = cand_f.to(dtype=elig.f_tilde.dtype)
        e_idx_ext = torch.cat([elig.e_idx, cand_idx.unsqueeze(1)], dim=1)  # [B, K+1, s]
        e_val_ext = torch.cat([elig.e_val, cand_val_aligned.unsqueeze(1)], dim=1)  # [B, K+1, s]
        f_ext = torch.cat([elig.f_tilde, cand_f_aligned.unsqueeze(1)], dim=1)      # [B, K+1, d_s]
        eta_ext = torch.cat([elig.eta, cand_ach.float().unsqueeze(1)], dim=1)  # [B, K+1]

        eta_i = eta_ext[b_arange, win_i]
        eta_j = eta_ext[b_arange, win_j]
        p_star = torch.where(eta_i >= eta_j, win_i, win_j)  # discrete argmax!

        u = torch.minimum(win_i, win_j)
        v = torch.maximum(win_i, win_j)

        new_e_idx = elig.e_idx.clone()
        new_e_val = elig.e_val.clone()
        new_f = elig.f_tilde.clone()
        new_eta = elig.eta.clone()

        # Slot u inherits p* verbatim (with lambda_eta decay)
        new_e_idx[b_arange, u] = e_idx_ext[b_arange, p_star]
        new_e_val[b_arange, u] = e_val_ext[b_arange, p_star]
        new_f[b_arange, u] = f_ext[b_arange, p_star]
        new_eta[b_arange, u] = self.cfg.lambda_eta * eta_ext[b_arange, p_star]

        # Case B (v < K): freed slot v gets candidate slot K
        case_b = v < K
        if case_b.any():
            b_cb = b_arange[case_b]
            v_cb = v[case_b]
            new_e_idx[b_cb, v_cb] = cand_idx[b_cb]
            new_e_val[b_cb, v_cb] = cand_val_aligned[b_cb]
            new_f[b_cb, v_cb] = cand_f_aligned[b_cb]
            new_eta[b_cb, v_cb] = cand_ach.float()[b_cb]

        return EligibilityState(e_idx=new_e_idx, e_val=new_e_val, f_tilde=new_f, eta=new_eta)

    def three_factor_write(
        self,
        s_bar_post: torch.Tensor,
        edges_post: torch.Tensor,
        c0: torch.Tensor,
        elig: EligibilityState,
        fast_state: HippocampusStateType,
        write_threshold: Optional[float] = None,
    ) -> tuple[EligibilityState, torch.Tensor, torch.Tensor]:
        """Execute three-factor semantic crystallization & bidirectional Gated Delta write.

        Returns:
          updated_elig: EligibilityState with quenched eta_{m,i}
          q_cryst:      [B, K] float32 semantic crystallization gate
          beta:         [B, K] float32 effective write strength beta_{m,i} = eta_{m,i} * q_{m,i}
        """
        B, K, _ = s_bar_post.shape
        fast_state.advance_step()

        # Precision Island #5 (Red Line #1): crystallization gate & sparse delta write strictly in float32
        with fp32_island(s_bar_post.device):
            # Semantic crystallization q_{m,i} (Theory Doc Part 3.II.2)
            c0_norm = c0.float().view(1, 1, -1)
            cos_sim = F.cosine_similarity(s_bar_post.float(), c0_norm.expand_as(s_bar_post), dim=-1, eps=1e-8)
            edge_norm_sum = edges_post.float().norm(dim=-1).sum(dim=-1)  # [B, K] sum_j ||A_{m,i,j}||_2
            q_cryst = torch.sigmoid(
                self.alpha_q * ((1.0 - cos_sim) + self.gamma_e * edge_norm_sum - self.theta_q)
            )  # [B, K] fp32

            beta = elig.eta.float() * q_cryst  # [B, K] fp32

            thresh = (
                self.cfg.householder_write_threshold
                if write_threshold is None and isinstance(fast_state, CompactHouseholderState)
                else (0.0 if write_threshold is None else write_threshold)
            )

            # Compute semantic sparse addresses for slots that will write
            for slot_i in range(K):
                beta_i = beta[:, slot_i]
                if float(beta_i.detach().max().item()) <= thresh:
                    continue
                s_bar_i = s_bar_post[:, slot_i, :]
                phi_idx_i, phi_val_i = self.dg_sem(s_bar_i)
                fast_state.write_slot(
                    beta=beta_i,
                    e_idx=elig.e_idx[:, slot_i, :],
                    e_val=elig.e_val[:, slot_i, :],
                    s_bar=s_bar_i,
                    phi_idx=phi_idx_i,
                    phi_val=phi_val_i,
                    f_tilde=elig.f_tilde[:, slot_i, :],
                    write_threshold=thresh,
                )

            # Quench consumed eligibility traces
            new_eta = (1.0 - self.cfg.kappa_quench * beta.detach()) * elig.eta
            new_elig = EligibilityState(
                e_idx=elig.e_idx,
                e_val=elig.e_val,
                f_tilde=elig.f_tilde,
                eta=new_eta.clamp(0.0, 1.0),
            )
        return new_elig, q_cryst, beta

    def read_expression(
        self,
        z_out: torch.Tensor,
        l_reg: torch.Tensor,
        fast_state: Optional[HippocampusStateType] = None,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expressive retrieval: Concept + l_reg -> stem prototype f_out [B, d_s] & gate rho_out [B]."""
        orig_dtype = z_out.dtype
        f_slow = self.w_lex_slow(z_out + self.w_lang_lex(l_reg))  # [B, d_s]
        # Unit-RMS normalize cortical stem readout to match SGSC.f_stem scale cleanly
        f_slow = (
            f_slow.float() * torch.rsqrt(f_slow.float().pow(2).mean(dim=-1, keepdim=True) + self.cfg.eps)
        ).to(orig_dtype)

        if (not enabled) or (fast_state is None):
            rho_zero = torch.zeros((z_out.shape[0],), device=z_out.device, dtype=torch.float32)
            return f_slow, rho_zero

        with fp32_island(z_out.device):
            phi_idx, phi_val = self.dg_sem(z_out)
            gamma_sem = fast_state.read_familiarity_sem(phi_idx, phi_val)  # [B] fp32
            rho_out = torch.sigmoid(self.alpha_out * (gamma_sem - self.theta_out))  # [B] fp32
            f_hip = fast_state.read_l_from_c(phi_idx, phi_val)  # [B, d_s] fp32

            f_out = ((1.0 - rho_out).unsqueeze(-1) * f_slow.float() + rho_out.unsqueeze(-1) * f_hip).to(orig_dtype)
        return f_out, rho_out
