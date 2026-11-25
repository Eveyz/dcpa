"""Unit & integration tests for DCPA-Turbo architecture, W&B resume continuity, and Git sync."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Any

import torch

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.modules.cortical import CorticalCarryState
from dcpa.modules.hippocampus import ColumnMajorFastWeights, CompactHouseholderState
from dcpa.ops.rope import apply_concept_rope_2d, compute_segment_scope_depths
from scripts.colab_git_sync import sync_repo_via_git
from train.watchdog_and_ckpt import CheckpointManager
from utils.wandb_tracker import WandbTracker


def test_concept_rope_2d_orthogonality_and_causal_scope_depth() -> None:
    torch.manual_seed(42)
    x = torch.randn(5, 4, 32, dtype=torch.float32)
    pos = torch.tensor([0, 3, 17, 64, 127], dtype=torch.int64)
    dep = torch.tensor([0, 2, 7, 15, 31], dtype=torch.int32)

    x_rot = apply_concept_rope_2d(x, pos, dep, theta_base=10000.0, scope_period=33.0)
    x_rec = apply_concept_rope_2d(x_rot, pos, dep, theta_base=10000.0, scope_period=33.0, inverse=True)
    assert (x_rec - x).abs().max().item() < 1e-5

    # Exact relative shift invariance: <R(p1, d1) q, R(p2, d2) k> == <R(p1-p2, d1-d2) q, k>
    q = torch.randn(1, 4, 32)
    k = torch.randn(1, 4, 32)
    p1, d1 = torch.tensor([29]), torch.tensor([6], dtype=torch.int32)
    p2, d2 = torch.tensor([11]), torch.tensor([2], dtype=torch.int32)
    q_abs = apply_concept_rope_2d(q, p1, d1)
    k_abs = apply_concept_rope_2d(k, p2, d2)
    q_rel = apply_concept_rope_2d(q, p1 - p2, d1 - d2)
    dot_abs = (q_abs * k_abs).sum(dim=-1)
    dot_rel = (q_rel * k).sum(dim=-1)
    assert (dot_abs - dot_rel).abs().max().item() < 1e-5

    # Strict `< cu_seqlens[m]` causality of `seg_start_depth[m]`
    raw_a = torch.tensor([list(b"fn f() { let a = [1, 2]; }")], dtype=torch.uint8)
    raw_b = torch.tensor([list(b"fn f() { let a = (((((({{{")], dtype=torch.uint8)
    cu = torch.tensor([0, 9, 17, 26], dtype=torch.int32)
    doc_s = torch.tensor([True, False, False], dtype=torch.bool)
    s_a, e_a, _ = compute_segment_scope_depths(raw_a, cu, doc_s)
    s_b, _, _ = compute_segment_scope_depths(raw_b, cu, doc_s)
    # Segment 2 starts at byte 17; mutating bytes 17..25 must NOT change seg_start_depth[2]!
    assert torch.equal(s_a[:3], s_b[:3])
    assert int(e_a[-1].item()) == 0  # balanced closing `}` returns depth to 0


def test_hippocampus_wy_parallel_matches_sequential_8_slot_householder() -> None:
    torch.manual_seed(2026)
    d_h, d_c, d_s, k_slots, s_act = 128, 32, 24, 8, 8
    wy = CompactHouseholderState(
        batch_size=1,
        d_h=d_h,
        d_concept=d_c,
        d_s=d_s,
        s_active=s_act,
        gamma_hip=0.995,
        w_max=64,
    )
    cm = ColumnMajorFastWeights(
        d_h=d_h,
        d_concept=d_c,
        d_s=d_s,
        gamma_hip=0.995,
        storage_dtype=torch.float32,
    )

    for _step in range(3):
        wy.advance_step()
        cm.advance_step()

        beta = torch.rand(1, k_slots) * 0.9
        # Force overlapping sparse indices across the 8 slots to exercise non-commuting reflections!
        pool_lex = torch.tensor([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47], dtype=torch.long)
        pool_sem = torch.tensor([2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46], dtype=torch.long)

        e_idx = torch.stack([pool_lex[torch.randperm(len(pool_lex))[:s_act]] for _ in range(k_slots)], dim=0).unsqueeze(0)
        phi_idx = torch.stack([pool_sem[torch.randperm(len(pool_sem))[:s_act]] for _ in range(k_slots)], dim=0).unsqueeze(0)
        e_raw = torch.rand(1, k_slots, s_act) + 0.1
        e_val = e_raw / e_raw.norm(dim=-1, keepdim=True)
        p_raw = torch.rand(1, k_slots, s_act) + 0.1
        phi_val = p_raw / p_raw.norm(dim=-1, keepdim=True)
        s_bar = torch.randn(1, k_slots, d_c)
        f_tilde = torch.randn(1, k_slots, d_s)

        wy.write_slots_wy_parallel(
            beta=beta,
            e_idx=e_idx,
            e_val=e_val,
            s_bar=s_bar,
            phi_idx=phi_idx,
            phi_val=phi_val,
            f_tilde=f_tilde,
            write_threshold=0.05,
        )
        for slot_i in range(k_slots):
            cm.write_slot(
                beta=beta[:, slot_i],
                e_idx=e_idx[:, slot_i, :],
                e_val=e_val[:, slot_i, :],
                s_bar=s_bar[:, slot_i, :],
                phi_idx=phi_idx[:, slot_i, :],
                phi_val=phi_val[:, slot_i, :],
                f_tilde=f_tilde[:, slot_i, :],
                write_threshold=0.05,
            )

    # Probe random readout vectors across active columns
    for _ in range(5):
        q_idx = pool_lex[:s_act].unsqueeze(0)
        q_val = torch.ones(1, s_act) / (s_act**0.5)
        c_wy = wy.read_c_from_l(q_idx, q_val)
        c_cm = cm.read_c_from_l(q_idx, q_val)
        assert (c_wy - c_cm).abs().max().item() < 1e-5

        qs_idx = pool_sem[:s_act].unsqueeze(0)
        l_wy = wy.read_l_from_c(qs_idx, q_val)
        l_cm = cm.read_l_from_c(qs_idx, q_val)
        assert (l_wy - l_cm).abs().max().item() < 1e-5

        fam_wy = wy.read_familiarity_lex(q_idx, q_val)
        fam_cm = cm.read_familiarity_lex(q_idx, q_val)
        assert (fam_wy - fam_cm).abs().max().item() < 1e-5


def test_cortical_sandwich_parallel_vs_streaming_parity() -> None:
    torch.manual_seed(99)
    cfg = DCPAConfig.tiny()
    model = DCPAModel(cfg).eval()

    raw = torch.tensor(
        [list("fn compute(a: i32) { let b = (a + 1) * 2; } // 你好世界".encode("utf-8"))],
        dtype=torch.uint8,
    )
    with torch.no_grad():
        sgsc_out = model.sgsc(raw, tau_s=0.35)
        m_tot = int(sgsc_out.f_stem.shape[0])
        assert m_tot >= 4
        doc_ids = torch.zeros((m_tot,), dtype=torch.long)
        pos_in_doc = torch.arange(m_tot, dtype=torch.long)
        seg_doc_start = torch.zeros((m_tot,), dtype=torch.bool)
        seg_doc_start[0] = True
        seg_start_d, seg_end_d, _ = compute_segment_scope_depths(raw, sgsc_out.cu_seqlens, seg_doc_start)

        h_enc_par, c_kv_par, _ = model.cortical.encode_parallel(
            c_slow=sgsc_out.c_slow,
            f_stem=sgsc_out.f_stem,
            doc_ids=doc_ids,
            pos_in_doc=pos_in_doc,
            seg_end_depth=seg_end_d,
        )

        # Build dummy pre-step graph trajectories and z_ref_prev to verify decode_parallel vs decode_step_streaming
        l_reg = model.pmd.get_lang_vec(torch.tensor([4], dtype=torch.long))
        l_reg_seg = l_reg.expand(m_tot, -1)
        pre_slots = torch.randn(m_tot, cfg.k_slots, cfg.d_concept)
        pre_edges = torch.tanh(torch.randn(m_tot, cfg.k_slots, cfg.k_slots, cfg.d_r))
        pre_sal = torch.randn(m_tot, cfg.k_slots)
        pre_phase = torch.arange(cfg.k_slots, dtype=torch.float32).unsqueeze(0).expand(m_tot, -1)
        z_stars = torch.randn(m_tot, cfg.d_concept)
        z_ref_prev = torch.cat([model.cortical.bos_concept.unsqueeze(0), z_stars[:-1]], dim=0)

        z_dec_par, copy_par, _ = model.cortical.decode_parallel(
            z_ref_prev=z_ref_prev,
            l_reg_seg=l_reg_seg,
            c_kv=c_kv_par,
            doc_ids=doc_ids,
            pos_in_doc=pos_in_doc,
            seg_start_depth=seg_start_d,
            seg_end_depth=seg_end_d,
            pre_slots=pre_slots,
            pre_edges=pre_edges,
            pre_salience=pre_sal,
            pre_phase=pre_phase,
        )

        carry = model.cortical.init_carry_state(1, dtype=torch.float32)
        h_enc_stream = []
        z_dec_stream = []
        copy_stream = []
        for m in range(m_tot):
            # Decode segment m strictly BEFORE pushing segment m into ring!
            carry_dec = CorticalCarryState(
                ring_kv=carry.ring_kv,
                ring_pos=carry.ring_pos,
                ring_depth=carry.ring_depth,
                ring_valid=carry.ring_valid,
                doc_pos=carry.doc_pos,
                ast_depth=seg_start_d[m : m + 1],
                prev_z_ref=carry.prev_z_ref,
                has_prev_seg=carry.has_prev_seg,
            )
            zd_m, cp_m = model.cortical.decode_step_streaming(
                l_reg=l_reg,
                carry=carry_dec,
                slots=pre_slots[m : m + 1],
                edges=pre_edges[m : m + 1],
                salience=pre_sal[m : m + 1],
                phase=pre_phase[m : m + 1],
            )
            z_dec_stream.append(zd_m[0])
            copy_stream.append(cp_m[0])

            he_m, carry = model.cortical.encode_step_streaming(
                c_slow_step=sgsc_out.c_slow[m : m + 1],
                f_stem_step=sgsc_out.f_stem[m : m + 1],
                seg_end_depth=seg_end_d[m : m + 1],
                carry=carry,
            )
            h_enc_stream.append(he_m[0])
            carry = CorticalCarryState(
                ring_kv=carry.ring_kv,
                ring_pos=carry.ring_pos,
                ring_depth=carry.ring_depth,
                ring_valid=carry.ring_valid,
                doc_pos=carry.doc_pos,
                ast_depth=carry.ast_depth,
                prev_z_ref=z_stars[m : m + 1],
                has_prev_seg=torch.ones((1,), dtype=torch.bool),
            )

        assert (torch.stack(h_enc_stream, dim=0) - h_enc_par).abs().max().item() < 1e-5
        assert (torch.stack(z_dec_stream, dim=0) - z_dec_par).abs().max().item() < 1e-5
        assert (torch.stack(copy_stream, dim=0) - copy_par).abs().max().item() < 1e-5


def test_contractive_gate_long_horizon_grad_and_slot0_zero_ulp() -> None:
    torch.manual_seed(777)
    cfg = DCPAConfig.micro()
    bcg = DCPAModel(cfg).bcg_pc
    bcg.train()

    n_steps = 20
    c_seq = torch.randn(n_steps, cfg.d_concept, requires_grad=True)
    prec_bu = torch.ones(1, cfg.d_concept)
    ach = torch.zeros(1)
    state = bcg.init_state(1)

    for t in range(n_steps):
        trig = torch.ones(1) if t == 1 else torch.zeros(1)
        out = bcg.step(
            c_fused=c_seq[t : t + 1],
            prec_bu=prec_bu,
            ach=ach,
            state=state,
            tau_m=0.5,
            use_gumbel=False,
            pin_trigger=trig,
        )
        state = out.state
        if t == 1:
            pinned_ref = state.slots[:, 0, :].detach().clone()
        elif t > 1:
            # Exact 0.0 ULP preservation on pinned Slot 0!
            assert torch.equal(state.slots[:, 0, :], pinned_ref)

    loss = out.g_ctx.pow(2).sum()
    loss.backward()
    assert c_seq.grad is not None
    assert torch.isfinite(c_seq.grad).all()
    assert float(c_seq.grad[0].norm().item()) > 0.0


class _FakeWandbRun:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.id = kwargs.get("id")
        self.kwargs = kwargs
        self.logged: list[tuple[int, dict[str, Any]]] = []
        self.summary: dict[str, Any] = {}
        self.finished = False
        self.url = f"https://wandb.ai/mock/{self.id}"

    def log(self, data: dict[str, Any], step: int = 0) -> None:
        self.logged.append((step, dict(data)))

    def finish(self) -> None:
        self.finished = True


class _FakeWandbModule:
    def __init__(self) -> None:
        self.runs: list[_FakeWandbRun] = []

    def init(self, **kwargs: Any) -> _FakeWandbRun:
        r = _FakeWandbRun(kwargs)
        self.runs.append(r)
        return r


def test_wandb_tracker_and_checkpoint_run_id_resume(tmp_path: Path) -> None:
    fake_wb = _FakeWandbModule()
    logs_dir = tmp_path / "nvme_logs"
    drive_logs = tmp_path / "drive_logs"
    ckpt_dir = tmp_path / "ckpts"

    tracker1 = WandbTracker(
        enabled=True,
        is_primary=True,
        project="dcpa-turbo",
        run_name="unit-test-run",
        mode="offline",
        log_every_steps=2,
        logs_dir=logs_dir,
        drive_logs_dir=drive_logs,
        wandb_module=fake_wb,
    )
    rid1 = tracker1.start(resume_step=0)
    assert rid1 is not None
    assert (drive_logs / "wandb_run.json").exists()

    tracker1.log_train_step(
        step=2,
        stage_id=2,
        metrics={"loss_total": 3.2, "loss_lang": 2.8, "spike_rate": 0.105, "mean_ach": 0.42},
        applied_lrs={"concept": 3e-4},
        kb_per_sec=120.5,
        peak_mem_gb=4.2,
        total_steps=10,
    )
    assert len(fake_wb.runs[0].logged) == 1
    assert "train/bpc" in fake_wb.runs[0].logged[0][1]

    # Verify CheckpointManager persists `wandb_run_id`
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ckpt_mgr = CheckpointManager(ckpt_dir=ckpt_dir)
    payload = CheckpointManager.build_payload(
        step=2,
        stage_id=2,
        model=model,
        optimizer=opt,
        scheduler_state={},
        dynamic_loss_multipliers={},
        dataloader_cursor={"batch_idx": 2},
        best_val_metric=4.0,
        config_dict=cfg.to_dict(),
        hardware_tier="cpu",
        wandb_run_id=rid1,
    )
    saved_pt = ckpt_mgr.save_healthy(payload)
    tracker1.finish()

    loaded = ckpt_mgr.load_checkpoint(saved_pt)
    assert loaded.get("wandb_run_id") == rid1

    # Simulate fresh Colab VM (empty local NVMe logs, resumed from Drive checkpoint / Drive logs)
    fresh_nvme_logs = tmp_path / "fresh_nvme_logs"
    tracker2 = WandbTracker(
        enabled=True,
        is_primary=True,
        project="dcpa-turbo",
        mode="offline",
        logs_dir=fresh_nvme_logs,
        drive_logs_dir=drive_logs,
        wandb_module=fake_wb,
    )
    rid2 = tracker2.start(resumed_run_id=loaded.get("wandb_run_id"), resume_step=2)
    assert rid2 == rid1
    assert fake_wb.runs[1].kwargs["resume"] == "allow"
    assert fake_wb.runs[1].kwargs["id"] == rid1
    tracker2.finish()


def test_colab_git_sync_clone_and_fast_forward_pull(tmp_path: Path) -> None:
    bare_origin = tmp_path / "origin.git"
    work_seed = tmp_path / "seed_repo"
    colab_checkout = tmp_path / "content_dcpa"

    subprocess.run(["git", "init", "--bare", str(bare_origin)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(work_seed)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(work_seed), "config", "user.email", "test@dcpa.local"], check=True)
    subprocess.run(["git", "-C", str(work_seed), "config", "user.name", "DCPA Test"], check=True)
    subprocess.run(["git", "-C", str(work_seed), "checkout", "-b", "feat/dcpa-turbo-parallel"], check=True, capture_output=True)

    (work_seed / "version.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(work_seed), "add", "version.txt"], check=True)
    subprocess.run(["git", "-C", str(work_seed), "commit", "-m", "init v1"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(work_seed), "push", str(bare_origin), "feat/dcpa-turbo-parallel"],
        check=True,
        capture_output=True,
    )

    res1 = sync_repo_via_git(
        colab_checkout,
        repo_url=str(bare_origin),
        branch="feat/dcpa-turbo-parallel",
    )
    assert res1["mode"] == "git_clone"
    assert res1["branch"] == "feat/dcpa-turbo-parallel"
    assert (colab_checkout / "version.txt").read_text(encoding="utf-8").strip() == "v1"

    # Push upstream commit v2 and verify `git pull --ff-only` updates cleanly
    (work_seed / "version.txt").write_text("v2-turbo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(work_seed), "add", "version.txt"], check=True)
    subprocess.run(["git", "-C", str(work_seed), "commit", "-m", "upgrade v2"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(work_seed), "push", str(bare_origin), "feat/dcpa-turbo-parallel"],
        check=True,
        capture_output=True,
    )

    res2 = sync_repo_via_git(
        colab_checkout,
        repo_url=str(bare_origin),
        branch="feat/dcpa-turbo-parallel",
    )
    assert res2["mode"] == "git_pull"
    assert res2["commit"] != res1["commit"]
    assert (colab_checkout / "version.txt").read_text(encoding="utf-8").strip() == "v2-turbo"

    # Verify offline Drive zip fallback when repo_url is unreachable
    fb_zip = tmp_path / "fallback.zip"
    with zipfile.ZipFile(fb_zip, "w") as zf:
        zf.writestr("version.txt", "from-zip\n")
    fb_dst = tmp_path / "offline_dcpa"
    res3 = sync_repo_via_git(
        fb_dst,
        repo_url=str(tmp_path / "nonexistent_repo.git"),
        branch="main",
        fallback_bundle_zip=fb_zip,
    )
    assert res3["mode"] == "bundle_zip"
    assert (fb_dst / "version.txt").read_text(encoding="utf-8").strip() == "from-zip"


def test_mode1_streaming_matches_mode2_and_mode3_with_active_pondering() -> None:
    """Verify Mode I (streaming carry) == Mode II (eval) == Mode III (parallel) when Pondering fires."""
    torch.manual_seed(2027)
    cfg = DCPAConfig.micro()
    cfg.tau_skip = 0.0  # Force System-2 pondering MoE to execute on EVERY concept segment!
    cfg.tau_halt = 0.0
    model = DCPAModel(cfg).eval()

    raw = torch.tensor(
        [list(b"fn solve(x: i32) { let y = (x + 3) * [1, 2][0]; return y; }")],
        dtype=torch.uint8,
    )
    lang_id = torch.tensor([4], dtype=torch.long)

    with torch.no_grad():
        carry0 = model.init_carry_state(batch_size=1, device=raw.device, dtype=torch.float32)
        # Mode I: streaming carry path (`graph_state is not None`)
        out_m1 = model(
            raw,
            lang_id,
            stage="D",
            tau_s=0.35,
            tau_m=0.25,
            use_broca_cq=False,
            fast_state=carry0.fast_state,
            graph_state=carry0.graph_state,
            elig_state=carry0.elig_state,
            cortical_state=carry0.cortical_state,
            h_init=carry0.sgsc_h,
            m_init=carry0.pmd_m,
            xi_init=carry0.pmd_xi,
            write_threshold_override=0.0,
        )
        # Mode II: isolated varlen eval path (`doc_cu_seqlens is not None and V <= 8`)
        doc_cu = torch.tensor([0, raw.shape[1]], dtype=torch.int32)
        out_m2 = model(
            raw,
            lang_id,
            stage="D",
            tau_s=0.35,
            tau_m=0.25,
            use_broca_cq=False,
            doc_cu_seqlens=doc_cu,
            write_threshold_override=0.0,
        )
        # Mode III: full-batch parallel path (`doc_cu_seqlens is None and graph_state is None`)
        out_m3 = model(
            raw,
            lang_id,
            stage="D",
            tau_s=0.35,
            tau_m=0.25,
            use_broca_cq=False,
            write_threshold_override=0.0,
        )

    assert float(out_m1.metrics["avg_ponder_steps"]) >= 3.9
    assert out_m1.s_trace is not None and out_m2.s_trace is not None and out_m3.s_trace is not None
    assert (out_m1.s_trace - out_m2.s_trace).abs().max().item() < 1e-5
    assert (out_m1.s_trace - out_m3.s_trace).abs().max().item() < 1e-5
    assert (out_m1.pmd.byte_logits - out_m2.pmd.byte_logits).abs().max().item() < 1e-5
    assert (out_m1.pmd.byte_logits - out_m3.pmd.byte_logits).abs().max().item() < 1e-5


def test_compiled_cortical_and_zero_sync_static_ponder_parallel(monkeypatch) -> None:
    """Verify `configure_model_compile` compiles `cortical` + `ponder_parallel` with 0 `torch.nonzero` calls."""
    from dcpa.compile_utils import configure_model_compile, unwrap_compiled_module

    torch.manual_seed(31415)
    cfg = DCPAConfig.micro()
    cfg.tau_skip = 0.0
    cfg.tau_halt = 0.0
    cfg.window_size = 16

    m_eager = DCPAModel(cfg).train()
    m_comp = DCPAModel(cfg).train()
    m_comp.load_state_dict(m_eager.state_dict())

    rep = configure_model_compile(
        m_comp, compile_mode="reduce-overhead", device=torch.device("cpu"), allow_cpu_compile=True
    )
    assert "cortical_sandwich_kernel" in rep.compiled_hotspots
    assert "bcg_ponder_parallel_kernel" in rep.compiled_hotspots

    # Forbid `torch.nonzero` during `ponder_parallel` to prove 0 GPU->CPU stream syncs!
    bcg_comp = unwrap_compiled_module(m_comp.bcg_pc)
    orig_nonzero = torch.nonzero
    nonzero_calls = [0]

    def _forbid_nonzero(*args, **kwargs):
        nonzero_calls[0] += 1
        return orig_nonzero(*args, **kwargs)

    for m_len in (19, 37, 53):
        z0 = torch.randn(m_len, cfg.d_concept, requires_grad=True)
        z0_c = z0.detach().clone().requires_grad_(True)
        eps0 = torch.randn(m_len, cfg.d_concept)
        surp0 = torch.ones(m_len) * 1.5
        sbar = torch.randn(m_len, cfg.k_slots, cfg.d_concept)
        cf = torch.randn(m_len, cfg.d_concept)
        pie = torch.ones(m_len, cfg.d_concept)

        z_e, ps_e, bal_e = m_eager.bcg_pc.ponder_parallel(z0, eps0, surp0, sbar, cf, pie)
        monkeypatch.setattr(torch, "nonzero", _forbid_nonzero)
        try:
            z_c, ps_c, bal_c = bcg_comp.ponder_parallel(z0_c, eps0, surp0, sbar, cf, pie)
        finally:
            monkeypatch.setattr(torch, "nonzero", orig_nonzero)

        assert nonzero_calls[0] == 0
        assert (z_e - z_c).abs().max().item() < 1e-5
        assert (ps_e - ps_c).abs().max().item() < 1e-5
        assert abs(float(bal_e.item() - bal_c.item())) < 1e-5
        (z_e.sum() + bal_e).backward()
        (z_c.sum() + bal_c).backward()
        assert z0.grad is not None and z0_c.grad is not None
        assert (z0.grad - z0_c.grad).abs().max().item() < 1e-5

    # Also verify compiled `CorticalSandwich` forward+backward across unaligned M=23 (pads to tile bucket)
    c_slow = torch.randn(23, cfg.d_concept, requires_grad=True)
    c_slow_c = c_slow.detach().clone().requires_grad_(True)
    f_stem = torch.randn(23, cfg.d_s)
    doc_ids = torch.tensor([0] * 11 + [1] * 12, dtype=torch.long)
    pos = torch.tensor(list(range(11)) + list(range(12)), dtype=torch.long)
    dep = torch.zeros(23, dtype=torch.int32)

    he_e, ckv_e, be_e = m_eager.cortical.encode_parallel(c_slow, f_stem, doc_ids, pos, dep)
    he_c, ckv_c, be_c = m_comp.cortical.encode_parallel(c_slow_c, f_stem, doc_ids, pos, dep)
    assert (he_e - he_c).abs().max().item() < 1e-5
    assert (ckv_e - ckv_c).abs().max().item() < 1e-5
    assert abs(float(be_e.item() - be_c.item())) < 1e-5
    (he_e.sum() + be_e).backward()
    (he_c.sum() + be_c).backward()
    assert c_slow.grad is not None and c_slow_c.grad is not None
    assert (c_slow.grad - c_slow_c.grad).abs().max().item() < 1e-5
