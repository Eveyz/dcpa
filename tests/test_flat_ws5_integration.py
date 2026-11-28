"""WS-5 End-to-End Integration Tests for DCPA-Flat v3.3 (`DCPAModelFlat`, Plugin Trainer, & 5k Killgate)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.interfaces import DCPAFlatConfig, LANG_DOMAINS_V3
from dcpa.model_flat import DCPAModelFlat
from dcpa.train_plugin import (
    load_plugin_pack_safetensors,
    save_plugin_pack_safetensors,
    train_additive_plugin_pack,
)
from gates.run_flat_killgate_5k import _make_synthetic_grid_batch, run_killgate


def test_parameter_budgets_48m_and_206m() -> None:
    """Verify parameter counts align with DCPA-Flat-48M and DCPA-Flat-206M hardware budgets."""
    m48 = DCPAModelFlat(DCPAFlatConfig.flat_48m())
    c48 = m48.count_parameters()
    # ~50M-62M trainable base (including 3-layer W=128 projections + +32KB semantic ring)
    assert 45_000_000 <= c48["trainable_total"] <= 68_000_000, f"Unexpected 48M budget: {c48}"
    assert c48["active_avg_per_seg"] < c48["trainable_total"] * 0.45

    m206 = DCPAModelFlat(DCPAFlatConfig.flat_206m())
    c206 = m206.count_parameters()
    assert 185_000_000 <= c206["trainable_total"] <= 235_000_000, f"Unexpected 206M budget: {c206}"
    assert c206["active_avg_per_seg"] < c206["trainable_total"] * 0.35


@pytest.mark.parametrize("use_win", [True, False])
@pytest.mark.parametrize("share_stem", [False, True])
def test_model_flat_all_stages_forward_backward(use_win: bool, share_stem: bool) -> None:
    """Test sensory, cortex, and full Hippocampus stages across R1 ablation toggles."""
    torch.manual_seed(42)
    cfg = DCPAFlatConfig.flat_micro()
    cfg.use_local_window_layers = use_win
    cfg.share_stem_ring = share_stem
    model = DCPAModelFlat(cfg)
    batch = _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["python"], seed=7, device=torch.device("cpu"))

    for stage in ("sensory", "cortex", "hippocampus"):
        model.set_stage_mode(stage)
        model.train()
        opt = torch.optim.AdamW(model.parameter_groups())
        opt.zero_grad(set_to_none=True)
        out = model(batch, stage=stage)
        assert torch.isfinite(out.loss_total).item(), f"Non-finite loss in stage={stage}"
        out.loss_total.backward()
        grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad)
        assert grad_ok, f"No valid gradients in stage={stage}"
        opt.step()
        model.post_optimizer_step(out)
        m = out.materialize_metrics()
        assert m["loss_lang"] > 0.0
        # Untrained step-0 SGSC has low compression ratio; S+1 trash slot safely absorbs any overflow
        assert 0.0 <= m["overflow_byte_ratio"] < 0.5
        if bool(out.bridge.byte_overflow.any().item()):
            assert float(out.pmd.eff_weight[out.bridge.byte_overflow].abs().max().item()) == 0.0


def test_unified_plugin_trainer_zero_drift_and_safetensors_roundtrip(tmp_path: Path) -> None:
    """Gate E contract: frozen-base plugin fine-tuning drops target BPC with strict 0.0 non-target drift."""
    torch.manual_seed(123)
    cfg = DCPAFlatConfig.flat_micro()
    model = DCPAModelFlat(cfg)
    dev = torch.device("cpu")

    sql_batches = [
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=11, device=dev),
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=12, device=dev),
    ]
    anchor_batches = [
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["en"], seed=21, device=dev),
        _make_synthetic_grid_batch(cfg, lang_id=LANG_DOMAINS_V3["zh"], seed=22, device=dev),
    ]

    pack, report = train_additive_plugin_pack(
        model,
        pack_id="sql_v3_test",
        target_batches=sql_batches,
        anchor_batches=anchor_batches,
        steps=10,
        experts_per_layer=1,
        lr=5e-3,
    )
    assert report.passed_zero_drift, f"Expected strict zero drift, got {report}"
    assert report.anchor_max_logit_drift == 0.0, f"Non-target anchor drift != 0.0: {report.anchor_max_logit_drift}"
    assert report.anchor_plugin_active_rate == 0.0

    # Persist & reload `.dcpa_expert.safetensors` and verify bit-exact match
    out_file = save_plugin_pack_safetensors(pack, tmp_path / "sql_v3_test.dcpa_expert.safetensors")
    reloaded = load_plugin_pack_safetensors(out_file)
    reloaded.validate(cfg.n_layers, cfg.d_concept, tau_min=cfg.plugin_tau_min)
    assert torch.equal(reloaded.proto_keys, pack.proto_keys)
    assert torch.equal(reloaded.tau, pack.tau)
    assert torch.equal(reloaded.w_d, pack.w_d)


def test_killgate_5k_smoke_runner(tmp_path: Path) -> None:
    """Run full 3-way homologous fork killgate script in fast `--smoke` mode."""
    out_json = tmp_path / "killgate_smoke.json"
    res = run_killgate(smoke=True, out_json=str(out_json))
    assert out_json.is_file()
    assert res["gate_b_stress"]["passed"] is True
    assert res["gate_e_plugin"]["passed_zero_drift"] is True
    assert res["gate_e_plugin"]["anchor_max_logit_drift"] == 0.0
    for k in ("run_a", "run_b", "run_c"):
        assert res["runs"][k]["val_bpc_en_zh"] > 0.0
        assert res["runs"][k]["val_bpc_code"] > 0.0


def test_model_flat_hotpath_ast_red_lines() -> None:
    """Verify `model_flat.py` forward hot path obeys zero-sync & no-bare-cuda red lines."""
    src = (_TRAIN_PY / "dcpa" / "model_flat.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute):
                    assert sub.attr not in {"nonzero", "repeat_interleave", "item", "tolist", "cpu"}, (
                        f"Forbidden op `.{sub.attr}` in DCPAModelFlat.forward"
                    )
    assert "torch.cuda" not in src, "Iron Law #1 violation: bare torch.cuda in model_flat.py"
