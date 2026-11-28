"""Contract and invariant tests for DCPA-Flat v3.3 frozen interfaces."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.interfaces import (
    ALLOWED_PROTOCOL_CTRL_BYTES,
    IDX_RECORD_FMT,
    IDX_RECORD_SIZE,
    LANG_DOMAINS_V3,
    LANG_ID_TO_NAME_V3,
    STYLE_DOMAINS_V3,
    STYLE_ID_TO_NAME_V3,
    CarrierGraphState,
    DCPAFlatConfig,
    PluginExpertPack,
    ScatterBridgeOut,
    StaticGridBatch,
    decode_fixed_point_loss_mask,
)


def test_protocol_constants_and_loss_mask_decoding() -> None:
    assert struct.calcsize(IDX_RECORD_FMT) == IDX_RECORD_SIZE == 16
    assert LANG_DOMAINS_V3 == {
        "en": 0,
        "zh": 1,
        "formal": 2,
        "python": 3,
        "sql_struct": 4,
        "javascript": 5,
    }
    assert LANG_ID_TO_NAME_V3 == {0: "en", 1: "zh", 2: "formal", 3: "python", 4: "sql_struct", 5: "javascript"}
    assert STYLE_DOMAINS_V3 == {
        "rigorous": 0,
        "concise_memo": 1,
        "expository": 2,
        "literary": 3,
        "strict_eng": 4,
    }
    assert len(STYLE_ID_TO_NAME_V3) == 5
    assert ALLOWED_PROTOCOL_CTRL_BYTES == frozenset({0x04, 0x11, 0x12, 0x13, 0x14, 0x1D, 0x1E, 0x1F})

    raw = torch.tensor([0, 1, 10, 15, 30], dtype=torch.uint8)
    dec = decode_fixed_point_loss_mask(raw)
    assert dec.dtype == torch.float32
    assert torch.allclose(dec, torch.tensor([0.0, 1.0, 1.0, 1.5, 3.0]))


def test_flat_config_presets_and_json_roundtrip(tmp_path: Path) -> None:
    for cfg in (DCPAFlatConfig.flat_micro(), DCPAFlatConfig.flat_48m(), DCPAFlatConfig.flat_206m()):
        assert cfg.vocab_size == 256
        assert cfg.d_concept % cfg.n_heads == 0
        assert cfg.grid_segs % cfg.window_size == 0
        assert cfg.grid_segs % cfg.c_scan == 0
        assert cfg.num_merge_pairs_full == cfg.k_slots * (cfg.k_slots - 1) == 56
        assert cfg.full_gate_temp == pytest.approx(0.05)
        assert cfg.occ_relax == pytest.approx(0.995)
        assert cfg.local_window_layers == (0, 3, 5)
        p = tmp_path / "flat_cfg.json"
        cfg.save_json(p)
        reloaded = DCPAFlatConfig.load_json(p)
        assert reloaded == cfg


def test_frozen_contract_validators() -> None:
    cfg = DCPAFlatConfig.flat_micro()
    P, T, S, d_s, D, K, d_r = cfg.grid_lanes, cfg.grid_bytes, cfg.grid_segs, cfg.d_s, cfg.d_concept, cfg.k_slots, cfg.d_r

    batch = StaticGridBatch(
        x_bytes=torch.zeros((P, T), dtype=torch.uint8),
        doc_id_byte=torch.zeros((P, T), dtype=torch.int32),
        doc_start_byte=torch.zeros((P, T), dtype=torch.bool),
        doc_end_byte=torch.zeros((P, T), dtype=torch.bool),
        loss_weight=torch.ones((P, T), dtype=torch.float32),
        doc_meta=torch.zeros((P * cfg.doc_meta_cap_per_lane, 3), dtype=torch.int32),
    )
    batch.validate()

    bridge = ScatterBridgeOut(
        f_stem=torch.zeros((P, S, d_s)),
        x0=torch.zeros((P, S, D)),
        seg_valid=torch.ones((P, S), dtype=torch.bool),
        seg_doc_id=torch.zeros((P, S), dtype=torch.int32),
        seg_reset=torch.zeros((P, S), dtype=torch.bool),
        seg_pin_trig=torch.zeros((P, S), dtype=torch.bool),
        seg_unpin_trig=torch.zeros((P, S), dtype=torch.bool),
        seg_start_depth=torch.zeros((P, S), dtype=torch.int32),
        seg_end_depth=torch.zeros((P, S), dtype=torch.int32),
        seg_l_reg=torch.zeros((P, S, 64)),
        byte_to_seg=torch.zeros((P, T), dtype=torch.int32),
        intra_seg_pos=torch.ones((P, T), dtype=torch.int32),
        byte_overflow=torch.zeros((P, T), dtype=torch.bool),
        seg_counts=torch.full((P,), S, dtype=torch.int32),
        prior_logits=torch.zeros((P, T, 256)),
        spikes=torch.zeros((P, T), dtype=torch.bool),
        s_tilde=torch.zeros((P, T)),
        mean_info=torch.zeros((P, S)),
        sem_norm=torch.zeros((P, S)),
    )
    bridge.validate(P, T, S, d_s, D)

    g_state = CarrierGraphState(
        slots=torch.zeros((P, K, D)),
        edges=torch.zeros((P, K, K, d_r)),
        occ=torch.full((P, K), 0.05, dtype=torch.float32),
        slot0_pinned=torch.zeros((P,), dtype=torch.bool),
    )
    g_state.validate(P, K, D, d_r)

    pack = PluginExpertPack(
        pack_id="sql_test",
        proto_keys=torch.nn.functional.normalize(torch.randn((cfg.n_layers, 2, D)), dim=-1),
        tau=torch.full((cfg.n_layers, 2), cfg.plugin_tau_min),
        alpha=torch.ones((cfg.n_layers, 2)),
        w_gu=torch.zeros((cfg.n_layers, 2, 2 * 64, D)),
        w_d=torch.zeros((cfg.n_layers, 2, D, 64)),
    )
    pack.validate(cfg.n_layers, D, tau_min=cfg.plugin_tau_min)
    with pytest.raises(ValueError):
        PluginExpertPack(
            pack_id="bad_tau",
            proto_keys=pack.proto_keys,
            tau=torch.full((cfg.n_layers, 2), 0.10),
            alpha=pack.alpha,
            w_gu=pack.w_gu,
            w_d=pack.w_d,
        ).validate(cfg.n_layers, D, tau_min=cfg.plugin_tau_min)
