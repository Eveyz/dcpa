"""Gate 1: SGSC Causality, Parallel Associative Scan Equivalence & UTF-8 Integrity.

Pass/Fail Quantitative Thresholds (Engineering Blueprint Chapter 7):
  (1) Randomly tamper input bytes at step t and after: max |p_{1:t}' - p_{1:t}| == 0.0 strictly!
  (2) Parallel associative scan vs sequential reference max abs error < 1e-5 (forward & backward).
  (3) Mixed Chinese + English + Emoji + Code UTF-8 byte stream: 100% of split segments are
      valid, complete UTF-8 strings (zero split multi-byte characters).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dcpa.config import DCPAConfig
from dcpa.modules.sgsc import SGSC
from dcpa.ops.scan import HAS_TRITON, parallel_diag_scan, sequential_diag_scan_ref
from utils.device import get_device, synchronize


def verify_gate1() -> None:
    print("=== Running Gate 1: SGSC Causality, Parallel Scan & UTF-8 Boundary Integrity ===")
    torch.manual_seed(2026)

    cfg = DCPAConfig.tiny()
    sgsc = SGSC(cfg)
    sgsc.eval()

    # -------------------------------------------------------------------------
    # Check (1): Strict zero-leakage causality on prior_probs p_{1:t}
    # -------------------------------------------------------------------------
    B, T = 4, 256
    x_orig = torch.randint(0, 256, (B, T), dtype=torch.uint8)
    with torch.no_grad():
        out_orig = sgsc(x_orig)

    for tamper_t in [0, 1, 17, 64, 128, 255]:
        x_tampered = x_orig.clone()
        x_tampered[:, tamper_t:] = ((x_tampered[:, tamper_t:].int() + 73) % 256).to(torch.uint8)
        with torch.no_grad():
            out_tamp = sgsc(x_tampered)
        # Note: prior_probs[:, 0:tamper_t+1] predicts bytes 0..tamper_t from states h_0..h_{tamper_t-1}
        # which depend ONLY on bytes 0..tamper_t-1! So even at tamper_t itself, p_{tamper_t} MUST be unchanged!
        diff = (out_orig.prior_probs[:, : tamper_t + 1, :] - out_tamp.prior_probs[:, : tamper_t + 1, :]).abs().max().item()
        assert diff == 0.0, f"GATE 1 FAIL: causality leak at t={tamper_t}, max diff={diff}"
    print("[PASS] Check 1/3: Strict causality verified (max |p_{1:t}' - p_{1:t}| == 0.0 for all tamper_t).")

    # -------------------------------------------------------------------------
    # Check (2): Parallel associative scan vs sequential reference (< 1e-5)
    # -------------------------------------------------------------------------
    for seq_len in [1, 7, 64, 257, 1024]:
        a = torch.sigmoid(torch.randn(3, seq_len, cfg.d_s, dtype=torch.float32))
        b = torch.randn(3, seq_len, cfg.d_s, dtype=torch.float32) * 0.2
        h0 = torch.randn(cfg.d_s, dtype=torch.float32) * 0.1

        a1 = a.clone().requires_grad_(True)
        b1 = b.clone().requires_grad_(True)
        h0_1 = h0.clone().requires_grad_(True)

        a2 = a.clone().requires_grad_(True)
        b2 = b.clone().requires_grad_(True)
        h0_2 = h0.clone().requires_grad_(True)

        h_par = parallel_diag_scan(a1, b1, h0=h0_1)
        h_seq = sequential_diag_scan_ref(a2, b2, h0=h0_2)

        fwd_err = (h_par - h_seq).abs().max().item()
        assert fwd_err < 1e-5, f"GATE 1 FAIL: forward scan diff={fwd_err} at T={seq_len}"

        g_out = torch.randn_like(h_par)
        (h_par * g_out).sum().backward()
        (h_seq * g_out).sum().backward()

        ga_err = (a1.grad - a2.grad).abs().max().item()
        gb_err = (b1.grad - b2.grad).abs().max().item()
        gh0_err = (h0_1.grad - h0_2.grad).abs().max().item()
        assert ga_err < 1e-5 and gb_err < 1e-5 and gh0_err < 1e-5, (
            f"GATE 1 FAIL: backward grad diff ga={ga_err}, gb={gb_err}, gh0={gh0_err}"
        )

    # Speed report (on CUDA + Triton if available, else MPS/CPU)
    dev = get_device("auto", deterministic=False)
    a_bench = torch.sigmoid(torch.randn(32, 4096, 192, device=dev, dtype=torch.float32), ).requires_grad_(True)
    b_bench = torch.randn(32, 4096, 192, device=dev, dtype=torch.float32, requires_grad=True)
    h0_bench = torch.zeros(192, device=dev, dtype=torch.float32, requires_grad=True)
    # Warmup
    out_w = parallel_diag_scan(a_bench, b_bench, h0_bench)
    out_w.sum().backward()
    synchronize(dev)
    t0 = time.perf_counter()
    a_bench.grad = None
    b_bench.grad = None
    h0_bench.grad = None
    out_b = parallel_diag_scan(a_bench, b_bench, h0_bench)
    out_b.sum().backward()
    synchronize(dev)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[PASS] Check 2/3: Parallel associative scan fwd+bwd matches sequential reference (err < 1e-5). "
        f"B=32, T=4096 on {dev.type} (triton={HAS_TRITON}): {elapsed_ms:.2f} ms."
    )
    if dev.type == "cuda" and HAS_TRITON:
        assert elapsed_ms < 4.0, f"GATE 1 FAIL: CUDA Triton scan took {elapsed_ms:.2f} ms >= 4.0 ms"

    # -------------------------------------------------------------------------
    # Check (3): 100% valid UTF-8 segment splitting on mixed ZH + EN + Emoji + Code
    # -------------------------------------------------------------------------
    mixed_texts = [
        "那只狴犴正趴在屋脊上吞云吐雾！🦀🔥🚀 A wug is a tiny desert bird. fn main() { println!(\"你好世界🌍\"); }",
        "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🥸🤩🥳",
        "中文汉字连续测试没有任何ASCII空格且故意超过十六字节限制一二三四五六七八九十零",
        "mix_😀_中_a_🚀_文_fn(x: &[u8]) -> Result<(), Error> { Ok(()) } 🎉✨",
    ]
    total_segs = 0
    for txt in mixed_texts:
        raw_bytes_py = txt.encode("utf-8")
        x = torch.tensor(list(raw_bytes_py), dtype=torch.uint8).unsqueeze(0)
        with torch.no_grad():
            out = sgsc(x)
        cu = out.cu_seqlens.tolist()
        reconstructed = []
        for s_i in range(len(cu) - 1):
            seg_bytes = raw_bytes_py[cu[s_i] : cu[s_i + 1]]
            assert len(seg_bytes) > 0, "Empty segment encountered"
            # Must decode as 100% valid UTF-8 without errors!
            decoded = seg_bytes.decode("utf-8", errors="strict")
            reconstructed.append(decoded)
            total_segs += 1
        assert "".join(reconstructed) == txt, "Segments did not reconstruct original UTF-8 text verbatim!"
    print(f"[PASS] Check 3/3: All {total_segs} segments across mixed ZH/EN/Emoji/Code decoded as 100% valid UTF-8.")
    print("=== GATE 1 PASSED ===")


if __name__ == "__main__":
    verify_gate1()
