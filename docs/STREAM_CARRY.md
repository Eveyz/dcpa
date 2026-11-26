# Stream Carry (A1) + BCG CUDA Graphs (A2)

## 为什么

Stages B/C 的 BCG-PC 图递推沿 segment 串行（离散 merge，不可 scan）。整文档喂入时每步串行深度
`m_max ≈ max_doc_bytes / ρ`（4KB 文档、CJK 可达 ~1000+ 步），而 batch 宽度只有文档数 V（~64）。
`bcg_tbptt_window=32` 早已在 32 个 segment 处截断梯度，所以跨窗口只剩前向状态需要连续。

## A1：跨 step 的有状态流（stream-TBPTT）

- 每个 micro-step：`S` 条字节流各推进 `stream_chunk_bytes`（默认 512B，UTF-8 安全切分）。
- 每条流的递推状态在 step 之间 **detach 后携带**：SGSC `h` + 上一字节 surprisal、PMD motor `m` + 下推栈 `Xi`
  + 上一字节、括号 scope depth、BCG-PC 图（slots/edges/salience/pin/phase）、cortical W-ring（最近 W 个
  segment 的 MLA latent + 位置 + 深度）、`doc_pos`、解码器因果移位用的 `z_refined[m-1]`。
- 结果：串行深度 ≈ `chunk / ρ`（与文档长度无关），BCG batch 宽度 ≈ `S`（256KB / 512B → 448 行 + 64 补位 = 512）。
- **精确性**：在 LIF spike 边界切分时，分块 + carry 与整文档一次前向的 logits 误差 ~1e-6
  （`tests/test_stream_carry.py`）。实际训练按字节切分，chunk 末尾会多一个强制 spike（与流式推理一致）。
- 实现要点：
  - 字节级扫描（SGSC/PMD/scope depth）在右填充的 `[rows, chunk]` 行布局上运行，用每行 `h0`/`xi0` 注入初值
    （因果 scan 对右填充是精确的，不改 Triton kernel）。下推栈用线性分解：
    `Xi = scan(零初值, 帧重置) + 1[t < 首个帧重置]·(M_t…M_0)·xi0`。
  - Cortical 编/解码器：块内滑窗注意力 + carried ring 注意力，用 log-sum-exp 合并（与联合 softmax 数学等价），
    pointer-copy 同样合并。
  - 文档在 chunk 中途结束时该行立即换新文档（同一 step 内补位，受 `max_entries` 静态上限约束）。
- 范围：仅 Stages B/C。Stage A 本身全并行；Stage D 海马体快权重暂不携带。resume / watchdog 回滚 / 换 stage 时
  所有流重置（从新文档开始）。

在 `feat/dcpa-turbo-parallel` 分支默认开启（配置里 `hardware.stream_carry: true`），可用 `--no-stream-carry` 关闭，或 `--stream-chunk-bytes 512` 调整分块大小。

## A2：BCG clause-chunk kernel 的 CUDA Graph

- 在 `feat/dcpa-turbo-parallel` 分支默认开启（`hardware.bcg_cudagraphs: true`，仅 CUDA 且 `compile_mode != none` 时生效），可用 `--no-bcg-cudagraphs` 关闭。
- clause-chunk kernel 额外以 `torch.compile(mode="reduce-overhead", dynamic=False)` 编译（cudagraph trees，前向+反向）。
  形状只依赖 2 的幂分桶的 batch 宽度和固定 `K_chunk`；A1 的静态 entry 上限让宽度恒定，图只录制少数几次。
- 仅在未启用 chunk 激活检查点时使用（显存超过 watermark 时自动回退普通 compiled kernel）；
  捕获出错时自动永久回退并告警。
- GPU 自检：`pytest tests/test_stream_carry.py -k cudagraph`（对比 CUDA Graph 与普通编译 kernel 的 loss/梯度，并连跑 3 个 carry step）。
