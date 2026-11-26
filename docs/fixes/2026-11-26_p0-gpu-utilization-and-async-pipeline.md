# 2026-11-26 · P0 CPU/GPU 利用率修复与异步流水线改造简报

- **日期**：2026-11-26
- **分支**：`feat/dcpa-turbo-parallel`
- **原则**：**零架构改动、零数学语义变更**（前向与反向数值对齐 `< 1e-5` 或位级一致），全部修复均不改变模型参数量与权重 key。

---

## 一、修复清单（做了什么）

### 1. 消除前向热路径 GPU→CPU 同步与每步常量分配（P0）
- **`repeat_interleave` 补齐 `output_size`**：消除 3 处无 `output_size` 导致的隐式同步：
  - `dcpa-train-py/dcpa/ops/ragged.py` (`ragged_local_offsets`)
  - `dcpa-train-py/dcpa/modules/pmd.py` (`doc_id_flat`)
  - `dcpa-train-py/dcpa/model.py` (`seg_doc_ids`)
- **无同步 `seg_doc_start`**：`dcpa-train-py/dcpa/model.py` 移除 `valid_docs.any()` + boolean 索引，改为纯 GPU 向量比较 `seg_doc_ids[1:] != seg_doc_ids[:-1]`。
- **CUDA+Triton 跳过无效 `.max().item()`**：在 `model.py`（Stage A）、`sgsc.py`（`row_layout` / CUDA+Triton）、`pmd.py`（CUDA+Triton）中跳过 Triton 忽略的 `max_span` 同步读取。
- **Mode III `seg_counts` D2H 三合一**：`dcpa-train-py/dcpa/model.py` 将 Mode III 内最多 3 次 `seg_counts`/`seg_offsets` `.to("cpu").tolist()` 合并为入口处 **1 次 `[V]` int32 D2H**，在 CPU 派生 `so_orig_cpu`、`carry_q_pad`、prefix-shrink 排序 `doc_order/inv_doc_order` 与 Stage 3 Broca CQ 偏移；之后整个 Mode III（Cortical Enc → BCG → Ponder → Dec → CQ → PMD → Loss）零同步连续发射。
- **缓存 EMA 权重**：`DCPAModel._get_c0_ema_weights_by_rem` 缓存 `c0_ema_weights_by_rem`，消除每 micro-step 16 次小 GPU 张量分配。

### 2. PMD Stream-Carry 双次 Stack Scan 合一（P0）
- **`dcpa-train-py/dcpa/ops/scan.py`**：为 `parallel_stack_scan`、`dcpa::tridiag_stack_scan_fwd/bwd`、纯 PyTorch 参考实现及 4 个 Triton stack kernel 增加 `reset_to_zero: bool = False`（默认 `False` 完全向后兼容）。
  - `reset_to_zero=True` 时：`t=0`（且 `~rm_0`）从 `xi_init` 启动，中段 `reset_mask` 复位至全零空栈，反向 `d_xi0` 仅在 `(t==0) & (~rm_0)` 累积。
- **`dcpa-train-py/dcpa/modules/pmd.py`**：`_motor_scan_rows_impl` 由原先 2 次 `parallel_stack_scan`（`xi_main + before_first_reset * xi_hom`）合并为 **1 次** `parallel_stack_scan(..., reset_to_zero=True)`，算力与激活显存减半。

### 3. Triton Diag Scan 双策略升级（降 HBM 读写 & 去 Python Pass 2）
- **`dcpa-train-py/dcpa/ops/scan.py`**：
  - **Regime 1（单 Pass 寄存器跨块传递）**：新增 `_triton_diag_single_pass_fwd_kernel` / `_bwd_kernel`（Grid = `(B, grid_d)`，块间状态 `h_carry`/`g_carry` 留寄存器），HBM 往返从 3 读 3 写降至 **1 读 1 写**（覆盖生产 `stream_carry` 场景）。
  - **Regime 2（单 CTA Triton Pass 2）**：新增 `_triton_diag_interchunk_scan_kernel`，直接输出 exclusive `carry`，用 1 次 Triton launch 替换原先 Python 层 `log2(N_chunks)` 次 `_kogge_stone_diag` + `torch.cat`。

### 4. Cortical Encoder 引入 FlexAttention（合并 `ring` LSE 路径 & 去 2W KV 复制）
- **`dcpa-train-py/dcpa/modules/cortical.py`**：
  - 在 `CorticalSandwich.encode_parallel` 统一构造 1 次共享 `BlockMask`（因果滑窗 + 同文档隔离 + padding 对角防 `0/0`），供全部 `enc_layers` 复用。
  - 自动处理非 2 次幂 `d_head`（如 `moe_210m` 的 `d_head=96` zero-pad 至 `128`，显式保留原始 `scale = 1/sqrt(Dh)`），消除 PyTorch 2.5 `_validate_embed_dim` 限制与 `M` 非 128 倍数越界风险。
  - `ring is not None` 走 `flex_attention(..., return_lse=True)` 后接 `_lse_merge_weights`（已在 PyTorch 2.5.1 与 2.7.1 验证 `grad_logsumexp` 反向支持及零重编译）。

### 5. Stage 3 字节损失权重 256-entry GPU LUT
- **`train/curriculum_state_machine.py`**：新增按 `(device, focal_weight)` 缓存的 `(code_lut[256], noncode_lut[256])`，替换 `build_byte_loss_weights` Stage 3 的 10 次 Python 循环与多轮 `torch.where` 为单次表查找。

### 6. `StreamCarryPacker` 异步预取 + Page-Locked Pin + CUDA `h2d_stream`（P0）
- **`train/train_dcpa.py`**：
  - `_DynamicRaggedCursorStream.next_batch` 新增 `pin_output: bool = True`，供 `StreamCarryPacker` 拉取原始 batch 时传 `pin_output=False`，避免无效 `cudaHostAlloc`。
  - 新增单线程顺序预取器 `_AsyncStreamCarryPrefetcher`：在后台线程完成 `StreamCarryPacker.next_batch` + 最终打包张量 `pin_batch_if_cuda`，同时按批次精确快照 `stream.cursor`（checkpoint / stage 切换 / Watchdog 回滚 100% 确定性）。
  - 训练循环引入专用 CUDA `h2d_stream`，在 `grad_accum_steps > 1` 时与前向/反向重叠下一 micro-step 的异步 H2D（`wait_event` + `record_stream`）。

### 7. Checkpoint 单次 `torch.save` 去重 + 异步落盘与 Drive 同步（P0）
- **`train/watchdog_and_ckpt.py`**：
  - 新增 `atomic_copy_or_link`（同文件系统 `os.link` + `os.replace` 零拷贝，跨盘回退 `fsync` 原子拷贝）与 `AsyncCheckpointWriter`。
  - `AsyncCheckpointWriter` 在 CUDA 上通过专用 `ckpt_stream` 将权重/优化器状态异步拷入可复用 pinned CPU 镜像（受 host RAM 上限保护），下步 `optimizer.step()` 前做零 CPU 同步的 `compute_stream.wait_event(d2h_done_event)`；后台单线程仅执行 **1 次** `atomic_torch_save`，其余目标通过 `atomic_copy_or_link` 与 Drive `staged_path` 复用。
- **`train/colab_drive_sync.py`**：`ABDriveSyncManager.save_ping_pong` 新增可选 `staged_path`，复用已落盘的本地 NVMe `.pt` 直接计算 BLAKE2b 并同步，不再二次 `torch.save`。
- **`train/train_dcpa.py`**：常规保存移除跨卡 `dist_barrier`（仅在 `any_interrupted` / `is_final_step` / Rollback / Resume 同步），并在 Rollback、紧急中断、最终导出及 `finally` 前强制 `async_ckpt.wait()` 确保落盘完整。

---

## 二、新增 / 调整的参数与阈值速查（勿随意改动）

| 位置 | 参数 / 常量 | 默认值 / 条件 | 说明 |
| --- | --- | --- | --- |
| `dcpa/ops/scan.py` | `parallel_stack_scan(..., reset_to_zero)` | `False`（PMD rows 传 `True`） | 控制中段 `reset_mask` 复位至 0 还是 `xi_init` |
| `dcpa/ops/scan.py` | Diag Single-Pass 阈值 | `(B * grid_d) >= 64 and n_chunks <= 64` | 覆盖 `stream_carry` 高 CTA 场景，防超长单流串行退化 |
| `dcpa/ops/scan.py` | Diag Triton Pass 2 阈值 | `1 < n_chunks <= 1024` | 覆盖单流 `T <= 262,144`；超出回退 PyTorch Kogge-Stone |
| `dcpa/modules/cortical.py` | `CorticalSandwich.use_flex_attention` | `True`（仅 CUDA SM>=80 + BF16/FP16 触发） | CPU/MPS/FP32 自动走原 tiled 路径；测试可用 `_force_flex_attention` |
| `dcpa/modules/cortical.py` | `_FLEX_BLOCK_SIZE` | `128` | `doc_ids` 对齐 128 倍数（兼容 PyTorch 2.5 `create_block_mask`） |
| `train/train_dcpa.py` | `eff_carry_prefetch` | `max(2, eff_prefetch_depth)`（仅 CUDA） | `_AsyncStreamCarryPrefetcher` 队列深度（CPU/MPS 为 0） |
| `train/train_dcpa.py` | YAML `checkpointing.async_checkpoint` | `True` | 开启后台异步序列化与 Drive 上传 |
| `train/train_dcpa.py` | YAML `checkpointing.async_pinned_mirror_gb` | `8.0` (GB) | 可复用 pinned CPU 快照镜像上限，超限自动回退非 pinned CPU |
| `train/colab_drive_sync.py` | `save_ping_pong(..., staged_path)` | `None` | 传入本地已写好的 `.pt` 路径以跳过重复 `torch.save` |

---

## 三、验证摘要

- **新增单测**：`tests/test_p0_perf_optimizations.py`（**10/10 PASSED**），覆盖 `reset_to_zero` 栈扫描前反向、`PMD._motor_scan_rows_impl` 单扫描等价、Cortical FlexAttention（含 `d_head=96` 与 `ring` LSE 合并）、Stage 3 256-LUT 位级等价、`seg_doc_start` 边界、`_AsyncStreamCarryPrefetcher` 确定性与回滚、以及 `AsyncCheckpointWriter` 单次 `torch.save` 与并发写入隔离。
- **回归测试**：
  - `tests/test_stream_carry.py`：**8/8 PASSED**（1 CUDA-only skipped）
  - `tests/test_trainer_lifecycle.py`：**8/8 PASSED**（含 4 阶段冻结、Watchdog 回滚、中断恢复确定性、A/B Ping-Pong 自愈）
  - `dcpa-train-py/tests/test_dcpa.py`：**8/8 PASSED**
  - `tests/test_dcpa_turbo_and_ops.py`：**8/8 PASSED**（PyTorch 2.7.1 环境）
  - `tests/test_sanity.py`、`test_data_pipeline.py`、`test_capabilities_upgrade.py`、`test_stage5_and_repl.py`、`test_v2_cognitive_pipeline.py`：**全通过**
  - 注：两处既有基线环境失败与本次改动无关（`test_step2` 第 1 段因本地 PyTorch 2.5.1 缺 `F.grouped_mm` 触发未改动的 `bcg_pc.py`；`test_v2_presets` 断言 `moe_210m` 参数上限 240M，而仓库基线配置本身为 337.7M）。`test_step2` 第 2/3/4 段（tridiag stack / chunked diag / TBPTT）已单独复测全过。
