# 2026-11-27 · Stage 2 BCG-PC 梯度爆炸、L4 CUDA OOM 与 Watchdog 防护修复简报

- **日期**：2026-11-27
- **分支**：`feat/dcpa-turbo-parallel`
- **事故背景**：Colab NVIDIA L4（22.03 GiB）续跑 `moe_210m`（`configs/v2/dcpa_210m_v2_cognitive.yaml`，W&B run `e7bed440804229ee`，从 step 30,000 起跑 Stage 2）时，30,001~37,300 步持续出现 `GN = 1e4..1.9e5` 与 `PC`/`L_m` 剧烈震荡；第 37,335 步起 `rho` 骤降至 `6.87–7.39`、`GN` 飙升至 `1.8e+09`、`PC=8.17`、`val_bpc_en=3.03`；随后第 37,551 步在 `pmd.py:715` 触发 CUDA OOM（`14.29 GiB allocated + 3.92 GiB private pools + ~3.77 GiB non-PyTorch = 21.98 / 22.03 GiB`）。
- **原则**：**零架构改动、0.0 ULP 前向数值一致**（100% 兼容已有 30k/35k checkpoint 与 `dcpa-core-rs` 位级对齐，不改变模型参数量与权重 key，不引入前向热路径 GPU→CPU 同步）。

---

## 一、故障根因定位

1. **Turbo `BCG-PC` 移除 `_DampedBoundedCarry` 导致 32 步 TBPTT 雅可比指数爆炸（主因）**
   - `DCPAModel.__init__` 曾将 `carry_grad_gamma` 从 `0.5` 改为 `1.0`、`carry_grad_max_rms` 改为 `2.0`（导致 chunk 间 `_DampedBoundedCarry` 失效），且 Turbo `_forward_clause_chunk_impl`（`S=16` 步）漏掉了 chunk 内每步 `_DampedBoundedCarry`。
   - 实际上 `Lambda_m ∈ [0.92, 0.998]` 不收缩 `edges`/`salience`/`s_merged_win` 与 `slots → c_hat → eps_0 → z_0` 闭环，在 Slot-0 Pinning 时还被强制置为 `1.0`，同时 Gumbel-STE merge 梯度按 `1/tau_m` 放大（Stage 2 中 `tau_m: 1.5 → 0.25`）。
   - **复现验证**（`moe_210m` `D=768`，32 步，`tau_m=0.25`）：无阻尼时 `dS[31→0]` 放大 `5.2e+07` 倍、`GN = 3.53e+05`（Slot-0 Pinning 再放大 `3–5x`）；恢复 `(gamma=0.5, max_rms=0.25)` 后 `GN` 降至 `3.30`（降幅 >10 万倍），且前向输出 **0.0 ULP 完全一致**。
2. **Stage 2 `cap_mix_ratio` 在第 37,335 步阶跃突变（`0% → 15%`）（引爆点）**
   - `cap_warmup_start_progress: 0.6667` 对应 `[12000, 50000]` 的精确步数 `12000 + ceil(0.6667 × 37999) = 37,334`（`completed_step = 37,335`）。整批切入含未见过控制字节（`\x1dS\n..\x1dE\n`、`\x1fA\n`）和 Slot-0 Pinning 的 `cap_*` 分片，瞬间拉低 `rho`（`9.5 → 6.87`）并引爆根因 1。
3. **CUDAGraph Trees 私有池膨胀（`3.92 GiB`）+ 水位线检测失明（OOM 直接原因）**
   - 动态 `while m_step < m_max:` + `use_prefix_shrink` 循环在 `rho` 波动与长尾 `m_max` 场景下产生大量不同 `(cur_v_0, cur_v_1, ...)` 轨迹，导致 `reduce-overhead` CUDAGraph Trees 持续分支并永久占用 `3.92 GiB` 私有池。
   - `bcg_checkpoint_vram_watermark_gb` 仅检查 `torch.cuda.memory_allocated()`（始终 `13.78 GiB < 14.0 GiB`），对私有池和 CUDA context 失明，导致直到只剩 `37 MiB` 也未触发 `use_ckpt_chunk`。
4. **Watchdog 与 Drive 断点健康门控盲区**
   - Watchdog Rule A 仅校验 `math.isfinite()`，放行有限值极端梯度爆炸（`GN = 1e5..1.8e9`）；`train_dcpa.py` 中 `wall_due` 触发 Drive A/B Ping-Pong 同步时未门控 `step_healthy_ok`，存在用发散权重覆盖健康 Drive 槽的风险。

---

## 二、修复清单（做了什么）

### 1. 恢复 Turbo `BCG-PC` 逐布与跨块 `_DampedBoundedCarry`
- **`dcpa-train-py/dcpa/modules/bcg_pc.py`**：
  - `_forward_clause_chunk_impl` 与 `forward_clause_chunk` 新增 `carry_gamma: float = 0.5, carry_max_rms: float = 0.25`（默认参数向后兼容）。
  - 在 `self.training and (carry_gamma < 0.9999 or carry_max_rms < 1e6)` 且 `k > 0` 时，于 `pre_*` 记录与 `forward_graph_step_no_ponder` 之前对 `(cur_slots, cur_edges, cur_sal)` 施加 `_DampedBoundedCarry.apply`，使 `K_c = 16` 与 `K_c = 1` 在前向与反向完全等价，并完整透传至 `_cudagraph_chunk_fn` 与 `_compiled_chunk_fn`。
- **`dcpa-train-py/dcpa/model.py`**：
  - `DCPAModel.__init__` 恢复 `self.carry_grad_gamma = 0.5`、`self.carry_grad_max_rms = 0.25`。
  - `_clause_chunk_checkpointed_fn`、chunked 循环及非 chunked 循环均采用 `carry_gamma < 0.9999 or carry_max_rms < 1e6` 判断，并将 `(carry_gamma, carry_max_rms)` 透传至 `torch_checkpoint` 与 `forward_clause_chunk`。

### 2. 真实剩余显存水位线防火墙 + CUDAGraph 线性头链上限
- **`utils/device.py`**（严格遵守 Iron Law #1，全部 `torch.cuda.*` 封装在本模块内）：
  - 新增 `get_reserved_memory_gb(device)`、`get_vram_free_and_total_gb(device)`（`torch.cuda.mem_get_info`）与 `is_vram_headroom_low(device, min_free_gb=3.5, min_free_frac=0.15)`。
- **`dcpa-train-py/dcpa/model.py`**：
  - `DCPAModel.__init__` 新增 `self.bcg_vram_min_free_gb = 3.5` 与 `self.bcg_cudagraph_max_chunks = 4`。
  - `use_ckpt_chunk` 在 `watermark_gb <= 0`、`allocated >= watermark_gb` 或 `_is_vram_headroom_low(device)` 任一成立时自动开启梯度检查点，同时停用后续 chunk 的 CUDAGraph 录制。
  - 将 CUDAGraph 严格限制在 `cur_v == V_eff` 的线性前 `bcg_cudagraph_max_chunks` 个 chunk（常规 512B StreamCarry 覆盖前 64 个概念即 95%+ 算力，prefix-shrink 尾部与低 `rho` 异常长尾无缝回退至普通 Inductor 编译核 `_compiled_chunk_fn`，彻底消除 CUDAGraph Tree 分叉膨胀）。

### 3. 平滑 Stage 2 `cap_mix_ratio` 线性暖启动斜坡
- **`train/curriculum_state_machine.py`**：Stage 2 `cap_mix_prob` 从原先的阶跃跳变改为按 `cap_warmup_ramp_progress`（默认 `0.08`，置 `0.0` 可恢复阶跃）从 `0.0` 线性爬坡至 `cap_mix_ratio`。
- **`configs/v2/dcpa_210m_v2_cognitive.yaml`**：显式配置 `cap_warmup_ramp_progress: 0.08`（约 3,040 步完成 `0% → 15%` 过渡：第 37,334 步柔和起步，第 ~40,374 步达目标 `15%`）。

### 4. Watchdog 有限值爆炸拦截与 Drive 断点健康门控
- **`train/watchdog_and_ckpt.py`**：
  - `SelfHealingWatchdog.__init__` 新增可配置阈值：`grad_norm_explosion_threshold=10_000.0`、`grad_norm_warmup_explosion_threshold=100_000.0`、`grad_norm_healthy_cap=1_000.0`、`max_finite_loss=100.0`。
  - `check_step`（Rule A）在 `optimizer.step()` 之前拦截超限的有限值 `grad_norm` 或 `loss_total`，触发标准回滚 + `lr *= 0.7` + 跳过毒批次。
  - `is_healthy_for_checkpoint(..., grad_norm=None)` 同步校验有限值 `loss_total` 与 `grad_norm <= grad_norm_healthy_cap`。
- **`train/train_dcpa.py`**：
  - 将 `grad_norm` 传入 `watchdog.is_healthy_for_checkpoint`。
  - `want_drive`（`wall_due` / `milestone_due` / `is_final_step`）与 `want_milestone` 全部绑定 `step_healthy_ok`（仅 `is_best` 依赖已验证的 `val/bpc` 新低），不健康步打印 `[DriveSync][SKIP]` 拒绝覆盖 Drive 槽位，且仅在真正落盘后调用 `wall_trigger.mark_saved` 以便下一个健康步自动补存。

---

## 三、新增 / 调整的参数与阈值速查（勿随意改动）

| 位置 | 参数 / 常量 | 默认值 / 条件 | 说明 |
| --- | --- | --- | --- |
| `dcpa/model.py` | `DCPAModel.carry_grad_gamma` | `0.5` | BCG-PC 逐步 carry 反向阻尼系数（0.0 ULP 前向恒等） |
| `dcpa/model.py` | `DCPAModel.carry_grad_max_rms` | `0.25` | BCG-PC 逐步 carry 反向单样本 RMS 上限（fp32 安全缩放） |
| `dcpa/modules/bcg_pc.py` | `forward_clause_chunk(..., carry_gamma, carry_max_rms)` | `0.5`, `0.25` | chunk 内 `k > 0` 阻尼参数，保证 `K_c=16` 与 `K_c=1` 等价 |
| `utils/device.py` | `is_vram_headroom_low(..., min_free_gb, min_free_frac)` | `3.5` GB, `0.15` | 实时空闲显存下限（L4 22G → `3.5G`；G4 95G → `14.25G`），含私有池 |
| `dcpa/model.py` | `DCPAModel.bcg_cudagraph_max_chunks` | `4` | CUDAGraph 仅用于 `cur_v == V_eff` 的线性前 4 个 chunk，防树分支 OOM |
| `train/curriculum_state_machine.py` | YAML `stage2.cap_warmup_ramp_progress` | `0.08` | Stage 2 `cap_mix` 线性爬坡占比（210M 约 3,040 步；48M 约 1,824 步） |
| `train/watchdog_and_ckpt.py` | `grad_norm_explosion_threshold` | `10_000.0` | warmup 后 Rule A 有限值梯度爆炸拦截阈值（warmup 期 `100_000.0`） |
| `train/watchdog_and_ckpt.py` | `max_finite_loss` | `100.0` | Rule A 有限值 `loss_total` 爆炸拦截阈值 |
| `train/watchdog_and_ckpt.py` | `grad_norm_healthy_cap` | `1_000.0` | warmup 后允许写入 `last_healthy.pt` / Drive 槽的 `grad_norm` 上限 |

---

## 四、验证摘要

- **新增回归测试**：`tests/test_stage2_stability_and_guards.py`（**5/5 PASSED**），覆盖：
  1. `moe_210m`（`D=768`）32 步 + `tau_m=0.25` + Slot-0 Pinning 梯度爆炸复现，验证 `_DampedBoundedCarry` 0.0 ULP 前向一致且 `GN` 降低 >100x；
  2. `DCPAModel` Stage B 下 `K_c=16` vs `K_c=1` 及 `torch.compile(backend="aot_eager", dynamic=False)` 前反向精确一致；
  3. `is_vram_headroom_low`（L4 / G4 / CPU）与低余量触发 `torch_checkpoint` 路径；
  4. Stage 2 `cap_mix_prob` 第 37,335 步平滑起步并在 42,000 步精准收敛至 `0.15`；
  5. Watchdog Rule A 有限值 `GN=2.8e8` / `loss_total=142.5` 拦截与端到端 Drive Ping-Pong 拒绝覆盖健康槽。
- **回归套件（PyTorch 2.7.0 / `modeva_llm` 与 `llm` 环境）**：
  - `tests/test_stream_carry.py`：**8/8 PASSED**（1 CUDA-only skipped）
  - `tests/test_sanity.py`：**7/7 PASSED**（含 Iron Law #1 静态 CUDA 检查与 4-tier 硬件契约）
  - `tests/test_p0_perf_optimizations.py`：**10/10 PASSED**
  - `tests/test_cli_overrides_and_autotune.py`：**5/5 PASSED**
  - `tests/test_trainer_lifecycle.py`：**8/8 PASSED**
  - `tests/test_v2_cognitive_pipeline.py`：**5/5 PASSED**
  - `tests/test_dcpa_turbo_and_ops.py`：**8/8 PASSED**
  - `tests/test_perf_and_compile.py`：**6/7 PASSED** + `W=32` vs `W=0` TBPTT 梯度余弦相似度 `1.000` (`loss_diff=0.000`)（`test_step2` 第 1 段因本地 PyTorch 2.7.0 CPU 缺 `F.grouped_mm` 触发未改动的 `SparseMicroMoE` BMM 路径，与本次改动无关）。

---

## 五、Colab 续跑恢复提示

- 续跑前务必检查 `/content/drive/MyDrive/DCPA_Turbo/checkpoints/210m/manifest.json` 中的 `step`：
  - 若 `step <= 37300`：断点健康，拉取本次修复后直接 `--resume auto` 续跑即可。
  - 若 `step > 37300`（已在发散期覆盖）：优先检查 `backup_slot` 步数是否 `<= 37300`，否则在 Cell 1 设置 `EXTRA_TRAIN_ARGS = "--resume /content/drive/MyDrive/DCPA_Turbo/checkpoints/210m/milestone_step_035000.pt"`（或健康 `best_model.pt`）回滚续跑。
