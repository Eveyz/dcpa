# 2026-11-28 · DCPA-Flat v3.3 大训前 P0 数学/架构审计修复与双栈吞吐加固简报

- **日期**：2026-11-28
- **分支**：`feat/dcpa-flat-v3-3`
- **关联提交**：`504998f`（代码实现与双栈测试）、`01fa1df`–`4ae46a1`（数据管线 HF 5.x 校验、流式 Drive 打包解压与 `My Drive/dcpa_flat` 路径隔离）
- **背景**：在正式租用 Colab RTX Pro 6000 / A100 启动 `killgate_5k` 与 `206M` 长程预训练前，对 DCPA-Flat v3.3 Python 训练栈与 Rust `no_std` 推理栈做了一次端到端独立审计与二次复核，集中清除会导致表征坍缩、错位绑定、死参数、显存/算力空转以及门控对比失真的 P0/P1 隐患。
- **原则**：所有涉及前向语义的改动均同步落地至 Rust `no_std` 引擎（保持 `0 allocs/byte`、`BrainSnapshotV3` 字节布局零膨胀、Gate 6 双栈对齐 `max_abs_err <= 1.5e-6`），并严格执行 5 条实现级护栏防止修旧 Bug 引入新 Bug。

---

## 一、问题根因与边界取舍

### 1. 必须在大训前清除的 P0/P1 根因
1. **JEPA 自监督分支平凡坍缩与伪 EMA 腐化**：原 `self.jepa_norm` 带可学习仿射权重并同时作用于预测端 `jepa_hat` 与目标端 `jepa_tgt`，在 `weight_decay=0.05` 与预测梯度夹击下会被压向 `0` 导致 `seg_jepa_err -> 0`（连带污染不确定性头 `u_n` 回归目标）；原 `jepa_target_online` 挂在实时 `x3` 上且 `RMSNorm` 径向梯度恒为 `0`，在 WD 下只会指数衰减直至触碰 `eps` 噪声，白占 ~1.18M 参数（206M）。
2. **优化器 Weight Decay 错伤一维参数与初始态**：`parameter_groups()` 曾将全部 `RMSNorm`、偏置、门控标量及图载体初始槽/边放进 `weight_decay=0.05`，长程训练会缓慢侵蚀归一化尺度与门控偏置。
3. **SGSC 单字节编码缺少局部 N-gram 感受野**：仅用单字节查表直接进入对角扫描，UTF-8 多字节汉字（3B）与代码标识符前缀缺少低层局部组合能力。
4. **G2 → 海马体槽位归属错配与死参数**：`ContractiveGraphCarrier.step_single` 未向海马体透传真实写入槽 `slot_star` 与合并对 `(win_i, win_j, merge_active)`，导致资格迹更新与实际图合并解耦；`HippocampusFlat.f_prec_cort` 无下游消费，属纯死参数；`rho_out = sigmoid(alpha_out * (g_sem - theta_out))` 在海马体空表（`g_sem == 0`）时仍保留非零底噪；内层 $K \le 8$ 三角求解在每块调用通用 `solve_triangular` 开销过高。
5. **Cortical 窗口注意力降回 FP32 + 精修门控硬阶跃**：`PreG1CorticalLayer` 将 `Q @ K^T` 与 `Attn @ V` 整个包在 `fp32_island` 中，损失 BF16 Tensor Core 吞吐；L7–L8 门控 `sigmoid(u_n - tau) * (u_n > tau)` 在阈值 `u_n = tau` 处存在 `0 -> 0.5` 不连续跳变；`206M/650M` 使用 `n_heads=8` 导致 `d_head=96/128` 不统一。
6. **训练器/Killgate/插件工程瓶颈与对比偏差**：
   - `train_flat.py` 存在每步双次 GPU 同步、DataLoader 单进程无 `pin_memory`、子模块未启用 `torch.compile`、`grad_accum > 1` 仅用最后 micro-step 的 `expert_counts` 更新 MoE 负载偏置、阶段切换冷启动无微热身及 `cap_mix_prob` 阶跃切入。
   - `run_flat_killgate_5k.py` 启动时预加载全量 5,800 个 batch 占满显存、缺少 CUDA BF16/compile、Run A 基线未与 Run C 对齐 padding 掩码与逐泳道 `lang_ids`、未拆分 `val_en_zh` 与 `val_code`、日志缺 `steady_kb_s` / `overflow_byte_ratio`。
   - `train_plugin.py` 采集隐状态原型时未按 `sample_mask` 过滤 lane 尾部 padding 段。

### 2. 经复核明确**不做**或**延后**的项目
- **拒绝将 `charge_e` 回灌进 `PMD.x_in`**：`charge_e` 依赖前序 `m_seq` 累积耗电，回灌会构成跨步非线性闭环并破坏 `parallel_diag_scan` 并行扫描；改为将 `lambda_halt` 从 `1.0` 下调至 `0.2` 消除梯度抢占。
- **保留 `x_seed = x3_prev` 不强换 `c_fused_prev`**：避免 Stage 3 → Stage 4 切入海马体时主干残差分布跳变。
- **延后至 `killgate_5k` / 底座训练后**：多卡 DDP、自回归 `chat-flat`、Rust 环缓存与 BF16 权重格式、多轮离线睡眠回放（均不影响单卡底座数学语义与权重布局）。

---

## 二、修复清单（做了什么）

### 1. JEPA 无参数 Unit-RMSNorm 与精修前难度解耦（护栏 #1）
- **`dcpa-train-py/dcpa/model_flat.py`**：
  - 删除 `jepa_target_online`、`jepa_target_ema`、`jepa_norm` 与 `_sync_jepa_ema`。
  - 实现无仿射参数 `_unit_rmsnorm(x)`（保持 $\|x\|_2 = \sqrt{D}$，避免 `F.normalize` 配合 `.pow(2).mean(-1)` 导致损失被除以 $D=768$ 缩小近千倍），使 `loss_jepa` 与 `seg_jepa_err` 稳定在 `O(1)` 量级。
  - `seg_difficulty` 里的 JEPA 误差项改用 L7–L8 精修前的 `z_hat_6` 计算，切断精修对不确定性目标的负反馈。

### 2. 优化器 5 组拆分与 Weight Decay 严格免除
- **`dcpa-train-py/dcpa/model_flat.py`**：
  - 重写 `parameter_groups()`：全部 `ndim < 2` 参数（`RMSNorm` 权重、偏置、门控标量等）以及 `init_slots_raw`、`init_edges_raw`、`sgsc.conv_w` 强制归入 `weight_decay = 0.0` 分组；所有分组显式附带 `"base_lr"` 键供余弦调度与阶段微热身精确缩放。

### 3. SGSC 文档隔离 `k=3` 因果深度可分离卷积（护栏 #2，Python + Rust 同步）
- **`dcpa-train-py/dcpa/modules/sgsc.py`**：
  - 新增 `conv_w [3, d_s]`，初始化为 `[1, 0, 0]`（零改动初始方差）。
  - 批前向禁用裸 `nn.Conv1d(..., padding=2)`（会令同一泳道内相邻两篇独立文档跨界污染），改用 `scan_reset` 严格掩码的 `t, t-1, t-2` 三阶移位加权，确保跨文档泄漏恒为 `0.00e+00`。
  - `SGSCFlatStreamState.prev2_bytes` 初始化改为哨兵 `[-1, -1]`，保证 `t=0, 1, 2...` 单步流式与批处理完全对齐。
- **`dcpa-core-rs/src/flat/{state.rs, weights.rs, engine.rs}`**：
  - `FlatSgscState::new/reset` 将 `prev2_bytes` 置为 `[-1, -1]`（不改变 `BrainSnapshotV3` 任何字节偏移或大小）。
  - `FlatWeights` 加载 `sgsc.conv_w`（删除已移除的 `hip_f_prec_cort_w/b`）。
  - `DcpaFlatEngine::sgsc_ingest_byte` 同步执行基于 `prev2_bytes` 哨兵保护的三阶加权。

### 4. G2 → 海马体真实槽归属、去死参数、幂零逆与平滑 `rho_out`（护栏 #3，Python + Rust 同步）
- **`dcpa-train-py/dcpa/modules/graph_carrier.py`**：
  - `step_single` 新增 `return_slot_info: bool = False`，输出精确的 `{slot_assign, win_i: u_star, win_j: v_star, merge_active}`（`u_star = min(win_i, win_j)`、`v_star = max(win_i, win_j)`，排除固定 Slot-0）；`DCPAModelFlat._g2_carrier_step_adapter` 透传 `return_slot_info=True`。
  - 新增 `enable_compiled_chunk` / `disable_compiled_chunk` 支持块级 `torch.compile`。
- **`dcpa-train-py/dcpa/modules/hippocampus_flat.py`**：
  - 移除死参数 `f_prec_cort`，`_compute_pi_cort` 返回 `torch.ones_like(c_slow, dtype=torch.float32)`。
  - 仅在理解侧内层 $K \le 8$ 块内（`L` 严格幂零，$L^8 = 0$）用精确无回代恒等式 $(I+L)^{-1} = (I-L)(I+L^2)(I+L^4)$ 替换 `solve_triangular`；表达侧 `[P, 256, 256]` 严格保留 `torch.linalg.solve_triangular`。
  - `rho_out` 改为 `g_sem * torch.sigmoid(alpha_out * (g_sem - theta_out))`（`g_sem == 0` 时严格等于 `0.0`），并同步更新 `model_flat.py` 流式路径与 `tests/test_flat_ws4_hippocampus.py` 教科书参考实现。
- **`dcpa-core-rs/src/flat/{state.rs, engine.rs}`**：
  - `FlatScratchPool` 新增 `car_slot_star`、`car_win_i`、`car_win_j`、`car_merge_active`，`step_carrier_single` 与 `step_segment` 同步执行精确的 `car_merge_active` 合并继承与 `car_slot_star` 归属更新，以及 `g_sem * sigmoid(...)` 的 `rho_out` 公式。

### 5. Cortical BF16 窗口注意力、平滑 L7–L8 门控与统一 `d_head = 64`（护栏 #4）
- **`dcpa-train-py/dcpa/modules/cortical_moe.py`**：
  - `PreG1CorticalLayer` 将 `Q @ K^T` 与 `Attn @ V` 移出 `fp32_island`，运行在原生 BF16 Tensor Core 上，仅掩码与 `softmax` 保留 FP32。
  - L7–L8 门控替换为连续平滑死区 `tanh(2.0 * F.relu(u_n - tau))`（训练、评估、流式及 Rust `engine.rs` 的 `tanhf(2.0 * (u_n - tau_think))` 全部对齐）。
- **`dcpa-train-py/dcpa/interfaces.py` & `configs/v3/dcpa_flat_{48m,206m,650m}_v3.yaml`**：
  - `lambda_halt: 1.0 -> 0.2`。
  - `flat_206m`: `n_heads: 8 -> 12`（`D=768 -> d_head=64`，CopyRing `d_s=192 -> d_k=16`）；`flat_650m`: `n_heads: 8 -> 16`（`D=1024 -> d_head=64`）；Rust `config.rs` 原生兼容无需改动。
  - `dcpa_flat_48m_v3.yaml`、`dcpa_flat_206m_v3.yaml` 开启 `hardware.compile_chunks: true`。

### 6. 训练器、Killgate 与插件校准加固（护栏 #5）
- **`dcpa-train-py/dcpa/model_flat.py` & `train/train_flat.py`**：
  - 新增 `enable_compiled_submodules()` / `disable_compiled_submodules()`；checkpoint 保存统一使用 `get_clean_state_dict(model)`（恢复时防御性剥离 `_orig_mod.` 前缀），并在权重导出与 Gate 验证前调用 `disable_compiled_submodules()`。
  - 接通 `build_static_grid_dataloader(num_workers, pin_memory)`（CUDA 自动启用 2–4 worker + `pin_memory=True` + `non_blocking=True` 异步 H2D）。
  - 移除每步双次 `synchronize()`（仅在日志/评估/保存边界同步）；`post_optimizer_step` 支持传入跨全部 `grad_accum` micro-batch 累积的 `expert_counts_override`。
  - 新增阶段切换后 500 步 `0.1 -> 1.0` 微热身以及每个阶段前 20% 窗口内 `cap_mix_prob` 线性爬坡。
- **`gates/run_flat_killgate_5k.py`**：
  - 改为流式 batch 迭代器（消除启动时 5,800 个 batch 预加载显存暴涨），开启 CUDA BF16 + 子模块编译并在 Gate B/E 前解包。
  - 对齐 Run A 与 Run C 的 padding `loss_weights` 掩码和真实逐泳道 `lang_ids`；拆分独立 `val_en_zh` 与 `val_code` 评估集；补齐 `Colab_DCPA_Flat_v3.ipynb` 所需的 `steady_kb_s` 与 `overflow_byte_ratio` 字段。
- **`dcpa-train-py/dcpa/train_plugin.py`**：
  - `_collect_layer_hidden_samples` 改用 `register_forward_pre_hook(..., with_kwargs=True)`，按 `sample_mask` 严格剔除无效尾段 padding 后再聚合原型与校准 `tau`。

---

## 三、新增 / 调整的参数与阈值速查（勿随意改动）

| 位置 | 参数 / 常量 | 默认值 / 条件 | 说明 |
| --- | --- | --- | --- |
| `dcpa/interfaces.py` | `DCPAFlatConfig.lambda_halt` | `0.2`（原 `1.0`） | 防止 PMD 停机惩罚过度抢占主语言梯度 |
| `dcpa/interfaces.py` & `configs/v3/` | `flat_206m.n_heads` / `flat_650m.n_heads` | `12` / `16`（原 `8`） | 统一 `d_head = 64` 适配 Tensor Core，CopyRing `d_k = 16` |
| `configs/v3/dcpa_flat_{48m,206m}_v3.yaml` | `hardware.compile_chunks` | `true` | CUDA 上启用子模块 `torch.compile`，保存/导出前自动解包 |
| `dcpa/modules/sgsc.py` & Rust `state.rs` | `SGSCFlat.conv_w` / `prev2_bytes` | `[3, d_s]` 初始化 `[1,0,0]` / `[-1,-1]` | 文档隔离 `k=3` 因果深度卷积，跨文档零泄漏 |
| `dcpa/modules/cortical_moe.py` & `engine.rs` | L7–L8 精修门控 | `tanh(2.0 * relu(u_n - tau_think))` | 阈值处连续无跳变，双栈位级公式一致 |
| `dcpa/modules/hippocampus_flat.py` & `engine.rs` | 表达侧 `rho_out` | `g_sem * sigmoid(alpha_out * (g_sem - theta_out))` | `g_sem == 0` 时严格为 `0.0`，消除空海马体底噪 |
| `dcpa/modules/hippocampus_flat.py` | 块内幂零逆阶数 | 仅用于理解侧 $K \le 8$ | 表达侧 `256×256` 严格保留 `solve_triangular` |
| `train/train_flat.py` | 阶段切换微热身 | `500` steps（`0.1 -> 1.0`） | 缓解 Stage 1→2 与 Stage 3→4 冷启动层梯度冲击 |
| `train/train_flat.py` | `cap_mix_prob` 爬坡期 | 阶段窗口前 `20%` 线性爬坡 | 防止控制字节分片阶跃切入引爆 `rho` |
| `train/train_flat.py` | CUDA `DataLoader num_workers` | `min(4, max(2, cpu_count // 2))` | 配合 `pin_memory=True` 与 `non_blocking=True` 消除喂数空泡 |

---

## 四、验证摘要

- **Rust `no_std` 与工作空间测试**：
  - `cargo check --no-default-features`（`dcpa-core-rs`）：**PASS**（0 warnings）
  - `cargo test --workspace`：**14/14 PASSED**
  - `cargo build --release`（`dcpa-cli` + `dcpa-sleep-daemon`）：**PASS**
- **Gate 6 双栈一致性与零分配验证（`gates/gate6_flat_v3_alignment.py`）**：
  - **PASS**：`heap_allocs = 0`；全部 17 个中间/输出张量 `cos_sim = 1.0000000`、`max_abs_err <= 1.192e-6`；插件零漂移与睡眠晋升验证通过。
- **5 条实现级护栏专项断言**：
  - **PASS**：`206M (H=12, d_head=64)` / `650M (H=16, d_head=64)`；WD 免除清单全绿；SGSC `k=3` 跨文档泄漏 `0.00e+00`、批/流差 `1.31e-06`；Unit-RMSNorm JEPA 初始 MSE `2.0625`（`O(1)` 量级）；`rho_out[0] == 0.0`。
- **Python 全量单元测试与冒烟验证**：
  - `pytest tests/test_flat_*.py`：**57/57 PASSED**（含同步参考实现后的 `tests/test_flat_ws4_hippocampus.py` 8/8）
  - `gates/run_flat_killgate_5k.py --smoke`：**PASS**
  - `train/train_flat.py --smoke`：**PASS**

---

## 五、二次交叉审计增量收尾（5k 门禁前必改项）

在 `504998f` 之后，针对第二轮深度复审确认的 4 组漏项与新发现的动力学不动点问题，完成了以下增量加固：

1. **`occ_relax = 0.995` + `full_gate_temp = 0.05` 根治 38 段静默抹除与“僵尸槽”不动点（Python / YAML / Rust 同步）**
   - 原 `occ_relax = 0.96` 会令未命中槽在 38 段（~360B）内跌破 `full_gate_center = 0.20` 被空槽分支静默覆盖且绕过海马体淘汰信号；调至 `0.995` 将寿命延长至 ~313 段（~3,000B）并保留归并打分器的时间新旧对比度。
   - **实测新发现**：单独把 `occ_relax` 调高至 `0.995` 时，原 `full_gate_temp = 0.10` 在 `min_free_rho = 0.603` 处仍保留 `1.75%` 的 Sigmoid 尾漏，正好与 `0.5%` 占用率衰减构成稳定不动点，持续向最冷槽混入随机向量并把范数压坍至解析值 `0.088`（5/15 随机种子在 512 步测试中触发 `< 0.15` 坍缩）。将 `full_gate_temp` 收紧至 `0.05`（最大斜率仍仅 `5.0`，尾漏降至 `0.033%`）后，15 个随机种子 × 512 步实测坍缩率从 `5/15` 降至 **`0/15`**，最差槽位范数由 `0.0762` 跃升至 **`0.6626`**。
   - 同步更新 [interfaces.py](../../dcpa-train-py/dcpa/interfaces.py)、4 份 `configs/v3/*.yaml` 与 Rust [config.rs](../../dcpa-core-rs/src/flat/config.rs)（一并补齐 Rust `default_lambda_halt = 0.2` 与 `flat_206m().n_heads = 12`）。

2. **`CorticalStackFlat` 方法级 `torch.compile` + 零同步精确有效段 `25%` 精修与第 75 百分位 `tau_think`**
   - 为 `CorticalStackFlat` 新增 `enable_compiled_methods()` / `disable_compiled_methods()`（独立编译 `_forward_pre_g1_impl` / `_forward_post_g1_impl` / `_forward_predict_and_refine_impl` 纯张量内核，零 `_orig_mod.` 键污染）并接入 `DCPAModelFlat.enable_compiled_submodules()`，补上占 GPU 算力 70%+ 的皮层编译缺口。
   - 训练态 `L7–L8` 保留静态 `k_ref = round(refine_frac * N_flat)`（绝不引入动态形状，也不因硬压 `0.17` 而在中文/代码高密段或 100% 有效单测中截断），在 GPU 上以零同步计算 `target_k = round(n_valid * refine_frac).clamp(1, k_ref)` 掩码 `sel_active = valid_sel & (arange(k_ref) < target_k)`，并用静态形状 1D `gather(0, (target_k - 1).unsqueeze(0))`（`fullgraph=True, dynamic=False` 原生零 unbacked symbol）更新精确的第 75 百分位 `tau_think`。

3. **`SGSCFlat` 断开 `s_ste_end` 伪梯度 + 嵌入/行归一化头 WD 豁免 + `StaticGridBatch.pin_memory()`**
   - [sgsc.py](../../dcpa-train-py/dcpa/modules/sgsc.py) 在 `norm_f(h_end + s_ste_end.detach() * agg)` 处断开残差尺度伪梯度，消除对 `theta_raw` / `log_temp_sgsc` 的噪声干扰。
   - [model_flat.py](../../dcpa-train-py/dcpa/model_flat.py) 将 `sgsc.{embed, w_out, lang_embed, style_embed}.weight` 纳入 `weight_decay = 0.0` 显式免除清单。
   - [interfaces.py](../../dcpa-train-py/dcpa/interfaces.py) 为 `StaticGridBatch` 补齐 `.pin_memory()`，真正激活 `DataLoader(pin_memory=True)` + `non_blocking=True` 异步 H2D。

4. **训练器与 `killgate_5k` 优化器/同步/微预热收口**
   - [train_flat.py](../../train/train_flat.py) 与 [run_flat_killgate_5k.py](../../gates/run_flat_killgate_5k.py) 统一启用 `AdamW(betas=(0.9, 0.95), fused=(device.type == "cuda"))`。
   - `train_flat.py` 将日志步双 `.item()` 合并为单次 `torch.stack(...).to("cpu")`，非日志 CUDA 步合并为单次 1 字节布尔归约并补齐 `grad_norm_t <= max_healthy_gn` 拦截（消除 9/10 步的 Watchdog 盲区）。
   - `run_flat_killgate_5k.py` 为 Run A / B / C 统一增加 200 步线性 LR 微预热（`0.1 -> 1.0`），并将 `opt` 构造前置到 `enable_compiled_submodules()` 之前。

### 留待 5k 门禁通过后、206M 五阶段大训前实施（不阻塞 5k 门禁）
- **Stage 4 海马体 35GB Autograd 显存炸弹排雷**：利用 `evict_weight` 严格仅 `(u*, v*)` 2 槽非零的数学性质，将理解侧缓冲与表达侧三角求解从 `K=8` 无损化简为 `2` 槽（显存降 4 倍、表达侧求解从 `256³` 降为 `64³`），叠加 `C=32` 块级 Checkpointing、删死张量 `prec_bu_seq`、以及 `is_assim` 保护高显著度 `elig_f`（同步 Rust `absorb_segment`）。
- **Stage 4 Window Dropout / 长距 Wug 混采** + **Stage 5 同步冻结 `sgsc.theta_raw` 与词干投影** + **Stage 4 海马体新解冻参数组 2x–3x LR 补偿**。
