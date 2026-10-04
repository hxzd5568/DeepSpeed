# async_load: 两阶段(异步)checkpoint 加载

> 背景：develop 分支上 hxzd5568 的 7 个 commit 引入了新的 checkpoint engine
> (CasyncEngine / DataStatesCheckpointEngine，基于 datastates-llm)。
> 本文档描述在其之上新增的 `async_load` 配置参数与两阶段加载 API。

## 目标

原同步加载 `load_checkpoint(dir)` 一次性同时加载 **weights** 与 **optimizer 状态**，
而训练中 forward/backward 只需要 weights，optimizer 状态仅在第一处 `optimizer.step()`
之前才真正需要。目标是：

1. 新增一个 config 参数 `async_load`，控制"是否启动异步加载 optimizer 状态"。
2. 新增两阶段加载 API：
   - 第一阶段（`stage=0`）只加载 weights，返回后即可开始 forward/backward；
   - 若 `async_load=True` 且 checkpoint engine 支持异步（如 CasyncEngine），
     optimizer 状态在后台线程中异步加载；
   - 在真正需要 update 之前调用一个 `wait`，确保 optimizer 状态就位。
3. 若 engine 不支持异步（如 TorchCheckpointEngine），提供**等价的两个同步函数调用**：
   第一次加载 weights，第二次加载 optimizer，由使用者在其上自行搭建异步机制。
4. 原始 `load_checkpoint` API 保持不变，原有行为不受任何影响。

## 路线

1. 阅读 hxzd5568 在 develop 分支的全部 commit（`cde7b599`、`d1545006`、`fcf0c560`、
   `a99fc6ff`、`1c84cdc0`、`632f16c6`、`6499f6bb`），弄清 casync/datastates engine
   的接入方式（config 解析、`create_checkpoint_engine` 选择逻辑、`wait()` 语义）。
2. 观察 `DeepSpeedEngine.load_checkpoint` / `_load_checkpoint` / `_load_zero_checkpoint`
   的加载流程，确认 weights 与 optimizer 状态在 ZeRO 下位于**不同的 checkpoint 文件**
   （`*_model_states.pt` vs `*_optim_states.pt`），天然可拆分。
3. 用 `load_optimizer_states=False` 复用既有 warmstart 路径实现 stage=0；
   stage=1 走 `_load_zero_checkpoint`（ZeRO）或从 stage=0 缓存的 client_state 里
   直接应用 optimizer 状态（非 ZeRO），二者合起来与同步加载语义等价。
4. 新增 `async_load` 配置项（顶层 key 或 `datastates_ckpt`/`casync_ckpt` dict 内），
   默认 `False`；engine 基类新增 `supports_async_load()` 能力位
   （Casync/DataStates engine 返回 True，Torch engine 返回 False）。
5. 异步实现：stage=0 里开一个 daemon 后台线程执行 stage=1 的完整逻辑；
   `wait`/stage=1 负责 join 线程并透传异常。
6. 端到端验证：ZeRO-0/1/2/3 下分别对比"两阶段加载 + 再训练一步"与
   "经典 load_checkpoint + 再训练一步"的权重结果完全一致。

## 方法（设计细节）

### 配置参数

`deepspeed/datastates/config.py` 中 `DeepSpeedDataStatesConfig` 新增：

```python
ASYNC_LOAD = "async_load"   # 默认 False
self.async_load = param_dict.get("async_load", False)
# 也支持写在 datastates_ckpt / casync_ckpt 配置 dict 里
```

config JSON 示例：

```json
{
  "zero_optimization": {"stage": 2},
  "async_load": true,
  "casync_ckpt": {"engine_type": "state_engine"}
}
```

### CheckpointEngine 能力位

- `deepspeed/runtime/checkpoint_engine/checkpoint_engine.py`：新增非抽象方法
  `supports_async_load() -> bool`，默认 `False`。
- `casync_checkpoint_engine.py` / `datastates_checkpoint_engine.py`：override 返回 `True`。

### 新公开 API（`deepspeed/runtime/engine.py`，纯新增代码）

```python
# 两阶段加载。stage=0 加载 weights；stage=1 加载 optimizer 状态。
# 返回 (load_path, client_state)，与 load_checkpoint 一致。
load_path, client_state = engine.load_checkpoint_stage(ckpt_dir, tag=tag, stage=0)
...
engine.wait_for_optimizer_states()          # 等价于：先确保 optimizer 状态就位
# 或 engine.load_checkpoint_stage(ckpt_dir, tag=tag, stage=1)  # 幂等
```

语义：

| 场景 | stage=0 | wait / stage=1 |
|------|---------|----------------|
| `async_load=False`（默认）或 engine 不支持 | 只加载 weights（同步返回） | 同步加载 optimizer 状态（wait 里若未加载则自动补做 stage=1） |
| `async_load=True` 且 engine 支持 | 加载 weights，并在后台线程启动 optimizer 加载 | join 后台线程，异常会在此处抛出 |

关键实现点：

- stage=0 内部调用 `load_checkpoint(load_optimizer_states=False)`（官方 warmstart 路径），
  因此 ZeRO-3 会先按 fp32 权重加载，之后 stage=1 的 `_load_zero_checkpoint` 会再次从
  `*_optim_states.pt` 恢复 fp32 master weights + optimizer 状态，最终结果与同步加载一致
  （中间一次 `_restore_from_bit16_weights` 造成的临时精度损失会被 stage=1 覆盖）。
- 非 ZeRO：optimizer 状态与权重在同一文件里，stage=0 已把整个 state_dict 读入内存，
  缓存在 `client_state['optimizer']`；stage=1 只做 `optimizer.load_state_dict`，零额外 I/O。
- MoE：optimizer 状态在独立的 `expp_rank_*_optim_states.pt` 文件中，stage=1 单独加载。
- 异步线程：`threading.Thread(daemon=True)`，`ctx` dict 记录 `optimizer_done` 与 `error`，
  wait/stage=1 时 `join()` 并重抛线程异常。
- tag 解析：stage=0 前按 `latest`/`latest_universal` 文件解析 tag（与 load_checkpoint 相同），
  并把解析后的 tag 存入 staged 上下文供 stage=1 使用。

## 结果

- 所有测试在 1 GPU 上通过（PyTorch 2.x, CUDA 13, deepspeed 0.18.2）：
  - ZeRO-0 / ZeRO-2 / ZeRO-3，`async_load=False`：stage0 后 forward diff = 0，
    optimizer step 数恢复正确，两阶段加载后再训一步的权重与经典 `load_checkpoint`
    再训一步的权重逐元素一致（max diff = 0）。
  - ZeRO-2，`async_load=True`（engine 支持异步）：后台线程加载路径同样通过。
  - `tag=None`（读 latest 文件）路径通过；stage=1 未先调 stage=0 抛 `RuntimeError`；
    stage 非法值抛 `ValueError`。
  - `tests/unit/runtime/test_ds_config_dict.py` 20 项全过（config 改动无回归）。
- 原始 `load_checkpoint` 及其余 API 完全未动，纯新增代码。
- Phase 1+2（ZeRO-3 权重快速加载）已实现并验证，详见下文"待办计划"章节的
  "验证结果（Phase 1+2 已全部通过）"。

## 困难 / 注意事项

1. **ZeRO-3 线程安全（已解决）**：早期实现在后台线程里直接执行
   `optimizer.load_state_dict`（内含 `_partition_all_parameters`），与主线程
   forward/backward 的参数 gather 竞争（实测复现 `free_param` assertion）。
   现已改为后台线程只做文件 I/O（预取），apply 在主线程 `wait` 中执行；
   见"待办计划"章节"Phase 1+2（已实现）"。
2. **ZeRO-3 时序要求**：与 `load_checkpoint` 相同，两阶段加载面向"干净"模型
   （不要在 `save_checkpoint` 后立即对同一 engine 加载）。stage=0 与 stage=1 之间
   允许跑 forward/backward（这正是异步的意义），但 stage=1 会重新分区 fp32 参数。
3. **universal checkpoint**：staged 加载针对标准（非 universal）checkpoint 格式设计
   与验证；universal 格式未充分测试。
4. **Casync/DataStates engine 的 `split_load` 自带 CUDA stream**：后台线程加载时与
   主线程 forward/backward 共享 GPU 内存与默认 stream，极端情况下会有显存抖动；
   datastates engine 的 load 本身仍是同步 API，真正的 I/O 重叠依赖其内部 C++ 异步
   restore 与 Python 线程的结合。
5. **NVMe offload**：`zero_nvme_offload_optimizer` 场景下 stage=1 复制 offloaded
   tensors（copytree），此路径未在测试环境验证。
6. **非异步 engine 的两阶段调用是同步的**：TorchCheckpointEngine 的 stage=0/stage=1
   均同步返回，异步机制（如进程池/线程池）由使用者自行叠加，见下例：

```python
# 非异步 engine 的等价两函数调用（使用者可自行异步化）
lp, cs = engine.load_checkpoint_stage(ckpt_dir, tag=tag, stage=0)   # 只加载 weights
# ... 使用者自己的异步调度 ...
lp, cs = engine.load_checkpoint_stage(ckpt_dir, tag=tag, stage=1)   # 只加载 optimizer 状态
```

---

## 待办计划：修复 ZeRO-3 下 stage=0 无收益的问题

### 问题诊断（已实测验证）

ZeRO-3 下两阶段加载目前**没有降低 GPU stall**，原因在 stage=0 复用了
`load_checkpoint(load_optimizer_states=False)` 的 warmstart 语义，而
`_load_checkpoint`（engine.py）里有：

```python
if self.zero_optimization_partition_weights() and not load_optimizer_states:
    checkpoint['module'] = get_fp32_state_dict_from_zero_checkpoint(load_dir)  # 读整个 optim 文件！
```

它把 `*_optim_states.pt` 完整读进来重建 fp32 权重。实测：把 optim 文件改名后
stage=0 直接 `FileNotFoundError`。于是 stage=0 读了两个文件（权重文件白读 + optim
文件全读），stage=1 又把 optim 文件读一遍——总 I/O 比同步加载还多一次。

**进一步实测发现（关键）**：ZeRO-3 的 `*_model_states.pt` 里 `module` 权重是
**空占位 tensor（shape [0]）**——分区状态下 `param.data` 被 `free_param` 释放，
真实权重只存在于 `*_optim_states.pt`（fp32 flat groups）。经典同步加载的权重恢复
完全来自 optim 文件的 fp32（`_rigid_load_state_dict` 恢复 fp32 再拷回 fp16）。
因此"只改 load、不动 save"对 z3 是**不可能的**：model 文件里根本没有权重可读。

| 文件 | 内容 | 大小（P = fp16 权重字节数） |
|------|------|------|
| `*_model_states.pt` | 空占位 + buffers + 元数据 | ~0 |
| `*_optim_states.pt` | fp32 master（2P）+ Adam moments（4P）+ ds_config | ~6P |

ZeRO-2 / 非 ZeRO 不受影响（model 文件里有真实 fp16 权重，stage=0 本就只读权重文件）。

### 计划（三阶段，可独立落地）

#### Phase 1：load 侧修复（不改 save，最小改动）

~~思路：z3 的 model 文件里本来就有 fp16 分区权重，stage=0 直接用它，不碰 optim
文件。~~（已证伪：z3 model 文件是空占位，见上文诊断。）实际落地为 Phase 1+2 合并：

#### Phase 1+2（已实现）：save 侧写 fp16 权重文件 + load 侧直读

**save 侧**（用户提出的思路——save 看到 `async_load` 且 zero3 时做设定）：
- 新增 `_zero3_async_layout_enabled()`：`async_load=True && zero3` 时生效；
- `_save_checkpoint` 在原文件**之外**额外把 fp16 分区权重（`param.ds_tensor`，
  含 buffers）写到独立文件 `*_fp16_weights.pt`；原 `*_model_states.pt`、
  `*_optim_states.pt` 保持原样 → 旧工具/旧版本完全兼容。

**load 侧**：
- `_load_checkpoint` 新增内部参数 `zero_fp32_reconstruct`（默认 None=原行为）；
- stage=0 走快速路径：检测 `*_fp16_weights.pt` 存在且 dp world size 一致时，
  只读该文件（P 字节），用 `_load_zero3_fp16_partition_state_dict` 把分区切片
  直接拷进 `param.ds_tensor`（无集合通信）；
- 旧 checkpoint（无 fp16 文件）自动 fallback 到 fp32 重建路径（慢但正确）。

**异步线程安全修复**（实现过程中发现并解决）：z3 下后台线程若直接执行
`optimizer.load_state_dict`（内含 `_partition_all_parameters`），会与主线程
forward/backward 的参数 gather 竞争（实测复现 `free_param` assertion 崩溃）。
现改为：**后台线程只做文件 I/O**（`_prefetch_staged_optimizer_states`，预取
zero optim state dicts / NVMe copytree / MoE optim 文件），
**apply 在主线程的 wait 里执行**（`_apply_prefetched_optimizer_states`，
含 prologue/epilogue）。universal checkpoint 与 `pipeline_loading_checkpoint`
场景自动禁用异步（退化为同步两阶段）。

#### Phase 3：进阶（可选，未实现）

权重文件改存 raw fp16 flat buffer + offset 索引（免 pickle），stage=0 用
`torch.frombuffer` + 单次 GPU memcpy，接近零反序列化；若 casync engine 提供
mmap / lazy load 则直接对接。

### 验证结果（Phase 1+2 已全部通过）

1. **负向测试**：改名 optim 文件后 stage=0 成功、forward diff = 0（证明只读
   fp16 权重文件）；wait 恢复 optim 文件后正常完成；
2. **一致性**：z3 `async_load=True` 两阶段加载 + 再训一步 vs 经典
   `load_checkpoint` + 再训一步，权重逐位一致（max diff = 0）；重复 3 次稳定；
3. **兼容性**：旧布局 checkpoint（无 fp16 文件）在新代码下自动走 fp32
   fallback；新布局 checkpoint 用经典 `load_checkpoint` 加载不受影响；
4. **回归**：z3/z2/z0 × async 开/关 全过；`tests/unit/runtime/test_ds_config_dict.py`
   20/20。

### 风险

1. stage=0 → wait 之间**禁止** `step()`（此窗口内 fp32 只是 fp16 的拷贝，
   精度有损）；forward/backward 无碍；
2. stage=0 需 pristine（分区状态）模型，与 `load_checkpoint` 约束相同；
3. **线程安全**：后台线程仅做 I/O（CPU 读取 / engine 内部流），apply 在主线程
   wait 中执行，已消除 z3 分区竞争；但 engine 的 `split_load` 在后台线程里
   `torch.cuda.empty_cache()` 等操作与主线程计算共享 allocator，极端情况下
   有显存抖动，需在目标集群验证；
4. **新布局 checkpoint 的旧版本兼容**：`*_fp16_weights.pt` 是额外文件，旧版本
   loader 会忽略它（标准文件未变），因此完全兼容；反之旧布局 checkpoint 在新
   代码下也正常（fallback）；
5. `async_load=True` 保存会多写一个 fp16 权重文件（约 P 字节，与 optim 文件
   相比可忽略）；
6. fp16 权重文件的 partition 布局与 DP world size 绑定（与 ZeRO optim 文件
   相同）；world size 变化时自动 fallback 到 fp32 重建路径；
7. universal checkpoint / pipeline loading checkpoint 不支持异步预取
   （自动退化为同步两阶段）。
