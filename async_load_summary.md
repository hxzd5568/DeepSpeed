# async_load 软件抽象层与 call chain 变化摘要

> 两阶段(异步)加载在 hxzd5568 的 casync/datastates engine 之上引入的抽象层变化。
> 详细背景见 async_load.md；本文只做新旧对比与调用链梳理。

## 一、抽象层变化（旧 → 新）

| 层次 | 旧（经典同步） | 新（async_load） | 说明 |
|------|---------------|------------------|------|
| 配置层 `datastates/config.py` | `DeepSpeedDataStatesConfig` 只有 `enabled/enabled_casync/config` | 新增 `async_load`（顶层 key 或 datastates_ckpt/casync_ckpt dict 内，默认 False） | 纯新增，不影响旧配置 |
| 引擎能力位 `checkpoint_engine/checkpoint_engine.py` | 无能力位 | 基类新增非抽象方法 `supports_async_load() -> bool`（默认 False）；Casync/DataStates engine override 为 True；Torch engine 保持 False | 用能力位让"引擎是否支持异步"与"用户是否开启异步"解耦 |
| 公开 API `runtime/engine.py` | 只有 `load_checkpoint(dir, tag)`，一次性同步加载 weights + optimizer | **纯新增** `load_checkpoint_stage(dir, tag, stage=0/1)` 与 `wait_for_optimizer_states()`；原 `load_checkpoint` 签名与行为完全不变 | 两阶段语义：stage0 只加载 weights；stage1/wait 加载 optimizer 状态 |
| 内部加载 `_load_checkpoint` | 无相关参数 | 新增内部参数 `zero_fp32_reconstruct`；`load_optimizer_states=False` 复用 warmstart 路径作为 stage0 | 非 ZeRO 时 stage0 已把整个 state_dict 读入内存并缓存在 `client_state['optimizer']`，stage1 零额外 I/O |
| 异步机制（engine 内部） | 无 | `_staged_checkpoint_state` ctx（load_dir/tag/optimizer_done/error/async_thread）；后台 daemon 线程只做文件 I/O 预取，**apply 在主线程 wait 中执行** | 线程只做 I/O 是 z3 线程安全的关键（避免后台 `optimizer.load_state_dict` 与主线程 gather 竞争） |
| 引擎不支持异步时 | — | `_async_load_enabled()` 门控 = config.async_load ∧ engine.supports_async_load() ∧ 非 universal ∧ 非 pipeline_loading；不满足则退化为**同步两阶段**（两个等价函数调用） | Torch engine 下 async_load=1 与 =0 结果一致（实测验证） |
| save 侧（z3） | 只写标准文件 | `_zero3_async_layout_enabled()`（async_load ∧ zero3）时额外写 `*_fp16_weights.pt`（fp16 分区权重 + buffers） | 旧版 loader 忽略该额外文件，完全兼容 |

## 二、call chain 变化

### 经典同步加载（旧，不变）

```
engine.load_checkpoint(dir, tag)
├─ _load_checkpoint(...)                          # 读 *_model_states.pt：weights + lr_scheduler + client_state
├─ [ZeRO] _load_zero_checkpoint(dir, tag)         # 读 *_optim_states.pt：fp32 master + optimizer states
│        └─ optimizer.load_state_dict(...)
├─ [NVMe offload] copytree(offloaded_tensors)
├─ [z3] checkpoint_event_prologue/epilogue 包裹
└─ [universal] optimizer.update_lp_params()
```

### 两阶段加载（新）

```
engine.load_checkpoint_stage(dir, tag, stage=0)
├─ tag 解析（latest / latest_universal，与 load_checkpoint 相同）
├─ checkpoint_event_prologue
├─ _load_checkpoint(load_optimizer_states=False, zero_fp32_reconstruct=False)
│   ├─ [z3 快路径] _try_load_zero3_fp16_weights → 只读 *_fp16_weights.pt
│   │             （文件不存在/DP 不匹配 → fallback get_fp32_state_dict_from_zero_checkpoint）
│   │             └─ _load_zero3_fp16_partition_state_dict 直接拷入 param.ds_tensor（无集合通信）
│   └─ load_module_state_dict（z0/1/2 走此路径，只读 *_model_states.pt）
├─ [ZeRO] optimizer._restore_from_bit16_weights()  # fp32 暂用 fp16 拷贝
├─ checkpoint_event_epilogue
├─ 构造 _staged_checkpoint_state ctx（含 client_state 缓存）
└─ [_async_load_enabled()] 启动 daemon 线程 _run_staged_optimizer_load
        └─ _prefetch_staged_optimizer_states       # 只做 I/O：
             kind='zero' → _get_all_zero_checkpoints（读全部 *_optim_states.pt）
             kind='nvme' → copytree(offloaded_tensors)
             kind='moe'  → 读 expp_rank_*_optim_states.pt

engine.wait_for_optimizer_states()                # 或 load_checkpoint_stage(stage=1)，幂等
└─ _load_optimizer_states_staged()
   ├─ join 后台线程；线程异常在此重抛
   ├─ [异步] _apply_prefetched_optimizer_states    # 主线程执行（z3 安全）：
   │     kind='zero' → optimizer.load_state_dict(zero_sd_list, ...)
   │     kind='nvme' → optimizer.reset_swap_buffers()
   │     kind='moe'  → optimizer.load_state_dict(optim_checkpoint)
   └─ [同步] _load_optimizer_states_impl           # 引擎不支持异步时的等价路径：
         ZeRO → _load_zero_checkpoint(load_optimizer_states=True)
         非 ZeRO → optimizer.load_state_dict(client_state['optimizer'])  # 零额外 I/O
```

### save 侧（z3 async 布局，新增）

```
engine.save_checkpoint
└─ _save_checkpoint
   └─ [zero3 ∧ async_load] _zero3_async_layout_enabled()
      └─ _get_zero3_fp16_partition_state_dict → 额外写 *_fp16_weights.pt
         （原 *_model_states.pt / *_optim_states.pt 保持不变）
```

## 三、关键设计原则

1. **纯新增**：`load_checkpoint` 及其内部路径零改动，新增 API 全部独立。
2. **能力位门控**：`async_load` 只表达意图，`supports_async_load()` 表达能力，二者取交才真正异步；否则同步两阶段，语义等价。
3. **后台线程只做 I/O**：apply 一律在主线程 wait 中执行，规避 z3 `_partition_all_parameters` 与 forward/backward gather 的竞争（实测复现过 free_param assertion，已按此修复）。
4. **z3 权重先行**：z3 的 `*_model_states.pt` 是空占位，真实权重在 fp32 master 里；save 侧多写 `*_fp16_weights.pt` 使 stage0 只读 P 字节（P=fp16 权重），否则 stage0 无收益。
5. **实测数据（4×4090, Qwen1.5-1.8B, seq2048, ga=16）**：
   - torch：async_load 无效（引擎无能力位），stage1-wait ≈ 6.3s；
   - datastates：async=1 时 stage1-wait 降至 0.69s(z1)/1.37s(z2)（残留为主线程 apply 的 H2D）；
   - compcheck：async=1 时 stage1-wait ≈ 0.005s（完全隐藏）。
