# hot_weights: checkpoint weights 内存热备

> 背景：在 `async_load`（两阶段加载，见 async_load.md）之上，新增 `hot_weights`
> 配置参数，实现 checkpoint **weights 的内存热备**：save 落盘完成后把权重快照
> 驻留在内存中的 engine 对象里，load 时优先从内存直接恢复权重（零磁盘 I/O），
> 使 weights 可以先于 optimizer 状态被"率先恢复"。
> 本文档同时记录 DeepSpeed 侧的改造计划与 cmpckp（datastates-llm，仓库路径
> /home/work/cmpckp）侧需要的相应修改。

## 目标

1. 新增 config 参数 `hot_weights`（默认 `False`），与现有 `async_load` 正交：
   - `hot_weights` 管 **weights 的来源**（内存热备 vs 磁盘文件）；
   - `async_load` 管 **optimizer 状态的加载时机**（后台线程 vs 同步）。
2. **ZeRO-0/1/2**：save 时把 fp16 module weights（`*_model_states.pt` 中
   `state['module']` 的内容）快照为独立的 CPU 副本，在 persist 完成后赋给
   engine 对象（`self._hot_weights`），使其不被回收；load 时先查该对象，
   命中则直接从内存恢复 weights。
3. **ZeRO-3**：save persist 完成后，从 fp32 master weights
   （`fp32_partitioned_groups_flat`）抽取权重并按模型 dtype cast 16
   （fp16/bf16），连同分片元数据（param_shapes、dp_world_size、buffers 等）
   存入热备对象；load 时校验分片布局一致后直接从内存拷回 `param.ds_tensor`。
4. **再次 save**：若热备对象已存在，在 persistent weight + optimizer
   **完整落盘之后**更新指针，热备始终指向最新的已落盘对象；persist 失败则
   保留旧指针。
5. 兼容：`hot_weights=False` 行为与现状完全一致；热备缺失/失效（结构变化、
   DP world size 变化）自动 fallback 到现有磁盘路径（z3 依次尝试
   `*_fp16_weights.pt` → fp32 重建）。

## 与 casync/datastates（cmpckp）引擎的关系（关键适配点）

cmpckp 的 `coalition_save`（/home/work/cmpckp/llm/datastates/engines/engine_state.py:463）
是**流水线分块**处理：压缩结果放入 `GpuArena`，`batch_commit` 后逐块
`arena.free_oldest()`（engine_state.py:544-551），因此引擎侧**不会让全部
weights 同时驻留内存**，persist 完成后其暂存 buffer 会被清掉。基于此确立两条
设计原则：

1. **热备必须是我们自己在 save 时刻克隆的独立 CPU 副本**，绝不能是"把传给
   引擎的 state_dict 引用存进 engine 对象让它不要消亡"。原因：
   - `module_state_dict()`（engine.py:2919）返回与活参数**共享 storage** 的
     tensor；z3 的 `_rigid_state_dict` 返回的 `FP32_FLAT_GROUPS`
     （stage3.py:2700）是**活 fp32 buffer 的引用**。持有引用会导致热备被后续
     训练污染成"最新权重"而非"落盘权重"，且白占 GPU 内存。
   - 引擎的流水线分块释放只作用于它自己的 staging 内存，与我们的副本无关。
2. **快照必须同步完成**（在 save 调用点、persist 之前），不能像 `async_load`
   那样丢给后台线程事后拷贝：源 tensor 在下一步训练时就会被改写。

代价：save 关键路径新增一次同步 D2H（P 字节，P = fp16 权重总字节数；z3 每 rank
约 P/dp_world_size）。torch engine 无感（save 本就同步）；对 casync/datastates
会部分抵消其异步收益 → v1 接受并实测，v2 由 cmpckp 侧配合消除（见下文
"cmpckp 侧修改"）。

## 配置参数

`deepspeed/datastates/config.py` 中 `DeepSpeedDataStatesConfig` 新增：

```python
HOT_WEIGHTS = "hot_weights"   # 默认 False
self.hot_weights = param_dict.get("hot_weights", False)
# 也支持写在 datastates_ckpt / casync_ckpt 配置 dict 里（与 async_load 同套路）
```

config JSON 示例：

```json
{
  "zero_optimization": {"stage": 3},
  "async_load": true,
  "hot_weights": true,
  "casync_ckpt": {"engine_type": "state_engine"}
}
```

## DeepSpeed 侧改造计划

### Phase A：config 与状态

- `deepspeed/datastates/config.py`：新增 `hot_weights` 解析（见上）。
- `deepspeed/runtime/engine.py` `__init__`：
  - `self._hot_weights = None`（已就绪的热备，指向最新已落盘对象）
  - `self._pending_hot_weights = None`（本次 save 固化、尚未 persist 的快照）
  - helper `_hot_weights_enabled()`：读 `self._config.datastates_config.hot_weights`。

### Phase B：save 侧（快照 + 指针提升）

1. **z0/1/2 快照**（`_save_checkpoint` 内，`checkpoint_engine.save` 返回之后、
   函数退出之前——与引擎同步压缩同窗口，保证与落盘内容一致）：
   `_pending_hot_weights = {'module': state['module'] 的 detach().cpu() 副本,
   'lr_scheduler': …, 'client_state': …, 'dp_world_size': …,
   'mp_world_size': …, 'buffer_names': …}`。注意 `state['module']` 与活参数共享
   storage，必须 D2H 拷贝成独立 CPU 副本。
2. **z3 快照**（`_save_zero_checkpoint` 内，`optimizer.state_dict()` 之后）：
   从返回的 `FP32_FLAT_GROUPS`（活 fp32 master 引用）按 group cast 到模型
   dtype（`self.bfloat16_enabled()` → bf16，否则 fp16）并拷贝到 CPU；同存
   `param_shapes`（`_get_zero_param_shapes()`，提供 flat group → 参数名的顺序
   映射）、`buffer_names`、`dp_world_size`、`mp_world_size`。z3 的 `*_model_states.pt`
   是空占位（见 async_load.md 诊断），真实权重只在 master 里，所以必须从这里抽。
3. **指针提升** `_promote_hot_weights()`：
   `self._hot_weights = self._pending_hot_weights; self._pending_hot_weights = None`。
   - 非 decoupled（torch engine）：`save_checkpoint` 尾部
     `checkpoint_engine.commit(tag)` **成功之后**（engine.py:3891-3892）；
   - decoupled（datastates/casync engine）：`_commit_decoupled_checkpoint`
     （engine.py:3901）的 `commit` 成功之后；该函数在
     `is_gradient_accumulation_boundary` 时被 `step()` 触发（engine.py:2521-2522），
     因此"persist 完成才换指针"的语义天然满足；
   - `commit()` 内部即 `wait(persist=True)`，抛异常则不提升、保留旧指针。
   - pending 只保存最近一次 save 的快照（多次 save 未 commit 时前一次被覆盖，
     可接受：热备只承诺指向最新已落盘对象）。
4. v1 明确不支持：MoE（`has_moe_layers`）、pipeline、universal checkpoint →
   打印 warning 并跳过热备。

### Phase C：load 侧（内存优先恢复）

- `load_checkpoint_stage(stage=0)`：tag 解析后、磁盘加载前先
  `_try_load_hot_weights()`：
  - **校验**：热备非空；`dp_world_size`/`mp_world_size` 与当前一致；z0/1/2
    校验 name 集合与 `buffer_names` 匹配；z3 校验热备 `param_shapes` 与当前
    `_get_zero_param_shapes()` 逐组一致。任一失败 → warning + 走现有磁盘路径。
  - **z0/1/2 命中**：用热备 dict 构造 checkpoint（module/lr_scheduler/
    client_state），复用 `load_module_state_dict`；
    `loaded_checkpoint_dp_world_size` 取自热备 → 权重**零磁盘 I/O**。
  - **z3 命中**：热备 flat fp16 groups + param_shapes 经
    `self.optimizer.unflatten` 恢复出各参数的分区切片，直接拷入
    `param.ds_tensor`（复用 `_load_zero3_fp16_partition_state_dict`
    engine.py:3367 的思路）；buffers 同样从热备拷入。
  - **未命中/失效**：z3 依次尝试 `*_fp16_weights.pt` → fp32 重建（现状）；
    z0/1/2 正常磁盘路径。
- `stage=1` / optimizer 状态：**不动**，仍从磁盘加载（热备只覆盖 weights）。
  与 `async_load=True` 组合：stage=0 weights 走内存热备（零 I/O），stage=1
  optimizer 状态照常后台预取。
- 可选 Phase D：经典 `load_checkpoint` 的 module 部分同样先查热备（optimizer
  状态仍走磁盘）。

### Phase E：验证

1. **负向**：save 后删除/改名 ckpt 文件 → stage=0 仍从内存恢复，weights 逐位
   一致（z0/z2/z3 × fp16/bf16）。
2. **一致性**：hot 恢复 + 再训一步 vs 磁盘 `load_checkpoint` + 再训一步，
   max diff = 0；重复多次。
3. **二次 save**：save(N) → 训练 → save(N+1) → commit，热备指向 N+1
   （与磁盘 N+1 逐位一致，证明"persist 完成后更新指针"生效）。
4. **失效 fallback**：改 DP world size / 改模型结构 / 热备为 None → 自动磁盘路径。
5. **回归**：`tests/unit/runtime/test_ds_config_dict.py`；async_load 两阶段
   测试全过。
6. **性能**：记录 save 路径 D2H 快照耗时、host 内存增量（≈P 字节）；
   casync 引擎下与 hot_weights=False 对比 step 时间。

## cmpckp（datastates-llm）侧需要相应的修改

仓库：/home/work/cmpckp/llm/datastates（engines/engine_state.py 的
`coalition_save`/`split_load` 与 datastates_core .so）。

### v1（DeepSpeed 单独落地，cmpckp 无强制修改）

事实依据：`coalition_save` 在 `checkpoint_engine.save()` 调用内**同步执行**
（CasyncEngine.save → `self.ckpt_engine.coalition_save`），且它对输入
state_dict 只读（`torch.cat`/`from_address` 均为读操作，不修改源 tensor）；
真正的异步只有 C++ `ckpt()` 落盘。因此 DeepSpeed 在 save 同步窗口内自行完成
CPU 快照即可，**不依赖引擎内存的任何生命周期**，引擎分块释放行为不影响热备。

### v2（优化，需 cmpckp 配合）

1. **暴露 CUDA stream（消除关键路径阻塞）**：
   `StateCheckpointEngine` 已有 `self.cuda_stream`（engine_state.py:322），但
   deepspeed wrapper（casync_checkpoint_engine.py）未暴露。建议：
   - wrapper 增加 `get_cuda_stream()`；deepspeed 把热备 D2H clone 挂到该
     stream 上，与压缩/暂存流水线重叠（训练默认 stream 不等待，clone 由
     wait(persist) 前同步点兜底）。
2. **engine 内 hot backup 模式（消除二次 D2H）**：
   `coalition_save` 流水线本身就要把每个 tensor 读到 host 侧，建议新增可选
   参数（如 `hot_backup=True` / config key），在 `_parse_state` 遍历时把
   weights 文件的 tensor **fork 一份未压缩的 pinned host 副本**（可复用
   `PinnedBytePool`，engine_state.py:275），persist 完成后以句柄
   （ptr/size/元数据）形式返回；deepspeed 的 `_promote_hot_weights` 直接
   指向该句柄，不再需要自己的 D2H。
3. **持久 region 豁免（配合上述句柄方案）**：
   - `load()` 开头会调 `self.ckpt_engine.clear_persistent_allocs()`
     （engine_state.py:727），若热备驻留在引擎管理的 host 内存中，**任何后续
     load 都会把它清掉**。需要引擎提供"hot weights 专属持久 region"：独立于
     `clear_persistent_allocs` 与 persist 后清理逻辑，由 deepspeed 显式
     `release()` 释放。
   - 同时 `split_load` 的 `LoaderSmartManager`/`PinnedBytePool` 内存账目
     （engine_state.py:82-141）需把持久 region 计入/排除，避免与加载暂存
     池互相踩踏。
4. **host 内存账目接口**：提供 host 缓存占用查询（当前 `host_cache_size`
   只用于初始化），便于 deepspeed 在 OOM 风险时 warning 或降级。

### 与 cmpckp 现有行为的冲突点（必须遵守）

- 不得让 deepspeed 持有 `sm[version][path]` / `state_manager` 内对象的引用：
  `coalition_save` 结束后这些对象由引擎生命周期管理，persist 完成即释放。
- `commit → wait(persist=True)` 的语义必须继续保证"返回即数据已持久化"，
  这是 hot_weights 指针提升正确性的前提（现状已满足，勿回归）。

## 风险

1. host 内存增加 ≈P 字节（热备副本）；多卡场景注意与 `host_cache_size`
   的叠加预算。
2. save 关键路径同步 D2H：torch 无感；casync/datastates v1 有少量阻塞，
   v2 由 cmpckp stream/hot_backup 消除。
3. z3 分片信息与 DP world size 绑定，world size 变化即失效（自动 fallback，
   正确但慢）。
4. decoupled 引擎下 save→commit 分离：pending 快照生命周期覆盖该窗口；
   多 save 未 commit 只保留最近快照（已注明）。
5. MoE / pipeline / universal 不支持（warning + 降级），后续按需扩展。
6. 与 `async_load` 组合时 stage=0 窗口内仍**禁止** `step()`（fp32 尚未从
   checkpoint 恢复），约束与 async_load.md 一致。
