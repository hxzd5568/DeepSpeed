# hot_weights_v2: 热备执行下沉到 cmpckp 侧（engine 内 fork，DS 只传参数）

> 背景：v1（hot_weights.md）把热备实现放在 DeepSpeed 侧——save 时 DS 自己对
> `state['module']` 做 `detach().cpu()` 快照。这造成：DS 一份阻塞 D2H + engine
> compress/persist 管线再对同一批 tensor 读一遍并压缩落盘，等于"重复 h2d"。
> 且未来一定使用 cmpckp（compcheck）引擎，热备理应复用 engine 已经发生的
> 数据搬运。
>
> v2 核心变化：**hot_backup 的实际执行全部下沉到 cmpckp 侧**，在
> `coalition_save` 的 `_parse_state` 遍历里（compress 决策之前）按"解析字典的
> 方式"把 tensor fork 成 pinned host 副本，用一个 `HotBackupManager` 对象托管；
> DeepSpeed 只在 save/load 时传递参数（`hot_backup=True` / `get_hot_backup()`），
> 不再自己拷贝、不再持有权重副本。

## 结论（可行性）

| ZeRO 阶段 | 可行性 | 前提与说明 |
|-----------|--------|------------|
| 0 / 1 / 2 | ✅ 可行 | 要求 `zero_optimization()`（模型文件 `state['optimizer']` 为 None，见 engine.py:4090）。热备对象 = 整个 `*_model_states.pt` 的 state_dict（module 真实 fp16 权重 + 元数据），约 P 字节 pinned host。 |
| 3 | ✅ 可行 | 模型文件是空占位（见 async_load.md 诊断），真实分区权重在 `*_fp16_weights.pt`（async_load Phase 1+2 布局）。`hot_weights` 与 `async_load` 一样触发该布局；热备对象 = fp16 文件内容 `{module: {...}}`，每 rank 约 P/dp_world_size。 |
| 非 zero（纯 fp32 / 无 zero 混合精度） | ❌ 不支持 | 模型文件里含完整 optimizer 状态（≈7P），热备会膨胀；warning + 降级为普通磁盘路径（与 v1 立场一致）。 |

结论：**zero 1/2/3 可行**。下述详细计划即按此设计。

---

## 一、总体架构

### 1.1 HotBackupManager（cmpckp 侧新类）

```python
class HotBackupManager:
    """托管一次 save 中 fork 出来的 weights 快照，保持其不消亡。"""
    def __init__(self, path):
        self.path = path
        self.tensors = {}        # mapped_key -> CPU tensor（pinned view 或 CPU clone）
        self.structure = None    # "可 restruct 的 dict"：_parse_state 返回的
                                 # lean_state_dict 对象（未 pickle），tensor 位置
                                 # 均被 mapped_key 字符串占位
        self.n_bytes = 0
        self._pool = None        # 专属 PinnedBytePool（Python 侧，不经 C++ 账目）
        self._extra = []         # 超池 pinned alloc 与 CPU clone 的持有者

    def fork(self, mapped_key, data, cuda_stream): ...   # 见 2.3
    def finalize(self, cuda_stream): ...                 # 两阶段拷贝的收尾，见 2.3
    def reconstruct(self) -> dict: ...                   # 见 2.5
    def release(self): ...                               # 显式释放（promote/shutdown 时）
```

生命周期（对应"用一个对象管理让他们不消亡"）：

- `StateCheckpointEngine.hot_backup`：已 promote 的、与最新落盘内容一致的热备对象；
- `StateCheckpointEngine._hot_pending`：本次 save 正在 fork、尚未 persist 的对象；
- **只有第二次 save 的 persist 成功后，`self.hot_backup = self._hot_pending` 重新赋值，
  旧 `HotBackupManager` 引用计数归零 → 其 tensors 与 structure（可 restruct 的 dict）
  才消亡**；persist 失败则旧指针保留（与 v1 语义一致）。

### 1.2 为什么不会被引擎的清理逻辑误杀（关键正确性）

- 热备内存使用 **Python 侧自建的 `PinnedBytePool`**（复用 engine_state.py:275 的类），
  **不经过** `state_manager`/`stage_engine` 的 C++ 账目，因此：
  - `wait(persist=True)` 里 `self.sm.clear()`（engine_state.py:1608）只清 C++ 侧，
    不影响热备；
  - `split_load`/`load` 开头调用的 `self.ckpt_engine.clear_persistent_allocs()`
    （engine_state.py:727/984）同样碰不到 Python pinned 池。
- 引擎的流水线分块释放（`GpuArena.free_oldest`、`batch_commit`）只作用于压缩
  暂存，与 fork 副本无关。
- 这也正是 v1 文档"cmpckp 侧修改"第 3 条（持久 region 豁免）的落地方式：**不向
  C++ 要豁免，直接用 Python 侧专属池**，改动最小且不触碰 C++ 内存账目。

---

## 二、cmpckp 侧详细改造

仓库：/home/work/cmpckp/llm/datastates/engines/engine_state.py（`state_engine`）。

### 2.1 新增成员与 API（`StateCheckpointEngine`）

```python
def __init__(...):
    ...
    self.hot_backup = None            # 已 promote
    self._hot_pending = None          # 本次 save 的 fork 中
    self._hot_capture = False         # 本次 save 是否启用 fork

def save(state_dict, path, hot_backup=False):        # 透传给 save_（可选路径）
def coalition_save(state_dict, path, hot_backup=False): ...

def get_hot_backup(self):             # 返回 self.hot_backup 或 None
def release_hot_backup(self):         # self.hot_backup.release(); self.hot_backup = None
def hot_backup_host_bytes(self):      # 账目查询（v1 文档第 4 条）
```

### 2.2 `coalition_save` 入口改造（engine_state.py:463）

```python
def coalition_save(self, state_dict, path, hot_backup=False):
    self._hot_capture = bool(hot_backup) and 'optim_states' not in path
    if self._hot_capture:
        # 覆盖上一个未 commit 的 pending（多 save 未 commit 只保留最近，与 v1 一致）
        if self._hot_pending is not None:
            self._hot_pending.release()
        self._hot_pending = HotBackupManager(path)
    try:
        ...原逻辑（见 2.3/2.4 的插桩点）...
    except Exception:
        self._hot_pending = None      # 失败不 promote（现有 sys.exit 行为保留）
        raise
    finally:
        self._hot_capture = False
```

防御：`'optim_states' in path` 时忽略 flag（optim 文件 6P，热备它无意义）。

### 2.3 `_parse_state` 插桩：compress 之前的 fork（engine_state.py:488-601）

要点：**mapped_key 的决策提前到 tensor 分支入口**（现分别在 506 与 512 两处
根据同一条件 `data.numel() <= min_size or not data.is_cuda` 生成，只需上提一次），
fork 与磁盘路径共用同一 mapped_key，保证 structure 引用的 key 与热备 tensor 的
key 严格一致。

```python
def _parse_state(key, data):
    ...
    if torch.is_tensor(data) and data.numel() != 0:
        is_small = (data.numel() <= min_size) or (not data.is_cuda)
        mapped_key = f"{'TENSOR' if is_small else 'CTENSOR'}{KEY_SEPARATOR}{key}"
        if self._hot_capture:
            # —— 热备 fork：在 compress 决策之前，先登记再拷贝（两阶段，见下）——
            self._hot_pending.fork(mapped_key, data, self.cuda_stream)
        if is_small:
            ...（原 small/delay_tensor 分支，mapped_key 改为使用上提变量）...
        else:
            ...（原 coalition / chunk 分支，mapped_key 改为使用上提变量）...
        snapshot = mapped_key
    ...
```

fork 采用两阶段（避免逐 tensor `pin_memory()` 的高开销、pool 需要总字节数）：

1. **登记阶段**（parse 遍历中，同步执行，只是存引用）：
   `self._pending_forks.append([mapped_key, data.contiguous() if data.is_cuda else data, data.is_cuda])`，
   累计 CUDA 字节数。
2. **拷贝阶段**（parse 结束后、`ckpt()` 之前，engine_state.py:644 前插入）：
   按累计字节 `PinnedBytePool(total_bytes)` 建池；对每个 CUDA tensor：

   ```python
   self.cuda_stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(self.cuda_stream):
       dst = pool.alloc(nbytes, dtype, shape)
       dst.copy_(src, non_blocking=True)      # 异步 D2H，与 compress/persist 流水线重叠
   self.tensors[mapped_key] = dst
   ```

   CPU tensor 直接 `clone()` 进 `self._extra`。
   单 tensor 超池时走 `PinnedBytePool.alloc` 的既有兜底（engine_state.py:286-290）。

说明：

- fork 在 **compress 决策之前**、位于 parse 遍历内部，因此 coalition / chunk /
  small 三条分支全部覆盖，且与"解析字典的方式组织 tensor"完全同构；
- fork 拷的是 `data.contiguous()` 的结果，与活参数共享 storage 的 view 在
  `coalition_save` 同步窗口内完成取值（与 v1 原则 2 一致，但不再由 DS 做）；
- 不涉及 compress 的 dtype 检查（fork 是纯字节拷贝），bf16 权重同样可 fork
  （engine 对 bf16 的 persist 限制是既有问题，与热备无关）。

### 2.4 promote：`wait(persist=True)` 内提升（engine_state.py:1592）

```python
def wait(self, persist=False, for_all=False):
    ...
    if persist:
        self.sm.clear()
        if self._hot_pending is not None:
            self.cuda_stream.synchronize()      # 兜底：确保所有 fork D2H 完成
            if self.hot_backup is not None:
                self.hot_backup.release()       # 旧热备消亡（"第二次 save 赋新值"）
            self.hot_backup = self._hot_pending
            self._hot_pending = None
```

- 语义与现有不变式绑定："`commit` → `wait(persist=True)` 返回即数据已持久化"，
  热备指针提升与其同点发生（v1 文档"必须遵守"的第 2 条自动满足）；
- `wait(persist=False)` 不提升；`shutdown()`/`__del__` 里加 `release_hot_backup()`。

### 2.5 `reconstruct()`：把"可 restruct 的 dict"还原成 state_dict

复用 `split_load` 中 `_reconstruct_state` 的遍历骨架（engine_state.py:1193-1231），
区别是 TENSOR/CTENSOR 占位符直接从 `self.tensors[mapped_key]` 取（返回 pinned view，
零拷贝），非字符串值原样返回（int/str/`ds_config` 对象/空占位 tensor）：

```python
def reconstruct(self):
    def walk(key, snap):
        if isinstance(snap, str) and snap.startswith(("TENSOR"+KEY_SEPARATOR, "CTENSOR"+KEY_SEPARATOR)):
            return self.tensors[snap]
        if isinstance(snap, list): return [walk(...) for ...]
        if isinstance(snap, dict): return {k: walk(...) for ...}
        return snap
    return walk("", self.structure)
```

structure 在 `coalition_save` 里 `pickle.dumps` 之前（engine_state.py:630）赋值：
`self._hot_pending.structure = lean_state_dict`。

### 2.6 可选：`save_`（非 coalition 路径）同样插桩

`DataStatesCheckpointEngine` wrapper 走 `save` → `save_`（engine_state.py:663，全
TENSOR、无压缩）。fork 逻辑相同且更简单（只有一条分支）。v2 优先落地
`coalition_save`（用户目标引擎），`save_` 列入 Phase 6（可选）。

---

## 三、DeepSpeed 侧详细改造（只传参数）

仓库：/home/work/softwares/DeepSpeed。

### 3.1 config（deepspeed/datastates/config.py）

```python
HOT_WEIGHTS = "hot_weights"          # 默认 False
self.hot_weights = param_dict.get("hot_weights", False)
if isinstance(self.config, dict) and HOT_WEIGHTS in self.config:
    self.hot_weights = self.config.get(HOT_WEIGHTS, self.hot_weights)
```

（与 `async_load` 同套路，顶层 key 或 `datastates_ckpt`/`casync_ckpt` 内均可。）

### 3.2 engine 基类能力位（deepspeed/runtime/checkpoint_engine/checkpoint_engine.py）

```python
def supports_hot_backup(self):   # 默认 False；datastates/casync wrapper 返回 True
    return False
def get_hot_backup(self):        # 默认 None，供 load 侧统一探测
    return None
```

wrapper（casync_checkpoint_engine.py / datastates_checkpoint_engine.py）：

```python
def save(self, state_dict, path, hot_backup=False):
    self.ckpt_engine.coalition_save(state_dict, path, hot_backup=hot_backup)
def get_hot_backup(self):  return self.ckpt_engine.get_hot_backup()
def release_hot_backup(self): self.ckpt_engine.release_hot_backup()
def supports_hot_backup(self): return True
```

### 3.3 save 侧（deepspeed/runtime/engine.py）

`_hot_backup_requested()`（新 helper）：

```python
def _hot_backup_requested(self):
    cfg = getattr(self._config, 'datastates_config', None)
    if not bool(getattr(cfg, 'hot_weights', False)): return False
    if not self.zero_optimization(): return False          # 非 zero 不支持（见结论表）
    if self.has_moe_layers or isinstance(self.module, PipelineModule): return False
    if self.load_universal_checkpoint() or self.zero_nvme_offload_optimizer(): return False
    return bool(getattr(self.checkpoint_engine, 'supports_hot_backup', lambda: False)())
```

**z0/1/2**（`_save_checkpoint`，engine.py:4112-4113）：

```python
if self.save_non_zero_checkpoint:
    if self._hot_backup_requested():
        self.checkpoint_engine.save(state_dict=state, path=save_path, hot_backup=True)
    else:
        self.checkpoint_engine.save(state_dict=state, path=save_path)
```

**z3**（engine.py:4120-4127）：把 `_zero3_async_layout_enabled()` 泛化为
`_zero3_fp16_layout_enabled()`（`async_load or hot_weights` 且 zero3 时写
`*_fp16_weights.pt`），并对该文件的 save 传 `hot_backup=True`。fp16 文件内容
`{module: ...}` 即热备内容；`*_model_states.pt`（空占位）与 `*_optim_states.pt`
不传 flag。

> 兼容性：`hot_weights=True` 但 `async_load=False` 时也会多写一个 fp16 文件
> （约 P/dp），与 async_load.md 风险 5 相同量级；旧 checkpoint（无 fp16 文件）
> 下热备自然不存在 → 自动走磁盘 fallback。

promote 无需 DS 侧代码：engine 在 `wait(persist=True)`（decoupled 引擎的
`_commit_decoupled_checkpoint`，engine.py:3901，于 step 的 gradient-accumulation
boundary 触发，engine.py:2521-2522）内自行提升。

### 3.4 load 侧（deepspeed/runtime/engine.py）

**z0/1/2**（`_load_checkpoint` 开头，engine.py:3542-3548）：

```python
hot_checkpoint = self._try_load_hot_checkpoint(load_optimizer_states)
if hot_checkpoint is not None:
    load_path = self._get_ckpt_name(load_dir, tag)   # 合成值，仅作返回
    checkpoint = hot_checkpoint
else:
    load_path, checkpoint, _ = sd_loader.load(...)
```

`_try_load_hot_checkpoint`：

```python
def _try_load_hot_checkpoint(self, load_optimizer_states):
    if load_optimizer_states or not self._hot_backup_requested(): return None
    if self.zero_optimization_partition_weights(): return None   # z3 走下方专线
    hb = self.checkpoint_engine.get_hot_backup()
    if hb is None: return None
    try: ckpt = hb.reconstruct()
    except Exception: warn; return None
    if ckpt.get('dp_world_size') != self.seq_dp_world_size \
       or ckpt.get('mp_world_size') != self.mp_world_size: warn; return None
    # 结构校验：module 名集合、buffer_names 与当前模型一致，否则 None
    return ckpt
```

命中后即走既有 `load_module_state_dict` / `client_state` / `lr_scheduler` 逻辑
（engine.py:3595 起），**权重零磁盘 I/O**；stage=1 / optimizer 状态照旧从磁盘
（与 `async_load=True` 组合时 stage=1 后台预取不变）。

**z3**（engine.py:3561-3572 的 fast-path 分支内）：仅替换 fp16 来源，其余（
`dp_world_size` 校验来自磁盘 model 文件、`_load_zero3_fp16_partition_state_dict`
拷贝）全部复用：

```python
fp16_weights = self._try_hot_zero3_fp16_weights()
if fp16_weights is None:
    fp16_weights = self._try_load_zero3_fp16_weights(load_dir, tag)
```

`_try_hot_zero3_fp16_weights`：`get_hot_backup()` → `reconstruct()` → 取
`d['module']`；逐 param 校验 `name in module` 且 `module[name].numel() ==
param.ds_tensor.numel()`（DP world size 变化 → 分片 numel 不匹配 → 自动 fallback
fp16 文件 → fp32 重建），返回 `{'module': module}` 或 None。命中后
`fp16_partition_load=True` → 直接拷入 `param.ds_tensor`，无集合通信、无磁盘 I/O。

> z3 的 stage=0 仍需读一次磁盘 `*_model_states.pt`（~0 字节，用于
> dp_world_size 等元数据），省下的是 fp16 权重文件（P/dp）的读取——收益主体。
> 完全免磁盘可后续把 `dp_world_size/mp_world_size` 标量并入 fp16 文件（可选优化）。

---

## 四、时序正确性论证（fork 与训练并发）

1. `save_checkpoint` 在训练循环同步执行；`coalition_save` 内 fork 的 CUDA 拷贝
   挂在 `self.cuda_stream`，且 `fork` 前 `cuda_stream.wait_stream(current_stream())`，
   保证读到的是上一次 optimizer step 落定后的权重。
2. decoupled 引擎的 `commit`（→ `wait(persist=True)`）发生在 step 的 boundary、
   `_take_model_step` **之前**（engine.py:2521-2522 → 2537）；`wait(persist=True)`
   内 `cuda_stream.synchronize()` 保证 fork 在权重被下一次 step 改写前完成。
   forward/backward 不改写 fp16 权重（z3 gather 写临时全量 buffer，不动
   `ds_tensor`），因此 save→commit 窗口内源数据稳定。
3. 后台线程只做 I/O 的 async_load 约束与本方案无关：fork 全部发生在主线程的
   `coalition_save` 调用内，仅 D2H 完成是异步的（由上述同步点兜底）。

## 五、分阶段落地计划

| Phase | 内容 | 交付物 |
|-------|------|--------|
| 1（cmpckp） | `HotBackupManager`（fork 两阶段、reconstruct、release）；`coalition_save` 插桩（mapped_key 上提、fork、structure 保存）；`wait(persist=True)` promote；`get_hot_backup/release_hot_backup/hot_backup_host_bytes` | cmpckp 单测：save(flag=True) → wait → reconstruct 与源逐位一致；第二次 save 后旧对象消亡（弱引用/内存观测） |
| 2（DS） | config `hot_weights`；wrapper 透传 + `supports_hot_backup`；base 默认位 | test_ds_config_dict 回归 |
| 3（DS save） | `_hot_backup_requested()`；z0/1/2 model 文件 save 传参；`_zero3_fp16_layout_enabled()` 泛化 + fp16 文件 save 传参 | z0/z1/z2 save 后 engine 内热备存在、字节数 = P |
| 4（DS load） | `_try_load_hot_checkpoint`（z0/1/2 全旁路）；`_try_hot_zero3_fp16_weights`（z3 专线） | 负向：删磁盘文件后 stage=0 仍从热备恢复 |
| 5（验证） | 见下"验证计划" | 全绿 |
| 6（可选） | `save_` 路径 fork；z3 fp16 文件并入 dp/mp 标量实现完全零磁盘；C++ 侧 retain 模式（省 fork 字节数，v3 方向） | — |

## 六、验证计划

1. **负向**：save → 删除/改名 ckpt 文件 → `load_checkpoint_stage(stage=0)` 从热备
   恢复，weights 逐位一致（z0/z1/z2/z3 × fp16/bf16）。
2. **一致性**：热备 stage=0 + 再训一步 vs 磁盘 `load_checkpoint` + 再训一步，
   max diff = 0；与 `async_load=True` 组合时 stage=1 语义不变。
3. **二次 save**：save(N) → 训练 → save(N+1) → commit，热备与磁盘 N+1 逐位一致；
   旧 HotBackupManager 已消亡（host 内存回落）。
4. **失效 fallback**：DP world size 变化 / 模型结构变化 / 热备 None / 非 zero
   配置 / MoE、pipeline、universal → 自动磁盘路径，无异常。
5. **回归**：async_load 两阶段全测、`tests/unit/runtime/test_ds_config_dict.py`。
6. **性能**：save 关键路径不再有 default-stream 同步 D2H；fork 与 compress/persist
   的重叠（记录 coalition_save 耗时分项）；host 内存增量 ≈P（z0/1/2 每 rank）或
   P/dp（z3 每 rank）。

## 七、与 v1 的差异对比

| 维度 | v1（DS 侧快照） | v2（cmpckp 侧 fork） |
|------|----------------|----------------------|
| 热备执行者 | DeepSpeed（`_save_checkpoint` 内 `detach().cpu()`） | cmpckp `coalition_save` 的 `_parse_state`（compress 前 fork） |
| DS 角色 | 拷贝 + 持有 `self._hot_weights` | 只传 `hot_backup=True` / `get_hot_backup()`，无副本 |
| D2H | 阻塞 default-stream 同步拷贝（关键路径），随后 engine 再读一遍压缩 | engine 内一次 fork，挂 cuda_stream 异步重叠，pinned 单份 |
| 组织方式 | DS 自建 dict | 与 parse 字典同构：mapped_key 扁平 dict + lean_state_dict（可 restruct 的 dict） |
| 生命周期 | `self._hot_weights`/`_pending` 双指针 | engine 内 `hot_backup`/`_hot_pending`，第二次 save 赋新值旧对象消亡 |
| promote 点 | `commit` 成功后 DS 显式调用 | `wait(persist=True)` 内自动提升（与"返回即已持久化"绑定） |
| torch engine | 支持 | 不支持（要求 cmpckp，`supports_hot_backup()` 门槛） |

## 八、风险

1. host 内存 ≈P 字节（z0/1/2 每 rank 全量；z3 每 rank P/dp），与 `host_cache_size`
   叠加预算；`hot_backup_host_bytes()` 供监控，超预算时 warning/降级。
2. fork 的 P 字节 D2H 与压缩后落盘流量并存（P + ~P/4）；省掉的是"第二次遍历、
   第二份 host 副本、default-stream 阻塞"，彻底省字节需 C++ retain 模式（v3）。
3. z3 分片与 DP world size 绑定：变化即失效 → 自动 fallback（正确但慢）。
4. decoupled 下 save→commit 窗口：多 save 未 commit 只保留最近 pending；
   persist 失败（engine 现有 sys.exit 行为）不提升、旧热备保留。
5. `hot_weights=True` 但 `async_load=False` 时 z3 会额外落盘 fp16 文件（与
   async_load 风险 5 一致）；stage=0 窗口内仍禁止 `step()`（约束不变）。
6. `StateCheckpointEngineAggregated`/`engine_state_h100` 不在本次范围（同构可移植）。
