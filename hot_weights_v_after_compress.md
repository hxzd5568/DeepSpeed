# hot_weights_v_after_compress: C++ 侧压缩后热备（单机共享段，已实测）

> 背景：v2（hot_weights_v2.md，已实现）在 `coalition_save` 的 `_parse_state` 里
> fork 一份 **未压缩** 的 pinned 副本（P 字节 raw D2H）。实测分析发现该 fork 的
> D2H 与引擎自身的压缩数据 D2H（`self.ckpt_engine.ckpt(...)` 触发，压缩后 ~P/4
> 字节经 `ctensor_stage_stream`/`flush_stream` 拷入 host pinned）**在 PCIe 上直接
> 竞争**，抵消了 casync 的收益。
>
> 本方案（v3）彻底消除竞争：**热备不再独立 fork，而是把 save 流水线本身搬到
> host 的压缩字节滞留于一块共享内存段（/dev/shm，shm_open + MAP_SHARED）**。
> 单机下每个 rank 映射同一段：保存 rank（z0/1/2 为 DP rank 0）在写完成回调里把
> 压缩字节 memcpy 进段内并维护 key 表，`split_load` 时任何 rank 都优先从该段
> 取字节（`torch.frombuffer` 直读共享映射），之后流程（GPU 解压 → copy 到
> target）**完全不变**。因为压缩/解压带宽是 H2D 的 ~10 倍，load 侧"共享段取
> 压缩字节 + GPU 解压"没有效率问题；save 侧**零额外 D2H**（唯一新增开销是写
> 完成后的 host→host memcpy，走内存带宽）。

## 一、可行性结论

- **zeRO 0/1/2**：热备对象 = `*_model_states.pt` 的压缩内容（P/4 量级）**加上
  恢复元数据**（`datastates_metadata`/`coalition_map`/`chunk_map` pickle 与
  header），load 完全零磁盘 I/O（文件删除后仍可恢复）。
- **zeRO 3**：热备对象 = `*_fp16_weights.pt` 的压缩内容（P/4/dp 每 rank），
  继续依赖 `_zero3_fp16_layout_enabled`（async_load/hot_weights 触发）布局。
- **非 zero / MoE / pipeline / universal**：与 v2 相同，warning + 降级。
- 恢复准确性：见第 5 节专项论证——字节一致、promote 原子性、path/version 匹配、
  既有校验全部继承，**无新增解析路径**。

## 二、现有 C++ 流水线（已通读仓库，插入点依据）

### 2.1 save 侧数据流

```
coalition_save (engine_state.py)
 ├─ 大 tensor: compress on GPU → CTensor → sm.add_var(compress_data, key)   # GPU tier
 │     └─ self.ckpt_engine.batch_commit(version, sm, path)                  # 逐 coal/chunk
 │         └─ state_io_engine_impl_t::batch_commit          (state_io_engine_impl.cpp:46)
 │             └─ state->get_next_ctensor(GPU_TIER, m)      (state_manager.cpp:162)
 │                 └─ core_engine->batch_region(m)          (core_impl.cpp:69)
 │                     └─ gpu_tier_t::flush_batch(m)        (gpu_tier.cpp:48)
 │                         └─ flush_batch_io_ 线程:          (gpu_tier.cpp:97)
 │                            ① successor_tier_->mem_pool->allocate(dest)   # host ring pool
 │                            ② cudaMemcpyAsync D2H (ctensor_stage_stream)
 │                            ③ host_tier_t::flush(dest)    → flush_q
 ├─ 小/CPU tensor: delay_tensor → sm.add_var(data, key) → ckpt(version, sm, path)
 │     └─ state_io_engine_impl_t::ckpt                     (state_io_engine_impl.cpp:17)
 │         └─ tier 循环: GPU tier 走 gpu_tier_t::flush (flush_q → flush_io_ 线程, gpu_tier.cpp:74)
 │                       HOST tier 走 host_tier->mem_pool->allocate + memcpy (core_impl.cpp:54)
 └─ host_tier_t::flush_io_ 线程 (host_tier.cpp:84)
     └─ file_handler->write(m)   # io_uring 异步写 (io_uring_handler.cpp:27) 或 pwrite (pwrite_handler.cpp:24)
         └─ 写完成回调: chunk_counter==0 → mem_pool->deallocate(m)   ← **热备插入点**
             (io_uring_handler.cpp:154-159 / pwrite_handler.cpp:47)
```

要点：
- 压缩字节 D2H 到 host ring pool（`host_tier_t`，size = `host_cache_size`，默认 1GB），
  写盘完成后立即 `deallocate` 回收到 ring。
- `wait_batch_stage`（Python 侧 arena 回收依赖，engine_state.py:544）只等
  `flush_bc_q.wait_item`（D2H 完成），**不等写盘** → 在写完成回调里做热备拷贝
  不会阻塞压缩/arena 流水线。

### 2.2 load 侧数据流

```
split_load (engine_state.py:982)
 └─ header = self.ckpt_engine.restore(version, path)   (state_io_engine_impl.cpp:103)
     ├─ 读 header_begin_offset + header（2 次小读，元数据）
     └─ 对 header 中每个 key：posix_memalign → persistent_allocs.push_back(buf)
         → core_engine->restore_region(m) → host_tier fetch → io_uring read  ← **热备旁路点**
     └─ 返回 JSON: {key: {ptr, size, offsets, dtype, shape}}
 └─ Python: 对 TENSOR/CTENSOR key 做 frombuffer(ptr).to(device) → decompress → copy target
```

要点：`restore` 对**每个 key** 都发盘读并返回 host 指针；Python 侧之后只按
JSON 指针取字节。把某几个 key 的"数据来源"从盘换成热区指针即可，**Python 流程
零改动**（"之后的流程不变"正是此意）。

## 三、C++ 侧设计

### 3.1 `mem_region_t` 扩展（include/common/mem_region.hpp）

```cpp
struct mem_region_t {
    ...
    bool        hot = false;        // 该 region 属于热备 save（仅 GPU tier 打标）
    std::string key;                // state provider 的 key（"coal0"/"chunk1"/"TENSOR|module|w"）
    ...
};
```

- 两个拷贝构造（`const mem_region_t*` / `shared_ptr` 版本）需同步拷贝 `hot/key`，
  保证 D2H 后 `dest`（gpu_tier flush 线程内新建）仍携带标记。
- `key` 在 `state_provider_t::get_next_chunk/get_next_ctensor` 填充
  （state_provider.cpp:138/155 处 `dest->key = data_key;`）。

### 3.2 `hot_weights_manager_t`（src/hot_weights_manager.{hpp,cpp}，已实现）

```cpp
class hot_weights_manager_t {
    struct entry { size_t offset; size_t size; size_t file_start_offset; };
    char*       buffer_ = nullptr;       // 固定 pinned 区域（posix_memalign + cudaHostRegister）
    size_t      capacity_ = 0;           // = 该文件未压缩字节总和（最坏上界，由 Python 传入）
    size_t      used_ = 0;               // 本次 save 已存压缩字节
    std::mutex  mtx_;                    // store(io_uring 完成线程) 与 begin/promote/lookup(主线程) 互斥
    std::map<std::string, entry> pending_map_;  // 本次 save 填充中
    std::map<std::string, entry> active_map_;   // 上次 persist 完成后 promote
    std::string pending_path_, active_path_;
    uint64_t    pending_version_ = 0, active_version_ = 0;
    bool        hot_valid_ = false;      // 仅 promote 后为 true；save 期间 false
public:
    void begin_save(uint64_t version, std::string path, size_t capacity);
        // ensure_capacity(capacity)（首次分配，其后复用；新容量≤旧容量直接复用）
        // hot_valid_ = false; used_ = 0; pending_map_.clear(); 记录 path/version
    void store(const std::shared_ptr<mem_region_t>& m);
        // 仅当 m->hot 时被调用：若 used_+m->size > capacity_ → 标记本次热备失败
        //   （hot_valid_ 保持 false，磁盘仍完整，warning）
        // memcpy(buffer_+used_, m->ptr, m->size);
        // pending_map_[m->key] = {used_, m->size, m->file_start_offset}; used_ += m->size;
    void promote();
        // 由 wait(persist=true) 完成路径调用：active_map_ = std::move(pending_map_);
        // active_path_/version_ = pending 值; hot_valid_ = true;
        // （语义 = "第二次 save 时管理对象被赋新值，旧 active_map_ 随之消亡"）
    bool lookup(uint64_t version, const std::string& path, const std::string& key,
                char*& ptr, size_t& size);
        // hot_valid_ && version==active_version_ && path==active_path_
        // && key in active_map_ → 返回 buffer_+offset/size
    void release();                     // shutdown 时释放
    size_t capacity() / used() / host_bytes();   // 供 Python 日志/账目
};
```

关键语义（共享段实现，已落地）：

1. **共享段**：`shm_open("/ds_hot_<fnv1a(path)>") + mmap(MAP_SHARED)`。布局：
   `[4KB 段头][数据区（压缩字节，按写完成顺序）][key 表（二进制序列化）]`。
   段头：magic / state(PENDING|COMMITTED) / version / generation /
   used/capacity/table 字节数 / **检查点文件身份(dev,ino,mtime_ns)** / path。
   写端（DP rank 0）写入，所有 rank（含从未 begin_hot_save 的读端）在
   `restore()` 里**惰性打开**同一段直接读取；跨节点 shm_open 失败 → 自动回磁盘。
2. **"只有第二次 save 时管理对象被赋新值，旧数据才消亡"**：
   - `begin_save`（第二次 save 开始）→ 段头置 PENDING（所有读端 lookup 立即
     miss → 回磁盘，杜绝混合代数据）；本进程同时 unlink 上一代 tag 的段；
   - `promote`（persist 完成）→ 写 key 表 + 收缩段到实际大小 + 原子置
     COMMITTED（release store），读端以 acquire 读头判定可见性；
   - persist 失败（FATAL 退出）→ 不 promote → 段保持 PENDING/INVALID，磁盘兜底。
3. **跨运行残留段防护**：段头记录检查点文件身份（dev/ino/mtime_ns），读端每次
   lookup stat 文件比对，文件被重写即判定残留 → 回磁盘。
4. **生命周期**：创建者（写端）在 shutdown unlink；`/dev/shm` 占用 = 收缩后的
   实际大小（压缩权重 + 表 + 4KB，实测 1.8B 模型 ≈1.38GB）。

### 3.3 `state_io_engine_impl_t` 接线（src/state_io_engine_impl.{hpp,cpp}）

```cpp
// 成员：std::shared_ptr<hot_weights_manager_t> hot_mgr;

// nanobind 暴露（nb_datastates_core.cpp）：
//   .def("begin_hot_save", &state_io_engine_impl_t::begin_hot_save, "version"_a, "path"_a, "capacity"_a)
//   .def("hot_host_bytes", &state_io_engine_impl_t::hot_host_bytes)

void state_io_engine_impl_t::begin_hot_save(version, path, capacity) {
    if (!hot_mgr) hot_mgr = std::make_shared<hot_weights_manager_t>();
    hot_mgr->begin_save(version, path, capacity);
}

void state_io_engine_impl_t::batch_commit(version, state, path) {
    ...
    state->get_next_ctensor(tier, m);          // m->key 在此填充
    if (hot_mgr && hot_mgr->is_pending_path(path)) m->hot = true;   // CTensor 恒为 GPU tier
    core_engine->batch_region(m);
}

void state_io_engine_impl_t::ckpt(version, state, path) {
    for (tier : {GPU, HOST_UNPINNED, HOST_PINNED}) {
        while (state->has_next_chunk(tier)) {
            ... get_next_chunk(tier, m);
            if (tier == GPU_TIER && hot_mgr && hot_mgr->is_pending_path(path)) m->hot = true;
            core_engine->ckpt_region(m);       // 仅 GPU tier 的 m->hot 会随拷贝构造传到写完成点
        }
    }
    ... header 两次写（同样打 hot 标，key 用合成名 "datastates_header_off" /
        "datastates_header"）...

    // ckpt() 开头对 hot save 自动扩容：hot_mgr->extend_capacity(
    //     state->get_file_offset() + sizeof(size_t) + state->get_state_meta().size()
    //     + get_fs_block_alignment())
    // —— 覆盖元数据 pickle 与 header，容量单调增长，二次 save 复用。
}

void state_io_engine_impl_t::wait(state, persist) {
    core_engine->wait(persist);
    if (persist) {
        state->release();
        if (hot_mgr && hot_mgr->has_pending()) hot_mgr->promote();   // ← promote 点
    }
}
```

注意：
- `wait(persist=true)` 的 `core_engine->wait(true)` 已等待 host_tier 全部写完成
  （core_impl.cpp:129-143），故 promote 时热区字节已完整；
- **全部 tier 打标**：GPU tier（权重张量）与 HOST tier（`datastates_metadata`/
  `coalition_map`/`chunk_map` pickle）都进热区；header 两段（`datastates_header_off`/
  `datastates_header`）同样打标。因为元数据在 load 侧被最先解析（split_load 先
  pickle.loads 出 lean_state_dict / coalition_map / chunk_map），热备它才能让
  weights 文件做到完全零磁盘 I/O。

### 3.4 写完成回调里 store（替换 deallocate 顺序）

io_uring（io_uring_handler.cpp:154-159）：

```cpp
if (chunk_counter[info.mem_region->internal_uid] == 0) {
    chunk_counter.erase(...);
    if (info.mem_region->hot) hot_mgr->store(info.mem_region);   // ← 先拷贝进热区
    mem_pool->deallocate(info.mem_region);                       // ← 再回收 ring
    perf_profiler.record_event(...);
}
```

pwrite（pwrite_handler.cpp:47）同样在 `mem_pool->deallocate(m)` 前插入。
文件处理器需要拿到 `hot_mgr` 引用（构造时传入，或经 base_file_handler 持有
`std::function<void(mem_region_t&)>` 回调，推荐回调方式以降低耦合）。

为什么放这里（而非 D2H 线程）：
- `wait_batch_stage`（arena 回收）只等 D2H 完成，写完成回调不在压缩关键路径上；
- store 是 host→host memcpy（内存带宽），与磁盘写在 io_uring 完成线程中天然重叠；
- 若担心 memcpy 延迟 ring 回收（大模型 P/4 > ring 1GB 时依赖回收），可把 store
  丢进独立线程队列（store 期间 region 由 shared_ptr 保活），列为可选优化。

### 3.5 `restore` 旁路（state_io_engine_impl.cpp:103，load 侧唯一改动点）

```cpp
restore 开头：header_begin_offset 与 header string 也先 `lookup("datastates_header_off"/
"datastates_header")`，命中则完全跳过文件访问（文件被删也能恢复）；未命中走原磁盘路径。

for (每个 key in header_json) {
    ...
    char* hot_ptr; size_t hot_size;
    if (hot_mgr && hot_mgr->lookup(version, path, key, hot_ptr, hot_size)
        && hot_size == data_size /* 与 meta offsets 一致 */) {
        entry["ptr"] = (uintptr_t)hot_ptr; entry["size"] = hot_size;
        entry["offsets"] = meta["offsets"]; /* dtype/shape 照抄 */
        out_json[key] = std::move(entry);
        continue;                            // 不 posix_memalign、不发盘读
    }
    ... 原有 posix_memalign + restore_region + persistent_allocs ...
}
```

- 热区指针**不**进 `persistent_allocs` → `clear_persistent_allocs()`
  （下次 load 开头调用）不会误 free；热区生命周期完全由 `hot_weights_manager_t` 管。
- header 两次小读（offset+header string）同样优先热区（文件被删也可恢复），
  未命中才走磁盘；无论命中与否，解析逻辑完全一致。
- 命中后 Python 侧 `frombuffer(ptr).to(device)` → decompress → split/copy 到
  target：与磁盘路径逐行相同。

## 四、Python 侧改动

### 4.1 cmpckp `engine_state.py`

```python
def coalition_save(self, state_dict, path, hot_backup=False):
    hot = bool(hot_backup) and 'optim_states' not in path
    ...
    version = get_checkpoint_version(path, self.last_ckpt_version)
    ...
    if hot:
        capacity = self._state_dict_bytes(state_dict)   # 递归求和 tensor nbytes（未压缩）
        self.ckpt_engine.begin_hot_save(version, path, capacity)
    ... 原流程（batch_commit / ckpt）不变 ...
```

- 删除 v2 的 `HotBackupManager` Python 类、`register_fork`/`finalize` 插桩、
  `wait` 里的 Python promote、`get_hot_backup/release_hot_backup`（或改为薄封装
  指向 C++ 的 `hot_host_bytes`）。
- `save_` 路径同法（可选）。

### 4.2 DeepSpeed 侧

- **save 侧不变**（v2 已落地的部分保留）：
  - `_hot_backup_requested()` 门槛、`checkpoint_engine.save(..., hot_backup=True)`
    传参、`_zero3_fp16_layout_enabled()`（async_load or hot_weights）。
- **`hot_weights` 开启时 `load_checkpoint` 默认进入两阶段路径**（新增）：
  `_hot_weights_default_two_stage()` 命中（config 开 + 引擎支持 + 非
  load_module_only）时，`load_checkpoint` 内部等价转调
  `load_checkpoint_stage(stage=0)` + `wait_for_optimizer_states()`——stage0 权重
  从热区瞬时恢复，optimizer 状态照旧（`async_load=True` 时后台预取），对调用方
  完全透明，物尽其用。
- **load 侧回退 v2 的旁路**（C++ 已透明化）：
  - 删除 `_try_load_hot_checkpoint` / `_hot_weights_structure_matches` /
    `_try_hot_zero3_fp16_weights`，以及 `_load_checkpoint` 开头的 hot 旁路与
    z3 分支里 `_try_hot_zero3_fp16_weights` 的调用；
  - load 代码回到"async_load 原始路径 + fp16 文件 fast path"，热命中完全发生在
    `checkpoint_engine.load` 内部（C++ restore lookup）。
- wrapper（casync/datastates）：`save(..., hot_backup=...)` 保留；`get_hot_backup`
  可删（DS 不再调用）或保留为 C++ 账目查询。

## 五、恢复准确性论证（重点）

1. **字节级一致**：热区字节与磁盘字节来自**同一份 D2H 结果**（同一个 ring pool
   区域，先写盘、写完成后同一指针 memcpy 入热区），load 从热区取字节再走**与
   磁盘完全相同的**解压/拷贝代码 → 逐位一致（可验证：热恢复 vs 磁盘恢复 max diff=0）。
2. **可见性/原子性**：`hot_valid_` 仅在 `wait(persist=true)`（写盘+fsync 完成后）
   promote 时置位；`begin_save` 立即置 false。任何时刻 lookup 要么命中"完整一代"
   的数据，要么回磁盘——**不存在混合代读取**。
3. **path/version 匹配**：lookup 要求 `version==active_version_ && path==
   active_path_`。加载其他 tag / 其他目录（含 symlink/copy）的 checkpoint 时
   自动回磁盘；DS 侧 tag 解析（latest 文件）不受影响。
4. **元数据同样热备、解析逻辑不变**：header JSON（offsets/shape/dtype）、
   `datastates_metadata`（lean_state_dict 结构 pickle）、`coalition_map`、
   `chunk_map` pickle 与 tensor 字节一起滞留热区，load 完全零磁盘 I/O；热区字节
   与磁盘字节逐位相同（同一 buffer、同一写路径），所有布局解析、coalition 拆分、
   chunk 重组逻辑**零新增路径**，split_load 流程原样。
5. **既有 DS 校验全部继承**：dp_world_size/mp_world_size 校验、z3 分区 numel
   校验、`loaded_checkpoint_dp_world_size`、strict load 等与磁盘路径逐条相同
   （因为 DS 侧看到的 dict 与磁盘加载产物一模一样）。DP world size 变化 →
   校验失败 → 报错/回退行为与现状一致。
6. **生命周期安全**：热区独立于 ring pool / `persistent_allocs` /
   `clear_persistent_allocs` / `LoaderSmartManager` 的 pinned_pool；
   `shutdown()` 释放。多 save 未 commit（decoupled）时 pending 覆盖语义与
   commit 一致："最后完成 persist 的 hot save" 胜出。
7. **并发**：store 发生在 io_uring 完成线程（与主线程通过 manager mutex 同步）；
   load 与 save 的串行假设与现状相同（训练中不会边 save 边 load 同一 engine）。
8. **性能正确性**：load 侧省去磁盘读（P/4 字节），改为 pinned host→GPU 传输 +
   GPU 解压；由于压缩/解压带宽 ≈ 10× H2D，P/4 的 H2D 量级 + 高速解压，无效率
   问题（用户论断成立）。save 侧零额外 D2H，新增开销仅为写完成后 host memcpy
   （P/4，内存带宽，与磁盘写重叠）。

## 六、内存/带宽账目

| 项 | v2（Python fork） | v3（C++ 压缩后热备） |
|---|---|---|
| save 侧额外 D2H | P（与引擎 D2H 竞争） | **0** |
| save 侧新增 host 流量 | P（raw） | P/4（host memcpy，内存带宽，写完成线程） |
| host 常驻 | P pinned | 容量 P（上界），实际占用 P/4 |
| load 权重来源 | 磁盘（stage=0） | 热区字节，省 P/4 磁盘读，GPU 解压后 copy |
| 对压缩/arena 关键路径 | 竞争 PCIe | 无（store 在写完成回调） |

## 七、分阶段落地

| Phase | 内容 |
|-------|------|
| C1 | C++ `hot_weights_manager_t` + 单测（store/lookup/promote/容量/失效语义，纯 CPU 可测） |
| C2 | mem_region_t 扩展（hot/key 及拷贝构造）、provider 填 key、batch_commit/ckpt 打标 |
| C3 | 写完成回调 store（io_uring + pwrite）；restore lookup 旁路；nanobind `begin_hot_save`/`hot_host_bytes`；wait 内 promote |
| C4 | cmpckp Python：coalition_save 计算容量并 begin_hot_save；删除 v2 Python fork |
| C5 | DS：load 侧回退 v2 旁路（保留 save 侧）；回归 + 集群验证 |

## 七·五、hot vs non-hot 对比测试（ddp1/hot_bench/）

仿照 ddp1/two_stage/ 新建 `ddp1/hot_bench/`：
- `train_hot_bench.py`：`--hot 0/1` 对比脚本（compcheck 引擎，默认
  `async_load=1`）；save 后显式 `_commit_decoupled_checkpoint()` 触发 persist +
  热区 promote；对同一 checkpoint 测 classic 加载与两阶段（stage0+wait）加载，
  记录 classic_s / stage0_s / stage1_s / two_stage_sum_s / max_weight_diff；
- `run_hot_bench.sh`：逐 stage 跑 nonhot 与 hot 两遍；
- `compare_hot.py`：合并 CSV，输出 nonhot vs hot 的 classic/stage0/stage1/合计
  与加速比。
- 预期：hot 的 stage0（权重可恢复时间）≈ 热区 H2D + 解压，远小于 nonhot 的
  磁盘读；hot 的 classic（默认两阶段）同样受益。

## 八、验证（已实测：4 卡 A/B，见 ddp1/hot_bench/results/stall_ab_report.md）

简化方案：**单脚本两次 4 卡运行**（`ddp1/hot_bench/hot_stall_bench.py`，
`--hot 1` / `--hot 0`，各自一个 4 卡进程，默认 master port）：

```
train 10 steps -> save_checkpoint -> wait(True) 落盘确认 + hot promote
-> 再 train 10 steps -> drop page cache + 冷启动 GPU optimizer state
-> 计时 load_checkpoint(tag)（barrier 后取 max = load 导致的 GPU 等待）
-> 再 train 2 steps 验证
```

实测（compcheck, Qwen1.5-1.8B, ZeRO-2, 4×4090, grad_accum=16，与 two_stage
参考同配置；窗口 = 16 个真实 fwd/bwd 微批）：

| 指标 | hot=1 | hot=0 |
|------|-------|-------|
| classic 同步墙钟 (max) | 3.454s | 3.356s |
| stage0 权重就绪 (max) | **0.525s** | 1.271s |
| 首个 step 残余 wait (call) | ~0.002s | ~0.002s |
| wait+drain（apply 排空，max） | 0.532s | 0.534s |
| **experienced GPU stall (stage0+wait+drain)** | **1.058s** | **1.805s** |

**hot 比 nonhot 少 0.75s（~41%）**，全部来自权重阶段（0.53s 共享段直读，4 卡
全命中 vs 1.27s 磁盘）；optimizer 加载在 16-micro 窗口内完全隐藏（call ~2ms，
剩余 ~0.35-0.53s 为主线程 apply 的 GPU 排空，两变体相同）。测量说明：stage0
后无 barrier（fwd/bwd 梯度累加无需跨卡同步；barrier 的 NCCL all-reduce 会与
async 后台预取线程的 GPU 流水线串行化、虚增 ~2s）。注意 ga=1 时窗口 backward
会崩于 `independent_gradient_partition_epilogue` 的 `zip(None, ...)`（ga=1 下
每个 backward 都是边界），与参考一致的 ga=16 无此问题。问题与修复记录于
`ddp1/hot_bench/results/stall_ab_report.md`。


## 九、风险与边界

1. store memcpy 使 ring pool 区域延迟回收 ~写时间 ×(memcpy/写盘时间比)，极端
   小 ring（host_cache_size 偏小）时可能拖慢写吞吐 → 观测；可选优化：store 独立
   线程 + shared_ptr 保活，或增大 host_cache_size。
2. 大 pinned 分配（容量=P）可能失败 → `cudaHostRegister` 失败时退化为普通
   malloc（H2D 稍慢，warning）。
3. 非 io_uring（pwrite）路径需同插 store；`split_load_bc`/旧 load 路径不在范围。
4. persist 失败（FATAL 退出）→ 不 promote，热备失效；与 v1/v2 "保留旧指针" 略有
   不同（此处旧数据可能已被新 save 部分覆盖，磁盘兜底更安全），已注明。
5. MoE/pipeline/universal/NVMe 与 v2 相同，不支持（DS 侧 gate 拦截）。
6. `StateCheckpointEngineAggregated`/`engine_state_h100` 不在本次范围（同构可移植）。

## 十、对 v2 已实现代码的处置清单

- 保留：DS `hot_weights` config、`_hot_backup_requested`、save 传参、
  `_zero3_fp16_layout_enabled`、wrapper `save(hot_backup=...)`/`supports_hot_backup`、
  base `supports_hot_backup()`。
- 删除/回退：cmpckp Python `HotBackupManager` 类与 fork 插桩（coalition_save/
  save_ 的 register_fork/finalize/structure）、Python 侧 `wait` promote、
  `get_hot_backup/release_hot_backup`（或改薄封装）；DS `_try_load_hot_checkpoint`/
  `_hot_weights_structure_matches`/`_try_hot_zero3_fp16_weights` 及 `_load_checkpoint`
  中两处旁路（z0/1/2 开头顶替 + z3 fp16 来源替换）。
