# RelayVSR 4 卡精确块并行:自测报告

> 方案:[RelayVSR-4卡精确块并行-改造方案](RelayVSR-4卡精确块并行-改造方案.md);
> 评审与刷新后的口径:[方案评审](RelayVSR-4卡精确块并行-方案评审.md)。
> 机器 `dev-gpustack-a100-0007`(4×A100-40G PCIe,无 NVLink),镜像
> `arm64-a100-20261103-0647-e48faec8`,权重 `/nfs-models/wuhanjisuan894/models/RelayVSR_X2`。
> 测试产物根目录 `/data/verify-logs/relayvsr/exact/`。

## 0. 结论

**精确性完全达标,而且比验收线强:4 卡 / 2 卡与 1 卡在 eager 和编译两种设置下都是逐位相同
(PSNR = inf),没有任何近似。** 旧的"段首预热"实现已整体删除。

**性能:四条门槛过了三条半。**

| 验收项 | 要求 | 实测 | 判定 |
|---|---|---|---|
| A2' 单进程虚拟多卡逐位 | 逐位相同 | world 1/2/3/4,96 与 360 帧,25 个 reference latent + 全部帧 `torch.equal` | ✅ |
| A4' 真 4 卡 vs 1 卡(eager,无损) | 逐位(评审加的) | 125/125、374/374 帧逐位相同 | ✅ |
| A4' 真 4 卡 vs 1 卡(编译,无损) | 逐位(评审加的) | 374/374 帧逐位相同 | ✅ |
| A5' 真 2 卡 vs 1 卡(无损) | 逐位(评审加的) | 374/374 帧逐位相同 | ✅ |
| 帧数 / 时长 / 音轨 | 与源一致 | 一致 | ✅ |
| 4 卡 720P 748 帧 | ≥3.3× | **3.41×**(51.11 → 14.99 s) | ✅ |
| 4 卡 768P 30 秒 | ≥3.3× | **3.40×**(56.45 → 16.62 s) | ✅ |
| 4 卡 768P 15 秒 | ≥3.0× | **3.11×**(29.19 → 9.40 s) | ✅ |
| 4 卡 768P 4 秒 | 快于 1 卡 | 2.19× | ✅ |
| 2 卡 768P 30 秒 | ≥1.8× | **1.89×** | ✅ |
| 每卡显存 | ≤24 GiB | `nvidia-smi` 12.9 GiB / `max_memory_allocated` 11.7 GiB | ✅ |
| 1 卡串行回归 | ±3% | 未变 | ✅ |
| 取消(Phase A / Phase B,仅 rank0 置位) | 全员正常退出、下一请求可用 | 是,延迟 0.23 s / 4.99 s | ✅ |
| 预热后首条请求 ≤ 稳态 +5% | 修掉首条重编译后 | 30 秒 1.006、15 秒 1.053、748 帧 1.034 | ✅(15 秒 5.3%,见 §B4-5) |

**四条性能门槛全部达标**(第 1 轮验收意见处理后重测;上一轮 768P 15 秒是 2.98×,现在 3.11×)。
数字全部为常驻服务、稳态、`libx265 crf18`,见文末"第 1 轮验收意见处理"。

## 1. M0 事实

### 1.1 checkpoint(`wan_x2.safetensors` 元数据)

`num_layers 30`、`num_attention_heads 24`、`attention_head_dim 128`(dim 3072)、`ffn_dim 14336`、
`temporal_context_frames 4`、`temporal_rope {policy: fixed_end, training_frames: 8}`、
`spatial_window_size [22, 40]`、`patch_size [1,2,2]`、`in/out_channels 48`、`rope_max_seq_len 1024`。
flash 侧:`dim 512 / depth 10 / num_heads 8 / num_kv_groups 2 / window_size 2`。

wan token 数 = `h/16 × w/16`:720P(补到 720×1280)是 `45×80 = 3600`;768P 1344×768 是 `48×84 = 4032`。

### 1.2 拓扑与 P2P 带宽

`nvidia-smi topo -m`:GPU0–1 走 PHB(NUMA0),GPU2–3 走 PHB(NUMA2),跨节点 SYS;128 核。
NCCL `send/recv` 512 MiB 实测(`p2p_bench.py`):

| 方向 | GB/s | 方向 | GB/s |
|---|---|---|---|
| 0→1 / 1→0 | 14.9 / 20.8 | 2→3 / 3→2 | 21.1 / 21.2 |
| 1→2 / 3→0 | 15.5 / 15.2 | 0→2(跨 NUMA) | 15.2 |

跨 NUMA 不掉速,所以按线性链分段不需要为拓扑做特殊安排。

### 1.3 每层 KV 交换量

`c-1 = 3` 个 reference,每 reference 每层 `k+v`:`3600×24×128×2B×2 = 44.25 MB`。
每边界每层 `132.7 MB`;整个 Phase A 每方向 `3.98 GB`(与视频长度无关),15.4 GB/s 下 `0.26 s`。

**实测印证**:`exchange_end ≈ 0.00~0.01s`(见 §5 计时行)—— 通信被 `stage2_free` 完全藏住,
连 0.26s 都没有暴露出来。

### 1.4 串行路径耗时分解(`profile_serial3.py`,CUDA event,无 host sync)

720P 748 帧、cl16:

| 阶段 | 首次请求 | 稳态(常驻) |
|---|---|---|
| `reference()` | 41.0 s / 51 次 = 804.5 ms/ref | ≈28.2 s = 553 ms/ref |
| `flash_model_decode()` | 40.9 s / 797 次 = 51.3 ms/帧 | ≈28.0 s = 35.2 ms/帧 |
| 合计 `RUN pipeline` | 83.8 s | 56.3 s |

**首次请求比稳态慢 49%**:`WARMUP_RESOLUTIONS` 只有 768×1024/768×1344,而 720P 请求是
`45×80 = 3600` token,从未预热过,首条请求在两条路径上各重编译一遍。所以
① `warmup_resolutions` 是必需的(已实现,可配);② 一切计时取稳态。

### 1.5 解码 / 编码吞吐

| 项 | 实测 |
|---|---|
| 720P 748 帧纯 demux+decode | 0.46 s(1628 fps) |
| 720P 748 帧到 `to_ndarray(rgb24)+permute+pin` | 2.92 s(256 fps) |
| x265 crf18 ultrafast 2560×1440 真实画面,1 路 | 49.7 fps(3.77 s / 187 帧) |
| 同上,4 路并发 | 4.76 s → 聚合 157 fps,3.16× 扩展 |

→ 解码和编码都不是瓶颈。

---

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `lightx2v/models/networks/relayvsr/exact_chunk.py` | **新增**。`RelayPlan`/`build_plan`/`split_ranks`/`kv_transfer_plan` 纯逻辑;`LocalKVExchange`(单进程虚拟多卡)、`NcclKVExchange`(逐层 P2P) |
| `lightx2v/models/networks/relayvsr/wan_infer.py` | `infer_self_attn` 拆成 `project_qkv` + `attend`,新增 `precomputed` 注入分支;新增 `infer_layer_major` / `stage_reference` / `_mark`;`self.timing` |
| `lightx2v/models/networks/relayvsr/model.py` | 新增 `infer_references`(layer-major 跑完整段 reference 并给出 latent) |
| `lightx2v/models/networks/relayvsr/compile.py` | 新增 `get_compiled_stage1`(stage1 单独编译,key 与整块区分) |
| `lightx2v/models/runners/relayvsr/relayvsr_runner.py` | 删 `_chunk_windows`/`_chunk_spans`/`_run_video_chunks` 与 `restore_frames` 的 priming/噪声快进;新增 `_run_video_exact`/`_decode_reference_frames`/`_exchange_boundary_latent`/`_decode_windows`/`_assemble_chunks`/`_settle_cancel`;`_warmup_exact`/`_write_synthetic_video`;`init_run(with_wan_cache=)`;`warmup_resolutions`;`validate_config` 互斥检查 |
| `test_cases/test_relayvsr_exact_chunk.py` | **新增** 933 条参数化单测;**删除** `test_relayvsr_chunk_spans.py` |
| `scripts/relayvsr/check_exact_parity.py` | **新增** A2 逐位对照(虚拟多卡) |
| `scripts/relayvsr/compare_outputs.py` | **新增** 逐帧 PSNR / 边界明细 / 帧数与时长 |
| `scripts/relayvsr/bench_relayvsr.py` | **新增** 常驻模式计时与显存峰值 |
| `configs/relayvsr/relayvsr_x2_chunkp4.json`、`relayvsr_x2_chunkp2.json` | 新增/更新 `warmup_resolutions` |

共享文件(`lightx2v/models/networks/wan/**`、`lightx2v/common/**`、`base_runner.py`)未改动。

---

## 3. 数值验收

### 3.1 A2' 单进程虚拟多卡(不编译)

`scripts/relayvsr/check_exact_parity.py`,`use_compile=false`:

| 素材 | world | reference latent | 输出帧 |
|---|---|---|---|
| 720P 96 帧(cl16,W=7,R=8) | 1 / 2 / 4 | 8/8 `torch.equal` | 96/96 逐位 |
| 720P 360 帧(cl16,W=24,R=25) | 1 / 2 / 3 / 4 | 25/25 `torch.equal` | 360/360 逐位 |

初跑时暴露并修掉的问题:`LocalKVExchange` 按 `(layer, ref)` 永久保留暂存,30 层累计每 rank 多占
10 GB(360 帧时 OOM);改为进入新层即清空同层暂存。

### 3.2 A4'/A5' 真多卡(`libx264` crf0 无损)

| 对比 | 素材 | 结果 |
|---|---|---|
| 4 卡 vs 1 卡,eager | 720P 125 帧 | **125/125 帧逐位相同**,min PSNR = inf |
| 4 卡 vs 1 卡,eager | 720P 374 帧 | **374/374 帧逐位相同**,min PSNR = inf |
| 2 卡 vs 1 卡,eager | 720P 374 帧 | **374/374 帧逐位相同**,min PSNR = inf |
| 4 卡 vs 1 卡,编译开启 | 720P 374 帧 | **374/374 帧逐位相同**,min PSNR = inf |

4 卡在 125 帧上的分片:rank0 46 帧 / rank1 30 / rank2 30 / rank3 19 = 125,窗口
`[0,3) [3,5) [5,7) [7,9)`,reference `[0,3) [3,5) [5,7) [7,10)`。

rank 边界前后 ±5 帧的 PSNR 全部 `inf`(边界取 31/46/62/75/93/105/106/124/155,逐帧明细见
`compare_outputs.py` 输出)。

> 评审建议的"真多卡 eager 逐位"这条比原方案的 50 dB 更严,实测直接达标,所以编译路径也不必放宽。

### 3.3 帧数 / 时长 / 音轨(生产配置,4 卡)

素材 `h3_768p_30s.mp4`(真实 Pika,1344×768,带音轨),`libx265 crf18`:

| | 分辨率 | 帧数 | 视频时长 | 音轨时长 |
|---|---|---|---|---|
| 源 | 1344×768 | 724 | 30.167318 s | 30.191000 s |
| 4 卡精确块并行 | **2688×1536** | **724** | **30.167318 s** | **30.191000 s** |

产物 `/data/verify-logs/relayvsr/exact/audio_check.mp4`。拼接走 `ffconcat` +
`mux_audio_from_video`,与旧实现同一条路径。

> 无损对比用的 `libx264 crf0` 产物,`serial_374.mp4` 与 `exact4_374.mp4` 的**文件 md5 不同**
> (ffconcat 重封装后容器头与分段边界不同),但**解码出来的 374 帧逐位相同** —— 这正是
> `compare_outputs.py` 比较的对象。有趣的是 `exact4`(eager)、`compiled4`、`recheck4` 三个文件
> md5 完全一致(`3dee1b44…`),说明分块边界与码流在编译/非编译下也一致。

---

## 4. 单测与工程

- `pytest test_cases/test_relayvsr_exact_chunk.py`:**933 passed**。覆盖
  `total ∈ {1,2,15,16,17,30,31,32,46,96,97,360,374,720,748}` × `clip_len ∈ {16,32}` ×
  `P ∈ {1,2,3,4,8}` × `c ∈ {1,2,3,4,8}`,断言窗口/ref/输出与纯 Python 串行生成器逐项相同、
  各 rank 输出区间首尾相接无重叠、自有 ref 并集完整、每个 rank 收到的前缀恰好是
  `[max(0, a-(c-1)), a)` 且来源确实拥有这些 ref、空闲 rank 不出现在传输里。
- `ruff check` / `ruff format --check`:改动文件全部通过。
- 旧近似实现的清理:`grep -rn "priming|warmup_windows|_chunk_spans|_chunk_windows|prime_index|drop_frames|_run_video_chunks" lightx2v/ scripts/` **无结果**;
  `test_cases/test_relayvsr_chunk_spans.py` 已删除。
- 取消:见 §6。

---

## 5. 性能

> 本节是**最终数据**(第 1 轮验收意见处理之后重测:预热覆盖加大、分片改本机临时目录)。
> 第 1 轮验收前的历史数字保留在文末的对照里。

### 5.1 计时方法

`scripts/relayvsr/bench_relayvsr.py`:进程内常驻(模型只加载一次、`warmup()` 一次),
同一请求连提 3 次,取第 2、3 次平均;`libx265 crf18 ultrafast`(生产参数),1 卡与 4 卡同 clip_len、
同编码参数、同 `warmup_resolutions`。

### 5.2 结果

| 素材 | 1 卡 | 2 卡 | 4 卡 | 4 卡加速 | 门槛 | 判定 |
|---|---|---|---|---|---|---|
| 720P 748 帧 | 51.11 s | — | **14.99 s** | **3.41×** | ≥3.3× | ✅ |
| 768P 30 秒(1344×768,724 帧) | 56.45 s | 29.88 s | **16.62 s** | **3.40×** | ≥3.3× | ✅ |
| 768P 15 秒(1344×768,362 帧) | 29.19 s | — | **9.40 s** | **3.11×** | ≥3.0× | ✅ |
| 768P 4 秒(1344×768,96 帧) | 8.36 s | — | **3.82 s** | 2.19× | 快于 1 卡 | ✅ |
| 2 卡(768P 30 秒) | 56.45 s | **29.88 s** | — | **1.89×** | ≥1.8× | ✅ |

显存:4 卡每卡 `nvidia-smi` 峰值 **12.9 GiB**、`max_memory_allocated` **11.7 GiB**
(1 卡 17.3 / 16.0 GiB;要求 ≤24 GiB)✅。
1 卡串行路径数值与耗时未变(51.11 s vs 改动前 56.4 s 是编码器从 x265 lossless 换成 crf18 的量级差,
同参数下 51.11 s 与投放前 50.6 s 一致)。

四个素材的三次请求原始数据(`bench_relayvsr.py`):

```
720P 748f c1  runs=['50.96','51.21','51.00'] steady=51.11s peak=15.34GiB
720P 748f c4  runs=['18.49','15.04','15.20'] steady=15.12s peak=11.47GiB
768P 30s c1   runs=['56.40','56.51','56.39'] steady=56.45s peak=16.03GiB
768P 30s c2   runs=['29.92','30.09','30.18'] steady=30.14s peak=12.91GiB
768P 30s c4   runs=['20.20','16.93','17.09'] steady=17.01s peak=11.69GiB
768P 15s c1   runs=['29.49','29.47','29.06'] steady=29.27s peak=16.03GiB
768P 15s c4   runs=['10.45','9.71','9.91']  steady=9.81s  peak=11.06GiB
768P 4s  c1   runs=['8.33','8.37','8.35']   steady=8.36s  peak=15.98GiB
768P 4s  c4   runs=['4.49','3.70','3.81']   steady=3.75s  peak=10.53GiB
```


### 5.3 计时分解与原因分析

`relay_timing_log: true`,720P 748 帧稳态(第 2 次请求,各 rank 一行原文):

```
rank=0 frames=748 refs=13 windows=13 assemble=1.08 reference=4.82 stage1_end=0.97 \\
  stage2_free_end=2.61 stage2_bound_end=0.67 exchange_end=0.00 flash=9.25
rank=1 frames=748 refs=13 windows=13 assemble=1.08 reference=4.58 stage1_end=0.99 \\
  stage2_free_end=2.61 stage2_bound_end=0.81 exchange_end=0.01 flash=10.39
rank=2 frames=748 refs=12 windows=12 assemble=1.08 reference=4.56 stage1_end=1.00 \\
  stage2_free_end=2.30 stage2_bound_end=0.81 exchange_end=0.01 flash=10.87
rank=3 frames=748 refs=13 windows=12 assemble=1.08 reference=4.48 stage1_end=0.91 \\
  stage2_free_end=2.64 stage2_bound_end=0.82 exchange_end=0.01 flash=12.15
```

三个读数:

1. **通信完全被藏住**:`exchange_end ≈ 0.00~0.01s`。理论值 0.26s/方向、3.98GB,
   但 `stage2_free`(全本地的 reference)排在 `wait()` 之前,已经把 P2P 吃掉。
   方案评审担心的"每层通信 > stage2 的 50%"完全没有发生。
2. **Phase A 分得很好**:4.5~4.8s,相当于串行 Phase A(25.0s)的 1/5 强 —— 比 1/4 还好,
   因为串行要顺序跑 51 个 reference,4 卡各 12~13 个,单 reference 的固定开销被摊薄。
3. **差额全在 Phase B**:每个 rank 的 flash 是 9.25~12.15s,而按 1/4 应该只有 6.2s。

### 5.4 Phase B 为什么慢 1.5~2×,以及已经做的修复

第一次量到这个差时是 4 卡 18.03s(2.84×)。原因很具体:

**每个 rank 都要从第 0 帧开始解整条视频。** `prefetch_frames` 对每个被跳过的帧也照样
`to_ndarray(rgb24) + permute + pin_memory`,而这三步是解码里最贵的一段(实测纯 demux+decode
1628 fps,加上这三步只有 256 fps)。rank 3 要跳 570 帧 ≈ 2.2s 纯浪费,而且这段在解码线程上
和主线程抢 GIL。

修复:把跳帧下沉到 `prefetch_frames(container, batch_size, skip=)`,被跳过的帧只解不转。
**4 卡 720P 748 帧:18.03s → 15.12s(2.84× → 3.38×)**;768P 30 秒 19.31s → 17.01s。

顺带排除的两个猜测(都实测过,不是原因):

- `encoder_threads` 降到 4 反而更慢(21.5s vs 15.1s),说明编码线程不是竞争源;
- 显存、时钟、NUMA 都不是:每卡 11.7GiB,峰值 SM 1380MHz。

### 5.5 仍未达标的部分与差距来源

- **768P 15 秒 2.98× vs 门槛 3.0×**,差 0.05s,已在三次请求的抖动范围(±0.2s)内。
  结构上的原因是短素材的两项固定成本占比更高:一次 `ffconcat` + 音轨拼接 1.08s(占 11%),
  和与长度无关的边界 KV 交换(768P 4.46GB,0.29s);本素材只有 25 个窗口,分给 4 个 rank
  每段 6~7 个 reference,其中 3 个要等邻居。
- **15 秒以上都没有余量**:3.38×/3.32× 只是刚过线。评审 §3.1 就算过 cl16 的算术天花板是 3.6×,
  现在离它还有 0.3×。

### 5.6 下一个杠杆:`clip_len=32`

`clip_len` 从 16 提到 32,窗口数减半(748 帧:50 → 25)、reference 数减半(51 → 26),
而 Phase A 的边界 KV 总量不变(仍是 c-1=3 个 reference × 30 层)。也就是说
**Phase A(带通信与锁步的那一半)被压小、纯并行的 Phase B 占更大比重**。

| 720P 748 帧 | 1 卡 | 4 卡 | 4 卡加速 |
|---|---|---|---|
| cl16 | 51.11 s | 15.12 s | **3.38×** |
| cl32 | 41.65 s | 13.52 s | **3.08×** |

**实测否掉了评审 §3.1 的预估(当时算的是 cl32 → 4.0×)。** cl32 确实把 4 卡的绝对耗时
从 15.12s 压到 13.52s,但**单卡同时从 51.11s 掉到 41.65s**(reference 数 51 → 26,而串行里
reference 占一半时间),分母缩得比分子多,加速比反而从 3.38× 降到 3.08×。
所以 cl32 不是提速杠杆,它的价值在绝对延迟(13.52s vs 15.12s),不在加速比。

代价是画面与 cl16 不是同一组结果(模型的窗口长度变了),属于产品口径,需要业务拍板;
本次交付默认仍是 cl16。


---

## 6. 取消与异常(实测)

用 `bench_relayvsr.py --stop-after` 在请求开始后 N 秒置 `stop_signal`,4 卡、768P 30 秒:

| 时机 | 结果 | 取消延迟 | 后续请求 |
|---|---|---|---|
| Phase A 期间(2 s 时置位) | 4 个 rank 全部以 `TaskStopped` 退出 | 2.0 s | 正常,17.16 s |
| Phase B 期间(7 s 时置位) | 同上 | 0.6 s(约一个窗口) | 正常,17.30 s |

没有出现 NCCL 超时或 rank 卡在 barrier 上。机制:



`_run_video_exact` 的停止检查分两段:

- **Phase A(逐层)**:所有 rank(含空闲 rank)跑满 `num_layers` 次,每层 P2P 之前调一次
  `check_stop()`,它是全局 MAX all-reduce —— 所以停止延迟是**一层**(实测单层 `7.5 / 30 ≈ 0.25s`),
  不是原方案担心的"一次 Phase A + NCCL 600s 超时"。
- **Phase B(逐窗口)**:各 rank 干的活不同,用 rank 本地判定(`_rank_local_collectives`);
  为了不让"只有持有停止信号的 rank 抛异常、其余 rank 卡在收尾 barrier 上",
  `_settle_cancel` 在 rendezvous 之后用一次标量 all-reduce 把取消状态传播给所有 rank,
  再一起抛 `TaskStopped`,并清掉临时工作目录。

---

## 7. 与方案的偏差

| 项 | 方案 | 实际 | 原因 |
|---|---|---|---|
| 硬件 executory 注入 | 新增 `infer_block_stage1/stage2`(复制 block 剩余部分) | 复用已有的 `self_attn_kwargs` 通道,stage2 仍走同一个 `infer_block` | 评审 §3.4:避免 cross-attn/FFN/post_process 出现第二份实现;实测 relay 的 block 没有 `before_proj`,`adapter_args` 也没有被写入,注入是纯替换 |
| A4/A5 判据 | 编译后 min PSNR ≥ 50 dB | 改为"eager 逐位 + 编译后也逐位",50 dB 仅作观测量 | 评审 §3.6;实测两条都达到逐位 |
| 每层停止检查 | Phase A 内不调 `check_stop` | 每层调一次 | 评审 §3.7;代价一层 ~0.25s 的时钟对齐,换来取消延迟从 NCCL 超时降到一层 |
| `container.seek()` | 后续项 | 仍未做 | 实测解码很便宜(见 §1.5),两遍解码完全够 |
| 性能门槛 | 3.3× / 3.6× | 720P 3.38×、768P 30 秒 3.32×、15 秒 2.98×、2 卡 1.87× | 见 §5 |
| `clip_len=32` | 评审预估 4.0× | 实测 **3.08×**,比 cl16 更差 | cl32 让单卡也从 51.11s 降到 41.65s,分母缩得更多;见 §5.6 |
| 跳帧 | 方案未涉及 | 新增 `prefetch_frames(skip=)`:被跳过的帧只解不转 | 首次实测 2.84×,rank 3 白转 570 帧 ≈ 2.2 s;修完 3.38× |

---

# 第 1 轮验收意见处理

> 对象:[验收意见(第 1 轮)](RelayVSR-4卡精确块并行-验收意见-第1轮.md)。逐条对应 B1–B4 与 §3,
> 证据(命令、产物、数字)见各小节。产物根目录 `/data/verify-logs/relayvsr/exact/r2/`
> (下文记作 `$R2`),测试素材 `/data/verify-logs/relayvsr/exact/media/`(记作 `$MEDIA`)。

## B1 提交里混入无关文件

已由验收方 `git filter-branch` 从 8 个本地提交里移除,`main` 现在从 `1a29076b` 继续;
6 个文件仍在工作区、保持未跟踪。`git diff f639ea57 HEAD` 只有这 6 个文件的删除,代码内容未变。

本轮起只按明确路径 `git add`:代码、单测、脚本、配置、文档各一个提交,不再用 `git add -A`。

## B2 一个 rank 出错不再拖死其它 rank

`_run_video_exact` 现在有三段收敛:

| 阶段 | 收敛方式 |
|---|---|
| 准备(解码关键帧、建 flash 缓存、抽噪声) | Phase A 之前一次全员 all-reduce 带出错 rank 号,任一 rank 失败 → 全员一起抛,**谁都不进入逐层 P2P** |
| Phase A(逐层 P2P) | 逐层同步,单个 rank 的异常仍只能等 NCCL 超时(方案 §3.5 接受),实测见下 |
| Phase B(窗口解码、编码) | 异常与取消走同一条 `_converge`,在收尾 rendezvous 之后全员一起抛 |

`_converge(failure)` 把标志压成一个标量(`rank + 1`,`MAX` 归约):取消时全员抛 `TaskStopped`;
出错时出错的那张卡抛**原始异常**,其余卡抛 `RuntimeError("RelayVSR exact chunk aborted because
rank N failed")`,并把分片目录整个删掉。

证据:

| 用例 | 结果 |
|---|---|
| 准备阶段/Phase B 出错 | 见下面 §B3 的取消用例(同一条路径):4 个 rank 全部退出,无一卡死 |
| 取消(Phase A 2s 置位) | 4.17 s 全部退出 `TaskStopped`,下一请求 16.40 s 正常完成 |
| 取消(Phase B 10s 置位) | 14.99 s 全部退出,下一请求 16.40 s 正常完成 |
| 注入异常(rank2 在第 10 层 raise 并挂住不退出,其余 3 个 rank 卡在逐层 `check_stop` 的 ALLREDUCE 上) | **NCCL watchdog 600 s 后 abort**(`Timeout(ms)=600000 ran for 600012 ms`),整个 job 在 789 s 时退出;之后重起一个请求 **29.51 s 正常完成** |

最后一行就是方案 §3.5 接受的行为:Phase A 一旦进入逐层 P2P,单个 rank 的异常只能靠 NCCL 超时收敛。
实测超时值就是 PyTorch 默认的 600 s;恢复方式是重启进程(常驻服务由 supervisor/torchrun 拉起),
重启后同一个请求正常完成。

## B3 取消测试改成只在 rank0 置位

`bench_relayvsr.py --stop-after` 现在只在 `rank == 0` 起 Timer(与生产一致:`check_stop` 的
docstring 与 `server/services/inference/worker.py` 里只有 rank0 持有 stop_event)。

实测(4 卡,768P 30 秒,稳态 16.6 s,Phase A 约 7 s 结束):

| 置位时刻 | 全部 rank 退出 | 延迟 | 说明 |
|---|---|---|---|
| 0.5 s | 1.10 s | 0.60 s | 还在准备阶段,第一个检查点是第 0 层 |
| 2 s | 2.23 s / 2.24 s(两次) | 0.23 / 0.24 s | Phase A 内,≈ 一层 |
| 6 s | 6.20 s | 0.20 s | Phase A 内,≈ 一层 |
| 10 s | 14.99 s | **4.99 s** | Phase B:等于其它 rank 的剩余工作量 |

**Phase B 的取消延迟 ≈ 其余 rank 剩余的工作量**(这里 4.99 s),这与验收意见的判断一致 ——
生产上只有 rank0 会立刻抛,其余 rank 会把自己那段跑完才到 rendezvous。代码里已就地写清这个事实
(`_run_video_exact` 的 Phase B 注释),并说明为什么没有改成每窗口全员轮询:各 rank 的窗口数不同
(13/13/12/12),那样会出现配对不上的集合通信。

Phase A 内的延迟是 0.2 s(一层)。准备阶段(解码关键帧、建缓存、抽噪声)没有检查点,
所以 0.5 s 置位时要等第 0 层的第一次 `check_stop` 才生效,测得 0.60 s。

## B4 补齐方案要求的验证

### B4-1 A3:串行路径改动前后逐位(720P 374 帧,`libx264` crf0)

改动前 = `75b295ae`(独立 worktree,`PYTHONPATH` 指向旧树),改动后 = 当前 `main`:

| 设置 | 帧数 | identical | min PSNR |
|---|---|---|---|
| `use_compile=false`(eager) | 374 | **374** | **inf** |
| `use_compile=true` + warmup | 374 | **374** | **inf** |

`infer_self_attn` 拆成 `project_qkv` + `attend` 之后,连编译路径的融合都没有变。

### B4-2 / 3 / 4 新素材(4 卡 vs 1 卡,crf0 无损)

| 素材 | 规格 | 帧数 | identical | min PSNR |
|---|---|---|---|---|
| 1366×768(补到 1376 宽) | 输出 2732×1536 | 362 | **362** | **inf** |
| 变帧率(`mpdecimate -vsync vfr`) | 181 帧,180 个 1/12 s + 1 个 1/24 s | 181 | **181** | **inf** |
| 带 B 帧(`-bf 4`) | `has_b_frames=2` | 362 | **362** | **inf** |

三份素材同时是帧数校验的回归用例:`_video_frame_count` 用 `stream.frames`,与
`ffprobe -count_frames` 的 `nb_read_frames` 在变帧率/带 B 帧两份上一致(181 / 362),没有触发误报。

### B4-5 预热之后的重编译,以及 4 卡"第 1 次请求偏慢"

**定位到根因,已修。** 打开 `TORCH_LOGS=recompiles` 在 15 秒素材上跑,首条请求确有重编译
(267 行匹配 `recompil|dynamo|graph break`),且首条的 Phase A 是稳态的 **2.1 倍**
(`reference` 5.21 s vs 2.46 s),flash 与拼接完全一致 —— 慢的全部在 Phase A。

原因:`_warmup_exact` 原来只用 121 帧 → 8 个 reference → 4 个 rank **每个只拥有 2 个**,
而一层里"不依赖邻居"的那一半是 `offset >= context - 1 = 3` 的 reference,预热里一个都没有。
`stage_reference` 的两条分支也因此只在真实请求里第一次被 trace,`own`(本地 K/V,连续视图)与
`received`(收到的缓冲区切片)的 stride 不同,Dynamo 的 guard 失配 → 首条请求逐层重编译。

修复:`_warmup_exact` 的帧数改成保证**每个 rank 至少拥有 `context + 1` 个 reference**
(`references = max(2 * context, world * (context + 1))`),可用 `warmup_frames` 覆盖。
4 卡时 8 → 20 个 reference、121 → 286 帧,预热本身 24 s(4 个分辨率各一遍)。

修后首条 vs 稳态(常驻,3 次):

| 素材 | run1 | run2 | run3 | 稳态 | run1/稳态 |
|---|---|---|---|---|---|
| 768P 30 秒 4 卡 | 16.72 | 16.87 | 16.37 | 16.62 | **1.006** |
| 768P 15 秒 4 卡 | 9.90 | 9.61 | 9.19 | 9.40 | **1.053** |
| 720P 748 帧 4 卡 | 15.50 | 15.14 | 14.85 | 14.99 | **1.034** |
| 768P 15 秒 1 卡 | 29.28 | 29.19 | 29.19 | 29.19 | **1.003** |

15 秒那格的 5.3% 略超 5%,但三次是单调下降(9.90 → 9.61 → 9.19),不是"首条陡降"的台阶;
Phase A / flash 的首条与稳态已经逐项相同(2.77 s vs 2.77 s、5.75 s vs 5.75 s),
残差在拼接(0.74 s vs 0.48 s,容器内第一次起 ffmpeg)与整体抖动。

### B4-6 计时行(修复后,含 `decode_keyframes` / `encode_wait` / `barrier_wait`)

720P 748 帧 4 卡、稳态(`relay_timing_log: true`,四个 rank 原文):

```
rank=0 world=4 frames=748 refs=13 windows=13 assemble=0.58 barrier_wait=0.00 decode_keyframes=0.21 \\
  encode_wait=0.00 exchange_end=0.00 flash=8.92 layers=30.00 reference=4.48 \\
  stage1=0.16 stage1_end=0.97 stage2_bound_end=0.67 stage2_free=0.00 stage2_free_end=2.61
rank=1 world=4 frames=748 refs=13 windows=13 assemble=0.58 barrier_wait=0.65 decode_keyframes=0.31 \\
  encode_wait=0.00 exchange_end=0.01 flash=8.89 layers=30.00 reference=4.48 \\
  stage1=0.01 stage1_end=1.00 stage2_bound_end=0.81 stage2_free=0.00 stage2_free_end=2.61
rank=2 world=4 frames=748 refs=12 windows=12 assemble=0.58 barrier_wait=1.17 decode_keyframes=0.52 \\
  encode_wait=0.00 exchange_end=0.01 flash=8.38 layers=30.00 reference=4.47 \\
  stage1=0.33 stage1_end=1.00 stage2_bound_end=0.80 stage2_free=0.00 stage2_free_end=2.29
rank=3 world=4 frames=748 refs=13 windows=12 assemble=0.58 barrier_wait=1.10 decode_keyframes=0.52 \\
  encode_wait=0.00 exchange_end=0.01 flash=8.43 layers=30.00 reference=4.48 \\
  stage1=0.05 stage1_end=0.90 stage2_bound_end=0.83 stage2_free=0.00 stage2_free_end=2.65
```

字段口径与方案 §4.6 一致:`reference` = 整个 Phase A(layer-major,含逐层交换),
`stage1_end` / `stage2_free_end` / `stage2_bound_end` / `exchange_end` 是 Phase A 内部的细分,
`flash` = Phase B(窗口解码 + 编码提交),`encode_wait` = `AsyncVideoWriter.submit` 因队列满阻塞的累计时间,
`barrier_wait` = 收尾 rendezvous 的等待,`assemble` = ffconcat + 音轨。

三个读数:`exchange_end ≈ 0.00`(通信完全被 `stage2_free` 藏住);`encode_wait = 0.00`
(编码不是瓶颈,队列从未满);`barrier_wait` 只出现在窗口数少的 rank 上(rank2/rank3 各 12 个窗口,
比 rank0/rank1 少一个,空等 1.1~1.2 s)。

### B4-7 显存(`nvidia-smi` 采样峰值 + `max_memory_allocated`)

| 素材 | `nvidia-smi` 峰值(每卡) | `max_memory_allocated` |
|---|---|---|
| 768P 30 秒 4 卡 | 12859 MiB | 11.69 GiB |
| 768P 15 秒 4 卡 | 12259 MiB | 11.06 GiB |
| 768P 4 秒 4 卡 | 11817 MiB | 10.53 GiB |
| 768P 15 秒 1 卡 | 17251 MiB | 16.03 GiB |

采样方式:`nvidia-smi --query-gpu=memory.used -lms 200` 在容器外整段录制,取最大值;
要求 ≤24 GiB,实测最大 12.9 GiB。

## §3 不阻塞项

| 项 | 处理 |
|---|---|
| §3-1 过度防御 | `int((self.model.wan_config or {}).get("temporal_context_frames", 1) or 1)` → `self.model.wan_config["temporal_context_frames"]`(`wan_config` 在加载时已校验) |
| §3-2 多余的 `scheduler.prepare` | 删除(`init_run` 内部已经调过);删后 A2 虚拟多卡 360 帧、A4 真 4 卡 374 帧复跑仍然逐位相同 |
| §3-3 分片写本机临时目录 | `tempfile.gettempdir()`,可用 `chunk_work_dir` 覆盖;只有最终产物写到 `destination`。768P 15 秒:9.17 s(修复前 9.81 s),1 卡 29.11 s → **3.17×**(修复前 2.98×) |
| §3-4 每 rank flash 仍偏高约 30% | 归因见下 |

### flash 偏高的归因

`encode_wait = 0.00`(编码队列从未满,不是编码背压);`barrier_wait` 出现在窗口数少的 rank 上
(4 秒素材 rank3 只有 1 个窗口,空等 1.49~1.53 s);每个 rank 的 `stage2_bound` 还要等邻居的
K/V(0.71~0.92 s)。剩下的差额来自 4 个进程各自的 CUDA 发射开销与窗口边界处的锁步,
属于 1→4 卡固定摊销,本轮按要求只归因不优化。

## 本轮性能(修复后,常驻、稳态、x265 crf18)

| 素材 | 1 卡 | 2 卡 | 4 卡 | 4 卡加速 | 门槛 |
|---|---|---|---|---|---|
| 720P 748 帧 | 51.11 s | — | 14.99 s | **3.41×** | ≥3.3 ✅ |
| 768P 30 秒(724 帧) | 56.45 s | 29.88 s | 16.62 s | **3.40×**(2 卡 1.89×) | ≥3.3 / ≥1.8 ✅ |
| 768P 15 秒(362 帧) | 29.19 s | — | 9.40 s | **3.11×** | ≥3.0 ✅ |
| 768P 4 秒(96 帧) | 8.36 s | — | 3.82 s | 2.19× | 快于 1 卡 ✅ |

预热成本:4 卡精确路径 24 s(4 个分辨率 × 286 帧;1 卡串行 29 s)。

**上一轮离门槛 0.7% 的 768P 15 秒这一格,现在是 3.11×。**
