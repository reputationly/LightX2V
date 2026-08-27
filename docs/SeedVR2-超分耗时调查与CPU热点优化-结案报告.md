# SeedVR2 超分耗时调查与 CPU 热点优化 结案报告

> 2026-08-23 · dev 节点 0017-0020(4×4×A100 PCIE 40G,ARM/鲲鹏,无 NVLink)· profile `seedvr2-3b/seg121-4card`
>
> **TL;DR:现网 5 秒素材超分耗时 965s、15 秒素材 20 分钟跑不完,根因不在 GPU——DiT 只占总耗时 3-4%,96% 花在两处 CPU 单线程大数组操作上。`_build_video_transform` 因 `cpu_offload=True` 把输入数据也放到 CPU,整条 bicubic 放大+edge pad 在单核上跑;`save_to_video` 的 RGB→BGR 负步长拷贝让 numpy 逐元素 materialise。两处各改一行后:单段 831s→129s,15 秒素材 20 分钟跑不完→120s,GPU 利用率 0%→81%。冷启动/首单惩罚、节点故障、NFS、分段数全部证伪。**

---

## 一、症状

体验区 1080P 文生视频走两段流水线(生成→超分),超分段异常慢且耗时不可预测:

| 现网任务 | 素材 | 段数 | 耗时 |
|---|---|---|---|
| `7ef857c0...` | 124 帧 (5s) | 2 | **965s** |
| `b22a...` | 243 帧 (10s) | 3 | **1253s** |

历史同模型任务耗时 113s / 225s / 353s / 1227s / 7081s,量级乱跳。前端轮询上限 6 分钟(`VIDEO_POLL_MAX_TIMES=90 × 4s`),用户看到进度条冻在 91% 不动。

## 二、根因

`py-spy dump` 在两处空白期抓到的栈直接定位:

```
前置空白(GPU util 0%,四张卡全空):
  rank0  interpolate (torch/nn/functional.py:4908)
         resize (torchvision/transforms/_functional_tensor.py:467)
         __call__ (.../transforms/area_resize.py:37)
         _build_video_transform (seedvr_runner.py:121)
  rank1  pad (torchvision/transforms/_functional_tensor.py:436)
         __call__ (.../transforms/divisible_crop.py:32)
         _build_video_transform (seedvr_runner.py:129)

尾部空白(GPU util 0%,一个 rank 在此停了 8m47s):
  rank1  save_to_video (lightx2v/utils/utils.py:249)   # .cpu().numpy()
  rank2  save_to_video (lightx2v/utils/utils.py:252)   # frames[..., ::-1].copy()
```

**热点 1 —— 前处理跑在 CPU。** `default_runner.py:143` 的 `set_init_device` 在 `cpu_offload=True` 时把 `init_device` 设为 CPU,而 `seedvr_runner.py:1048` 用它承载**输入视频张量**(该开关本意只管模型权重)。于是 `NaResize`(bicubic 放大)+ `DivisibleCrop`(edge pad)整条链在单个 CPU 核上跑,torchrun 还默认把 `OMP_NUM_THREADS` 设为 1。

**热点 2 —— 一次纯浪费的通道翻转。** `save_to_video` 的 `frames[..., ::-1].copy()` 把 RGB 翻成 BGR,只为喂 ffmpeg 的 `-pix_fmt bgr24`。反转最后一维会让数组变负步长,numpy 只能逐元素 materialise。

微基准(96 帧 1920×1080×3,一段的量):

| 操作 | CPU | GPU |
|---|---|---|
| area resize 96 帧 (1344×768→1920×1104) | **43.7s** | **0.04s** |
| `frames[..., ::-1].copy()` | **23-166s** | `torch.flip` **0.00-0.03s** |

两个 CPU 数字都极不稳定(同一进程内重复三次:23.0 / 36.9 / 105.6 秒),因为多个 rank 同时做 GB 级数组分配与拷贝时互抢内存带宽。这也解释了现网耗时为何毫无规律。

**改造前的时间构成**(362 帧 / 4 段):前置 587s + DiT 37s + 尾部 239s。**GPU 实际计算占比 3-4%。**

## 三、改动(2 files, +25/-5)

**`lightx2v/models/runners/seedvr/seedvr_runner.py:121`** — `_build_video_transform` 开头加一行,把变换搬到加速器:

```python
img = img.to(AI_DEVICE, non_blocking=True)
```

先上传再放大还顺带**减少 2.5 倍传输量**(源分辨率比放大后小);输出留在 device 上,`vae_encode` 本就要搬过去,`run_vae_decoder` 的 `color_fix="gpu"` 也直接读 `self._input`,省掉每段一次 H2D。

**`lightx2v/utils/utils.py:249`** — 删掉 `frames[..., ::-1].copy()`,ffmpeg 输入管道 `bgr24` → `rgb24`(lossless 与常规两个分支都改)。要的就是手上已有的布局,拷贝彻底消失而非转移。

## 四、验证(四节点 × 单段/双段/四段)

| 素材 | 帧数 | 段数 | 改造前 | 改造后 |
|---|---|---|---|---|
| short105 | 106 (4.4s) | **1** | **831.5s** | **129.2s** |
| h3 t2v 产物 | 124 (5s) | 2 | 120.8s(现网 965s) | **107.6s** |
| 用户素材 | 362 (15s) | 4 | **>20 分钟未完成** | **119.6 / 121.2 / 126.3 / 143.6s** |

关键不只是变快,而是**性能悬崖消失**:改造前单段 831s、双段 120s、四段跑不完,毫无规律;改造后 106→362 帧、1→4 段全部落在 107-144s。耗时改由 GPU 计算主导,而 GPU 计算是线性且稳定的。

**资源与正确性**(362 帧 / 4 段,1Hz 采样):

- GPU 利用率 **81-82%** 平均、峰值 100%、四卡 85% 时间在忙(改造前抓栈时为 **0%**)
- 峰值显存 **33147 MiB / 40960**,比改造前(profiles 记录 31.9G)只涨 **1.2 GB**——正好等于 `_input` 留在 GPU 的代价(96 帧 × 1920×1104×3 × bf16)。该增量有硬上界:分段保证每段 ≤ `sr_segment_length=121` 帧,最坏情况约 33.5 GB,**无 OOM 风险**
- 颜色:同 seed 新旧输出逐帧比对,同序得分 **5.56** vs R↔B 交换得分 **48.16**;15 秒素材对源比对 **3.71 vs 40.15**,均 COLORS MATCH。残留 ~1%/通道差异来自 bicubic 在 CPU/GPU 的实现差异
- 输出规格 1920×1080 / 24fps / 15.08s / aac 音轨完整

## 五、排除清单(全部证伪,勿重走)

| 假设 | 证据 | 判定 |
|---|---|---|
| **冷启动 / 首单惩罚** | 改造前:0019 **冷**(启动后从未跑过任何任务)120.8s vs 0018 **热** 118.0s;改造后:0019 冷 129.6s vs 0020 热 130.8s | **无差异(<2%)**。曾据 `Load models cost 419-538s` 与 VAE 的 `weights_mmap=True` 推断存在首单惩罚,**错**:DiT 权重走 `networks/seedvr/model.py:111` 的 `torch.load` 一次性读入常驻内存,不是 mmap,闲置多久都不掉 |
| 节点故障(0017 慢) | 0017 现网 965s;同机同素材同分段重跑 **120s** | 节点无罪 |
| NFS 带宽 | 顺序读 **267-314 MB/s**,写 **90 MB/s** | 正常 |
| 容器 CPU 受限 | `NanoCpus=0 CpusetCpus= `,128 核,load average 1.0 | 无限制 |
| ffmpeg 编码慢(`-crf 12 -preset slow`) | 容器内实测编码 124 帧 1080P **5.2s**(`-threads 4` 在输入侧,x264 照样吃满核:user 1m39s / real 5.2s);concat 是 `-c copy` | 无罪 |
| 镜像/代码版本回归 | 镜像 `lightx2v:arm64-a100-latest` 2026-08-15 06:00 构建,与跑出 113s 的那次是同一个 | 无变更 |
| 实例 flap / 重调度 | gpustack-server 日志仅一条 `assigned to instance 503`,无 re-dispatch;`RestartCount=0` | 无 |
| 主机内存被 VACE/h3 挤占 | seedvr2 独占节点(`docker ps` 仅 seedvr2 + gpustack-worker);251G 中 available 100G | 无竞争 |
| 段数少导致慢 | 106 帧 **1 段** 831s 反而比 125 帧 **2 段** 120s 慢 7 倍 | 方向相反,是每 rank 帧数触发 CPU 热点的超线性 |

## 六、两个证伪的进一步优化(勿重试)

**① 调大 / 关闭 VAE tiling —— 全部更慢,且触发 OOM**

动机是 tile 512/overlap 64 在 1920×1104 上切出 15 个 tile = 3.93M 像素,而画面只有 2.12M,存在 1.85 倍重复计算。实测(362 帧,四台并行同素材):

| 变体 | 耗时 | 峰值显存 | OOM 日志 |
|---|---|---|---|
| **base512(现配置)** | **109.6s** | **33147 MiB** | 0 |
| tile1024 | 182.5s | 40431 MiB | 0 |
| tile2048 | 144.5s | 40427 MiB | **14** |
| use_tiling_vae=false | 142.2s | 40427 MiB | **14** |

**结论:tiling 不是浪费,是拿少量重复计算换显存平稳。** 去掉它省下的计算被显存压力下的分配器抖动与 OOM 重试吃掉还倒亏。`vae_tile_size=512` 已是最优,不要动。

> 实验教训:第一轮通过嵌套 heredoc 改容器内 config **没有生效**,四台跑的都是 512,得到一组噪声数据。改成"生成文件 → `docker cp` → 从启动日志验证 `'use_tiling_vae': X, 'vae_tile_size': Y` 确实加载"后才拿到可信结果。**改容器内配置后必须从启动日志验证生效。**

**② `save_to_video` 用 pinned staging buffer —— 微基准与端到端结论打架,判死**

微基准显示 pinned 压倒性占优(分配成本也不是问题,PyTorch 的 CachingHostAllocator 会复用):

```
run0  pin_alloc=0.52s  pin_copy=0.04s  合计=0.55s  |  pageable .cpu()=74.79s
run1  pin_alloc=0.00s  pin_copy=0.04s  合计=0.04s  |  pageable .cpu()=18.22s
run2  pin_alloc=0.00s  pin_copy=0.04s  合计=0.04s  |  pageable .cpu()=35.73s
```

但端到端 0020 从 109.6s 恶化到 **448.8s**。两者能同时成立的解释是:**该 ARM 平台 host 内存子系统底噪极大**——同一段 pageable 拷贝在不同时刻测出过 0.13 / 18 / 35 / 75 / 157 秒,波动上千倍,且 pinned 与 pageable 混用会互相干扰。收益上限只有尾部 16 秒,却带来无法解释的 4 倍劣化风险,**已回退**。

## 七、遗留与待办

1. **改动目前是容器内热替换**(`docker cp` + `docker restart`),原文件备份为 `/opt/LightX2V/lightx2v/{utils/utils.py,models/runners/seedvr/seedvr_runner.py}.orig` 可回滚;**长期生效需重建 `lightx2v:arm64-a100-latest` 镜像**
2. `_build_sr_segments` 的段数只由 `seg_len=121` 决定,**与可用卡数无关**:124 帧 → 2 段 → 两张卡全程闲置。按 world size 分段(124 帧分 4 段)理论收益约 1.7×,但段越多跨段接缝越多、每段时间上下文从 62 帧缩到 39 帧,**属速度换质量,需出样片验收后再定**
3. new-api 侧 `VIDEO_POLL_MAX_TIMES=90 × 4s = 6 分钟`轮询上限:改造后 SR 段 107-144s 已不会触发超时,但余量仅 2 倍,可考虑放宽
4. SeedVR2 官方效率数据([arXiv 2506.05301](https://arxiv.org/pdf/2506.05301))报的是 **720p / 100 帧**,不是 1080p;第三方评测(SwiftVR)指出 3B 变体在 1440p 以上只能靠 VAE tiling 勉强跑。**把它推到 1080p 输出本就是它不擅长的区间**,这是选型层面的约束

## 八、调试手法备查

- **py-spy 抓栈是本次的决定性工具**:`pip install --target /tmp/pyspy py-spy` 装进容器(不污染 site-packages),`py-spy dump --pid <rank pid>`。在"日志空白 + GPU 0%"的区间抓,一次就定位到行号。用完 `rm -rf /tmp/pyspy`
- 找任务落在哪个实例:gpustack-server 日志 `grep <task_id>` 拿 `assigned to instance N`,或四台并行 `docker logs | grep -c <task_id>`
- 直连引擎复现(不走 new-api、不计费):`POST http://<ip>:<proxy_port>/v1/tasks/video/`,body 见 `lightx2v/server/schema.py:75` 的 `VideoTaskRequest`,关键字段 `video_path / sr_ratio / resize_mode / save_result_path / seed`
- 阶段耗时靠日志时间戳差:`Processing segment` → `infer_main cost` → `center crop SR output` → `RUN pipeline cost`,三段间隔分别对应前处理、DiT、decode+落盘
- 颜色回归:同 seed 新旧输出各取一帧,比 `|R_old-R_new|+|B_old-B_new|`(同序)与 `|R_old-B_new|+|B_old-R_new|`(交换),前者显著小才算通过
