# InfiniteTalk 实验测试报告(音频驱动数字人, 4步蒸馏)

> 日期:2026-07-10 · 环境:gpustack 集群计算节点(dev-gpustack-a100-0010, A100 PCIE 40G)
> 镜像:`lightx2v:arm64-a100-latest` · harness:`run_batch.sh it_distill`
> 模型:MeiGen-AI InfiniteTalk(Wan2.1-I2V-14B 底座 + 音频 adapter),配 lightx2v 4步蒸馏 DiT

## 结论先行:音频数字人引擎选型定案 = InfiniteTalk

与 Wan2.2-S2V 同素材(seko_input.png + seko_input.mp3, 5s)对比,**全维度胜出**:

| 维度 | Wan2.2-S2V(40步) | **InfiniteTalk(4步蒸馏)** |
|---|---|---|
| 生成耗时 | 1318s(22分钟) | **130s(2.2分钟, 快10倍)** |
| 画质(肉眼) | 有点糊、人物说话生硬 | **更清晰、更自然** |
| 口型同步 | 可对上 | 可对上 |
| 帧率/帧数 | 16fps / 77帧 | **25fps / 125帧** |
| 显存峰值 | 20.9G | **15.5G** |
| 长视频 | num_repeat 分段(线性耗时,22s音频=105分钟) | **原生分段续写**(官方配置跑过300s) |
| 多人对话 | ✗ | ✅(multi 模式, 权重已下) |
| 上游补丁需求 | 5 个 | 2 个(轻得多) |

S2V 定位降级为画质对照基线,生产不用。

## 权重四件套(共 ~90G + 顺带的 134G quant 变体, 均在 NFS)

| 组件 | 路径(/nfs-data/models/) |
|---|---|
| 底座(T5/CLIP/VAE + 原版DiT) | `Wan2.1-I2V-14B-480P` |
| 4步蒸馏 DiT(32.8G bf16) | `Wan2.1-Distill-Models/wan2.1_i2v_480p_lightx2v_4step.safetensors` |
| 音频 adapter(单人 9.3G) | `MeiGen-AI/InfiniteTalk/single/infinitetalk.safetensors`(另有 multi/ 和 quant_models/) |
| 音频编码器 | `TencentGameMate/chinese-wav2vec2-base` |

下载脚本:`scripts/download_infinitetalk.sh`(ModelScope 优先+续传+自检)。
注:官方内部配置用的"蒸馏DiT+音频adapter联合蒸馏"版(`ITAudioAdaptorV6.1`)未公开;本次用公开 4步蒸馏 DiT + adapter 组合,实测口型/画质均过关。

## 实测数据(480p, 5s, 4步, 无CFG, 单卡 block offload)

| 项 | 值 |
|---|---|
| 加载 | 1301s(NFS 冷读 ~40G, 一次性成本;dbg 轮 340s) |
| 生成 | **130s** |
| 显存峰值 | 15.5G(bf16 DiT block offload + adapter/CLIP 常驻) |
| 产物 | 896×448 / 125帧 / 25fps / 1.6MB,清晰自然 |

## A100 配置要点(`configs/infinitetalk/a100/infinitetalk_480p_single_distilled.json`)

照官方 `infinitetalk_480p_single_distilled.json` 改 4 处:
1. 四个权重路径换公开版(上表)
2. attn 全部 `torch_sdpa`(镜像无 sageattention;fp8/sage 是 Hopper/4090 路线)
3. **`rope_type: "torch"`**(默认 flashinfer 镜像没装 → 'NoneType' object is not callable,Z-Image 同款坑)
4. `cpu_offload: true + offload_granularity: block`(抄官方 4090 配方;InfiniteTalk 的推理类**原生支持 offload**,不像 S2V 要自己补)+ `t5_cpu_offload` + `video_duration: 5`

## 上游 bug(已修,挂卷生效,建议提 PR)

| # | 文件 | 问题 |
|---|---|---|
| 1 | `infer/infinitetalk/transformer_infer.py:236` | **torch_sdpa 与 sage/flash 的输出形状契约不一致**:audio cross-attn 传 4D (t,s,h,d),sage 返回拍平 2D,torch_sdpa 返回 3D → 下游 proj 的 torch.mm 崩("mat1 must be a matrix")。修复:proj 前按需拍平。官方只测过 sage/flash 路径 |
| 2 | (配置层)rope_type 默认 flashinfer | 未装即 NoneType not callable,配置显式 torch 即可(与 Z-Image 报告的坑同源,建议上游把默认改为带回退) |

## 深度测试(A/B/C 期, 2026-07-10)

### 蒸馏质量终审:零可见损失 ⭐

同素材对比 **40步无蒸馏基线(enable_cfg + 双引导, 2569s/43分钟)** vs **4步蒸馏(130s/2.2分钟)**:
**肉眼无差** → 蒸馏 20 倍加速白拿,生产一律用蒸馏版。
每步成本恒定(~31.6s/步, block offload 480p),总耗时差 = 步数差(40→4) + 免 CFG。

### 肉眼验收(2026-07-10, 全通过)

- **长音频(15s, 多段续写)**:段间过渡自然、无跳帧——motion_frame 上下文机制有效,长视频能力可用 ✅
- **720p**:画质无异常,自然 ✅
- 40步 vs 4步:无差(见上) ✅

### 性能数据(480p/5s, 单卡 block offload, 除注明外; 安静宿主)

| 实验 | 生成耗时 | 显存峰值 | 结论 |
|---|---|---|---|
| **4步蒸馏(生产形态)** | **130s** | 15.5G | 基准 ✅ |
| 40步无蒸馏基线 | 2569s(43分钟) | 15.8G | 画质=4步蒸馏 → 蒸馏 20× 白拿 |
| 长音频 15s(全音频) | 327s | 15.6G | 时长=min(音频,config);按秒成本 21.8s/s 反而更优(共享编码) |
| **720p 蒸馏** | **363s** | **27.1G** | 40G 卡从容;≈2.8×480p 耗时 |
| 单卡常驻(不 offload) | **OOM** | >39.5G | 蒸馏DiT 28.6G+adapter 9.3G+CLIP 装不下 → **单卡 bf16 必须 block offload** |
| 步数扫描 6/8 | 作废 | — | **infinitetalk 不吃请求体 infer_steps**(调度器按 config 初始化;wan 系吃)→ 服务化注意 |

### 压测①:单卡单实例稳态(生产形态, 安静宿主, N=6 丢预热)

| 指标 | 值 |
|---|---|
| **稳态耗时** | **103s/条**(5s 480p;预热首条 124s;iter2-6 全部 103s 零方差) |
| 显存峰值 | 16.0G |
| 容器内存峰值 | **72.0G**(锁页权重 38G + 运行时) |
| CPU 峰值 | 2605%(约 26 核, VAE/存盘段) |
| 宿主可用最低 | 169.1G |
| **节点容量红线** | **256G 内存节点最多 3 实例**(4×72G=288G 爆内存;瓶颈是内存不是显存!调度器看不见,人工限制) |
| 单节点吞吐 | 3 实例 × 60/103 ≈ **1.75 条/分 ≈ 105 条 5s 视频/小时** |

### 4卡 ulysses:bf16 判负(三连败, 机理完整)

1. **+cpu_offload**:每 rank 独立 pin 全量权重 ~60G×4=240G+ → host 内存打爆(锁页不可回收, 险僵机)
2. **裸常驻**:ulysses 只切激活、权重每卡全量复制 37.9G/卡 → 单卡都装不下(38.9G OOM 实证), 4 卡同理
3. **+load_from_rank0**:rank0 在 GPU 整装暂存全量权重再广播 → 38.9G 爆 40G 卡
→ InfiniteTalk bf16 在 40G 卡上没有多卡形态。

### int8 四后端对决:triton 逆袭夺冠 ⭐(2026-07-10 下午)

权重:`Wan2.1-Distill-Models/wan2.1_i2v_480p_int8_lightx2v_4step.safetensors`(17G, lightx2v 官方 int8 蒸馏 DiT)+ bf16 adapter,**常驻显存 30.5G**(免 offload)。

| dit_quant_scheme | 结果 | 结论 |
|---|---|---|
| int8-torchao | 228s(慢 2.2×) | weight-only 反量化开销, A100/ARM 吃不到 INT8 算力(Z-Image 同款教训) |
| int8-vllm | 崩(`scaled_int8_quant` None) | ARM vllm wheel 无编译算子 |
| int8-q8f / int8-sgl | 不可用 | 镜像未装 q8_kernels/sgl_kernel |
| **int8-triton** | **冷态 124s / 热态稳态 93.8s** | **真 INT8 路径,反超 bf16 offload 9%** ✅ |

画质:int8 与 bf16 同分辨率对比**无差**("运动中衣服文字微糊"两者同样存在,是 480p+生成模型共性,非量化损伤;该指标本身可作画质敏感评测点)。

### 🏆 生产形态终审(以此为准)

| | bf16 offload | **int8-triton 常驻(冠军)** |
|---|---|---|
| 稳态耗时(5s 480p) | 103s | **93.8s** |
| 显存 | 16.0G | 30.5G |
| **容器内存** | 72G(pin 权重) | **16G** |
| 实例数/256G 节点 | 3(内存红线) | **4**(卡数即上限) |
| **节点吞吐** | 1.75 条/分 | **2.56 条/分(+46%)** |

**部署定案:int8-triton 常驻 × 4 实例/节点 ≈ 154 条 5s 视频/小时/节点。**
备注:triton 首跑有 per-shape autotune(冷态 124s),实例启动后预热一发即稳。

### 加餐三连(2026-07-10 下午, 三节点并行, 全部通过)

| 实验 | 耗时 | 显存 | 结论 |
|---|---|---|---|
| **720p int8-triton 单卡常驻** | **345s** | 37.6G(距上限 2.4G) | 比 720p bf16 offload(363s)还快,720p 也归 int8 阵营 ✅ |
| **多人对话(multi 模式)** | **148s**(480p/5s) | 31.6G | multi adapter + 双音轨(逗号分隔 p1.mp3,p2.mp3)+ sample_shift 11;功能通,口型肉眼待验 |
| **4卡 ulysses int8 + load_from_rank0** | **稳态 54.4s**(零方差) | 27.9G/卡 | int8 使广播暂存(~28G)装进 40G 卡,bf16 时代的三连败全部翻案;**1.7× 单卡**(PCIe 无 NVLink 折损) |

### 多形态终极压测(T1-T6, 2026-07-10 下午, 五节点并行, 全部零方差)

| 实验 | 形态 | 稳态/条 | 关键结论 |
|---|---|---|---|
| T4 ⭐ | **单卡×4 实例满载并发**(480p) | **91-93s/实例** | **并发零干扰**——与独占(93.8s)持平,2.6 条/分/节点坐实 |
| T1 | 2卡 ulysses·同NUMA·独占 | 72.8s | 2卡加速比 1.29× |
| T2 | 2卡·跨NUMA | 75.8s | **NUMA 惩罚 +3s(+4%)** → 部署绑同NUMA卡对(GPU0,1/GPU2,3) |
| T3 | 双2卡实例并发(各绑一NUMA对) | 72.4-73s | 并发无劣化 |
| E3 | 4卡 ulysses(480p) | 54.4s | 加速比 1.73× |
| T5 | 720p 单卡稳态 | **312.8s** | (单发 345s 含 720p 形状 autotune 冷态) |
| T6 ⭐ | **720p 4卡 ulysses** | **136.7s** | **加速比 2.29×**——分辨率越高并行效率越好(DiT 平方涨 vs 串行线性涨),验证"算量放大→逼近线性"假说 |

Amdahl 拆账:480p 串行 ~41s + 可并行 ~53s;720p 串行 ~79s + 可并行 ~234s(推 2卡 720p ≈196s)。
串行部分 = VAE 解码 + 音频/CLIP 编码 + 存盘 + 通信,不随卡数下降 → **PCIe 无 NVLink 机器上多卡永远到不了线性,吞吐场景单卡×N 恒为最优**。

### 部署形态菜单(最终, 全部实测; 耗时均为 25fps 原生输出、未挂 RIFE)

| 档位 | 形态 | 单条延迟 | 节点吞吐 | GPU成本/条 |
|---|---|---|---|---|
| **480p 默认** | **单卡×4 实例** | 93s | **2.62 条/分** 🏆 | 1.55 GPU·分 |
| 480p 均衡 | 2卡(同NUMA)×2 实例 | 73s | 1.65 条/分 | 2.43 GPU·分 |
| 480p 低延迟 | 4卡×1 实例 | 54s | 1.10 条/分 | 3.63 GPU·分 |
| 720p 标准 | 单卡×4 实例 | 313s | 0.77 条/分 | 5.2 GPU·分 |
| **720p 低延迟** | 4卡×1 实例 | **137s** | 0.44 条/分 | 9.1 GPU·分 |
| 多人对话 | 单卡×4(multi 配置, 独立实例) | 148s | 1.62 条/分 | 2.5 GPU·分 |

RIFE 插帧:InfiniteTalk 原生 25fps 已流畅,**默认不挂**;若产品要求全平台统一 32fps,实例配置加 `video_frame_interpolation` 块 + 网关 target_fps 控制(+3-5s/条)。

### 长内容工况实测(T7/T8, 720p×4卡, 生产主场景)

| 时长 | 生成耗时 | 显存 | 容器内存峰值 |
|---|---|---|---|
| 5s(T6) | 136.7s | ~28G | ~16G |
| 15s(T7) | 382.5s | 31.3G | 108G |
| **60s(T8)** | **1455s(24.3分钟)** | 31.3G(全程平稳) | **184.9G ⚠️** |

- **时间严格线性**(~24s 每秒视频, 段数摊薄一次性开销后略优于外推)→ 按秒计费成立。
- **显存与时长无关**(分段生成, 每段恒 81 帧)✅。
- **⚠️ host/容器内存随时长增长**(解码帧累积):60s 已达 185G, 逼近 240G 容器上限 → **单任务音频时长红线 ≈ 60-75s**;更长内容(3-5分钟口播)网关须按 ~60s 切分多任务生成再拼接, 或待上游实现分段流式落盘(可改造点:每段 VAE 解码后即写盘释放, 参考 seedvr 的 segment 落盘模式)。
- **用户拍板的生产架构(2026-07-10)**:主场景=720p 长音频 → **4卡 ulysses ×1 实例/节点为主力**(60s 音频 24分钟 vs 单卡 63分钟, 单任务延迟优先);混合舰队保留 1-2 台单卡×4 节点服务 480p 短内容/多人/画布预览("480p 预览 → 720p 精渲染"流程)。

### 多实例/部署级发现(压测方法论级结论)

1. **同宿主邻居容器加载权重会把推理拖慢 2-4 倍**(130s→297-520s):block offload 每步从锁页内存搬 33G,与邻居的权重加载抢内存带宽。→ 性能数字必须安静宿主测;生产错峰启动;共置实例的稳态吞吐要打折估算。
2. **多实例并发冷启动会互相拖垮**:3 容器并发冷读 40G + NFS 下载 → 单容器加载 340s 恶化到 >1800s 超时。NFS 读带宽 + host 内存(每容器 pin 全量权重)双瓶颈。
3. **⚠️ 4卡 ulysses 禁止配 cpu_offload**:每 rank 独立 pin 全量权重(~60G×4=240G+),256G 内存必打爆(锁页不可回收,险些僵机;cgroup --memory=240g 保住了宿主)。**多卡一律常驻显存(NOOFL)**,权重 33G/rank 进 40G 显存。与 I2V"bf16 多卡 CPU OOM"同根,机理=per-rank pin。
4. NFS 运维:队列运行中不要覆盖 NFS 上正在执行的脚本(Stale file handle);新节点安装(读镜像 tar)与权重冷读错峰。

## 总表(参照 Wan2.2-I2V 报告样式)

### 权重路径(全在 NFS, 计算节点视角 /nfs-data/)

| 用途 | 路径 | 大小 |
|---|---|---|
| 底座(T5/CLIP/VAE) | `models/Wan2.1-I2V-14B-480P` | ~43G |
| 蒸馏 DiT bf16 480p/720p | `models/Wan2.1-Distill-Models/wan2.1_i2v_{480p,720p}_lightx2v_4step.safetensors` | 各 31G |
| **蒸馏 DiT int8 480p/720p(生产)** | `models/Wan2.1-Distill-Models/wan2.1_i2v_{480p,720p}_int8_lightx2v_4step.safetensors` | 各 16G |
| 音频 adapter 单人/多人 | `models/MeiGen-AI/InfiniteTalk/{single,multi}/infinitetalk.safetensors` | 各 9.3G |
| 音频编码器 | `models/TencentGameMate/chinese-wav2vec2-base` | 1.8G |

### 全量实验矩阵(480p/5s/4步蒸馏为基准, 安静宿主)

| # | 形态 | 生成耗时 | 显存峰值 | 容器内存 | 状态/结论 |
|---|---|---|---|---|---|
| 1 | 单卡 bf16 + block offload | 103s(稳态) | 16.0G | 72G | ✅ 亚军(内存红线→3实例/节点) |
| 2 | 单卡 bf16 常驻 | — | >39.5G | — | ❌ OOM(权重37.9G装不下) |
| 3 | **单卡 int8-triton 常驻** | **93.8s(稳态)** | 30.5G | **16G** | ✅ **生产默认**(4实例/节点) |
| 4 | 单卡 int8-torchao 常驻 | 228s | 31.3G | 15G | ❌ weight-only 慢2.2× |
| 5 | 单卡 int8-vllm / q8f / sgl | — | — | — | ❌ ARM 缺算子/未装 |
| 6 | **4卡 ulysses int8 + load_from_rank0** | **54.4s(稳态)** | 27.9G/卡 | 低 | ✅ 延迟档(1.7×单卡) |
| 7 | 4卡 bf16(offload/常驻/rank0 三姿势) | — | — | — | ❌ 三连败(pin爆host/权重超卡/广播爆卡) |
| 8 | 40步无蒸馏基线 | 2569s | 15.8G | 72G | ✅ 画质=4步 → 蒸馏20×白拿 |
| 9 | 长音频 15s(3段续写) | 327s | 15.6G | 72G | ✅ 段间自然;时长=min(音频,config) |
| 10 | **720p int8-triton 单卡** | **345s** | 37.6G | ~16G | ✅ 画质=bf16;距40G上限2.4G |
| 11 | 720p bf16 offload | 363s | 27.1G | ~72G | ✅ 备选 |
| 12 | **多人对话(multi)** | **148s** | 31.6G | ~16G | ✅ 双人口型各对各音轨 |

### 画质肉眼验收汇总(全通过)

| 对比 | 结论 |
|---|---|
| 4步蒸馏 vs 40步基线 | 无差(蒸馏零损失) |
| int8 vs bf16(同480p) | 无差("运动中衣服文字微糊"两者皆有,是480p+生成模型共性) |
| 720p int8 vs 720p bf16 | 无差 |
| vs Wan2.2-S2V | InfiniteTalk 更清晰自然(S2V 偏糊生硬) |
| 长音频段间过渡 | 自然无跳帧 |
| 多人口型 | 各对各的音轨 ✅ |

## 🚀 生产部署建议(最终)

1. **默认档(吞吐/成本最优)**:**单卡 int8-triton 常驻 × 4 实例/节点**,480p/25fps,93.8s/5s条,**2.56 条/分/节点(~154 条/小时)**;实例启动后**预热一发**(triton per-shape autotune,冷态124s→热态93.8s)。
2. **档位菜单**:720p 档(345s/条,int8 单卡)、多人档(148s/条)、延迟档(4卡 ulysses 54.4s/条,吞吐减半,仅延迟敏感场景按需开)。
3. **出包前置**:`infinitetalk/transformer_infer.py` 补丁(torch_sdpa 4D 形状)必须合入镜像(现靠挂卷);连同 S2V/SeedVR2 共 8 个补丁一起出包。
4. **服务化参数**:时长走请求体 `video_duration`(runner 读 input_info ✓);**步数只吃 config**(蒸馏4步锁死,不暴露);`target_fps` 显式传(单独用=32 走 RIFE,画布中间步=16);计费按音频秒数(时长=min(音频,video_duration))。
5. **部署纪律**:多实例**错峰启动**(NFS+内存带宽);节点上限 4 实例(int8 形态;bf16 形态只有 3);压测/回归必须在安静节点跑。
6. **1080p**:不由生成出,走"720p 生成 → SeedVR2 3B 超分 → RIFE"链路(canvas 设计文档 §5.8)。

## 待办 / 下一步

- [ ] 720p 版配置(`infinitetalk_size: infinitetalk-720` + 720p 蒸馏 DiT——Distill-Models 里有 720p 版可补下)
- [ ] 长音频实测(22s seko 全长,验证原生分段续写的连贯性与耗时曲线)
- [ ] 多人模式(multi adapter 已在 NFS)
- [ ] 服务化:gpustack 部署配置 + new-api 能力注册(音频数字人原子能力,引擎=InfiniteTalk)
- [ ] quant_models/ 里 134G 用不上的 fp8 变体可清理(int8 变体可留作后续实验)
