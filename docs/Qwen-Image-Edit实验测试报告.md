# Qwen-Image-Edit-2511 图生图/编辑 实验测试报告

> 模型:Qwen-Image-Edit-2511(i2i 图像编辑),LightX2V server(Docker)
> 平台:4×A100 PCIE 40G · 鲲鹏920 ARM · 节点 `dev-gpustack-a100-0001`
> 镜像:`crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest`
> 日期:2026-07-05 · 配套:`Qwen-Image-实验测试报告.md`(t2i/加速/压测)、`qwen-image-单机部署方案.md`
> 一句话结论:**Edit-2511 六类编辑(风格/增物/改色/换背景/加文字/改光照)全部可用。⭐ Lightning 离线合并 8步 + `qwen25vl_cpu_offload:false` = 生产最优,单张 38s(base 25步 ~2min)。之前误判的"hang"实为 `cpu_offload` 把文本编码器 offload 到 ARM CPU、VLM 前向跑 7min(非死锁),`qwen25vl_cpu_offload:false` 让其留 GPU → 文本编码 7min→11s。attn_type 必须 torch_sdpa;prompt 含引号须用 json.dumps 转义。**

---

## 1. 环境与权重

| 项 | 值 |
|---|---|
| 基座 | `/data/models/Qwen-Image-Edit-2511`(bf16,transformer 5 分片) |
| 配置 | `/data/lightx2v_configs/qwen_edit_2511_a100_base.json` |
| 启动 | `--model_cls qwen_image --task i2i --model_path /data/models/Qwen-Image-Edit-2511`,单卡 GPU,`--memory=110g --init` |
| 端点 | `POST /v1/tasks/image/`,请求体带 `image_path`(URL/base64/本地路径)+ `prompt`(编辑指令) |

配置关键字段(与 t2i 同一套 A100 适配,见主报告 §5 踩坑):
```jsonc
{
  "infer_steps": 25,
  "prompt_template_encode": "<|im_start|>system\nDescribe the key features of the input image ... alter or modify ...<|im_end|>...",
  "prompt_template_encode_start_idx": 64,
  "resize_mode": "adaptive",
  "attn_type": "torch_sdpa",          // ⚠️ 非 sage_attn2(黑图)
  "rope_type": "torch",
  "enable_cfg": true, "sample_guide_scale": 4.0,
  "CONDITION_IMAGE_SIZE": 147456, "USE_IMAGE_ID_IN_PROMPT": true,
  "cpu_offload": true, "offload_granularity": "block"
}
```

> **红线**:edit 也是 offload 重负载,计入"单机 ≤3 offload 实例(内存,混合流量选 2 稳)"配额(见部署方案)。本实验起 1 个 edit 实例(GPU2,端口 8002),同时保留 1 个 t2i(GPU0)。

## 2. 源图(t2i merged8 生成,多分辨率多题材)

存于 `/data/_editsrc/`,均 2.0-2.3M、无转置:

| 文件 | 题材 | 分辨率 |
|---|---|---|
| src_0_1x1 | 人像(雀斑少女) | 1328×1328 |
| src_1_16x9 | 山湖日出 | 1664×928 |
| src_2_3x2 | 红色敞篷车 | 1584×1056 |
| src_3_4x3 | 北欧客厅 | 1472×1104 |
| src_4_9x16 | 猫看窗外 | 928×1664 |
| src_5_16x9 | 汉堡薯条 | 1664×928 |
| src_6_3x4 | 灯塔悬崖 | 1104×1472 |
| src_7_1x1 | 水果静物 | 1328×1328 |

## 3. 编辑指令配对(6 条,覆盖 5 类)

| # | 源图 | 编辑类型 | 指令 |
|---|---|---|---|
| 1 | src_1(湖景) | 风格转换 | turn this into an oil painting with thick brush strokes |
| 2 | src_3(客厅) | 局部增加 | add a fluffy orange cat sleeping on the sofa |
| 3 | src_2(红车) | 颜色替换 | change the car color from red to glossy dark blue, keep everything else |
| 4 | src_0(人像) | 背景替换 | replace the background with a sunny tropical beach, keep the person unchanged |
| 5 | src_5(汉堡) | 加文字(Qwen 强项) | add a small chalkboard sign that reads "BURGER $9" |
| 6 | src_4(猫) | 光照/时间 | change the scene to night time with moonlight through the window |

## 4. 结果

**6/6 全部出图成功,肉眼无明显崩坏。** 单卡 GPU2、base 25步+CFG,每张 ~2min。

| # | 类型 | 源图 | base 出图/耗时 | **merged 出图/耗时** | 肉眼评价 |
|---|---|---|---|---|---|
| 1 | 风格转换(→油画) | src_1 湖景 | 2.4M / ~2min | 2.4M / **24s** | ✅ 结构保持、质感到位 |
| 2 | 局部增加(沙发加猫) | src_3 客厅 | 1.9M / ~2min | 1.8M / **24s** | ✅ 加物融入场景 |
| 3 | 颜色替换(红→蓝车) | src_2 红车 | 1.8M / ~2min | 1.9M / **25s** | ✅ 改色、其余保持 |
| 4 | 背景替换(→沙滩) | src_0 人像 | 1.7M / ~2min | 1.8M / **27s** | ✅ 主体保持、背景换 |
| 5 | 加文字("BURGER $9") | src_5 汉堡 | 1.8M / ~130s | 1.8M / **24s** | ✅ 正常(文字精度待放大看) |
| 6 | 光照/时间(→夜景月光) | src_4 猫 | 1.2M / ~2min | 1.5M / **25s** | ✅ 全局光照重构 |

> **merged 6/6 全成,热态 21.6s/张**(压测均值;连发 24-27s)。比 base ~2min **快 5.4×**。均须 `qwen25vl_cpu_offload:false`。
> **质量(肉眼逐图对比,同源图/prompt/seed)**:**base(25步)保真度略优于 merged(8步)**——符合蒸馏预期,8步省时间换一点细节。差距不大,多数场景 merged 够用;**质量敏感场景(精修、细节/文字关键)用 base**。

> 评价为**粗粒度肉眼过**(用户目视 6 张无崩坏),非严格盲评。若上生产,建议对**文字精确度(#5)**、**主体一致性(#4)**、**细节保真** 再放大逐图复核。产物 `/data/_editout/edit_N.png`,均 >50KB 无黑图。

## 5. 踩坑

- **attn_type 必须 torch_sdpa**(sage_attn2 黑图,同 t2i)。
- **image_path 支持容器内本地路径**(`/data` 已挂载),也可 URL/base64。
- `resize_mode=adaptive`:输出尺寸随输入图自适应(6 张各分辨率均正常)。
- **⚠️ prompt 含双引号会破坏 JSON**:指令 `reads "BURGER $9"` 直接手拼进 curl `-d '{"prompt":"..."}'` → 内层 `"` 未转义 → 服务器拒收、返回空 task_id(#5 首次失败即此)。**请求体用 `python json.dumps` 构造,别手拼字符串**——接 GPUStack/new-api 时同理,适配器侧必须正确转义用户 prompt。
- **⚠️ `aspect_ratio` 默认 16:9 把输出塌成横图**:不传 aspect_ratio → 所有输入(方/竖图)输出恒定 1664×928。**edit 请求必须显式带 aspect_ratio**(指定比例)或 `""`(跟随输入)。详见 §9。
- **⚠️ 误判 hang**:文本编码器 offload 到 CPU 时任务长时间 `processing`+GPU≈0,极像死锁,实为 7min CPU 编码;poll 超时要给足、先看 pipeline 分解。见 §6。
- **⚠️ 内存监控别用 `free` 的 used 列**:offload 的 DiT 存在 **共享内存(Shmem)**,`free` used 列**不计 shmem**,会严重低估(实测 3 副本 used 显 19G、实为 Shmem 180G)。量真实内存用 `/proc/meminfo` 的 `Shmem` + `MemAvailable`。**每 offload 实例 ~60G shmem**,单机(251G)副本上限 ≈ 3(第 4 个全局 OOM-kill)。**注:t2i 报告的"主机内存"列已同步修正为 Shmem 口径(t2i 也是 offload,~60.2G shmem/实例);t2i/edit 峰值副本数都随分辨率(16:9 3副本、1:1 2副本)。**

## 6. 结论 / 生产建议

**一句话:Qwen-Image-Edit-2511 base 六类编辑(风格/增物/改色/换背景/加文字/改光照)全部可用,~2min/张,base 可直接上生产。**

- **可用性**:6/6 通过,肉眼无崩坏;各编辑类型都能响应指令且保持主体结构。适合做通用图像编辑后端。
- **速度**:base 25步+CFG ~2min/张,比 t2i base 更慢(多了图像编码 + i2i 条件)。**单卡串行**,和 t2i 共守"单机 ≤3 offload 实例(内存,混合流量选 2 稳)"红线。
- **⭐ 提速路径(已跑通)**:离线合并 `Qwen-Image-Edit-2511-Lightning-8steps-bf16`(720/720 命中,38.1G merged 文件)→ 8步 merged 配置 + **`qwen25vl_cpu_offload:false`** → **单张 38s**。DiT(8步)23.8s + 文本编码 11s + VAE 3.3s;GPU 显存峰 ~20G/40G。
- **"hang"真相(重要教训)**:之前判 merged edit "hang" 是**误判**。它一直在跑、会完成,只是 `cpu_offload:true` 把文本编码器 Qwen2.5-VL(15G)也 offload 到 **ARM CPU**,VLM 前向(edit 还要编码输入图)在 CPU 上跑 **~7min(436s)**,而 poll 超时 200s → 误以为死锁。pipeline 分解实测:481s = 文本编码 436s + DiT 42s + VAE 2s。**加 `qwen25vl_cpu_offload:false` 让文本编码器留 GPU,7min→11s,总时 481s→38s(12.6×)。**
- **诊断经验**:edit「卡住」先看 pipeline 分解(`docker logs | grep 'cost.*seconds'`)是不是文本编码占大头 + GPU util 是否 0(=在 CPU 跑),别急着判 hang;poll 超时给足(edit 冷态可 >200s)。
- **生产结论:edit 用 merged 8步 + `qwen25vl_cpu_offload:false`(38s),base 作全质量备选(~2min)。**
- **接入注意**:prompt 转义(§5)、异步轮询任务态在实例内存(多副本靠客户端容忍 404)。

## 7. base vs merged 8步 对比 —— ⭐ 热态稳态压测(`test_qwen_image_stress.sh MODE=edit/edit_m`)

> 方法同 t2i:单容器连发 6 张(同一输入图 src_1 16:9)、丢首张、取后 5 张均值。均 `qwen25vl_cpu_offload:false`。

| 档 | 步数/CFG | 加载 | **热态稳态** | GPU util峰 | 显存峰(稳态/冷) | 结论 |
|---|---|---|---|---|---|---|
| base(25步) | 25+CFG(50前向) | 121s | **116.7s** | 100% | 19.8G / 27.5G | 全质量备选 |
| **merged Lightning(8步)** | 8 无CFG(8前向) | 86s | **21.6s** | 99% | 19.8G / 27.5G | ✅ **生产最优,快 5.4×** |

- 两者 **GPU util 99-100% = compute-bound**(文本编码器已在 GPU,瓶颈是 DiT 步数);merged 8步/无CFG vs base 25步/CFG = **5.4× 提速**(≈ 50/8 前向比)。
- 显存两者相同(~20G 稳态,27.5G 冷峰),40G 富余一半——**edit 单卡显存不是约束**(和 t2i 一致)。
- 稳定性极佳(merged ±0.05s、base ±1s)。

## 8. 多副本吞吐(`test_qwen_image_4cards.sh MODE=edit_m`)⭐

| 副本 | 吞吐 | 相对单副本 | 真实主机内存(Shmem+used) | 每卡显存 | 状态 |
|---|---|---|---|---|---|
| 1 | 0.046 img/s | 1× | ~60G | ~20G | ✅ |
| 2 | 0.091 img/s | **1.98×** | ~120G | ~20G | ✅ |
| **3** | **0.126 img/s** | **2.74×** | **~180G(available 剩 50G)** | ~20G | ✅ **实用上限** |
| 4 | — | — | 需 ~240G > 251G | — | ❌ **加载即 OOM-kill**(全局内存不足) |

**edit 多副本近线性扩展到 3 副本,4 副本被主机 251G 内存卡死。(峰值副本数随分辨率:16:9 类中低分辨率 3 副本近线性、1:1 大图 2 副本封顶——t2i/edit 同规律,见 t2i 报告 §4.3 隔离测试。)**
- **吞吐扩展好的根因**:edit 文本编码器常驻 GPU(`qwen25vl_cpu_offload:false`),DiT offload 预取占 ARM CPU 但争抢轻 → 1→2→3 副本近线性(1.98× / 2.74×)。t2i 把文本编码器 offload 到 CPU,2 副本抢 CPU 就打折、3 副本负优化。
- **⚠️ 4 副本硬限(实测 OOM)**:每 edit 实例 offload DiT 占 **~60G 共享内存(Shmem,不可回收)**;3 副本 Shmem 已 **180G**、available 仅剩 **50G**,第 4 个要 60G → **内核全局 OOM-kill(`constraint=CONSTRAINT_NONE / global_oom`,exit 137)**。非 cgroup(`--memory`)限制、非 drop_caches 可解(180G 是 shmem,可回收页缓存才 ~10G)。**要 4 卡需加内存条。**
- **注意监控口径**:`free` 的 used 列**不含 shmem**,会严重低估(脚本原报 19.7G,实为 180G);量真实内存看 `/proc/meminfo` 的 **Shmem** + `MemAvailable`。
- 队列:`max_queue_size` 默认 10;并发请求按副本数分摊(单实例灌 >10 会溢出,非真失败)。
- **⚠️ 试过 int8 解内存换 4 卡,net 亏**:把 merged DiT 离线量化 int8(38G→20G),①no-offload:装进 GPU(idle 35.5G)、shmem 归零,但**生成反量化 buffer 顶到 40.3G OOM**;②int8+offload:GPU 峰 19G 不 OOM、shmem 砍到 30G/实例(可上 4 副本),但**每图 72s(慢 3.3×,int8-torchao 在 A100 无 INT8 tensor core)** → **4 副本 int8 = 0.056 img/s < 3 副本 bf16 = 0.126**。**结论:int8 弃用,bf16 3 副本(0.126)就是 edit 吞吐上限。**

## 9. 输出尺寸 / 比例控制 ⭐(重要接入规范)

**edit 输出尺寸完全由请求的 `aspect_ratio` 决定,与输入图比例无关**(源码 `qwen_image_runner.get_custom_shape`):

| 输入 | `aspect_ratio` | 输出 | 说明 |
|---|---|---|---|
| 1328² | `"1:1"` | 1328×1328 | 显式比例 → 固定桶(~1.5MP) |
| 1328² | `"9:16"` | **928×1664** | **可强制任意比例,与输入无关** |
| 1328² | `""`(空) | 1024×1024 | 跟随输入比例,但缩到 ~1MP(偏小) |
| 928×1664 | `""`(空) | 768×1376 | 跟随输入,~1MP |
| 928×1664 | `"9:16"` | 928×1664 | 显式比例 |

- **⚠️ 默认坑**:`ImageTaskRequest.aspect_ratio` schema 默认 `"16:9"` → 不传时**所有图恒定塌成 1664×928**(方图/竖图全变横图)。之前 6 张编辑就都塌了。
- **接入规范**:
  - 要**指定输出比例** → 传对应 `aspect_ratio`(7 选:`16:9/9:16/1:1/4:3/3:4/3:2/2:3`,~1.5MP)。
  - 要**跟随输入原比例** → 传 `aspect_ratio:""`(注意缩到 ~1MP,偏小)。
  - 要**非标准尺寸** → 传 `target_shape:[h,w]`。
  - **别用默认**(=16:9),new-api/GPUStack 适配器须显式带 aspect_ratio。

## 10. 全配置实测对比表 ⭐(和 t2i 报告 §2 同结构)

> 分辨率 1:1(源图 16:9,1664×928);生成为热态。均须 `qwen25vl_cpu_offload:false`。

| # | 配置 | 步数/CFG | 卡/并行 | 加载 | 显存峰(稳/冷) | 真实主机内存(Shmem+used) | 生成热态 | 吞吐(img/s) | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **merged 单卡** ⭐ | 8 无CFG | 1 | 86s | 20G / 27.5G | ~60G | **21.6s** | 0.046 | ✅ **延迟最优** |
| 2 | merged 2 副本 | 8 | 2×1 | 57s(热) | 20G/卡 | ~120G | ~22s | **0.091(1.98×)** | ✅ 吞吐 |
| **3** | **merged 3 副本** ⭐ | 8 | 3×1 | 97s | 20G/卡 | **~180G(剩50G)** | ~24s | **0.126(2.74×)** | ✅ **吞吐上限** |
| 4 | merged 4 副本 | 8 | 4×1 | OOM | — | 需 ~240G>251G | — | — | ❌ 全局 OOM-kill(每实例 60G shmem×4 超 251G) |
| 5 | **base 全质量 单卡** | 25+CFG | 1 | 121s | 20G / 27.5G | ~60G | **116.7s** | 0.009 | ✅ 质量档(保真略优于8步) |
| 6 | merged **无** `qwen25vl_cpu_offload:false` | 8 | 1 | — | — | — | **481s** | — | ❌ 反例:文本编码器落 CPU 跑 7min |
| 7 | 多卡 TP / ulysses | — | 2-4 | — | — | — | — | — | ❌ 继承 t2i(TP慢1.43×/ulysses奇数崩/4卡死机) |
| 5b | base 未蒸馏 8步 | 8 | 1 | — | 20G | ~60G | 22.2s | — | ❌ 快但**欠拟合糊**(base 没为8步训练;蒸馏才行) |
| 6 | merged **无** `qwen25vl_cpu_offload:false` | 8 | 1 | — | — | — | **481s** | — | ❌ 反例:文本编码器落 CPU 跑 7min(误判成 hang) |
| 7 | 多卡 TP / ulysses | — | 2-4 | — | — | — | — | — | ❌ 继承 t2i(TP慢1.43×/ulysses奇数崩/4卡死机) |
| 8 | int8 no-offload(离线量化 merged→int8 20G) | 8 | 1 | 快 | idle 35.5G,**生成顶 40.3G OOM** | Shmem≈0 | — | — | ❌ 装得下但生成反量化 buffer 顶爆 40G |
| 9 | int8 + offload | 8 | 1 / 4 | — | GPU峰 19-33G | Shmem 30G/实例 | 热态 **49s** | **4副本 0.079(实测)** | ❌ 内存减半能上4卡,但每图慢 2.3×,0.079 < bf16 3卡 0.126 |
| 10 | **lazy_load(磁盘 offload,权重在 SFS)** | 8 | 1 | 快 | ~20G | **Shmem 0.5MB!** | **132s** | ~0.03 | ❌ shmem 归零但每图慢 6×(per-block 开销,GPU util 1%) |
| 11 | **lazy_load(block 拷本地 VBS 盘)** | 8 | 1 | 快 | ~20G | **Shmem 0.5MB** | **126s** | ~0.03 | ❌ 本地盘也 126s ≈ SFS 132s → **瓶颈是 per-block 开销不是磁盘** |

**四条硬结论:**
1. **延迟最优 = merged 8步 单卡(21.6s)**;质量敏感用 base 25步(116.7s,保真略优)。
2. **吞吐最优 = 单机 3 副本 0.126 img/s(2.74× 近线性)**——edit 文本编码器在 GPU、不抢 ARM CPU,比 t2i(2副本封顶)扩展好;**卡在 60G-pinned×3=180G 内存,4 副本全局 OOM**。
3. **必开 `qwen25vl_cpu_offload:false`**(否则文本编码 7min);多卡/int8/lazy_load 全部实测判负。
4. **想破 3 副本上限(内存)的三条路全试全输**:int8(慢2.3×)、lazy_load(慢6×)、多卡(TP慢/ulysses崩)。要真 4 卡满速只能**加内存条**(bf16 4×60G pinned)或**改引擎让多进程共享 pinned 权重**(工程量大)。

**edit 用到的优化手段(对齐 t2i)**:`attn_type=torch_sdpa`(防黑图)· `rope_type=torch` · `cpu_offload=block`(DiT)· **`qwen25vl_cpu_offload=false`(文本编码器留GPU,edit 关键,7min→11s)** · Lightning 离线合并(`dit_original_ckpt`)· `aspect_ratio` 显式指定(防默认16:9塌图)。

## 11. 探索历程 ⭐(问题 → 尝试 → 结果 → 再优化,全程复盘)

这一节按时间线记录整个调优过程,包括**走过的弯路、误判、和最终排除的负结果**——供后人少踩坑。

### 阶段 0:起点
- edit base(25步+CFG)能跑,6 类编辑肉眼可用,但 **~2min/张太慢**。目标:上 Lightning 蒸馏加速,并摸清副本/内存边界。

### 阶段 1:Lightning 离线合并 → 误判成 "hang"
- **做法**:把 `Qwen-Image-Edit-2511-Lightning-8steps-bf16` LoRA 用 `merge_qwen_lora.py` 离线焊进 base DiT(720/720 命中,产出 38.1G merged 单文件),配 8步 `dit_original_ckpt` 配置。
- **问题**:merged 8步推理**长时间 `processing`、GPU util≈0**,poll 200s 超时 → 我判成"死锁/引擎 bug",还差点判合并失败。
- **排查**:拉全 pipeline 日志分解,发现**任务其实在跑、会完成**,只是 `RUN pipeline cost 481s`,其中 **DiT 只 42s、文本编码占 ~436s(7min)**。
- **根因**:`cpu_offload:true` 把**文本编码器 Qwen2.5-VL(15G)也 offload 到 ARM CPU**,VLM 前向(edit 还要编码输入图)在 CPU 上跑 7 分钟。读代码 `qwen25_vlforconditionalgeneration.py:73` 有独立开关 `qwen25vl_cpu_offload`。
- **解决**:配置加 **`qwen25vl_cpu_offload:false`**(文本编码器留 GPU,才 15G,显存够)→ 文本编码 7min→11s,**总时 481s→38s(冷)/21.6s(热),12.6×**。
- **教训**:edit「卡住」先看 pipeline 分解 + GPU util,别急着判 hang;poll 超时要给足。

### 阶段 2:多副本吞吐 —— edit 意外比 t2i 强
- **做法**:`test_qwen_image_4cards.sh` 起 N 个单卡实例压吞吐。
- **结果**:1→0.046 / 2→0.091(1.98×)/ 3→**0.126(2.74×)**,近线性。**edit 扩展性远好于 t2i**(t2i 2副本仅 1.67×、3副本负优化)。
- **原因**:edit 文本编码器在 GPU(不抢 ARM CPU),只有 DiT offload 预取占 CPU,争抢轻。

### 阶段 3:4 副本 OOM → 挖出 shmem 真相
- **问题**:第 4 副本加载崩,`docker inspect` = `OOMKilled=true exit=137`,dmesg = `global_oom / shmem-rss 35-60G`。
- **弯路**:我先按 `free` 的 used 列判断内存,报"3副本才19G",被质疑后重查 `/proc/meminfo`:**Shmem=180G**!`free` used **不计 shmem**,严重低估。
- **根因**(读代码 + agent 分析):`cpu_offload` 下整份 38G DiT 被 `torch.empty(pin_memory=True)` 拷成 **CUDA pinned 内存(common/ops/utils.py:90)**,以 shmem 计,**每进程 ~60G、不可跨进程共享**。3副本=180G,4副本=240G+系统>251G → 全局 OOM。
- **结论**:edit 单机吞吐上限 = **3 副本(0.126)**,被 251G 物理内存卡死。

### 阶段 4:优化尝试 A —— int8 量化减内存
- **动机**:int8 DiT 只 20G(bf16 38G 的一半),内存减半 → 或可上 4 卡。
- **做法**:`converter.py --linear_type int8 --save_by_block` 把 merged DiT 量化成 int8(19.6G)。
  - **A1 int8 no-offload**(装进 GPU):idle 35.5G、**Shmem 归零**,但**生成时反量化 buffer 顶到 40.3G → OOM**。
  - **A2 int8 + offload**:GPU 峰 19G 不 OOM、**Shmem 砍到 30G/实例**,4 副本全起(120G<251G)→ **能上 4 卡!** 但热态 **49s/张**(int8-torchao 在 A100 无 tensor core,慢 2.3×)。
- **结果**:int8 4副本实测 **0.079 img/s < bf16 3副本 0.126**。**内存解了、能上 4 卡,但每图慢太多,4 卡追不回** → 弃。
- **弯路**:我一度拿 int8 冷态 72s 外推 4×=0.056,被质疑"压测了没/算冷启没",重跑热态压测才得 49s/0.079 实测。另外 stress 脚本注入逻辑会误删配置里的 int8 字段(测出 base bf16 8步 22s),也修了。

### 阶段 5:优化尝试 B —— lazy_load 磁盘 offload(想让多进程共享权重)
- **动机**:agent 分析指出 pinned 内存不可共享是 4 卡瓶颈;`lazy_load` 模式只常驻 2-block 双缓冲、权重运行时从磁盘按需读,理论上多进程共享 page cache。
- **做法**:`converter.py --save_by_block`(不量化)切 bf16 block,配 `lazy_load:true`。
- **结果**:**Shmem 从 60G → 0.5MB!**(内存彻底解决)——但**每图 132s(慢 6×)**,GPU util 1%(全程等 I/O)。
  - 怀疑 SFS 慢 → 把 block 拷到本地 VBS 盘重测:**126s**,和 SFS 132s 几乎一样。
- **根因**:瓶颈**不是磁盘速度**,是 lazy_load **每步每 block 的 `safe_open + get_tensor + pin copy`(每图 480 次)**串行开销,换本地盘也省不掉;DiT 从 15s 涨到 70s。
- **结果**:每图 126-132s,4 卡也才 ~0.03 img/s,**比啥都差** → 弃。

### 阶段 6:定论
- 三条破 3-副本上限的路(int8 / lazy_load / 多卡)**全部实测判负**,没一条能超过 bf16 pinned 3 副本 0.126。
- **edit 生产终极方案:延迟走 merged 单卡 21.6s,吞吐走 bf16 pinned 3 副本 0.126 img/s。** 想真 4 卡满速只能加内存条或改引擎共享 pinned 权重。

### 一路踩的坑汇总(除上面的根因外)
- sage_attn2 黑图 → torch_sdpa;`aspect_ratio` 默认 16:9 塌图 → 显式传;prompt 含双引号破坏 JSON → `json.dumps` 构造;docker 容器僵尸难杀 → kill+sleep+rm+`--init`,`live-restore` 未开(D 态卡 NFS);`--port` 写错(8005 vs 映射 8000)→ 空响应误判 OOM;绑核(cpuset+OMP)反而慢 22%(offload 吃核数);`free` used 漏 shmem 严重低估内存;stress 注入误删 int8 字段。

## 12. 待办
- [x] ~~Edit-2511-Lightning 离线合并 + 跑通~~:合并成功、`qwen25vl_cpu_offload:false` 后 **38s 跑通** ✅
- [ ] base(25步)vs merged(8步)**编辑保真度** 逐图盲比(耗时已知,看质量差)。
- [ ] 放大逐图复核质量:文字精确度(#5)、主体一致性(#4)、细节保真。
- [ ] 6 条编辑指令用 merged 8步 全量重跑(现只验了 #1 油画),补 §4 的 merged 结果列。
- [ ] 更多编辑类型:去物体、换姿势/表情、多轮连续编辑、局部 mask(若引擎支持 image_mask)。
