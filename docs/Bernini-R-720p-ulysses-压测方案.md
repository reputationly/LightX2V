# Bernini-R 720p 4卡ulysses 压测与插帧实验方案

> 日期:2026-07-20 · 环境:dev-gpustack-a100-0021~0025(20×A100 PCIE 40G,**无 NVLink**,鲲鹏 ARM,256G 内存/swap 仅 3G)
> 镜像:`lightx2v:arm64-a100-latest` · 模型:Bernini-R-14B int8-triton(继承 Wan2.2-T2V-A14B,双专家 MoE 4步蒸馏)
> 定位:720p 生产形态已拍板 = **4卡 ulysses 直出**(卡充足不省卡,见 memory `bernini-720p-ulysses4-production`)。本方案把这条路**压到底**:锁热态耗时、验插帧、量吞吐、探边界、留画质证据。
> 状态:**方案(未执行)**。下面每组给 目的/配置/命令/度量/判据,结果表留空待填。

---

## 0. 已知基线(单次冷跑,dev-0024,待本方案复核)

| 项 | 值 | 备注 |
|---|---|---|
| 模型冷加载(NFS) | 352s | 一次性;server 常驻后不复现 |
| DiT step1(high) | **24.8s** | **含 triton per-shape autotune 冷编译**,不可摊薄到热态 |
| DiT step2(high) | 10.7s | |
| DiT step3(low) | 13.2s | |
| DiT step4(low) | 10.1s | |
| VAE decode | 2.9s | 未开 tiling |
| 单条 end2end(rank0) | ~61s(冷) | |
| 产物 | 720×1280 / 81帧 / 16fps / 11MB | |

→ 关键悬念:**热态整条到底多少**(估 ~45s,`step1` 掉到 ~10s)?这是对外报的生产数,E0 必须钉死。

---

## 0.5 关联报告的已知结论(**直接改本方案假设,别再当未知**)

来自同架构(Wan2.2-A14B MoE 双专家 int8)的既有实测,Bernini-R = 标准 T2V 变体,数值可迁移:

- **ulysses 多卡加速是实打实的,不是"全灭"**:I2V 480p 单卡 97s → 4卡 36s(~2.7×);VACE R2V DiT 1080s → 317.6s(**3.37×**,DiT 占比 87% Amdahl 友好)。→ **H3/E4 的假设从"担心通信吃光"上调为"量到底 ~3×"**。记忆里 W2"通信全灭"指的是**别的并行/跨节点/load_from_rank0**,不是单机 seq-p ulysses。
- **720p×帧数的显存曲线已有 I2V 实测表**(832×1104,同 0.92MP 面积预算,int8 4卡):81帧 35.3G / **121帧 37.9G✅** / **161帧 40.3G 顶满** / 201帧 OOM。→ **E2 直接有预期数**:Bernini 是纯 t2v(无 image cond,比 i2v 更轻),121 帧几乎必过,大概率能摸到 161。
- **时长上限 = 显存上限,画质不先崩**:i2v 外推到 161 帧(720p 10s)肉眼仍正常。Bernini 纯 t2v 无 flf2v 的末尾锚点问题,长视频画质应更稳。
- **RIFE 插帧已定案可用**:VACE R2V 16→32fps 插帧仅 **+~34s**(480p),"丝滑很多",挂 RIFE 定案;既定铁律"只在链路最后一步插"。→ E1 默认 32fps 正确;720p 的插帧代价更高(全分辨率帧)需实测。配置/脚本有先例:`wan22_moe_vace_a100_int8_rife.json` / `vace_rife.sh`。
- **铁律(VACE §六,沿用)**:多卡**禁 cpu_offload**;MoE **禁 load_from_rank0**;**triton 首请求需预热**(= E0 丢第1条);**加载 303-352s 是 4 rank 争 NFS 带宽 → 生产常驻实例,不按请求拉起**。
- **⚠️ 对编辑线的强预警**:VACE 720p 4卡常驻 **三审死刑**——因为 VACE 有"参考图/控制视频 latent 拼接"的额外上下文分支,38G 常驻 + 720p 序列瞬时峰把 40G 挤爆(P10 两轮补丁仍差 774MB,回滚)。**Bernini 纯 t2v 720p ulysses 能跑通(本方案基线,已出片),恰恰因为它没那条分支**。→ **编辑线(v2v/i2i)一旦 concat source latents,就变得像 VACE,720p 编辑很可能撞同一堵 OOM 墙**。编辑线的 720p 要么走单卡+offload(VACE 4步LoRA 单卡 258s 的路),要么 480p 编辑 + SR。**这条决定编辑线 720p 的形态选择,必须早验。**

---

## 1. 待验证的关键问题(假设)

- **H1 热耗时**:server 常驻下第 2+ 条稳态 end2end ≈ ?(假设 ~40-45s,step1 从 24.8s→~10s)。冷/热差 = triton 冷编译代价,量出来写进 SLA(server 崩溃重启首条会再冷)。⚠️ 注意 I2V 报告 720p 81帧 int8 4卡是 **87s**——比我估的 45s 慢一倍,但 i2v 带 image-condition latent(in_dim 36 拼图)更重;Bernini 纯 t2v 应更快。E0 的实测数以谁为准要**当场核**(45s 还是 87s 差一倍,直接影响容量规划)。
- **H2 插帧(RIFE)**:720p 16fps → 32/48/60fps,画质增益 vs 加时?VACE 480p 16→32 只 +34s;**720p 全分辨率帧插值代价更高**,且 RIFE 在解码后帧序列上跑,720p×高帧数**可能显存爆**——要测 OOM 边界与耗时曲线。
- **H3 通信代价**:已知 ulysses 在本集群有效(I2V 2.7× / VACE 3.37×)。本组只需**量 Bernini 720p 的具体加速比**,确认 720p 长序列下 all-to-all 占比是否恶化(序列越长通信量越大)。
- **H4 吞吐口径**:4卡 ulysses 单节点**并发=1**(一条吃满 4 卡)。节点吞吐(条/分)对比 480p 单卡×4 并发,谁高?卡充足下 ulysses 的"每条更快但不并发"净吞吐如何?
- **H5 边界**:720p×81 峰值显存 / 40G 余量?**121/161 帧**——I2V 表明 121=37.9G✅、161=40.3G 顶满、201 OOM;Bernini 纯 t2v 更轻,**验证是否 121 稳过、161 摸顶、201 死**。VAE tiling 开关对显存/耗时的影响?
- **H6 长稳**:server 常驻连续出片 1h+,有无显存泄漏 / 耗时漂移 / NCCL 掉链子。
- **H7 集群线性度**:5 台 dev 各跑 720p ulysses,集群吞吐是否线性叠加(NFS 冷读、错峰启动是否互相拖累)。
- **H8 画质证据**:720p ulysses 直出 vs 480p 单卡 + SeedVR2 2× 超分 —— 决策已定走前者,但留对照证据备查。

---

## 2. 度量口径与采集方法(统一,照 S2V 报告)

| 指标 | 采集 |
|---|---|
| 加载 | server 首启日志 `Load models cost`(冷) |
| **热态每步 / 整条** | server 日志 `Run DiT cost` / `🚀 infer_main cost` / `RUN pipeline cost`,**丢第1条,取第2..N 条中位数**(N≥10) |
| 显存峰值/rank | 后台 `nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 >> gpu.csv`,取 max(照 `scripts/smoke/vace_stress.sh` 监控段) |
| 宿主内存 | `free -g` 每 2s 采样,取 floor/peak |
| 通信占比(H3) | 对比 seq_p=4/2/1 的每步 DiT 时间;加速比 = t(单卡)/t(4卡),通信代价 = 4 - 加速比(理想 4×) |
| 吞吐 | 连跑 K 条计总墙钟 → 条/分/节点;× 节点数 → 条/分/集群 |
| 产物规格 | `ffprobe` 出片:分辨率/帧数/fps/时长/码率 |
| 插帧质量 | 人眼 + 相邻帧运动连贯性肉眼判(有无鬼影/果冻);同 prompt+seed 对照 |

- **复现脚本**:`scripts/smoke/test_bernini_720p_stress.sh` —— 照 `test_infinitetalk_stress.sh` 规范改写的**单容器复用**压测(起 server + 后台 CSV 监控 + 连发 N 丢首条 + 聚合汇总 + ffprobe 核分辨率)。一个脚本靠 env 覆盖 E0/E1/E2/E4/E6。scp 脚本 + 两个 config 到 `/nfs-models/_transfer/`。
- **真实 server API**(以 `test_infinitetalk_stress.sh` 为准,非猜测):`POST /v1/tasks/video/`(→task_id)、`GET /v1/tasks/$TID/status`(completed/failed)、`GET /health`(200)。
- **关键差异(Bernini vs InfiniteTalk 模板)**:① `model_cls=wan2.2_moe_distill task=t2v`(无 image/audio);② **禁 `load_from_rank0`**(Bernini 是 Wan2.2 MoE 双专家,开了破坏专家路由;InfiniteTalk 是 Wan2.1 单 DiT 才能开);③ 请求体走 t2v(prompt + target_video_length [+ target_fps])。
- **配置**:`configs/wan22_bernini/bernini_r_14b_t2v_720p_ulysses4_int8.json`(基线)、`..._rife_int8.json`(插帧)。

---

## 3. 实验矩阵

### E0 — 热态基线(钉死生产耗时)【最高优先】
- **目的**:H1。拿到 720p ulysses 稳态 end2end + 每步 breakdown。
- **配置**:基线 ulysses4 int8。**单容器复用**(server 常驻)。
- **命令**:
  ```
  tmux new -s e0 -d 'N=11 bash /nfs-models/_transfer/test_bernini_720p_stress.sh > /nfs-output/e0.log 2>&1'
  tail -f /nfs-output/e0.log      # 看加载→连发→汇总(中位/吞吐/峰值 都自动打)
  ```
- **度量**:脚本汇总的稳态中位/均值 end2end(丢首条冷)+ 显存/内存峰值;细粒度 step1-4 热态从容器日志 `docker logs bernini720-stress-* | grep 'Run DiT cost'`。
- **判据**:热态 end2end 稳定(方差<10%);step1 掉到与 step2 同量级(证实 24.8s 确为冷编译)。这个数 = 对外生产 SLA。

### E1 — RIFE 插帧(16fps → 32/48/60)
- **目的**:H2。插帧画质增益 vs 加时 vs 显存;找 720p 下 RIFE 的 OOM 上限。
- **前置**:先下 RIFE 权重 → `python tools/download_rife.py`,把 flownet.pkl 放到 config 里 `video_frame_interpolation.model_path`(`..._rife_int8.json` 已占位 `/nfs-models/.../rife/flownet.pkl`,按实际改)。
- **配置**:`..._rife_int8.json`,`target_fps` 分别 **32 / 48 / 60**。
- **命令**:`CFG_BASE=/nfs-models/_transfer/bernini_r_14b_t2v_720p_ulysses4_rife_int8.json TARGET_FPS=32 bash test_bernini_720p_stress.sh`(32/48/60 各跑一次;脚本会把 target_fps 注入 config + 请求体)。
- **度量**:① 插帧新增耗时(整条 - E0 同 prompt);② VAE decode 后帧序列显存峰值(720p×160帧 for 60fps 5s);③ 出片 fps/帧数核对;④ 画质:60fps 有无果冻/鬼影,慢镜头(swan 倒影、火焰)最容易露馅。
- **判据**:32fps 应几乎零代价且更顺滑 → 生产默认;48/60 若显存爆或鬼影明显则标注上限。**注意 fps 只是"补帧变顺",不改变生成时长/内容**(记忆 `sr-canvas-only-decision`:fps 透传不插帧的老结论此处被 RIFE 场景覆盖,单独记)。

### E2 — 帧数/时长边界(81 / 121 / 161 / 201)
- **目的**:H5。摸 720p ulysses 的帧数天花板;每帧摊销随长度怎么变。
- **预期(I2V 已铺路,832×1104 同面积)**:81=35.3G、121=37.9G✅、161=40.3G 顶满、201 OOM。Bernini 纯 t2v 更轻,天花板可能≥161。
- **命令**:`FRAMES=121 bash test_bernini_720p_stress.sh`(依次 121/161/201,脚本注入 target_video_length)。
- **度量**:显存峰值/rank、每步 DiT、是否 OOM;OOM 时 server 是否优雅 failed 不崩(I2V 实测 health 保持 200 → 单容器可继续下一档)。
- **判据**:确认稳过档(≤121 建议生产)、顶满档、死亡档;长视频超天花板走 480p 或分段。

### E3 — VAE tiling 开关
- **目的**:H5。720p VAE decode 显存 vs 耗时权衡(基线未开 tiling)。
- **配置**:基线 + `"use_tiling_vae": true`。
- **度量**:VAE decode 显存峰值 & 耗时,开/关对比。
- **判据**:若不开也不爆(E0 已知 2.9s 且整体没 OOM)→ 生产不开(更快);若 E2 的 121帧靠 tiling 才不爆 → 长视频专开。

### E4 — 并行度阶梯(量 Bernini 720p 的 ulysses 加速比)
- **目的**:H3。同架构已证 ulysses 有效(I2V 2.7× / VACE 3.37×),本组只补 **Bernini 720p 具体加速比**、确认长序列下通信是否恶化。
- **配置**:改 `parallel.seq_p_size`(**4 / 2**)与 `--nproc_per_node`。seq_p=1 单卡:int8 720p 单卡(I2V 实测)OOM,Bernini 大概率也 OOM → 单卡对照改用 480p,或直接以 VACE/I2V 加速比为锚,只测 720p 的 t4/t2。
- **度量**:每步 DiT 热态 t4 / t2;t2/t4 看接近 2× 的程度(反推 720p all-to-all 占比)。
- **判据**:预期 720p 加速比 ≥2.5×;若明显低于 480p 加速比,说明 720p 长序列通信量上升吃收益,记录为长视频/更高分辨率的注意点。**加速比 = 决策量化背书。**

### E5 — 节点吞吐对照(720p-ulysses 串行 vs 480p 单卡×4 并发)
- **目的**:H4。同一台 4 卡节点,两种形态的**节点吞吐(条/分)**与**单条延迟**。
- **A(720p ulysses)**:server 常驻,连打 K=20 条,总时间 → 条/分。
- **B(480p 单卡×4)**:`bench_bernini.sh` 4 卡各并发一实例(480p 基线 config),连打各 5 条 → 条/分。
- **度量**:A/B 各:单条延迟、节点条/分、单条产物分辨率。
- **判据**:卡充足决策已选 A(要 720p 画质);此组量化"选 A 相对 B 的吞吐代价 X%",写进容量规划(每节点每分钟出几条 720p)。

### E6 — 长稳压测(server 常驻 1h+)
- **目的**:H6。连续出片下的显存泄漏 / 耗时漂移 / NCCL 稳定性(无 NVLink 长跑最怕 all-to-all 偶发挂)。
- **命令**:`N=80 bash test_bernini_720p_stress.sh`(≈1h+;脚本自带 CSV 全程采样)。
- **度量**:monitor.csv 里 80 条 end2end 时序(有无单调上升=泄漏/漂移);GPU/host 内存 floor 是否随时间抬升;`docker logs` 有无 NCCL timeout/watchdog。
- **判据**:耗时平稳(漂移<10%)、内存不涨、无 NCCL 挂 → 可常驻生产;否则记录需定期重启周期。

### E7 — 集群扇出(5 台 × 720p ulysses)
- **目的**:H7。5 台 dev(0021-0025)各起一个 720p ulysses server,同时压,集群吞吐是否线性(NFS 冷读争抢、错峰启动)。
- **命令**:5 台按 memory `dev-fleet-0021-0025` 铁律**错峰 120s** 起 server;各自 probe 20 条。
- **度量**:单台条/分 × 5 vs 实测集群条/分(线性度);NFS 冷加载阶段是否互相拖慢。
- **判据**:线性度 >0.85 → 直接线性扩容;否则记录 NFS/启动瓶颈(错峰、预热策略)。

### E8 — 画质对照(留证据,非决策)
- **目的**:H8。720p ulysses 直出 vs 480p 单卡 + SeedVR2 2× 超分(→1080p 近似)。
- **命令**:同 prompt+seed;A=720p ulysses;B=480p 基线出片 → SeedVR2 sr_ratio 2.0(见 memory `seedvr2-docker-sr-ratio-locked`)。
- **度量**:并排肉眼(细节、纹理、伪影、时序一致性);两条链路总耗时+占卡。
- **判据**:纯留档。若某些题材 B 明显更好可作为"SR 兜底"备选,但主路仍 A。

---

## 4. 风险与预案

| 风险 | 触发 | 预案 |
|---|---|---|
| triton 每容器冷编译(无持久 cache) | server 崩溃/重启,首条又 ~25s/step | SLA 写明"首条冷";探 `TRITON_CACHE_DIR` 挂 NFS 是否能跨进程复用(仓库当前无,需试);server 尽量长驻 |
| NCCL 无 NVLink all-to-all 挂/慢 | E4 加速比低 / E6 长跑偶发 timeout | 去注释 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 走 host 中转;E4 若加速比<1.5× 上报重议 |
| RIFE 720p 高帧数 OOM | E1 的 48/60fps | 降 target_fps;RIFE 侧分块/降批;标注 720p RIFE 上限 fps |
| **编辑线 720p 撞 VACE OOM 墙** | 编辑线(v2v/i2i)concat source latents 后跑 720p 4卡 | VACE 720p 4卡已三审死刑(额外上下文分支挤爆 40G)。编辑线 720p **不假设能复用本方案的纯 t2v ulysses**;先按 480p 编辑+SR 或单卡+offload 设计,720p 编辑另立实验验证 |
| 256G host + swap 仅 3G | 多 server / offload 叠加 | 720p ulysses 不开 cpu_offload(E0 已验不爆);多节点错峰,避免同时冷加载挤 host |
| NFS 冷读 352s 拖慢集群 | E7 同时启动 | 错峰 120s;或预热(先单条 warm 再放流量) |

---

## 5. 结果模板(待填)

**E0 热态基线**

| 项 | 冷(第1条) | 热(第2..11中位) |
|---|---|---|
| step1 / step2 / step3 / step4 |  |  |
| VAE |  |  |
| end2end(rank0) | ~61 |  |

**E1 插帧**

| target_fps | 出片帧数 | 插帧加时 | VAE后显存峰 | 画质(鬼影/果冻) |
|---|---|---|---|---|
| 32 |  |  |  |  |
| 48 |  |  |  |  |
| 60 |  |  |  |  |

**E4 并行度**

| seq_p | 每步DiT(热) | 加速比 vs 1 | 通信代价 |
|---|---|---|---|
| 1 |  | 1× | — |
| 2 |  |  |  |
| 4 |  |  |  |

**E5 节点吞吐**

| 形态 | 单条延迟 | 节点条/分 | 分辨率 |
|---|---|---|---|
| 720p ulysses(串行) |  |  | 720p |
| 480p 单卡×4(并发) |  |  | 480p |

(E2/E3/E6/E7/E8 同样留空表)

---

## 6. 执行顺序与待办

- [ ] **E0 先跑**(半天出生产 SLA)——scp `serve_720p_ulysses4.sh` + `warm_latency_probe.sh` + 两个 720p config 到 `/nfs-models/_transfer/`。**当场核对热态到底 ~45s 还是 ~87s(I2V 参考),差一倍**;出片 ffprobe 核实 720×1280(I2V §8 坑)
- [ ] 首次跑前 `curl` 确认 server 端点/schema(`lightx2v/server/schema.py` 与路由前缀),修 `warm_latency_probe.sh` 的 `EP`
- [ ] 压测统一走 `scripts/smoke/test_bernini_720p_stress.sh`(照 `test_infinitetalk_stress.sh` 规范):E0=`N=11`、E1=`TARGET_FPS=32`、E2=`FRAMES=121`、E4=`GPUS=0,1`、E6=`N=80`;首跑前 `curl` 确认 `/v1/tasks/video/` 端点与 schema 字段名与 `test_infinitetalk_stress.sh` 一致
- [ ] `python tools/download_rife.py` 下 RIFE 权重 → 填 `..._rife_int8.json` 的 model_path → E1
- [ ] E4 量加速比(决策背书)、E5 量节点吞吐(容量规划)
- [ ] E2/E3 边界、E6 长稳、E7 集群、E8 画质证据
- [ ] 结果回填本文 §5,复核 §0 冷基线
- [ ] 复现脚本/配置随编辑线一起 review(**未 commit**)
