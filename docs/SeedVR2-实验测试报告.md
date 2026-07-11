# SeedVR2 视频超分 实验测试报告

> 日期:2026-07-09 · 环境:gpustack 集群计算节点(dev-gpustack-a100-0010)
> GPU:A100 PCIE 40G 单卡 · 镜像:`lightx2v:arm64-a100-latest` · harness:`scripts/smoke/test_model.sh`(server 模式)
> 目标:AIGC 720p 视频 → 1080p 超分,选型 3B / 7B普通 / 7B锐化

## 结论先行

1. **选 3B**:画质与 7B 肉眼无差(AIGC 720p 输入),显存峰值 27.5G vs 39.9G(7B 距 40G 卡上限仅 1G,换素材极易 OOM),加载快 3 倍。
2. **7B 时间成本≈0 但不值**:单步扩散下 DiT 耗时占比小(生成 154s vs 148s,仅 +4%),瓶颈在 VAE 编解码 + 分段拼接(CPU),但显存风险和加载时长都劣于 3B。
3. 与官方结论一致:**论文用户研究里 3B(蒸馏自 7B)偏好度反超 7B**;官方明确警告轻退化输入(720p AIGC)易过度生成细节/过锐化,两个模型都有此风险。
4. server 模式 `sr_ratio` 锁死 2.0:720p→1080p 够用(被 target 1080p cap);**480p→1080p 实测只能到 1664×960**(832×480 源 ×2.0,到不了 1080p 需 ~2.25×)→ 要支持必须给 `lightx2v/server/schema.py` 的 `VideoTaskRequest` 加 `sr_ratio` 字段(1 行,未做待需求)。
5. **⚠️ 默认配置对 >81 帧视频有硬伤级接缝跳变**:`sr_segment_length:81` 把 121 帧切两段独立扩散,拼接是丢重叠帧后 ffmpeg concat 硬切(`seedvr_runner.py:628`),第 80/81 帧处(24fps ≈3.3s)画面跳变,三个模型版本都复现、原片无。**修复已验证(2026-07-09 肉眼确认跳变消失):≤121 帧用单段配置**(`seedvr2_3b_seg121.json`:`sr_segment_length:121` + `vae_tile_size:384`)——3B 单段 121 帧实测:加载 50s / 生成 278s / **显存峰值 37.2G**(<40G 可行,但比分段版 148s 慢 88%,是无缝的代价);更长视频(>121帧)要根治需改上游拼接逻辑(重叠帧 cross-fade 融合而非硬切)。**生产建议:≤5s 视频一律用单段配置。**

## 测试数据(同一输入:hq_coffee_rain_720p121.mp4,1280×720 / 121帧 / 24fps)

| 版本 | 加载 | 生成(121帧) | 显存峰值 | GPU利用峰值 | 宿主内存峰值 | 产物大小 | 输出 |
|---|---|---|---|---|---|---|---|
| 3B | 30s | 148s | 27.5G | 100% | 42.8G | 15.7MB | 1920×1088 / 121帧 / 24fps |
| 7B 普通 | 90s | 154s | **39.9G ⚠️** | 100% | 74.9G | 15.5MB | 同上 |
| 7B 锐化 | 90s | 154s | **39.9G ⚠️** | 100% | 74.9G | 17.2MB(+11%,高频细节更多) | 同上 |

- 折算速度(3B):148s / 121帧 ≈ 1.22 s/帧;5 秒视频(121帧@24fps)超分约 2.5 分钟。
- 121 帧按 `sr_segment_length:81` + `sr_overlap:1` 分两段(81+41);段间 CPU 飙到 15 核(拼接/换段),显存尖峰出现在 VAE 阶段(DiT 阶段仅 ~10G)。
- 输出 1088 而非 1080:`DivisibleCrop(16)` 对齐;fps 自动跟随源视频。

## 多素材扫描(3 版 × 3 素材,九连跑全部成功)

| 素材 | 源规格 | 输出 | 备注 |
|---|---|---|---|
| graded_horse_beach(马/沙滩,毛发纹理) | 1280×720/121帧 | 1080p 级 | 三版肉眼对比待补 |
| cmp_lx2v_morning_720p121 | 1280×720/121帧 | 1080p 级 | 同上 |
| wan22_lightning_fp8_480P_5s_8step | 832×480 | **1664×960(到不了1080p)** | sr_ratio=2.0 封顶实锤 |

- 480p 素材 7b_sharp 单条:加载 100s、生成 56s、显存峰值 28.0G(分辨率低,峰值也低于 720p 的 39.9G)。

## 画质肉眼对比(待补)

- [x] coffee_rain(AIGC 720p):3B / 7B / 7B锐化 三版肉眼无差
- [x] 马/沙滩、morning 两条 720p:**三版同样肉眼无差,锐化版未见明显过冲** → **最终结论:选 3B**(画质相同、显存 27.5G vs 39.9G、加载快 3 倍)
- 附:产物文件大小与版本无稳定关系(coffee 锐化版最大、马/morning 3B 最大),码率随内容浮动,不构成画质信号
- [x] **时序跳变**:分段拼接导致 3.3s 处跳变(见结论 5);**单段配置修复已肉眼验证通过(2026-07-09)**
- [x] ~~480p 组产物只有 2s~~ 已澄清:**源文件本身就是 49帧/2.04s**(文件名"5s"名不副实的历史产物);超分产物 49帧/24fps/2.04s 与源逐帧一致,**SR 链路对帧数/帧率忠实无损**

## 跑法(复现)

```bash
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest \
NAME=seedvr2-3b MODEL_CLS=seedvr2 TASK=sr NP=1 GPUS=1 STEPS=1 \
EXTRA_VOL="-v /data/smoke/seedvr_runner.py:/opt/LightX2V/lightx2v/models/runners/seedvr/seedvr_runner.py:ro" \
MODEL_PATH=/nfs-data/models/ByteDance-Seed/SeedVR2-3B \
CFG=/data/smoke/seedvr2_3b.json \
VIDEO_PATH=/nfs-data/outputs/hq_coffee_rain_720p121.mp4 \
PROMPT="video super resolution" \
OUT=/data/outputs/seedvr2_3b_coffee_1080p.mp4 \
bash /data/smoke/test_model.sh
```

- 7B:`CFG=/data/smoke/seedvr2_7b.json`(锐化版 `seedvr2_7b_sharp.json`),`MODEL_PATH` **仍指 3B 目录**(7B 仓库缺 vae/emb,借 3B 的;config `dit_original_ckpt` 指 7B DiT,普通/锐化只差这一行)。
- 配置:`configs/seedvr/a100/seedvr2_{3b,7b,7b_sharp}.json`(cpu_offload + tiling_vae + 分段)。
- `STEPS=1` 必须显式给(harness 默认发 infer_steps:4,会覆盖单步扩散配置)。
- `VIDEO_PATH` 必须是**容器内可达路径**(server 不下载视频;NFS 挂卷 `/nfs-data/...`)。

## 踩坑记录

1. **torchvision 新版移除 `read_video_timestamps`** → `seedvr_runner.py::_probe_video` 直接 ImportError(try 只包了调用没包 import)。已修:import 失败回退 PyAV 读时间戳/fps(与读帧的三级 fallback 同思路),补丁文件挂卷覆盖容器内源码(`EXTRA_VOL`),免重建镜像。**值得给上游提 PR**。
2. **节点 8000 端口被占**(host 网络容器)→ harness 加 `PORT` 参数(映射改 `-p $PORT:8000`)。
3. **镜像 tag 变更**:节点上是 `arm64-a100-latest`,harness 默认的旧日期 tag 不存在 → 跑时 `LX_IMG` 覆盖。
4. 管理节点(238)无 `/nfs-data`/`/data` 软链(仅计算节点有,指向 `/nfs-models/wuhanjisuan894`),浏览 NFS 用真实路径。

## 官方/社区参考结论

- 论文([arXiv 2506.05301](https://arxiv.org/html/2506.05301v1)):3B 蒸馏自 7B 初始模型;用户研究(VideoLQ+AIGC28 共 50 条)**3B 偏好度高于 7B**,作者归因于蒸馏阶段有效。
- [官方模型卡](https://huggingface.co/ByteDance-Seed/SeedVR2-7B):原型模型;对重退化/大运动不鲁棒;**轻退化输入(720p AIGC)倾向过度生成细节、偶发过锐化**。
- 社区经验:退化重/写实素材用 3B(7B 会放大瑕疵);干净素材 7B 才显微细节优势;锐化版适合动漫/风格化,皮肤易塑料感、会放大源里的 ringing([HF 讨论](https://huggingface.co/ByteDance-Seed/SeedVR2-7B/discussions/2))。

## 待办

- [ ] 画质肉眼对比结论回填(上表)
- [ ] 需要 480p→1080p 时:schema 加 `sr_ratio` 字段(1 行)+ 重跑验证
- [ ] 可选:7B 显存峰值贴上限,若必须用 7B 试调小 `vae_tile_size`(默认 512)压峰值
