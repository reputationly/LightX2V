# Wan2.2-VACE 实验测试报告

> 日期:2026-07-11 · 模型:Wan2.2-VACE-Fun-A14B(MoE 双专家, 40步无蒸馏, CFG开)
> 环境:4×A100 PCIE 40G / ARM 鲲鹏920 / 256G RAM(见交接文档) · 镜像 lightx2v:arm64-a100-latest
> **结论先行:①bf16 全形态 OOM, int8-triton 是唯一活路;②Lightning 4步蒸馏LoRA 移植成功(1088s→82.5s, 13×), 需一行补丁;③LoRA 与 40步是两种人格——LoRA=强参考锚定+轻微动态(编辑器), 40步=弱锚定+强动态(创作者), 按场景二选一;④生产推荐:编辑类能力全走 4步LoRA 480p(82s)+SeedVR2 超分。**

## 一、通关史:四次 OOM 到 int8 通车

| 尝试 | 配置 | 结果 |
|---|---|---|
| 1. bf16 + T5 offload | 480p 81帧 | ❌ OOM(38.5G+) |
| 2. + rope_chunk + VAE offload | 同上 | ❌ OOM(38.9G) |
| 3. + 降到 49帧 | 同上 | ❌ OOM |
| 4. **int8-triton 离线转换**(高/低噪各~18G) | 480p 81帧, model 粒度 offload | ✅ **通车** |

- bf16 单专家 28G 常驻 + VACE 额外上下文分支(参考图/控制视频 latent 拼接)把 40G 挤爆——比标准 t2v 更吃显存。
- int8 转换:`convert` 工具离线量化高/低噪两个 DiT → `/nfs-data/models-int8/Wan2.2-VACE-Fun-int8/{high,low}_noise`(block 目录格式, triton loader 直接吃)。

## 二、实测数据(480p 832×480, 81帧, 40步, seed 42)

| 形态 | 耗时 | 备注 |
|---|---|---|
| R2V 单卡 int8(model offload) | **1238s(20.6min)**, DiT 1080s(27s/步) | 首片画质"还不错"(用户判决), 专家切换 1.9s |
| **R2V 4卡 ulysses int8**(禁offload, 双专家常驻~36G/卡, 无OOM) | **总625s(10.4min)**, DiT 317.6s | **DiT 提速 3.37×, 全场最佳多卡收益**(DiT占比87%, Amdahl 友好);加载 303s(4 rank 争NFS带宽);画质=单卡肉眼无差(2026-07-11 用户判决)→多卡无损 |
| V2V 原片直喂(单卡, src_video=R2V首片, 水墨风提示词) | 总1101s, pipeline 1015.6s | 链路通(PyAV补丁生效);**但输出≈输入, 提示词拗不动**——自然RGB视频把像素钉死, 真重绘必须喂预处理控制视频(见 canny 实验) |
| V2V canny 控制(canny边缘视频→水墨重绘) | 完成(2.8M) | ✅ **"确实变成了水墨风格, 内容跟原始的差不多"(用户判决)**——语义重绘成立:控制信号给结构、提示词给内容 |
| R2V + RIFE 16→32fps | pipeline 1121s(插帧仅+~34s) | VACE→VFI 链路通, 成本可忽略;**"丝滑很多"(用户判决)→原子能力形态挂RIFE定案** |
| MV2V inpaint 一号(mask区不填充) | 完成 | ⚠️ 输出≈原片——runner 把 mask 区原像素也编码送模型(reactive latent), 模型照抄重建 |
| MV2V inpaint 二号(mask区**灰填**, 官方预处理做法) | 总1336s | ✅ **机制成立**:mask区(上半身)确实按提示词重画, mask外(裙子)如实保留;480p 下语义执行含糊(黄衣≈肤色)。**画布要点:mask 圈准完整物体+提示词具体+src 必须灰填** |
| MV2V outpaint(60%画面四周扩展) | 总1214s | **"挺自然"(用户判决)✅**——画布扩图能力可用 |
| 首帧续写 extend(仅首帧+运镜提示词) | 总1197s | ❌ **"不连贯+手变形"(用户判决)**——不推荐;首帧动画走生产 i2v |
| 纯 t2v(全空输入) | 完成(8.0M) | 路径通;480p 糊(config锁定), 质量不评——生产有专职 t2v, 此模式无意义 |

### 2.1 Lightning 4步蒸馏 LoRA 移植(压轴, 全部成功)

- **弹药**:lightx2v/Wan2.2-Lightning `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0`(high/low 各1.2G, ModelScope 已下 NFS)→ `convert_lightning_to_x2v_lora.py` 转 x2v 格式(400组全转0跳过)→ `lora_dynamic_apply: true` 在线应用于**现有 int8 权重**(量化模型唯一合法路径, 权重零改动)
- **必需补丁**:`WanVaceModel.__init__` 未透传 lora_path/lora_strength(上游动态LoRA只实现到标准Wan类)——一行签名修复, 出包第9号补丁
- **速度**:480p 单卡 pipeline **1088s→82.5s(13.2×)**;每步 27s→14s(免CFG减半)×步数 40→4(10×)
- **A/B 终审**(4对题材:写实女子/夜景仙女棒/猫冲浪/烹饪手部, 同seed同prompt同参考图):
  - **LoRA 人格=编辑器**:强参考图锚定(构图/姿势忠实, 卡通参考图都原样保留), 只做轻微动态调整, 画面稳
  - **40步 人格=创作者**:偏离照片更远、新增动作更多, 但易失控(写实女子对照臂镜头狂晃)
  - canny 控制在 4步下依然成立(水墨重绘 OK, 风格化程度略弱于40步的纯黑白水墨)
- **720p 两条路**:4卡常驻❌**三审死刑**(P10两轮补丁:mask数学挪CPU→像素张量驻CPU+use_tiling_vae 分块编解码, 墙从VAE拆到DiT pre_infer patchify 仍差774MB——38G常驻+720p序列瞬时峰在40G卡无解, P10已回滚);**单卡+model offload ✅**(offload形态VAE编码时GPU全空)——pipeline 258s/端到端344s
- **压测定形**(0013-0015 三档):单实例基线 418s;**双实例稳态 485s(膨胀16%)✅=生产密度**;4实例❌(host内存最低61MB濒死+4×冷启把NFS打进病态:3×T5并发读18min);错峰启动升格为必须

## 三、模式覆盖(六种玩法 = src_video/src_mask/src_ref_images 三原语组合)

| 模式 | 输入 | 判决 |
|---|---|---|
| R2V 参考图 | ref_images | ✅ 人物特征保持可用, 多卡/RIFE 均无损 |
| V2V 重绘 | src_video(**必须预处理控制视频**, 原片直喂=复读) | ✅ canny 线稿→水墨重绘成功(结构保持+风格重画);depth/pose 等 annotator LightX2V 未带, 画布需上游预处理节点 |
| MV2V inpaint | src_video+mask(白=重画, **src的mask区必须灰填**) | ✅ 机制成立(不灰填=照抄);编辑质量中等 |
| MV2V outpaint | 缩小画面+周边mask白 | ✅ "挺自然" |
| 首帧续写 | 首帧+灰帧+时序mask | ❌ 不连贯/手变形, 用生产 i2v 替代 |
| 纯 T2V | 全空 | 【待填】(有专职 t2v 模型, 仅验路径) |

**静态视频转运镜的结论**:VACE 干不了(V2V 结构钉死镜头, extend 质量不行)——正解=取帧走生产 i2v+运镜提示词;重度需求再评估 Wan2.2-Fun-Control-Camera 专用权重。

## 四、ARM 镜像补丁(出包清单新增)

- **vace_processor.py decord→PyAV 兜底**:ARM 镜像 decord 是空壳包(无 `bridge` 属性), V2V/MV2V 模式必崩。补丁在 `load_video_batch` 加 try/except, 落到 `_PyAVVideoReader`(接口对齐 decord 所需子集:avg_fps/frame_timestamp/get_batch)。x86 不受影响;PyAV 解码与 decord 像素级一致, 仅预处理毫秒级差异, 画质/速度零影响。

## 五、生产形态拍板

| 场景 | 形态 | 耗时(480p 81帧) |
|---|---|---|
| **画布编辑类**(R2V/inpaint/outpaint/canny重绘)= 默认 | **4步LoRA + int8-triton 单卡 + SeedVR2 3B 超分(1664×960)** | **生成82s + SR 80s ≈ 2.7min**;终验判决"轻微锐化、可接受"✅ |
| ~~创作类 40步 4卡~~ | **不上线**(用户拍板:625s 太长撑不起产品体验;大动态需求走生产 t2v/i2v;ul4 配置留档) | ~625s |
| 720p 原生 = 不推荐 | 单卡offload 可跑(344s)但清晰度差(1280×720已ffprobe核实, 是真画质弱非配置问题);4卡常驻❌VAE前处理OOM | 比配方链更慢、更糊、分辨率更低, 三输 |

## 六、生产接入清单

1. **server schema 缺字段**:请求体没有 `src_video`/`src_ref_images`/`src_mask`——VACE 只能 CLI 跑, 上服务要加 schema 字段+落盘逻辑(同 SeedVR2 sr_ratio 的改造路数)
2. **插帧**:原子能力形态挂 RIFE(配置见 `wan22_moe_vace_a100_int8_rife.json`);画布编排时只在链路最后一步插(既定拍板)
3. **多卡形态已验证**:4卡 ulysses int8 双专家常驻 ~36G/卡无 OOM, 端到端 20.6min→10.4min(生成段 3.37×)。**生产建议 4卡**;加载 303s 建议常驻实例而非按请求拉起
4. **铁律沿用**:多卡禁 cpu_offload;MoE 禁 load_from_rank0(w1b 实证);triton 首请求 autotune 需预热
5. **40步无蒸馏是速度天花板**:社区暂无 VACE-Fun 蒸馏权重, 20min/条(单卡)决定了它只适合低频编辑场景, 不适合高并发生成

## 六、配置/脚本索引

- 配置:`configs/wan22/a100/wan22_moe_vace_a100_int8{,_ul4,_rife}.json`(bf16 版 `wan22_moe_vace_a100.json` 仅存档, 40G 不可用)
- 脚本:`scripts/smoke/vace_int8.sh`(R2V 首跑) / `vace_batch.sh`(ul4/v2v) / `vace_v2v2.sh`(v2v+PyAV补丁) / `vace_rife.sh`(插帧验证) / `download_vace_model.sh`
- 产物:outputs/ 下 `vace_r2v_int8.mp4` / `vace_r2v_ul4.mp4` / `vace_v2v_int8.mp4` / `vace_r2v_rife.mp4`
