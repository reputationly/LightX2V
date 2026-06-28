# 视频超分/补细节 选型备查(720p→1080p)

> 记录时间:2026-06-28
> 结论:**专心搞 SeedVR2(LightX2V 原生内置)。FlashVSR 暂时不弄。**

## 一、需求背景

把 AI 生成的干净 720p 视频放大/锐化到 1080p,并希望**补充画面细节**。

## 二、能不能"补细节"——看模型类型

| 类型 | 代表 | 是否真"补细节" |
|---|---|---|
| GAN/回归式 | Real-ESRGAN、FlashVSR | 半补:增强已有边缘、重建合理纹理,锐化好,但凭空新细节有限 |
| **扩散式** | **SeedVR2**(字节 2025) | ✅ 真补:生成模型,能脑补原本不存在的细节(发丝/毛孔/纹理)。代价:慢、可能改动原内容、不保证完全忠实 |

一句话:想"补细节"就得用扩散式 SeedVR2;FlashVSR/ESRGAN 是"变清晰",不是"无中生有"。

## 三、LightX2V 支持情况(已查仓库)

### SeedVR2 —— 原生完整支持 ⭐(后续主攻)

- 网络/runner/scheduler 全套:
  - `lightx2v/models/networks/seedvr/`
  - `lightx2v/models/runners/seedvr/`
  - `lightx2v/models/schedulers/seedvr/`
  - VAE:`lightx2v/models/video_encoders/hf/seedvr/`
- 现成配置:
  - `configs/seedvr/seedvr2_3b.json`
  - `configs/seedvr/seedvr2_7b.json`
  - `configs/seedvr/4090/seedvr2_3b.json`、`configs/seedvr/4090/seedvr2_7b.json`
- 不用自己接,**下权重 + 用现成 config 就能跑**。
- 显存:40GB A100 跑 **7B** 应该 OK,紧张就用 **3B**。

### FlashVSR —— 暂缓(走外部依赖,ARM 有坑)

- `lightx2v/models/runners/vsr/vsr_wrapper.py`,`from diffsynth import FlashVSRTinyPipeline`
- 即通过第三方库 **diffsynth** 接入,而 diffsynth **没装在 ARM 镜像里**(和 decord 同类坑,ARM 安装待确认)。
- **本轮不弄。** 以后真要轻量/实时超分再回来搞。

### Real-ESRGAN

- 不在 LightX2V 内,是外部独立工具。本轮不考虑。

## 四、下一步(SeedVR2 主攻待办)

- [ ] 看 `configs/seedvr/seedvr2_7b.json` + seedvr runner,理清:怎么跑、要下哪个权重、输入输出格式
- [ ] 下 SeedVR2 权重(7B 优先,显存紧用 3B)
- [ ] 跑通 720p→1080p 一条 demo,确认补细节效果与耗时
- [ ] 决定是否纳入正式流水线(或对比:生成阶段直接出 1080p 是否更省)

## 五、一个值得先想清楚的点

720p→1080p 只有 1.5×。如果 LTX/Wan 流水线能**直接出 1080p**,可能比"先 720 再超分"更省事、画质更原生。SeedVR2 的价值更多在**补细节/修复**,而非单纯放大。
