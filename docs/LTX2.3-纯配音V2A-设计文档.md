# LTX-2.3 纯配音(V2A)设计文档

> 目标:给一段**已有视频**只生成配套音轨,画面像素**逐帧完全不变**(严格 video→audio dubbing / foley)。
> 状态:设计阶段(未实现)。本文用于评审与实现对齐。

---

## 1. 背景与动机

### 1.1 需求
输入一段视频 + 一段文字描述(想要的声音:说话 / 脚步 / 环境音 / 音效),输出**画面不变、只多一条 AI 音轨**的视频。核心硬约束是**画面绝对不变**——不是"尽量接近",而是逐像素相同。

### 1.2 现状:LTX-2.3 已有能力与缺口
LTX-2.3 在 LightX2V 中是**音视频联合扩散**模型,视频/音频各有独立 VAE(`video_encoders/hf/ltx2/{video_vae,audio_vae}`)。已接入 4 个任务:

| 任务 | 输入 | 输出 | 画面是否重渲染 |
|------|------|------|----------------|
| `t2av` | 文本 | 视频+音频 | 是(全新生成) |
| `i2av` | 图+文本 | 视频+音频 | 是 |
| `v2av` | 控制视频+文本 | 视频+音频 | **是**(IC-LoRA union-control,画面被重渲染) |
| `ltx2_s2v` | 音频(+参考图) | 视频 | 是(音频当条件驱动视频) |

**缺口**:没有"画面不动、只出音频"的纯 V2A。`v2av` 最接近但会重渲染画面(把输入视频降采样 0.5 当 IC-LoRA 参考),不满足硬约束。

### 1.3 关键洞察:基础设施已经对称
`ltx2_s2v` 做的正是**反向**操作:冻结输入**音频** latent、只去噪**视频**。纯 V2A 就是它的**镜像**:冻结输入**视频** latent、只去噪**音频**。二者共用同一套调度器 mask 机制,因此 V2A **无需改动核心算法**,只需新增一个 input-encoder 分支 + 输出混流。

---

## 2. 核心机制(逐行核实)

### 2.1 调度器的双模态 denoise mask
文件:`lightx2v/models/schedulers/ltx2/scheduler.py`

- **保留 clean 的融合**(`post_process_latent`,~L857):
  ```python
  return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)
  ```
  `mask=0` → 每步输出恒等于 `clean`(编码后的干净 latent)。

- **初始化不加噪**(`prepare_*_latents`,~L574):
  ```python
  scaled_mask = mask * noise_scale
  latent = (noise * scaled_mask + latent * (1 - scaled_mask))
  ```
  `mask=0` → `latent` 恒等于 `initial_*_latent`,不被噪声污染。

- **timestep 归零**(`video_timesteps_from_mask` / `audio_timesteps_from_mask`):
  `timesteps = mask * sigma` → `mask=0` 的 token 全程 timestep=0,模型视其为已完成去噪的干净条件。

- **每步更新**(`step_post`,~L877):视频、音频各自算 velocity 前先过 `post_process_latent`。冻结模态因 mask=0,velocity 计算后仍稳定停在 clean 值。

**结论**:把视频 mask 全置 0、`initial_video_latent`=编码后的输入视频,则视频 latent 全程冻结;音频 mask=1(默认)正常去噪,并通过联合注意力**以冻结的干净视频为条件**生成音频。这与 `i2av` 用干净关键帧条件化、`ltx2_s2v` 冻结音频的做法同源,属 in-distribution。

---

## 2.5 方案对比:为什么必须做 V2A 改造(v2av 重绘 vs V2A 冻结)

有人会问:现成的 `v2av` 已经能"吃视频、出带音频的视频",为什么还要新做 V2A?因为 **v2av 会把画面完全重画,不满足"画面绝对不变"**。

### 机制差异
`v2av`(`scheduler.py` `_append_reference_video_latents` ~L667):把输入视频当成**额外的参考 token** 附加进去(`mask = 1-strength`,strength=1.0 时参考被冻结成强条件),但**真正输出的那路视频主 latent,`denoise_mask` 仍为 1,是从纯噪声重新去噪出来的**。参考只"强烈引导",不"复制"——输出的每个像素都是模型重画的。

V2A 则相反:把**主视频 latent 直接 mask=0 冻结**,输出画面由 `-c:v copy` 复用原视频流,像素零损失。

### 变化程度对比

| 维度 | v2av(重绘) | V2A(冻结,本方案) |
|------|-------------|---------------------|
| 构图 / 布局 / 运动 | 贴得很近(强引导) | **完全一致** |
| 逐像素 | **完全不同**(全是模型重画) | **逐像素相同** |
| 人脸身份 / 皮肤纹理 | 漂移("像但不是同一人") | **不变** |
| 文字 / logo / 小物体 / 手指 | 易变形(细节最先崩) | **不变** |
| 清晰度 | 参考降采样 0.5 + VAE 往返 → 细节损失 | **原清晰度** |
| 色彩 / 光照 | 可能轻微偏移 | **不变** |
| 音频 | ✅ 生成 | ✅ 生成 |

### 判定
- "画面绝对不变"是**硬要求** → 现有任何任务(v2av / t2av / i2av / s2v)都做不到,**只能走 V2A 改造**(冻结主 latent + 原视频混流)。
- 若可接受"很像即可、允许重绘" → 用现成 v2av 零改造,但需接受上表漂移。

本项目取硬要求,故采用 V2A。

### 2.2 参照实现:`ltx2_s2v`(镜像模板)
`_run_input_encoder_local_ltx2_s2v`(`ltx2_runner.py` ~L493)已示范冻结一路模态:
```python
self.initial_audio_latent = encoded            # 编码输入音频
self.audio_denoise_mask   = torch.zeros(1, f_audio, mel_bins)   # 冻结音频
...                                            # 视频侧走正常去噪
```
V2A 把"音频"换成"视频"即可。

---

## 3. 设计方案

### 3.1 新任务:`v2a`
`model_cls` 仍为 `ltx2`。输入 `--video_path`(原视频)+ `--prompt`(声音描述),输出原画面 + 生成音轨的 mp4。

### 3.2 数据流
```
输入视频 ──(全分辨率解码)──► pixels
   │
   ├─► video_vae.encode ─► initial_video_latent ─┐
   │                                             │  video_denoise_mask = 0(冻结)
   │                                             ▼
prompt ─► text_encoder ─────────────────► 联合去噪(只去噪音频)
                                              │  audio_denoise_mask = 1(全去噪)
                                              ▼
                                        audio latent ─► audio_vae.decode ─► 生成音轨
                                              │
原始视频文件 ────────────────────────────────┴─► ffmpeg -c:v copy 混流 ─► 输出 mp4
                                                    (画面像素零损失)
```

### 3.3 为什么画面能"绝对不变"
保存时**不使用** VAE 解码出的视频帧,而是用 `ffmpeg -i 原视频 -i 生成音轨 -map 0:v:0 -map 1:a:0 -c:v copy -shortest out.mp4`,直接复制原视频流。像素零损失由**混流方案**保证,与模型/VAE 精度无关。(VAE 编码输入视频仅用于给音频提供条件,其解码结果被丢弃。)

---

## 4. 改动清单

### 4.1 `lightx2v/models/runners/ltx2/ltx2_runner.py`(核心)

**(a) 新方法 `_run_input_encoder_local_v2a`**
以 `_run_input_encoder_local_v2av`(~L384,负责读视频/定帧长/VAE 编码)为骨架,按 s2v 的冻结逻辑改写:
- `_clear_ltx2_reference_audio_state()` + `_clear_ltx2_reference_video_state()` 防状态泄漏。
- 读 `self.input_info.video_path`(必填,空则报错)。
- **全分辨率**加载(`load_video_conditioning`),**不做** v2av 的 `ref_downscale_factor=0.5`。
- 帧数**向上取整**到 `1+8k`(复制末帧补齐,而非裁尾——裁尾会让被复制的原片尾帧无配音、成静音尾;多生成的零点几秒音频由 mux 的 `-shortest` 裁掉),设 `self.input_info.target_video_length`,再 `get_latent_shape_with_target_hw()` 得 video/audio latent shape(保证音视频帧长对齐)。
- `self.video_vae.encode(pixels)` → `self.initial_video_latent`。
- **冻结视频**:`self.video_denoise_mask = torch.zeros(1, F, H, W)`(unpatchified,shape 同 `run_vae_encoder` 返回值)。
- **放开音频**:`self.initial_audio_latent = None`;`self.audio_denoise_mask = None`(scheduler 默认 ones=全去噪)。
- **禁用 IC-LoRA 参考**:`self._ref_video_latent = None`(不走 union-control 重渲染分支)。
- 记录原视频路径:`self._v2a_source_video = video_path`。
- `run_text_encoder(self.input_info)`,返回 `{"text_encoder_output": ...}`。

**(b) `init_modules`**(~L101)分派:
```python
elif self.config["task"] == "v2a":
    self.run_input_encoder = self._run_input_encoder_local_v2a
```

**(c) VAE encoder 加载门**(`load_vae`,~L206):把 `"v2a"` 加入需要 VAE encoder 的任务元组 `("i2av","ltx2_s2v","v2av")`。

**(d) 输出混流**(`process_images_after_vae_decoder`,~L936):
- `task=="v2a"` 时,用生成音轨 + **原始视频文件**混流,替代 `save_video` 的解码视频输出。
- 新增 helper（见 4.2）。

### 4.2 `lightx2v/utils/utils.py`
新增 `mux_generated_audio_onto_video(src_video, gen_audio, out_path, tempo=1.0)`:
- 参照已有 `mux_audio_from_video`(~L360)的 ffmpeg 调用范式(`imageio_ffmpeg.get_ffmpeg_exe()`)。
- 生成波形以 f32le PCM 走 stdin 喂 ffmpeg(免 wav 库依赖),`-map 0:v:0 -map 1:a:0 -c:v copy`。
- 音频滤镜链 `atempo(可选) → apad`:
  - **fps 对齐**:音视频都在模型 24fps 时间轴上生成(模型空间严格对齐),非 24fps 源由 `tempo=src_fps/24` 变速(保音高,超 0.5~2.0 自动级联)——模型时间 `f/24` 的音频事件变速后恰落在该帧真实播放时刻 `f/src_fps`,任意 fps 精确同步零漂移;
  - **画面永不截**:`apad` 无限静音垫尾,使完整复制的视频流恒为 `-shortest` 的最短方。
- 失败兜底:退回 `save_video` 的常规写法并 `warning`。
- **v2a 不支持 `return_result_tensor`**:像素零损失依赖文件级 stream-copy,无法以解码 tensor 兑现(那是有损 VAE 重渲染 + 含补帧),入口即报错。
- **`save_result_path` 必填**:文件输出是 v2a 唯一出口,入口即校验非空——否则整轮编码/去噪跑完会静默不产出任何文件。

### 4.3 `lightx2v/utils/set_config.py`(~L209)
把 `"v2a"` 加入 `["i2v","s2v","rs2v","ltx2_s2v","v2av"]`,使 `target_video_length` 走 `1+8k` 取整校验(音视频帧长对齐所必需)。

### 4.4 `lightx2v/infer.py`(~L110)
`--task` 的 `choices` 加入 `"v2a"`。`--video_path` / `--prompt` / `--save_result_path` 已存在,无需新增参数。

### 4.5 新配置 `configs/ltx2/ltx2_3_v2a.json`
以 `configs/ltx2/ltx2_3.json`(基础联合 AV 模型 `ltx-2.3-22b-dev`,含 `mm_guider` 音频 `cfg_scale=7.0`)为底:
- **不挂** union-control IC-LoRA(不驱动画面)。
- `"use_upsampler": false`(画面来自原视频、冻结不变;上采样会改视频帧数破坏音视频对齐)。
- 保留 `mm_guider` / `enable_cfg`,音频侧强引导。
- 分辨率/帧数占位即可,运行时由输入视频覆盖。

### 4.6 新运行脚本 `scripts/ltx2/run_ltx2_3_v2a.sh`
镜像 `scripts/ltx2/run_ltx2_3_s2v.sh` 风格(无硬编码路径:`lightx2v_path`/`model_path`/`VIDEO_PATH` 变量 + `source base.sh`):
```bash
--model_cls ltx2 --task v2a \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_v2a.json \
--video_path "${VIDEO_PATH}" \
--prompt "..." --save_result_path .../output_ltx2_v2a.mp4
```

---

## 5. 风险与验证

### 5.1 风险
| # | 项 | 评估 | 处置 |
|---|-----|------|------|
| 1 | **画面像素零损失** | 架构保证(`-c:v copy` 复制原流,不经 VAE 重解码) | 无需模型验证,`ffprobe` 核对即可 |
| 2 | **音频质量 / 与画面同步**(唯一经验性风险) | 架构 in-distribution(s2v 冻结音频的镜像),但音质/对位需人耳确认 | 一次 smoke 跑;若不足,可像 v2av 挂 foley/audio LoRA 增强 |
| 3 | **音视频帧长对齐** | 已由"从输入视频推 `target_video_length` 再算两者 shape"保证(沿用 v2av/s2v) | 奇偶帧/极短片段各跑一次确认 |
| 4 | **上游是否有专用 V2A/foley 权重** | 首版用基础 AV 模型;若官方有专用 LoRA 更优 | 实现前确认 LTX-2.3 release 是否含 foley/audio LoRA |

### 5.2 端到端验证
1. 备一段短测试视频(3~5s / 24fps,帧数满足 `1+8k`)。
2. 跑 `scripts/ltx2/run_ltx2_3_v2a.sh`(单卡 bf16;prompt 描述期望声音)。
3. 检查:
   - `ffprobe` 确认输出**视频流与原视频逐帧一致**(分辨率/帧数/编码 copy),新增一条 audio 流。
   - 人眼比对画面无变化;人耳听音频与画面内容/动作是否匹配、有无破音/错位。
4. 边界:极短/极长、无音频源、奇数帧(触发 snap)各跑一次确认不崩。

---

## 6. 改动影响面小结
- 纯**新增**任务分支,不改动 t2av/i2av/v2av/s2v 既有路径与调度器算法。
- 触及文件:`ltx2_runner.py`、`utils/utils.py`、`set_config.py`、`infer.py` + 新增 1 配置 1 脚本。
- 无新增权重依赖(首版复用基础 AV 模型)。
