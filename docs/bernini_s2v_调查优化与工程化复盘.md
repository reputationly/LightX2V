# Bernini-R-S2V 调查、优化与 LightX2V 工程化复盘

> 日期：2026-07-27
> 状态：身份跟随问题已解决；首帧发白已通过可配置的工程策略规避；4×A100
> 端到端回归通过。
> 代码状态：修改保留在 LightX2V 工作区，按要求暂未提交、未推送。

## 1. 文档目的

本文记录 Bernini-R-S2V 从问题交接到最终工程化的完整过程，包括：

- 身份不跟参考图的现象、假设和实验矩阵；
- 对官方 Bernini、Wan S2V 和 ComfyUI 实现的对照分析；
- 最终根因及其代码层解释；
- 4×A100 性能和显存约束；
- 首帧发白/重影的测量、判断和当前处理；
- LightX2V 内的模型接入、双专家路由、配置、服务编排和 NFS 资产管理；
- 已排除方向、当前限制和建议的后续优化。

本文是调查复盘和后续研发依据。面向部署的简版说明见
`docs/bernini_s2v.md`。

---

## 2. 最终结论摘要

### 2.1 身份问题的确定根因

身份不跟参考图的根因不是权重损坏、提示词、CFG、`omega_img`、FP8，也不是
蒸馏 LoRA。

真正原因是早期 LightX2V 接入把 Bernini `context_latents` 错误划入了
**zero-timestep modulation** 区域。

Bernini 的 context token 必须：

- 使用独立的 source-id rotary phase；
- 不添加原生 Wan S2V 的 condition-mask embedding；
- 使用当前扩散 timestep 的调制。

只有原生 Wan S2V 的 reference/motion conditioning token 使用 zero
timestep。旧实现用一个序列边界同时表示“输出边界”和“timestep 调制边界”，
因此错误地把追加在目标 token 后面的 Bernini context token 当成原生 S2V
条件 token。

修复后，序列被明确表示为：

```text
[ target video tokens | Bernini context tokens | native ref/motion tokens ]
  <--- output/audio --->
  <------- current diffusion timestep -------->
                                            <--- zero timestep --->
```

对应两个独立边界：

- `target_seq_len`：目标视频 token 的末尾，用于输出 head 和音频注入；
- `timestep_seq_len`：目标视频加 Bernini context token 的末尾，用于 timestep
  调制切分。

同 seed、同输入、同配置的修复前后对照证明：

- 修复前是无关人物；
- 修复后回到参考图本人；
- 眼镜、Kinsta 上衣、麦克风、水杯、熔岩灯和房间构图均跟随参考图；
- 不需要增加独立的图像 CFG，也不需要 `omega_img`。

### 2.2 首帧发白的结论

首帧问题表现为：

- 第一帧整体亮度突然升高；
- 人物和手部带轻微叠影/过渡感；
- 从后续几帧开始快速恢复正常；
- 整段视频并没有持续过曝。

原始身份修复视频的亮度测量：

| 帧 | YAVG |
|---:|---:|
| 0 | 62.03 |
| 1 | 57.14 |
| 2 | 55.80 |
| 4 | 54.44 |
| 8 | 51.78 |
| 稳态 | 约 50.7 |

这个形态不像采样器或提示词导致的全局曝光偏差，更像 causal VAE 在“前置参考
latent”和“生成 latent”拼接边界产生的一帧解码过渡。

当前证据支持这一判断，但尚未通过 VAE 内部逐层对拍证明，因此应把它视为
**高可信工程判断**，而不是已经严格证明的模型理论根因。

当前生产策略为：

```text
仅将最终视频的第 0 帧替换为第 1 帧
```

该策略：

- 只影响 16 fps 下的 62.5 ms；
- 不改变帧数；
- 不移动音频时间轴；
- 不影响第二帧之后的生成序列；
- 通过 `s2v_stabilize_first_frame` 控制，默认关闭，只在 Bernini 生产配置开启。

修复后前两帧的 YAVG 分别为 57.1697 和 57.1676，原来的首帧亮度尖峰已消失。

### 2.3 工程归属

正式运行归属为 LightX2V：

- 模型 runner、context 条件接入、双专家路由在 LightX2V；
- API 服务和 4 卡序列并行在 LightX2V；
- 生产配置、容器构建、启动脚本和资产清单在 LightX2V；
- 模型权重和生成结果在 NFS；
- Bernini 仓只作为模型来源和算法参考，不再是部署时依赖。

---

## 3. 环境和正式资产

### 3.1 验证环境

| 项目 | 值 |
|---|---|
| Worker | `dev-gpustack-a100-0025` |
| GPU | 4×A100 40 GB |
| 主机内存 | 约 251 GB |
| 序列并行 | Ulysses，`seq_p_size=4` |
| 生产分辨率面积上限 | `399360` |
| 采样步数 | 20 |
| CFG | 关闭，scale 1 |
| shift | 8 |

### 3.2 NFS 模型目录

```text
/nfs-data/models/Bernini-R-S2V
/nfs-data/models/Bernini-R-S2V-lx2v-high
/nfs-data/models/Bernini-R-S2V-lx2v-low
/nfs-data/models/Wan2.2-S2V-14B
```

空间占用：

| 目录 | 大小 | 用途 |
|---|---:|---|
| `Bernini-R-S2V` | 约 62 GB | 原始 Bernini 双专家资产 |
| `Bernini-R-S2V-lx2v-high` | 约 31 GB | LightX2V 高噪声块式专家 |
| `Bernini-R-S2V-lx2v-low` | 约 31 GB | LightX2V 低噪声块式专家 |
| `Wan2.2-S2V-14B` | 约 46 GB | 共享 VAE、T5、wav2vec 等 S2V 资产 |

高、低噪声专家均采用 `--save_by_block` 布局，每个目录要求至少包含：

```text
config.json
diffusion_pytorch_model.safetensors.index.json
non_block.safetensors
block_0.safetensors
...
block_39.safetensors
```

低噪声目录中的公共资产通过 NFS 绝对软链接复用：

```text
Wan2.1_VAE.pth
  -> /nfs-data/models/Wan2.2-S2V-14B/Wan2.1_VAE.pth

models_t5_umt5-xxl-enc-bf16.pth
  -> /nfs-data/models/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth

wav2vec2-large-xlsr-53-english
  -> /nfs-data/models/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english
```

资产布局由
`configs/wan22/bernini_s2v_assets_nfs.json` 统一声明。启动服务前会检查两个专家
的 40 个 block、non-block、索引、配置以及关键共享资产。

### 3.3 正式验证产物

首帧已稳定的 LightX2V 仓内实现回归结果：

```text
/nfs-data/bernini_s2v_out/lightx2v_repo_20260727/bernini_s2v_repo_first_frame_fixed.mp4
```

前五帧抽帧：

```text
/nfs-data/bernini_s2v_out/lightx2v_repo_20260727/first_frames
```

身份已修复、但尚未做首帧稳定处理的结果：

```text
/nfs-data/bernini_s2v_out/identity_fix_20260727/bernini_s2v_identity_fixed.mp4
```

身份修复前的同 seed 对照：

```text
/nfs-data/bernini_s2v_out/identity_fix_20260727/pre_fix_control.mp4
```

当前回归视频规格：

| 项 | 值 |
|---|---:|
| 分辨率 | 832×448 |
| 视频帧率 | 16 fps |
| 视频帧数 | 77 |
| 时长 | 4.813 s |
| 音频 | 有 |
| 文件大小 | 1,169,628 bytes |

---

## 4. 模型关系和条件路径

### 4.1 模型关系

```text
Wan2.2 架构
└── ByteDance/Bernini-R
    ├── high-noise expert
    ├── low-noise expert
    └── 多任务 in-context 渲染能力
        └── Bernini-R-S2V
            └── Bernini 骨干 + Wan2.2-S2V 音频模块
```

Bernini-R-S2V 不是一套与 Wan S2V 完全无关的结构。它同时依赖：

- Bernini 的 in-context 视觉条件机制；
- Wan S2V 的参考、运动和音频注入机制；
- Wan2.2 的高、低噪声双专家采样。

### 4.2 官方 Wan S2V 与 Bernini 的身份路径不同

官方 Wan2.2-S2V 的身份主要通过原生 `reference_latent`/ref token 进入模型。

Bernini 的参考身份通过 `context_latents` 进入模型。其参考图片经 VAE 编码后，
作为独立的视觉 context stream 追加到序列，并使用不同的 source ID。

早期接入虽然把 context token 追加进了 attention 序列，但它们的 timestep
调制区域错误，所以“token 在场”并不等于“条件有效”。

### 4.3 提示词中的 `image0`

Bernini 官方 r2v/S2V 范式会在自然语言中明确引用 `image0`，例如：

```text
The man from image0 is speaking to the camera, keeping exactly the same face,
hair, glasses, clothes, room and lighting from image0.
```

这对稳定任务语义和场景一致性有帮助，但不能代替正确的视觉 context 条件。

实验中，打开和关闭无效 context 后，使用相同 `image0` 详细提示词得到的构图和
物体近乎一致，说明当时的眼镜、蓝衣、麦克风和熔岩灯主要来自文字描述，而不是
参考图条件真正生效。

---

## 5. 调试过程和实验矩阵

### 5.1 初始现象

初始 Bernini-R-S2V 可以生成：

- 正常的说话动作；
- 大致合理的口型；
- 符合文字描述的眼镜、蓝色上衣和录音室；

但输出人物不是参考图本人。更关键的是，多次配置变化后经常稳定地产生相似的
陌生人物，说明参考身份条件没有真正控制生成。

### 5.2 建立官方 Wan S2V 基线

首先用相同输入管线测试官方 Wan2.2-S2V：

- 参考人物身份正确；
- Kinsta 上衣、眼镜和工作室细节正确；
- 口型和动作正常。

这组基线排除了以下问题：

- 输入图片传错；
- 图片预处理或裁剪完全损坏；
- 音频路径有误；
- API 请求没有传图；
- VAE、音频编码和整体 S2V runner 完全不可用。

### 5.3 历史实验矩阵

| # | 模型 | Context | 原生 ref | 提示词 | CFG | shift | 结果 |
|---:|---|---|---|---|---:|---:|---|
| 0 | 官方 Wan S2V | 不适用 | 开 | 普通描述 | 4.5 | 3 | 身份正确 |
| 1 | Bernini 双专家 | 开 | 关 | 普通描述 | 4.5 | 3 | 陌生人，偏卡通 |
| 2 | Bernini 双专家 | 开 | 关 | 空提示词 | 1 | 8 | 明显跑飞 |
| 3 | Bernini 双专家 | 开 | 关 | `image0` 详述 | 4.5 | 8 | 场景吻合，但脸不对 |
| 4 | Bernini 双专家 | 关 | 开 | 同一 `image0` 详述 | 4.5 | 8 | 与 #3 近似 |
| 5 | Bernini 双专家 | 开 | 关 | `image0` 详述 | 1 | 8 | 仍为陌生人 |
| 6 | Bernini 双专家 | 开 | 关 | 官方原句和素材 | 1 | 8 | 无法复现官方人物 |
| 7 | Bernini 双专家 | 开，修正 timestep | 关 | `image0` 详述 | 1 | 8 | 参考身份恢复 |

关键证据：

1. #3 与 #4 近似，说明旧 context 路径几乎没有贡献。
2. #6 使用官方素材和文案仍失败，说明问题不只是提示词措辞。
3. #7 只修正 context token 的 timestep 边界，身份立即恢复。
4. #7 没有引入 `omega_img`，因此独立图像 CFG 不是本问题的必要条件。

### 5.4 对 source-id RoPE 的检查

Bernini 会给不同视觉来源分配独立 source ID，并在标准时空 RoPE 上叠加额外
相位：

```text
angle_j = source_id / theta^(2j/d)
```

LightX2V 的复数乘实现与 ComfyUI 的二维旋转矩阵实现做过数值对拍，最大角度
误差约 `4.4e-16`，属于浮点噪声。

因此 source-id 旋转方向和计算公式不是最终根因。

### 5.5 找到 timestep 边界错误

旧实现使用 `original_seq_len` 同时承担：

- 输出和音频注入边界；
- 当前 timestep 与 zero timestep 的切分边界。

在只有原生 Wan S2V token 时，这两个边界可以相同。但加入 Bernini context
之后，它们不再相同：

```text
output boundary   = target
timestep boundary = target + context
```

如果仍使用 target 末尾作为 timestep 边界，context token 会被错误地使用
zero timestep embedding。它们虽然进入 self-attention，但处于错误的调制分布，
实际身份约束非常弱。

这解释了此前所有现象：

- 开关 context 输出近似；
- 文字能控制场景，但参考脸不能控制身份；
- 修改 CFG、shift 和提示词不能从根本上修复；
- source-id RoPE 数学正确仍无效。

---

## 6. 最终代码设计

### 6.1 `pre_infer.py`

文件：

```text
lightx2v/models/networks/wan/infer/s2v/pre_infer.py
```

新增配置：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `s2v_context_latents` | `false` | 是否追加 Bernini context latent |
| `s2v_ref_token` | `true` | 是否保留原生 Wan S2V ref token |
| `rope_theta` | `10000.0` | source-id rotary 基数 |

主要处理：

1. 参考图片经 VAE 编码后作为 `context_latents` 传入。
2. 每个 context latent 经 `patch_embedding` 转为 token。
3. context token 追加在 target video token 后。
4. 每个 context stream 分配从 1 开始的 source ID。
5. 在标准 RoPE 上乘 source-id rotary phase。
6. context token 不添加原生 S2V condition-mask embedding。
7. 分别记录 `target_seq_len` 和 `timestep_seq_len`。
8. 原生 ref/motion token 仍保留 zero timestep 行为。

### 6.2 `s2v_model.py`

文件：

```text
lightx2v/models/networks/wan/s2v_model.py
```

Ulysses 会把全局序列切到不同 rank。代码必须分别保留：

- 全局输出边界；
- 全局 timestep 边界；
- 当前 rank 对应的局部 timestep 边界。

如果序列并行阶段再次把两个边界合并，即使单卡正确，4 卡仍会重新出现调制
错误。因此这部分不是单纯的性能适配，而是 4 卡正确性的一部分。

### 6.3 `wan_s2v_runner.py`

文件：

```text
lightx2v/models/runners/wan/wan_s2v_runner.py
```

包含三类改动。

第一类是参考图 context 注入：

```python
dit_inputs["s2v"]["context_latents"] = [ref_latents]
```

第二类是 `wan2.2_s2v_moe` runner：

- 显式读取高、低噪声专家路径；
- 服务启动时检查专家资产是否完整；
- 根据 scheduler timestep 在两个专家间切换；
- 仅保持当前专家常驻，切换时释放另一专家；
- 每个专家都使用 `WanS2VModel`，从而共享相同的 S2V/context 修复。

第三类是首帧稳定：

```python
video[:, :, 0] = video[:, :, 1]
```

该逻辑位于首个 clip 的 VAE 解码和裁剪之后、motion history 更新之前，仅在
配置开关启用时执行。

---

## 7. 双专家和采样配置

生产配置：

```text
configs/wan22/a100/bernini_s2v_moe_4gpu.json
```

关键参数：

```json
{
  "infer_steps": 20,
  "sample_shift": 8,
  "sample_guide_scale": 1.0,
  "enable_cfg": false,
  "boundary": 0.875,
  "s2v_context_latents": true,
  "s2v_ref_token": false,
  "s2v_stabilize_first_frame": true,
  "max_area": 399360,
  "cpu_offload": false,
  "seq_parallel": true,
  "parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses"
  }
}
```

专家路由规则：

```text
timestep >= 875 -> high-noise expert
timestep <  875 -> low-noise expert
```

当前采用 20 步 BF16 路径作为稳定生产基线。官方 ComfyUI workflow 中的 FP8
和 4 步蒸馏 LoRA 是性能优化方向，不是身份正确性的前提。

---

## 8. 性能、显存和容量边界

历史性能数据：

| 配置 | 生成耗时 | 显存峰值 | 结论 |
|---|---:|---:|---|
| 1 卡、block offload、CFG 4.5 | 约 777 s | 约 25.8 GB | 可运行但很慢 |
| 1 卡、block offload、CFG 1 | 约 463 s | 约 24.0 GB | 关闭 CFG 明显提速 |
| 4 卡、Ulysses、无 offload、CFG 1 | 约 186–198 s | 39,679 MiB/卡 | 当前 A100 基线 |
| 4 卡、面积 518400 | OOM | 接近 40 GB 上限 | A100 40 GB 不适合 |

需要特别注意：

- Ulysses 切的是序列，不切模型权重；
- 每个 rank 都需要完整专家权重；
- 40 GB 卡只剩约 7 GB 放激活；
- 4 卡时不能启用模型级 CPU offload。

模型级 offload 会在主机端形成多份锁页内存，四个 rank 可能合计约 130 GB，
容易导致节点内存压力和服务不稳定。

当前生产选择：

```text
4×A100 + seq parallel + no CPU offload + max_area 399360
```

---

## 9. 首帧发白的详细分析

### 9.1 解码路径

首个 clip 会把参考 latent 前置到生成 latent：

```python
decode_latents = torch.cat([ref_latents, generated_latents], dim=2)
image = vae_decoder(decode_latents)
image = image[:, :, -infer_frames:]
```

Wan VAE 是带时间因果结构的解码器。即使最终裁剪到生成区域，生成区的第一帧仍
可能受前置参考 latent 和时间卷积缓存影响。

首帧同时出现亮度突变和叠影，而不是纯粹饱和裁剪，支持“边界过渡”判断。

### 9.2 为什么没有直接调低曝光

不建议对整段视频做固定曝光修正，因为：

- 只有第一帧异常；
- 后续帧亮度正常；
- 全局调暗会损伤整段视频；
- prompt 中增加 `overexposed` 负面词无法修复 latent 边界。

也不优先对第一帧做线性亮度缩放，因为它除了偏亮还有轻微结构重影。单纯降低
亮度不能去掉叠影。

### 9.3 为什么当前选择复制第二帧

第二帧已经进入正常解码状态，且与首帧时间仅相差 62.5 ms。复制第二帧：

- 同时消除亮度尖峰和边界叠影；
- 不需要重新采样或重新解码；
- 不引入额外颜色变换；
- 对音画同步影响最小；
- 行为确定，容易开关和回归。

### 9.4 后续若要做模型级根治

建议按以下顺序调查：

1. 保存 VAE 输入 latent 和未经裁剪的全部解码帧。
2. 分别测试“带参考 latent 解码”和“只解码生成 latent”。
3. 对齐 VAE latent 时间索引、`drop_first_motion` 和最终帧裁剪索引。
4. 测试多丢弃一个边界帧后是否仍需复制。
5. 检查 VAE temporal cache/warm-up 对第一生成帧的影响。
6. 对比逐 clip 解码与整段一次性解码。
7. 用逐帧 YAVG、直方图距离、LPIPS 和相邻帧光流量化跳变。

在没有完成上述对拍前，不应声称已经修复 VAE 内部根因。当前实现是经过验证的
生产规避策略。

---

## 10. LightX2V 工程化落地

### 10.1 仓内文件

| 文件 | 作用 |
|---|---|
| `lightx2v/models/networks/wan/infer/s2v/pre_infer.py` | Context token、source-id RoPE、双边界 |
| `lightx2v/models/networks/wan/s2v_model.py` | Ulysses 下保留输出和 timestep 边界 |
| `lightx2v/models/runners/wan/wan_s2v_runner.py` | Context 传递、MoE runner、首帧稳定 |
| `configs/wan22/a100/bernini_s2v_moe_4gpu.json` | 4×A100 生产配置 |
| `configs/wan22/bernini_s2v_assets_nfs.json` | NFS 模型资产清单 |
| `scripts/server/start_bernini_s2v_a100.sh` | 资产预检和服务启动 |
| `scripts/server/start_bernini_s2v_a100_docker.sh` | Docker 服务编排 |
| `dockerfiles/Dockerfile_bernini_s2v_a100` | A100 生产镜像 |
| `docs/bernini_s2v.md` | 简版部署说明 |
| 本文 | 调试、优化和工程化复盘 |

### 10.2 服务启动

宿主机直接启动：

```bash
bash scripts/server/start_bernini_s2v_a100.sh
```

容器化启动：

```bash
docker build \
  -f dockerfiles/Dockerfile_bernini_s2v_a100 \
  -t lightx2v:bernini-s2v-a100 \
  .

bash scripts/server/start_bernini_s2v_a100_docker.sh
```

启动脚本会：

1. 读取 NFS 资产清单；
2. 检查两个专家和共享资产；
3. 创建输出目录；
4. 设置 4 卡和 CUDA 内存参数；
5. 通过 `torchrun` 启动 LightX2V API server；
6. 使用 `wan2.2_s2v_moe` runner。

### 10.3 容器源码一致性踩坑

第一次构建只覆盖了三个本次修改的 Python 文件。基础 A100 镜像内的
LightX2V 版本比当前仓库旧，结果出现：

```text
ImportError: cannot import name 'WanS2VOffloadTransformerInfer'
```

原因不是 Bernini 修改本身，而是：

- 新版 `s2v_model.py`；
- 旧版 `transformer_infer.py`；
- 两者被拼进同一个运行环境。

最终 Dockerfile 改为复制当前仓库的完整 `lightx2v/` Python 源码，而不是只
覆盖少量文件：

```dockerfile
COPY lightx2v lightx2v
```

CUDA、PyTorch 和系统依赖仍继承经过验证的 A100 基础镜像。这样既避免重装大型
GPU 依赖，又保证 LightX2V Python 模块来自同一个代码版本。

这是后续构建必须保留的原则：

```text
可以复用基础运行环境，但不能混用不同提交的 LightX2V Python 模块。
```

### 10.4 NFS 构建快照

本次完整源码构建上下文：

```text
/nfs-data/_build/lightx2v_bernini_s2v_full_20260727
```

构建日志：

```text
/nfs-data/_build/lightx2v_bernini_s2v_full_20260727/build.log
```

经过端到端回归的节点本地镜像标签：

```text
lightx2v:bernini-s2v-repo-20260727-v2
```

该标签目前是 0025 节点本地镜像，不等同于已推送到镜像仓库。正式发布前应按
团队镜像命名规范重新打 tag 并推送。

---

## 11. 验证结果

### 11.1 静态检查

以下检查已通过：

- 修改的三个 Python 文件可编译；
- 两个 shell 脚本通过 `bash -n`；
- 两个 JSON 配置可解析；
- `git diff --check` 通过；
- 仓库 pre-commit 中的 ruff、ruff-format、尾空格、冲突标记等检查通过。

本地 Mac 环境没有完整的 CUDA、Torch 和 LightX2V 运行依赖，因此真正的模块
导入和推理回归在 0025 的 GPU 容器内完成。

### 11.2 GPU 容器导入

完整源码镜像在 GPU 容器内成功导入，并确认注册：

```text
registered = True
runner = WanS2VMoeRunner
```

### 11.3 端到端回归

回归确认：

- 高噪声专家正常加载；
- 低噪声专家正常切换；
- 4 rank NCCL/Gloo 初始化成功；
- Ulysses 序列并行运行成功；
- 峰值显存约 39,679 MiB/卡；
- GPU 推理阶段利用率约 99%–100%；
- 生成视频包含音频；
- 参考人物身份和主要场景元素保持；
- 首帧亮度尖峰和叠影已消失。

修复后前十帧 YAVG：

```text
57.1697
57.1676
55.7664
55.2051
54.4202
53.6142
52.8315
51.9499
51.8168
50.9684
```

首帧与第二帧因视频编码可能存在极小像素差异，但亮度和视觉内容已经一致，不再
出现原来的异常边界帧。

---

## 12. 已排除的方向

以下方向已经有足够对照证据，不应作为下一轮首选：

1. **输入图片错误**

   官方 Wan S2V 在相同输入链路下可以正确锁定身份。

2. **Bernini 权重损坏**

   官方实现和最终修复结果均证明权重具备参考身份能力。

3. **只靠提示词解决**

   `image0` 详述能改善场景，但无法修复失效的视觉 context。

4. **CFG 大小是根因**

   CFG 4.5 和 CFG 1 都无法修复旧 context；修正 timestep 后 CFG 1 即可。

5. **必须实现 `omega_img`**

   最终身份恢复没有使用独立图像 CFG。

6. **source-id RoPE 公式错误**

   已与参考实现数值对拍。

7. **FP8 或 4 步蒸馏 LoRA 是身份前提**

   当前 BF16、20 步路径已经恢复身份。FP8/LoRA 属于后续性能优化。

8. **首帧发白是整段曝光问题**

   亮度逐帧快速收敛，只有边界第一帧异常。

---

## 13. 当前限制和风险

### 13.1 A100 40 GB 余量很小

生产配置已接近显存上限。增加分辨率、context 数量、CFG 双路计算或更长序列都
可能 OOM。

### 13.2 当前首帧方案是规避，不是 VAE 内部根治

首帧复制策略稳定、简单，但会让视频开头有两张近似相同的帧。16 fps 下通常不
明显，但对高帧率、精确动作起始或逐帧分析场景需要进一步研究 VAE 边界。

### 13.3 首次请求会有专家加载延迟

双专家采用单专家常驻策略以适配 40 GB 显存。首次高噪声加载和切到低噪声时会有
明显 NFS 权重加载开销。

后续可以评估：

- 更快的本地 NVMe 缓存；
- 权重预热；
- 更细的加载并发控制；
- 量化专家；
- 在更大显存 GPU 上双专家同时常驻。

### 13.4 资产预检还可以更严格

当前检查文件是否存在，但没有检查：

- 文件大小；
- SHA256；
- safetensors 是否可读；
- index 中声明的 shard 是否全部存在；
- wav2vec 权重文件的完整清单。

生产资产发布建议增加带版本号和 checksum 的 manifest。

### 13.5 修改尚未提交

按当前要求：

- LightX2V 工作区内修改已保留；
- 没有 Git commit；
- 没有 push；
- 原有未跟踪文件没有被覆盖或删除。

---

## 14. 后续优化建议

### P0：补自动化正确性回归

建议增加最小测试，覆盖：

- context 开关关闭时，官方 Wan S2V 路径行为不变；
- context 开启时，`target_seq_len < timestep_seq_len`；
- context span 的 cond mask 为零；
- native ref/motion span 使用 zero timestep；
- Ulysses 每个 rank 的局部 timestep 边界正确；
- 首帧稳定开关默认关闭、开启时只替换一帧；
- 高/低专家缺文件时启动快速失败。

### P1：建立标准视觉评测

固定一组输入和 seed，记录：

- ArcFace/InsightFace 身份相似度；
- SyncNet 唇音同步；
- 首帧与后续帧亮度差；
- 相邻帧 LPIPS；
- 人脸区域和全图的光流；
- 生成耗时和各阶段耗时；
- GPU 峰值和 NFS 读取量。

仅凭肉眼判断适合排障，但不适合长期回归。

### P1：模型资产版本化

建议 manifest 增加：

```text
model_family
source_revision
conversion_revision
dtype
block_count
sha256
shared_asset_revision
compatible_lightx2v_revision
```

并在镜像启动时打印资产版本，避免“代码更新但仍加载旧转换权重”。

### P1：镜像发布和可追溯性

正式镜像 tag 应至少包含：

- LightX2V Git SHA；
- Bernini 资产版本；
- CUDA/PyTorch 基础镜像版本；
- 架构标识；
- 构建日期。

例如：

```text
lightx2v:bernini-s2v-<git-sha>-a100
```

### P2：4 步蒸馏和量化

在保持 20 步 BF16 基线不变的前提下，单独验证：

- Bernini-R LightX2V high/low noise LoRA；
- 4 步 scheduler 和专家切换位置；
- FP8 权重；
- 身份相似度是否下降；
- 首帧边界是否变化；
- 性能提升和显存下降。

不要把 4 步、FP8 和身份修复混在同一个实验中，否则难以定位回归来源。

### P2：VAE 首帧根因研究

如果业务不能接受重复一帧，再投入 VAE 级研究。优先做 latent/解码索引对拍，
不要先做全局色彩后处理。

### P2：服务运行指标

建议暴露：

- 当前加载专家；
- 专家切换耗时；
- NFS 权重读取耗时；
- 每阶段显存峰值；
- T5/VAE/audio/DiT 分阶段时延；
- 首帧亮度异常计数；
- 任务失败类型。

---

## 15. 推荐的后续工作顺序

1. 让业务方先查看当前 NFS 回归视频，确认人物和动作主观质量。
2. 保留 20 步 BF16 配置作为黄金基线。
3. 补充自动化序列边界测试和固定 seed 回归。
4. 完善模型 manifest 和 checksum。
5. 按团队规范构建并推送带 Git SHA 的镜像。
6. 再独立评估 4 步 LoRA、FP8 和更快的专家加载。
7. 只有业务明确不能接受首帧复制时，再做 VAE 内部根因研究。

---

## 16. 核心经验

这次问题最重要的经验不是某个参数，而是条件 token 的“语义边界”必须在工程
实现中被显式表达。

新增一种 token 后，需要逐项确认：

- 它是否参与 self-attention；
- 使用哪种 RoPE/source ID；
- 是否带 condition-mask embedding；
- 使用当前 timestep 还是 zero timestep；
- 是否参与音频注入；
- 是否进入输出 head；
- 在 sequence parallel 切分后边界是否仍正确。

旧实现只确认了“context token 已追加到序列”，但没有确认它在 timestep 调制
上的语义。最终正是这个边界错误让参考图条件看似存在、实际上几乎失效。

对于模型工程，能进入张量图不等于条件已经正确接入；位置编码、调制、mask、
输出范围和并行切分共同决定了 token 的真实语义。
