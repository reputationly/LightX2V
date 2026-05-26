# LTX 2.3 在鲲鹏 ARM + A100 上的本地化 POC — 交接文档

> 面向接手的 agent：读完本文应当能完全复现当前进度、理解每个失败的根因、并直接接着「最新计划」往下做，无需回看历史会话。
> 最后更新：2026-05-26（已完成单卡 A100 121 帧验证）。

---

## 0. 一句话现状

**已跑通。** LightX2V server 模式下，LTX 2.3 distill 1.1 已在 **单张 A100 40GB** 上完成 `1280×768 / 121 帧 / 24fps` 生成验证，5 条 prompt 全部成功，`blackdetect` 均未发现黑屏区间。

核心修复有两处：
- gemma-3-12b 文本编码器从“整模型搬 GPU / 或全 CPU 慢跑”改为 **逐层 GPU 流式前向**，解决文本编码 OOM 和 ARM CPU 极慢问题。
- LTX VAE decode 的 generator 消费过程包进 `torch.inference_mode()`，并把 VAE tiling 调小到 `256px / 16 frames`，解决保存视频阶段显存暴涨 OOM。

当前推荐启动脚本：`/data/start_ltx_server_single_fixed.sh`。当前批量验证脚本：`/data/run_ltx5.sh`（已修 `status` 字段并改为 121 帧）。

---

## 1. 任务目标

用 **server 常驻模式**（`lightx2v.server`，FastAPI+uvicorn）在本地把 **LTX 2.3 蒸馏版 v1.1** 跑起来，用与之前 Wan 测试**相同的 5 条提示词**、**尽量一致的视频规格（1280×768 / 121 帧）** 出片。允许下载权重。

5 条提示词的命名（沿用 Wan 测试）：`night_market` / `hummingbird` / `mountain_drone` / `coffee_rain` / `horse_beach`。

---

## 2. 硬件与环境

| 项 | 值 |
|---|---|
| 服务器 | `111.172.214.29`，SSH 别名 `edt-vpn`，root 用户 |
| 架构 | 鲲鹏 ARM **aarch64** |
| GPU | 4 × A100 **40GB** PCIe（sm_80） |
| CUDA | 12.8 |
| 宿主内存 | 251GB（swap 仅 3GB；测试中未触发 swap） |

### 镜像（`docker images | grep lightx2v`）
| 镜像 | 用途 |
|---|---|
| `lightx2v-arm64:ltx` | **当前 LTX 用的镜像**，transformers **4.57.1** + huggingface_hub 0.35.3（这个组合能跑通 LTX）。Dockerfile：`LightX2V/dockerfiles/Dockerfile_aarch64_cu128`（A100 sm_80 专用，torch 2.11.0 cu128，flash_attn 2.7.4.post1，sageattention；通用镜像需要保留 Wan2.2 Lightning 的 int8 能力，`torchao` 是必需运行依赖，`q8_kernels/sgl-kernel/SpargeAttn` 仍按可用性保留或可选编译）。 |
| `lightx2v-arm64:common-fixed` | 2026-05-26 新构建的 A100 通用 app 镜像；base 复用 `lightx2v-arm64:ltx`，通过 `Dockerfile_aarch64_app` 只烘入当前 Python 代码，镜像 id `f055ea1604cf`。LTX 服务启动 smoke 已通过；Wan2.2 4 卡 `torchrun -m lightx2v.server` 已进入 int8-torchao 量化权重加载路径，未提交生成任务。 |
| `lightx2v-arm64:server` | Wan 用的常驻镜像 |
| `lightx2v-arm64:local` | 早期镜像 |

### 模型路径（服务器）
- DiT 蒸馏权重：`/data/models/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors`（46GB bf16，**只有 bf16，无预量化 int8 权重**）
- 上采样器：`/data/models/Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- gemma 文本编码器：`/data/models/google/gemma-3-12b-it-qat-q4_0-unquantized`（unquantized bf16，~24GB）
- 注意 `/data/models/Lightricks/LTX-2.3/` 下多个 `.safetensors` 是软链到 `LTX-2___3/` 真实目录。

### 网络限制
- 宿主 **无法访问** `files.pythonhosted.org`（PyPI）。装包必须用清华镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...`

---

## 3. lightx2v.server 用法速查

- 启动：容器内 `python -m lightx2v.server --model_cls ltx2 --task t2av --model_path <dir> --config_json <json> --host 0.0.0.0 --port 8000`（多卡用 `torchrun --nproc_per_node=N`）。
- API：
  - `POST /v1/tasks/video/` body `{prompt, negative_prompt, save_result_path, target_video_length, infer_steps, seed}`（请求字段会覆盖 config）→ 返回 `{task_id, ...}`
  - `GET /v1/tasks/{id}/status` → 返回字段 **`status`**（值：`pending`/`processing`/`completed`/`failed`，失败时有 `error`）。**注意不是 `task_status`！** 早期脚本读错字段导致轮询死循环（见 §7 run_ltx5.sh 坑）。
  - `GET /health` → 200 表示就绪
  - metrics 在 8001 端口

---

## 4. 根因分析（最重要，务必读）

### 4.1 OOM 不在 DiT，在 gemma 文本编码器
- 现象：单卡 121 帧约 16s（开 upsampler 时）就 OOM，且 vae_cpu_offload 开不开都一样快地 OOM → 说明 OOM 发生在**计算早期**，不是结尾的 VAE 解码。
- 抓到完整 traceback：OOM 落在 `model.py:152 encode_text → base_encoder.py:54 precompute → transformers/gemma3 forward → RMSNorm`。
- 代码定位：`LightX2V/lightx2v/models/input_encoders/hf/ltx2/model.py` 的 `encode_text`（约 137–156 行）在 `cpu_offload=True` 时，会把**整个 gemma 模型（~24GB bf16）一次性 `.to(AI_DEVICE)`** 做前向，算完再搬回 CPU。
- 叠加常驻的 DiT non-block 权重 + VAE + 上采样器（~14GB）→ ~38GB → 在 gemma 前向（RMSNorm）时 OOM。空闲常驻其实只有 ~2–3GB（DiT 用 block offload 流式，gemma 平时在 CPU）。

### 4.2 为什么 DiffSynth 用 cpu offload 没问题
DiffSynth 的 `enable_vram_management()` 是**逐层/逐模块流式 offload**（一次只把 gemma 的一层放上 GPU，峰值几百 MB）。lightx2v 的 ltx2 gemma offload 是**全有或全无**（整 24GB 模型整体搬）。`offload_granularity:block` 只作用于 **DiT transformer blocks**（常驻 2 块），**不管 gemma**。
→ 这正是「per-layer 流式」方向的依据（§9）。

### 4.3 高分辨率工作流（官方机制）
LTX 高分辨率 = 低分辨率生成 + ×2 空间上采样。`use_upsampler:True` 时 DiT 主干 pass 在 `target_height//2 × target_width//2` 跑（`ltx2_runner.py:196-198`），之后上采样器把 latent ×2 到目标分辨率，再 VAE 解码。官方 `ltx2_3_distill_offload.json` 默认 768×512；`ltx2_3_distill_upsample_offload.json` 目标 1536×1024（应是给 80GB 卡用）。

### 4.4 瓶颈转移后的新 OOM（gemma 上 CPU 之后）
gemma 不再占 GPU 后，OOM 转移到 **DiT 主干 + 上采样器 + VAE 解码在 768×1280 下的激活**。属于**激活受限**（activation-bound），不是权重受限。这决定了多卡序列并行（ulysses）理论上应该能解（切激活），而张量并行/权重量化解不了根本（见 §6）。

---

## 5. 关键文件清单（服务器 `/data` 下）

### 配置（`/data/lightx2v_configs/`）
| 文件 | 说明 |
|---|---|
| **`ltx2_3_distill_v11_hq.json`** | **单卡基准配置**（本 POC 的主配置）。已包含：`infer_steps:8, target_video_length:121, 768×1280, cpu_offload:true, offload_granularity:block, vae_cpu_offload:true, use_upsampler:true, use_tiling_vae:true`，upsampler/dit/gemma ckpt 路径齐全，`distilled_sigma_values_upsample:[0.909375,0.725,0.421875,0.0]`。全文见服务器。 |
| **`ltx2_3_distill_v11_hq_ul4.json`** | 4 卡 ulysses 配置 = 上面单卡配置 + `parallel:{seq_p_size:4, seq_p_attn_type:ulysses}`。 |
| `ltx2_3_distill_v11_hq_4card.json` | **旧的** ulysses 配置，**已过时**（无 upsampler/tiling，vae_cpu_offload:false）。别用，用 ul4。 |
| `ltx2_3_distill_v11_hq_tp4.json` | 张量并行 4 卡配置（gemma 仍复制，解不了，弃用）。 |
| `ltx2_3_distill_v11_hq_int8.json` | int8-torchao 尝试，**失败**（KeyError weight_scale，见 §6）。 |

### 启动脚本（`/data/`）
| 文件 | 说明 |
|---|---|
| **`start_ltx_server_single_fixed.sh`** | **当前推荐启动脚本**（单卡 GPU0）。挂载全部 LTX 修复补丁：`ltx2_model.py`、`ltx2_base_encoder.py`、`ltx2_vae_model.py`、`ltx2_runner.py`、`ltx2_media_io.py`。启用 `LTX_GEMMA_ON_CPU=1` + `LTX_GEMMA_LAYERWISE_GPU=1`，并设置 VAE tiling：`LTX_VAE_SPATIAL_TILE=256`、`LTX_VAE_SPATIAL_OVERLAP=32`、`LTX_VAE_TEMPORAL_TILE=16`、`LTX_VAE_TEMPORAL_OVERLAP=8`。 |
| **`start_ltx_server_single.sh`** | **单卡启动**（GPU0）。已加 `-e LTX_GEMMA_ON_CPU=1` 和 `-v /data/patches/ltx2_model.py:<容器内 model.py 路径>` 挂载补丁。`.bak` 是改之前的备份。 |
| `start_ltx_server_single_debug.sh` | 调试启动脚本。比 fixed 多 `LTX_DEBUG_STATS=1`，会在日志中打印 LTX latent/RGB 数值统计；定位黑屏/OOM 时使用，不建议常驻线上使用。 |
| **`start_ltx_server_ul4.sh`** | 4 卡 ulysses 启动（torchrun nproc=4），同样挂补丁 + 环境变量，用 ul4 配置。脚本顶部有 `docker rm -f` + 等待容器名消失的循环。 |
| `start_ltx_server.sh` / `_int8.sh` | 旧的多卡 / int8 启动脚本。 |
| `start_lx2v_server.sh` | **Wan 常驻服务**的启动脚本（POC 结束后可能要恢复 Wan）。 |

### 补丁与探针
| 文件 | 说明 |
|---|---|
| **`/data/patches/ltx2_model.py`** | **gemma-on-CPU 补丁文件**（见 §8）。通过 `-v` 挂载覆盖容器内 `/opt/LightX2V/lightx2v/models/input_encoders/hf/ltx2/model.py`，免重建镜像。已加 `import os`，`encode_text` 用 env `LTX_GEMMA_ON_CPU=1` 门控：跳过整模型搬 GPU，gemma 在 CPU 算，只把输出 `v_context/a_context` 搬到 AI_DEVICE。 |
| **`/data/patches/ltx2_base_encoder.py`** | **gemma 逐层 GPU 流式补丁**。覆盖容器内 `lightx2v/models/input_encoders/hf/ltx2/gemma/encoders/base_encoder.py`。`LTX_GEMMA_LAYERWISE_GPU=1` 时，手动复现 transformers Gemma3 text forward：embedding/rotary/layer/norm 逐模块搬到 GPU，用完搬回 CPU，输出 hidden states 供 feature extractor/embeddings processor 使用。 |
| **`/data/patches/ltx2_vae_model.py`** | **VAE decode 修复补丁**。覆盖容器内 `lightx2v/models/video_encoders/hf/ltx2/model.py`。视频/音频 decode 包 `torch.inference_mode()`；VAE tiling 支持环境变量配置，fixed 脚本使用 `256px spatial tile + 16 frame temporal tile`。这是 33/49/81/121 帧 VAE 保存阶段 OOM 的关键修复。 |
| **`/data/patches/ltx2_runner.py`** | LTX runner 诊断补丁。`LTX_DEBUG_STATS=1` 时打印 Stage1/upsampler/Stage2/VAE 前 latent 的 shape/min/max/mean/std/finite 统计；fixed 脚本未开启该变量。 |
| **`/data/patches/ltx2_media_io.py`** | LTX 保存链路诊断补丁。`LTX_DEBUG_STATS=1` 时打印保存前第一块 RGB video chunk 统计；fixed 脚本未开启该变量。 |
| **`/data/ltx_one.sh`** | **单 prompt 探针**（正确读 `status` 字段 + 每 5s 记录 GPU0 显存峰值）。用法 `bash /data/ltx_one.sh <帧数>`，默认 81。 |
| `/data/run_ltx5.sh` | 5 prompt 批量脚本。**已修复**：轮询读取 `status` 字段；输出目标已改为 `ltx_${name}_768p121.mp4`，`target_video_length:121`。备份：`/data/run_ltx5.sh.bak_ltxfix`。 |
| `/data/outputs/ltx_probe*.log` | 各次探针日志（见 §6 结果表）。 |
| `/data/outputs/ltx_run5_768p121_fixed.log` | 5 条 prompt 批量验证日志。 |
| `/data/outputs/ltx_smoke_night_market.mp4` | 早期冒烟产物（1.3MB，非本规格）。 |

### 当前验证产物（服务器 `/data/outputs/`）
| 文件 | 结果 |
|---|---|
| `ltx_night_market_768p121.mp4` | 1280×768，121 帧，24fps，5.041667s，`blackdetect=none`，大小约 2.0MB。 |
| `ltx_hummingbird_768p121.mp4` | 1280×768，121 帧，24fps，5.041667s，`blackdetect=none`，大小约 989KB。 |
| `ltx_mountain_drone_768p121.mp4` | 1280×768，121 帧，24fps，5.041667s，`blackdetect=none`，大小约 1.6MB。 |
| `ltx_coffee_rain_768p121.mp4` | 1280×768，121 帧，24fps，5.041667s，`blackdetect=none`，大小约 633KB。 |
| `ltx_horse_beach_768p121.mp4` | 1280×768，121 帧，24fps，5.041667s，`blackdetect=none`，大小约 855KB。 |

---

## 6. 所有尝试与结果（按时间）

测试方法统一：启动对应 server → `/health` 200 后 → `bash /data/ltx_one.sh <帧数>` 提交单 prompt（night_market），轮询 `status` 并记录 GPU 峰值。

| # | 配置 | 帧数 | 结果 | 峰值显存 | 关键观察 |
|---|---|---|---|---|---|
| 1 | 单卡，gemma 默认（上 GPU） | 121 | **OOM** | ~38GB | OOM 在 gemma RMSNorm，~16s 就挂。根因点。 |
| 2 | 单卡 + vae_cpu_offload | 121 | **OOM** | — | 更快挂（15s），证明 OOM 不在 VAE。 |
| 3 | 单卡 int8-torchao + offload | 121 | **失败** | — | `KeyError weight_scale`。auto-quant（`weight_auto_quant:true`）与 cpu_offload block buffer 不兼容；offload 预分配 buffer 要读 `.weight_scale`，而 on-load 量化是 lazy 的。int8+offload 需要**磁盘上的预量化权重**，但 LTX 2.3 只有 bf16，且 `tools/convert/converter.py` **不支持 ltx2**（只支持 wan_dit/qwen_image_dit 等）→ 无离线量化路径。 |
| 4 | 单卡 ulysses（gemma 仍上 GPU） | — | **OOM** | — | gemma 先 OOM，与并行无关。 |
| 5 | **单卡 + gemma-on-CPU 补丁** | 121 | **进展!** 跑过编码，DiT/解码尾部 OOM | 39567MiB | gemma 在 CPU 算 ~5min（GPU 全程 ~2GB），然后 GPU 爬升 4→7→17→39.5GB，在解码尾 OOM（339s）。**证明 gemma-on-CPU 解决了文本编码 OOM**。 |
| 6 | 单卡 + gemma-on-CPU | 81 | **OOM（更靠后）** | 40015MiB | DiT 8 步全跑完（顶在 39.5GB），降到 21GB 转入解码，又爬回 40GB OOM（300s）。失败在 **VAE 解码/上采样尾**。 |
| 7 | 单卡 + gemma-on-CPU + **use_tiling_vae** | 81 | **OOM** | 40423MiB | 开了分块解码仍挂，但失败点变成 DiT/上采样爬升中（6→40GB，294s）。说明 VAE 已被 tiling 约束住，**剩余压力在 DiT+上采样器激活本身** → 激活受限。 |
| 8 | **4 卡 ulysses + gemma-on-CPU + tiling** | 121 | **未完成（编码极慢）** | GPU 全程 2183MiB×4 | 4 rank 加载耗时 ~18min（每 rank 各加载 gemma 到 CPU）。提交后 **>67min（4021s）GPU 从未爬升**，4 个 python 进程 State R、各 ~44–64% CPU、无 swap、内存充足 → **不是死锁，是 gemma bf16 在 ARM CPU 上 ×4 rank 并发、内存带宽争抢导致极慢**。ulysses 让每个 rank 都各算一遍 gemma（冗余），是慢的主因。**未能跑到 DiT 阶段，所以「ulysses 能否解 DiT 高分辨率 OOM」尚未被证实。** |
| 9 | **单卡 + gemma 逐层 GPU + debug stats** | 17 | **成功** | 15165MiB | 66s 完成，1280×768/17 帧。证明逐层 GPU gemma 能绕过 CPU 慢编码并进入 DiT/upsampler/VAE 全链路；`blackdetect=none`。 |
| 10 | 单卡 + gemma 逐层 GPU，VAE 未修 | 49 | **OOM** | 40395MiB | Stage1/upsampler/Stage2 latent 数值正常，OOM 明确发生在 VAE decode iterator 被保存链路消费时。 |
| 11 | 单卡 + gemma 逐层 GPU，VAE 未修 | 33 | **OOM** | 40375MiB | 同样在 VAE 保存阶段 OOM。由此确认问题不是 DiT/upsampler 激活，而是 VAE generator decode 未被 `inference_mode` 包住 + tiling 过粗。 |
| 12 | **单卡 + gemma 逐层 GPU + VAE inference_mode + 256/16 tiling** | 33 | **成功** | 15787MiB | 56s 完成，1280×768/33 帧，`blackdetect=none`。VAE OOM 修复生效。 |
| 13 | 同 #12 | 49 | **成功** | 16373MiB | 50s 完成，1280×768/49 帧，`blackdetect=none`。 |
| 14 | 同 #12 | 81 | **成功** | 17473MiB | 56s 完成，1280×768/81 帧，`blackdetect=none`。 |
| 15 | **同 #12，目标规格** | 121 | **成功** | 18879MiB | 65s 完成，1280×768/121 帧，24fps，5.041667s，`blackdetect=none`。 |
| 16 | **同 #12，5 prompt 批量** | 121×5 | **全部成功** | 单条探针峰值约 18.9GB | `night_market` 85s、`hummingbird` 86s、`mountain_drone` 70s、`coffee_rain` 71s、`horse_beach` 70s。5 个文件均为 1280×768/121 帧/24fps，`blackdetect=none`。 |

---

## 7. 已解决 / 已排除

- ✅ **文本编码 OOM 根因**：gemma 整模型上 GPU。最终用 **gemma 逐层 GPU 流式** 解决；早期 gemma-on-CPU 只作为定位手段，速度不可接受。
- ✅ **高分辨率机制**：低分辨率生成 + ×2 上采样器，`use_upsampler` 把 DiT 主干分辨率减半。
- ✅ **VAE 解码/保存阶段 OOM**：根因是 `video_vae.decode()` 返回 generator，真正 decode 在保存链路消费时发生；原实现没有把 generator 消费过程包进 `torch.inference_mode()`，叠加默认 VAE tiling `512px / 64 frames` 过粗。已用 `inference_mode` + `256px / 16 frames` tiling 解决。
- ✅ **黑屏验证**：17/33/49/81/121 单 prompt 以及 5 条 121 帧批量产物均经 `ffmpeg blackdetect` 检查，无黑屏区间；debug stats 中 latent 全为 finite，非全 0。
- ❌ 排除 `vae_cpu_offload` 解 OOM（OOM 不在 VAE）。
- ❌ 排除 int8-torchao auto-quant（与 offload 不兼容；无 ltx2 离线量化路径）。
- ❌ 排除张量并行（gemma 仍复制到每卡）。
- ❌ 排除「离线预计算 embedding」路线：`process_captions.py` 是**官方 `Lightricks/LTX-2` 仓库的训练侧脚本**（`packages/ltx-trainer/scripts/`，配 `process_dataset.py`），产物是给训练 dataloader 的 `.precomputed/conditions/`，**不是 lightx2v 推理链路**；且 lightx2v 的 `ltx2_runner.run_text_encoder`（约 677–700 行）**没有读缓存 embedding 的分支**，硬接需自写加载路径，不划算。

---

## 8. gemma-on-CPU 补丁详情（`/data/patches/ltx2_model.py`）

原 `encode_text`（`LightX2V/lightx2v/models/input_encoders/hf/ltx2/model.py` 本体也是这样）：
```python
def encode_text(self, prompts):
    if self.cpu_offload:
        self.text_encoder = self.text_encoder.to(AI_DEVICE)   # 整 24GB 搬上卡 → OOM 点
    result = []
    for prompt in prompts:
        v_context, a_context, _ = self.text_encoder(prompt)
        result.append((v_context, a_context))
    if self.cpu_offload:
        self.text_encoder = self.text_encoder.to("cpu")
    return result
```
补丁后（文件顶部加了 `import os`）：
```python
def encode_text(self, prompts):
    gemma_on_cpu = os.environ.get("LTX_GEMMA_ON_CPU", "") == "1"
    if self.cpu_offload and not gemma_on_cpu:
        self.text_encoder = self.text_encoder.to(AI_DEVICE)
    result = []
    for prompt in prompts:
        v_context, a_context, _ = self.text_encoder(prompt)
        if gemma_on_cpu:
            v_context = v_context.to(AI_DEVICE)
            a_context = a_context.to(AI_DEVICE)
        result.append((v_context, a_context))
    if self.cpu_offload and not gemma_on_cpu:
        self.text_encoder = self.text_encoder.to("cpu")
    return result
```
前提：`cpu_offload:true` 时 `load_text_encoder` 会把 gemma 整个建在 CPU（`text_encoder_device=cpu`），所以补丁只是「不再搬上卡」。
**注意**：`AI_DEVICE` 是字符串（如 "cuda"），由 `from lightx2v_platform.base.global_var import AI_DEVICE` 导入。`GemmaTextEncoder` 没有 `self.config`，别依赖它。

---

## 9. 当前稳定基线与后续方向

### 稳定基线：bf16 + 单卡 fixed
当前推荐作为 LTX 2.3 distill 1.1 的稳定基线：
```bash
ssh edt-vpn 'bash /data/start_ltx_server_single_fixed.sh'
```

该脚本仍使用 `lightx2v-arm64:ltx` 镜像，但通过 `/data/patches/*.py` bind mount 注入修复。启动后单卡 GPU0 常驻约 8.3GB；单条 `1280×768 / 121 帧` 探针峰值约 18.9GB，5 条 prompt 已全部验证通过。

### 镜像化建议
短期 POC 用 bind mount 没问题；长期部署建议把本地已修复代码 bake 进新镜像，例如：
```bash
docker build -f dockerfiles/Dockerfile_aarch64_app -t lightx2v-arm64:common-fixed .
```
这是推荐的收尾构建方式：基础依赖镜像不重编译，只把当前 LightX2V Python 代码烘进 app 层。然后启动脚本可去掉 5 个 `/data/patches/*.py` 文件挂载，只保留环境变量：
```bash
LTX_GEMMA_ON_CPU=1
LTX_GEMMA_LAYERWISE_GPU=1
LTX_VAE_SPATIAL_TILE=256
LTX_VAE_SPATIAL_OVERLAP=32
LTX_VAE_TEMPORAL_TILE=16
LTX_VAE_TEMPORAL_OVERLAP=8
```

已知构建提速点：
- **GitHub Actions 要走 base/app 分层**：先在 A100 服务器上低频构建并上传依赖 base（例如 `arronlee/lightx2v:arm64-cu128-a100-base`），GitHub Actions 再用 `Dockerfile_aarch64_app` 只做 `COPY .` + `pip install -e . --no-deps`。Actions 不能引用服务器本地的 `lightx2v-arm64:ltx` tag，必须用 registry 上已 push 的 base tag。
- **通用服务镜像不要按 LTX-only 裁剪**：Wan2.2 Lightning 仍需要 int8 能力，至少要保留 `torchao`；`q8_kernels/sgl-kernel/SpargeAttn` 这类加速或量化组件可以做成可选编译，但不应因为 LTX 当前 bf16 基线就从通用镜像能力里删除。
- **分离 base 层和 app 层**：`Dockerfile_aarch64_cu128` 用于少量重建的通用依赖 base；日常代码修复用 `Dockerfile_aarch64_app`，从已有 base 复制代码并 `pip install -e . --no-deps`，通常只需几十秒。
- **保留 Docker layer cache**：不要频繁改 `cu128` Dockerfile 中 pip/编译依赖层；LTX 代码改动只应命中最后 `COPY . /opt/LightX2V` 和 `pip install -e .` 层，或直接走 app 层。
- **flash-attn / SageAttention / q8_kernels 预构建**：如果后续需要重建 base，优先在服务器上复用已有 wheel/source/cache，或者单独做依赖 base 镜像；不要在每次业务代码变更时重新编译 CUDA 扩展。
- **服务器构建优先于 GitHub Actions**：ARM64 + CUDA 编译在 GitHub 上很容易超时；服务器本地构建可复用已有镜像层和 wheel/source 缓存。

GitHub Actions 示例：
```yaml
- name: Build and push ARM64 app image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: dockerfiles/Dockerfile_aarch64_app
    platforms: linux/arm64
    push: true
    build-args: |
      BASE_IMAGE=arronlee/lightx2v:arm64-cu128-a100-base
    tags: |
      arronlee/lightx2v:arm64-a100-latest
```

依赖升级时才重建并上传 base；平时只改 LightX2V Python 代码，走 app 镜像即可。

### 下一步：4 卡 ulysses
现在 gemma 已经逐层 GPU 流式，4 卡 ulysses 不再会卡在 ARM CPU 文本编码。可直接改造 `/data/start_ltx_server_ul4.sh`，挂载同样补丁或使用 fixed 镜像，并添加上述环境变量。建议测试顺序：
1. 先测 `1280×768 / 121 帧`，确认 ulysses 与修复兼容。
2. 再测更高规格，例如官方 upsample 默认接近 `1536×1024`。
3. 若目标是速度，要注意 gemma 可能仍每 rank 各算一次；4 卡主要收益在 DiT/upsampler 激活切分和更高规格可跑。

### int8 路线暂不建议主推
LTX 2.3 当前只有 bf16 safetensors；`weight_auto_quant:true` 与 block offload 组合曾失败于 `KeyError weight_scale`，且 `tools/convert/converter.py` 暂不支持 ltx2 离线预量化。Wan2.2 的 int8 成功经验不能直接套到 LTX2.3。若后续需要 LTX int8，应作为独立工程项：补 converter → 生成带 scale 的磁盘 int8 权重 → 再验证 offload/画质/速度。

### 对 Wan2.2 / 其他模型的影响
本次行为修复集中在 LTX2 路径：`input_encoders/hf/ltx2`、`runners/ltx2`、`video_encoders/hf/ltx2`、`ltx2_media_io`。Wan2.2 不走这些类。`LTX_DEBUG_STATS` 默认关闭，fixed 脚本也未设置该变量。

### A100 attention/backend 边界
本项目当前只面向 A100 部署，但要覆盖 A100 上的多个模型，至少包括 Wan2.2 Lightning 和 LTX 2.3 distill 1.1。因此“通用镜像”指 **A100 通用模型服务镜像**，不是 H100/5090 后端全覆盖。

- LTX 2.3 当前验证配置：`attn_type=flash_attn2`，bf16 + offload fixed。
- Wan2.2 Lightning 当前验证配置：`flash_attn2` 或 `sage_attn2`，`dit_quant_scheme=int8-torchao`，可 4 卡 ulysses。
- `flash_attn3`、`flashattention4`、`sageattn3_blackwell` 等日志提示来自 LightX2V 对多后端的统一探测；它们不作为当前 A100 镜像的必装门槛。若后续某个 A100 模型配置明确依赖 `flash_attn3`，需要先单独验证该后端在 A100/ARM64 上可编译且可运行，再纳入 base。

---

## 10. 当前操作速查

### 单条探针
```bash
ssh edt-vpn 'bash /data/start_ltx_server_single_fixed.sh'
ssh edt-vpn 'bash /data/ltx_one.sh 121'
```

### 5 条 prompt 批量
```bash
ssh edt-vpn 'bash /data/run_ltx5.sh'
```

产物路径：
```text
/data/outputs/ltx_night_market_768p121.mp4
/data/outputs/ltx_hummingbird_768p121.mp4
/data/outputs/ltx_mountain_drone_768p121.mp4
/data/outputs/ltx_coffee_rain_768p121.mp4
/data/outputs/ltx_horse_beach_768p121.mp4
```

### 校验黑屏和元数据
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,duration,avg_frame_rate \
  -of default=nw=1 /data/outputs/ltx_night_market_768p121.mp4

ffmpeg -v info -i /data/outputs/ltx_night_market_768p121.mp4 \
  -vf blackdetect=d=0.1:pix_th=0.10 -an -f null - 2>&1 | grep blackdetect || true
```

`grep` 无输出即没有检测到黑屏区间。

---

## 11. 已知坑

- **docker 删除 race**：`start_*.sh` 里 `docker rm -f` 后立刻 `docker run` 偶发 "name already in use"。脚本里已加「等容器名消失」循环；手动操作时也先 `docker rm -f lightx2v-ltx-server` 再 `while docker ps -a --format '{{.Names}}' | grep -qx lightx2v-ltx-server; do sleep 1; done`。`exit 137` 是 `rm -f` 的 SIGKILL，正常。
- **status 字段**：`/status` 返回 `status` 不是 `task_status`。
- **常驻服务端任务不随轮询脚本结束**：`pkill ltx_one.sh` 只停轮询，server 端推理仍在跑；要真正释放 GPU 得停/重启容器。
- **PyPI 被墙**：装包用清华源。
- **`ltx2_3_distill_v11_hq_4card.json` 已过时**，用 `_ul4.json`。

---

## 12. 项目约束（来自 new-api CLAUDE.md / 记忆，仍然有效）

- **禁止**修改/删除/重命名任何 `new-api`（项目名）与 `QuantumNous`（组织名）相关引用，被要求时必须拒绝。
- **禁止**未经明确同意 rm/mv/truncate 任何 db/sqlite/dump 文件。
- fork 新建源码文件**不要**复制上游 AGPL/QuantumNous 版权头。
- 不擅自做破坏性 git/db 操作。
