# LTX 2.3 在鲲鹏 ARM + A100 上的本地化 POC — 交接文档

> 面向接手的 agent：读完本文应当能完全复现当前进度、理解每个失败的根因、并直接接着「最新计划」往下做，无需回看历史会话。
> 最后更新：2026-05-27（已完成单卡 A100 121 帧验证；已完成代码提交/CI 修复；**ARM64 出包已迁移到阿里云 ACR + Docker Hub 双地分发，见 §13**）。

---

## 0. 一句话现状

**已跑通。** LightX2V server 模式下，LTX 2.3 distill 1.1 已在 **单张 A100 40GB** 上完成 `1280×768 / 121 帧 / 24fps` 生成验证，5 条 prompt 全部成功，`blackdetect` 均未发现黑屏区间。

核心修复有两处：
- gemma-3-12b 文本编码器从“整模型搬 GPU / 或全 CPU 慢跑”改为 **逐层 GPU 流式前向**，解决文本编码 OOM 和 ARM CPU 极慢问题。
- LTX VAE decode 的 generator 消费过程包进 `torch.inference_mode()`，并把 VAE tiling 调小到 `256px / 16 frames`，解决保存视频阶段显存暴涨 OOM。

当前推荐启动脚本：`/data/start_ltx_server_single_fixed.sh`。当前批量验证脚本：`/data/run_ltx5.sh`（已修 `status` 字段并改为 121 帧）。

出包进度（已完成，详见 §13）：
- ARM64 通用镜像出包已落地「阿里云 ACR + Docker Hub 双地分发」：base 在 A100 服务器构建后**直连推 ACR**，再用 crane 同步到 Docker Hub；日常 app 出包由 GitHub Actions `FROM Docker Hub base` 双推两地，约 4.5 分钟。
- 地址：ACR `crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v`、Docker Hub `arronlee/lightx2v`。
- 起因（已解决）：国内直推 Docker Hub 大 layer 被 GFW reset（torch 6.45GB 单层不可切 + Docker Hub 不支持断点续传），换国内 ACR 直连后一次过、零 RST。

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
      arronlee/lightx2v:arm64-a100-YYYYMMDD-HHMM-<shortsha>
      arronlee/lightx2v:arm64-a100-latest
```

依赖升级时才重建并上传 base；平时只改 LightX2V Python 代码，走 app 镜像即可。

### 已提交的本地代码
本地 Mac 仓库 `/Users/reputationly/Desktop/code/api/LightX2V` 已 push 到 `origin/main`。关键提交：

| commit | 说明 |
|---|---|
| `6651fcc0` | LTX2 A100 offload/黑屏修复：gemma 逐层 GPU、VAE decode inference_mode、debug stats、`.dockerignore`、`Dockerfile_aarch64_app`、交接文档。 |
| `e3726195` | GitHub Actions 改为 `Dockerfile_aarch64_app`，自动生成 `arm64-a100-YYYYMMDD-HHMM-<shortsha>` 与 `arm64-a100-latest` 两个 tag。 |
| `01f56a46` | ruff 自动修复 import 顺序；CI lint 已通过。 |

注意：`Dockerfile_aarch64_cu128` 只新增 `torchao` 并把 q8 注释改为通用 q8f 可选路径，**没有删除** Wan2.2 int8/q8/sgl/SpargeAttn 相关能力。通用镜像仍面向 A100 上的 Wan2.2 Lightning + LTX 2.3。

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

---

## 13. ARM64 镜像出包方案（阿里云 ACR + Docker Hub 双地分发）

> 本节记录 2026-05-26～27 把 ARM64（鲲鹏 / A100 sm_80）通用镜像出包流程从「卡在 Docker Hub」迁移到「阿里云 ACR + Docker Hub 双地」的完整过程、根因、最终架构与操作手册。**取代 §0 / §9 中关于 Docker Hub 上传的过时描述。**

### 13.1 目标
通用 A100 镜像（同时服务 Wan2.2 Lightning + LTX 2.3 distill 1.1 等），用 GitHub Actions 出 app 包，部署到国内 A100（鲲鹏 ARM）节点。注意构建 base 的机器与部署机器是**同一台**（`edt-vpn` = 111.172.214.29）。

### 13.2 最初的死结：国内直推 Docker Hub 被 GFW reset
服务器经 xray（v2rayA，systemd 管理，监听 127.0.0.1:10809）推 29GB base 到 Docker Hub，大 layer 反复 `connection reset by peer`。根因三条：
1. 最大层是 **torch 单个 pip 包 6.45GB**，一个 pip 包就装在一层里，**无法再切小**；
2. **Docker Hub 不支持大 blob 断点续传**（单层必须一口气 PUT，断了整层重来）；
3. 链路撑不住单层连续约 15 分钟（实测上行 ~7MB/s），最好的节点也只撑约 3 分钟就被 RST。

排除掉的方向（都无效）：
- **换 CF 节点**：v2rayA 里 40+ 个「节点」其实是同一个 vless 配置（同 uuid、同伪装域名 `edt.ovaijisuan.com`、同 ECH+fragment）的不同 Cloudflare 入口。延迟最低的 `saas.sin.fan` 反而每 16s 断，比 `arron.cf.090227`（约 3min）更差。GFW 针对的是流量特征（同 SNI），不是单纯某个 IP。
- **关 ECH / 关 fragment / 换协议**：`fragment` 是 `tlshello` 模式只管握手；协议（vless/ws）是订阅服务端写死的、客户端改不了；真正能治长连接 RST 的 XHTTP 需要服务端支持，订阅节点没有。
- **切层**：torch 6.45GB 单包切不动。

### 13.3 解决：换阿里云上海 ACR 直连
- 服务器在国内、上海 ACR 也在国内 → **直连**，绝不走 VPN（出墙绕回又慢又断）。
- **必须配 NO_PROXY**：docker daemon 的 systemd proxy drop-in（`/etc/systemd/system/docker.service.d/*.conf`）里 `NO_PROXY` 末尾加 `.aliyuncs.com`，然后 `systemctl daemon-reload && systemctl restart docker`，否则 docker 会把发往上海的流量也塞进 xray 出墙绕回。
- 用**公网地址**，不是 `-vpc` 那个（VPC 地址只有阿里云 ECS 在同一 VPC 内才能访问）。
- 结果：29GB **一次过、零 RST**。

### 13.4 最终架构：base/app 分层 + 双地分发

| 阶段 | 做法 | 耗时 |
|---|---|---|
| **base 构建**（升级依赖时，低频） | A100 服务器 `dockerfiles/Dockerfile_aarch64_cu128` 本地编译 → 直连推 ACR | 几小时（编译 flash_attn/Sage 等） |
| **base 同步到 Docker Hub**（base 重建后跑一次） | 手动触发 `sync-base-to-dockerhub.yml`，crane registry→registry 直传 | ~12 分钟 |
| **日常出包**（改代码） | push → `build-arm64-docker.yml` 在 GHA ARM64 runner 上 `FROM Docker Hub base` → 叠代码层（`Dockerfile_aarch64_app`） → 双推 ACR + Docker Hub | ~4.5 分钟 |
| **Release 记录** | app 镜像双推成功后，`build-arm64-docker.yml` 自动创建 GitHub Release，release tag 与镜像版本 tag 相同 | 自动 |

地址与 tag：
- ACR：`crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v`
- Docker Hub：`arronlee/lightx2v`
- base tag：`arm64-cu128-a100-base`；app tag：`arm64-a100-<YYYYMMDD-HHMM>-<shortsha>` + `arm64-a100-latest`
- GitHub Release tag：同 app 版本 tag，例如 `arm64-a100-20260527-0608-27f65081`；Release notes 内记录 ACR/Docker Hub 两套镜像地址、latest tag、base image 与 commit。
- 两边仓库都设**公开**，拉取免登录。

### 13.5 两条 workflow
- `.github/workflows/sync-base-to-dockerhub.yml`：手动触发（workflow_dispatch），用 `crane copy` 把 ACR base 搬到 Docker Hub。只登录目标 Docker Hub（源 ACR 公开仓免登录）。仅 base 重建后跑。
- `.github/workflows/build-arm64-docker.yml`：push 触发。`FROM Docker Hub base`（GHA 国外拉 Docker Hub 快），叠代码层，双推 ACR + Docker Hub；push 成功后自动创建/更新 GitHub Release。

### 13.6 实测耗时对比（为什么这么设计）

| 路径 | 耗时 | 结论 |
|---|---|---|
| 服务器直推 Docker Hub | ∞（RST，0 层成功） | 死结，放弃 |
| 服务器直连推 ACR | 一次过 | ✅ base 落地 ACR |
| GHA 国外拉**上海** base 构建 | 38min | 太慢，弃 |
| crane 同步 ACR→Docker Hub | 12min | ✅ 一次性/低频 |
| GHA 拉 **Docker Hub** base 双推 | **4.5min** | ✅ 日常出包 |

### 13.7 操作手册
**A. 服务器推 base 到 ACR**（base 重建后）
```bash
# 1. 让 docker 直连 ACR：编辑 proxy drop-in，NO_PROXY 末尾加 .aliyuncs.com
#    Environment="NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,.aliyuncs.com"
systemctl daemon-reload && systemctl restart docker
docker info | grep -i "no proxy"   # 确认含 .aliyuncs.com
# 2. 登录 + 打 tag + 推
docker login crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com
docker tag <本地 base> crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-cu128-a100-base
while ! docker push crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-cu128-a100-base; do sleep 5; done
```
**B. 同步到 Docker Hub**：GitHub Actions → 手动触发 `Sync base image (ACR -> Docker Hub)`。
**C. 日常出包**：push 代码即可，`Build ARM64 Docker Image` 自动跑，4.5 分钟双推两地。
**D. GHA secrets**：`ACR_USERNAME`/`ACR_PASSWORD`（推 ACR）、`DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN`（账号 arronlee，同步/双推 Docker Hub）。

### 13.8 已知坑
- **顺序依赖**：base 必须先跑 Sync 推到 Docker Hub，build 才能 `FROM` 到；base 重建后记得重跑 Sync，否则 build 失败。
- **NO_PROXY 不配 = 白换**：不配的话 docker 仍把上海流量塞进 xray。
- **Docker Hub 仓库要设 public**：对外分发 + 拉取免登录。
- **上游 x86 是另一套**：`dockerfiles/Dockerfile` 单体、`FROM pytorch/pytorch`，一个 Dockerfile 里源码编译全部扩展，从国外推 `lightx2v/lightx2v`（Docker Hub）+ `registry.cn-hangzhou.aliyuncs.com/yongyang/lightx2v`（阿里云杭州）。x86 能单体是因为有预编译 wheel + 国外网络好；ARM64 没这条件才拆 base/app + 走国内 ACR。

---

## 14. 通用镜像可用性验证（LTX2.3 + Wan2.2 Lightning 实测出片）

> 2026-05-27 用发布到 ACR/Docker Hub 的新镜像 `arm64-a100-latest`（已 bake 修复、**无需任何 bind mount 补丁**）实测两个模型端到端出片。**结论：镜像可用，LTX2.3 和 Wan2.2 Lightning 都能出干净片**；Wan2.2 必须走预量化 int8 配置（在线 LoRA 路径会雪花，见 §14.7）。

### 14.1 硬件与镜像

- 服务器 `edt-vpn`（111.172.214.29，root），鲲鹏 ARM **aarch64**，4×A100 **40GB** PCIe（sm_80），CUDA 12.8，宿主内存 251GB。
- 镜像：`crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest`（与 Docker Hub `arronlee/lightx2v:arm64-a100-latest` 同 digest）。国内 A100 拉 ACR 走国内带宽、快。
- 镜像 smoke 验证（`docker run --rm --gpus all <img> python -c "import torch,flash_attn,lightx2v"`）：`torch 2.11.0+cu128 / cuda True / flash_attn 2.7.4.post1 / sageattention import OK / lightx2v 在 /opt/LightX2V`。
- 当前状态：`lightx2v-wan-int8` 容器在 GPU0 常驻（int8 Wan server）。

### 14.2 权重路径（服务器 `/data/models`）

| 模型/组件 | 路径 |
|---|---|
| LTX2.3 DiT 蒸馏 | `Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors`（46GB bf16） |
| LTX2.3 上采样器 | `Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| LTX2.3 gemma 文本编码器 | `google/gemma-3-12b-it-qat-q4_0-unquantized`（bf16 ~24GB） |
| Wan2.2 base MoE | `Wan-AI/Wan2.2-T2V-A14B`（high_noise + low_noise 各 6 shard，含 T5 `models_t5_umt5-xxl-enc-bf16.pth`） |
| Wan2.2 Lightning LoRA（在线，**雪花，勿用**） | `lightx2v/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/`（high/low 各 1.2GB） |
| Wan2.2 预量化 int8（**干净，要用**） | `wan22_t2v_int8`（1217 LoRA 合）、`wan22_t2v_int8_seko`（Seko 合）；各含 `high_noise_model` / `low_noise_model`（`distill_model_partN.safetensors`+index） |

### 14.3 配置文件（`/data/lightx2v_configs`）

| 模型 | 配置 | 说明 |
|---|---|---|
| LTX2.3 | `ltx2_3_distill_v11_hq.json` | 单卡基准，8步/121帧/768×1280/flash_attn2/upsampler/tiling ✅ |
| Wan2.2 int8 单卡 | `test_480p_int8_prequant.json` | 预量化 int8，480p/49帧，单卡 ✅ |
| Wan2.2 int8 720p 多卡 | `cmp_720p_seko.json`(4卡 Seko) / `cmp_720p_1217.json`(4卡) / `test_720p_int8_ulysses2.json`(2卡) | 预量化 int8 + ulysses |
| ⚠️ **勿用** | `wan22_t2v_lightning_single.json` / `_ulysses2.json` | `lora_dynamic_apply:true` 在线 LoRA → **雪花废片** |

### 14.4 启动 server（新镜像，无 bind mount 补丁）

LTX2.3（脚本 `/data/start_ltx_new.sh`）：
```bash
IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
docker rm -f lightx2v-ltx-new 2>/dev/null
docker run -d --name lightx2v-ltx-new --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES=0 -e LTX_GEMMA_ON_CPU=1 -e LTX_GEMMA_LAYERWISE_GPU=1 \
  -e LTX_VAE_SPATIAL_TILE=256 -e LTX_VAE_SPATIAL_OVERLAP=32 -e LTX_VAE_TEMPORAL_TILE=16 -e LTX_VAE_TEMPORAL_OVERLAP=8 \
  "$IMG" python -m lightx2v.server --model_cls ltx2 --task t2av \
  --model_path /data/models/Lightricks/LTX-2.3 \
  --config_json /data/lightx2v_configs/ltx2_3_distill_v11_hq.json --host 0.0.0.0 --port 8000
```

Wan2.2 Lightning（预量化 int8，脚本 `/data/start_wan_int8.sh`）：
```bash
docker rm -f lightx2v-wan-int8 2>/dev/null
docker run -d --name lightx2v-wan-int8 --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=0 \
  "$IMG" python -m lightx2v.server --model_cls wan2.2_moe --task t2v \
  --model_path /data/models/Wan-AI/Wan2.2-T2V-A14B \
  --config_json /data/lightx2v_configs/test_480p_int8_prequant.json --host 0.0.0.0 --port 8000
```
就绪判据：`curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health` 返回 200（LTX 加载 ~5min，Wan int8 ~3min）。

### 14.5 提交任务 + 验证产物

- **LTX**：`bash /data/ltx_one.sh 121`（提交 night_market、轮询 `status` 字段、每 5s 记 GPU0 显存峰值；产物 `/data/outputs/ltx_test_probe.mp4`）。
- **Wan**：`POST /v1/tasks/video/`，body `{prompt, negative_prompt, save_result_path, seed}`，轮询 `GET /v1/tasks/{id}/status`（脚本 `/data/wan_verify.sh`）。
- **验证产物**：
  ```bash
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,avg_frame_rate "$F"   # 规格
  ffmpeg -v info -i "$F" -vf blackdetect=d=0.1:pix_th=0.10 -an -f null - 2>&1 | grep blackdetect          # 黑屏
  ffmpeg -y -i "$F" -vf "select=eq(n\,20)" -vframes 1 frame.png                                            # 抽帧肉眼看雪花
  ```
  ⚠️ **雪花用 blackdetect 检测不出**（不是黑屏），必须抽帧肉眼看；旁证：雪花帧（高熵噪声）压缩后单帧 png 体积异常大（雪花 720p 单帧 2.4MB vs 干净 480p 645KB）。

### 14.6 实测结果

| 模型 | 配置 | 规格 | 耗时 | 显存峰值 | 画质 |
|---|---|---|---|---|---|
| LTX2.3 蒸馏1.1 | `ltx2_3_distill_v11_hq` 单卡 | 1280×768 / 121帧 / 24fps | 86s | 18.9GB | 干净 ✅ |
| Wan2.2 Lightning | `test_480p_int8_prequant` int8 单卡 | 832×480 / 49帧 | 56s | 34.8GB | 干净 ✅ |
| Wan2.2 Lightning | `cmp_720p_seko` int8 **4卡 ulysses** | 1280×720 / 49帧 | 51s | 34.6GB/卡 | 干净 ✅ |
| Wan2.2（对照·勿用） | `wan22_t2v_lightning_single` 在线LoRA | 1280×720 / 121帧 | 284s | 33GB | ❌ **雪花废片** |
| LTX2.3（4卡对照·勿用） | `ltx2_3_distill_v11_hq_ul4` **4卡 ulysses** | 1280×768 / 121帧 | >400s 未完 | GPU 全 0% | ❌ gemma 卡 CPU |

### 14.7 踩过的坑

1. **Wan 雪花**：`lora_dynamic_apply:true` 在线 LoRA 在 Wan2.2 T2V 上有 bug（2026-05-24 已定位，见 `new-api/docs/local-video-poc-checklist.md` §2.10.X），与镜像/sageattention/boundary 无关。解法：预量化 int8 配置。
2. **端口 8000 already allocated**：旧容器没停干净就 run 新的。先 `docker rm -f` 所有相关容器（`lightx2v-ltx-* / lightx2v-wan-* / lightx2v-server`）、确认 `docker ps | grep 8000` 为空再启。
3. **`sageattention not found` 日志**：只是 LightX2V 在探测高级变体（sageattn3 / flashattention4 / sageattn3_sparse），**基础 `sage_attn2` 实际可用**（`import sageattention` OK）。
4. **`utils_patched.py`**：只是给 ffmpeg 加调色滤镜（`saturation=0.78,gamma=1.05,colorbalance`）的补丁，**非功能必需**，不挂不影响能否出片。
5. **SSH/出墙网络不稳**：操作服务器用「短命令 + 重试循环」；长操作（push / 测速 / 生成）用 `nohup` 后台写日志 + 短 ssh `tail` 读日志，避免长连接半路断。
6. **LTX 不再需要 patches**：gemma 逐层 GPU、VAE inference_mode 等修复已在 `origin/main`（`6651fcc0`）、即在镜像里，启动只需环境变量，不用 `/data/patches/*.py` bind mount。

### 14.8 目前还存在的问题

1. **Wan2.2 在线 LoRA 雪花 bug 未修**（上游代码问题）。绕过：用预量化 int8。**若要上任意新 LoRA**，需先用 `tools/convert/converter.py --device cpu` 离线把 LoRA 合进 base 并 int8 量化，再用 `*_quantized_ckpt` 配置，不能走 `lora_dynamic_apply`。
2. **Wan2.2 单卡 720p 未实测**（但 **4卡 ulysses 720p 已验证 51s 干净**，见 §14.6/§14.9）：int8 单卡只验证了 480p（峰值 34.8GB/40GB）；720p/121帧激活更大，单卡可能 OOM，故 720p 走多卡 ulysses 预量化配置（`cmp_720p_*` / `test_720p_int8_ulysses2`），多卡需 `--shm-size=32g`。
3. **调色补丁 `utils_patched.py` 未进仓库**：线上若要那套饱和度/gamma 调色，需把改动提交进 `lightx2v/utils/utils.py` 后重出镜像。
4. **可选后端未装**：sageattn3 / flashattention4 / sageattn3_sparse / decord 未装；LTX/Wan 基础推理不需要，animate 模型需 decord。
5. **GHA 国外拉上海 base 慢**（38min）：已用「base 同步到 Docker Hub + build 从 Docker Hub 拉」降到 4.5min，见 §13。

### 14.9 并行与量化能力结论（2026-05-27 4卡实测）

| 模型 | 单卡 | 4卡 ulysses | int8 量化 | 最终建议 |
|---|---|---|---|---|
| **LTX2.3 蒸馏1.1** | 1280×768/121帧 **86s ✅** | ❌ >400s 未完（gemma 卡 CPU、GPU 全 0%） | ❌ 不支持 | **固定单卡 bf16** |
| **Wan2.2 Lightning** | int8 480p 56s ✅ | int8 720p 51s ✅ | ✅ 必须（避雪花） | 单卡或多卡，**必须预量化 int8** |

- **LTX 4卡 ulysses 不实用**：gemma 每 rank 冗余、在 ARM CPU 上串行慢算，4 张卡 GPU 全程 0% 利用率（与 §6 #8 同现象，即使逐层 GPU 也没改善）。**LTX 测试/部署都别加 4卡**，固定单卡。
- **LTX 不支持 int8 量化**：`tools/convert/converter.py` 不支持 ltx2（只支持 wan_dit/qwen_image_dit 等）→ 无离线预量化路径；在线 `weight_auto_quant` 与 block offload 不兼容（`KeyError weight_scale`）；官方只发 bf16。故 **LTX 只能 bf16**。
- **Wan 4卡 ulysses 有效**：无 gemma 瓶颈，ulysses 切序列/激活分摊 + 加速，能上 720p。但 **ulysses 不切权重**（每卡全量 int8 ~28GB），省权重只能靠量化（已 int8）或 offload，**加卡不减每卡权重显存**。
- 两模型量化能力正好相反：**LTX 固定 bf16、Wan 必须 int8**。

---

## 15. 新镜像验收基础用例（出 Docker 后回归测试）

> **每次出新 ARM64 A100 镜像后，跑以下 3 个基础用例确认镜像可用。** 覆盖两个模型 × 单卡/多卡 × bf16/int8。镜像统一用 `IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest`（或具体版本 tag）。权重/配置全在服务器 `edt-vpn`，路径见 §14.2 / §14.3。

### 15.0 前置（每次必做）
```bash
# 1. 拉新镜像（国内拉 ACR 快；base 层已在本地只拉 app 层）
docker pull crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
# 2. smoke：确认 torch/CUDA/flash_attn/lightx2v 都在
docker run --rm --gpus all "$IMG" python -c "import torch,flash_attn,lightx2v;print('cuda',torch.cuda.is_available(),'fa',flash_attn.__version__)"
#   期望：cuda True / fa 2.7.4.post1
# 3. 起容器前先清旧容器（避免 8000 端口冲突）
docker rm -f lightx2v-ltx-new lightx2v-wan-int8 lightx2v-wan-ul4 lightx2v-ltx-server lightx2v-server 2>/dev/null
```

### 用例总览

| # | 模型 | 卡 | 配置 | 关键权重 | 预期（出片干净） |
|---|---|---|---|---|---|
| 1 | LTX2.3 蒸馏1.1 | 单卡 | `ltx2_3_distill_v11_hq.json` | `Lightricks/LTX-2.3` + `google/gemma-3-12b-...` | 1280×768 / 121帧 / 24fps / ~86s / ~18.9GB |
| 2 | Wan2.2 Lightning | 单卡 | `test_480p_int8_prequant.json` | `wan22_t2v_int8` + `Wan-AI/Wan2.2-T2V-A14B`(T5/VAE) | 832×480 / 49帧 / ~56s / ~34.8GB |
| 3 | Wan2.2 Lightning | 4卡 ulysses | `cmp_720p_seko.json` | `wan22_t2v_int8_seko` + `Wan-AI/Wan2.2-T2V-A14B` | 1280×720 / 49帧 / ~51s / ~34.6GB/卡 |

> ⚠️ **红线**：Wan 一律走预量化 int8（勿用 `wan22_t2v_lightning_single.json` 等 `lora_dynamic_apply:true` 配置 → 雪花）；LTX 一律单卡（勿加 4卡 ulysses → gemma 卡 CPU）。

### 15.1 用例 1 — LTX2.3 单卡（bf16）
```bash
docker run -d --name lightx2v-ltx-new --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES=0 -e LTX_GEMMA_ON_CPU=1 -e LTX_GEMMA_LAYERWISE_GPU=1 \
  -e LTX_VAE_SPATIAL_TILE=256 -e LTX_VAE_SPATIAL_OVERLAP=32 -e LTX_VAE_TEMPORAL_TILE=16 -e LTX_VAE_TEMPORAL_OVERLAP=8 \
  "$IMG" python -m lightx2v.server --model_cls ltx2 --task t2av \
  --model_path /data/models/Lightricks/LTX-2.3 \
  --config_json /data/lightx2v_configs/ltx2_3_distill_v11_hq.json --host 0.0.0.0 --port 8000
# 等 health 200（~5min 加载）后提交：
bash /data/ltx_one.sh 121          # 产物 /data/outputs/ltx_test_probe.mp4
```
现成脚本：`/data/start_ltx_new.sh`。通过标准：status=completed、`elapsed≈86s`、`peak≈18.9GB`、`ffprobe` 为 1280×768/121帧/24fps、`blackdetect` 无、抽帧肉眼无异常。

### 15.2 用例 2 — Wan2.2 Lightning 单卡（int8 预量化）
```bash
docker run -d --name lightx2v-wan-int8 --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=0 \
  "$IMG" python -m lightx2v.server --model_cls wan2.2_moe --task t2v \
  --model_path /data/models/Wan-AI/Wan2.2-T2V-A14B \
  --config_json /data/lightx2v_configs/test_480p_int8_prequant.json --host 0.0.0.0 --port 8000
# 等 health 200（~3min）后提交：
bash /data/wan_verify.sh           # 产物 /data/outputs/wan_verify.mp4（POST /v1/tasks/video/ + 轮询）
```
现成脚本：`/data/start_wan_int8.sh`。通过标准：completed、`elapsed≈56s`、832×480/49帧、抽帧无雪花。

### 15.3 用例 3 — Wan2.2 Lightning 4卡 ulysses（int8 预量化，720p）
```bash
docker run -d --name lightx2v-wan-ul4 --gpus all --shm-size=32g -p 8000:8000 -p 8001:8001 -v /data:/data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$IMG" torchrun --nproc_per_node=4 --master_port=29524 -m lightx2v.server \
  --model_cls wan2.2_moe --task t2v --model_path /data/models/Wan-AI/Wan2.2-T2V-A14B \
  --config_json /data/lightx2v_configs/cmp_720p_seko.json --host 0.0.0.0 --port 8000
# 等 health 200（4 rank 加载 ~5-8min）后提交（同用例2，改 save_result_path）：
```
现成脚本：`/data/start_wan_ul4.sh`。通过标准：completed、`elapsed≈51s`、1280×720/49帧、每卡显存 ~28GB（ulysses 不切权重，正常）、抽帧无雪花。多卡必须 `--shm-size=32g`。

### 15.4 通用产物验证
```bash
F=/data/outputs/<产物>.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,avg_frame_rate -of default=nw=1 "$F"   # 规格
ffmpeg -v info -i "$F" -vf blackdetect=d=0.1:pix_th=0.10 -an -f null - 2>&1 | grep blackdetect || echo 无黑屏        # 黑屏
ffmpeg -y -i "$F" -vf "select=eq(n\,20)" -vframes 1 frame.png                                                       # 抽帧肉眼看雪花
```
> 雪花 `blackdetect` 检测不出，**必须抽帧肉眼看**；旁证：雪花帧（高熵噪声）单帧 png 体积异常大（720p 雪花 2.4MB vs 干净 1.5MB）。

### 15.5 验收红线（任一不满足即镜像/配置有问题）
1. 三个用例都 `status=completed`、规格与上表一致、**抽帧无雪花**。
2. 耗时不显著劣化（LTX ~86s / Wan 单卡 ~56s / Wan 4卡 ~51s，±30% 内）。
3. 不出现 `KeyError weight_scale`（int8 量化路径正常）、不出现 OOM。
4. **不踩红线**：Wan 不用 `lora_dynamic_apply` 配置、LTX 不加 4卡。

---

## 16. 标准内容测试集 + 压测方案

> 在 §15 基础验收通过后，用本节做**内容质量测试**（8 个 prompt × 2 模型）和**能力压测**（找 OOM 边界）。Prompt 按官方推荐结构化：主体+外观+动作+环境+光影+镜头运动+风格。

### 16.1 测试矩阵
8 个 prompt（4 真人短剧 + 4 动漫，4 风格各异）**在两个模型上都跑**：
- **Wan2.2 Lightning**：4 卡 ulysses + int8 预量化（`cmp_720p_seko.json`，720p），`task t2v`。
- **LTX2.3 蒸馏1.1**：单卡 bf16（`ltx2_3_distill_v11_hq.json`，1280×768），`task t2av`。

共 16 次生成。每次验证：`status=completed` + `ffprobe` 规格 + **抽帧 Read 比对画质**（雪花/崩坏/语义偏离，见 §15.4）。

### 16.2 八个 Prompt（中英文最终版）

**真人短剧（亚洲人 + 简单剧情，中文优先）**

1. `cafe_reunion` 雨天咖啡馆重逢
   - 中：暖色调电影感画面。一位二十多岁的亚洲年轻女性独自坐在咖啡馆靠窗的位置，手捧一杯热咖啡，窗外下着小雨、玻璃上挂满水珠。她抬头透过窗户看到一位老朋友推门进来，脸上露出惊喜的微笑，放下杯子起身挥手。浅景深，背景虚化的暖黄灯光，镜头缓缓推近。
   - EN: Warm cinematic tone. A young Asian woman in her twenties sits alone by a café window holding a hot coffee, light rain streaking the glass. She looks up, sees an old friend pushing the door open, her face lighting up with a surprised smile as she sets down the cup and stands to wave. Shallow depth of field, blurred warm golden bokeh, slow dolly-in.

2. `night_market_wok` 夜市小吃摊
   - 中：霓虹夜市，手持跟拍。一位亚洲中年男厨师在小吃摊前熟练颠勺翻炒，锅中火光腾起、热气与油烟升腾，旁边几位顾客排队等候、边看边聊。背景是模糊的彩色霓虹招牌。烟火气十足，暖橙色调。
   - EN: Neon night market, handheld tracking shot. A middle-aged Asian male cook skillfully tosses food in a wok at a street stall, flames leaping and steam rising. A few customers wait in line, chatting and watching. Blurred colorful neon signs behind. Lively street-food atmosphere, warm orange tone.

3. `subway_commute` 清晨地铁通勤
   - 中：冷色调，固定机位。早高峰的地铁车厢里，一位穿西装的亚洲上班族站在扶手旁低头刷手机，周围乘客拥挤。列车缓缓进站、车门打开，人流涌动进出。车厢内日光灯偏冷白，窗外站台灯光掠过。写实纪实风格。
   - EN: Cool tone, static shot. Inside a crowded rush-hour subway car, an Asian office worker in a suit stands by the handrail scrolling his phone, surrounded by commuters. The train slows into the station, doors slide open, crowds flow in and out. Cool white fluorescent light, platform lights sweeping past windows. Realistic documentary style.

4. `park_taichi` 公园晨练太极
   - 中：金色晨光，慢镜头环绕。一位亚洲老人在公园的树下缓慢打太极拳，动作舒展平和，几片落叶从空中飘过。清晨阳光透过树叶洒下斑驳光影，薄雾笼罩。镜头围绕老人缓缓旋转，宁静祥和。
   - EN: Golden morning light, slow orbiting shot. An elderly Asian man practices Tai Chi slowly under a tree in a park, movements graceful and calm, a few leaves drifting by. Morning sunlight filters through leaves casting dappled light, thin mist. Camera slowly orbits him. Serene and peaceful.

**动漫（4 种区分明显的风格）**

5. `anime_sakura` 日系赛璐璐
   - 中：日本动画赛璐璐风格，鲜艳明亮。樱花树下，一位身穿校服的少女抬头仰望，微风吹起她的长发和飘落的粉色花瓣。蓝天白云，阳光明媚。清晰的线条和色块，经典 TV 动画质感。
   - EN: Japanese anime cel-shaded style, vibrant and bright. Under a cherry blossom tree, a schoolgirl looks up, a gentle breeze lifting her long hair and the falling pink petals. Blue sky with white clouds, bright sunshine. Clean lines and flat color shading, classic TV-anime look.

6. `ghibli_field` 吉卜力水彩
   - 中：吉卜力工作室水彩手绘风格，柔和温暖。乡间绿色的田野上，一个少年张开双臂奔跑，远处是翻涌的云海和连绵的山丘。柔和的自然光，细腻的水彩笔触，治愈系氛围。
   - EN: Studio Ghibli hand-painted watercolor style, soft and warm. A boy runs with arms outstretched across green countryside fields, rolling hills and a sea of clouds in the distance. Soft natural light, delicate watercolor brushwork, healing nostalgic atmosphere.

7. `cyberpunk_girl` 赛博朋克霓虹
   - 中：赛博朋克动画风格，高对比霓虹。雨夜的未来都市街道，霓虹灯牌倒映在湿漉漉的地面上，一位机械义体少女缓缓回眸，发丝间闪烁蓝紫色光芒。强烈的青色与品红撞色，电影级氛围。
   - EN: Cyberpunk anime style, high-contrast neon. A rainy-night futuristic city street, neon signs reflecting on wet ground. A cyborg girl slowly turns to look back, blue-purple glow shimmering through her hair. Strong cyan-magenta color clash, cinematic mood.

8. `ink_crane` 国风水墨
   - 中：中国水墨动画风格，大量留白。青绿山水之间，一只白鹤展翅缓缓掠过，云雾在山峰间缭绕流动。淡雅的墨色晕染，写意笔触，古典诗意，宁静悠远。
   - EN: Chinese ink-wash animation style, generous negative space. Among blue-green mountains and rivers, a white crane glides slowly with spread wings, mist swirling between peaks. Elegant ink washes, freehand brushstrokes, classical poetic, tranquil and distant.

### 16.3 执行
1. 起两个 server（不同时，单机 8000 端口复用）：Wan 用 §15.3 启动、LTX 用 §15.1 启动。
2. 每个 prompt 提交，`save_result_path` 命名 `/data/outputs/<model>_<slug>.mp4`（如 `wan_cafe_reunion.mp4` / `ltx_cafe_reunion.mp4`）：
   ```bash
   curl -s -X POST http://localhost:8000/v1/tasks/video/ -H 'Content-Type: application/json' \
     -d '{"prompt":"<上面对应 prompt>","negative_prompt":"low quality, blurry, distorted","save_result_path":"/data/outputs/<model>_<slug>.mp4","seed":42}'
   # 轮询 GET /v1/tasks/{id}/status 至 completed
   ```
3. 逐个抽帧 Read 比对画质（雪花/手部崩坏/语义偏离/风格是否到位）。

### 16.4 Wan2.2 4卡 ulysses int8 压测（已完成，2026-05-27）

> **API 字段修正**：覆盖分辨率的正确字段是 `target_shape:[H,W]`（如 `[720,1280]`），**不是** `target_height`/`target_width`（server 会静默忽略这两个字段）。覆盖帧数用 `target_video_length`（有效）。

#### 分辨率上限说明

**Wan2.2 A14B 官方最高支持 720p（1280×720）**，不原生支持 1080p/1440p/4K。传入更高分辨率（通过 `target_shape`）会触发模型内部 tensor shape 不匹配（RoPE/位置编码仅针对 720p token 数设计），task 立即 failed。若需要 1080p 输出，只能用 720p 生成后超分后处理（如 RealESRGAN）。

#### 720p 帧数爬升（4 卡 ulysses + int8 Seko，`cmp_720p_seko.json`）

| 帧数 | 时长(@16fps) | 结果 | 耗时 | 显存峰值/卡 |
|---|---|---|---|---|
| 49 | ~3.1s | ✅ | 45s | 34.2GB |
| 81 | ~5.1s | ✅ | 80s | 35.7GB |
| 121 | ~7.6s | ✅ | 131s | 38.0GB |
| **161** | **~10.1s** | ✅ **上限** | **192s** | **40.3GB** |
| 201 | — | ❌ OOM | — | — |

→ **Wan2.2 4卡 720p 稳定上限：161 帧（≈10s@16fps，192s，40.3GB/卡）**

#### cpu_offload 突破 161 帧尝试（失败）

| 方案 | 结果 | 根因 |
|---|---|---|
| 4 卡 + `cpu_offload:true` | ❌ CPU OOM，OOM killer SIGKILL rank 1 | 4 rank 各持独立 CPU 权重副本，4×28GB=112GB+，超出宿主内存上限 |
| 单卡 + `cpu_offload:true`（无 torchrun） | ❌ 201 帧 task completed 但视频全黑 | int8 预量化 + offload 单卡推理路径 bug（未深究，不作为生产选项） |

**结论：cpu_offload 在当前配置下不可用。161 帧是 Wan2.2 4卡的真实稳定上限。**

#### Wan2.2 最终能力边界

| 指标 | 值 |
|---|---|
| 最高分辨率 | **720p（1280×720）** |
| 720p 最大帧数 | **161 帧 ≈ 10s@16fps** |
| 161 帧耗时（4 卡） | **192s** |
| 161 帧显存峰值/卡 | **40.3GB** |
| 内容测试推荐规格 | 720p / 49 帧（51s，质量与速度均衡） |

### 16.5 LTX2.3 单卡 bf16 压测（已完成，2026-05-27）

启动脚本 `/data/start_ltx_new.sh`，配置 `ltx2_3_distill_v11_hq.json`（1280×768 / 8步 / flash_attn2 / block offload）。

> LTX2.3 **不支持 int8 量化**（converter 不支持 ltx2，在线 auto-quant 与 block offload 不兼容），只能 bf16。**不加 4 卡 ulysses**（gemma 每 rank 冗余，4 卡 GPU 全 0%，极慢）。

#### 帧数维度（输出 1280×768，`target_video_length` 爬升）

| 帧数 | 时长@24fps | 结果 | 耗时 | 显存峰值 |
|---|---|---|---|---|
| 121 | 5s | ✅ | 86s | 18.9GB |
| 241 | 10s | ✅ | 100s | 23.2GB |
| 481 | 20s | ✅ | 196s | 31.5GB |
| 641 | 27s | ✅ | 281s | 36.9GB |
| 721 | 30s | ✅ | 322s | 40.1GB |
| **801** | **33s** | ✅ **上限** | 372s | ~40GB |
| 961 | 40s | ❌ OOM | — | — |

→ **LTX 单卡 768×1280 帧数上限：801 帧 ≈ 33s@24fps**（961 OOM）。显存随帧数近似线性（每 +80 帧约 +2.7GB）。

#### 分辨率维度（固定 49 帧）

> ⚠️ **`target_shape:[H,W]` 传的是 upsampler 前的 base 分辨率，最终输出 ×2**（ffprobe 实测）。即要 4K 输出传 `[1088,1920]`，要 5K 传 `[1440,2560]`。分辨率须为 **32 的倍数**（LTX 官方硬要求，所以 1080→1088、4K 的 2160→2176）。

| target_shape (base) | 实际输出 | 结果 | 耗时 | 显存峰值 |
|---|---|---|---|---|
| [768,1280] | 2560×1536（2.5K） | ✅ | 100s | 22.3GB |
| [1088,1920] | **3840×2176（4K）** | ✅ | 181s | 31.2GB |
| [1440,2560] | **5120×2880（5K）** | ✅ | 393s | 39.5GB |
| [2176,3840] | 7680×4352（8K） | ❌ OOM | — | — |

→ **LTX 单卡 49 帧分辨率上限：5K（5120×2880，39.5GB）**，8K OOM。

#### 资源 & 说明
- GPU 顶满 40GB；**系统内存峰值仅 46.7GB**（远未到 251GB 瓶颈，内存不是约束）。
- 分辨率与帧数此消彼长：高分辨率时帧数要降，长视频时分辨率要降。
- 重启后首个任务有 ~300s warmup（torch 编译），之后稳定。
- **OOM 后显存不自动释放，必须重启 server 才能继续测**（否则下一档假性 OOM）。

---

## 17. 两模型能力对比与选型结论

### 17.1 能力对比

| 维度 | Wan2.2 Lightning（4卡 int8） | LTX2.3 蒸馏1.1（单卡 bf16） |
|---|---|---|
| 最高分辨率 | 720p（1280×720） | **5K（5120×2880）** |
| 最大时长 | 161 帧 / ~10s | **801 帧 / ~33s** |
| 基线耗时 | 720p/49帧 51s（4卡） | 768×1280/121帧 86s（单卡） |
| 量化 | **必须 int8 预量化**（否则雪花） | **只能 bf16**（不支持 int8） |
| 多卡 | 4卡 ulysses 有效（切激活） | 单卡（4卡 gemma 卡 CPU，无效） |
| cpu_offload | 不可用（4卡内存 OOM / 单卡黑屏） | 显存富余，无需 |
| 内容·真人短剧 | **优秀**，运动自然 | 尚可，偶尔不清晰 |
| 内容·动漫 | **优秀**，4 风格到位 | **差**，接近静止，不适合 |

### 17.2 选型建议
- **真人/写实 + 动漫**：优先 **Wan2.2 Lightning**（内容质量高、运动自然），规格限 720p / 10s 内。
- **需要高分辨率（4K/5K）或长时长（>10s）**：用 **LTX2.3**，但**仅限真人/写实**（动漫几乎静止，不要用 LTX）。
- **4K 交付**：也可 720p 生成 + 外部超分（RealESRGAN），通常比直接超高分辨率生成更稳更省。

### 17.3 整体结论
新 ARM64 A100 镜像（`arm64-a100-latest`）经完整验证：**Wan2.2 Lightning 与 LTX2.3 都能在单机 4×A100 40G 上稳定出片**，能力边界、内容质量、量化/并行约束均已摸清记录。镜像可用于生产部署。

---

## 18. LTX2.3 多卡/多实例并行探索（结论：单卡最优）

> 动机：LTX 单卡 121 帧 86s，想用满单机 4×A100 提升吞吐/速度。试了 4 种方案，**全部失败**，根因是 gemma 文本编码器的并发困境。

### 18.1 各方案实测

| 方案 | 配置 | 结果 | 实测数据 |
|---|---|---|---|
| 4 卡 ulysses | `seq_p_size:4`，gemma 逐层 GPU | ❌ 卡死 | 4 卡 GPU 全 0% 利用率，>400s 不出片，gemma 每 rank 冗余争 CPU |
| 3 卡 ulysses | `seq_p_size:3` | ❌ 同上 | gemma 冗余瓶颈与卡数无关 |
| 3 实例数据并行（gemma 留 CPU） | 3 个 docker 各绑 1 卡，`LTX_GEMMA_ON_CPU=1` | ❌ 严重劣化 | 单任务 **600–710s**（单实例 86s 的 7–8 倍）；8 任务 round-robin，906s 只出 6 个 |
| 多实例 + gemma 整搬 GPU | 关 `LTX_GEMMA_ON_CPU` | ❌ OOM | 2 实例 49 帧都 OOM（GPU 顶满 40431MiB，247s failed） |
| 3 实例 + CPU 绑核 | 各绑 1 个 NUMA node 32 核（`--cpuset-cpus`）+ 1 GPU，gemma 留 CPU | ❌ 仍劣化 | 3 并发各 539-559s（不绑核 600-710s，仅快 ~15%，仍是单卡 86s 的 6 倍）；证实瓶颈是**内存带宽**而非核竞争 |

### 18.2 根因：gemma 困境
LTX 的 gemma-3-12b 文本编码器（bf16 ~24GB）是并发瓶颈，两条路都堵死：
- **留 CPU**（`LTX_GEMMA_ON_CPU=1` + 逐层 GPU）：单卡不 OOM（常驻仅 ~2GB），但多实例/多卡 encode 时**争抢 ARM CPU 内存带宽** → 严重拖慢。CPU 绑核（每实例独占一个 NUMA node 32 核）实测只快 ~15%（650s→550s），证实主因是**内存带宽竞争**而非核竞争——绑核也救不了。
- **整搬 GPU**（关 `LTX_GEMMA_ON_CPU`）：不争 CPU，但单卡 24GB gemma + DiT + 上采样器 + 激活 **> 40GB → OOM**。

Wan2.2 没有这个问题（用 T5 + int8 量化，4 卡 ulysses 有效），所以 Wan 能多卡、LTX 不能。

### 18.3 结论
**LTX2.3 最优部署 = 单实例单卡 + gemma 逐层 GPU（86s/121帧）。** 单机内多卡 ulysses、多实例数据并行、gemma 上 GPU 全部走不通。要提升总吞吐只能**横向扩多机**（每机 1 卡 1 实例 + 上层负载均衡），单机多卡对 LTX 无效。

---

## 19. 内容测试详细数据（8 prompt × 2 模型，镜像 `0302-23264f35`）

8 个 prompt：`cafe_reunion`/`night_market_wok`/`subway_commute`/`park_taichi`（真人短剧）+ `anime_sakura`/`ghibli_field`/`cyberpunk_girl`/`ink_crane`（动漫 4 风格），完整中英文见 §16.2。

### 19.1 Wan2.2 4卡 int8（`cmp_720p_seko.json`，720p/49帧）
| prompt | 耗时 | prompt | 耗时 |
|---|---|---|---|
| cafe_reunion | 55s | anime_sakura | 45s |
| night_market_wok | 50s | ghibli_field | 50s |
| subway_commute | 45s | cyberpunk_girl | 50s |
| park_taichi | 50s | ink_crane | 45s |

画质（肉眼）：**8 个全部优秀** —— 真人写实、运动自然，动漫 4 风格区分到位。

### 19.2 LTX2.3 单卡 bf16（`ltx2_3_distill_v11_hq.json`，1280×768/121帧）
| prompt | 耗时 | prompt | 耗时 |
|---|---|---|---|
| cafe_reunion | 80s(含warmup) | anime_sakura | 70s |
| night_market_wok | 70s | ghibli_field | 65s |
| subway_commute | 70s | cyberpunk_girl | 65s |
| park_taichi | 70s | ink_crane | 65s |

画质（肉眼）：**真人短剧尚可**（偶尔不够清晰）；**动漫差**（运动微弱、接近一张图播 3 秒，不适合 LTX）。

> 注：每个 server 进程首次推理有 ~80–305s 的 torch 编译 warmup，之后稳定。两次（旧/新镜像）内容测试画质一致。
