# MiniMax-H3 接入进展与交接文档

> 日期：2026-08-04
> 状态：**单卡 480p 已验收通过并可用；4 卡 768p 卡在「缺一份命名匹配的量化权重」，其余全部打通**
> 接手请先读第 1 节（结论）和第 6 节（下一步选项），再按第 3-5 节复现环境。

---

## 0. 模型背景（决定期望值，先看）

MiniMax-H3 是 2026-08-03 开源的**全模态生成系统**：文/图/视频/音频混合上下文 → **视频 + 原生立体声音频联合生成**。

- DiT 是 **33B dense 单流 Transformer**（`num_layers=50`, `hidden_size=5376`, `num_attention_heads=56`），
  其中约 13B 在 AdaLN 分支（官方称推理时可预计算缓存、不必加载，但目前没有开关能跳过加载）
- 文本/视觉编码器是 **Qwen3-VL-32B**，只取第 50 层 hidden state（bf16 约 51.5G）
- 视觉 VAE 是 f16t4d24（空间 16×、时间 4× 压缩），再经 1×2×2 patch → **等效空间下采样 32×**
- 输出固定 24fps、32kHz 立体声，时长 4-15s

**两条必须知道的边界**（不是部署问题，是开源范围问题）：

1. **本地只能出 768p**。2K 靠 **H3-Regenerate-2K，未开源**，只有官方 API。
2. **输入质量全靠 H3-Context-IR，也未开源**。官方原话是它「对最终输出质量至关重要」，
   并提供 `docs/VIDEO_PROMPT_WRITING_GUIDE_*.md` 让社区自建。
   → **本地效果注定低于海螺网页版**，评估时别拿网页版当基准。

仓库里**两套格式并列**，这一点是后面踩坑的根源：

| 路径 | 格式 | 谁吃它 | 分片命名 |
|---|---|---|---|
| `FL2VA/`、`Ref2VA/` | **原始 checkpoint** | SGLang / vLLM-Omni | `model-0000X-of-00013.safetensors` |
| 仓库根目录 | **diffusers 模块化** | diffusers / DiffSynth | `diffusion_pytorch_model-0000X-of-00014.safetensors` |

两者权重 **sha256 完全不同**（重新分片过），不能互相软链复用。

---

## 1. 结论速览

| 路线 | 状态 | 数据 |
|---|---|---|
| **NF4 单卡 480p（DiffSynth）** | ✅ **已验收，可用** | 832×480/124帧/50步 = **265.6s**（5.31 s/step），显存峰值 **34.4G/39.5G**，出片带 32kHz 立体声。用户已确认画质、音画同步、5 秒运动幅度均可用 |
| **4 卡 768p（vllm-omni）** | ⚠️ **打通到最后一关，卡在权重格式** | 并行/offload/加载路径/attention 后端全部验证可用；缺一份**命名匹配 vllm-omni 的量化权重** |
| bf16 4 卡（不量化） | ❌ 不可能 | DiT 61.7G + TE 62G，ulysses 是复制权重不是切，40G 卡装不下 |

**一句话交接**：480p 这条随时可以推进产品化；768p 这条所有工程障碍已清除并留下补丁，唯一缺口是权重。

---

## 2. 机器与环境

### 2.1 计算节点

```
dev-gpustack-a100-0030      ssh -p 43055 root@111.172.214.16
```

- 4 × A100-PCIE-40GB（**sm_80，无 NVLink**），实际可用 39.49 GiB/卡
- 鲲鹏 920 **aarch64**，251G 内存（swap 仅 3G）
- **4 × Mellanox MT4125（ConnectX-6 Dx）100Gb RoCE，全部 Active/LinkUp**
  → 聚合 50GB/s，**比节点内 PCIe(~20GB/s) 还快**，多机方案比通常直觉更可行
- NFS：`/nfs-data/models` 与 `/nfs-models/wuhanjisuan894/models` 是同一份（前者是计算节点上的软链）

### 2.2 管理节点

下载都在管理节点执行（`dev-gpustack-manager`）。0030 上**没装 pip**，跑不了下载脚本。

### 2.3 镜像（关键：不需要重建）

| 镜像 | 用途 | 环境 |
|---|---|---|
| `reputationly/lightx2v:arm64-a100-latest` | DiffSynth 单卡路线 | py3.10 / torch 2.11+cu128 / **自带 flash_attn + sageattention** |
| `reputationly/vllm-omni:arm64-a100-latest` | vllm-omni 多卡路线 | py3.12 / torch 2.11+cu130 / **vllm 0.26.0 / bnb 0.50.0**（2026-08-11 实测复核；此前记的 vllm 0.25.0 / bnb 0.49.2 已过期） |

⚠️ `arm64-a100-latest` 是**浮动 tag**，内容随出包滚动，版本号只在核对当天有效。要复现请钉日期 tag，例如
`arm64-a100-20260809-0612-3f4fe637`（digest `sha256:d71c261f…3146ae`，2026-08-11 核为 vllm 0.26.0 / bnb 0.50.0）。

⚠️ `gpustack:lx2v-dev` 镜像里**没有 torch**，别用。
⚠️ 两个镜像都有 ENTRYPOINT，调试要加 `--entrypoint bash`。

---

## 3. 权重清单（全在 NFS）

| 目录 | 大小 | 内容 | 来源 | 用途 |
|---|---|---|---|---|
| `MiniMax-H3/FL2VA/` | 135G | bf16 原始 checkpoint：transformer 62G + text_encoder 63G + video_vae 9.8G + audio_vae 578M + processor/tokenizer | ModelScope `MiniMax/MiniMax-H3` | vllm-omni 的输入；画质基线 |
| `MiniMax-H3/docs`、`scripts` | <1M | 提示词指南 + 官方复现 curl 脚本 | 同上 | **必读**，见第 0 节 |
| `MiniMax-H3-NF4/` | 33G | DiffSynth 打包的 NF4：fl2va DiT 16.0G + ref2va DiT 16.0G + text-encoder 14.3G + video_vae 1.5G + audio_vae 0.26G | ModelScope `DiffSynth-Studio/MiniMax-H3-NF4` | **单卡路线，已验收** |
| `MiniMax-H3-W4A16/` | 36G | auto-round W4A16，10 分片 | HF `Ar4ikov/MiniMax-H3-transformer-W4A16-RTN` | ❌ **命名对不上，见 5.3** |
| `MiniMax-H3-FL2VA-W4A16/` | 32K | 软链拼装目录（transformer→W4A16，其余→FL2VA） | 本地构造 | 同上，失败 |

绝对路径前缀：`/nfs-data/models/`（0030 上）或 `/nfs-models/wuhanjisuan894/models/`（管理节点上）。

### 3.1 下载脚本

`LightX2V/scripts/download_minimax_h3.sh`（管理节点 `/root/download_minimax_h3.sh`，md5 `5acde67205a6e713418b719c1c80dd1f`）

```bash
tmux new -s dl -d 'MODELS="nf4_fl2va docs" bash /root/download_minimax_h3.sh'   # 32G，最小可跑集
tmux new -s dl -d 'MODELS="fl2va" bash /root/download_minimax_h3.sh'            # 135G bf16
tmux new -s dl -d 'MODELS="w4a16" bash /root/download_minimax_h3.sh'            # 36G
tail -f /nfs-models/wuhanjisuan894/dl_minimax_h3.log
```

标签：`nf4`(48G) / `nf4_fl2va`(32G) / `fl2va`(134G) / `ref2va`(62G,去重后) / `diffusers`(134G,去重后) / `w4a16`(35G) / `docs` / `assets`。

脚本特性：ModelScope 主源→hf-mirror 回退、断点续传、**整目录 sha256 软链去重**（`Ref2VA` 与 `FL2VA` 的
`text_encoder/video_vae/audio_vae/processor/tokenizer` 逐文件同 sha，省 72.4G；根目录的 `text_encoder/tokenizer/processor`
也同 sha，再省 62.1G）、末尾逐文件大小审计。

**下载速度差异极大，别误判为卡住**：
- 魔搭有的仓：**60~200 MB/s**
- **HF-only 的仓（如 w4a16）：单线程 hf-mirror 只有 3~4 MB/s**。实测 4 连接分块并行每条仍有
  4.7-5.0MB/s（合计 ~19MB/s，**提速 4.6×**）→ hf-mirror 限单连接速率、不限总带宽。
  脚本目前**还没实现分块并行**，这是一个明确的待优化点。

---

## 4. ✅ 已验收路线：NF4 单卡（DiffSynth）

### 4.1 环境（容器 `h3smoke`，已在 0030 上常驻）

```bash
docker run -d --name h3smoke --gpus all --shm-size 32g --ipc=host \
  -v /nfs-data:/nfs-data -v /nfs-models:/nfs-models -v /root/h3:/work \
  --entrypoint bash reputationly/lightx2v:arm64-a100-latest -lc "sleep infinity"
```

DiffSynth-Studio 2.0.18 已装在 `/usr/local/lib/python3.10/dist-packages`，源码在 `/work/DiffSynth-Studio`。

**装的时候踩的三个坑（重装必看）**：

1. **`sentencepiece` 无 aarch64 轮子**，会把整个 `pip install -e .` 拖崩。
   H3 走 `Qwen2TokenizerFast` 不需要它 → `pip install -e . --no-deps`，再补 `pandas`、`peft`。
2. **GitHub 从 0030 连不通**（`git ls-remote` 直接超时）→ Mac 上 clone 后 `tar czf` + scp 上传。
3. bitsandbytes 0.50.0 **有 aarch64 官方轮子**，`pip install bitsandbytes` 直接装得上，import 正常。

### 4.2 冒烟脚本

`LightX2V/scripts/smoke_minimax_h3_nf4.py`（0030 上 `/root/h3/`，容器内 `/work/`）

```bash
docker exec h3smoke bash -lc "cd /work && python3 smoke_minimax_h3_nf4.py --dry-run"   # 环境自检，不加载权重
docker exec h3smoke bash -lc "cd /work && CUDA_VISIBLE_DEVICES=0 python3 -u smoke_minimax_h3_nf4.py \
  --steps 50 --out /work/h3_t2va_50.mp4"
```

脚本要点：
- **默认 `--offload none` 全常驻**，刻意不照抄官方 README 的 disk offload。
  理由：offload 每步搬 16G 权重，50 步 = 800GB 过 PCIe（无 NVLink 实测 ~20GB/s），**纯传输就 30+ 分钟**。
- 权重全部指向 NFS，并强制 `HF_HUB_OFFLINE=1` 等三个 offline 开关，杜绝隐式联网卡死。
- 对 `ModelConfig` 做**运行时签名自省**（DiffSynth 主线在快速迭代），参数名对不上会打印并剔除，
  不会等加载完 30G 才抛 TypeError。
- 落 metrics JSON：显存峰值 / s per step / token 数。

**真实 API 签名**（照模型卡写会错，已核对源码 `diffsynth/pipelines/minimax_h3_audio_video.py`）：
- FL2VA 用 `keyframes=[PIL]` + `keyframe_indices=[0|-1]`（0=首帧，-1=尾帧），**不是** `input_image`/`end_image`
- `num_frames` 会被向上吸附到最近的 **17n+5**（124 = 17×7+5，正好不用调）
- pipeline 默认 768×1344（≈92k token），**在 40G 上必爆**，必须显式压到 480p

### 4.3 实测结果

| 指标 | 数值 |
|---|---|
| 出片 | 832×480 / 24fps / **5.175s** / H.264 + **AAC 32kHz 立体声** |
| 推理 | **265.6s**（5.31 s/step × 50） |
| 加载 | NFS 冷读 78.7s，热态 33.9s |
| 显存峰值 | 加载 32.1G / **推理 34.4G**（卡上 39.5G） |
| 序列长度 | ~12k token（31 latent 帧 × 15 × 26） |

产物在 0030 的 `/root/h3/h3_t2va_50.mp4`（50 步正片）和 `h3_probe.mp4`（10 步试跑）。

**吞吐形态**：34.4G/39.5G 意味着**一卡一实例**，4 卡 = 4 并发，每条 4.4 分钟 → **约 0.9 条/分钟/节点**。

**两个预判被实测推翻，记下来免得重犯**：
1. 我预估 10~20 分钟/条，**实测 4.4 分钟** —— NF4 反量化开销没有想象中伤。
2. 担心 ARM 上 flash-attn 装不上只能退 SDPA —— **lightx2v 镜像里 flash_attn 和 sageattention 都已经有了**，
   速度好看有它们一半功劳。

**仍需验证**：`ref2va` 属参考/编辑类任务，低比特最易崩（参考 Bernini v2v 雪花结案：编辑类任务禁 int8），
画质必须与 bf16 对照着看。ref2va 的 NF4 DiT 已在 `MiniMax-H3-NF4/` 里（16G），未测。

---

## 5. ⚠️ 未完成路线：4 卡 768p（vllm-omni）

### 5.1 为什么必须是 vllm-omni

- **DiffSynth 的 H3 pipeline 零并行支持**（grep 无 ulysses/sp/dist），库里 xfuser 基建只接了 `wan_video` 和 `mova`
- **vllm-omni 上游有完整 H3 支持**，且是唯一给出 4 卡配方的框架：
  ```
  24741961 [Model] Add MiniMax H3 diffusion support (#5691)
  a4ea67a2 [Model] Add soundfile fallback for MiniMax H3 audio loading (#5699)
  900a7f08 [Model] Add MiniMax H3 T2VA accuracy test (#5709)
  ```
- 官方 recipe `recipes/MiniMaxAI/MiniMax-H3.md`，但**是在 4×B300(288G HBM) 上验的**，
  实测 FL2VA 209帧 1248×768 = 86.96s。**recipe 里零量化提及**——量化是我们自己加的。

⚠️ **本地 `api/vllm-omni` 工作树正在做回调/进度上报功能，不要动它。**
用独立工作树：
```bash
cd api/vllm-omni && git fetch upstream
git worktree add ../vllm-omni-h3 upstream/main --detach   # 本地曾落后上游 214 个提交
```

### 5.2 环境（容器 `voh3`，已在 0030 上常驻）

```bash
docker run -d --name voh3 --gpus all --shm-size 32g --ipc=host \
  -v /nfs-data:/nfs-data -v /root/h3:/work \
  --entrypoint bash reputationly/vllm-omni:arm64-a100-latest -lc "sleep infinity"

docker exec voh3 bash -lc "cd /work/vllm-omni-h3 && \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0 pip install -e . --no-deps"
```

**不用重建镜像**：上游 H3 代码在镜像自带的 vllm 0.25.0 上 **import 通过**（H3 只用 vLLM 稳定 API：
`get_tensor_model_parallel_world_size` / `init_logger` / `linear` / `default_weight_loader` / `QuantizationConfig`）。
只有一条 major.minor 软警告（`warn_if_misaligned_vllm_version`），不影响运行。

已验证可用：**4 卡 ulysses 组建成功**（`sp_size=4, ulysses=4, ring=1`）、**FLASH_ATTN 在 sm_80 解析成功**、
所需 CLI 参数齐全（`--usp` / `--text-encoder-tp-size` / `--vae-patch-parallel-size` / `--quantization` /
`--enable-layerwise-offload` / `--enable-distributed-layerwise-offload`）。

### 5.3 四轮实验记录

启动脚本都在 0030 的 `/root/h3/`，日志在同目录。

| 轮次 | 脚本 / 日志 | 配置 | 结果 |
|---|---|---|---|
| 1 | `serve_h3_4gpu.sh` / `serve.log` | `--quantization bitsandbytes --text-encoder-tp-size 4 --usp 4` | ❌ 12/13 分片 CUDA OOM（39.4G/40G，差 126MB） |
| 2 | 同上 / `serve2.log` | 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ❌ **无效**，占用反升到 38.88G → 证明不是碎片，是真的差这么多 |
| 3 | `serve_h3_lw.sh` / `serve_lw.log` | 加 `--enable-layerwise-offload` | ❌ 仍 OOM |
| 4 | `serve_h3_w4a16.sh` / `serve_w4.log` | 预量化 W4A16 + 补丁 + layerwise offload，**不传 `--quantization`** | ⚠️ **加载期 OOM 解决**，倒在权重命名 |

#### 根因 1（已修复）：在线量化被硬编码「先上 GPU 再压」

`vllm_omni/diffusion/model_loader/diffusers_loader.py::load_model` 打印：

```
Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
```

62G bf16 中间态放不进 40G 卡。**块流式 offload 救不了，因为它管推理期不管加载期。**

判 offline 的条件原本只有两个：`data_type == "mx_fp"`（AutoRound MXFP8，Blackwell 格式）和
`is_checkpoint_quantized` —— 而**后者在 vllm-omni 和 vllm 0.25.0 里从未被赋值过**，等于永远走在线分支。

**补丁（已打在 `api/vllm-omni-h3` 工作树 + 容器内，单测过，未提交）**：
抽出 `_is_checkpoint_quantized()`，除原有两条外再认：
- `is_checkpoint_*_serialized`（复用 `factory._disk_marks_serialized` 已有的命名约定，新方法免改）
- 方法本身是静态的：auto-round / inc / gptq / awq / compressed-tensors / modelopt（它们没有在线路径）

在线的 bitsandbytes 和扩散 int8 仍判 False，行为不变。**生效标志**：

```
diffusers_loader.py:396  Offline-quantized model with CPU offload, loading weights directly on CPU
```

第 4 轮已确认生效：四个 rank 全部打印该行，**GPU 占用从 39.4G 降到 501 MiB**。

#### 根因 2（未解决，当前拦路虎）：权重命名对不上

第 4 轮倒在：

```
ValueError: Following weights were not initialized from checkpoint:
{'transformer.blocks.14.attn.qkv_proj.weight', 'transformer.blocks.9.mlp.fc1.weight', ...}
```

| | `Ar4ikov/...W4A16-RTN` | FL2VA 原始（vllm-omni 吃的） |
|---|---|---|
| 命名 | `transformer_blocks.49.attn.to_out.0.qweight` | `blocks.0.attn.out_proj.weight` |
| q/k/v | 分开（`to_q`/`to_k`/`to_v`） | **融合**（`qkv_proj`） |
| 键数 | 1238 | 535 |
| **交集** | **1 个** | |

**Ar4ikov 量化的是仓库根目录那份 diffusers 格式**，而 vllm-omni 读的是 FL2VA 原始格式。
而且 vllm-omni 的 H3 **没有任何映射层**：

```python
def load_weights(self, weights):
    """Load exact H3 checkpoint names with logical TP-aware loaders."""
    param = params.get(name)      # 纯字典精确匹配
```

另外该 checkpoint `iters: 0`，是纯 RTN 无校准（仓名 `-RTN` 名副其实），**画质预期本来就低于真 auto-round**。

### 5.4 显存账（供下一位估算）

官方 recipe 的 4 卡配置在 A100-40G 上的账：

| 组件 | 每卡 | 备注 |
|---|---|---|
| DiT | **复制**，不切 | `--usp/--ulysses-degree` 是序列并行，只切激活不切权重；recipe 明写 "TP left at 1" |
| Text Encoder | 51.5G ÷ N | `--text-encoder-tp-size N` 是**真切**（N 须整除 64 heads / 8 KV heads → 4 卡可选 1/2/4） |
| 激活 | usp4 把 768p 的 92k token 切成 **23k/卡** | 与已验证的 480p（12k）同量级 |

→ **DiT 常驻是唯一死结，跟卡数无关**。加节点只能缓解激活和 TE，解决不了 DiT。
只有「量化 + 块流式 offload」能解。

块流式的可行性账（H3 有 50 层）：单块权重 ≈ 35G/50 ≈ 0.70 GiB，PCIe ~20GB/s → 搬运 **~35ms**；
单块计算外推 200-300ms → 搬运占 12-18%，**双缓冲理论上掩盖得掉**。
不确定项：4 个 rank 抢同一 root complex 的聚合带宽、鲲鹏 NUMA 拓扑（部署要绑 GPU0,1 / GPU2,3 同 NUMA）。

---

## 6. 下一步选项

**推荐优先级从高到低：**

### 选项 1：改加载器做逐 shard 量化（推荐）

不依赖任何外部权重，用已有的 135G bf16 就能跑，而且是根因 1 的**正确修法**，改完可以提回上游。

**当前调用链**（`vllm_omni/diffusion/model_loader/diffusers_loader.py`，行号为打补丁后）：

```
load_model()                                      # L370
├── L384-397  决定 load_device
│              在线量化 → load_device = "cuda", offload_after_quant = True
├── L~420     _init_from_load_format(...)         # 建模型骨架
├── L440      self.load_weights(model)            # ⚠️ 把全部 13 个 shard 的 bf16 权重
│                                                 #    一次灌进 GPU params → 62G → 12/13 处 OOM
├── L441      self._process_weights_after_loading(model, target_device)
│                                                 #    L518/L537 逐 module 调
│                                                 #    quant_method.process_weights_after_loading()
│                                                 #    → bnb quantize_4bit()
└── L443-445  if offload_after_quant: model.to("cpu")
                                                  #    整体挪回 —— 太晚了，OOM 已经发生
```

**病灶**：L440 的「全部加载」和 L441 的「全部量化」是两个串行的整体阶段，
GPU 峰值 = **完整 bf16 模型**，而不是「已量化部分 + 当前一个 shard」。

**改法**：把这三步交错成 per-module（或 per-shard）流水：
> 加载一个模块的权重 → 立刻 `process_weights_after_loading` 量化它 → 立刻 `.to("cpu")` → 释放显存 → 下一个

峰值就降到 `已量化NF4(累积在CPU) + 单个模块的bf16`，40G 绰绰有余。

**动手位置**：
- `load_weights()`（L542）和 `_get_weights_iterator()`（L272）是权重流的来源，天然是按 shard 迭代的
- `_process_weights_after_loading()`（L469）已经是**遍历 module** 的写法（L518/L537 两处调用），
  把它拆成「可对单个 module 调用」的形式即可复用
- 注意 `_has_online_quant()`（L451）判断的是 upstream vLLM 的 `uses_meta_device`（如 online FP8），
  与本改动是两回事，别混

**验证判据**：
1. 日志中 `Multi-thread loading shards: X/13` 推进过程中，`nvidia-smi` 显存**不应单调爬升到 39G**，
   应在某个远低于 40G 的值上下波动
2. 能推进过 13/13 并出现 `Application startup complete`
3. 出片与单卡 480p 同 seed 对比，画质不应有肉眼可见退化

预估半天，风险中等。改完建议提回上游（这是通用 bug，不止影响 H3）。

### 选项 2：自己量化出命名匹配的权重
用 auto-round / GPTQ 从 `FL2VA/transformer` 原始 bf16 量化，**保持原始命名**。
挑战：auto-round 是给 HF transformers/diffusers 模型设计的，要适配 vllm-omni 的自定义 H3 模型；
33B 校准要 GPU 时间和校准数据。预估 1-2 天。

**做之前先看这两条社区情报，能大幅降低难度**（数据来自 `DeepBeepMeep/MiniMax-H3`，
Wan2GP 作者，专攻低显存，他的数字是实测不是推演）：

**① 剪枝能省 23.2G，比预期多**

```
MiniMax-H3-FL2VA_bf16.safetensors          61.73 GB
MiniMax-H3-FL2VA-pruned_bf16.safetensors   38.56 GB    ← 剪掉 23.2 GB
MiniMax-H3-FL2VA-pruned_int8_convrot       20.62 GB    ← 再叠 int8
```

正好对上官方说的「约 13B 在 AdaLN 分支，推理时可预计算缓存、不必加载」（13B×2 ≈ 26G，实测省 23.2G）。
**若在原始格式上做同样剪枝，DiT 62G → 38.5G，再叠 4bit 只剩 ~10G —— 单卡都装得下。**
这条路已被社区验证可行，不是理论推演。

**② TE 只需保留 50 层**

```
Qwen3-VL-32B-Instruct-layer50_bf16.safetensors        47.97 GB
Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8        24.89 GB
qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf                    13.58 GB
```

注意 `layer50` 这个命名 —— 大家都只保留到第 50 层（H3 只取第 50 层 hidden state），后面的层直接丢。
我们目前是把完整的 63G TE 整个加载进去的，这里有明显的优化空间。

⚠️ 这两份权重本身都是 **ComfyUI 扁平单文件格式，vllm-omni 读不了**，只能借鉴思路不能直接用。

### 选项 3：写权重转换脚本
diffusers 命名 → 原始命名 + qkv 融合 + GPTQ scales/zeros 重排。
**风险最高**：GPTQ 打包下融合 q/k/v 要处理 group_size 128 对齐和 zero point 布局，错了是**静默的数值错误**，
转完必须做数值一致性验证。预估 1 天+。

### 选项 4：等 / 换硬件
H3 才开源一天，很可能几周内社区就有 vllm-omni 格式的量化权重。或者换 80G 卡则一切迎刃而解。

### 选项 5：先做 480p 这条的产品化
`ref2va` 画质验证（**必做**，编辑类任务对低比特最敏感）、`fl2va` 首尾帧验证（权重已在本地，
脚本已按真实 API 写好）、多实例吞吐压测、GPUStack 接入。

**多机方向的补充**：本节点有 4×100Gb RoCE 全部在线，聚合 50GB/s **比节点内 PCIe 还快**，
多机方案比通常直觉可行得多。但如上所述，加节点解决不了 DiT 常驻，
所以应该在选项 1/2 跑通之后再考虑——那时扩节点是**降延迟**，不是**救 OOM**。
届时还要落实：vllm-omni 的多机编排（vllm 走 Ray，`--num-gpus` 是否只管单机需确认）、
NCCL 走 RDMA 的配置（`NCCL_IB_HCA` / GDR）。

---

## 6.5 从零到能跑（复制粘贴即可）

两个容器 `h3smoke` / `voh3` **目前都还在 0030 上运行着**，权重也都在 NFS。
正常情况下直接用即可，不用重建。下面是万一容器没了的重建步骤。

```bash
ssh -p 43055 root@111.172.214.16

# ---------- A. 单卡 NF4（已验收路线） ----------
docker run -d --name h3smoke --gpus all --shm-size 32g --ipc=host \
  -v /nfs-data:/nfs-data -v /nfs-models:/nfs-models -v /root/h3:/work \
  --entrypoint bash \
  crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest \
  -lc "sleep infinity"

docker exec h3smoke bash -lc '
  pip install -q bitsandbytes
  cd /work/DiffSynth-Studio && pip install -q -e . --no-deps && pip install -q pandas peft
  python3 -c "from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline; print(\"OK\")"'

docker exec h3smoke bash -lc "cd /work && python3 smoke_minimax_h3_nf4.py --dry-run"          # 自检
docker exec h3smoke bash -lc "cd /work && CUDA_VISIBLE_DEVICES=0 python3 -u \
  smoke_minimax_h3_nf4.py --steps 50 --out /work/h3_t2va_50.mp4"                              # 正片
# 期望：265s 左右，显存峰值 34.4G，产出 832x480/24fps/5.175s 带 AAC 32kHz 立体声

# ---------- B. 多卡 vllm-omni（待优化路线） ----------
docker run -d --name voh3 --gpus all --shm-size 32g --ipc=host \
  -v /nfs-data:/nfs-data -v /root/h3:/work \
  --entrypoint bash \
  crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest \
  -lc "sleep infinity"

docker exec voh3 bash -lc '
  cd /work/vllm-omni-h3 && SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0 pip install -e . --no-deps
  cd / && python3 -c "from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer; print(\"OK\")"'

# 复现第 4 轮（会走到权重命名报错为止，用来确认补丁生效）
docker exec voh3 bash -lc 'cd /work && setsid nohup ./serve_h3_w4a16.sh > /work/serve_w4.log 2>&1 </dev/null &'
sleep 200 && docker exec voh3 bash -lc 'grep -E "Offline-quantized|Online quantization" /work/serve_w4.log'
# 期望看到 4 行 "Offline-quantized model with CPU offload, loading weights directly on CPU"
```

**源码位置**：
- 宿主机 `/root/h3/vllm-omni-h3/`（容器内 `/work/vllm-omni-h3/`），editable 安装，改完即生效**无需重装**
- Mac 上 `~/Desktop/code/api/vllm-omni-h3/`（git worktree，detached at `upstream/main`）
- 两边改完记得同步：
  `scp -P 43055 <file> root@111.172.214.16:/root/h3/vllm-omni-h3/<相对路径>`

### 补丁全文（未提交，重建环境时需重新应用）

`vllm_omni/diffusion/model_loader/diffusers_loader.py`，在 `_resolve_custom_pipeline_cls()` 之后、
`class DiffusersPipelineLoader` 之前插入：

```python
# Quantization methods that only ever produce packed weights offline: a config for
# any of these can only have come from the checkpoint's own config.json, never from
# a bare --quantization flag.
_STATIC_QUANT_METHODS = frozenset(
    {
        "auto-round",
        "inc",
        "gptq",
        "gptq-marlin",
        "awq",
        "awq-marlin",
        "compressed-tensors",
        "modelopt",
    }
)


def _is_checkpoint_quantized(quant_cfg: object) -> bool:
    """Return True when the checkpoint already holds packed quantized weights.

    Such weights must be loaded straight to CPU under CPU offload: staging them on
    the accelerator first is pure waste, and for large DiTs it OOMs before the
    offload-back ever runs. Four signals, most specific first:

    1. ``is_checkpoint_quantized`` — explicit opt-in for configs that set it.
    2. ``data_type == "mx_fp"`` — AutoRound MXFP8, which carries no serialized flag.
    3. ``is_checkpoint_*_serialized`` — the same naming convention
       ``factory._disk_marks_serialized`` keys off, so new methods need no edit here.
    4. the method is static (see ``_STATIC_QUANT_METHODS``).
    """
    if getattr(quant_cfg, "is_checkpoint_quantized", False):
        return True
    if getattr(quant_cfg, "data_type", None) == "mx_fp":
        return True
    for key, val in vars(quant_cfg).items():
        if val and key.startswith("is_checkpoint_") and key.endswith("_serialized"):
            return True
    get_name = getattr(quant_cfg, "get_name", None)
    if callable(get_name):
        return str(get_name()).lower().replace("_", "-") in _STATIC_QUANT_METHODS
    return False
```

并把 `load_model()` 里的判断替换掉：

```diff
         offload_after_quant = False
         if load_device == "cpu" and self.quant_config is not None and device is not None:
             quant_cfg = self.quant_config
-            is_offline = getattr(quant_cfg, "data_type", None) == "mx_fp" or getattr(
-                quant_cfg, "is_checkpoint_quantized", False
-            )
+            is_offline = _is_checkpoint_quantized(quant_cfg)
             if not is_offline:
```

补丁自测（不需要 GPU）：

```bash
docker exec voh3 bash -lc 'cd / && python3 -c "
from vllm_omni.diffusion.model_loader.diffusers_loader import _is_checkpoint_quantized as f
class C:
  def __init__(s,n,**kw): s._n=n; s.__dict__.update(kw)
  def get_name(s): return s._n
assert f(C(\"bitsandbytes\")) is False      # 在线，行为不变
assert f(C(\"int8\")) is False              # 在线，行为不变
assert f(C(\"auto-round\")) is True         # 静态
assert f(C(\"gptq\", is_checkpoint_gptq_serialized=True)) is True
assert f(C(\"auto-round\", data_type=\"mx_fp\")) is True
print(\"patch OK\")"'
```

---

## 6.6 排障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `error: [E3020] [404] 获取模型文件失败` + 静默回退 hf-mirror | 新版 `ms` CLI 位置参数只吃字面文件名，不吃 glob | 下载脚本已修（先用清单 API 展开）。自己写命令时别传 `FL2VA/*` |
| 下载只有 3-4 MB/s | 该仓 ModelScope 没有，回退到了 hf-mirror | 正常现象。hf-mirror 限单连接不限总带宽，4 连接分块可提速 4.6× |
| `ModuleNotFoundError: sentencepiece` / `pip install -e .` 整体失败 | sentencepiece 无 aarch64 轮子 | `pip install -e . --no-deps`，再补 `pandas` `peft` |
| `git clone` 卡住不动 | GitHub 从 0030 连不通 | Mac 上 clone → `tar czf` → scp |
| `gpustack: error: unrecognized arguments: bash -lc ...` | 镜像有 ENTRYPOINT | `docker run --entrypoint bash <img> -lc "..."` |
| `packaging.version.InvalidVersion: Invalid version: 'dev'` | 打包时排除了 `.git`，setuptools_scm 拿不到版本 | `SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0 pip install -e . --no-deps` |
| 起服务的命令执行后没有日志文件 | `pkill -f "vllm serve"` 匹配到自己所在 shell 把自己杀了 | 写成 `pkill -f "vllm ser[v]e"`；后台起用 `setsid nohup ... < /dev/null &` |
| `EOFError` / `TCPStore Broken pipe` / `Orchestrator initialization failed` | 这些都是**下游噪声**，某个 rank 先死导致 | 别看这些，`grep` 时排除 `ProcessGroupNCCL\|TCPStore\|Broken pipe\|c10::Error\|frame #\|EOFError`，找第一现场 |
| `torch.OutOfMemoryError` 在 `Multi-thread loading shards: 12/13` | 根因 1，见 5.3 | 用预量化 checkpoint，或做选项 1 的改造 |
| `ValueError: Following weights were not initialized from checkpoint` | 根因 2，权重命名对不上 | 见 5.3。确认 checkpoint 是从 **FL2VA 原始格式**量化的，不是 diffusers 格式 |
| `vllm serve --help` 看不到 `--usp` 等参数 | 默认只打印 87 行分组概览 | `vllm serve --omni --help=all`（2097 行） |

---

## 7. 杂项踩坑（省时间用）

- **新版 `ms` CLI 位置参数只吃字面文件名，不吃 glob**。传 `FL2VA/*` 会 404（E3020）然后静默回退 hf-mirror
  （慢 20-60 倍）。下载脚本已改为先用清单 API 展开成真实文件名。
- **`pkill -f "vllm serve"` 会匹配到自己所在的 shell 命令行把自己杀掉** → 写成 `pkill -f "vllm ser[v]e"`。
- `vllm serve --help` 只有 87 行分组概览，要看全量参数得 `vllm serve --omni --help=all`（2097 行）。
- 社区量化版筛选结论（HF+魔搭共 10 个，只有 2 个能进）：
  - **NVFP4 系全灭**（lilcheaty / rockerBOO / Abiray）——需 Blackwell sm_100+，A100 是 sm_80
  - **ComfyUI 系全灭**（gordonz int4-convrot / Abiray Convrot / Abiray GGUF / tsolful INT4Mixed）——
    扁平单文件无 config/index/tokenizer，SGLang/vLLM/diffusers/DiffSynth 都读不了
  - **MLX 系**（ddalcu）是 Apple Silicon 专用
  - `benjiaiplayground/MiniMax-H3_quant` 是**空壳**，58 个文件全是 assets/docs，没有权重
  - 能进的两个：DiffSynth NF4（已用）、Ar4ikov W4A16（格式对不上，见 5.3）

---

## 8. 相关文档

- `docs/InfiniteTalk-优化总结与Wan2.2移植清单.md` —— offload vs 常驻的实测对比
  （bf16+block offload 103s vs int8 常驻 93.8s，**offload 惩罚约 +10%**）；Amdahl 拆账
- `../Bernini/docs/bernini-常驻块流式优化与生产验证报告.md` —— 块流式的完整实现与验证
  （⚠️ 其中「块流式比常驻快 2.83×」的红利**不适用于 H3**，根因是 Bernini 有高/低噪声双专家、
  切换要整体搬运；H3 是单流 dense，没有专家切换）
- `../vllm-omni/docs/vLLM-Omni-大模型量化与Offload可行性调研.md` —— A100 sm_80 的量化红线
  （FP8 不可用，只剩 weight-only INT8 / W4A16）；vllm-omni 的 offload 基建全在
  `vllm_omni/diffusion/offloader/`，**扩散专用**
- `docs/LTX2.3-单条延迟优化记录.md` —— 逐层流式 offload 的真实秒级成本（gemma 正向编码 27s）
- `MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_*.md`（NFS 上）—— H3-Context-IR 未开源，
  提示词质量全靠这个，**做效果评估前必读**
