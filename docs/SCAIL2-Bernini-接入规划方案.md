# SCAIL-2 & Bernini 接入 LightX2V — 落地规划方案

> 状态:规划(未开工)。日期 2026-07-19。
> 结论先行:**两个都是 Wan 骨架的衍生模型,走 LightX2V「原生接入」(runner/model/config)可行,复用度高。** 上游/官方核对见文末来源。
> 阅读前置:`support_new_model` skill、`docs/InfiniteTalk-优化总结与Wan2.2移植清单.md`、`docs/Wan2.2-VACE-实验测试报告.md`(VACE 的 v2v/mask/ref 链路是 Bernini/SCAIL 的直接模板)。

---

## 0. 一页纸对比

| 维度 | **SCAIL-2**(智谱 ZAI) | **Bernini / Bernini-R**(字节 ByteDance) |
|---|---|---|
| 任务 | 角色动画 / 换人(reference 角色图 + 驱动视频 → 动画) | 统一视频生成+编辑:t2i/i2i/t2v/v2v/rv2v/r2v |
| 骨架 | **Wan2.1-I2V-14B** fork(3 段 RoPE:reference/video/pose + 双 mask) | 渲染器 **Wan2.2-T2V-A14B** fork;full 版另加 **Qwen2.5-VL-7B** 语义规划器(MLLM) |
| 变体 | 14B(单一) | Bernini-R 14B / Bernini-R **1.3B** / full Bernini(planner+renderer) |
| LightX2V 关系 | 4 步蒸馏 LoRA(`Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64`)+ 官方 DPO LoRA | 4 步蒸馏 LoRA 对(high/low noise,社区已按 LightX2V 蒸馏) |
| 上游接入现状 | LightX2V 主仓**未原生支持**;社区经 ComfyUI/WanGP 用 LoRA 跑 | 同上 |
| 依赖坑 | sat→safetensors 需 `convert.py`;Python 3.10–3.12 | 硬依赖 VeOmni;`diffusers==0.35.2 / transformers==4.57.3 / torch2.7.1+cu126`;**FA3 只在 Hopper**(A100 回退 SDPA) |
| License | 仓库 Apache-2.0 / HF 卡 MIT(有出入) | Apache-2.0 |

**接入优先级建议:先 SCAIL-2 后 Bernini。** SCAIL-2 = 单一 Wan2.1 DiT + 输入编码改造,和我们已验的 Wan2.1/VACE 路径最近;Bernini-R 次之(Wan2.2 渲染器,复用 VACE/int8 经验);**full Bernini 最后**(要拉 Qwen2.5-VL planner,是新子系统)。

---

## 1. 需要下载的权重清单

> 约定:落 `/nfs-data/models/<repo>`,int8 转好落 `/nfs-data/models-int8/`。起容器必带 `-v /nfs-data:/nfs-data -v /data:/data`。优先魔搭(ModelScope)镜像,HF 作回退(见 [[nfs-weights-storage]])。

### 1.1 SCAIL-2
| 组件 | 来源 | 说明 / 大小 | 可复用? |
|---|---|---|---|
| SCAIL-2 主权重(DiT) | `zai-org/SCAIL-2`(HF) | sat 格式 `model/1/fsdp2_rank_0000_checkpoint.pt`,**需 `convert.py --scail-dir … --save-path …safetensors`** 转 wan 分支 | 新下(~28G bf16 量级) |
| Wan2.1 VAE | 仓库自带 `Wan2.1_VAE.pth` | 与我们已有 Wan2.1 VAE 同源 | ✅ 大概率复用现有 |
| umt5-xxl 文本编码器 | 仓库自带 `umt5-xxl/` | Wan2.1 系通用 T5 | ✅ 复用现有 Wan2.1 |
| DPO LoRA(可选,提质) | `Comfy-Org/SCAIL-2`(loras) | 小,增强偏好对齐 | 新下(小) |
| LightX2V 4 步蒸馏 LoRA | `Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64` | 加速核心 | ✅ **我们 Wan2.1-Distill-Loras 里应已有**(InfiniteTalk 用过),核对文件名 |

社区备选(不建议生产):`xocialize/SCAIL-2-bf16`(MLX,实验)、`realrebelai/SCAIL-2_GGUF`(ComfyUI 低显存)。

### 1.2 Bernini
| 组件 | 来源 | 说明 / 大小 | 可复用? |
|---|---|---|---|
| Bernini-R 14B 渲染器 | `ByteDance/Bernini-R-Diffusers` | Wan2.2 渲染器微调,**先接这个** | 新下(~28G) |
| Bernini-R 1.3B(轻量档) | `ByteDance/Bernini-R-1.3B-Diffusers` | 低显存/快,Wan2.1-1.3B 微调;40G 上做快验很划算 | 新下(小,~3G) |
| full Bernini(planner+renderer) | `ByteDance/Bernini-Diffusers` | 含 MLLM 语义规划器,**二期** | 新下(大) |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | full Bernini 的 planner;单独 ~16G | 新下(除非已有) |
| Wan2.2 VAE / 文本编码器 | 随 Diffusers 仓库 or 复用 Wan2.2 | 与 Wan2.2-VACE/I2V 同源 | ✅ 大概率复用 |
| LightX2V 4 步 LoRA 对 | `rzgar/Bernini-R-LightX2V-4step-loras` | `..._high_noise.safetensors` + `..._low_noise.safetensors`;strength 1.0/1.0,4 步,dpmpp_2m_sde/sgm_uniform | 新下(小,LightX2V 出品) |

---

## 2. NFS 已有可复用盘点(需上机 `ls` 核对)

复用点(能省下载和转换的大头):
- **Wan2.1 VAE / umt5-xxl**:`/nfs-data/models/Wan2.1-I2V-14B-480P` 与 InfiniteTalk 权重里都带 → SCAIL-2 的 VAE+T5 直接指过去,**只需下 DiT**。
- **Wan2.2 VAE / 文本编码器**:`/nfs-data/models/Wan2.2-VACE-Fun-A14B`、`Wan2.2-I2V-A14B` 里的 VAE/T5 → Bernini-R 复用。
- **Wan2.1/2.2 蒸馏 LoRA**:`/nfs-data/models/Wan2.2-Lightning`(VACE 已转 x2v 格式)、`Wan2.1-Distill-Loras`(交接文档记录已下)→ SCAIL-2 的 Wan2.1 4 步 LoRA **很可能已在盘**,核对文件名即可。
- **int8 转换工具链**:`scripts/convert_int8.sh` / `convert_lightning_to_x2v_lora.py` 现成,两模型都能套(见 §3.4)。

**需净下**:SCAIL-2 DiT、Bernini-R DiT(14B+1.3B)、Qwen2.5-VL-7B(二期)、两组模型专属小 LoRA。核对命令:
```bash
ls -la /nfs-data/models/ | grep -iE "Wan2.1-I2V|Wan2.2-VACE|Wan2.2-I2V|Distill-Loras|Lightning|Qwen2.5-VL"
ls /nfs-data/models/Wan2.1-Distill-Loras/ 2>/dev/null    # 找 lightx2v_cfg_step_distill_lora_rank64
```

---

## 3. LightX2V 工程改造(按模块)

参照 registry:runner 名如 `@RUNNER_REGISTER("wan2.2_moe_vace")`(`wan_vace_runner.py`)。task 分支在 `wan_runner.py:200/867` 的白名单里判定。

### 3.1 SCAIL-2 —— 新 runner + 输入编码是核心
- **runner**:`lightx2v/models/runners/wan/wan_scail2_runner.py`,`@RUNNER_REGISTER("wan2.1_scail2")`,base 继承 `wan2.1`(i2v)。
- **难点(与标准 Wan2.1 的差异)**:**3 段 RoPE(reference / video / pose)+ 双 mask 条件**。要在 input-encoder / infer 阶段把「参考角色图、驱动视频、pose 序列、两张 mask」拼进 latent 与位置编码。这是主要工作量,不是简单换权重。
- **权重加载**:sat→safetensors 转换后,写 weight 映射(参考现有 wan2.1 weight 类)。
- **task**:新增 `scail2`(角色动画)到 `wan_runner.py` 任务白名单。
- **config**:`configs/wan/wan_scail2_a100.json`(bf16 单卡)+ `..._int8.json`,路径全走 `/nfs-data/...`,**不硬编码**(遵循 skill 规范)。

### 3.2 Bernini-R —— 最接近 VACE,复用度最高
- **runner**:`wan_bernini_runner.py`,`@RUNNER_REGISTER("wan2.2_bernini")`,继承 `wan2.2_moe`。
- **复用 VACE 链路**:v2v/rv2v/r2v 的「源视频 + 参考图 + mask」输入编码,**直接借 `wan_vace_runner.py` 已跑通的 src_video/src_mask/src_ref_images 处理**(VACE 报告里 R2V/inpaint/outpaint 六模式已验)。
- **1.3B 快验档**:先用 Bernini-R-1.3B 在 40G 单卡 bf16 打通全链路(小、快、省显存),再上 14B。
- **task**:新增 `v2v` / `rv2v` / `r2v`(部分可映射到现有 vace 语义)。

### 3.3 full Bernini(二期)—— 加 MLLM planner 子系统
- 新增「语义规划」前置:input-encoder 里挂 Qwen2.5-VL-7B,产出 target semantic embedding 喂给渲染器。
- 这是**新推理阶段**(类似给 pipeline 插一个 VLM 前处理),显存 +~16G,40G 单卡吃紧,可能要 planner 与 renderer **分卡/分时**。二期再评估,一期先只做 Bernini-R(纯渲染器,和 Wan2.2 一样)。

### 3.4 加速与量化(复用既有铁律)
- **4 步蒸馏 LoRA**:两模型都有 LightX2V LoRA → 套 `lora_dynamic_apply:true` + `convert_lightning_to_x2v_lora.py`(VACE 的 P9 补丁经验:runner `__init__` 要透传 `lora_path`,上游常漏)。预期 10×+ 提速。
- **int8-triton**:14B 在 **40G A100 上 bf16 大概率 OOM**(VACE/InfiniteTalk 均如此),**int8-triton 离线转换是唯一活路**(torchao 后端慢,别用 —— 见 [[infinitetalk-deep-test-verdict]])。用 `scripts/convert_int8.sh` 转 DiT。
- **注意力**:Bernini 官方要 FA3(Hopper),**我们 A100 无 FA3 → 回退 SDPA/FA2**(和 S2V 的 fa3 硬依赖补丁同类问题,`docs` 有先例)。ARM 宿主还要留意 decord→PyAV 兜底(VACE P8 补丁)。

### 3.5 server schema(接入服务/GPUStack 前必做)
现 `lightx2v/server/schema.py::VideoTaskRequest` **已具备** VACE 时补的字段:`src_video` / `src_mask` / `src_ref_images` / `video_path` / `image_path`。
- SCAIL-2 需**新增字段**:`pose_video`(pose 驱动序列)、`ref_image`(参考角色,可复用 `src_ref_images`)、必要时第二 mask。
- Bernini v2v/rv2v 基本可复用现有字段,`guidance_mode` / `task_type` 建议加入。

---

## 4. GPUStack 调用 —— 要改什么

结论(源码级,见 [[gpustack-lightx2v-integration]]):**GPUStack 侧无需改代码即可调**,机制是 custom backend 容器 + model route `generic_proxy:true` 原样透传 LightX2V 异步 API(`POST /v1/tasks/video` → task_id → 轮询)。所以「能不能调」= 能,前提是 LightX2V 这两个模型先在 server 里跑通。

需要动的是 **LightX2V + 部署配置**,不是 GPUStack 内核:
1. **server schema 补字段**(§3.5)—— 否则 pose/ref/mask 传不进去。
2. **新后端 spec / 启动命令**:`community-inference-backends` 的 `spec.yaml` 里为 scail2/bernini 各加一个 profile(`--model_cls wan2.1_scail2` / `wan2.2_bernini`、对应 config、`{{model_path}}` 占位)。参照现有 lx2v-launcher profile(commit `516b9306` 已为 InfiniteTalk/SeedVR2/VACE 加过 task inference profile)。
3. **单实例先行**:task_id 亲和性坑未解前(多实例轮询会打到别的实例查不到),PoC 用**单实例**;生产要 sticky session 或 task 状态入共享存储(Redis)。
4. **显存打分**:GPUStack scorer 按 LLM 设计,视频 DiT(+VAE 峰值,Bernini 还 +VLM)预估不准 → 建议**整卡独占**调度。
5. **大文件回传**:视频经网关有大小/超时风险 → 让 LightX2V 直吐对象存储 URL,不走网关回传。

---

## 5. 分阶段路线图

| 阶段 | 目标 | 关键动作 | 验收 |
|---|---|---|---|
| **P0 侦察**(0.5d) | 核对上游代码 + NFS 复用面 | clone `zai-org/SCAIL-2`、`bytedance/Bernini`;`ls` NFS 对齐 §2;确认 convert.py / 依赖版本 | 下载清单锁定、复用项确认 |
| **P1 SCAIL-2 PoC**(2–3d) | 单卡 bf16 跑出一条角色动画 | 下 DiT+转 safetensors;写 runner + 3 段 RoPE/双 mask 输入编码;bf16 单卡 | 输出与上游 pipeline **画质对齐**(skill 要求 parity) |
| **P2 SCAIL-2 加速+服务**(2d) | 4 步 LoRA + int8,进 server | 套蒸馏 LoRA;int8-triton 转换;schema 补 pose/ref;单实例 server | 提速≥10×、40G 不 OOM、`/v1/tasks/video` 可调 |
| **P3 Bernini-R PoC**(2–3d) | 1.3B→14B 视频编辑跑通 | 复用 VACE 输入链路;先 1.3B 打通再 14B;int8 | v2v/rv2v 输出正确 |
| **P4 Bernini-R 加速+服务**(2d) | LightX2V high/low LoRA + server | 下 `rzgar/...` LoRA 对;schema/task 补齐 | 4 步稳定、进 server |
| **P5 GPUStack 上线**(1–2d) | 两模型注册为后端 | 加 spec profile、单实例、整卡独占、对象存储回传 | 经 `/model/proxy/<id>/...` 端到端可调 |
| **P6(可选)full Bernini** | 加 MLLM 语义规划 | 挂 Qwen2.5-VL planner;planner/renderer 分卡 | 复杂指令遵循提升 |

---

## 6. 主要风险 / 硬件适配(4×A100 40G,无 NVLink,鲲鹏 ARM)

- **40G 显存**:14B bf16 单卡/多卡多半 OOM(VACE/InfiniteTalk 前车之鉴)→ int8-triton 常驻是主路;多卡通信在无 NVLink+ARM 上历来提不动(W2 全灭),**优先单卡 int8 + 多实例**而非张量并行。
- **FA3 缺失**:Bernini 官方要 Hopper FA3,A100 回退 SDPA/FA2,需补 attention 回退(有 S2V 补丁先例)。
- **ARM 依赖**:decord 空壳 → PyAV 兜底(VACE P8);bnb/量化算子在 ARM 的可用性提前验(HunyuanImage3 POC 里 bnb-ARM 可用有记录)。
- **依赖冲突**:Bernini 钉死 `diffusers 0.35.2 / transformers 4.57.3 / torch2.7.1`,与本仓其它模型可能冲突 → 大概率**独立镜像/venv**(HunyuanImage3 也是独立 transformers 4.57.1)。
- **License**:SCAIL-2 仓库 Apache / HF 卡 MIT 有出入,商用前确认。

---

## 7. 来源(官方 + 上游核对)
- LightX2V 主仓支持清单(**不含** Bernini/SCAIL):https://github.com/ModelTC/LightX2V
- SCAIL-2 官方:https://github.com/zai-org/SCAIL-2 · 权重 https://huggingface.co/zai-org/SCAIL-2 · DPO LoRA `Comfy-Org/SCAIL-2`
- Bernini 官方:https://github.com/bytedance/Bernini · 渲染器 https://huggingface.co/ByteDance/Bernini-R · LightX2V 4 步 LoRA https://huggingface.co/rzgar/Bernini-R-LightX2V-4step-loras
- 社区集成(旁证):Wan2GP 12.21 更新加 SCAIL-2+LightX2V/Bernini 1.3B https://github.com/deepbeepmeep/Wan2GP · ComfyUI SCAIL-2 教程 https://docs.comfy.org/tutorials/video/zai/scail2
- 内部依据:[[gpustack-lightx2v-integration]] [[nfs-weights-storage]] [[vace-verdict-lora-editor]] [[infinitetalk-deep-test-verdict]] [[server-hardware-spec]]
</content>
</invoke>
