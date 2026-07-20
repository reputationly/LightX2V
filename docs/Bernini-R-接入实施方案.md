# Bernini-R 接入 LightX2V — 可落地实施方案(代码级)

> 状态:待实施。日期 2026-07-19。作者依据当前 main 代码分析(带 file:line)。
> 前置:权重已下 NFS(见 [[scail2-bernini-weights-downloaded]]);总体规划见 `docs/SCAIL2-Bernini-接入规划方案.md`。
> 范围:**只做 Bernini-R(渲染器)**。full Bernini(含 Qwen2.5-VL planner)不在本方案,二期另议。SCAIL-2 更远,单独排期。

---

## 0. TL;DR / 决策

- **Bernini-R = Wan2.2-T2V-A14B 的 MoE 微调**,磁盘结构 `transformer/` + `transformer_2/`(高/低噪双专家)完全对齐本仓 `wan2.2_moe` runner。**不需要新写 MoE 调度,复用现有 `Wan22MoeRunner`。**
- **唯一硬骨头 = 格式转换**:Bernini 是 HuggingFace **Diffusers 分片格式**,而 LightX2V 只认自家 native 格式。但——**本仓 `tools/convert/converter.py` 已内置 `wan_dit` 的 Diffusers↔native 双向 key 映射**(`--direction backward` 就是 Diffusers→native),且能在同一趟里顺带 int8/fp8 量化。转换路径已通,不用从零写。
- **加速 LoRA 已对口**:`rzgar/Bernini-R-LightX2V-4step` 的 high/low 两个 LoRA,直接填 `lora_configs` 的 `high_noise_model`/`low_noise_model`(本仓已支持 per-expert LoRA)。
- **架构已核实(2026-07-19)**:`transformer/config.json` = `WanTransformer3DModel`,`in_channels:16 / num_layers:40 / heads:40 / head_dim:128 / ffn:13824 / text_dim:4096`,与 Wan2.2-T2V-A14B **完全一致**;抽取 1095 个 key,顶层前缀仅 `['blocks','condition_embedder','patch_embedding','proj_out','scale_shift_table']` —— **零编辑专用模块**(无 vace_/ref_/扩通道)。**⇒ converter `wan_dit` backward 全命中,t2v/i2v 零架构改动可跑通。**
- **VAE = 16 通道 Wan2.1 VAE**(`in_channels:16` 决定;Wan2.2-T2V-A14B 本就用 Wan2.1 VAE)→ config `vae_type:"wan"` + **直接复用现成 `Wan2.1_VAE.pth`**,零转换。
- **编辑任务(v2v/rv2v/r2v)不在权重里**:纯 T2V 架构 ⇒ 编辑靠 **in-context 序列拼接**(源视频/参考图 VAE 编码后当额外 token 拼进输入序列),逻辑在 Bernini 上游 pipeline,**不碰模型**,P4 照抄即可。

**落地顺序:先 t2v/i2v 跑通 parity(纯生成,零架构改动)→ 读上游 pipeline 抄 in-context 拼接 → 补编辑链路。**

---

## 1. 结构映射(Bernini-R → 本仓 wan2.2_moe)

| Bernini-R Diffusers 目录 | 内容 | 映射到 LightX2V |
|---|---|---|
| `transformer/` (54G, fp32) | 高噪专家 DiT | `high_noise_model/`(转换后 native safetensors)`wan_runner.py:726-744` |
| `transformer_2/` (54G, fp32) | 低噪专家 DiT | `low_noise_model/` 同上 |
| `text_encoder/` (11G, 5 分片) | umt5-xxl T5 | 转成单体 `.pth`(见 §4.2) |
| `vae/` (485M) | VAE(Wan2.2 系,待 config 确认) | 转成单体 `.pth`(见 §4.3) |
| `tokenizer/` | umt5 tokenizer | 直接复用(HF 目录格式,本仓就要这个)`wan_runner.py:138` |
| `scheduler/` `config.json` | 采样器/超参 | 读出 `boundary`/`shift`/`in_channels` 填我们的 config JSON |

**双专家切换机制**(现成,`wan_runner.py:633-722` `MultiModelStruct`):按 timestep 与 `boundary_timestep = boundary*1000` 比较选专家;蒸馏版走 `boundary_step_index`(`wan_distill_runner.py:55-77`)。Bernini 的 `scheduler/config.json` 里应有对应 boundary,照抄。

---

## 2. 核心工作项:权重转换(3 件)

工具:`tools/convert/converter.py`,CLI 支持 `--source`(目录/文件)`--direction {forward,backward}` `--model_type wan_dit` `--output/--output_name/--output_ext` `--quantized/--bits/--linear_dtype`(可同趟量化)。

### 2.1 DiT ×2(Diffusers→native,可选顺带 int8)
```bash
R=/nfs-models/wuhanjisuan894/models
OUT=/nfs-models/wuhanjisuan894/models-x2v/Bernini-R-14B    # native 落这
# 高噪专家
python tools/convert/converter.py --model_type wan_dit --direction backward \
  --source $R/Bernini-R-Diffusers/transformer \
  --output $OUT/high_noise_model --output_name model --output_ext .safetensors --single_file
# 低噪专家
python tools/convert/converter.py --model_type wan_dit --direction backward \
  --source $R/Bernini-R-Diffusers/transformer_2 \
  --output $OUT/low_noise_model --output_name model --output_ext .safetensors --single_file
```
- fp32→转换时可加 dtype 降 bf16(砍半到 ~28G/专家)。40G A100 上 **bf16 双专家常驻大概率仍 OOM**(VACE/InfiniteTalk 前例)→ 追加 `--quantized --bits 8 --linear_dtype int8` 出 int8-triton 版(参 [[vace-verdict-lora-editor]])。
- ⚠️ **转换后必须核对 key 命中率**:Bernini 若在 Wan2.2 基础上加了模块(编辑相关),这些 key 不在 `wan_dit` 映射表里会残留/丢弃,转换日志要看 unmapped key 数(见 §5)。

### 2.2 T5(Diffusers 5 分片 → 单体 .pth)
本仓 T5 加载只认单体 `.pth`(`t5/model.py:832` `load_weights`→`load_state_dict`),不吃 Diffusers 分片。**但 SCAIL-2 已自带一份 `models_t5_umt5-xxl-enc-bf16.pth`,且 Wan2.1-I2V-14B-480P 里也有** → **优先直接复用现成 umt5-xxl .pth,跳过转换**(umt5-xxl 是 Wan 系通用件)。仅当 Bernini 的 T5 权重与标准 umt5 有出入时才需转。

### 2.3 VAE —— 已定:复用 Wan2.1 VAE,零转换
`in_channels:16` 已确认 Bernini-R 用 **16 通道 Wan2.1 VAE**(非 Wan2.2 48 通道)。config 里设 `vae_type:"wan"`(走 `WanVAE`,`wan_runner.py:78`)+ 指向现成 `Wan2.1_VAE.pth`(SCAIL-2 / Wan2.1-I2V-14B-480P 里都有,或 Bernini 自带 `vae/` 转一份)。**优先复用现成 .pth,不转。**

---

## 3. 代码改动(file-by-file)

**乐观情形(Bernini-R 架构 == Wan2.2,纯微调):几乎零改动,复用 `wan2.2_moe` + 一个 config 即可。** 下面列全量潜在改动点,按需启用。

### 3.1 若架构等同 Wan2.2 → 无需新 runner
直接用 `model_cls: "wan2.2_moe"` + 新 config(§6)。t2v/i2v 立即可跑。**先验证这条路。**

### 3.2 若需要标识/微调差异 → 新 runner(薄封装)
`lightx2v/models/runners/wan/wan_bernini_runner.py`:
```python
@RUNNER_REGISTER("wan2.2_bernini")
class WanBerniniRunner(Wan22MoeRunner):   # 复用 MoE 双专家/切换/LoRA
    ...  # 仅覆盖有差异的部分(如编辑输入编码,见 §5)
```
注册机制:`registry_factory.py:68` + 派发 `infer.py:50` `RUNNER_REGISTER[config["model_cls"]](config)`。

### 3.3 编辑任务(v2v/rv2v/r2v)—— 见 §5 决策后再动
若走 VACE 式,复用 `wan_vace_runner.py` 的 `prepare_source`/`run_vae_encoder`(`wan_vace_runner.py:42-163`)与 `VaceInputInfo`;若走通道拼接式,改 input-encoder(`default_runner.py:109-126` 派发处加 `_run_input_encoder_local_v2v`)。

### 3.4 新 task 字符串的注册清单(全端 8 处,缺一不通)
若新增 `v2v`/`rv2v`/`r2v` task,须同步:
1. `infer.py:110` CLI `choices`
2. `wan_runner.py:94` image-encoder 白名单
3. `wan_runner.py:200` / `:867` VAE-encoder 白名单
4. `default_runner.py:109-126` input-encoder 派发
5. `schedulers/wan/scheduler.py:51` task 分支
6. `networks/wan/weights/transformer_weights.py:567` image-encoder 判定
7. `disagg/utils.py:286`(若用 disagg)
8. `server/schema.py` 已有 `src_video/src_mask/src_ref_images`(VACE 时补过),v2v 基本够;SCAIL 才需再加 pose 字段
> 若能把 Bernini 编辑复用现有 `vace` task 语义,则大部分白名单已就绪,改动最小。

---

## 4. server / GPUStack(复用既有结论)

- **server**:`VideoTaskRequest`(`server/schema.py:75-88`)已含 `src_video/src_mask/src_ref_images/video_path/image_path`,请求→runner 数据流已验(`worker.py:100-101` `update_input_info_from_dict`)。Bernini v2v 大概率零 schema 改动。
- **GPUStack**:内核不用改,custom backend + `generic_proxy` 透传异步 API(源码级已定,[[gpustack-lightx2v-integration]])。只需加一个后端 profile:`model_cls=wan2.2_bernini` + 新 config,单实例先行、整卡独占、视频吐对象存储 URL。

---

## 5. 必须先解决的开放问题(挡编辑任务)

> **已坐实(2026-07-20,读上游 `github.com/bytedance/Bernini` 源码,本地在 `../Bernini`)**:in-context 机制 = **序列维 token 拼接 + source-id RoPE**,非通道拼接、非 VACE 分支、非 CLIP。核心文件:`bernini/models/transformer_wan.py`(`patch_vae_latent`:446 / source-id RoPE:274-289)、`bernini/models/wan_diffusion.py`(`GEN_Wanx22.sample`:274-615 条件与 guidance)、`bernini/pipeline.py`(VAE 编码)。

### 5.1 in-context 配方(可直接照抄实现)
唯一架构 add-on = `use_src_id_rotary_emb`(RoPE 乘 per-source 相位,**零新增权重**;target=source_id 0=恒等,context=1..n,n>5 时 `linspace(1,5,n)` 分数 id 防外推)。核心原语 `patch_vae_latent`:每个"源"(噪声target、源视频、每张参考图)各自 VAE 编码→归一化`(z-mean)/std`→Conv3d patch_embed(16→inner, k=stride=(1,2,2))→**按序列维 cat**(context 在前、target 在最后),全序列双向注意力,最后只取 target token 段作预测。**无空间 mask**;mask 仅是"哪些 token 是 target"的布尔选择。

**逐任务输入构造**(target latent `[1,16,t,h/8,w/8]`):
- **v2v / mv2v**:context=[源视频(id1)];guidance_mode `v2v`/`v2v_chain`/`v2v_apg`。mv2v≡v2v,只换提示词(`task_type` 只影响提示词增强模板,不进 sampler)。
- **rv2v**:context=[源视频(id1)+参考图(id2..)];guidance_mode `rv2v`(链式 ∅→V→VI→VTI)。
- **r2v**:context=[仅参考图(id1..N,≤8,>5 插值)];无源视频;guidance_mode `r2v_apg`(APG 链)。
- **i2i**:单源图当 image context,`num_frames=1`;guidance_mode `t2v_apg`/`v2v_apg`。
- **t2v/t2i**:无视觉 context,`t2v`/`t2v_apg`。**⇒ 已跑通,印证 in-context 对纯生成是空集。**

**guidance_mode 组合**(每种=跑几次 forward 再线性组合,`_fwd` 返回 target 段 velocity;∅/V/VI/I=无/视频/视频+图/仅图 上下文栈,T=正文本):
- `v2v`(2次):`ε_VI + ω_txt·(ε_VTI − ε_VI)`
- `rv2v`(4次链式):`ε_∅ + ω_vid·(ε_V−ε_∅) + ω_img·(ε_VI−ε_V) + ω_txt·(ε_VTI−ε_VI)`
- `r2v_apg`(3次,APG):∅/I/TI 转 x 后走 `normalized_guidance_chain`
- 双专家边界后所有 ω×`omega_scale`(0.75/0.8)。

### 5.2 LightX2V 落点(三块改动,共用现有 int8 权重)
1. **RoPE**:给 wan RoPE infer 加 source-id 相位分支(`use_src_id_rotary_emb`);target 段传 id0(恒等,不影响 t2v)。
2. **input-encoder**:新增 `_run_input_encoder_local_bernini_edit`——VAE 编码源视频/参考图、按序列维拼 token、建 target-mask、分配 source_id。
3. **guidance/scheduler**:按 `guidance_mode` 跑多组合 forward 并线性组合(t2v 仍是现有单路,不受影响)。
4. **schema/task**:server 加 `guidance_mode` 字段;task 复用 `v2v`/`r2v` 或新增,走 §3.4 的注册清单。

> 仍需 0023 的原生 parity 标尺对照画质(见测试计划)。以下旧表保留作机制排除记录:

**Bernini 如何注入"源视频/参考图"做 v2v/rv2v/r2v?** 下表机制已被权重结构排除到只剩 in-context:

| 机制 | 判据 | 改动量 |
|---|---|---|
| (a) i2v 式**通道拼接** | `transformer/config.json` 的 `in_channels` > 标准(如 32/48 而非 16) | 小,改 input-encoder 拼 latent |
| (b) VACE 式**旁路上下文分支** | 权重里有 `vace_*` / 额外 `patch_embedding` / `before_proj` 类 key | 中,复用 `WanVaceModel` 那套(`vace_model.py`) |
| (c) **参考图走 image_encoder**(CLIP) | 有 CLIP 权重且 config `use_image_encoder` | 小,复用现有 i2v 图编码 |

**P0 必做**(拿到即可判):
```bash
R=/nfs-models/wuhanjisuan894/models
cat $R/Bernini-R-Diffusers/transformer/config.json          # 看 in_channels / 模块名
cat $R/Bernini-R-Diffusers/scheduler/config.json            # 看 boundary/shift/flow
cat $R/Bernini-R-Diffusers/config.json                      # 顶层 pipeline 描述
python3 -c "from safetensors import safe_open; import glob;
f=sorted(glob.glob('$R/Bernini-R-Diffusers/transformer/*.safetensors'))[0];
[print(k) for k in list(safe_open(f,'pt').keys())[:40]]"    # 抽样 key 名, 比对 Wan2.2
```
外加**读 Bernini 上游 modeling 代码**(`github.com/bytedance/Bernini` 的 Diffusers pipeline/transformer 定义)确认 t2v/i2v/v2v 各自的条件构造。**t2v/i2v 不依赖这个,可并行先跑。**

---

## 6. Config 模板(先 t2v/i2v)

参照 `configs/wan22/wan_distill_moe_flf2v_with_lora.json` 与 `configs/wan22_vace/a800/bf16/wan22_moe_vace.json`。新建 `configs/wan22_bernini/bernini_r_14b_distill.json`(路径不硬编码,走 config):
```json
{
  "infer_steps": 4,
  "target_video_length": 81,
  "target_height": 480,
  "target_width": 832,
  "self_attn_1_type": "sage_attn2",
  "cross_attn_1_type": "sage_attn2",
  "sample_guide_scale": [1.0, 1.0],
  "sample_shift": 5,
  "enable_cfg": false,
  "boundary_step_index": 2,
  "denoising_step_list": [1000, 750, 500, 250],
  "cpu_offload": true,
  "offload_granularity": "model",
  "vae_type": "wan",
  "vae_name": "Wan2.1_VAE.pth",
  "high_noise_original_ckpt": "/nfs-models/wuhanjisuan894/models-x2v/Bernini-R-14B/high_noise_model/model.safetensors",
  "low_noise_original_ckpt":  "/nfs-models/wuhanjisuan894/models-x2v/Bernini-R-14B/low_noise_model/model.safetensors",
  "lora_configs": [
    {"name": "high_noise_model", "path": "/nfs-models/wuhanjisuan894/models/loras/Bernini-R-LightX2V-4step/Bernini-R_LightX2V_high_noise.safetensors", "strength": 1.0},
    {"name": "low_noise_model",  "path": "/nfs-models/wuhanjisuan894/models/loras/Bernini-R-LightX2V-4step/Bernini-R_LightX2V_low_noise.safetensors",  "strength": 1.0}
  ],
  "lora_dynamic_apply": true
}
```
> `boundary_step_index`/`sample_shift`/`sample_guide_scale`/`denoising_step_list` 用 Bernini `scheduler/config.json` 的真值校准;VAE/T5 路径经 config 或 model_path 目录解析。1.3B 版另出一个 config(单专家,不用 high/low 两段)。

---

## 7. 分阶段实施 + 验收

| 阶段 | 动作 | 验收 |
|---|---|---|
| **P0 摸底**(0.5d) | 跑 §5 的 config/key dump;读 Bernini 上游 modeling;确认 VAE/T5 是否可复用 | 编辑注入机制定性、key 映射覆盖率清楚 |
| **P1 转 1.3B**(0.5d) | 先转小的 1.3B(单专家,5.3G)→ native | converter 无报错、key 全命中 |
| **P2 t2v/i2v 跑通 1.3B**(1–2d) | `wan2.2`(dense,1.3B 单专家)+ config,bf16 单卡 | 出视频、与 Bernini 上游 pipeline **画质 parity** |
| **P3 转 14B + MoE 跑通**(1–2d) | 转 transformer/transformer_2 → 挂 high/low LoRA 4 步;int8 版 | 40G 不 OOM、4 步稳定、提速达标 |
| **P4 编辑任务**(2–3d) | 按 P0 结论接 v2v/rv2v/r2v(VACE 复用 或 通道拼接)| v2v/rv2v 输出正确 |
| **P5 server + GPUStack**(1–2d) | 新 config profile、单实例、整卡独占、对象存储回传 | `/model/proxy/<id>/v1/tasks/video` 端到端 |

---

## 8. 风险 / 硬件(4×A100 40G,无 NVLink,鲲鹏 ARM)

- **fp32 权重**:转换时务必降 bf16,否则单专家 54G 直接爆盘+爆显存。
- **40G OOM**:14B bf16 双专家常驻大概率 OOM → **int8-triton**(转换同趟出);多卡通信无 NVLink+ARM 提不动 → 优先单卡 int8 + 多实例(参 [[infinitetalk-deep-test-verdict]])。
- **FA3 缺失**:Bernini 官方要 Hopper FA3,A100 回退 SDPA/sage_attn2(config 里已用 `sage_attn2`)。
- **依赖冲突**:Bernini 官方钉 `diffusers 0.35.2 / transformers 4.57.3 / torch2.7.1`,**但我们只借它的权重、用本仓推理栈**,所以**转换这一步**可能需要在独立 venv 跑(为读 Diffusers 权重),推理不受影响。
- **key 覆盖率**:若 §5 发现 Bernini 加了 Wan2.2 没有的模块,`wan_dit` 映射表要补规则(`converter.py:35+` `get_key_mapping_rules`),这是唯一可能要改转换器的地方。

---

## 9. 立即可执行的下一步
1. 在服务器跑 §5 的 4 条 dump 命令,把 `transformer/config.json`、`scheduler/config.json`、抽样 key 贴回 → 我据此定编辑注入机制、校准 config 数值。
2. 并行:转 1.3B(§2.1 单专家)+ 起 t2v 冒烟,先把"能出图"这条主干打通。

相关:[[scail2-bernini-weights-downloaded]] [[vace-verdict-lora-editor]] [[infinitetalk-deep-test-verdict]] [[gpustack-lightx2v-integration]] [[server-hardware-spec]]
</content>
