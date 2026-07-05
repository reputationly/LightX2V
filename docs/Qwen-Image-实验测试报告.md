# Qwen-Image 系列(文生图 / 图生图 / Lightning 加速)实验测试报告

> 模型:Qwen-Image-2512(t2i 基座)、Qwen-Image-Edit-2511(i2i 编辑)、Qwen-Image-2512-Lightning 8 步蒸馏(离线合并)
> 平台:4×A100 PCIE 40GB · 鲲鹏920 ARM · LightX2V server(Docker),节点 `dev-gpustack-a100-0001`
> 日期:2026-07-05(qwen25vl开关+分辨率归因+shmem口径修正)
> 一句话结论:**Qwen 在 A100 上 `attn_type` 必须 `torch_sdpa`(`sage_attn2` 出纯黑图 NaN);DiT 靠 `cpu_offload=block` 塞进 40G(预取掩盖=免费),但**文本编码器必须 `qwen25vl_cpu_offload=false` 留 GPU**(否则在 ARM CPU 上拖慢)。生产最优 = bf16 单卡 merged8 + qwen25vl开关 **热态 17.0s**(旧配置28.2s);base 25步 108.9s。吞吐:多实例负载均衡,**单机最多 3 副本**(每实例 60.2G shmem,4副本>251G OOM),峰值副本随分辨率(16:9 3副本 0.164 / 1:1 2副本 0.114);int8/lazy_load/多卡想破上限全试全负(见 edit 报告 §11)。Lightning 加速必须离线合并,用 `dit_original_ckpt`。t2i/i2i/8步快版三任务均跑通。**

---

## 1. 硬件与环境

| 项 | 规格 |
|---|---|
| GPU | NVIDIA A100 **PCIE 40GB × 4**,无 NVLink |
| CPU | 鲲鹏920 **ARM aarch64** |
| 容器镜像 | `crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest` |
| 挂载 | `/data`(softlink → `/nfs-models/wuhanjisuan894`)、`/nfs-data` |

> 起容器必带:`--gpus '"device=0"'`、`--memory=240g`、`-v /data:/data -v /nfs-data:/nfs-data`、`-e PYTHONPATH=/opt/LightX2V`、**`--init`**(见 §5 僵尸容器坑)。

---

## 2. 权重路径

| 用途 | 路径 | 大小 |
|---|---|---|
| **Qwen-Image-2512**(t2i 基座,bf16 diffusers) | `/data/models/Qwen-Image-2512` | ~58G |
| ├ transformer(DiT,9 分片) | `…/transformer/diffusion_pytorch_model-0000X-of-00009.safetensors` | ~40G |
| ├ text_encoder(Qwen2.5-VL) / vae | `…/text_encoder`、`…/vae` | ~15G / — |
| **Qwen-Image-Edit-2511**(i2i 基座,bf16) | `/data/models/Qwen-Image-Edit-2511` | ~58G |
| └ transformer(DiT,5 分片) | `…/transformer/…-0000X-of-00005.safetensors` | — |
| **Lightning LoRA**(bf16 4/8 步) | `/data/models/loras/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-{4,8}steps-V1.0-bf16.safetensors` | 811M / 811M |
| ├ fp32 版 | `…-{4,8}steps-V1.0-fp32.safetensors` | 1.6G |
| └ fp8/int8 版(**A100 sm80 不可用**) | `…fp8_e4m3fn_scaled_*.safetensors`、`…int8_*.safetensors` | 各 20G |
| **⭐ 离线合并 DiT**(base+8步LoRA,bf16 单文件) | `/data/models/merged/qwen_2512_lightning_8step_merged.safetensors` | **38.1G** |

---

## 3. 核心配置(A100 实测可用)

### 3.1 文生图基座 Qwen-Image-2512(`qwen_2512_a100_base.json`)

```jsonc
{
  "infer_steps": 25,
  "aspect_ratio": "1:1",
  "prompt_template_encode": "<|im_start|>system\n...detailing the color, shape...<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
  "prompt_template_encode_start_idx": 34,
  "attn_type": "torch_sdpa",          // ⚠️⚠️ 不能用 sage_attn2(黑图),见 §5
  "rope_type": "torch",               // 镜像无 flashinfer
  "enable_cfg": true, "sample_guide_scale": 4.0,
  "cpu_offload": true,                // 58G base 单卡 40G 必须 offload
  "offload_granularity": "block"      // dense DiT 用 block 没问题(MoE 才必须 model)
}
```

### 3.2 图生图/编辑 Qwen-Image-Edit-2511(`qwen_edit_2511_a100_base.json`)

在 3.1 基础上改:`prompt_template_encode` 换成 edit 版模板 + `prompt_template_encode_start_idx: 64`;加 `"resize_mode": "adaptive"`、`"CONDITION_IMAGE_SIZE": 147456`、`"USE_IMAGE_ID_IN_PROMPT": true`。启动 `--task i2i --model_path /data/models/Qwen-Image-Edit-2511`;提交时请求体带 `"image_path": "<URL/base64/本地路径>"`。

### 3.3 Lightning 8 步快版(`qwen_2512_a100_lightning_merged.json`)

```jsonc
{
  "infer_steps": 8,
  "attn_type": "torch_sdpa", "rope_type": "torch",
  "enable_cfg": false, "sample_guide_scale": 4.0,   // 蒸馏关 CFG
  "cpu_offload": true, "offload_granularity": "block",
  "qwen25vl_cpu_offload": false,   // ★★ 文本编码器留 GPU:单图 28→17s、2副本吞吐 +48%(edit 探索反哺)
  "dit_original_ckpt": "/data/models/merged/qwen_2512_lightning_8step_merged.safetensors"
  // ★ 不写 lora_configs!DiT 用离线合并好的单文件;model_path 仍指 base(编码器/VAE 从那读)
}
```

- 三者都是 `model_cls=qwen_image`,走 **image 端点** `/v1/tasks/image/`。
- `model_path` 始终指 base 目录;`dit_original_ckpt`(单文件)优先于 `model_path/transformer`(glob 分片)。

---

## 4. 速度 / 显存 —— ⭐ 热态稳态压测(`test_qwen_image_stress.sh`)

> 方法同 Z-Image:单容器连发 6 张、**丢首张预热、取后 5 张均值** + GPU util 峰 + 显存峰。分辨率 1:1(1328×1328)。

| 配置 | 步数/CFG | 加载 | **热态稳态** | GPU util峰 | 显存峰 | 判定 |
|---|---|---|---|---|---|---|
| **bf16 单卡 merged8 + `qwen25vl_cpu_offload:false`** ⭐ | 8 无CFG | 176s | **17.0s** | 99% | 26.7G | ✅ **单卡最优(文本编码器留GPU)** |
| bf16 单卡 merged8(文本编码器offload到CPU) | 8 无CFG | 125s | 28.2s | 99% | 17.8G | ⚠️ 旧配置,慢 1.66× |
| **bf16 单卡 base** | 25+CFG(50前向) | 106s | **108.9s** | **100%** | 17.8G | ✅ 全质量基线 |
| int8 单卡(+offload) | 25+CFG | 91s | **218.8s** | 100% | 17.1G | ❌ 慢 2.0× |

> ⭐ **`qwen25vl_cpu_offload:false` 对 t2i 也有效(edit 探索反哺)**:文本编码器(Qwen2.5-VL,~15G)默认随 `cpu_offload` 一起 offload 到 ARM CPU,拖慢;留 GPU 后单图 **28.2→17.0s(1.66×)**,显存 17.8→26.7G(仍<40G)。**t2i 生产配置应加此开关。**

**关键发现(推翻了直觉):bf16 单卡 GPU util 100% + 显存峰仅 17.8G → 是 compute-bound,不是 offload-bound。**
- block offload 是**预取流水线**:算当前 block 时后台预取下一个,PCIe 传输被计算完全掩盖 → offload 在这里**几乎不是速度惩罚,只是让 58G 塞进单卡 40G 的必要手段**。
- 108.9s 就是实打实算 25步×CFG(=50 次大 DiT 前向 @ 1.7MP)。merged8 8步无CFG(=8 前向)→ 28.2s,**3.86× 提速**,印证"步数是主因"。
- **显存峰仅 17.8G ≪ 40G → 显存根本不是约束**,这是下面"int8 无价值"的根因。

### 4.1 多卡:三条路全部实测判死 ❌

Qwen 24 head(可被 2/3/4/6/8 整除),但在 40G A100 + 共享 SFS 上多卡全废:

| 路线 | 结果(实测) | 根因 |
|---|---|---|
| 2卡 TP + offload | **能跑但慢 1.43×**:稳态 **155.9s** vs 单卡 108.9s | TP 每层 all-reduce 通信 + 双卡各自 offload 传输,盖过拆分计算收益(单卡已 100% util) |
| 2卡 TP(`tensor_p`)无 offload | **启动 OOM**(单卡占 38.86G) | TP 建了 device_mesh 但**不切权重显存**,每 rank 仍驻完整 38G DiT |
| 2卡 ulysses + offload | 加载 71s 但**每张生成崩** | `tensor a(6889) vs b(6890)`:1:1 latent token=83²=6889 奇数,切不开 seq_p=2 |
| 4卡 ulysses + offload | 冷缓存 >21min 未起;**热缓存重试仍整机死机**(SSH 断) | 4 rank offload init 挤爆 ARM CPU,与 4 副本死机同因 |
| int8 单卡无 offload | 加载过、**生成 OOM** | int8 反量化回 bf16 临时 buffer + 编码器常驻 + CFG 双份 → 顶满 40G |

→ **结论:Qwen 多卡在此硬件全部有害无益。TP 慢 1.43×(实测)、ulysses 奇数 token 崩、4卡死机、无 offload 必 OOM。**

> **统一硬约束(实测 4+ 次)**:任何"4 路"Qwen offload 负载——**4 副本 或 4 卡并行**——在加载/初始化阶段就把 **ARM CPU 挤爆 → 整机死机(SSH 断,需强制重启)**。根因是 offload 权重处理是 CPU 重活,4 份同时干超出 ARM CPU 承载。**必须错峰启动,单机并发度 ≤ 2-3。** 与并行方式无关,是 offload 引擎在此硬件的天花板。

### 4.2 int8:显存不缺时纯亏(与 Z-Image 同向)

- int8+offload 218.8s vs bf16+offload 108.9s = **慢 2.0×**(int8-torchao 是 weight-only,算前反量化回 bf16,不吃 INT8 算力,反增开销)。
- int8 想"缩小权重→单卡不 offload→快"的算盘破产:**无 offload 生成即 OOM**(反量化 buffer),**躲不开 offload**。
- 显存峰 17.1G ≈ bf16 的 17.8G,**int8 连显存都没省到**(offload 下本就只驻几个 block)。
- (注:int8 用原版 Qwen-Image、bf16 基线用 2512,架构同(60层/24head/25步),速度比可比;未单跑 bf16 原版精确配对,~2× 结论不受影响。)

### 4.3 多副本吞吐(`test_qwen_image_4cards.sh`,merged8 1:1)⭐

> ⭐ **重大更新(edit 探索反哺)**:加 `qwen25vl_cpu_offload:false`(文本编码器留 GPU,不抢 ARM CPU)后,t2i 单图 28.2s→**17.0s**、吞吐大幅提升。下表分"旧(文本编码器 offload 到 CPU)"和"新(留 GPU)"。

| 副本 | 旧 吞吐(文本编码器CPU) | **新 吞吐(qwen25vl=false)** | 新/单图 | 真实内存(Shmem) |
|---|---|---|---|---|
| 1 | 0.046 | **0.057** | 17.5s | ~60G |
| **2** | 0.077(封顶,1.67×) | **0.114(2.0× 近线性)⭐峰值** | 17.5s | ~120G |
| 3 | 0.068(负优化) | **0.103(仍<2副本,略回退)** | 29s | ~180G |
| 4 | 整机死机 | 第4实例 OOM(60G×4>251G),但**整机不死**(qwen25vl 让4路init不再挤爆CPU) | — | — |

**核心结论(qwen25vl 前后都成立):Qwen t2i 单机吞吐峰值副本数随分辨率变——1:1 峰值 2 副本(0.114),16:9 峰值 3 副本(0.164);4 副本内存墙(60G shmem×4>251G)。**
- **qwen25vl 双重提升**:文本编码器从 ARM CPU 挪到 GPU → 单图 28.2→**17s(1.66×)**、2副本 0.077→**0.114(+48%)**。根因:旧配置文本编码器在 CPU,2 副本抢 ARM CPU → 打折;新配置不抢 → 2 副本近线性。
- **峰值副本数随分辨率变(实测隔离,非任务差异)**:
  - **@ 1:1(1.76MP)**:2 副本 0.114 峰值,**3 副本回退 0.103**(每图 17→29s)。
  - **@ 16:9(1.5MP)**:**3 副本近线性 0.164**(2.9× 单实例 0.055),不回退。
  - **根因是分辨率不是任务**:控制变量测 t2i@16:9 得 0.164(和 edit@16:9 0.126 同为近线性),证明"t2i峰值2 vs edit峰值3"纯是**分辨率(激活大小)差异**——大图激活大→DiT offload 每步搬运多→3 实例并发抢 PCIe/内存带宽更早撞墙;非 t2i/i2i 任务本身区别。同分辨率下两者行为一致。
- **shmem 精确实测**(定时采集器抓 3 副本错峰启动):Shmem 阶梯 **60.2 → 120.4 → 180.7G**(每实例 **60.2G**),available 掉到 **49.5G** → 第 4 个的 60G 塞不下 = 全局 OOM。
- **内存真相(修正)**:每实例 offload DiT 占 **~60G shmem(CUDA pinned,不可共享)**;3副本180G、4副本240G+系统>251G → 4 副本全局 OOM。**注:早期用 `free` used 列报的 28/47/64G 是错的(漏 shmem),真实是 60G/实例**;量真实内存看 `/proc/meminfo` 的 Shmem。
- **必须 `--memory≥100g` + 错峰启动**(见 §5)。
- 与 z-image 天壤之别:z-image 不 offload、每实例才 ~1.7G,4 副本线性到 0.53 img/s;**t2i 被 offload 拖累(60G shmem/实例),单机最多 3 副本(16:9 0.164 / 1:1 峰值2副本0.114)**。
- **破 2 副本上限的路全试全负**(见 edit 报告 §11):int8(慢2×)、lazy_load(慢6×)、多卡(全废)——要真 4 卡满速只能加内存条或改引擎共享 pinned 权重。

---

## 5. 踩坑记录 ⭐(Qwen 系列专属,新模型大概率复用)

| 坑 | 现象 | 解 |
|---|---|---|
| **⚠️ sage_attn2 出黑图** | Qwen-Image(head_dim=128 + `enable_cfg=true`)用 `sage_attn2`(INT8 量化注意力)→ NaN → **纯黑图**,产物仅 5.1KB,`completed` 假绿 | **`attn_type` 改 `torch_sdpa`**(纯 torch SDPA,Ampere 100% 正确、零依赖)。实测 torch_sdpa 2.4M 正常 / sage_attn2 5.1K 全黑。**判黑图:产物 <50KB 即失败** |
| **运行时 LoRA 合并卡死** | 配 `lora_configs` + `cpu_offload=true` → 加载完 9 分片后逐 block 在 CPU 上合并,CPU 291% 死转、10+min 无新日志、永不 ready | **离线预合并**(见 §6),启动改用 `dit_original_ckpt` 加载合并好的单文件,无运行时合并 |
| **官方 lora_merger.py 0 命中** | `tools/extract/lora_merger.py` 只认 `.lora_up/.lora_down` 老格式,对 Qwen/diffusers 格式(`.lora_A/.lora_B`)一个都不匹配 | 写 `scripts/merge_qwen_lora.py`,**复用 `lightx2v.utils.lora_loader.LoRALoader`**(支持全格式 + `transformer_blocks.N` key 映射),与运行时 `apply_lora` 完全同语义 |
| **僵尸容器杀不掉** | `docker rm -f qwen-smoke` 报 `PID ... is zombie` / `did not receive an exit event` | `docker kill <n>; sleep 2; docker rm -f <n>`;仍残留 `systemctl restart docker`。起容器**加 `--init`** 从根上避免 |
| **合并容器 import 崩** | `import lightx2v` 强制 `init_ai_device: cuda`,不挂卡报 `AI device 'cuda' is not available` | 离线合并容器也要 `--gpus '"device=0"'`(合并本身仍 CPU-only,卡只为过 import) |
| **rope flashinfer 缺失** | `rope_type` 默认 flashinfer,镜像没装 → `'NoneType' object is not callable` | `rope_type: torch` |
| **fp8/int8 LoRA 不可用** | A100 sm80 无 fp8/nvfp4 tensor core | 只用 `*-bf16.safetensors` 版 LoRA |
| **⚠️ 文本编码器默认被 offload 到 CPU 拖慢** | `cpu_offload:true` 会连 Qwen2.5-VL 文本编码器(~15G)一起 offload 到 ARM CPU → 拖慢(t2i 单图 28.2s、多副本抢 CPU 卡在 2 副本) | 配置加 **`qwen25vl_cpu_offload:false`**(文本编码器留 GPU,才 15G,显存够)→ 单图 28.2→**17.0s**、2副本吞吐 +48%。edit i2i 更严重(不加 → 文本编码 7min,见 edit 报告 §6) |
| **⚠️ `free` used 列漏 shmem,严重低估内存** | offload DiT 存 **CUDA pinned 内存(计入 Shmem)**,`free` 的 used 列不计 shmem → 3 副本实报 19G、**实为 180G** | 量真实内存看 `/proc/meminfo` 的 **Shmem** + `MemAvailable`;每 offload 实例 **~60.2G shmem**,单机(251G)副本上限 3 |

---

## 6. 离线 LoRA 合并流程(一次性,永久可用)

脚本 `scripts/merge_qwen_lora.py`:读 base 分片 → 复用 `LoRALoader.apply_lora`(与运行时同语义)→ 存单文件 bf16。

```bash
# CPU-only(卡只为过 import),ARM bf16 matmul 偏慢,本次 720 层 ~38min,读+写 ~40G 各一次
docker run --rm --gpus '"device=0"' --memory=200g \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  "$IMG" python /data/merge_qwen_lora.py \
    --base-transformer /data/models/Qwen-Image-2512/transformer \
    --lora /data/models/loras/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors \
    --out  /data/models/merged/qwen_2512_lightning_8step_merged.safetensors \
    --strength 1.0
```

- 关键校验:日志 `[map] lora pairs=720 diffs=0 | 命中 base=720`(命中数=pairs 数才对),`[merge] applied=720`,`[done] -> ... (38.1 GB)`。
- ARM CPU bf16 慢是一次性代价;可加 `--compute-dtype fp32`(更准、RAM 翻倍)。
- 同法可产 Edit-2511-Lightning、4 步版等的合并权重。

---

## 7. 结论 / 生产建议

| 维度 | 结论(实测) |
|---|---|
| **生产最优(延迟)** | **bf16 单卡 merged8(8步)+ `qwen25vl_cpu_offload:false` = 17.0s/张**(旧配置文本编码器offloadCPU 28.2s);要全质量则 base 25步 108.9s |
| 瓶颈性质 | **compute-bound**(util 100%),DiT offload 被预取掩盖;但**文本编码器别 offload**(qwen25vl 开关),否则在 ARM CPU 上慢 |
| 显存 | merged8 峰值 26.7G(含文本编码器留 GPU 15G),仍 <40G |
| 多卡 | ❌ 全废:4卡冷启动>21min / TP不切显存OOM / ulysses奇数token崩 |
| int8 | ❌ 慢 2.0×、显存没省、无offload即OOM → 有害无益(同 z-image) |
| **吞吐方案** | 多实例负载均衡 + `qwen25vl:false`;**单机最多 3 副本**(每实例 **60.2G shmem**,4副本240G>251G OOM);峰值副本随分辨率:**16:9 3副本 0.164** / 1:1 2副本 0.114;每实例 `--memory≥100g` + **错峰启动** |
| **真实内存口径** | 每实例 **60.2G shmem**(CUDA pinned,不可共享);`free` used 列**漏 shmem**,量内存看 `/proc/meminfo` 的 Shmem |
| 必改配置 | `attn_type=torch_sdpa`(非 sage_attn2)、`rope_type=torch`、`cpu_offload=true/block`、**`qwen25vl_cpu_offload=false`** |

## 8. 复现命令(`scripts/smoke/test_qwen_image_stress.sh`)

```bash
# 单卡热态(报告 §4 数据)
MODE=base    GPUS=0 bash /data/test_qwen_image_stress.sh   # bf16 25步 108.9s
MODE=merged8 GPUS=0 bash /data/test_qwen_image_stress.sh   # bf16 8步  28.2s
MODE=int8orig GPUS=0 OFFLOAD=1 bash /data/test_qwen_image_stress.sh   # int8 218.8s
# 多卡(均失败,留作证据;READY_TO 调大超时,崩溃会秒报)
MODE=base GPUS=0,1   PTYPE=ulysses            READY_TO=1800 bash /data/test_qwen_image_stress.sh  # token 奇数崩
MODE=base GPUS=0,1   PTYPE=tp OFFLOAD=0        bash /data/test_qwen_image_stress.sh               # OOM
MODE=base GPUS=0,1,2,3 PTYPE=tp OFFLOAD=0 READY_TO=3000 bash /data/test_qwen_image_stress.sh
# env: MODE(base|merged8|edit|bf16orig|int8orig) GPUS PTYPE(ulysses|tp) OFFLOAD(1/0) N ASPECT STEPS QUANT_CKPT READY_TO KEEP
```

## 9. 待办 / 后续

- [ ] 接回 GPUStack(Phase A Custom 后端 + 薄透传),经 new-api 端到端(生图链路 + OBS 落盘)。
- [ ] Edit-2511 的 Lightning 加速(同 §6 离线合并 `Qwen-Image-Edit-2511-Lightning`)。
- [ ] 多副本吞吐实测(N×单卡 merged8,仿 z-image `test_z_image_4cards.sh`)出 img/s。
- [x] ~~分辨率扫描(转置 bug)~~:实测 6 比例宽高全正确,**无 z-image 的横竖转置 bug**,aspect_ratio 可直接用。像素:1:1=1328²、16:9=1664×928、3:2=1584×1056、4:3=1472×1104、9:16=928×1664、3:4=1104×1472(均 ~1.5-1.7MP)。
- [x] ~~autotune 缓存跨分辨率保留~~:实测序列 `1:1→16:9→9:16→1:1→16:9→9:16`,**重访 1:1 = 16s(热),不因中间切别的分辨率而重新冷启**(同 z-image)。首图最冷 20s(全局CUDA预热),per-shape 冷启很小(~1-2s)。**生产:通用实例池,启动预热各分辨率一遍,之后任意分辨率热态;不用按分辨率绑实例。**
- [x] ~~多卡加速~~:实测三路全废(§4.1),不再追。
- [x] ~~int8 省显存换速度~~:实测慢 2.0× 且躲不开 offload(§4.2),弃用。
