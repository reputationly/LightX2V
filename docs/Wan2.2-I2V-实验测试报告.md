# Wan2.2-I2V-A14B 实验测试报告

> 模型:Wan2.2-I2V-A14B 4-step 蒸馏(LightX2V)
> 平台:4×A100 PCIE 40GB · 鲲鹏920 ARM · LightX2V server
> 日期:2026-06-28
> 一句话结论:**int8 画质无损、int8 4卡是生产最优解;bf16 多卡在本机不可行;长视频走 480p,720p 最长 10s。**

---

## 1. 硬件与环境

| 项 | 规格 |
|---|---|
| GPU | NVIDIA A100 **PCIE 40GB × 4**,**无 NVLink**(走 PCIe) |
| CPU | 鲲鹏920 **ARM aarch64** 128 核 |
| 内存 | **256GB**,**swap 仅 3GB**(内存压力大易 OOM/卡死) |
| 本地盘 | 无,权重在 NFS(华为云 SFS,软链 `/data/models → /nfs-data`) |
| 容器镜像 | `crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721` |

> ⚠️ 起容器必须带 `-v /nfs-data:/nfs-data`(权重在 NFS)。

---

## 2. 权重路径

| 用途 | 路径 | 大小 |
|---|---|---|
| 基座(VAE / T5) | `/nfs-data/models/Wan-AI/Wan2.2-T2V-A14B` | — |
| 蒸馏 DiT(bf16)high | `/nfs-data/models/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors` | 57G |
| 蒸馏 DiT(bf16)low | `/nfs-data/models/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors` | 57G |
| **int8 DiT** high | `/nfs-data/models-int8/Wan2.2-I2V-720p-int8/high_noise/` | ~14G |
| **int8 DiT** low | `/nfs-data/models-int8/Wan2.2-I2V-720p-int8/low_noise/` | ~14G |

> Wan2.2 是 **MoE 双专家**:high_noise + low_noise,推理按 `boundary_step_index` 切换。
> int8 由 `scripts/convert_int8.sh` 离线量化(int8-torchao,块格式,敏感层 embedder/norm 自动留 bf16)。

---

## 3. 脚本与配套

| 脚本 | 作用 |
|---|---|
| `scripts/smoke/test_model.sh` | 通用 harness:起容器(单卡 python / 多卡 torchrun)、等 health、提交、轮询、记录加载/生成耗时+峰值显存。带 `--memory=240g` 防宿主 OOM。支持 `LAST_FRAME`/`NEG_PROMPT`/`RESIZE_MODE` |
| `scripts/smoke/test_wan_i2v.sh` | i2v/flf2v 测试:生成配置 + 调 harness。开关 `RES`(480/720)、`CASES`、`TASK`(i2v/flf2v)、`LAST_FRAME` |
| `scripts/smoke/test_wan_i2v_stress.sh` | **时长压测(单容器复用)**:起一次 int8 4卡 server,循环不同 `target_video_length` 提交,找显存/画质上限 |
| `scripts/convert_int8.sh` | 批量 bf16→int8(docker 内跑 converter) |
| `scripts/download_models.sh` | ModelScope 优先下载模型 |

**官方 flf2v 参考**:`scripts/wan22/run_wan22_moe_flf2v.sh`、`examples/wan/wan_flf2v.py`、配置 `configs/wan22/wan_distill_moe_flf2v_int8.json`。

---

## 4. 核心配置(int8 4卡,实测可用)

```jsonc
{
  "infer_steps": 4,
  "target_video_length": 81,            // 时长=帧数/16fps;请求体可覆盖
  "target_height": 720, "target_width": 1280,  // 720p 面积,仅 resize_mode=null 时生效(见 §8)
  "sample_guide_scale": [3.5, 3.5], "sample_shift": 5.0, "enable_cfg": false,
  "boundary_step_index": 2,             // MoE high/low 专家切换点
  "denoising_step_list": [1000, 750, 500, 250],
  "self_attn_1_type": "sage_attn2", "cross_attn_1_type": "sage_attn2", "cross_attn_2_type": "sage_attn2",
  "cpu_offload": false,                 // int8 28G 单卡装得下,不用 offload
  "t5_cpu_offload": true, "vae_cpu_offload": false,
  "use_image_encoder": false, "rope_type": "torch",
  "dit_quantized": true, "dit_quant_scheme": "int8-torchao",
  "high_noise_quantized_ckpt": "/nfs-data/models-int8/Wan2.2-I2V-720p-int8/high_noise",
  "low_noise_quantized_ckpt":  "/nfs-data/models-int8/Wan2.2-I2V-720p-int8/low_noise",
  "parallel": { "seq_p_size": 4, "seq_p_attn_type": "ulysses" }   // 4卡;单卡删掉本行
}
```

- **bf16 单卡** 需改:`cpu_offload:true` + `offload_granularity:"model"`(MoE 必须 "model" 粒度,一次换一个 28.5G 专家;用 "block" 会黑屏),并把 quantized 换成 `high/low_noise_original_ckpt` 指向 bf16 文件。
- `model_cls=wan2.2_moe_distill`,`task=i2v`(或 `flf2v`)。

---

## 5. 画质 / 速度 / 显存对比

### 5.1 480×832(81帧,5s)

| 用例 | 加载 | 生成 | 峰值显存 | 状态 |
|---|---|---|---|---|
| bf16 单卡(offload) | 181-250s | 56-61s | 36.4G | ✅ |
| int8 单卡 | 110-160s | 97s | 37.1G | ✅ |
| **int8 4卡 ulysses** | 90s | **36s** | 32.9G | ✅ **最优** |
| bf16 4卡 | — | — | — | ❌ **CPU OOM**(见 §6.2) |

### 5.2 832×1104(720p,81帧,5s)

| 用例 | 生成 | 峰值显存 | 状态 |
|---|---|---|---|
| bf16 单卡(offload) | 183s | 40.3G | ✅ 险过(死贴 40G,无余量) |
| int8 单卡 | — | OOM | ❌ |
| **int8 4卡 ulysses** | 87s | 35.3G | ✅ **生产最优** |

### 5.3 画质结论

**int8 ≈ bf16,肉眼无差,量化无损可用。** 与 LTX2.3(int8 画质崩)完全相反 —— Wan 是 int8 一等公民。

---

## 6. 原因分析(关键)

### 6.1 int8 单卡为什么比 bf16 单卡还慢?
int8-torchao 在 **A100/ARM 上不走 INT8 tensor core**,矩阵乘仍按 bf16 算,还多了反量化开销 → int8 单卡(97s)反而比 bf16 单卡(56s)慢。
**int8 的价值是显存(28G 单卡装得下,免 offload),不是速度;提速靠多卡。**

### 6.2 bf16 4卡为什么必 CPU OOM?
ulysses 每个 rank **复制一份完整 bf16 模型到 CPU**(供 offload 流式):`4 rank ×(57G 模型 + 11G T5)≈ 276G > 256G 内存` → 加载阶段就 `SIGKILL(-9)`。
harness 的 `--memory=240g` cgroup 上限把容器干净杀掉,**宿主不挂**(双保险)。→ **bf16 想多卡,必须先量化成 int8。**

### 6.3 高分辨率下 bf16+offload 反而比 int8 更扛 OOM
显存 = **权重 + 激活**。`cpu_offload` 搬得走权重,**搬不走激活**(前向计算时激活必须在 GPU)。
- **bf16+offload(model 粒度)**:GPU 上一次只放 1 个专家(28.5G),720p 大激活(~12G)还塞得下 → 40.3G 险过。
- **int8(无 offload)**:两个专家都常驻(28G)+ 反量化 buffer → 挤爆 → OOM。

所以 720p 单卡:**bf16 险过、int8 OOM**。但两者都没实用余量,生产仍用 **int8 4卡**。

### 6.4 多卡为什么能线性提速 / 扛更高分辨率?
ulysses **按序列(token = H×W×帧)切分到各卡**,每卡只算 1/N 的激活 → 既线性提速,又让**激活 ÷ 卡数**,这正是会随分辨率/帧数膨胀的部分。所以高分辨率/长视频靠多卡,不是靠 offload。

---

## 7. 时长压测(int8 4卡)

> 时长 = 帧数 / 16fps;帧数须 = **4n+1**(VAE 时间步长 4)。

### 7.1 480p(480×832)

| 帧数 | 时长 | 生成 | 峰值 | 状态 |
|---|---|---|---|---|
| 81 | 5.06s | 35s | 32.9G | ✅ |
| 121 | 7.56s | 56s | 33.8G | ✅ |
| 161 | 10.06s | 71s | 34.7G | ✅ |
| 201 | 12.56s | 92s | 35.5G | ✅ |
| 241 | 15.06s | 112s | 36.3G | ✅ |

**显存非瓶颈**:帧数翻 3 倍只涨 3.4G(~21MiB/帧,ulysses 切序列 ÷4)。按斜率估 **~25s 才 OOM**。生成 ~0.48s/帧。

### 7.2 720p(832×1104)

| 帧数 | 时长 | 生成 | 峰值 | 状态 |
|---|---|---|---|---|
| 81 | 5.06s | 87s | 35.3G | ✅ |
| 121 | 7.56s | 132s | 37.9G | ✅ |
| **161** | **10.06s** | **193s** | **40.3G** | ✅ 顶满(剩 ~0.16G) |
| 201 | 12.56s | — | OOM | ❌ |
| 241 | 15.06s | — | OOM | ❌ |

**720p 天花板 = 161 帧(10s)**,~62MiB/帧(480p 的 3×)。生产建议 **≤121 帧(7.5s/37.9G 有余量)**。

> OOM 时 server **优雅 failed 不崩**(health 保持 200),单容器压测可继续后续档。

### 7.3 时长结论

| 需求 | 推荐 |
|---|---|
| 长视频(>10s) | **480p**,实测 15s、估计 ~25s,且生成快 |
| 720p 高清 | 最长 10s(161帧);稳妥 ≤7.5s |

✅ **画质已实测确认**:480p 15s(241帧)、720p 10s(161帧)**肉眼均正常**,无漂移/重复/变糊。蒸馏模型虽在 81 帧训练,外推到测试上限画质仍稳 → **时长真实上限 = 显存上限**(画质没有先到)。

---

## 8. 分辨率机制(踩坑记录)

server 路径下 wan i2v 的分辨率,由 `run_vae_encoder`(`wan_runner.py:504`)决定,只看 `resize_mode` 是否为 None:

- **非 None**(请求默认 `resize_mode="adaptive"`,schema.py:83,base.py:172 无条件塞)→ 走预处理 = **480×832**
- **None** → 走 `max_area = config.target_height × target_width` → **按图比例算出 832×1104**

**修法:匹配在线版 720p,在请求体传 `resize_mode: null`** + config `target_height:720/target_width:1280`。
- ❌ `target_shape` 字段对 wan i2v **无效**(`get_latent_shape_with_target_hw` 全仓无人调用)。
- ❌ config 里的 `fixed_area`/`resize_mode` 对 i2v 也无效(被请求默认覆盖)。
- 脚本封装:`RES=720` 自动传 `resize_mode=null`。

---

## 9. 首尾帧模式(flf2v)

**同一个 int8/bf16 权重直接支持**,不用换模型(`configs/model_pipeline.json` 明确列了 `flf2v → Wan2.2-I2V-A14B-distill`)。

与 i2v 仅两处不同:
1. `task: flf2v`(不是 i2v)
2. 请求体多传 `last_frame_path`(尾帧图)

官方蓝鸟示例:
- 首帧 `assets/inputs/imgs/flf2v_input_first_frame-fs8.png`(蓝鸟蹲地)
- 尾帧 `assets/inputs/imgs/flf2v_input_last_frame-fs8.png`(蓝鸟展翅飞天)
- 正向(英文)+ 负向(中文),见 `examples/wan/wan_flf2v.py:42-43`

脚本封装:`TASK=flf2v LAST_FRAME=<尾帧> bash test_wan_i2v.sh`。

---

## 10. 测试方法 / 复现命令

> 前置:`scp scripts/smoke/test_*.sh edt-vpn:/data/`,起容器前 `docker rm -f $(docker ps -aq --filter name=wan-i2v)`。

```bash
# 480p 三连(bf16 / int8 单卡 / int8 4卡)
SEED=504166 IMAGE=/opt/LightX2V/assets/inputs/imgs/img_0.jpg \
PROMPT="..." bash /data/test_wan_i2v.sh

# 720p int8 4卡(匹配在线版 832×1104)
RES=720 SEED=504166 IMAGE=... PROMPT="..." bash /data/test_wan_i2v.sh

# 720p 全档(含单卡,验证 int8 单卡 OOM / bf16 单卡险过)
RES=720 CASES="bf16 int8 int8_ul4" SEED=504166 IMAGE=... PROMPT="..." bash /data/test_wan_i2v.sh

# 时长压测(单容器复用)
bash /data/test_wan_i2v_stress.sh                 # 480p,扫 81~241
RES=720 bash /data/test_wan_i2v_stress.sh         # 720p
FRAMES_LIST="81 161 241 321" bash /data/test_wan_i2v_stress.sh

# 首尾帧(flf2v,蓝鸟示例)
RES=720 TASK=flf2v SEED=343632 \
IMAGE=/opt/LightX2V/assets/inputs/imgs/flf2v_input_first_frame-fs8.png \
LAST_FRAME=/opt/LightX2V/assets/inputs/imgs/flf2v_input_last_frame-fs8.png \
PROMPT="CG animation style, a small blue bird takes off ..." \
bash /data/test_wan_i2v.sh
```

**产物**:`/data/outputs/wan_i2v_<时间戳>/` 或 `wan_i2v_stress_<res>/`,每轮独立时间戳目录,不混。

---

## 11. 一页速查

| 维度 | 结论 |
|---|---|
| 画质 | int8 ≈ bf16,无损 |
| 生产最优 | **int8 4卡 ulysses**(480p 36s / 720p 87s) |
| int8 单卡 | 慢(97s,无 tensor core),价值=显存 |
| bf16 单卡 | 需 offload+model 粒度;720p 险过但巨慢 |
| bf16 多卡 | ❌ 不可行(CPU OOM,256G 装不下 4×复制) |
| 480p 时长 | ≤15s 实测 / ~25s 估计,显存非瓶颈 |
| 720p 时长 | ≤10s(161帧顶满),稳妥 ≤7.5s |
| 720p 开关 | 请求体 `resize_mode: null` |
| 首尾帧 | `task=flf2v` + `last_frame_path`,同权重 |
| 长视频画质 | ✅ 480p 15s / 720p 10s 肉眼正常,时长上限 = 显存上限 |
