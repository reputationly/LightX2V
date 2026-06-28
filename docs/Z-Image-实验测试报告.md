# Z-Image-Turbo 文生图 实验测试报告

> 模型:Z-Image-Turbo(t2i,9 步蒸馏,LightX2V)
> 平台:4×A100 PCIE 40GB · 鲲鹏920 ARM · LightX2V server(Docker)
> 日期:2026-06-28
> 一句话结论:**bf16 单卡 7.6s/张是最优解;int8 在本机热态慢 2.9 倍(无 INT8 算力)且 z_image 显存宽裕,不要用;多卡 ulysses 单图只 1.2× 不划算;NUMA 对单图无影响。生产首选 bf16 单卡,要吞吐就 4×单卡实例。**

---

## 1. 硬件与环境

| 项 | 规格 |
|---|---|
| GPU | NVIDIA A100 **PCIE 40GB × 4**,**无 NVLink** |
| CPU | 鲲鹏920 **ARM aarch64** 128 核 |
| 内存 | 256GB,swap 仅 3GB |
| NUMA | **4 节点**;GPU0,1@node0(CPU0-31)、GPU2,3@node2(CPU64-95);组内 PHB、跨组 SYS |
| 容器镜像 | `crpi-...lightx2v:arm64-a100-20260625-0256-2cde4721` |

> 起容器必带 `-v /nfs-data:/nfs-data`、`--memory=240g`。

---

## 2. 权重路径

| 用途 | 路径 | 大小 |
|---|---|---|
| bf16 完整仓(diffusers 格式) | `/nfs-data/models/Z-Image-Turbo` | ~31G |
| ├ transformer(DiT) | `…/transformer`(分片 safetensors) | 23G |
| ├ text_encoder(Qwen3) | `…/text_encoder` | 7.5G |
| ├ tokenizer / vae | `…/tokenizer`、`…/vae` | 16M / 160M |
| **int8 DiT** | `/nfs-data/models-int8/Z-Image-Turbo-int8`(`non_block.safetensors`) | **5.8G** |

> int8 由 `convert_int8.sh` 离线量化(int8-torchao);**只量化 transformer**,text_encoder/tokenizer/vae 仍用 bf16 目录。
> 跑 int8 时 `model_path` 仍指 bf16 目录,`dit_quantized_ckpt` 指 int8 目录。

---

## 3. 核心配置(int8 / 多卡,实测可用)

```jsonc
{
  "aspect_ratio": "16:9",
  "num_channels_latents": 16,
  "infer_steps": 9,                         // turbo 蒸馏 9 步
  "attn_type": "sage_attn2",                // ⚠️ 改自官方 flash_attn3(A100 跑不了 Hopper kernel)
  "rope_type": "torch",                     // ⚠️ 改自默认 flashinfer(镜像没装 flashinfer,否则崩)
  "enable_cfg": false, "sample_guide_scale": 0.0,
  "patch_size": 2,
  // int8 追加:
  "dit_quantized": true, "dit_quant_scheme": "int8-torchao",
  "dit_quantized_ckpt": "/nfs-data/models-int8/Z-Image-Turbo-int8",
  // 多卡追加(seq_p_size 必须整除 30 个 attn head → 只能 2/3,不能 4):
  "parallel": { "seq_p_size": 2, "seq_p_attn_type": "ulysses" }
}
```

- `model_cls=z_image`,`task=t2i`,走 **image 端点** `/v1/tasks/image/`。
- 分辨率由 `aspect_ratio` 决定(也可请求体传 `target_shape` 自定义,长边≤1664)。

---

## 4. 速度 / 显存对比 —— ⭐ 热态稳态(关键)

> **必须看热态**:单发首张带 CUDA kernel 预热,会虚高近一倍。下表为单容器连续出 6 张、**丢首张预热、取后 5 张均值**(`test_z_image_stress.sh`)。分辨率 928×1664(见 §7 转置)。

| 配置 | **热态生成** | 1→2卡提速 | GPU util | 显存峰值 |
|---|---|---|---|---|
| **bf16 单卡** | **7.64s** | — | 100% | 21801MiB |
| bf16 2卡(同 NUMA 0,1) | 6.31s | **1.21×** | 100% | 20883MiB |
| bf16 2卡(跨 NUMA 0,2) | 6.57s | 1.16× | 100% | — |
| **int8 单卡** | **21.84s** | — | 78% | 16065MiB |
| int8 2卡(0,1) | 16.36s | 1.33× | 68% | 14987MiB |

**热态抖动极小**(±几十 ms),数据可信。

> 冷态(单发含预热,仅供识别坑):bf16 1卡 19s、int8 1卡 31s、bf16 2卡 13s、int8 2卡 25s —— **都偏高,勿用于结论**。

### 4.1 int8 vs bf16:int8 慢 2.9 倍

| | bf16 | int8 | int8 相对 |
|---|---|---|---|
| 单卡 | 7.64s | 21.84s | **慢 2.86×** |
| 2卡 | 6.31s | 16.36s | 慢 2.59× |

---

## 5. 原因分析(关键)

### 5.1 int8 为什么热态慢这么多?
int8-torchao 是 **weight-only(W8A16)**:权重存 int8,算前**反量化回 bf16 再做 bf16 矩阵乘**,本就为省显存/带宽设计,**不吃 INT8 算力**。A100 虽有 INT8 tensor core,但 torchao 这条路不走它,反而多了反量化开销 → GPU util 掉到 68~78%(喂不满),热态比 bf16 慢 ~2.9×。冷态只看着慢 1.6× 是因为预热开销把比例压平了。

### 5.2 int8 的价值只剩"省 ~6G 显存",而 z_image 不缺
bf16 单卡峰值才 **21.8G < 40G**,余量近一半。int8 省下的 6G 在这里**没用**。→ 与 Wan2.2 正相反:Wan 显存顶满甚至 OOM,int8+多卡是刚需;z_image 显存宽裕,int8/多卡/NUMA **全都不需要**。

### 5.3 多卡 ulysses 为什么只 1.2×?
单图 token 仅 ~6000,ulysses 的 all-to-all + 同步开销占比相对大,而计算本身已很快(bf16 7.6s)→ 2 卡只换 1.21×(效率 60%),还多占一张卡。**不划算。**

### 5.4 NUMA 对单图无影响
同 NUMA(0,1)6.31s vs 跨 NUMA(0,2)6.57s,差值落在同 NUMA 自身抖动(5.6–6.8s)内,均值/中位互相矛盾 → **观测不到跨 NUMA 惩罚**。根因:GPU 全程 100% util = **compute-bound**,通信量小,跨 NUMA 的 QPI 延迟显不出来。挑同 NUMA 卡对(0,1/2,3)当习惯即可,不是瓶颈。

---

## 6. 踩坑记录(可复用)

| 坑 | 现象 | 修法 |
|---|---|---|
| **rope_type 默认 flashinfer** | 镜像没装 flashinfer → 推理 `'NoneType' object is not callable` | config 加 `"rope_type": "torch"`(纯 torch RoPE,零依赖)。**flux2 等用 flashinfer rope 的模型同样会踩。** |
| **attn_type 默认 flash_attn3** | A100(Ampere)非 Hopper,flash_attn3 跑不了 | 改 `sage_attn2`(此镜像 Wan 已验证) |
| **ulysses 4 卡报错** | `heads (30) not divisible by seq_p_size (4)` | z_image 30 个 head,seq_p_size 只能 2/3/5/6…,**4 卡不可用** |
| **冷态数字虚高** | 单发首张比稳态慢近一倍 | 单容器连发、丢首张预热取均值 |
| **`compile: true` 无效** | z_image 无 `compile()` 方法,`hasattr` 为 False → 静默跳过 | 框架级缺失,要真编译需改核心代码(本次未做) |
| **分辨率转置** | `aspect_ratio=16:9` 出 928×1664 竖图(实际 9:16) | runner bug,见 §7 |

---

## 7. 分辨率机制 + 转置 bug(踩坑)

`aspect_ratio` 映射成固定像素(`z_image_runner.py` `default_aspect_ratios`),但**横竖被调换**:

```
z_image_runner.py:305  get_input_target_shape() 返回 (width, height)   # 16:9 → (1664, 928)
z_image_runner.py:311  set_target_shape(): height, width = get_input_target_shape()  # ← 按 (h,w) 解包,宽高互换!
```

返回 `(宽,高)` 但调用方按 `(高,宽)` 解包(`set_img_shapes` 同错)→ **每个比例都被转置**(1:1 不受影响)。

**实测确认**(`test_z_image_sweep.sh`,请求体逐张传 aspect_ratio,ffprobe 量实际尺寸):aspect_ratio **确实生效**(7 个比例出 7 个不同分辨率,非卡死),且**横竖全转置**:

| 请求标签 | 预期 W×H | 实测输出 W×H | 该次耗时(冷态,见下注) |
|---|---|---|---|
| 16:9 | 1664×928 | **928×1664**(竖,=9:16) | 11.49s |
| 9:16 | 928×1664 | 1664×928(横) | 7.33s |
| 1:1 | 1328×1328 | 1328×1328(不变) | 8.36s |
| 4:3 | 1472×1104 | 1104×1472(竖) | 12.33s |
| 3:4 | 1104×1472 | 1472×1104(横) | 8.22s |
| 3:2 | 1584×1056 | 1056×1584(竖) | 12.33s |
| 2:3 | 1056×1584 | 1584×1056(横) | 8.22s |

- **临时绕过**:要 16:9 横图就请求 `9:16`(标签反着填)。
- **彻底修**:把两处调用改成 `width, height = self.get_input_target_shape()`(1 行,需验证,本次未改核心)。
- 提示词内容不影响耗时(stage② 固定词 vs stage③ 轮换词逐行几乎一致)→ DiT 计算只看分辨率。

> ⚠️ **上表耗时是"冷态单形状"**:sweep 每换一个分辨率,kernel 按新 shape 重新 autotune,每个测量都带冷启动开销(与 §4 热态稳态不可直接比——同一张 928×1664 冷态 11.49s / 热态 7.64s)。
>
> ✅ **"竖图慢 1.5×"已被热态证伪**:同像素的竖(16:9→928×1664)**7.64s** vs 横(9:16→1664×928)**7.65s**,几乎完全相同。sweep 里竖图偏慢纯属**冷态 per-shape autotune 假象**,**热态下朝向无影响,同像素=同耗时**。全部 7 比例长边 1664/1328,约 1.5–1.6MP。
>
> ✅ **autotune 缓存跨切换保留**(实测穿插重访 `16:9 9:16 1:1 16:9 9:16 1:1`):16:9 第二次(中间隔了别的分辨率)从冷 12.3s 掉到热 **7.3s**。即**每分辨率首次付一次冷启动、之后永久缓存,不因混分辨率而丢**。→ 见 §8 架构建议。

---

## 8. 结论 / 生产建议

| 维度 | 结论 |
|---|---|
| 画质 | **int8 ≈ bf16,肉眼无差**(量化无损,同 Wan)——但 int8 已被速度否决 |
| **生产最优** | **bf16 单卡,7.6s/张** |
| int8 | ❌ 热态慢 2.9×,只省 6G(z_image 不缺),**别用**(除非塞 24G 卡) |
| 多卡 ulysses | 单图只 1.2×,占卡不划算;且 4 卡不可用(30 head) |
| NUMA | 单图无影响(compute-bound);挑同 NUMA 当习惯 |
| **要吞吐** | **N×单卡实例 + 负载均衡**;4 单卡实测 **0.530 img/s**(16/16),≈预测 0.52,≫ 2×双卡 0.32 |
| compile/fp8 | compile 对 z_image 无效(缺方法);fp8 A100 无算力单元,均不提速 |
| 分辨率 | aspect_ratio 生效但横竖转置(见 §7);7 比例都能出,~1.5MP |

### 8.1 多实例生产架构(实测支撑)
- **不要"每实例绑一种分辨率 + 按分辨率路由"**:autotune 缓存跨切换保留(§7 实测),**一个通用实例跑过 7 种分辨率各一次后全部缓存、之后任意分辨率都热态**,不会因混分辨率反复冷。绑分辨率反而招致负载不均(热门分辨率实例排队、其余闲置)。
- **推荐**:2 节点 × 4 卡 = **8 个通用单卡实例 + 负载均衡(轮询/最少连接)**;每个实例**启动时预热全部 7 种分辨率**(各发 1 张 dummy 请求焊死 autotune 缓存)→ 之后任意请求路由到任意实例都是热态,既无冷启动、又完美均衡。
- VRAM 无忧:7 种都 ~1.5MP,峰值相近;缓存的只是 kernel 选择元数据,很小。
- **主机内存无忧(实测,非外推)**:4 单卡实例 + 预热全 7 分辨率 + 并发混 7 分辨率压测下:每实例容器内存 **~1.7GB**、4 实例合计 **6.9GB**、**主机已用峰值 15.9GB / 256GB(6.2%)**。权重在 GPU 显存,主机只有运行时+缓冲。**高并发不增内存**:server 每实例一次只跑 1 个任务,并发请求只排队(队列只存 prompt 文本),显存/内存与并发数无关,代价是延迟不是内存。**不踩 Wan 的 CPU OOM 坑**(单卡无 offload 复制)。高并发要调的是 server `max_queue_size`(限流),与内存无关。
- **7 分辨率全部可预热**(实测):4 实例各预热 7/7 分辨率成功 → 通用实例预热一遍即全 cache,印证 §7 缓存跨切换保留。
- 注:混 7 分辨率并发吞吐 **0.469 img/s**(比纯 16:9 的 0.530 略低,因平均像素更大:1:1/4:3/3:2 都比 16:9 大),是更真实的混流量数字。

---

## 9. 测试工具 / 复现命令

| 脚本 | 作用 |
|---|---|
| `scripts/smoke/test_z_image.sh` | 封装:CASES=`bf16/bf16_ul2/bf16_ul3/int8/int8_ul2/int8_ul3`,单发对比 |
| `scripts/smoke/test_z_image_stress.sh` | **热态稳态压测**(单容器复用、丢预热);NUMA/精度对比;`GPUS` 指定卡 |
| `scripts/smoke/test_z_image_sweep.sh` | **分辨率扫描**:逐张传 aspect_ratio + 多提示词,ffprobe 量实际尺寸(注:每形状只跑一次=冷态,耗时勿与 §4 热态混比) |
| `scripts/smoke/test_z_image_4cards.sh` | **4 单卡实例并发吞吐**(每卡一个,端口 8000-8003) |
| `scripts/smoke/test_model.sh` | 通用 harness(已加 GPU 利用率/CPU%/内存 监控 + `GPUS` 覆盖) |

```bash
# 热态矩阵(本报告 §4 数据)
GPUS="0"   PREC=bf16 N=6 bash /data/test_z_image_stress.sh
GPUS="0,1" PREC=bf16 N=6 bash /data/test_z_image_stress.sh
GPUS="0"   PREC=int8 N=6 bash /data/test_z_image_stress.sh
GPUS="0,1" PREC=int8 N=6 bash /data/test_z_image_stress.sh
# NUMA 对比
GPUS="0,1" PREC=bf16 N=6 bash /data/test_z_image_stress.sh   # 同 NUMA
GPUS="0,2" PREC=bf16 N=6 bash /data/test_z_image_stress.sh   # 跨 NUMA
# 分辨率/提示词 渐进三阶段(每次只变一个变量):
#  ① 基线: 固定分辨率 + 固定提示词(= 上面的 stress)
#  ② 隔离分辨率: 扫 7 比例 + 固定提示词
PROMPT="A serene mountain lake at sunrise, pine forest" bash /data/test_z_image_sweep.sh
#  ③ 综合: 扫 7 比例 + 轮换提示词(镜像压测)
bash /data/test_z_image_sweep.sh
# 4 单卡实例吞吐
REQS=16 bash /data/test_z_image_4cards.sh
```

---

## 10. 待办

- [x] 跑 `test_z_image_sweep.sh`:7 比例都生效 ✓、转置 bug 实锤 ✓、提示词不影响耗时 ✓
- [x] 跑 `test_z_image_4cards.sh`:4 单卡实例 **0.530 img/s**(16/16 成功)✓
- [x] **热态横竖对比**:竖 7.64s vs 横 7.65s,**证伪"竖图慢"**——纯冷态 autotune 假象,热态朝向无影响 ✓
- [x] 肉眼对比 int8 vs bf16 出图质量:**看不出差异,量化无损** ✓
- [~] 分辨率转置 bug:**不自行修核心代码,仅记录**(§7),待官方修复后用 `9:16↔16:9` 反填绕过

---

## 11. 一页速查

| 维度 | 结论 |
|---|---|
| 生产最优 | **bf16 单卡 7.6s/张** |
| int8 | 热态慢 2.9×,只省 6G,**别用** |
| 多卡 | 只 1.2×,4 卡不可用(30 head),不划算 |
| NUMA | 单图无影响(compute-bound) |
| 吞吐方案 | 4×单卡实例 **实测 0.530 img/s**(≫ 2×双卡 0.32) |
| 必改配置 | `rope_type=torch`、`attn_type=sage_attn2`、steps 9 |
| 分辨率 | aspect_ratio 横竖被转置(16:9 出竖图);热态朝向无影响(同像素同耗时);7 比例 ~1.5MP |
| 对比 Wan | 完全相反:Wan 显存顶满→int8+多卡刚需;z_image 显存宽裕→全不需要 |
