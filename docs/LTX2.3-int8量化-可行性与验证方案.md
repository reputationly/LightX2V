# LTX2.3 int8 量化 — 可行性与验证方案

> ⚠️ **实测结论(2026-06-25,已执行,失败):** 本方案已完整跑通并验证 —— int8 权重产出/加载/常驻都成功(DiT 常驻仅 20GB),但 **① 速度无提升(≈bf16,int8-triton 没吃到 INT8 tensor core 收益)② 画质崩(5/5 不可用,W8A8 全块量化、未跳敏感层)**。**结论:int8 对 LTX2.3 单条延迟目标无用。** 详见记忆 `ltx23-int8-no-speed-bad-quality`。下文保留作复现记录,命令均为**实际跑通**的版本。
>
> 状态:已执行的可行性记录。2026-06-25。
> 原目标:把 LTX2.3 22B 蒸馏 DiT 离线 int8 量化,使其 **~23GB 单卡 40GB 全驻留(免 offload)**,提升单条延迟且不降吞吐。

---

## 1. 背景与判断

- 当前模型:`ltx-2.3-22b-distilled-1.1.safetensors`(46GB bf16,**已是 8 步蒸馏**)。
- 单卡只能靠 block offload 跑(峰值 18.9GB,但权重每步从 CPU 流式,有开销);多卡因 46GB 装不下而 OOM(见 `视频生成平台-轻量自建方案设计.md` §6.1)。
- **关键**:LTX2 推理端已支持可插拔量化 mm(`transformer_weights.py`:`mm_type = config["dit_quant_scheme"]` → `MM_WEIGHT_REGISTER[mm_type]`),与 Wan int8 同一套机制。现仅用于 fp8,但 **int8 mm 是现成的**。
- **A100(sm_80)有 INT8 tensor core**(不像 fp8 要 sm_90)→ int8 GEMM 比 bf16 快,适合 A100。
- 46GB → **~23GB**(int8 1 byte/param;只量化 transformer 块,pre/post 保持 bf16)→ 单卡 40GB 全驻留,留足激活空间。

## 2. 现成的可用件 / 缺口

| 件 | 现状 |
|----|------|
| 推理端 int8 mm | ✅ 现成:`int8-q8f` / `int8-triton` / `int8-sgl` / `int8-torchao` / `int8-vllm`(`@MM_WEIGHT_REGISTER`) |
| Wan int8 参照 | ✅ `dit_quant_scheme:"int8-q8f"`(蒸馏配置)+ `dit_quantized_ckpt` |
| converter int8 | ✅ 实际 flag:`--quantized --bits 8 --linear_type int8 --non_linear_dtype torch.bfloat16`,可 `--device cpu`(注:`--linear_type` 不是 `--linear_dtype`;无 `--quant_scheme`) |
| converter 的 ltx2 key 映射 | ❌ `--model_type` 没列 ltx2;**但可保留默认 `--model_type wan_dit` + CLI override `--key-idx 4 --target-keys ... --ignore-keys none`**(无需改代码;注意是 `--target-keys` 连字符,不是 `--target_keys`) |
| 离线 int8 权重格式 | converter 产 weight+weight_scale → 推理 int8 mm 直接消费(**绕开** POC 那次"在线 auto-quant + offload → KeyError weight_scale"的坑) |

## 3. 关键风险(决定 go / no-go)

1. **画质**(头号风险):蒸馏 + int8 双重压缩,LTX 视频可能出现糊/色偏/雪花。**必须和 bf16 抽帧对比验证**。
2. **int8 mm 在 ARM A100 镜像里哪种能用**:`int8-q8f` 需 q8_kernels(你们 base 标注为"可选编译",**可能没装**);`int8-torchao` 镜像有 torchao 但启动有 `Unable to import torchao Tensor objects` 警告;`int8-sgl` 需 sgl-kernel;`int8-triton` 需 triton。**先确认哪种真能 import+跑**。
3. **LTX2 的 key 命名**:target_keys/key_idx 要对上 LTX2 DiT 的权重 key(注意力:to_q/to_k/to_v/to_out.0;FFN:net.0.proj/net.2),需先 dump key 确认。
4. **gemma 不受影响**:int8 只压 DiT;文本编码器(gemma)仍是另一段开销,但那是另一回事(且 server 模式下每请求很快)。

## 4. 分阶段验证方案

### Phase 0 — 镜像里确认可用的 int8 mm(秒级,先做)
在容器内逐个测哪种 int8 mm 能 import:
```
docker run --rm --gpus all <IMG> python -c "from lightx2v.common.ops.mm.<...> import *"  # 或直接看 MM_WEIGHT_REGISTER 注册时是否报缺依赖
```
判据:挑一个**能干净导入**的 scheme。**实测**:`int8-triton`(triton 3.6.0)和 `int8-vllm`(vllm 0.21.0)可用,`int8-torchao`(0.17.0,有 Tensor 子类警告)风险,`int8-q8f`/`int8-sgl` **MISS**(q8_kernels/sgl_kernel 没装)。**最终用 `int8-triton`。**

### Phase 1 — 用 converter 产出 LTX2 int8 权重(实际跑通的命令)
1. dump LTX2 DiT 权重 key(`model.diffusion_model.transformer_blocks.N.<模块>.…`),确认模块名在 dotted-key 的第 4 段 → `key_idx=4`。
2. 跑 converter(容器内,**必须带 `--gpus all`** 否则 import 期 `check_ai_device` 报错;量化负载仍走 `--device cpu`):
```bash
docker run --rm --gpus all -v /data:/data "$IMG" python /opt/LightX2V/tools/convert/converter.py \
  --source /data/models/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
  --output /data/models/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1-int8 \
  --output_name ltx2_int8 \
  --model_type wan_dit \
  --key-idx 4 \
  --target-keys attn1,attn2,audio_attn1,audio_attn2,audio_to_video_attn,video_to_audio_attn,ff,audio_ff \
  --ignore-keys none \
  --quantized --bits 8 --linear_type int8 --non_linear_dtype torch.bfloat16 \
  --device cpu
```
   产出:量化 1632 个 tensor(48块×34线性层),46GB→26GB(int8 17.7GB + 非量化 8.6GB),76 个分块 + `*.index.json`。**无需改 converter.py**(CLI override 足够)。

### Phase 2 — 配 config + 单卡跑
基于 `ltx2_3_distill_v11_hq.json` 改:
```
"dit_quantized": true,
"dit_quant_scheme": "int8-triton",     # Phase0 实测可用
"dit_quantized_ckpt": ".../ltx-2.3-22b-distilled-1.1-int8",
"skip_fp8_block_index": [],             # converter 量化了全部 48 块,推理端须一致
"cpu_offload": false,                   # DiT int8 ~20GB 全驻留
"gemma_cpu_offload": true               # 关键:否则 cpu_offload=false 会把 gemma 也推 GPU → OOM
```
单卡 server 起,同 prompt/seed 出片。**实测:加载成功、DiT 常驻 20GB,但速度=bf16、画质 5/5 崩 → 此路对延迟目标不通。**

### Phase 3 — 画质 + 性能验证(go/no-go)
- **画质**(关键):和 bf16 单卡产物**同 prompt/seed 抽帧逐一对比**(肉眼 + png 体积),重点看雪花/糊/色偏/细节丢失。
- **性能**:显存峰值(应 <40GB 全驻留、无 offload)、DiT 耗时(int8 应快于 bf16+offload)、加载耗时(23GB 应快于 46GB)。

### Phase 4 — 决策
- **画质 OK** → int8 定为 LTX 单卡主路(更快 + 吞吐不变);**多卡(TP/ulysses)因 23GB 也自动装得下**,要单条更快可再叠 TP-4(此时无需 TP-loader 改造)。
- **画质不行** → 试 per-channel int8 / 调 `skip_*_block_index` 保护敏感层 / 换 int8 mm scheme;仍不行再回 TP-loader 方案。

## 5. 与 TP-loader 方案的关系

| | int8 量化 | TP-loader 改造 |
|---|---|---|
| 单条延迟 | ✅ 直接快(免 offload + INT8 core) | ~1.7×(并行 DiT) |
| 吞吐 | ✅ 1 卡/任务,不降 | ❌ 4 卡/任务,降 |
| 多卡 | 附带解锁(23GB 装得下) | 这就是它本身 |
| 工程量 | 中(主要是验画质 + converter 适配) | 中高(重写分布式加载) |
| 主风险 | 画质 | 分布式正确性 |

**结论**:int8 应**优先于** TP-loader 验证 —— 它同时给"单卡更快 + 吞吐不降 + 多卡自动可行",且很可能让 TP-loader 变得不必要。**TP-loader 仅在 int8 画质不达标、且确需多卡进一步压延迟时才回头做。**

## 6. 待确认输入

- Phase 0 的镜像 int8 mm 测试结果(定 scheme)。
- LTX2 DiT 权重 key 结构(定 target_keys/key_idx)。
- converter 跑量化所需的临时磁盘/内存(46GB 读 + 23GB 写)。
