# Wan2.2-S2V 实验测试报告(音频驱动数字人)

> 日期:2026-07-09 · 环境:gpustack 集群计算节点(dev-gpustack-a100-0010, A100 PCIE 40G)
> 镜像:`lightx2v:arm64-a100-latest`(torch 2.11) · harness:`scripts/smoke/test_model.sh` + `run_batch.sh`
> 权重:`/nfs-data/models/Wan2.2-S2V-14B`(bf16 46G, 官方唯一形态, 无蒸馏版)

## 结论先行

1. **S2V 在 A100 上必须打补丁才能跑**:上游为"与官方数值对齐"把自注意力/交叉注意力**硬编码 flash_attn3**(Hopper 专属),且 **S2V + cpu_offload 组合上游从未实现**。本次共打 5 个补丁后跑通(见 §补丁清单,全部值得提上游 PR)。
2. **可行形态 = bf16 + block offload 单卡**:480p / 5s / 40步 → **生成 22 分钟**(每步 31s)、显存峰值 20.9G、口型肉眼正常。40G 卡余量大,720p 同配置理论可跑(更慢,未测)。
3. **int8 判死**:S2V 推理路径(`wan_ops.py`)全程裸 `torch.mm`,量化哪层崩哪层;规避后(只量化 cross_attn+ffn)模型 34G 仅比 bf16 省 1.3G,无价值。**"要多卡先 int8"的 I2V 经验对 S2V 不适用**——S2V 是稠密单 DiT(28G/rank),4 卡 bf16 host 内存 ~160-200G 可能挤进 256G,还有 `load_from_rank0` 开关可救(未测)。
4. **40 步无蒸馏是生产硬伤**:lightx2v 的蒸馏火力全在 t2v/i2v,对"快数字人"的答案是自家闭源 SekoTalk → S2V 大概率永远不会有官方蒸馏。**生产级音频数字人建议转向 InfiniteTalk**(MeiGen-AI, Wan2.1 底座, LightX2V 原生支持且带 4 步蒸馏配置, 权重四件套 ~90G 已安排下载)。S2V 价值定位 = 画质对照基线。
5. **时长控制**:config `num_repeat`(每段 80 帧/5s;null=按音频长度切段,22s 音频切 5 段 → 105 分钟)。**请求体 `video_duration` 字段对 wan2.2_s2v 无效**(那是 seko_talk/audio runner 的字段)——服务化时网关需把时长翻译成 num_repeat 并做 per-request 覆盖(同 target_fps 机制)。

## 实测数据(480p, 5s/80帧, 40步, CFG=4.5, 单卡)

| 项 | 值 |
|---|---|
| 加载 | 111s(NFS 冷读 46G) |
| 生成 | **1318s(22 分钟)**,每步 31s(含 block offload 每步 ~56G 权重过 PCIe) |
| 显存峰值 | 20.9G(双缓冲仅驻 2 块 ≈1.4G + audio_inject 常驻 + 激活) |
| 宿主内存 | 65.8G(28G DiT pin + T5 + 容器) |
| 产物 | 832×448 / 77帧 / 16fps / 1.15MB,口型与音频同步、人像一致、无伪影 |

- 不开 offload 的显存账:DiT 28G 常驻 + 激活 → 480p 也 OOM(38.5G→39.1G 需求 > 39.5G 可用),**40G 卡单卡必须 offload**。
- 输入素材:镜像自带 `/opt/LightX2V/assets/inputs/audio/seko_input.{png,mp3}`。

## 补丁清单(挂卷覆盖生效,文件在 NFS smoke/;均为上游 bug,建议提 PR)

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `wan_s2v_runner.py` | cpu_offload 时输入留 CPU、VAE 已搬 CUDA → 混合设备 conv3d 报"slow_conv3d 无 CUDA 实现" | encode 前显式 `.to(AI_DEVICE)` |
| 2 | `s2v/wan_ops.py` | `flash_attention` 无 fa3 直接 raise(A100 无解) | **torch SDPA 回退**(varlen 打包前,按 k_lens 建 mask;数值等价) |
| 3 | `common/ops/tensor/tensor.py` | cpu_offload 时 DefaultTensor 只有 pin_tensor,裸 `.tensor` 访问 AttributeError | `__getattr__` 兜底:pin→device 按需搬 |
| 4 | `s2v/transformer_infer.py` | — | 新增 **WanS2VOffloadTransformerInfer**:标准块权重走 manager 双缓冲轮换,audio_inject(小)常驻 GPU,块间注入 |
| 5 | `s2v_model.py` | `_init_infer_class` 无条件用非 offload 推理类(**S2V+offload 级联问题总根源**) | 按 cpu_offload 选择推理类 |

另:官方配置 `configs/wan22/wan_s2v.json` 的 `"vae_dtype": "torch.float32"` 字符串与当前代码不兼容(配置腐化,`torch.tensor(dtype=str)` 直接崩)→ 删掉该字段用默认 bf16。

## 跑法(复现)

```bash
tmux new -s s2vbf -d 'bash /data/smoke/run_batch.sh s2v_bf16'
# 配置 configs/wan22/a100/wan_s2v_bf16.json:
#   cpu_offload:true + offload_granularity:block + use_tiling_vae + t5_cpu_offload
#   max_area:399360(480p) + num_repeat:1(5s一段) + attn 全部 torch_sdpa
# run_batch.sh 的 s2v_bf16 case 挂 5 个补丁文件(EXTRA_VOL)
```

## int8 尸检记录(为什么判死)

1. 第一版转换(`wan_dit` 默认 ignore ca/audio)**丢弃**音频编码器权重 → 加载 KeyError。
2. 重转保留 audio(bf16) → `mm_weight_autocast_nd` 裸 `torch.mm(bf16, int8)` 崩 → S2V 自定义算子路径不支持量化权重。
3. 再转 self_attn 也保 bf16(只量化 cross_attn+ffn) → 过了矩阵乘,撞 fa3 硬依赖(当时未打 SDPA 补丁);但此时模型 34G vs bf16 35.3G,**收益仅 1.3G,已无意义**。
4. 结论:除非上游重构 wan_ops 支持量化 MM 包装,S2V int8 不可行。

## 待办 / 下一步

- [ ] 口型质量与 InfiniteTalk(4步蒸馏)对比 → 决定"音频数字人"原子能力引擎选型
- [ ] 可选:720p(同配置改 max_area,预计每步 ~60-90s)、4卡 ulysses bf16 + load_from_rank0(720p 提速方案)
- [ ] 可选:社区野路子——Wan2.1 I2V 蒸馏 LoRA 挂到 S2V 跑 4-8 步(runner 支持 lora_configs,ComfyUI 社区已验证可用)
- [ ] 5 个补丁整理提上游 PR
