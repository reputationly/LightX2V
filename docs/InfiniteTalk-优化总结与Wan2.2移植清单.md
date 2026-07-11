# InfiniteTalk 优化战役总结 & Wan2.2 t2v/i2v 移植清单

> 日期:2026-07-11 · 基于 `docs/InfiniteTalk-实验测试报告.md` 的全部实验(T1-T8/O系列)
> 本文回答两个问题:①这轮优化的方法论沉淀;②哪些成果能直接搬到 Wan2.2 t2v/i2v 生产线。

---

## 一、优化过程复盘(从 22 分钟到 54 秒的路径)

| 阶段 | 动作 | 结果 | 方法论 |
|---|---|---|---|
| 起点 | S2V 40步 bf16 offload | 22 分钟/5s条 | 先跑通再优化 |
| 换模型 | → InfiniteTalk 4步蒸馏 | 130s(10×) | **选型>调优**:有蒸馏的模型天然赢 |
| 蒸馏审判 | 40步基线 vs 4步, 同seed肉眼 | 无差 → 20×白拿 | **蒸馏损失必须实测**,不能想当然 |
| 量化后端对决 | torchao/vllm/q8f/sgl/triton 五选 | **triton 93.8s vs torchao 228s(2.4×差距)** | **同一权重不同 kernel 天壤之别**;逐后端排雷 |
| 形态对决 | offload vs 常驻 vs 多卡×{1,2,4} | 常驻 int8 单卡 93.8s;4卡 54.4s(1.7×) | Amdahl 拆账:串行部分(VAE/通信)决定多卡天花板 |
| 并发验证 | 单卡×4 / 2卡×2 满载 | **零干扰**(91-93s) | 外推的容量必须满载实测 |
| 分辨率规律 | 480p vs 720p 的多卡加速比 | 1.73× vs **2.29×** | **算量越大并行效率越高**(DiT平方涨/串行线性涨) |
| 长内容工况 | 5s/15s/60s 曲线 | 时间线性(24s/秒);显存恒定;**host内存随时长涨→60-75s红线** | 资源曲线要测到生产工况长度 |
| 通信优化 | fp8_comm/tensor_fusion/head_parallel | 测试中(O1-O3) | 无NVLink机器的通信是主要损耗 |

**踩坑沉淀(部署铁律)**:安静宿主测数(邻居加载拖慢推理2-4倍)/多实例错峰启动/多卡禁cpu_offload(per-rank pin爆host)/load_from_rank0是多卡加载入场券(且需权重≤显存)/triton首跑autotune须预热/长任务全进tmux。

---

## 二、Wan2.2 t2v/i2v 移植清单(按预期收益排序)

### ⭐ W1: int8-torchao → int8-triton(已结案 2026-07-11:**转正,生产配置已切换**)

**证据**:InfiniteTalk 同权重下 triton 93.8s vs torchao 228s(2.4×),机理(weight-only反量化开销)与模型无关。
**Wan2.2 实测**(同 seed A/B, triton loader 吃 block 目录格式无碍):

| 形态 | torchao → triton | 生产影响 |
|---|---|---|
| i2v 480p 单卡(w1a) | 55 → 31s(1.77×) | — |
| **i2v 720p 4卡(w1b, 生产形态)** | 86 → **68s**(1.26×) | **生产 i2v 提速 21%** |
| t2v 720p 4卡(w1c, 生产形态) | 141 → 133s(1.06×) | 提速 6%(t2v seko 权重 DiT 占比低, 提升被通信摊薄) |

**画质终审**:三对同 seed 肉眼对比"都看不出差异"(2026-07-11 用户判决)→ 零画质代价。
**落地**:`configs/deploy/wan22_{i2v,t2v}_int8_4card_a100.json` 已改 `dit_quant_scheme: int8-triton`;t2v 同时固化 `boundary_step_index: 2`(原依赖 launcher 注入,配置自身跑不起来的隐患)。待随出包发布+灰度。
**运维注意**:triton 首请求 per-shape autotune(冷态偏慢),预热一发再对外。

### ~~W2: seq_p 通信优化~~(已结案 2026-07-11:全灭)

- O1 fp8_comm / O2 +tensor_fusion:❌ 崩于 `scaled_fp8_quant` 缺失(依赖 vllm 编译算子, ARM wheel 没带——与 int8-vllm 同死因)
- O3 head_parallel:❌ 无收益(134s ≈ 基线 136.7s)
→ **无 NVLink + ARM 栈上, ulysses 通信成本无软件手段可省**;54.4s(480p)/136.7s(720p)即多卡最终数字。此结论同样适用于 wan2.2 的 4卡生产形态,勿再尝试。

### ~~W3: load_from_rank0~~(已结案 2026-07-11:**MoE 不适用**)

w1b 实证:`load_from_rank0: true` + Wan2.2 MoE 双专家 **直接崩**(该机制按单 DiT 假设广播,与 MultiModelStruct 双模型不兼容)——此前"MoE int8 可过"的推测作废。**仅限单 DiT 模型使用**(InfiniteTalk/S2V 等);Wan2.2 生产配置禁配此项。

### W4: 运维方法论(零成本全量移植)

- 错峰启动/预热请求(triton autotune per-shape)/安静宿主压测纪律
- 压测工具 `test_infinitetalk_stress.sh` 泛化:CFG_BASE 指向任意 wan 配置即可复用(监控CSV/稳态统计/防撞名)
- NUMA 绑定(2卡形态 +4%)

### W5: 画质专项(已结案 2026-07-11,详见《Wan2.2-真实感调查报告.md》)

**判决:模型全线无罪(蒸馏 40步vs4步 同seed"都没有CG感"、int8 无损、运镜 720p 下动静皆真),主犯是提示词/题材,从犯是分辨率(480p 显糊)。** 同一套生产权重,素句瀑布"好假"→摄影级提示词+720p+SeedVR2 收尾"很真实"。处方:提示词增强(主杠杆)+ 真实感场景引导 i2v + ≥720p + 链路末尾 SR。

### 不移植/不适用

- int8 判死结论(那是 S2V 特有的 wan_ops 裸矩阵乘问题;wan2.2 标准路径量化健康)
- 长内容 host 内存红线(t2v/i2v 单条 5-10s, 帧累积可忽略)
- 多人/audio 相关

---

## 三、执行顺序建议

1. **W1 立测**(半小时, 现有权重+一行配置):先单卡 i2v int8-triton vs torchao 同seed;赢了推 4卡与 t2v
2. W2 等 O1-O3 判决后搬运
3. W5-3(蒸馏审判)与 W5-1(SeedVR2解药)回答"模糊/CG"——用户挑样本后开跑
4. W3/W4 并入出包与部署规范
