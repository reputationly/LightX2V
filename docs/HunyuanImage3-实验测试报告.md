# HunyuanImage-3.0-Instruct-Distil(NF4)A100 实验测试报告

> 日期:2026-07-16 | 测试人:reputationly + Claude | 节点:dev-gpustack-a100-0020
> **【终审判决:判死结案,不接入】** 技术上单节点可跑(2卡 t2i 73.8s / i2i 95.8s),
> 但对样全维度输给在产 Qwen-Image-Edit:单图编辑画质"差不多"却慢 3 倍+、占卡 2-3 倍;
> **多图融合(唯一候选差异化)在原生分辨率 OOM(4图=18.7k token,eager MoE 路由矩阵单次要 20.9G),
> 而 Qwen-Edit 同任务 1 卡直接生成**。权重与配方留档,复活条件见 §10。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 节点 | dev-gpustack-a100-0020(鲲鹏920 ARM,4×A100 PCIE 40G 无 NVLink,256G RAM) |
| 容器镜像 | `crpi-...aliyuncs.com/reputationly/vllm-omni:arm64-a100-20260714`(torch 2.11.0+cu130,Python 3.12) |
| GPU 占用 | 宿主 GPU0 有他人服务(勿动),测试用 GPU1/2/3(容器内 cuda:0/1/2) |
| 起容器 | `docker run -d --name hy3-poc --gpus '"device=1,2,3"' --memory=240g -v /nfs-models:/nfs-models <镜像> sleep infinity` |
| 必装依赖 | **`transformers==4.57.1` + `tokenizers==0.22.0`(官方钉版,5.14 必崩)**、`bitsandbytes`(ARM+cu130 4bit 实测可用)、`accelerate` |

## 2. 权重

| 项 | 值 |
|---|---|
| 权重 | `EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4-v2`(bnb NF4,双重量化,**46G/11 分片**) |
| NFS 路径 | `/nfs-models/wuhanjisuan894/models/HunyuanImage-3.0-Instruct-Distil-NF4-v2` |
| 其他版本判死 | bf16 原版 168G(官方要求 8×80G)❌;NVFP4 版(sm_120 Blackwell 专属,A100 加载即崩)❌;INT8 版 81G(2 卡放不下、3 卡无收益)不推荐 |

## 3. A100 核心配置(必改项)

```python
AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True, torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "24GiB", 1: "33GiB", "cpu": "200GiB"},  # 2卡必须不对称:cuda:0 少装权重留激活/VAE
    attn_implementation="sdpa",      # 唯一支持;flash_attn3 Hopper 专属
    moe_impl="eager",                # ARM 无 flashinfer
    moe_drop_tokens=True,
)
model.load_tokenizer(MODEL)          # 必须手动挂,否则 generate_image 崩

model.generate_image(
    prompt=..., image=[...],         # i2i 传图(≤3 张)
    bot_task="image",                # 【铁律】直生;think/recaption 判死见 §5
    use_system_prompt="en_vanilla",
    diff_infer_steps=8,              # 蒸馏版固定 8 步(guidance 2.5 内嵌)
    image_size="auto", infer_align_image_size=True,  # i2i 对齐原图尺寸
)
```

## 4. 速度/显存矩阵(均为热态,NFS 热缓存加载 46-53s / 冷 196s)

| 形态 | 任务 | 耗时 | 每步 | 显存峰值 | 备注 |
|---|---|---|---|---|---|
| 3卡 auto | t2i 1024² 8步 | **87.0s**(±3%) | 10.9 s/it | 28.5G | 冷热无差,免 warmup |
| 3卡 auto | i2i 单图 | **115.4s** | 11.8 s/it | 30.9G | 含 17s 读图定尺寸段 |
| **2卡 24/33GiB** | t2i 1024² | **73.8s** ✅ | 9.1 s/it | 39.2G(cap33 时) | **比 3 卡快 15%**(跨卡跳数少) |
| **2卡 24/33GiB** | i2i 单图 | **95.8s** ✅ | 9.8 s/it | **38.4/38.3G ⚠️** | 距红线 600MB,单图 1024 是上限 |
| 3卡 | i2i think_recaption | **4797s(80min)** ❌ | — | 35.8G | **判死**:自回归 ~6.7s/token |

**吞吐结论**:2卡×2实例/节点 = t2i ~1.63 张/分(vs 3卡单实例 0.69),**2.4×**,GPU0 还可混部单卡模型。

## 5. 关键发现

1. **任何要模型自己吐字的模式全部判死**。`think_recaption` 一张图 80 分钟(700 token CoT × 6.7s/token);`recaption` 同理(~17min)。**`bot_task="image"` 是唯一活路**——与群友 5090 实测结论一致且在 A100 上更极端。
2. **enhance 与"糊"无关**(三配方对样:素prompt/富prompt/80min官方 全一样)。"糊"根源=i2i 整图重生成的重编码损耗,且**输入相关**:真实照片输入画质 OK(红裙实测 ≈ Qwen-Edit 水平);超锐 AI 图二次编辑显软。兜底方案:SeedVR2 SR(已有原子能力)。
3. enhance 的真实价值是那段"专业改写文本",**可外置**:网关侧用小 LLM 按 TI2I 模板(改什么+显式锁定不变项)改写用户指令 → 走快路,115s 拿到等效效果。80min 那次产出的 recaption 可做模板参考(存 `hy3_i2i_run2.log`)。
4. **卡数越少越快**:层串行流水无并行计算,3卡→2卡减少跨卡搬运反提速 15%。1 卡判死(46G>40G 且 **bnb 4bit 拒绝 CPU dispatch**,报错不降级);动态搬运(block swap)只换密度不换速度,须移植 EricRollei 实现,暂缓。
5. **多节点判死**:HF device_map 不跨节点;vLLM TP 无 NVLink 判死区(节点有 mlx5 网卡未验,无动机再验)。
6. **线程并发判死(原生代码)**:单例 pipeline/scheduler(`_step_index/dt/sample`)+ `self.post_token_len` 等请求态挂共享对象,多线程必串台。**原生 batch 可用替代**(`prompt` 接受 `list[str]`),未测,为可选优化项。
7. 蒸馏版免 CFG(`cfg_distilled`)、guidance 2.5 内嵌;i2i 会自动先花 ~15s 做一次"读图定尺寸"前向(max_new_tokens=1,不可省)。

## 6. 部署建议(若立项接入)

- **形态**:2卡×2实例/节点(`CUDA_VISIBLE_DEVICES` 分组 + `max_memory={0:"24GiB",1:"33GiB"}`);**多图融合请求路由到 3 卡 profile**(2 卡必 OOM)。
- **路线**:第二引擎范式(IndexTTS 模板)——官方 remote code 包异步 FIFO API(P1)→ arm64 镜像(P2,基于 vllm-omni 或 LightX2V base + transformers 4.57.1 钉版)→ GPUStack 内置 backend(P3);**不走 LightX2V 原生接入**(AR+MoE 混合架构与 DiT 管线不兼容,P0.5 判死)。
- **产品定位**(待最终对样裁决):Qwen-Edit(1卡/更快/已在产)守常规单图编辑;HY3 的差异化 = 多图融合、复杂指令推理类编辑。若对样无明显优势,POC 结案归档即可。
- 输入限制建议:单图 ≤1024、多图走 3 卡、prompt 建议网关侧富化。

## 7. 坑点全录

| 坑 | 现象 | 解法 |
|---|---|---|
| transformers 5.14 | `StaticLayer.lazy_initialization() missing 'value_states'`(KV cache API 变) | 钉 **4.57.1** + tokenizers 0.22.0 |
| `pip install transformers[tiktoken]` | blobfile 在内网源找不到 | 去掉 extras 直装 |
| `huggingface-cli` | 新版 hub 已废弃 | 用 `hf download` |
| hf-mirror 大文件 504 | 单分片反复 Gateway Timeout(回源抽风) | 重试循环(实测 1-2h 自愈);Mac 代理直连 HF 作保底 |
| 忘 `load_tokenizer` | generate_image 崩 | 加载后必调 |
| 2卡对称配额 26GiB | bnb 4bit 拒绝 CPU dispatch,加载即 ValueError | 不对称 `24GiB,33GiB` |
| 输出别名 | i2i think 模式会自改宽高比(768×1024) | 快路 + `infer_align_image_size=True` |
| GPU0 邻居服务 | 24.7G 常驻(eCoreProc_*) | 勿动;POC 用 device=1,2,3 |

## 8. 测试资产

- 测试脚本:`LightX2V/scripts/poc/hy3_t2i_test.py`(t2i/i2i/热态/每卡配额)、`hy3_concurrent_test.py`(batch/线程/群友拓扑,未跑)
- NFS 副本与日志:`/nfs-models/wuhanjisuan894/hy3_*.{py,log,png}`(t2i 4张、i2i 狐狸 3配方、红裙 2/3卡)
- 80min CoT 样本:`hy3_i2i_run2.log`(外置改写模板参考)

## 9. 对样裁决记录(2026-07-16 结案依据)

| 科目 | Qwen-Image-Edit(在产) | HY3-NF4 | 胜负 |
|---|---|---|---|
| 单图编辑画质(红裙) | 基准 | "差不多"(用户裁定) | 平 |
| 速度/占卡 | 1 卡,更快 | 2 卡 96s / 3 卡 111s | **Qwen 胜** |
| 多图融合(男+女+鹦鹉+柴犬,4 实体) | **同 prompt 直接生成** | 原生分辨率 OOM(18.7k token,MoE one_hot 单次 20.9G);须降采样妥协 | **Qwen 胜** |
| 运维 | 省心 | 显存贴红线、配额手调、多图须限流 | **Qwen 胜** |

**判决:不接入,POC 结案。**

### 9.1 结案后补测(07-17,用户要求,全部维持原判)

| 补测项 | 结果 |
|---|---|
| **INT8 版**(jamesw767,81G/17片) | t2i 热态 **94.4s**(比 NF4-3卡慢 8%、比 2卡慢 28%)、必须 4 卡;**狐狸换季同题一样糊** → 三连败出局。注:该仓 config/代码不配套(缺 model_version 等),须用 NF4 仓的 py+config 打补丁才能跑(补丁已留在权重目录,orig_backup 为原件) |
| **糊的终审归因** | NF4/INT8 同糊 → **量化无罪**;根因=i2i 整图重生成(输入压缩为 4096 latent token,每 token 扛 16×16 像素,高频细节先天丢失)+ 8 步蒸馏无力重建细节。模型固有,本集群无解 |
| **flashinfer MoE**(vllm-omni 镜像自带 0.6.13) | **与 bnb 量化权重互斥**:cutlass_fused_moe 按权重 shape 推断维度,bnb 打包成 uint8 blob 后 shape 错乱,ValueError 崩。官方 flashinfer 路径仅适配 bf16 满血。**多图 OOM 与 87s 速度的翻案窗口就此关闭** |
| **1+3 拓扑**(真 4 卡,base 钉 cuda:0 + 32 层摊 3 卡) | 3 图融合仍 OOM:层卡 14.8G 权重 + **~8G KV(17.7k 上下文)** + 18.6G eager 路由尖峰 = 41.5G > 39.5G。**eager MoE 下 40G 卡任何摆法都塞不下 ≥3 参考图**(实测 7 种组合全灭) |
| **降采样输入** | 无效:image processor 把任意输入重采样回 1024 桶,cond token 数写死(3600+1024/图),与输入分辨率无关 |

**通用沉淀(超出本案)**:① bnb 量化 × flashinfer fused MoE 互斥——任何 MoE 大模型想在 40G 集群量化跑都撞此墙;② eager MoE 路由 `F.one_hot` 显存随序列长度爆炸(~1.05G/千token),长上下文多模态 MoE 的第一堵墙;③ HF remote code 社区量化仓质量参差,config/代码不配套时可用"可靠仓的 py+config + 目标仓的权重+quantization_config"嫁接。

## 10. 归档与复活条件

- **留档**:NF4 权重(46G,NFS)、测试脚本 ×2、全部日志/样张、本报告。容器 `hy3-poc` 可删(`docker rm -f hy3-poc`)。
- **复活条件**(满足其一再评估):① 集群升级 80G 卡(bf16/INT8 + 原生分辨率多图有空间);② 官方发布小参数或图编辑专用蒸馏版;③ 业务出现 Qwen-Edit 明确做不了、且 HY3 官方 API 验证能做的编辑需求(先用 fal.ai 验证,再谈自部署)。
- 本次沉淀的可复用资产:ARM+bnb 4bit 可用性结论、transformers 4.57.1 兼容性配方、HF remote code 多卡 max_memory 不对称调配法、"外置 prompt 富化替代模型内 enhance"的设计模式。
