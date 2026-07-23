# Bernini-R v2v 片尾雪花调查结案报告

> 2026-07-21 · dev 节点 0021-0025(5×4×A100 PCIE 40G,ARM/鲲鹏,无 NVLink)· 约 30 组对照实验
>
> **TL;DR:片尾大颗粒雪花的根因 = int8-triton W8A8 线性层量化 × v2v 编辑路径的交互。LightX2V 的 v2v 移植实现本身正确(bf16 下与原生引擎同源同果)。生产配方定稿:bf16 + 蒸馏 LoRA 4 步 + 无引导 + model 粒度专家 offload,单卡 49 帧 DiT 113s。编辑类 in-context 任务从此禁用 int8。**

---

## 一、症状与结论

v2v(in-context 视频编辑)出片在**视频时间维的后段**出现大颗粒雪花,49/81 帧、单卡/4 卡均按相对位置出现在尾部;t2v 同权重同量化完全干净。

最终证据链(其余全部排除后,唯一差异变量收口):

| 实验 | 线性层 | 注意力 | 结果 |
|---|---|---|---|
| v2v26 | **int8** | SDPA | 尾雪 |
| v2v27 | **bf16** | SDPA | **干净** |
| 原生引擎 ×2(编辑+重建) | bf16 | SDPA/FA | **干净** |
| t2v40(int8 / bf16 各一) | 均测 | — | 均干净 |

**机理假说**(经验结论,机理未做逐层定位):v2v 的注意力/线性层输入是"干净 context token + 噪声 target token"拼接的双峰分布,W8A8 动态量化的误差在该分布下超过编辑锚定信号的信噪比;后段帧对 context 锚定依赖最重(运动积累),量化噪声在此最先显形。t2v 单流分布下同一量化无害——与 [Wan2.2 int8 无损结论](../docs/InfiniteTalk-优化总结与Wan2.2移植清单.md) 不矛盾,该结论限生成类单流任务。

## 二、排除清单(全部无罪)

调参类(全部对雪花无效,但过程中修正了真实的用法错误):

| 嫌疑 | 实验 | 判定 |
|---|---|---|
| 引导方式(普通 CFG / APG / 无引导) | v2v5/7/17/18/19 vs v2v6 | ω 只放大雪花,不是根因;**但 ω=4 普通 CFG 确实毁蒸馏编辑(40 步下雪花瀑布)** |
| 步数(4/8/40) | v2v6/10/22 | 无效 |
| 蒸馏 LoRA(有/无、强度 0.75) | v2v13/22 | **蒸馏无罪**(bf16 下编辑质量可用,int8 才是真凶) |
| sample_shift(5→3) | v2v23 | 无效 |
| sage_attn2(→SDPA) | v2v26 | 无效(int8 下换注意力后端仍雪) |
| int8 多步累积 | t2v40 int8(干净) | 排除"int8 在 40 步下必炸" |
| 源视频质量 | 09.mp4 肉眼 + VAE 往返测试(逐帧 std 健康、出片干净) | M1 读取/编码无罪 |
| 组件替换(VAE/T5/tokenizer) | 与 Bernini-R-Diffusers 自带件指纹比对 | 全部为标准原版(umt5-xxl / Wan2.1 VAE,latents_mean 一致) |
| 移植结构(RoPE 对齐/source-id 相位/mask/timestep) | 对上游 transformer_wan.py / wan_diffusion.py 逐行核对 + 逐帧范数仪器 | 全部一致;范数无爆炸,\|diff\|/frame 尾部平滑升 2-3×(模型固有,被 ω 放大) |

## 三、原生引擎 ARM 跑通(金标准 + 备选路线)

官方 Bernini 仓在本集群完整跑通,配方:

- 底座:lightx2v arm64 镜像;`pip uninstall torchao`(与 diffusers 0.35.2 冲突)→ `pip install diffusers==0.35.2 transformers==4.57.3 accelerate==0.34.2 ftfy einops imageio imageio-ffmpeg`(tuna 源)
- VeOmni v0.1.11:Mac clone → NFS;**先 cp 到容器本地再装**(NFS 并发构建会产生 build/ 套娃)+ `--no-deps --ignore-requires-python`(要求 py3.11,3.10 实测可用)
- decord 垫片:`scripts/smoke/native_shim/decord.py`(PyAV 实现 VideoReader 最小 API),PYTHONPATH 前置
- 显存:单卡 40G 不够(fp32 per-token temb 一次要 4.6G),**2 卡 `torchrun --nproc-per-node 2 infer_multi_gpu.py --ulysses 2` 通过**
- 权重:`Bernini-R-Diffusers` 全自包含,`--config` 直指权重目录即可

结果:水墨编辑与重建两条 49f/40 步各 ~12 分钟(18-19s/it),全程干净。GPUStack 嵌入可走 IndexTTS 模式(FastAPI 异步任务 API + generic_proxy),作为多任务(mv2v/rv2v/r2v/i2i)扩展的备选路线。

## 四、耗时汇总(480p,seed 42,冷启动含加载)

| 配方 | 卡数 | 帧数 | DiT | 端到端(冷) | 备注 |
|---|---|---|---|---|---|
| **v2v28 生产候选:bf16+蒸馏4步+NOCFG** | 1 | 49 | **113.6s** | 335s(载 bf16+LoRA ~220s) | 常驻 server 后单请求 ≈ 2 min |
| v2v29 同上 + APG ω=4 | 1 | 49 | 220.9s | 415s | 双前向;风格反而更弱(APG 平行阻尼保源),留作轻编辑选项 |
| v2v27 bf16 基座 40 步 NOCFG(SDPA) | 1 | 49 | 847.6s | 1083s | 判决实验,非生产 |
| t2v40 bf16 对照 | 1 | 49 | 374.9s | 618s | bf16 通路验证 |
| 原生引擎 40 步 APG | 2 | 49 | ~12min | ~17min(含依赖安装) | 金标准 |
| v2v30 2 卡 ulysses + 81 帧 | 2 | 81 | — | **判死** | 三连 OOM(200g/250g cgroup、无上限主机 OOM):双 rank 各持双专家的匿名内存峰值 >250G,256G 主机放不下;与 InfiniteTalk 铁律"多卡禁 cpu_offload(per-rank pin 爆 host)"完全吻合 |
| v2v32 单卡 block 流式 + 81 帧 | 1 | 81 | 356s | **判死(全黑)** | **block offload × MoE 双专家 = LightX2V 既有 bug**(v2v25 非蒸馏 + v2v32 蒸馏,两变体全黑,单流模型如 S2V 无此问题);且 85s/步说明 PCIe 重叠效率也低,即使修好也不快 |
| v2v31 fp8-triton(e4m3)| 1 | 49 | — | **判死(硬件)** | 权重转换成功(53.6→13.4G/专家,-75%),但推理首个前向 triton 编译报 `fp8e4nv not supported in this architecture`(仅 e4m3b15/e5m2)——**A100/SM80 不支持 e4m3,fp8 算力是 Hopper SM90 专属** |
| v2v31 fp8-sgl | 1 | 49 | — | **判死(依赖)** | `sgl_kernel is None`——sgl-kernel ARM64 编不出(镜像 Dockerfile 该步 `|| echo skip` 容错跳过)。q8f/vllm/pertensor 底层同依赖 e4m3 tensor core 或 x86 kernel,不再试。**fp8 路线在本集群物理封死,需 H100/L40S** |

## 五、生产配置定稿

- **配方**:bf16 原始权重(`Bernini-R-14B-bf16/{high,low}_noise`,converter.py `--direction backward` 自 Diffusers fp32 转出)+ 社区 4 步蒸馏 LoRA(dynamic apply)+ `enable_cfg:false` + `cpu_offload:true, offload_granularity:"model"`(专家级换入换出,boundary 处一次,开销≈0)
- **形态**:**49 帧单卡为主力档**(`bernini_r_14b_v2v_bf16_distill.json`,DiT 113s 已验);**81 帧档 2/4 卡 bf16 均判死**——双(多)rank 各持双专家的匿名内存实测峰值 >250G(200g/250g cgroup 与无上限主机 OOM 三连杀),256G 主机放不下。81 帧出路:①原生引擎 2 卡(已验 12 分钟);②fp8 权重(14G/专家,双专家全驻 GPU、RAM 减半,待画质验证);③给 MoE lazy 路径补"切换时释放旧专家"(~10 行,MultiDistillModelStruct.get_current_model_index,待评审)
- **提示词协议**:任务 system prompt **只拼 cond**(上游 pipeline.py:943/299);风格类编辑用 mv2v system prompt;长指令式提示词(改什么+保留什么);标准 Wan2.2 中文负向词
- **铁律新增**:①编辑类 in-context 任务禁 int8;②`offload_granularity:"block"` 与 wan2.2_moe(非蒸馏)组合会**黑屏**(v2v25),必须用 `"model"`;③APG(`guidance_mode:"v2v_apg"`,已实现进 model.py)编辑力度弱于无引导,仅轻编辑场景使用

## 六、调试资产(随本次改动入库)

- smoke 脚本 env 开关矩阵:`NP/GPU/TASK/FRAMES/SYS/NOCFG/STEPS/OMEGA/STRENGTH/APG/DBG/SHIFT/ATTN/BASE40/BF16D`,现场 python 覆写临时 config
- `scripts/smoke/v2v_vae_roundtrip.py`:M1 隔离测试(读→编→解→存,逐 latent 帧统计)
- `scripts/smoke/native_shim/decord.py`:ARM decord 垫片
- model.py:`v2v_apg` 引导 + `v2v_debug_norms` 逐帧范数仪器
- 孤儿容器教训:tmux kill 只杀 docker 客户端,容器残留占显存 → 脚本已加命名容器+启动自清理
- 内存运维教训:①docker cgroup 把容器读文件的 page cache 也记账,大权重任务 `--memory` 上限要按"匿名+缓存"合计给;②**权重转换任务必须设上限**(匿名内存无上界,不设限会把节点打到 ssh 失联,0024 实例);③推理任务匿名内存有已知上界时可放开上限让内核全局回收;④测数前 `drop_caches` 给干净起点,复测时保留缓存加速加载

## 七、引擎选型:LightX2V 移植 vs 原生引擎

| 维度 | LightX2V 移植 | Bernini 原生引擎 |
|---|---|---|
| 正确性 | v2v bf16 已验干净(与原生同源同果);int8 判死 | 官方实现,金标准 ✓ |
| 任务覆盖 | 仅 t2v/t2i/v2v;mv2v/rv2v/r2v/i2i 每个都要重走移植+调试 | 全任务白送 |
| 49 帧延迟 | **DiT 113s,热请求 ≈2 min/条,单卡**(实测) | 40 步 12 min/2 卡(实测);merge 蒸馏 LoRA 估 ~2 min(未验证) |
| 81 帧延迟 | 无解(2 卡 OOM / block offload 黑屏;fp8 待验) | 估 ~20 min/2 卡;merge LoRA 估 ~3 min(未验证) |
| 单节点密度 | 内存瓶颈:每实例 ~67G → 3 实例/节点 | 每实例 2 卡+~140G → 1 实例/节点 |
| 服务化 | lx2v-launcher 现成,GPUStack 已调研 | 需自建 FastAPI 包装(估 2-3 天,IndexTTS 先例) |
| 工程债 | block offload×MoE bug、UniPC 未移植、新任务=新工程 | decord 垫片/VeOmni py3.10 绕过;跟上游容易 |
| 运维 | 与现有生产线同引擎单轨 ✓ | 双引擎双轨 |

**决策**:v2v 上线走 LightX2V(49 帧单卡档);产品确认需要 mv2v/rv2v/r2v 中 ≥2 个任务时,启动原生引擎产品化(一次包装全任务通吃);原生环境永久保留为质量金标准(回归对拍)。

## 八、待办

1. v2v30 结果回填(2 卡 ulysses × model offload 组合验证)→ 生产配置终稿
2. 同 seed 单卡 vs 2 卡 SP 数值一致性抽查
3. 720p v2v 显存探测(压测方案 E 系列,见 `docs/Bernini-R-720p-ulysses-压测方案.md`)
4. server schema(v2v 任务参数)+ lx2v-launcher profile
5. 原生引擎产品化立项评估(多任务扩展时)
6. bf16 常驻实例的节点混部规划(单卡 v2v × N + t2v int8 共存)
