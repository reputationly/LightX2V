# 《RelayVSR 4 卡精确块并行》方案评审

> 评审对象:[RelayVSR-4卡精确块并行-改造方案](RelayVSR-4卡精确块并行-改造方案.md)(未提交)。
> 评审方式:逐条对照当前代码(`75b295ae`),并在 `dev-gpustack-a100-0007` 上补跑了方案 M0 需要的事实。
> 结论:**方案成立,可以交给开发**。§2.1 的串行语义与代码逐条一致;我原先唯一担心的"每层 KV 通信"
> 实测比文档自己设的停下阈值低一个数量级,不是风险。下面 8 条是需要改的地方。

---

## 1. 结论:可行性已经用实测数据兜住了

### 1.1 checkpoint 事实(M0 ①,已完成)

`wan_x2.safetensors` 元数据:

| 键 | 值 |
|---|---|
| `num_layers` | 30 |
| `num_attention_heads` / `attention_head_dim` | 24 / 128(dim = 3072) |
| `ffn_dim` | 14336 |
| `temporal_context_frames` | **4**(所以 `c-1 = 3`) |
| `temporal_rope` | `{policy: fixed_end, training_frames: 8}` |
| `spatial_window_size` | `[22, 40]` |
| 其它 | `patch_size [1,2,2]`、`in/out_channels 48`、`rope_max_seq_len 1024`、`use_block_causal_attention true` |

flash 侧:`dim 512 / depth 10 / num_heads 8 / num_kv_groups 2 / window_size 2`
(与 `model.py::validate_checkpoint_configs` 的期望一致)。

wan token 数 = `h/16 × w/16`:720P(补到 720×1280)是 `45×80 = 3600`;768P 1344×768 是 `48×84 = 4032`。
**注意 720P 的 grid 和 `WARMUP_RESOLUTIONS` 里的 768×1024、768×1344 都不同**,见 §3.2。

### 1.2 拓扑与 P2P 带宽(M0 ②③,已完成)

`nvidia-smi topo -m` / `nvidia-smi topo -p2p r`:无 NVLink;GPU0–GPU1 走 PHB(NUMA0,核 0-31),
GPU2–GPU3 走 PHB(NUMA2,核 64-95),跨节点是 SYS;整机 128 核。

NCCL `send/recv` 512 MiB 实测(`/data/verify-logs/relayvsr/profile/p2p_bench.py`):

| 方向 | GB/s |
|---|---|
| 0→1 / 1→0 | 14.9 / 20.8 |
| 2→3 / 3→2 | 21.1 / 21.2 |
| 1→2 / 3→0 / 0→2(跨 NUMA) | 15.5 / 15.2 / 15.2 |

**跨 NUMA 不掉速**(15 GB/s),所以按"线性链"分段的 rank 顺序不用为拓扑做特殊安排;
只有 1→2 这一个边界跨节点,和 PHB 一样快。

### 1.3 每层 KV 交换量(M0 要求"算出来")

`c-1 = 3` 个 ref,每个 ref 每层 `k+v`:

```
k: 3600 tokens × 24 heads × 128 × 2B = 22.12 MB
k+v                                  = 44.25 MB / ref / 层
每边界每层 = 3 × 44.25               = 132.7 MB
整个 Phase A 每方向 = × 30 层        = 3.98 GB      → 15.4 GB/s 下 0.26 s
```

768P(4032 tokens)是 `4.46 GB → 0.29 s`。

对照 Phase A 的实际耗时(每 rank 13 个 ref,稳态 0.553 s/ref = 7.2 s):

| | 30s 素材(51 ref) | 4s 素材(8 ref) |
|---|---|---|
| KV 交换 / 方向 | 0.26 s | 0.26 s(**与长度无关**) |
| Phase A / rank | 7.2 s | 1.1 s |
| 占比(单向;收发不重叠按双倍算) | 3.6%(≤7.2%) | 24%(≤48%) |

→ **§10 的"每层通信 > stage2 的 50%"不会触发,差一个数量级**。30s 素材上通信不是问题;
只有 4s 这种短素材才会因为"边界 KV 是与长度无关的固定量"而吃到 24% —— 这与文档"4s 只要求比 1 卡快"
的保守设定一致,是对的。

### 1.4 串行路径耗时分解(M0 ④,已完成)

`profile_serial3.py` 用 CUDA event 区间计时(不插 host sync,扰动可忽略),720P 748 帧、`clip_len=16`、
本镜像 `arm64-a100-20261103-0647-e48faec8`:

| 阶段 | 首次请求(warmup 之后) | 稳态折算(见 §3.2) |
|---|---|---|
| `reference()` | 41.03 s / 51 次 = **804.5 ms/ref** | ≈28.2 s = 553 ms/ref |
| `flash_model_decode()` | 40.86 s / 797 次 = **51.3 ms/帧** | ≈28.0 s = 35.2 ms/帧 |
| `preprocess` / `postprocess` | 0.35 / 0.20 s | 同量级 |
| 合计(`RUN pipeline`) | **83.8 s** | **56.3 s** |

两个可直接用的结论:

1. **单卡耗时几乎全在 ref 前向和 flash 解码上,各占一半**;解码 + 编码 + 拼接合计不到 2 s
   (对比 1.2 节:748 帧完整 RGB 解码 2.92 s,但在 `prefetch_frames` 线程里被藏掉了)。
   所以 Phase A / Phase B 两条都能按 4 卡摊,Amdahl 的串行尾巴只有 1.3 s 的 ffconcat + 音轨。
2. ref 前向 804 ms 对应 `2 × 3.8e9 × 3600 ≈ 2.3e13 FLOP` → 约 **29 TFLOPS(≈9% MFU)**,
   属于**发射/小 tile 受限**而不是算力受限。这对方案是好事:每 rank 少 3/4 的 ref,基本线性回收。

### 1.5 编码吞吐(M0 ⑤,已完成)

真实 2K 画面(`p4cl16-1.mp4`,2560×1440)取 187 帧,`libx265 crf18 ultrafast pools=16`:

| | 耗时 | 吞吐 |
|---|---|---|
| 1 路 | 3.77 s | 49.7 fps |
| 4 路并发 | 4.76 s(4×187 帧) | 聚合 157 fps,3.16× 扩展 |

每 rank 187 帧 ≈ 4.2~4.5 s,对约 14 s 的 GPU 时间完全藏得住(线程里跑),**编码不是瓶颈**。
(用随机噪声帧测会得到完全错误的结果,不要那样测。)

---

## 2. §2.1 串行语义逐条核对:全部与代码一致

| 文档条款 | 代码事实 | 判定 |
|---|---|---|
| 窗口网格 `step=L-1`,窗口 k 覆盖 `[k·step, k·step+L)` | `restore_frames` 的 `window = [start, *islice(..., L-1)]`,`start` 每轮取上一窗口末帧 | ✅ |
| `W = 1 if total<=L else 1+ceil((total-L)/step)`;ref 数 `R=W+1`(`total==1` 时 1) | 748 帧 → W=50、R=51,与实测 `reference()` 调用 **51 次**吻合 | ✅ |
| ref `j`(j≥1)= 帧 `min(j·step, total-1)` | 末窗可能是短窗(748 帧时末窗只有 13 帧),`min(...)` 正好覆盖;`total≡1 (mod step)` 时尾窗长度为 1 会被整段丢弃、不产生 ref — 公式也成立 | ✅ |
| 每个 ref 从同一 generator 依次抽一次 `prepare_latents` | `RelayVSRScheduler.prepare_latents` 用 `torch.randn(..., generator=self.generator)`,且 `reference()` 里每个 ref 恰好调一次 | ✅ |
| ref 前向 = `preprocess → wan_pre.infer → 30 层 → infer_non_blocks → unpatchify → latents-noise_pred → [:,:,0] → flash_model_prepare_condition` | `reference()` + `RelayWanPostInfer.infer` + `scheduler.step_post` | ✅ |
| K/V 写槽位 `j % c`,读取前 `visible=min(j+1,c)` 个槽、**按槽位顺序**拼(不是时间顺序);槽 s 存 ref `j-((j-s)%c)` | `infer_self_attn` 里 `cache[slot*tokens:...] = k`,`key = cache[0][:k_rope[0].shape[0]]`;`prepare_input` 给出 `positions = tuple(frame-((frame-slot)%window)-origin ...)` | ✅ **这是最容易写错的地方,文档抓对了** |
| RoPE `origin = max(0, j-training_frames+1)`,`q_rope` 用 `frame-origin` | `prepare_input` 逐字如此 | ✅ |
| 注意力 = `SpatialAttentionSpec(1,h,w,wh,ww,c,key_frames=visible,query_frame_offset=visible-1)` | `infer_self_attn` 逐字如此;`h=w` 分界用 `height>wh or width>ww` | ✅ |
| flash:`condition(ref w)` 起点、`condition(ref w+1)` 终点,窗口开头 `history.reset()`,逐帧 `decode(offset)` | `restore_frames` 逐字如此 | ✅ |
| 输出:窗口 0 全发,其余不发 offset 0;裁剪后 `postprocess` | `if first_window or offset > 0`,再 `output[..., :h*scale, :w*scale]` | ✅ |

**文档漏掉的一条**(建议补进 §2.1):`reference()` 里除前向之外还有 `scheduler.step_pre(0)` /
`step_post()`,`self.scheduler.latents` 与 `noise_pred` 都是**逐 ref 重建**的临时量,不构成跨 ref 状态;
真正跨 ref 的只有 `reference_count`、wan ring 缓存、generator 三样 —— 而这三样在文档 §2.2 里都点了。
另外 `reset_infer_states(x, context)` 走的是 `x.shape[0]`(每个 ref 相同)与固定的 cross context,
所以在 layer-major 里**设一次就够**,不必每层每 ref 重设(§4.2 已提到,建议写实成"只设一次")。

结论:文档 §2.1 可以直接当正确性基准交付,不需要再和开发对一轮口径。

---

## 3. 需要改的 8 处

### 3.1 性能门槛:3.3× 就是**算术上限附近**,不是保守值(回答"要不要等 M0")

按 §1.4 的分解直接算(720P 748 帧、`clip_len=16`、稳态):

```
Phase A: 28.2 / 4 = 7.05 s  + 0.26 s 通信   ≈ 7.3 s
Phase B: 28.0 / 4 = 7.0  s
IO/拼接:                                       1.3 s(ffconcat + 音轨,实测)
合计 ≈ 15.6 s        →  56.3 / 15.6 = 3.6×
```

也就是:**3.6× 是这一版设计在 cl16 下的天花板,3.3× 只留 9% 余量**;再扣掉 rank 间不均衡和
P2P 气泡,实际很可能落在 3.2~3.5×。

建议(不必等 M0 再调,现在就能定):

- **3.3× 保留为"必须达到",3.6× 明确写成"目标/加分"**,并接受"cl16 下达不到 3.6× 不算失败";
> **实施后更正(2026-11-04)**:cl32 已实测,**不是提速杠杆** —— 4 卡绝对耗时从 15.12s 降到
> 13.52s,但单卡同时从 51.11s 降到 41.65s,加速比反而 3.38× → 3.08×。本节下面这条预估作废,
> 详情见[自测报告 §5.6](RelayVSR-4卡精确块并行-自测报告.md)。
>
> 另外实测 4 卡 720P 748 帧是 **3.38×**(不是本节估的 3.6×),差额来自 Phase B 每个 rank
> 1.42~1.75× 的单帧开销与 1.08s 拼接;把跳帧的 RGB 转换省掉之后才从 2.84× 补到 3.38×。

- ~~把**唯一的免费杠杆写进文档**:`clip_len=32`。window 数从 50 降到 25、ref 从 51 降到 26,
  单卡实测 47.08 s(cl16 是 56.4 s),而**边界 KV 总量不变**(仍 3.98 GB)。按同样的分解:
  `Phase A 14.4/4+0.26 ≈ 3.9 s` + `Phase B 27.1/4 ≈ 6.8 s` + `1.3 s` ≈ **12.0 s → 4.0×**。
  这是把 Phase A(带通信和 P2P 同步的那一半)压小、让纯并行的 Phase B 占更大头。
  前提是产品能接受 cl32 的画面(和 cl16 不是同一组结果),需要业务拍板;
- 1 卡基线**必须用稳态**(§7.3 已写"取第 2、3 次",正确),M0 ④ 也要补上同一个口径,理由见 §3.2。

### 3.2 首次请求比稳态慢 49% —— warmup 没覆盖真实请求的形状

实测:同一条 720P 748 帧请求,**warmup 之后的第一条 83.8 s,稳态 56.3 s**(1.49×)。原因不是预热不够,
而是**预热分辨率与素材分辨率不匹配**:`WARMUP_RESOLUTIONS = ((768,1024),(768,1344))`,
而 720P 请求补到 720×1280 → wan grid `45×80 = 3600`,是一个从未预热过的 token 数,首条请求在
ref 和 flash 两侧各重编译一遍(两侧都恰好是 1.49×,是同一个原因)。

影响三处:

1. §3.6 的 `warmup_resolutions` 不是"顺带优化",**是必需的**;并且必须把**测试/生产素材的分辨率**
   写进去(720P 素材要加 `[720,1280]`),否则 §7.3 的 3.3× 会拿"4 卡稳态"对"1 卡首条"，
   或者反过来;
2. M0 ④ 的"单 ref 平均耗时 / 单帧 flash 耗时"必须取**稳态**(常驻服务第 2、3 次),否则整条
   KV 预算会被高估 1.5 倍;
3. A6 的"预热后零重编译"要以**真实请求分辨率已在 `warmup_resolutions` 里**为前提判定。

### 3.3 §1 的"现状"描述已过时,删代码的清单要重写

文档写"段首重放 `temporal_context_frames` 个窗口" —— 那是 `5481c6a9`;`910e2578` 之后当前代码是
"**窗口对齐切分 + 最多 `c` 个窗口重放 + 单窗口 priming(只读上一窗口末帧)+ 噪声快进**"
(`_chunk_windows` 返回窗口网格,`restore_frames(priming=...)`)。要删的清单精确到:

- `_chunk_windows`、`_chunk_spans`(后者已无人调用,是死代码);
- `_run_video_chunks` 里的 `warmup_windows` / `first_window_index` / `drop_frames` / `islice(start-1)` / `priming`;
- `restore_frames` 的 `priming` 形参、`self.reference_count = prime_index`、噪声快进循环,
  以及那句**现在已经是错的 docstring**("成片与单卡逐位相同" —— 实测 44.38 dB,最差帧 41.4 dB)。
  §7.5 的 `grep` 关键词里请加上 `warmup_windows|priming|prime_index|drop_frames`。

### 3.4 §4.2/§4.3:不要复制 `infer_block`,注入通道已经存在

`infer_block(block, x, pre, self_attn_kwargs=None)` 已经接受 `self_attn_kwargs`,并且
`BaseTransformerInfer.run_block` → `self.infer_block(block, *args)` 会把 relay `run_block` 组好的
`attention` 字典原样传进来。所以 stage2 完全可以**继续调用同一个 `infer_block`**,只是把
`self_attn_kwargs` 换成 `{q_rope,k_rope,kv_store}`;在 relay 的 `infer_self_attn` 里加一个分支:

```
kv_store 为空 → 现在这条路(写 ring → 读 ring 前缀)
kv_store 非空 → 从 kv_store 按槽位顺序取 k/v(不再碰 ring)→ attend
```

这样 cross-attn / FFN / post_process / offload / compile 全部和串行共用**同一份代码段**,
不会再出现"逐行对应"的漂移风险,stage1 只剩 `pre_process` + 新的 `project_qkv`(约 6 行)。

顺带两个细节:

- relay 的 block **没有** `before_proj`(`infer_block` 里是 `hasattr(...)` 判空),§4.2 里
  "before_proj 残差"那句话可以删掉;
- `get_compiled_block` 是按 `(block_idx, block)` 缓存的,同一个 `infer_block` 在有/无 `kv_store`
  下会产生两套图,每个 block 自己的 `recompile_limit` 已被 `RelayCompileMixin` 提到 32,够用;
  但 warmup 必须把**两套图**都跑一遍(§3.6 的 M4 退出条件请写明"stage1 与 stage2 两套图各预热一次");
- `mark_sequence_dynamic` 要把注入的 k/v 一起标动态(现有 `run_block` 标的是 ring cache)。

### 3.5 一处会被 A2 首先撞到的点:KV 布局

串行读的是**预分配 ring 缓冲的前缀视图**(连续),精确路径是**新 concat 出来的张量**。
两者数值相同、顺序相同,triton 按 stride 取值,理论上逐位一致;但这仍是 A2 最可能失败的地方。
建议 M2 第一个断言就打在这里(`k`/`v` 与串行 `cache[0][:visible*tokens]` 直接 `torch.equal`),
而不是等整条视频跑完再比。

### 3.6 A4/A5 的门槛判据要换(这条最重要)

"4 卡(编译)vs 1 卡(编译)最差帧 ≥ 50 dB"**不是算法正确性判据**。8 bit 视频上
"每个像素差 1 LSB"已经是 48.1 dB,`≥50 dB` 意味着几乎 2/3 的像素完全不同都算过线。
它把"算法是否等价"和"Inductor 融合边界是否恰好一致"混在了一个数里:stage1/stage2 分成两段编译后,
融合点必然和串行不同,一个**完全正确**的实现也可能只有 40 dB 出头。

建议改成两条判据:

| 编号 | 内容 | 要求 |
|---|---|---|
| A2' | **真 4 卡、`use_compile=false`**、720P 96 帧与 360 帧 | 每个 ref 的 `latent` 与串行 `reference()` 序列 `torch.equal`;全部输出帧与串行 eager **逐位相同** |
| A4' | 编译开启,4 卡 vs 1 卡 | 报告 min/mean PSNR,门槛降到 **≥40 dB**,并**明确它只是编译差异的观测量**,不用于判算法对错 |

这样:算法错了 A2' 一定失败(而且能定位到第一个不同的 ref),编译差异不会误杀正确实现。
A2 现在只覆盖"单卡虚拟多 rank",而它恰恰验证不了真实的 NCCL 字节一致性和真实的 rank 边界 ——
把 A2' 加到真 4 卡上是零成本的(跑同一个 eager 版本即可),价值最高。

### 3.7 取消:每层一个标量 stop 检查,别接受 600 s 超时

§3.5 说"Phase A 内部不调 `check_stop`,异常/取消延迟上限约一次 Phase A,本期接受"。这条对一个
常驻服务来说不能接受:NCCL watchdog 是 **600 s**,期间服务是挂住的。

改法很便宜:要求**所有 rank(含空闲 rank)都跑满 `num_layers` 次循环**,每层在 P2P 之前调一次
`check_stop()`(它本来就是一次标量 MAX all-reduce,`base_runner.py:404`)。空闲 rank 只是不参与
P2P 计划(计划为空),照样能到齐。代价:30 次小 all-reduce ≈ **1 ms 量级**(相对 7.2 s 的 Phase A),
换来取消延迟从"一次 Phase A × NCCL 超时"降到"一层"。文档 §3.5 的"否则空闲 rank 无法配对"
只在"空闲 rank 跳过整个循环"时才成立,把"必须跑满层数"写成硬约束即可。

### 3.8 §1 第 2 条对解码的批评过重,§3.4 的"两遍解码"完全没问题

实测 748 帧:纯 demux+decode **0.46 s(1628 fps)**,带 `to_ndarray(rgb24)+permute+pin` **2.92 s(256 fps)**。
也就是 rank3 补齐"解码整条前缀但不转换"只要 **约 0.35 s**,而且完全被 Phase A 藏住。
所以:`container.seek()` 可以**永久不做**(§9 已这么写,正确),§3.4 的"两遍解码、内存有界"也没问题;
§1 里"后面的 rank 要先把前面所有帧解码 + 转 RGB + pin 才能开工"这句请改掉——那正是当前
`islice` 实现的问题,不是精确方案的问题,别让开发以为这块要优化。

---

## 4. 对你三个问题的回答

**① 性能门槛要不要等 M0?** 不用等,3.3× 站得住(§3.1 的分解)。但要同时接受:
cl16 下 3.6× 是天花板;把 `clip_len=32`(预测 4.0×)作为一个"需要业务拍板"的备选写进文档;
1 卡基线一律取稳态。

**② 直接删旧实现不留开关?** 同意,代码最干净,而且旧实现本身就是"近似",留着会被人误开。
三个前提:
1. 旧实现的产物(`/nfs-data/lx2v_test/relayvsr/cp_p4nw2.mp4`、`p4cl16-*.mp4` 等)作为对照基线留在
   NFS 上,验收时拿来对比"新方案 vs 旧近似 vs 单卡";
2. 每个里程碑一个 commit(§6 已要求),M3 之前任何一步都可以 `git revert` 回旧实现;
3. 删 `restore_frames` 的 `priming` 时**顺手改掉那句错 docstring**(§3.3)。

**③ 768P 素材?** 机器上已经有真实的 1344×768 素材:`/nfs-data/lx2v_test/assets/sr_768p_15s.mp4`
(372 帧、24 fps、15.5 s),可以直接当 A4/A5 的 768P 15 s 用例,不必从 720P 缩放。
还缺的是 **1024×768 和 1366×768**(验证宽补齐到 1376):这两个只能问业务要原始 H3 片段;
拿不到就用 720P 缩放,并在报告里注明"仅用于测速,不用于画质结论"(文档 §7.4 已经这么写了,正确)。

---

## 5. 保留的风险(开发要盯住的)

| 风险 | 触发点 | 处置 |
|---|---|---|
| KV 布局差异导致 A2' 不逐位 | M2 第一个断言 | 先只比 `k/v` 与 ring 前缀;真的不一致就把"接收端写进同样大小的 ring 缓冲"作为备选(数值不变,只是布局一致) |
| P2P 气泡:每层 3 个 ref 的 stage2 要等邻居 | M3/M5 | 文档 §3.2 的"先算 j≥a_r+c-1、再 wait"是对的;若仍见气泡,把 P2P 提前到上一层 stage2 期间 post |
| rank 不均衡(末位 rank 13 refs + 整条前缀解码) | M5 计时行 | 解码实测只有 0.35 s,暂时可以不处理;余数给前面的 rank 这个方向是对的 |
| 编译后仍重编译 | M4 | `warmup_resolutions` 必须含真实素材分辨率(§3.2) |
| 4 s 短素材:边界 KV 与长度无关 → 占 24% | §7.3 只要求"比 1 卡快" | 已是保守设定,保持 |

---

## 6. 附:本次实测的复现物

| 文件 | 用途 |
|---|---|
| `/data/verify-logs/relayvsr/profile/profile_serial3.py` | 串行路径 CUDA event 分解(不插 host sync) |
| `/data/verify-logs/relayvsr/profile/p2p_bench.py` | 4 卡两两 P2P 带宽(512 MiB,NCCL) |
| `/data/verify-logs/relayvsr/profile/enc_bench.py` | 真实 2K 画面 x265 编码吞吐(1 路 vs 4 路) |
| `/data/verify-logs/relayvsr/profile/io_bench.py` | 720P 748 帧解码吞吐(带/不带 RGB 转换) |

跑法(镜像 `arm64-a100-20261103-0647-e48faec8`):

```bash
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -e PYTHONPATH=/opt/pydeps:/repo/dockerfiles/decord_shim:/repo \
  -v /data/lx2v-pydeps:/opt/pydeps -v /data/lx2v-sync:/repo \
  -v /nfs-models:/nfs-models:ro -v /nfs-data:/nfs-data:ro -v /data/verify-logs:/data/verify-logs \
  -w /repo <image> -lc "python /data/verify-logs/relayvsr/profile/profile_serial3.py \
    --model_cls relayvsr_x2 --task sr --model_path /nfs-models/wuhanjisuan894/models/RelayVSR_X2 \
    --config_json /repo/configs/relayvsr/relayvsr_x2.json \
    --video_path /nfs-data/lx2v_test/relayvsr/src_720p_30s.mp4 --sr_ratio 2 --seed 42 \
    --save_result_path /data/verify-logs/relayvsr/profile/serial3.mp4"
```
