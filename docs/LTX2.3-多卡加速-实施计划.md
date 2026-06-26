# LTX2.3 多卡加速 — 实施计划

> 2026-06-26。目标:为**高分辨率/长视频**场景让 LTX2.3 单条出片用上多卡(TP/ulysses)提速。
> 前置结论见:`LTX2.3-单条延迟优化记录.md`、`LTX2.3-int8量化-可行性与验证方案.md`、记忆 `ltx23-multicard-oom-40gb` / `ltx23-int8-no-speed-bad-quality`。

---

## 1. 为什么做 / 什么时候才值得做

单条 100s 的分段(121帧/1280×768,profiler 实测):gemma 44s(44%)+ **DiT 31s(31%)** + VAE/音频/存盘 24s。负向跳过优化后已是 86s。

- **当前规格多卡不值**:DiT 只占 31%,4 卡天花板 ~1.28×(86→~66s),还占 4 卡/任务(吞吐掉 4 倍)。
- **大规格才值**:DiT 随**分辨率**(空间 token,注意力 ~O(N²))和帧数增长;gemma 恒定 ~27s。规格一大,DiT 占比冲到 60-70% → 多卡 2-3×;且单卡可能**激活 OOM**,ulysses 切激活成刚需。
- **结论**:多卡是"为大视频备的弹药"。先验证收益是否随规格放大,再决定投入。

## 2. 当前已知约束(实测)

| 方案 | 40GB 单卡能跑? | 卡点 |
|------|:---:|------|
| bf16 单卡 + block offload | ✅(峰值 18.9GB) | — |
| bf16 ulysses-4 | ❌ CPU OOM | 每 rank 复制 46GB → 4×=184GB 撑爆内存 |
| bf16 TP-4 | ❌ GPU OOM | `_load_weights_from_rank0` 在 rank0 GPU 暂存整 46GB > 40GB |
| **int8 单卡** | ✅(常驻 20GB) | 速度=bf16、画质崩(全块量化) |
| **int8 ulysses-4** | ✅(预期,每卡 20GB) | 待验证;画质需精修 |
| **int8 TP-4** | ✅(预期,rank0 暂存 20GB) | 待验证;画质需精修 |

## 3. Roadmap

### Phase 0 — 廉价验证:多卡到底提不提速(先做,~半小时)
**用现有(崩画质)int8 权重 + ulysses-4,只测速度不看画质。**
- int8 20GB/卡 → ulysses 装得下(bf16 那次是 CPU 复制爆,int8 不会)。
- config:`ltx2_int8_triton.json` + `parallel:{seq_p_size:4, seq_p_attn_type:"ulysses"}`;启动 `torchrun --nproc_per_node=4 -m lightx2v.server ...`。
- **同时测一个大规格**(更高分辨率 / 更多帧):看 ① DiT 占比 ② 单卡是否 OOM ③ 4 卡相对单卡省多少。
- **决策门**:DiT 能被拆(如 31s→~12s)→ 继续 Phase 1/2;拆不动 → 两条都放弃。

### Phase 1 — bf16 + TP 加载器修复(满血画质,优先)
TP 分片权重 → bf16 装得下(46/4≈11.5GB/卡),**无 int8 画质问题**。
1. 改 `lightx2v/models/networks/ltx2/model.py::_load_weights_from_rank0`:每 rank 只读自己 1/4 分片,**不在 rank0 上铺整模型**。两种改法:
   - A(简单):rank0 在 **CPU** 读+切,只把每片发对应 GPU(NCCL 需先把片搬 GPU)。
   - B(干净):每 rank 用 safetensors **惰性切片**直接从磁盘读自己那片到 GPU(无 rank0 暂存、无大块 NCCL、并行读盘)。
2. bind-mount 改后的文件测 `ltx2_tp4_test.json`:不 OOM + 出片 + 和 bf16 基准**画质一致** + DiT 提速。
3. 走 fork 补丁还是上游 PR:per-rank 分片加载对所有"卡比模型小"的人都有用,可考虑 PR。

### Phase 2 — int8 + ulysses(超大视频的最后手段)
只在"目标规格下 TP 的激活也装不下、必须 ulysses 切激活"时才需要。
1. int8 重量化加 skip 敏感块 `[0,43,44,45,46,47]`(官方 fp8 配置同款;首块+末5块)。
   - converter:`--ignore-quant-keys transformer_blocks.0.,transformer_blocks.43.,...,transformer_blocks.47.`
   - 推理 config:`skip_fp8_block_index:[0,43,44,45,46,47]` 与之一致。
2. 单卡测画质,和 5 条 bf16 基准片(`/data/outputs/baseline/bf16_*.mp4`)逐条抽帧对比。
3. 不够再跳块内 `to_gate_logits` + 跨模态注意力(`audio_to_video`/`video_to_audio`)。
4. 画质达标 → int8 + ulysses-4 测大规格出片。

### Phase 3 — 在目标规格上定方案
- 中大规格、TP 激活够 → **bf16 TP**(满血,首选)。
- 超大规格、TP 激活爆 → **int8 ulysses**(切激活,唯一能跑)。

## 4. 关键决策点(待定)

1. **要支持的最大视频规格**(分辨率 × 时长)= ?  ← 决定这两条重活到底要不要做、做到哪。
2. Phase 0 的多卡 DiT 提速是否成立(数据说话)。
3. bf16 TP 在目标规格下激活是否装得下(决定是否需要 int8 ulysses)。

## 5. 注意事项 / 资产

- **int8 那 26GB 权重(`/data/models/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1-int8/`)Phase 0 要用,务必先别删。**
- bf16 基准片 `/data/outputs/baseline/bf16_*.mp4` + `/data/gen_baseline.py`:画质对比基准,保留。
- 代码改动用 bind-mount 测(像负向跳过那样),验证通过再提交 + 出包。
- 排序原则:**Phase 0 先验证收益 → bf16 TP 优先(满血)→ int8 ulysses 兜极端规格。** 避免在未验证前提下投入硬工程。
