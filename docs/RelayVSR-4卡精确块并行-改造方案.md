# RelayVSR 4 卡精确块并行(exact chunk_p)改造方案

> 角色分工:本文由分析/验收方编写,开发 agent **按本文实施**。本文没写到的情况、或发现代码事实与本文描述不一致,
> **停下来上报,不要自行发挥**(尤其不要为了达到性能目标引入任何近似)。开发完成后按 §7 自测,
> 提交 §8 的自测报告,由验收方验收。
>
> 相关文档:[RelayVSR_X2-实测-2026-11-03](RelayVSR_X2-实测-2026-11-03.md)(现状数据)、
> [上游同步设计文档](上游同步-2026-11/设计文档.md)(SwiftVR 精确块并行的先例)。

---

## 0. 目标与范围

| 项 | 内容 |
|---|---|
| 业务场景 | 768P 输入(1024×768 / 1344×768 / 1366×768 等)、4–30 秒、24fps 左右的视频,`relayvsr_x2` 原生 2× 超分(输出 1536P) |
| 硬件 | 4 × A100-40G **PCIe**(无 NVLink),arm64 主机;机器 `dev-gpustack-a100-0007` |
| 目标 1(硬性) | 4 卡输出与 1 卡串行输出**数学等价**:不允许任何预热近似、接缝、交叉淡化。验收线见 §7.2 |
| 目标 2 | 4 卡相对 1 卡的加速比:30 秒级 ≥ 3.3×(目标 3.6×+),见 §7.3 |
| 范围内 | `chunk_p_size>1` 的视频路径整体替换为精确方案;计时埋点;相关单测、配置、对比脚本 |
| 范围外 | seq_p 路径(不动);图像路径(不动);1 卡串行路径的数值(**不得改变**);CUDA Graph、NUMA 绑核、INT8、换编码器(均为后续项,见 §9) |

---

## 1. 现状与问题

当前 `chunk_p_size=4`(提交 `5481c6a9`、`910e2578`)的做法:每个 rank 负责一段连续窗口,段首**重放
`temporal_context_frames` 个窗口**(reference + flash 全跑,输出丢弃)来近似恢复串行状态。

问题:

1. **不精确**:Wan 的 reference 链每层可回看 `c-1` 个历史 reference(`c = temporal_context_frames`),
   30 层叠加后感受野为 `层数 × (c-1)` 个 reference,对 4–30 秒视频基本覆盖全片 —— 有限预热永远做不到逐位一致。
   实测对齐版与单卡均值 44.4dB,最差帧未测;接缝处存在跳变风险。
2. **慢**:预热窗口的 flash 解码是纯浪费;后面的 rank 要先把前面所有帧解码 + 转 RGB + pin 才能开工。
   实测 748 帧 720P:1 卡 56.4s,4 卡对齐版 26.5s(1.78×)。
3. 我们 SwiftVR 自己做过一模一样的"连续切块 + 预热近似",已被上游精确方案替换
   (见上游同步设计文档 §2 第 107 行)。RelayVSR 现在重蹈覆辙。

---

## 2. 原理:为什么可以做到精确并行

### 2.1 串行语义(唯一的正确性基准)

以下全部来自 `lightx2v/models/runners/relayvsr/relayvsr_runner.py::restore_frames` 与
`lightx2v/models/networks/relayvsr/wan_infer.py`,**新实现必须逐条复现**。记 `L = clip_len`(默认 16),
`step = L - 1`,`total` 为帧数,`c = temporal_context_frames`(从 checkpoint 元数据读)。

1. **窗口**:第 `w` 个窗口起点帧 `w*step`,长度 `min(L, total - w*step)`;窗口 0 一定存在,其余窗口
   仅在长度 > 1 时存在。窗口数 `W = 1 if total <= L else 1 + ceil((total - L) / step)`。
2. **reference(关键帧)**:ref 0 = 第 0 帧;之后每个长度 > 1 的窗口贡献一个 ref = 该窗口最后一帧。
   即 ref `j`(j≥1)对应帧 `min(j*step, total-1)`;ref 总数 `R = W + 1`(`total == 1` 时 `R = 1`)。
   ref `j` 调用时 `reference_count == j`。
3. **噪声**:每个 ref 调一次 `scheduler.prepare_latents(padded_h, padded_w)`,按 ref 顺序从同一个
   generator(`scheduler.prepare(seed)`)依次抽取。第 `j` 个 ref 用第 `j` 次抽取。
4. **ref 前向**:`lr = preprocess(frame)` → `wan_pre.infer(latents.to(bf16), lr)` → 30 层 block →
   `infer_non_blocks` → unpatchify → `latents = latents - noise_pred.float()` →
   `latent = latents[:, :, 0].to(bf16)` → `flash_model_prepare_condition(latent)` 得到 condition。
5. **Wan 自注意力的 KV**(`RelayWanTransformerInfer.infer_self_attn`):每层把本 ref 的 K/V
   (`norm_k` 之后、**未旋转**、bf16)写入 frame ring 槽位 `slot = j % c`;注意力读取前
   `visible = min(j+1, c)` 个槽位,**按槽位顺序**拼接(不是时间顺序);槽位 `s` 中存的是 ref
   `j - ((j - s) % c)`;RoPE 位置由 `prepare_input(frame=j)` 给出(`positions` 元组按槽位排列,
   `origin = max(0, j - training_frames + 1)`)。注意力调用为
   `SpatialAttentionSpec(1, h, w, wh, ww, c, key_frames=visible, query_frame_offset=visible-1)`。
6. **flash 解码**:窗口 `w` 使用 condition(ref `w`) 作起点、condition(ref `w+1`) 作终点
   (`total == 1` 时终点 = 起点);窗口开头 `flash_model_history.reset()`;窗口内按 offset 0..len-1
   逐帧 `flash_model_decode(lr, conditions, history, offset)`。
7. **输出**:窗口 0 输出全部帧;其余窗口不输出 offset 0(它是上一窗口末帧)。输出裁剪为
   `[..., :h*scale, :w*scale]` 后 `postprocess`。

### 2.2 依赖关系

- ref `j` 的**输入**只有"第 j 次抽取的噪声 + 第 j 个关键帧的低清图",都可以提前得到;ref 的**输出不回灌**给后续 ref。
- ref 之间唯一的依赖:**第 ℓ 层**的注意力需要前 `c-1` 个 ref 在**同一第 ℓ 层**的 K/V;而第 ℓ 层的
  K/V 只由该 ref 第 ℓ 层的**输入**算出(与同层其它 ref 的注意力结果无关)。
- flash 窗口之间完全独立(history 每窗口 reset),只依赖两端 ref 的 latent。

所以:把 ref 按段分给各 rank,**所有 rank 按层同步推进**——每层先算本段全部 ref 的 Q/K/V,
把段尾 `c-1` 个 ref 的 K/V 发给下一个 rank,再做注意力及其后的部分。这与 SwiftVR 上游"ReAE 各层边界经
ring P2P 交换"是同一类结构,结果与串行数学等价;若各 ref 的计算仍按单 ref 形状逐个执行,
则与串行**逐位相同**(§7.2 A2 验证这一点)。

---

## 3. 总体设计

### 3.1 切分

- 以**窗口**为单位连续切分给 `P = chunk_p_size` 个 rank:`base = W // P`,`extra = W % P`,
  前 `extra` 个 rank 各多 1 个窗口(余数给前面的 rank,因为最后一个 rank 还要多算 1 个 ref)。
- rank `r` 拥有窗口 `[ws_r, we_r)`,拥有 ref `[ws_r, we_r)`;**最后一个非空 rank** 额外拥有 ref `W`
  (= R-1)。窗口数为 0 的 rank 是空闲 rank,不参与计算,只参与 §3.5 规定的同步点。
- rank `r` 的窗口 `we_r - 1` 需要的终点 ref `we_r` 属于下一个非空 rank(它的第一个 ref),在 Phase A 结束后
  点对点拿该 ref 的 **latent**(`[1,48,h,w]` bf16,768P 约 1.6MB),本地再算 condition。
- rank `r` 需要解码的帧区间 `[ws_r*step, min((we_r-1)*step + L, total))`;输出帧区间:
  `ws_r == 0` 时从第 0 帧开始,否则从 `ws_r*step + 1` 开始,到区间末尾。各 rank 输出区间首尾相接、无重叠、
  并集 = 全片(§7.1 单测保证)。

### 3.2 Phase A:分层同步计算 reference

每个非空 rank:

1. **噪声**:`scheduler.prepare(seed)` 后循环 `j = 0 .. b_r-1`(`b_r` 为本 rank 最后一个 ref + 1)逐次调用
   `prepare_latents`,只保留 `j ≥ a_r` 的结果(`clone`)。**必须逐个抽取**,不能一次抽大张量再切
   (Philox 偏移不同,结果不同)。
2. **pre-infer**:对每个自有 ref 单独调用 `wan_pre.infer`(保持与串行相同的张量形状)。
3. **逐层(layer-major)**,对 block `ℓ = 0..num_layers-1`:
   - **stage1**(逐 ref):`pre_process` 取调制量 → `norm1` + modulate → `q/k/v` 投影与 `norm_q/norm_k`,
     得到每个 ref 的 `q, k, v`(未旋转,形状与串行完全相同)。
   - **exchange**:按 §3.2.1 的计划,把本段尾部 ref 的 `k, v` 发给需要它们的 rank,接收本段前缀
     ref 的 `k, v`。用 `dist.batch_isend_irecv`。
   - **stage2**(逐 ref):为 ref `j` 按 §2.1 第 5 条**槽位顺序**拼出 `visible` 个 ref 的 K/V
     (来源:本段 stage1 结果或接收到的前缀),取 `prepare_input(frame=j)` 的 `q_rope/k_rope`,
     调用与串行**同一个**注意力函数及参数 → `self_attn_o` → cross-attn → FFN → `post_process`,
     得到 ref `j` 的下一层输入。
   - **通信隐藏(要求实现)**:stage2 中不依赖前缀的 ref(`j ≥ a_r + c - 1`)先算,
     再 `wait()` 通信,最后算前 `c-1` 个 ref。stage2 各 ref 之间互不依赖,顺序不影响结果。
4. **post**:逐 ref `infer_non_blocks` → `wan_post.infer` → `latents - noise_pred.float()` →
   `latents[:, :, 0].to(bf16)`。

#### 3.2.1 KV 交换计划

- rank `r`(自有 ref `[a_r, b_r)`)需要的前缀 ref:`[max(0, a_r - (c-1)), a_r)`。每个前缀 ref 的所属
  rank 按切分表查出;按来源 rank 分组成连续区间。**前缀可能跨多个 rank**(短视频每段 ref 数 < c-1 时),
  计划必须支持。
- 发送方发送的都是**自己的** ref(不转发别人的),因此同一层内所有收发可以一次 `batch_isend_irecv`
  完成,没有跨 rank 的串行链。
- 每层每个区间的 K/V 打包成一个张量 `[2, m, tokens, heads, head_dim]` 再发,减少 P2P 次数。
- 计划是 `(spans, c)` 的纯函数,所有 rank 独立算出同一份,不需要额外通信。
- NCCL 注意:`batch_isend_irecv` 前该进程组必须已经做过一次全员集合通信(进程组已有
  `broadcast_object_list`,保持这个顺序即可);没有收发任务的 rank 这一层直接跳过。
- 通信量估算(需在 M0 用真实配置算出并写进报告):每个 rank 边界每层
  `(c-1) × tokens × dim × 2(k,v) × 2B`;768×1376 输入时 `tokens = 48×86 = 4128`。

### 3.3 Phase B:flash 解码

- 拿到下一个 rank 的首 ref latent 后,对本 rank 需要的每个 ref 调用一次
  `flash_model_prepare_condition`(相邻窗口共享同一个 ref 的 condition,不重复算)。
- 窗口 `w ∈ [ws_r, we_r)` 完全按 §2.1 第 6、7 条执行;使用本 rank 自己的 `flash_model_history`。
- 输出送 `AsyncVideoWriter` 写本 rank 的分片文件;Phase B 期间 `check_stop` 用 rank 本地判定
  (沿用 `_rank_local_collectives`)。

### 3.4 视频 IO

- **总帧数**:用 demux 计数(只读包、不解码)得到 `total`,与 `stream.frames` 不一致时打 warning;
  所有 rank 用同一方法算出相同值(或 rank0 算出后广播 —— 推荐广播,保证一致)。
  解码过程中若实际帧数与预期不符,**直接报错**,不得静默截断或补帧。
- **区间解码**:rank 从头解码时,对区间起点之前的帧**只解码、不调用 `to_ndarray`、不 pin**。
  Phase A 需要的关键帧:第一遍扫描本区间时只转换关键帧;Phase B 再按现有 `prefetch_frames` 流式读本区间
  (两遍解码,内存有界)。`container.seek()` **不做**(变帧率 / B 帧下按帧号定位有风险),列为后续项。
- **编码线程**:未在配置中显式给出 `encoder_threads` 时,`chunk_p_size>1` 下取
  `min(16, max(1, len(os.sched_getaffinity(0)) // chunk_p_size))`(与 SwiftVR `swiftvr_runner.py:358` 一致)。
- **拼接**:沿用现有 `_run_video_chunks` 的分片 + `ffconcat` + `mux_audio_from_video` 逻辑。

### 3.5 同步点、取消与异常

- 全员(含空闲 rank)必须到达的同步点:请求开始(广播 `total`、工作目录)、Phase A 开始前的
  `check_stop`(全局 MAX all-reduce)、Phase A 结束后的 `check_stop`(全局)、首 ref latent 交换、
  分片写完后的 barrier、拼接后的 barrier。
- Phase A 内部(逐层 P2P)不调用 `check_stop`(否则空闲 rank 无法配对);取消延迟上限约为一次 Phase A 的耗时,
  在报告中给出实测值。
- 任一 rank 在 Phase A 抛异常时,其余 rank 会阻塞在 P2P 上直到 NCCL 超时。本期接受这个行为,但要求:
  异常 rank 必须先打印完整 traceback;报告中写明实测的超时时间和服务恢复情况。

### 3.6 编译与预热

- stage1、stage2 分别 `torch.compile`,编译缓存 key 与串行的 `infer_block` 区分开
  (例如 `("stage1", block_idx)`),沿用 `mark_sequence_dynamic` 和 `compile_options`
  (`emulate_precision_casts=True`、`pattern_matcher=False`)。注意力 geometry 中的
  `visible`(1..c)会产生最多 c 个变体,这与串行相同。
- `run_warmup`:`chunk_p_size>1` 时用合成帧走一遍**精确路径**(所有 rank 一起),帧数要保证每个 rank 至少
  `c+1` 个 ref,使 `visible = 1..c` 的变体和 P2P 都被预热到。
- 预热分辨率改成可配置:新增配置 `warmup_resolutions`(列表,缺省为类常量 `WARMUP_RESOLUTIONS`)。
  768P 生产配置写 `[[768, 1024], [768, 1344], [768, 1376]]`(1366 宽补齐到 1376)。
- 报告中给出:`TORCH_LOGS=recompiles` 下,预热之后正式请求是否还有重编译(要求为 0)。

### 3.7 配置

- `parallel.chunk_p_size > 1` 即启用精确方案;**删除旧的近似实现**(预热重放、priming、噪声快进)。
  不保留开关(旧实现可从 git 历史找回)。
- 精确方案与 `seq_p_size>1`、`cpu_offload`、`lazy_load` 互斥:在 `validate_config` 里明确报错。
- 新增可选配置:`relay_timing_log`(bool,默认 false,开启后输出 §4.6 的计时行)。
- 更新 `configs/relayvsr/relayvsr_x2_chunkp4.json`,新增 `relayvsr_x2_chunkp2.json`,两者都写上
  `warmup_resolutions`。

---

## 4. 代码改动清单

> 原则:串行路径(`chunk_p_size=1`)**数值不能变**,只允许把公共的计算抽成函数供两条路径复用;
> 所有新逻辑放在 relayvsr 目录内;不修改 `lightx2v/models/networks/wan/**` 等共享文件。

### 4.1 新文件 `lightx2v/models/networks/relayvsr/exact_chunk.py`(纯逻辑 + 通信)

```python
@dataclass(frozen=True)
class RelayPlan:
    total: int
    clip_len: int
    step: int
    windows: int          # W
    refs: int             # R
    def window_frames(self, w: int) -> range: ...   # 窗口 w 覆盖的帧
    def emitted_frames(self, w: int) -> range: ...  # 窗口 w 输出的帧
    def ref_frame(self, j: int) -> int: ...

def build_plan(total: int, clip_len: int) -> RelayPlan: ...

@dataclass(frozen=True)
class RankSpan:
    rank: int
    windows: range        # [ws, we)
    refs: range           # 自有 ref(最后一个非空 rank 含 R-1)
    decode_frames: range  # 需要解码的帧
    emit_frames: range    # 需要输出的帧

def split_ranks(plan: RelayPlan, world: int) -> list[RankSpan]: ...

@dataclass(frozen=True)
class KVTransfer:
    src: int
    dst: int
    refs: range           # 连续的全局 ref 区间,均属于 src

def kv_transfer_plan(spans: list[RankSpan], context: int) -> list[KVTransfer]: ...

class KVExchange(Protocol):
    def start(self, layer: int, own_kv: dict[int, tuple[Tensor, Tensor]]) -> None: ...
    def wait(self) -> dict[int, tuple[Tensor, Tensor]]: ...   # 全局 ref 号 -> (k, v)

class NcclKVExchange: ...   # batch_isend_irecv 实现
class LocalKVExchange: ...  # 单进程"虚拟多 rank"实现,供 §7.2 A2 使用
```

`RelayPlan` / `split_ranks` 的窗口、ref、输出语义必须与 §2.1 完全一致,用 §7.1 单测对照串行生成器验证。

### 4.2 `lightx2v/models/networks/relayvsr/wan_infer.py`

- 把 `infer_self_attn` 拆成两个由串行和精确路径**共用**的函数:
  - `project_qkv(phase, x, shift_msa, scale_msa) -> (q, k, v)`
  - `attend(phase, q, keys, values, q_rope, k_rope, visible, height, width) -> attn_out`(含 `self_attn_o`)
  串行 `infer_self_attn` = `project_qkv` → 写 ring → 读 ring → `attend`,运算顺序与现在完全一致。
- 新增 block 级两阶段函数:
  - `infer_block_stage1(block, x, pre) -> (q, k, v)`
  - `infer_block_stage2(block, x, pre, q, keys, values, q_rope, k_rope, visible) -> x`
    (内部为 `attend` → cross-attn → FFN → `post_process`,与 `WanTransformerInfer.infer_block`
    的剩余部分逐行对应;cross-attn 的 context 必须像现有 `run_block` 一样替换为
    `self.cross_cache[block_idx]`)
- 新增 `infer_layer_major(weights, hidden: list[Tensor], first_ref: int, height, width, rope, exchange) -> list[Tensor]`:
  实现 §3.2 第 3 步;调用 `reset_infer_states` / 设置 `_output_tokens` 等与串行 `infer` 等价的前置状态;
  需要编译时走 §3.6 的编译缓存。

### 4.3 `lightx2v/models/networks/relayvsr/model.py`

- 新增 `RelayVSRModel.infer_references(noises: list[Tensor], lrs: list[Tensor], first_ref: int, exchange) -> list[Tensor]`:
  完成 §3.2 第 2–4 步,返回每个自有 ref 的 `latent`(bf16,`[1,48,h,w]`)。
- `flash_model_prepare_condition` / `flash_model_decode` 不改。

### 4.4 `lightx2v/models/runners/relayvsr/relayvsr_runner.py`

- 删除:`_chunk_windows`、`_chunk_spans`、`_run_video_chunks` 中的预热与 priming 逻辑、`restore_frames`
  的 `priming` 参数及噪声快进(恢复为上游串行语义)。
- 新增 `_run_video_exact(input_info, source, destination, fps, target)`:IO(§3.4)→ Phase A(§3.2)→
  首 ref latent 交换 → Phase B(§3.3)→ 拼接,同步点按 §3.5。
- `init_run`:精确路径不创建 Wan frame ring cache(省显存),flash history 照旧创建。
- `run_warmup`:§3.6。
- `validate_config`:§3.7 的互斥检查。

### 4.5 其它

- `test_cases/test_relayvsr_chunk_spans.py`:替换为 §7.1 的新单测(旧测的 `_chunk_spans` 已删除)。
- `scripts/relayvsr/compare_outputs.py`(新):对比两个视频,输出逐帧 PSNR(相同记为 inf)、min/mean、
  低于阈值的帧号、指定边界帧附近 ±5 帧的 PSNR、帧数、视频时长、音轨时长。
- `scripts/relayvsr/bench_encoder.py`(新):用当前 `AsyncVideoWriter` 的编码参数,对合成的
  2048×1536 / 2688×1536 帧测 1 路与 4 路并发编码 fps。
- `configs/relayvsr/relayvsr_x2_chunkp4.json`、`relayvsr_x2_chunkp2.json`:§3.7。
- `docs/RelayVSR_X2-实测-2026-11-03.md`:末尾追加一节,链接到自测报告。

### 4.6 计时埋点(`relay_timing_log: true` 时)

每个请求每个 rank 输出一行(串行路径同样输出对应字段):

```
[RelayVSR][timing] rank=r world=P frames=N refs=n windows=m decode_keyframes=… noise=… wan_pre=… \
wan_stage1=… wan_comm_wait=… wan_stage2=… wan_post=… flash_cond=… flash=… encode_wait=… \
barrier_wait=… assemble=… total=…
```

只在阶段边界 `torch.cuda.synchronize()`(`wan_comm_wait` 每层同步一次);关闭时不得有额外同步。
`encode_wait` = `AsyncVideoWriter.submit` 因队列满而阻塞的累计时间。

---

## 5. 实施步骤(按顺序;每步达到退出条件再进入下一步)

| 步骤 | 内容 | 退出条件 |
|---|---|---|
| **M0 事实采集** | ① 打印 `wan_x2.safetensors` 元数据中的 `num_layers`、`num_attention_heads`、`attention_head_dim`、`temporal_context_frames`、`temporal_rope.training_frames`、`spatial_window_size`、FFN 隐层维度(键名以元数据为准);② `nvidia-smi topo -m`、`nvidia-smi topo -p2p r`、`lscpu`、`numactl -H`;③ GPU 两两之间 P2P 带宽(torch 小脚本,256MB 张量 `send/recv`);④ 在串行路径加 §4.6 埋点,测 768P 30s 素材的单 ref 平均耗时、单帧 flash 平均耗时、`encode_wait`;⑤ `bench_encoder.py` 结果 | 数据写进报告 §8.1;算出每层 KV 交换量与预计耗时。**若预计每层通信时间 > stage2 耗时的 50%,停下上报** |
| **M1 纯逻辑** | `exact_chunk.py` 中 `build_plan`、`split_ranks`、`kv_transfer_plan` + §7.1 单测 | 单测全过 |
| **M2 单卡虚拟多 rank** | §4.2、§4.3 拆分与 `infer_layer_major`;`LocalKVExchange`;单进程内用虚拟 P=1/2/3/4 跑 Phase A + Phase B | §7.2 A2(eager 逐位相同)、A3(串行回归)通过 |
| **M3 真多卡** | `NcclKVExchange`、`_run_video_exact`、IO、同步点、拼接 | 4 卡、2 卡能跑通 §7.4 全部素材;A4、A5 通过 |
| **M4 编译与预热** | §3.6 | 预热后正式请求零重编译;A6 通过 |
| **M5 性能** | 通信隐藏、编码线程;跑 §7.3 | 填完报告 §8.3;未达标时附计时分解与原因分析 |
| **M6 收尾** | 删除旧代码、ruff、文档 | §7.5 全部满足 |

---

## 6. 开发约束

1. 不得为了性能引入近似(批量 GEMM 合并多个 ref、改变注意力 K/V 顺序、改变 RoPE origin 等)。
   这类优化若有价值,在报告里写成提案,由验收方决定。
2. 每个 ref 的 GEMM / 归一化 / 注意力调用保持与串行**相同的张量形状和调用顺序**(层内跨 ref 的顺序可以变)。
3. 不修改共享文件(`lightx2v/models/networks/wan/**`、`lightx2v/common/**`、`base_runner.py` 等)。确有必要时先上报。
4. 遵循仓库风格:ruff(`line-length = 200`)、不加模板化注释和过度防御;内部配置路径不做输入格式穷举
   (参考 `.agents/skills/remove-ai-code-smell`)。
5. 不提交权重、测试视频、输出视频;测试产物放 `/nfs-data/lx2v_test/relayvsr/exact/` 下并在报告中给出路径。
6. 每完成一个里程碑提交一次 commit(中文提交信息,风格同 `git log`),不 push。

---

## 7. 验收标准

### 7.1 单测(CPU,`pytest test_cases/test_relayvsr_exact_chunk.py`)

- **串行对照生成器**:在测试里用纯 Python 复刻 `restore_frames` 的帧流(帧用整数代替),得到串行的
  ref 帧号序列、每个窗口的帧、输出帧序列。
- 参数:`total ∈ {1, 2, 15, 16, 17, 30, 31, 32, 96, 97, 360, 374, 720, 748}`、`clip_len ∈ {16, 32}`、
  `P ∈ {1, 2, 3, 4, 8}`、`c ∈ {1, 2, 3, 4, 8}`。断言:
  1. `build_plan` 的窗口 / ref / 输出与串行生成器逐项相同;
  2. `split_ranks`:各 rank 的 `emit_frames` 首尾相接、无重叠,并集 = `range(total)`;每个 rank 的
     `decode_frames` 覆盖其全部窗口帧和自有 ref 帧;自有 ref 并集 = `range(R)`,无重复;
  3. `kv_transfer_plan`:每个非空 rank 收到的 ref 恰好是 `[max(0, a_r-(c-1)), a_r)`,且每个区间的 `src`
     确实拥有这些 ref;
  4. 空闲 rank(W < P)不出现在任何传输里。

### 7.2 数值(GPU)

所有对比统一使用**无损编码**,避免 x265 有损编码本身带来的差异:在配置中覆盖
`"video_codec": "libx264", "video_crf": 0`(yuv420p 无损),用 `compare_outputs.py` 比较。

| 编号 | 内容 | 要求 |
|---|---|---|
| A2 | 单卡、`use_compile=false`,虚拟 P=1/2/3/4(`LocalKVExchange`),768P 96 帧与 360 帧素材 | 所有 ref latent 与串行 `reference()` 序列 `torch.equal`;所有输出帧与串行 eager 输出**逐位相同** |
| A3 | `chunk_p_size=1` 串行路径,改动前提交 vs 改动后,768P 360 帧,编译开启 | 全部帧 PSNR = inf |
| A4 | 真 4 卡精确方案 vs 1 卡串行,编译开启,§7.4 全部素材 | 逐帧 **min PSNR ≥ 50dB**(目标为 inf,如实报告);帧数 = 源帧数;报告每个 rank 边界前后 ±5 帧的 PSNR |
| A5 | 真 2 卡,同 A4 | 同 A4 |
| A6 | 生产配置(x265、crf18),4 卡 | 能播放;帧数 = 源帧数;视频时长与源相差 ≤ 1 帧;音轨存在且时长与源相差 ≤ 1 帧 |

### 7.3 性能

条件:空机;服务模式常驻(`torchrun --nproc_per_node=4 -m lightx2v.server ...`,1 卡用
`relayvsr_x2.json` 起单进程服务);同一请求连续提交 3 次,取第 2、3 次 `RUN pipeline` 耗时的平均;
1 卡与 4 卡使用相同的 `clip_len`、编码参数和预热分辨率。

| 素材 | 要求(4 卡 / 1 卡) | 目标 |
|---|---|---|
| 720P 748 帧(历史基准,1 卡 56.4s) | ≥ 3.3× | ≥ 3.6× |
| 768P 30 秒 | ≥ 3.3× | ≥ 3.6× |
| 768P 15 秒 | ≥ 3.0× | ≥ 3.4× |
| 768P 4 秒 | 快于 1 卡,如实报告 | — |
| 2 卡(768P 30 秒) | ≥ 1.8× | ≥ 1.9× |

附加要求:

- 每卡显存峰值 ≤ 24GiB(768P 30 秒,`nvidia-smi` 采样峰值);
- 1 卡串行耗时与改动前相差不超过 ±3%;
- 未达标时**不得**通过近似手段凑数;提交 §4.6 计时分解,并说明瓶颈(通信、编码、解码、CPU 发射等)。

### 7.4 测试素材

放在 `/nfs-data/lx2v_test/relayvsr/exact/`:

1. 已有的 720P 125 / 374 / 748 帧(`/nfs-data/lx2v_test/relayvsr/`);
2. 768P:优先用真实 H3 产出的 1024×768、1344×768 片段;没有就从 720P 素材用
   `ffmpeg -vf scale=1344:768` 生成 4 秒(96 帧)、15 秒(360 帧)、30 秒(720 帧)各一条,再生成一条
   1366×768 的 15 秒片段(验证补齐到 1376 宽);
3. 变帧率片段:`ffmpeg -i <15s> -vf mpdecimate -vsync vfr vfr_15s.mp4`;
4. 带 B 帧的片段(x264 默认参数编码即可)。

### 7.5 工程

- `ruff check` 与 `ruff format --check` 对改动文件通过;`pytest test_cases/test_relayvsr_exact_chunk.py` 通过;
- 旧的近似实现已全部删除,`grep -n "priming\|warmup_windows\|_chunk_spans" lightx2v/` 无结果;
- 取消测试:4 卡运行 30 秒素材,分别在 Phase A 期间、Phase B 期间发出停止请求,所有 rank 均正常退出本请求,
  下一个请求能正常完成;报告取消延迟;
- 改动文件清单写进报告(验收方据此更新上游同步设计文档 §3 的保留清单)。

---

## 8. 自测报告(开发 agent 提交)

文件:`docs/RelayVSR-4卡精确块并行-自测报告.md`,包含:

1. **M0 事实**:checkpoint 配置值、拓扑 / P2P 带宽、串行计时分解、编码基准、每层 KV 交换量与预计耗时;
2. **数值**:A2–A6 结果表(素材、P、min/mean PSNR、是否逐位相同、帧数、时长),边界帧 PSNR 明细,
   `compare_outputs.py` 原始输出路径;
3. **性能**:§7.3 表格(1 卡 / 2 卡 / 4 卡耗时、加速比、显存峰值),每个 rank 的 §4.6 计时行原文;
4. **重编译**:`TORCH_LOGS=recompiles` 结论;
5. **取消 / 异常**:实测行为和延迟;
6. **改动文件清单与 commit 列表**;
7. **偏差与遗留**:与本文不一致的地方、未达标项及原因、建议的后续优化。

---

## 9. 后续项(本期不做,报告里可给数据和建议)

| 项 | 说明 |
|---|---|
| flash 解码 CUDA Graph | flash 模型很小(dim 512),发射开销占比可能较高;需要先看 §4.6 计时 |
| NUMA / 绑核 | 每个 rank 绑到离 GPU 最近的 NUMA 节点 |
| 编码瓶颈 | A100 没有 NVENC;若 `bench_encoder.py` 显示 4 路 x265 合计吞吐不够,再评估 preset / 线程配置 |
| `container.seek()` | 必须按 PTS 校验帧号,变帧率素材通过后才能启用 |
| 批量 GEMM | 多个 ref 合并成一次大 GEMM 可能更快,但数值会变,需要单独评估并由验收方决定 |
| 一卡多进程(MPS) | 每卡约 17GB,40G 卡可放 2 个进程;若 SM 利用率低再评估 `chunk_p_size=8` |

---

## 10. 遇到以下情况立即停下上报

- M0 发现 `temporal_context_frames`、RoPE 策略或注意力调用与 §2.1 描述不符;
- A2 做不到逐位相同,且无法定位到某个具体算子;
- 每层通信预计或实测超过 stage2 耗时的 50%;
- 显存超过 24GiB 且无法在不改变数值的前提下降下来;
- 需要修改 §6 第 3 条列出的共享文件;
- 任何需要引入近似才能继续的情况。
