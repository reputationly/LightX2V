# LTX2.3 单条延迟优化记录

> 2026-06-25 在 A100(4×40GB,sm_80,ARM/鲲鹏)上对 LTX2.3 22B 蒸馏单卡单条出片(1280×768/121帧/8步)做的一轮延迟优化。**净结果:单条 ~102s → ~86s(1.19×),无损,已落地。**

---

## 1. 起点:单条 ~100s 的分段(121帧,server 路径,profiler 实测)

| 阶段 | 耗时 | 占比 | 能并行(TP)? |
|------|------|------|:---:|
| **gemma 文本编码** | **44s** | **44%** | ❌ |
| **DiT 去噪**(stage1 18s + stage2 上采样 13s) | 31s | 31% | ✅ |
| VAE 解码 | 1s | 1% | ❌ |
| 音频生成(vocoder)+ 存视频(t2av) | ~24s | 24% | ❌ |
| **RUN pipeline 合计** | **100.5s** | | |

**关键洞察:瓶颈是 gemma(44%),不是 DiT(31%)。** 任何"并行 DiT"的方案天花板都被 gemma + VAE + 存盘(共 68%)锁死。

---

## 2. 试过的杠杆与结论

| 杠杆 | 结果 | 原因 |
|------|------|------|
| **int8 量化 DiT** | ❌ 无加速 + 画质崩(5/5 不可用) | int8-triton 没吃到 INT8 tensor core 收益;W8A8 全块量化未跳敏感层。详见 `LTX2.3-int8量化-可行性与验证方案.md` / 记忆 `ltx23-int8-no-speed-bad-quality` |
| **TP 多卡(并行 DiT)** | ❌ 天花板仅 ~1.25× | DiT 只占 31%;且 TP 在 40GB 上加载 OOM(见记忆 `ltx23-multicard-oom-40gb`) |
| **gemma 不量化常驻 GPU** | ❌ OOM | gemma bf16 24GB + DiT block-offload ~16GB > 40GB(实测 t=33s 冲到 40431MiB 崩) |
| **✅ CFG 关闭时跳过负向 prompt 编码** | ✅ **44s→27s,总 102s→86s,无损** | 见下 §3 |

---

## 3. 落地的优化:CFG 关闭时跳过负向编码

### 原理
- 蒸馏配置 `enable_cfg=false`(`sample_guide_scale=1`,guidance 已蒸进模型)。
- `pre_infer.py` 的 `infer_condition` 分支:CFG 关时只取 `v_context_p`(正向),**`v_context_n`(负向)从不被消费**。
- 而 `run_text_encoder` 原本**无条件编码正向+负向 2 个 prompt** → 负向那次纯属浪费。

### 改动(`lightx2v/models/runners/ltx2/ltx2_runner.py::run_text_encoder`,commit `e2883f85`)
```python
if self.config.get("enable_cfg", False):
    v_context_p, a_context_p, v_context_n, a_context_n = self.text_encoders[0].infer(prompt, neg_prompt)
else:
    # CFG 关:负向 context 不被消费,跳过其编码;占位符不会被用到
    ((v_context_p, a_context_p),) = self.text_encoders[0].encode_text([prompt])
    v_context_n, a_context_n = v_context_p, a_context_p
```

### 实测(night_market,seed 42,121帧)
| 指标 | 改前 | 改后 |
|------|------|------|
| Run Text Encoder | 44s | **27s** |
| RUN pipeline | 100.5s | **85.5s** |
| 单条总耗时 | 102s | **86s** |
| 产物 | 2061426 字节 | **md5 完全一致(无损)** |

> gemma 27s 而非 22s:正向 prompt 比空负向长,单独编正向就要 27s;省掉的是负向那 ~17s。

### 安全性(边界已核)
- `enable_cfg=true`:走原分支,完全不变。
- i2v / 其他 task:`run_text_encoder` 一致,只看 enable_cfg。
- cfg_parallel / mm_guider:都在 `if enable_cfg:` 内,不受影响。
- 占位符设为正向副本(非 None),即便下游误读也不 crash。

---

## 4. 部署注意
- 当前服务器用 **bind-mount** 注入补丁:`-v /data/patches/ltx2_runner.py:/opt/LightX2V/.../ltx2_runner.py`。
- 代码已进 `origin/main`(`e2883f85`)→ **下次出 app 包**(`build-arm64-docker.yml`)即烘入镜像,届时**不再需要 bind-mount**。

---

## 5. 剩余杠杆与建议
- 仅剩 **gemma 正向编码 27s**(仍走逐层搬运)。要砍它需**量化 gemma 让其常驻 GPU**(int4 ~6GB),但 bitsandbytes 未装、ARM 支持不确定;torchao 在但有 Tensor 子类警告。**投入产出比差,暂不做。**
- **结论:接受 ~86s 为 LTX2.3 单卡单条基线;平台层面靠多实例拼吞吐。**
