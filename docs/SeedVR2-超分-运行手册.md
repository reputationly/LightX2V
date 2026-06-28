# SeedVR2 视频超分 运行手册(LightX2V / ARM A100)

> 记录时间:2026-06-28
> 目标:720p AIGC 视频 → 1080p,补细节。
> 结论先行:**现有 ARM 镜像即可跑(不需要 decord),3B 最轻、7B 也能在 40GB A100 装下。**

---

## 0. 三个最关键的点(先看)

1. **不需要 decord,不用重出镜像。**
   runner 读视频用 `torchvision.io.read_video`,三级 fallback,最后一级用 **PyAV(`av`,镜像已装)**。见 `lightx2v/models/runners/seedvr/seedvr_runner.py` 的 `_get_read_video()`。

2. ⚠️ **官方警告:720p AIGC 输入容易"过度补细节 / 过锐化"。**
   原话:"由于强大的生成能力,该方法面对轻度退化的输入(例如 720p 的 AIGC 视频)时倾向于过度生成细节,偶尔导致过锐化。"——正好是本场景。细节能补,但可能用力过猛,需调参/验证。

3. ⚠️ **7B 权重缺文件。** 官方 7B 仓库缺 `pos_emb.pt` / `neg_emb.pt`。
   做法:`model_path` 指向 **3B 仓库**(取 vae+emb),config 里 `dit_original_ckpt` 指向 7B 的 DiT。**跑 7B 要同时下 3B + 7B。**

---

## 1. LightX2V 里的资产位置(已确认存在)

- 网络:`lightx2v/models/networks/seedvr/`
- runner:`lightx2v/models/runners/seedvr/seedvr_runner.py`(`@RUNNER_REGISTER("seedvr2")`)
- scheduler:`lightx2v/models/schedulers/seedvr/`
- VAE:`lightx2v/models/video_encoders/hf/seedvr/`
- 配置:`configs/seedvr/seedvr2_3b.json`、`seedvr2_7b.json`、`configs/seedvr/4090/{3b,7b}.json`
- 脚本:`scripts/seedvr2/run_seedvr2_3b_sr.sh`、`run_seedvr2_7b_sr.sh`、`run_seedvr2_3b_image_sr.sh`

## 2. 权重清单与下载

一个 `model_path` 目录需包含:

| 文件 | 用途 |
|---|---|
| `seedvr2_ema_3b.pth` / `seedvr2_ema_7b.pth` | DiT 主干 |
| `ema_vae.pth` | 视频 VAE(编码+解码同一个) |
| `pos_emb.pt` | 预计算正向文本 embedding |
| `neg_emb.pt` | 预计算负向文本 embedding |

runner 默认路径(`seedvr_runner.py`):
- DiT = `dit_quantized_ckpt` > `dit_original_ckpt` > `{model_path}/seedvr2_ema_{model_size}.pth`
- VAE = `{model_path}/ema_vae.pth`
- emb = `{model_path}/{pos,neg}_emb.pt`

**下载(HF,无官方 ModelScope;ARM 拉 HF 建议走 hf-mirror):**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ByteDance-Seed/SeedVR2-3B",
                  local_dir="/nfs-data/models/ByteDance-Seed/SeedVR2-3B",
                  local_dir_use_symlinks=False, resume_download=True)
# 7B 另外再下一份 ByteDance-Seed/SeedVR2-7B(只需其中的 seedvr2_ema_7b.pth)
```

> 权重放 NFS(`/nfs-data`),起容器记得 `-v /nfs-data`(见项目 NFS 备忘)。

## 3. 运行命令

### 3B(推荐先跑,最轻)

```bash
python -m lightx2v.infer \
  --model_cls seedvr2 --task sr --sr_ratio 1.5 \
  --video_path /path/to/720p.mp4 \
  --model_path /nfs-data/models/ByteDance-Seed/SeedVR2-3B \
  --config_json configs/seedvr/seedvr2_3b.json \
  --save_result_path save_results/out_seedvr2_3b.mp4
```

### 7B(注意 model_path 用 3B 仓库 + config 指 7B DiT)

`configs/seedvr/seedvr2_7b.json` 里设:
```json
"dit_original_ckpt": "/nfs-data/models/ByteDance-Seed/SeedVR2-7B/seedvr2_ema_7b.pth"
```
命令:
```bash
python -m lightx2v.infer \
  --model_cls seedvr2 --task sr --sr_ratio 1.5 \
  --video_path /path/to/720p.mp4 \
  --model_path /nfs-data/models/ByteDance-Seed/SeedVR2-3B \
  --config_json configs/seedvr/seedvr2_7b.json \
  --save_result_path save_results/out_seedvr2_7b.mp4
```

### 单图超分

```bash
python -m lightx2v.infer --model_cls seedvr2 --task sr --sr_ratio 2.0 \
  --image_path assets/inputs/imgs/frame_1.png \
  --model_path /nfs-data/models/ByteDance-Seed/SeedVR2-3B \
  --config_json configs/seedvr/seedvr2_3b.json \
  --save_result_path save_results/out.png
```

运行前 `source scripts/base/base.sh` 需要先 export `lightx2v_path` 和 `model_path`(脚本里有)。

## 4. 关键参数(config_json)

| 参数 | 默认 | 说明 |
|---|---|---|
| `target_height` / `target_width` | 1080 / 1920 | 输出上限;`sr_ratio` 放大后超过此值会被 cap |
| `infer_steps` | 1 | **单步扩散**,所以快 |
| `--sr_ratio` (CLI) | 2.0 | 放大倍率;**720→1080 用 1.5 最贴合**,2.0 会被截到 1080p |
| `resize_mode` | adaptive | adaptive=不强制裁到 target;非 adaptive 会中心裁/缩放到 target |
| `color_fix` | gpu | 小波重建做色彩校正,防止偏色(cpu/gpu/off) |
| `cpu_offload` | true | 省显存,40GB 必开 |
| `offload_granularity` | model | 整模型级 offload |
| `use_tiling_vae` | true | VAE 分块解码,省显存(`vae_tile_size` 512 / `vae_tile_overlap` 64) |
| `sr_segment_length` | 81 | 长视频按帧分段处理(防 OOM) |
| `sr_overlap` | 1 | 段间重叠帧,拼接时去重 |

**分辨率计算逻辑**(`_build_video_transform`):
`resolution = min((ori_h*ori_w)^0.5 * sr_ratio, (target_h*target_w)^0.5)`,再 `DivisibleCrop(16,16)`。
即:按 sr_ratio 放大,但面积被 target 限死。所以 1080p 输出基本由 config 的 target 保证。

## 5. 显存与提速(40GB A100)

- **3B**:cpu_offload + tiling,40GB 很宽裕。
- **7B**:同样开 offload+tiling 可装下;吃紧就缩 `sr_segment_length` / `vae_tile_size`。
- **fp8 量化**(可选,后续试):`configs/seedvr/4090/` 用 `dit_quant_scheme: fp8-q8f` + `dit_quantized_ckpt` 指向 fp8 safetensors。本镜像有 `q8_kernels`,理论可用;需先拿到/转出 fp8 权重。
- 官方多卡:序列并行 sp,1×H100-80G 跑 720p,4 卡跑 1080p/2K。LightX2V 的 offload+tiling 就是为单卡小显存设计的(有 4090/24GB 配置佐证)。

## 6. 待办

- [ ] 下 3B 权重到 NFS,先跑通一条 720p→1080p
- [ ] 肉眼看效果:重点确认是否"过锐化/过度补细节"(官方已警告 AIGC 720p 易过冲)
- [ ] 若过冲:调小 `sr_ratio`、试 `color_fix` 不同档、或考虑 3B(生成更弱、可能更稳)
- [ ] 对比 7B vs 3B 质量/耗时
- [ ] 可选:试 fp8-q8f 量化提速
- [ ] 决定:超分后挂 vs 生成阶段直接出 1080p,哪条更划算

## 参考

- [SeedVR2-3B (HF)](https://huggingface.co/ByteDance-Seed/SeedVR2-3B)
- [SeedVR2-7B (HF)](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)
- [SeedVR 官方仓库](https://github.com/ByteDance-Seed/SeedVR)
