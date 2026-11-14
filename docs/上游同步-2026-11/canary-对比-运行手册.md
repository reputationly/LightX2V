# 新镜像 canary 对比运行手册(2026-11-14 起)

目的:把 `merge/upstream-2026-11b` 打成不可变 tag 的镜像,用 GPUStack canary 模型在**真实网关链路**上
和现网 `relayvsr` 对比,判据是「耗时不多于现网、显存不高于现网、成片与现网一致」。

已完成的前置工作(2026-11-14):合并 + 冲突处理 + 代码级 A/B(同镜像挂载合并代码:稳态 9.00 s vs 8.98 s、
峰值显存一致、成片 md5 相同),见 `docs/上游同步-2026-11/合并评审-2026-11-14.md` §9。
本手册只覆盖「构建 → canary → 网关对比 → 清理」。

## 0. 环境变量

```bash
GPUSTACK=http://58.48.42.236
KEY=<GPUStack API key>          # 不要写进仓库
IMG_REPO=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v
BRANCH=merge/upstream-2026-11b
```

## 1. 构建不可变镜像

```bash
git push origin "$BRANCH"
gh workflow run build-arm64-docker.yml --ref "$BRANCH" -f push_latest=false
gh run list --workflow=build-arm64-docker.yml --limit 3          # 等 status=completed / conclusion=success
```

- `push_latest=false`(默认)只推 `arm64-a100-<yyyymmdd-hhmm>-<sha8>`,**不动现网浮动 tag**
  `arm64-a100-latest`;现网 5 个 `relayvsr` 副本完全不受影响。
- 取 tag:`gh run view <run-id> --log | grep 'arm64-a100-'` 或看 workflow 输出的 `IMAGE_TAG`。
- app 层只做「拷代码 + `pip install -e --no-deps`」,合并未引入新依赖(`requirements.txt` 未变,
  `pyproject` 只多一个可选 extra `tbdub`),无需重建 base。

## 2. 建 canary 模型(4 卡,指定空节点)

```bash
TAG=arm64-a100-20261114-XXXX-<sha8>
curl -sS -X POST "$GPUSTACK/v2/models" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d @- <<JSON
{
  "source": "local_path",
  "local_path": "/nfs-models/wuhanjisuan894/models/RelayVSR_X2",
  "name": "relayvsr-canary",
  "replicas": 1,
  "categories": ["video"],
  "backend": "LightX2V",
  "backend_version": "1.0.0",
  "image_name": "$IMG_REPO:$TAG",
  "backend_parameters": ["--model-cls relayvsr_x2"],
  "gpu_selector": {"gpus_per_replica": 4, "gpu_ids": ["dev-gpustack-a100-0007:cuda:0","dev-gpustack-a100-0007:cuda:1","dev-gpustack-a100-0007:cuda:2","dev-gpustack-a100-0007:cuda:3"]},
  "distributed_inference_across_workers": false,
  "placement_strategy": "spread",
  "owner_principal_id": 1,
  "access_policy": "authed"
}
JSON
```

- 用 `image_name` 钉住版本(比新增 backend version 干净),回滚 = 删掉 canary 或把模型切回原镜像。
- `--model-cls relayvsr_x2` + 4 卡 → launcher 自动选 `relayvsr/chunkp4`
  (`configs/relayvsr/relayvsr_x2_chunkp4.json`),与现网一致。
- 等就绪:`GET /v2/models/<id>` 的 `ready_replicas=1`;4 卡冷启动约 6–7 分钟(加载 48 s + 预热 ~320 s),
  GPUStack 的健康检查探测 `/ready`,预热完成前一直 503,不会提前导流。

## 3. 网关链路对比(同一输入、同一参数)

```bash
# 3.1 上传源片(gpustack-ui 路径:一次一字段,multipart)
curl -sS -X POST "$GPUSTACK/v1/videos/inputs" -H "Authorization: Bearer $KEY" \
  -F field=video -F file=@/path/to/sr_768p_15s.mp4
# → {"input_refs": {"video": ["inputs/.../xxx-video.mp4"]}, ...}

# 3.2 提交(canary 与现网各 3 次,串行,记录 wall time)
curl -sS -X POST "$GPUSTACK/v1/videos" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
  "model": "relayvsr-canary",
  "task_type": "sr",
  "sr_ratio": 2,
  "seed": 42,
  "input_refs": [{"field": "video", "path": "inputs/.../xxx-video.mp4"}]
}'

# 3.3 轮询 / 取片
curl -sS "$GPUSTACK/v1/videos/<task_id>" -H "Authorization: Bearer $KEY"
curl -sS "$GPUSTACK/v1/videos/<task_id>/content" -H "Authorization: Bearer $KEY" -o out.mp4
```

判据:

| 项 | 判定 |
|---|---|
| 端到端耗时 | canary 稳态 ≤ 现网稳态(现网基线见评审文档 §9,机器与输入需一致) |
| 每卡峰值显存 | canary ≤ 现网 12275 / 12203 MiB |
| 成片 | 与现网同请求产物 `md5` 相同(`sr_ratio=2` 路径;浮点 ratio 只比对 PSNR) |
| 预热与首条 | 实例日志里 `warmup cost`、`RUN pipeline cost` 首条与稳态;首条 ~2× 两边都应出现 |

## 4. 取证据(不用 ssh)

```bash
# 实例日志:预热耗时、pipeline cost、报错
curl -sS "$GPUSTACK/v2/model-instances/<instance_id>/logs?tail=2000" -H "Authorization: Bearer $KEY"
# 实例与 GPU 绑定
curl -sS "$GPUSTACK/v2/model-instances?model_id=<canary_model_id>" -H "Authorization: Bearer $KEY"
```

节点级核对(可选,SSH 已通):`docker inspect <cid>` 看 `RepoDigests` 是否等于第 1 步的 tag,
`nvidia-smi` 看峰值。

## 5. 清理与回滚

```bash
curl -sS -X DELETE "$GPUSTACK/v2/models/<canary_model_id>" -H "Authorization: Bearer $KEY"
# 现网 relayvsr(id 64)全程只读,不动;要切版本时改 model 的 image_name 后逐副本滚动
```

## 6. 现状备注

- 现网 `relayvsr`:5 副本 × 4×A100-40G(0016–0020),`LightX2V 1.0.0` → 浮动 tag
  `arm64-a100-latest`,实际 digest `sha256:d1872e01e53d…`(2026-11-04T03:01Z 创建)。
- 浮动 tag 的风险仍在:任何容器重启都可能拉到新镜像。切版本时优先用不可变 tag 钉 `image_name`,
  核对 `docker inspect` 的 digest 后再放量。
