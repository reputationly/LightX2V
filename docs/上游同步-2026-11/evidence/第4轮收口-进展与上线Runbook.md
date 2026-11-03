# 第 4 轮收口:进展与上线 Runbook(2026-11-03)

## 已完成

| 项 | 结果 |
|---|---|
| 2K 封顶(§1) | 4 份 `configs/swiftvr/a100/*.json` 的 `max_target_height/width` = 1440/2560;代码缺省值同步改成 2K |
| 回退 empty_cache(§2) | `961ccc52` 的逐块 `empty_cache` 与后来加的段末 `empty_cache` 都已删除,SwiftVR 解码路径与上游一致 |
| 2K 冒烟(§3) | 4C 双跑 sh=1440 连跑 3 次:completed 15.2/12.9/14.0s,峰值 22595→23535→23535 MiB(**不随请求上涨**);sh=2160 被压回 **2540×1452**(面积 = 1440×2560,零形变)并打 `exceeds the 2560x1440 ceiling` warning(4 条) |
| 合入 main(§6) | `main` 快进到 `e48faec8` 并已 push(期间 main 无新提交) |
| 旧工作区清理 | worktree `../LightX2V-merge` 与分支 `merge/upstream-2026-08` 已删;其 4 个未提交改动先导出成补丁:`~/Desktop/code/api/LightX2V-merge-uncommitted-20260828.patch`(143 行,内容是 SwiftVR 部署说明与 `seg_warmup_chunks` 注释) |
| 真镜像构建(§4) | `push_latest=false`,产出不可变 tag **`arm64-a100-20261103-0647-e48faec8`**,digest `sha256:d611187bd47293e28fe0a6f20b8400aea78dd8a356a6b43ae7840cd92af595f2`;未推浮动的 `-latest` |
| 真镜像复验(不挂代码) | SeedVR2 4C 124f:**completed 84.0s / 28965 MiB / 1920×1104 / 124 帧**;SwiftVR 4C 1080p:**completed 9.7s / 17185 MiB / 1890×1080 / 124 帧**;镜像内 `av 18.1.0`、`ffmpeg 4.4.2` 含 **libx264 + libx265**;`decord` 垫片来自 `/opt/decord_shim/decord.py`,`_NDArray.to_dlpack()` 可用(返回 PyCapsule) |
| 门禁 | 233 单测通过;ruff check/format 全绿;白名单违规 0;audit rc=0;配置键门禁 rc=0 |

## 上线(§5):遇到一个平台限制,需要裁决

**限制**:GPUStack 2.2.3 里 LightX2V 是 **built-in backend**(`id=3, is_built_in=true, backend_source=built_in`),
它的版本表不允许通过 API 追加固定版本:

```
PUT /v2/inference-backends/3  (带 default_version) → 400 "Built-in backend 'LightX2V' cannot have default_version set"
PUT /v2/inference-backends/3  (不带 default_version,追加 1103.1.0.1) → 400 "Custom backend version '1.0.0' must end with '-custom'"
PUT /v2/inference-backends/3  (版本改名 1103.1.0.1-custom,保留 1.0.0) → 同样 400(它要求**所有**版本都以 -custom 结尾)
```
当前仍是:`versions = {1.0.0 → ...:arm64-a100-latest}`,`default_version = 1.0.0`(未被我改动)。
改动前备份:`/data/verify-logs/backend3-before.json`(在 0004 上)。

**三条可行路线**:

1. **建 custom backend `LightX2V-custom`**(推荐):按 built-in 的同款参数新建
   (`common_parameters = [--profile, --model-cls, --task]`、`health_check_path=/ready`、`custom_framework=cuda`),
   版本 `1103.1.0.1-custom` 指向不可变 tag;然后建 `seedvr2-canary`(1 副本 × 4 卡)走真实网关验证,
   通过后把 `seedvr2` 切到该 backend_version 逐副本滚动,回滚就切回 built-in `1.0.0`。
   代价:集群里多一个 backend 实体。
2. **改 GPUStack 内置定义**:在 gpustack 仓把 LightX2V 内置版本表里的 floating tag 改成不可变 tag 后重新部署
   (改动平台代码,得跟平台组协同)。
3. **推浮动 tag**:`push_latest=true` 再构建一次,把 `arm64-a100-latest` 挪到新摘要 —— 这是验收明确要避免的做法,
   只在 1、2 都不通时作为最后手段。

**另需**:canary 的"真实网关链路"验证需要 new-api 的地址与 API key(本地没有),需要你提供或由你执行那一步。

## 上线进展(2026-11-03 15:05):canary 已部署并验证 —— 用**模型级 image_name** 绕开 built-in backend 限制

发现 `ModelCreate/ModelUpdate` 支持 `image_name` 字段,模型可以自己钉镜像,不必改 backend 版本表。
这比之前提的三条路线都干净(不加 backend 实体、不动浮动 tag):

```
POST /v2/models
{"name":"seedvr2-canary","backend":"LightX2V","backend_version":"1.0.0",
 "image_name":".../lightx2v:arm64-a100-20261103-0647-e48faec8",
 "replicas":1,"backend_parameters":["--model-cls seedvr2"],
 "local_path":"/nfs-models/wuhanjisuan894/models/ByteDance-Seed/SeedVR2-3B",
 "gpu_selector":{"gpu_ids":[4 张空闲卡],"gpus_per_replica":4},"cluster_id":1, ...}
```

| 验证项 | 结果 |
|---|---|
| 创建 | `seedvr2-canary` 1 副本,`ready_replicas=1/1`,`image_name` 已钉成不可变 tag |
| 实跑镜像 | worker 上容器 `seedvr2-canary-n78b3-run-0` 镜像 **`d611187bd472`** = 新镜像 digest(`docker inspect` 核对) |
| 生产形态请求 | 走 launcher 门面(`/v1/tasks/video/`,payload 带 `model/task_type/user_id/size:"1920x1080"/sr_ratio:4.0/target_short_edge/seed/negative_prompt/infer_steps`)→ **completed**,产物 **1904×1104 / 124 帧** |
| 兼容层行为 | 日志 `dropping request fields it does not support: negative_prompt, target_short_edge`(WARNING,非 4xx) |

**仍缺的一步**:new-api → GPUStack 门面这一跳(拿不到 new-api 地址/API key)。canary 目前仍在
`dev-gpustack-a100-0004` 上跑着(4 卡),供那次真网关验证使用。

**滚动上线的执行命令**(第 4 步剩余部分,待 new-api 验证通过后执行):

```bash
# 1) 把生产模型钉到同一个镜像(不影响正在跑的副本,只有被重建的副本才会用新镜像)
PUT /v2/models/38  {...,"image_name":".../lightx2v:arm64-a100-20261103-0647-e48faec8"}
# 2) 逐副本滚动:replicas 3→4(或按 GPUStack 的实例逐个删除,让它重建),每步核对
docker inspect <seedvr2-*-run-0> --format '{{.Image}}'   # 期望 d611187bd472
# 3) 回滚:PUT /v2/models/38 {...,"image_name":null}  ← 回到 built-in 的浮动 tag
```

## canary 复核(2026-11-03 15:20):两种现网形态 payload 都通过

| payload 形态 | 结果 | 兼容层日志 |
|---|---|---|
| 现代形态(`size:"1920x1080"` + `target_short_edge` + `negative_prompt` + `infer_steps` + `model`/`task_type`/`user_id`) | completed,**1904×1104 / 124 帧** | `dropping request fields it does not support: negative_prompt, target_short_edge`;`dropping request fields unknown to this service: model, task_type, user_id` |
| 旧形态(`target_shape:"1920x1080"` 字符串 + `target_video_length` + `num_frames`) | completed,**1904×1104 / 124 帧** | 同上(字符串 size 被忽略、旧键被重命名/丢弃,均 WARNING 非 4xx) |

## 切换机制的两个方向(在 canary 上验证)

- **钉住**(生产要用的方向):`PUT /v2/models/63 {"image_name":"<不可变 tag>", ...}` →
  模型 `image_name` 生效,重建出来的容器镜像 ID = `d611187bd472`(新镜像)✓
- **回滚**(`image_name: null` → 回到 built-in 的浮动 tag):PUT 会把字段改成 null ✓,但**正在跑的容器不会自己重建**
  (GPUStack 只在副本重建时换镜像)。所以回滚的实际操作是:把 `image_name` 置 null → 逐个副本重建
  (缩容/扩容或删掉实例让它重建)→ 用 `docker inspect` 核对应回到 `cdd5271f`(现网旧镜像)。
- 注意:`PUT /v2/models/{id}` 必须带完整 body(含 `source`/`cluster_id`/`gpu_selector`),否则 422。
