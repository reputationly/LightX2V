# LTX-2.3 V2A 配音 — 端到端产品化设计方案

> 覆盖:LightX2V 引擎适配、GPUStack 调度层(launcher/门面)、gpustack-ui、new-api(gpustackplus 渠道)、体验区「视频配乐体验中心」、AudioX 视频配乐下线、接口调用全流程。
> 基于三个本地代码库逐文件核实(LightX2V / ../gpustack + ../gpustack-ui / ../new-api),结论附文件:行号证据。
> 状态:设计稿 v2(引擎侧 v2a 已实现合入 main `e46eb951`,待 smoke;平台三层均未接入)。
> **v2 关键变更(产品拍板)**:AudioX 的视频配乐功能下线;平台 task_type `v2a` 的**任务契约**重定义为「视频→配好音的视频(.mp4)」——task_type 是抽象任务形态,可挂任意多个模型,LTX-2.3 是该契约下的首个实现(非独占);不引入新名、不做按-backend 消歧。

---

## 1. 背景与产品定义

引擎侧已完成 LTX-2.3 `v2a` 纯配音任务(见 `docs/LTX2.3-纯配音V2A-设计文档.md`):输入视频 + 声音描述 prompt(可选)→ 输出**原视频画面逐帧零损失 + AI 生成音轨的 mp4**。声音能力:音效/环境音/BGM/单人与多人对话(引号内逐字台词)。

**产品定义(本次拍板)**:
- 平台「视频配乐」= 上传视频 → 拿回**配好音的视频**;`v2a` 是这个任务形态的 task_type(kind=video,输出 .mp4),**可挂多个模型**,LTX-2.3 为首发实现;
- 现有 AudioX 的「视频生音」(音乐页 tab,输出 .wav 纯音频)**下线**——契约不同(出音频文件),不能与新契约共用 v2a;AudioX 其余无视频输入的能力(t2a 文生音效等)不受影响;
- 用户在模型列表中选择具体模型(能力标签「视频配乐」过滤),未来新配音模型上架仅需部署+渠道绑定+配价,平台层零代码。

## 2. 全链路架构总览

```
用户(体验区/API)
   │ ①上传视频(base64/data-url ≤50MB) + prompt(可选), POST /pg/videos 或 /v1/videos
   ▼
┌─────────────────── new-api(网关层)───────────────────┐
│ gpustackplus 渠道(ChannelType=59)                     │
│ ②鉴权/计费预扣 ③输入物化: 视频写 NFS inputs/…        │
│ ④提交门面: POST {gpustack}/v1/videos                  │
│    body: model / task_type=v2a / prompt / input_refs   │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌─────────────────── GPUStack(调度层)──────────────────┐
│ 视频任务门面 /v1/videos(异步队列+sweeper+准入背压)   │
│ ⑤least-pending 选实例, 路由头静态绑定                  │
│ ⑥透传引擎: POST worker:{port}/v1/tasks/video/          │
│    input_refs → video_path(容器内 NFS 路径)           │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌────────────────── LightX2V(引擎层)───────────────────┐
│ launcher(profile: ltx2-v2a)→ python -m lightx2v.server│
│   --task v2a --config_json ltx2_3_v2a.json             │
│ ⑦冻结视频latent仅去噪音频 ⑧-c:v copy 混流原视频       │
│ ⑨成品 mp4 写 NFS save_result_path                      │
└──────────────────────┬─────────────────────────────────┘
                       ▼
⑩网关轮询 done → nfs_path → 搬 OBS → 签名 URL → 前端 <video> 播放
```

**平台既有契约(全部复用,零新建)**:
- 输入物化(2026-07-07 定案):用户输入由**网关统一物化到共享 NFS**,调度层只收**可信路径引用**(`input_refs`,防目录穿越+租户隔离校验,gpustack `videos.py:380-434`);URL/SSRF 处理在网关(new-api `nfsinput.go:6-9`),引擎只见本地路径;
- 异步任务:提交→轮询→取件,状态机 QUEUED→ASSIGNED→RUNNING→DONE/FAILED(`video_generation_task.py:12-41`),sweeper 兜底(24h 硬死线);
- 成品交付:NFS → OBS 归档 → 签名 URL(new-api `service/media_ingest.go:25-40`);`VideoProxy` 按 content-type 透传(`video_proxy.go:209-223`,默认 `video/mp4`),对 mp4 天然兼容。

## 3. 关键设计决策

### D1: `v2a` 任务契约重定义为「视频 → 配好音的视频」;AudioX 视频配乐下线

**核心概念:task_type 是任务形态(抽象类),不绑定任何单一模型。** 平台机制本来如此:`task_type` 只决定任务契约(路由到哪类引擎 API、输出什么格式),**具体执行哪个模型由请求 `model` 字段决定**("routing is by model",gpustack `videos.py:713` 注释原话);同一 task_type 下可多模型并存(t2v/i2v 现状即是),用户在模型列表里选,计费按模型配价。

**现状**:`v2a` 当前的契约被 vLLM-Omni AudioX 定义为「视频→音频文件(.wav)」——gpustack `videos.py:102` `_AUDIOGEN_TASK_TYPES={"t2a","v2a","v2m","tv2a","tv2m","svs"}`,`:283-284` v2a→kind audiogen,`:297` 输出 .wav;new-api 音乐页「视频生音」tab 用 `<audio>` 渲染。

**决策**:`v2a` 契约重定义为「视频→配好音的视频(.mp4)」:
- **同一个 task_type 只能有一种输入/输出契约**(门面按 task_type 定 engine kind 与输出后缀)——AudioX 退出 v2a 的原因是**契约不同**(出音频文件),不是"名额让给 LTX";
- **LTX-2.3 是新契约下的首个实现,非独占**:未来任何"输入视频、输出配好音视频"的模型直接上架进该 task_type(部署 + 渠道绑定 + 能力标签 + 配价,平台层零代码),用户模型下拉框中选择;
- gpustack 改动:`v2a` 移出 `_AUDIOGEN_TASK_TYPES` → 自动落入 `_engine_kind` 默认分支 kind=`video`(`videos.py:285`)、输出 `.mp4`。**一行集合改动,6 个调用点全部自动生效**;
- `tv2a` 一并下线(v2a 契约里 prompt 本来就是可选字段,不需要"带文本"单列一个 task_type);
- 引擎内外同名(平台 task_type = 引擎 --task = `v2a`)。

**下线边界(明确不动的)**:AudioX 的 t2a(文生音效)保留;`v2m/tv2m`(视频生音乐)与 `svs` 的去留**待产品确认**(见 §7-开放问题——若"视频配乐"语义收敛到 LTX,v2m/tv2m 建议同批下线,LTX 的 BGM 能力覆盖其场景)。

### D2: LightX2V 不做 video URL 下载 / multipart 降级

按平台输入物化契约,URL 下载/SSRF 校验在网关层,引擎只见可信 NFS 路径——现有 v2a 的 `video_path` 透传恰好正确。引擎仅补一个防御:mux 源路径非本地文件时**显式报错**(当前会静默回退 VAE 解码输出,破坏像素契约)。multipart `video_file` 仅 standalone 调试有用,降 P2。

### D3: 历史数据与切换期

- 已完成的 AudioX v2a 任务:产物 .wav 已归档 OBS,任务行存的是绝对路径,回放走 content-type,**不受改判影响**;
- 切换期纪律:改判上线前 **drain 存量 AudioX v2a 队列**(QUEUED/RUNNING 清零再发版),否则 sweeper 重派会把老任务按新路由派给 LTX 实例;
- new-api 侧下架音乐页 tab 后,直连 API 的存量 v2a 调用方(若有)会拿到 LTX 语义的结果(mp4 而非 wav)——发布说明里明示这是 breaking change。

## 4. 分层改动清单

### 4.1 LightX2V 引擎层(本仓)

| # | 改动 | 文件 | 优先级 |
|---|------|------|--------|
| E1 | mux 源路径非本地文件时显式 raise(替代静默回退) | `lightx2v/utils/utils.py` | P0 |
| E2 | launcher 新增 ltx2 profile(见下) | `deploy/gpustack-lx2v-launcher/profiles.yaml` | P0 |
| E3 | server schema 补 `reference_video_frame_cap` 透传 | `lightx2v/server/schema.py` | P1 |
| E4 | multipart `video_file`(standalone 调试) | `lightx2v/server/api/tasks/video.py` | P2 |

**E2 profile 设计**(格式对齐既有条目,数值待 smoke 标定):
```yaml
# LTX-2.3 v2a pure dubbing (task=v2a): freeze source video latents, denoise
# audio only, stream-copy original pixels at mux. Single card bf16 (22B multi-
# card OOMs on 40G A100). Duration/input caps enforced UPSTREAM
# (gateway VideoModelConfig): suggest ≤15s in, ≤720p.
ltx2:
  variants:
    - name: ltx2-v2a/bf16-1card
      gpus: 1
      task: v2a
      config_json: /opt/LightX2V/configs/ltx2/ltx2_3_v2a.json
```
依据:LTX2.3 22B 在 40G A100 多卡 ulysses/TP 均 OOM,只能单卡 bf16(实测存档);v2a 无 upsampler、视频侧冻结不迭代,显存应低于生成任务,offload 开关按 smoke 结果定。

### 4.2 GPUStack 调度层(../gpustack)

| # | 改动 | 位置(已核实) |
|---|------|----------------|
| G1 | `_AUDIOGEN_TASK_TYPES` 移除 `"v2a"`(与 `"tv2a"`)→ v2a 自动落 kind=video/.mp4;确认 `:705` audiox_task 回填随集合移除自动失效 | `routes/videos.py:102,283-285,297,705` |
| G2 | 输入字段:复用现成 `video: video_path` 映射,零改动 | `videos.py:148` |
| G3 | 延迟表加 ltx2 条目(准入背压估算,值待 smoke) | `config/config.py:203` |
| G4 | (可选)LightX2V backend `common_parameters` 登记 `--task/--profile` | `schemas/inference_backend.py:118-120` |
| G5 | 切换期:发版前 drain 存量 AudioX v2a 任务(§3-D3) | 运维动作 |
| G6 | 模型目录同步:AudioX 条目描述与区块注释移除 v2a/tv2a 广告,注明移交 LTX-2.3(广告面必须与路由面一致;不做兼容别名) | `assets/model-catalog.yaml:3817,3826` |

无需改:BackendEnum(LightX2V 已内置)、CategoryEnum(自动归 video)、异步队列/sweeper/上传下载/鉴权、`_engine_kind` 函数本体。

### 4.3 gpustack-ui(../gpustack-ui)

| # | 改动 | 位置(已核实) |
|---|------|----------------|
| U1 | `VideoTaskType`/`KNOWN_TASK_TYPES` 加 `'v2a'`;`inferVideoTaskType` 识别 ltx 模型名 | `src/pages/playground/video/task-inputs.ts:6,17,29-45` |
| U2 | `videoTaskInputs` 加 `v2a: [{field:'video', kind:'video', required:true}]`(照抄 `sr`) | `task-inputs.ts:53-126` |

上传→提交→轮询→`<video>` 播放全复用 `use-text-video.ts`,无新组件/路由。

### 4.4 new-api 后端 gpustackplus 渠道(../new-api)

| # | 改动 | 位置(已核实) |
|---|------|----------------|
| N1 | `validTaskTypes`:`v2a` 已在表内保留;移除 `tv2a` | `relay/channel/task/gpustackplus/adaptor.go:60-67` |
| N2 | `inferTaskType`:加 ltx2 模型名分支→`v2a`;**移除 audiox 对 v2a/tv2a 的推断**(:418 分支收窄到 t2a 等保留项) | `adaptor.go:411-453` |
| N3 | 物化:`v2a` 继续走现成 `materializeAudioXVideoInputs`(`metadata.video`→`FieldVideo`,输入形态相同,建议顺手改名 `materializeVideoDubInputs`) | `adaptor.go:289-329,696-715` |
| N4 | `VideoModelConfig` 加 ltx2 条目(首发建议 ≤720p 入/≤15s/≤50MB) | `common/media_model_config.go:289-316` |
| N5 | 能力标签:`VideoCapabilities` 加「视频配乐」;`MusicCapabilities` **移除「视频生音」**;前端 `VIDEO_CAPABILITIES` 同步(两处一致) | `constant/model_capability.go:19-26,56-66` + `videoPlayground.constants.js:14` |
| N6 | 模型注册计费:渠道模型映射加公开名(建议 `ltx2-v2a`);按次价格 JSON 配置;**解绑 audiox 的 v2a 计费项** | 渠道配置 + `setting/ratio_setting/model_ratio.go:368,397` |

输出侧零改动(content-type 驱动,mp4 天然兼容)。

### 4.5 体验区(new-api 前端 web/classic)

**新增:「视频配乐体验中心」**(语音模型板块):

| # | 改动 | 位置 |
|---|------|------|
| F1 | 语音页 `pages/Audio/index.jsx` 加「视频配乐」子 tab(推荐,成本最低;备选独立页 `pages/VideoDub/`+菜单+路由) | `pages/Audio/` |
| F2 | **模型选择器**:按「视频配乐」能力标签过滤展示(循体验区惯例,支持多模型并存——今天只有 ltx2-v2a,未来新模型上架自动出现);输入:复用 `MediaFileInput.jsx`(kind='video',base64 ≤50MB)+ prompt(可选)+ 负向提示词 | 现成组件 |
| F3 | 提交 hook:仿 `useMusicGeneration.js:988`(`metadata.video` + `task_type:'v2a'`),POST `/pg/videos`;轮询/历史复用 `VideoHistoryPanel` | `hooks/` |
| F4 | **结果渲染用 `VideoChatArea.jsx` 的 `<video>` 播放器**(MusicChatArea 硬编码 `<audio>`,不可用) | 现成组件 |
| F5 | 预置 prompt 快捷按钮:脚步/雨声/引擎/环境氛围/BGM/对白示例(引号台词) | 常量文件 |

**下线:音乐页「视频生音」tab**:

| # | 改动 | 位置 |
|---|------|------|
| F6 | `MUSIC_MODES` 移除 `v2a` 模式(tab 消失);清理 `MUSIC_VIDEO_UPLOAD_MAX_MB` 等仅该 tab 使用的常量与 `MusicModelConfig.videoMaxMB` 引用 | `constants/musicPlayground.constants.js:109-128,301` + `pages/Music/index.jsx:128-135` |

### 4.6 计费与红线(管理端配置)

- 按次计价(任务链路 per-call 预扣+提交后校正,`relay_task.go:190-265`);单价待 M0 smoke 实测耗时定档(参考同为单卡的 SeedVR2/VACE 梯度);
- 红线走 `VideoModelConfig`:≤720p / ≤15s / ≤50MB 首发(引擎 >361 帧有显存警告;上游强制,循 infinitetalk/seedvr2 "red lines enforced UPSTREAM" 惯例);
- 非 24fps 输入引擎已 atempo 精确对齐,无需网关拦;产品文案标注 24fps 素材最佳。

## 5. 接口调用全流程

### 5.1 体验区流程
```
① 前端: 上传视频(FileReader→base64)+ prompt(可选)
② POST /pg/videos   {model:"ltx2-v2a", metadata:{task_type:"v2a", video:"data:video/mp4;base64,...", prompt:"..."}}
③ new-api: 鉴权→计费预扣→物化视频到 NFS→ POST {gpustack}/v1/videos
     {model, task_type:"v2a", prompt, user_id, input_refs:[{field:"video", path:"inputs/v2a-ltx2/2026/07/24/<uid>/<gid>-video.mp4"}]}
④ gpustack: 准入(超阈值 429)→建任务→选实例→ POST worker/v1/tasks/video/
     {video_path:"/nfs-input/…mp4", prompt, negative_prompt, save_result_path:"/nfs-output/…/<task>.mp4", seed…}
⑤ LightX2V: v2a 推理(冻结视频→去噪音频→atempo/apad 混流原画面)→ 写 save_result_path
⑥ 轮询 GET /pg/videos/{id} → gpustack poll-on-GET 拉引擎状态
⑦ done → OBS → 签名 URL → GET /pg/videos/{id}/content → <video> 播放/下载
```

### 5.2 直接 API(OpenAI-video 兼容)
```
POST {new-api}/v1/videos           Authorization: Bearer <token>
  {"model":"ltx2-v2a","metadata":{"task_type":"v2a","video":"<data-url|https-url>","prompt":"footsteps on wood, quiet room ambience"}}
→ {"id":"task_xxx","status":"queued"}
GET  {new-api}/v1/videos/task_xxx           → queued|in_progress|completed|failed
GET  {new-api}/v1/videos/task_xxx/content   → mp4
```

### 5.3 超时/背压(现成,标定即可)
| 层 | 参数 | 现值 | 动作 |
|----|------|------|------|
| gpustack 准入 | `lightx2v_video_max_queue_wait_seconds` | 150s | 沿用(v2a 改判后自动按 video 阈值) |
| gpustack 延迟表 | `lightx2v_model_latency_seconds` | 按模型 | 加 ltx2 条目(smoke 定) |
| gpustack 兜底 | `_TASK_MAX_AGE_HOURS` | 24h | 沿用 |
| 门面上传上限 | `_MAX_UPLOAD_BYTES` | 256MiB | 沿用(网关 50MB 先拦) |

## 6. 上线顺序

1. **M0 引擎 smoke**(阻塞一切):`run_ltx2_3_v2a.sh`,素材 120帧@30fps;产出显存/耗时(→延迟表、profile、定价)+ 音质同步人工评估;
2. **M1 引擎收尾**:E1 + E2,出镜像(浮动 tag,GPUStack 零改动生效);
3. **M2 调度层**:G1(v2a 改判)+ G3 + G5(drain);gpustack-ui U1/U2;内网 video playground 验通;
4. **M3 网关后端**:N1-N4 + N6;curl 走 §5.2 全流程验通;
5. **M4 体验区**:F1-F5 新中心 + F6 音乐页下线 + N5 标签;
6. **M5 上线**:红线/单价定档、发布说明(v2a 语义变更 breaking change 明示)。

## 7. 风险与开放问题

| # | 风险/待决 | 处置 |
|---|-----------|------|
| 1 | **音频质量/同步是唯一未验证的经验性风险** | M0 首要验证;不佳则评估官方 foley/audio LoRA |
| 2 | AudioX `v2m/tv2m`(视频生音乐)、`svs` 是否随「视频生音」同批下线 | **待产品确认**;建议同批(LTX 的 BGM 能力覆盖,语义收敛) |
| 3 | 22B 单卡显存 vs 输入时长上限 | M0 实测 5s/10s/15s 三档定红线 |
| 4 | 存量 v2a API 调用方语义变更(wav→mp4) | 发布说明明示 breaking;切换前 drain 队列(G5) |
| 5 | 逐字台词对口型依赖原视频嘴型匹配(dubbing 固有约束) | 体验区预置 prompt 以环境音/音效/BGM 为主,台词场景标注最佳实践 |
