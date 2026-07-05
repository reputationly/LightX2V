# Qwen-Image 单机最优部署方案(4×A100 40G · 鲲鹏920 ARM)

> 节点:`dev-gpustack-a100-0001`(4×A100 PCIE 40G,鲲鹏920 ARM 128核 4-NUMA,251G RAM,权重在 SFS)
> 引擎:LightX2V server(Docker,镜像 `…/reputationly/lightx2v:arm64-a100-latest`)
> 数据来源:`Qwen-Image-实验测试报告.md`(t2i)+ `Qwen-Image-Edit实验测试报告.md`(edit)全部实测。日期 2026-07-05。
> 一句话:**t2i 和 edit 都用 2 个单卡实例/节点(GPU0+GPU2,不绑核,错峰启动)。必开 `qwen25vl_cpu_offload:false`(单图快 1.66×、吞吐 +48%)。2 实例是"任何分辨率都不亏"的稳妥解;多卡/int8/lazy_load 想上更多卡全试全负。**

---

## 1. 推荐方案(t2i + edit 通用,直接照做)

| 项 | 值 | 为什么 |
|---|---|---|
| **实例数** | **2 副本/节点(t2i 和 edit 各自)** | **稳**:任何分辨率不回退、内存安全(2×60G=120G<251G)。3 副本仅 16:9 类中低分辨率划算、大图(1:1)反而回退+内存逼近上限 → 混合流量选 2 |
| 引擎档(t2i) | **merged8**(离线合并 8 步蒸馏) | **17.0s/张**(qwen25vl 后),比 base 108.9s 快 6.4× |
| 引擎档(edit) | **merged 8步 i2i** | **21.6s/张**,比 base 25步 116.7s 快 5.4×;质量敏感场景用 base |
| **★必开开关** | `qwen25vl_cpu_offload: false` | 文本编码器留 GPU(才15G):t2i 单图 28→**17s**、2副本吞吐 0.077→**0.114**;edit 不加会文本编码 7min(误判成 hang) |
| GPU | **GPU0 + GPU2**(各占一个 NUMA) | 分散显存/PCIe |
| **不绑核** | 去掉 `--cpuset-cpus` / `OMP_NUM_THREADS` | 实测绑核吞吐掉 22%(offload 吃核数,限核=限流);让两实例共享全部 128 核 |
| 每实例内存 | `--memory≥100g`(用 110g) | 每实例 **~60G shmem**(CUDA pinned DiT,不可共享);<78g cgroup 会 thrash |
| 容器 | `--init` + `--restart no` | 防僵尸;`restart always` 开机同时拉起→死机 |
| 启动 | **严格错峰**:起一个→等 ready→再起下一个 | 同时启动 = ARM CPU 被 offload init 挤爆死机 |
| 端口 | 容器内都 8000,宿主映射 8000/8001 | 实例隔离 |
| 剩余 GPU1/GPU3 | 纯 Qwen 场景闲置(或跑 z-image,z-image 不 offload、可 4 卡满载 0.53 img/s) | offload 内存天花板:3+副本 shmem 逼近 251G |

**产能(2 副本,qwen25vl 后)**:t2i **~0.114 img/s**(1:1)/ **~0.11**(16:9,3副本可到0.164);edit **~0.091 img/s**。要更高吞吐:加机器,或用低分辨率上 3 副本。

---

## 2. 全配置实测对比表 ⭐(全部真机跑过,均 `qwen25vl_cpu_offload:false`)

> 生成为**热态**(丢首张)。完整逐项见两份测试报告(t2i §2/§4、edit §10)。**主机内存用真实 Shmem 口径**(`free` used 列漏 shmem、会低估)。

| # | 配置 | 卡/并行 | 步数 | 生成(热态) | 显存峰 | 每实例shmem | 吞吐 | 结论 |
|---|---|---|---|---|---|---|---|---|
| **1** | **t2i merged8 单卡** ⭐ | 1 | 8 | **17.0s** | 26.7G | ~60G | 0.057 | ✅ **t2i 延迟最优** |
| **2** | **t2i merged8 2 副本** ⭐ | 2×1 | 8 | 17.5s | 26.7G/卡 | ~60G | **0.114(2.0×)** | ✅ **推荐(稳)** |
| 3 | t2i merged8 3 副本 @1:1 | 3×1 | 8 | 29s(拖慢) | — | ~60G | 0.103(回退) | ⚠️ 大图 3 副本负优化 |
| 3b | t2i merged8 3 副本 @16:9 | 3×1 | 8 | 18s | — | ~60G | **0.164(近线性)** | ✅ 中低分辨率可上 3 |
| 4 | t2i 4 副本 | 4×1 | 8 | — | — | — | — | ❌ 60G×4>251G 全局 OOM |
| 5 | t2i base 全质量 单卡 | 1 | 25+CFG | 108.9s | 17.8G | ~60G | 0.009 | ✅ 全质量档 |
| **6** | **edit merged 单卡** ⭐ | 1 | 8 | **21.6s** | ~20G | ~60G | 0.046 | ✅ **edit 延迟最优** |
| **7** | **edit merged 2 副本** ⭐ | 2×1 | 8 | ~22s | ~20G/卡 | ~60G | **0.091(1.98×)** | ✅ **edit 推荐(稳)** |
| 8 | edit merged 3 副本 @16:9 | 3×1 | 8 | ~24s | ~20G/卡 | ~60G | 0.126 | ✅ 中低分辨率可上 3 |
| 9 | edit base 全质量 单卡 | 1 | 25+CFG | 116.7s | ~20G | ~60G | 0.009 | ✅ 保真略优 |
| 10 | int8 +offload | 1 / 4 | 8 | 49s | 19-33G | 30G | 4副本 0.079 | ❌ 慢2.3×,4卡也<bf16 3卡 |
| 11 | lazy_load 磁盘offload | 1 | 8 | 126-132s | ~20G | **0.5MB** | ~0.03 | ❌ shmem归零但慢6×(per-block开销)|
| 12 | 多卡 TP / ulysses | 2-4 | — | — | — | — | — | ❌ 全废(TP慢/ulysses奇数崩/4卡死机) |

### 表的四条硬结论
1. **只用单卡实例**:多卡(TP/ulysses,#12)非慢即崩即死机。
2. **必开 `qwen25vl_cpu_offload:false`**:文本编码器留 GPU,t2i 单图 28→17s、2副本吞吐 +48%;edit 不加会文本编码 7min。
3. **副本上限 = 3(内存)**:每实例 60G shmem(pinned,不可共享),4副本 240G+系统>251G OOM。**峰值副本随分辨率**:1:1 峰值2副本、16:9 峰值3副本。**混合流量选 2 副本最稳。**
4. **破上限的路全试全负**:int8(慢2.3×)、lazy_load(慢6×)、多卡(全废)——要真 4 卡满速只能加内存条或改引擎共享 pinned 权重。

---

## 3. 启动命令(绑核双实例,严格错峰)

```bash
IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
CFG=/data/lightx2v_configs/qwen_2512_a100_lightning_merged.json
MP=/data/models/Qwen-Image-2512

run_inst(){  # $1=name $2=gpu $3=hostport  —— 不绑核、不限 OMP(实测绑核掉 22%)
  docker run -d --name "$1" --init --restart no \
    --gpus "\"device=$2\"" --memory=110g \
    -p "$3":8000 -v /data:/data -v /nfs-data:/nfs-data \
    -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMG" python -m lightx2v.server --model_cls qwen_image --task t2i \
    --model_path "$MP" --config_json "$CFG" --host 0.0.0.0 --port 8000
  until curl -sf "http://localhost:$3/v1/tasks/queue/status" >/dev/null 2>&1; do sleep 5; done
  # 预热一张焊 autotune
  curl -s -X POST "http://localhost:$3/v1/tasks/image/" -H 'Content-Type: application/json' \
    -d '{"prompt":"warmup","save_result_path":"/data/_smoke/warm_'"$3"'.png","aspect_ratio":"1:1"}' >/dev/null
  echo "$1 ready ✅"
}

run_inst qwen-t2i-0 0 8000   # 实例0:GPU0
run_inst qwen-t2i-1 2 8001   # ★上一个 ready 后才起:GPU2
```

- 上面的 `qwen_2512_a100_lightning_merged.json` **已含 `qwen25vl_cpu_offload:false`**(必须);启动后每实例发 1 张 warmup 焊 autotune。
- **图生图(edit)另起 2 个实例**(同样 2 副本,与 t2i 分开占卡或分节点):
  ```
  --task i2i --model_path /data/models/Qwen-Image-Edit-2511 \
  --config_json /data/lightx2v_configs/qwen_edit_2511_a100_lightning_merged.json
  ```
  提交带 `image_path` + **显式 `aspect_ratio`**(不传默认 16:9,会把方图/竖图塌成横图)。edit 也是 offload 重负载,同样计入"≤3 实例、混合选2"配额。

---

## 4. 接入层

- 负载均衡:轮询 8000/8001;LightX2V 任务态在实例内存,轮询打错实例返回 404,**new-api GPUStackPlus 适配器已容忍 404 继续轮**,多副本可用。
- 健康检查:`GET /v1/tasks/queue/status`(200);加载期 2-6min,初始延迟给足。
- 限流:`max_queue_size`(默认10)是唯一阀门,排队只存文本不占显存;按 SLA 调小让上游快速失败重试。
- 判黑图:产物 <50KB 视为失败(sage_attn2 黑图假绿的教训;本方案 torch_sdpa 已规避)。

---

## 5. 运维红线(踩过 5+ 次死机换来的)

1. **绝不 4 路齐发**(4 实例 或 4 卡)= ARM CPU 挤爆 → 整机死机、SSH 断,只能云控台强制重启。
2. **实例必须错峰启动**,`--restart no` + 开机脚本按序拉起(§3 的 `run_inst` 顺序调用);**禁用 `--restart always`**(开机同时拉起=复现死机)。
3. **每实例 `--memory≥100g`,不要 `--cpuset-mems`**(锁 NUMA 内存会顶爆)。
4. **换容器 `kill+sleep+rm -f`,新容器带 `--init`**;lightx2v 进程常 D 态卡 NFS,`docker rm -f` 单发常失败。
5. 监控 `load` 与 GPU util:**load>50 且 util 长期 0 = offload thrash**,查是不是有实例在重复重启/内存不足。

---

## 6. 已完成 / 待补
- [x] ~~绑核复测~~:绑核吞吐掉 22%,弃用(不绑核)。
- [x] ~~`qwen25vl_cpu_offload:false`~~:t2i 单图 28→17s、2副本 0.077→0.114;edit 文本编码 7min→11s。**已入生产配置。**
- [x] ~~Edit-2511 Lightning 离线合并~~:merged 8步 21.6s 跑通(见 edit 报告)。
- [x] ~~分辨率/转置~~:Qwen **无** z-image 的转置 bug;autotune 缓存跨分辨率保留(启动预热各比例即可)。
- [x] ~~int8 / lazy_load 破副本上限~~:全试全负(int8慢2.3×、lazy_load慢6×)。
- [ ] 接回 GPUStack(Custom 后端 + 薄透传 → new-api + OBS)。
- [ ] 开机自启脚本:按序错峰拉起(禁 `--restart always`)。
