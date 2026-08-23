#!/usr/bin/env bash
# =============================================================================
# 下载 MiniMax-H3(2026-08-03 开源的全模态视频+原生立体声音频生成系统)到 NFS。
# 沿用 download_bernini_s2v.sh 的套路:ModelScope 主源(国内快)→ hf-mirror 回退,
# 实时测速 + 断点续传 + 失败追踪 + 整目录软链去重。只写新目录, 不碰已有模型。
#
# ⚠️ 原始仓很大(全量 464G), 但逐文件 sha256 比对后可整目录软链去重, 实际落盘 330G。
#    A100-40G 上真正跑得动的其实是 NF4 量化版(48G, 单卡), 见下面的"跑得动吗"。
#
# 原始仓结构(两套格式并列, 别一股脑全下):
#   FL2VA/        134.2G  原始 checkpoint, 任务 t2va + fl2va(文生 / 首帧 / 尾帧 / 首尾帧)
#   Ref2VA/       134.2G  原始 checkpoint, 任务 ref2va(≤9图 / ≤3视频 / ≤3音频 混合参考)
#                         ↑ 这两个给 SGLang / vLLM 用(--model-variant fl2va|ref2va)
#   根目录/       195.9G  diffusers 模块化格式(modular_model_index.json + transformer/
#                         transformer_ref/ vae/ audio_vae/ text_encoder/ ...),给 diffusers 用。
#                         注意:它的 transformer 是重新分片的独立副本(命名 diffusion_pytorch_
#                         model-*-of-00014, 与 FL2VA 的 model-*-of-00013 sha 全不同), 不能软链复用。
#   assets/       0.1G    官方 demo 视频/图, 默认不下
#   docs/ scripts/ README/LICENSE   提示词写作指南 + 复现用 curl 脚本, 很小, 默认下
#
# 逐文件 sha256 核实过的去重关系(本脚本自动做整目录软链, DEDUP=0 可关):
#   Ref2VA/{text_encoder,video_vae,audio_vae,processor,tokenizer} == FL2VA 的同名目录  → 省 72.4G
#   根/{text_encoder,tokenizer,processor}                          == FL2VA 的同名目录  → 省 62.1G
#   只有 FL2VA/transformer(61.7G) 与 Ref2VA/transformer(61.7G) 是真·两份不同权重。
#
# 标签(默认 "nf4 fl2va docs"):
#   nf4         48.0G  DiffSynth-Studio/MiniMax-H3-NF4(bitsandbytes 4bit) —— **A100-40G 首选**
#                      fl2va DiT 16.0 + ref2va DiT 16.0 + text_encoder 14.3 + video_vae 1.5 + audio_vae 0.3
#                      单路只需 DiT+TE+两个 VAE ≈ 32G, 单卡装得下;另拉原仓 processor/tokenizer(几 MB)
#   nf4_fl2va   32.0G  只要 fl2va 那一路的 NF4(省掉 ref2va DiT)
#   w4a16       35.2G  Ar4ikov/MiniMax-H3-transformer-W4A16-RTN(auto-round int4) —— **Plan B, 默认不下**
#                      标准 diffusers 分片格式(有 config/index/quantization_config), W4A16 在 sm_80 有
#                      marlin 系 kernel, auto-round 带校准, 画质预期优于 RTN/NF4。但两个硬伤:
#                      ① 只量化了 DiT, text encoder 仍需原仓 bf16 的 62G, 单卡照样装不下;
#                      ② 上传 3 小时、downloads=0, 零验证。只在 NF4 画质不达标时才试。
#   fl2va      134.2G  原始 bf16 FL2VA 全量(text_encoder 62.1 + transformer 61.7 + video_vae 9.7 + audio_vae 0.6)
#   ref2va      61.8G  只下 Ref2VA/transformer + model_index.json, 其余整目录软链自 FL2VA
#                      (DEDUP=0 时退化为 134.2G 全量下载)
#   diffusers  133.8G  根目录 diffusers 格式(text_encoder 软链自 FL2VA; DEDUP=0 时 195.9G)
#   docs        <1M    README / LICENSE / docs / scripts / *.json
#   assets      0.1G   官方 demo 视频
#
# ⚠️ A100 40G 跑得动吗:先看清楚再下。
#   - H3-Omni-Transformer 是 33B dense(bf16 ≈ 61.7G), text encoder 是 Qwen3-VL-32B(≈62G);
#   - 官方给的 `--num-gpus 4 --ulysses-degree 4` 是序列并行, **每卡都要放全量权重**, 61.7G > 40G,
#     4×A100-40G 直接放不下 → bf16 只能走 TP / offload, 参考 LTX2.3 22B 那次的教训;
#   - 另有约 13B 参数在 AdaLN 分支, 官方说推理时可预计算缓存、不必加载, 能省一截;
#   - 本机 256G 内存(swap 仅 3G), 多进程各自 CPU 侧持 60G+ 权重容易 OOM, 起服务要错峰;
#   - 所以先验 NF4:单卡 ~32G 装得下, DiffSynth-Studio 有现成 pipeline, bnb 在鲲鹏 ARM 上
#     已由 HunyuanImage3.0 那次 POC 验过可用。但记住 NF4 是 weight-only, **只省显存不提速**
#     (A100 不吃 INT4 算力 + 反量化开销, 同 Z-Image int8 热态慢 2.9× 的教训);
#     且 ref2va 属参考/编辑类任务, 低比特最容易崩(Bernini v2v 雪花那次的结论), 画质要重点比对。
#   社区量化版逐个核过(2026-08-03, HF+魔搭共 10 个), 结论是**只有 NF4 和 W4A16 两个能进来**:
#     NVFP4 系(lilcheaty / rockerBOO / Abiray 的 nvfp4 部分) —— NVFP4 要 Blackwell sm_100+,
#                                                              A100 是 sm_80, 无硬件支持, 全灭;
#     ComfyUI 系(gordonz int4-convrot / Abiray Convrot / Abiray GGUF / tsolful INT4Mixed) ——
#                扁平单文件打包、无 config/index/tokenizer, SGLang/vLLM/diffusers/DiffSynth 都读不了
#                (同 Bernini-S2V 跳过 int8-convrot 的理由);
#     MLX 系(ddalcu 的 8bit/4bit) —— Apple Silicon 专用, 与服务器无关;
#     benjiaiplayground/MiniMax-H3_quant —— 空壳, 58 个文件全是 assets/docs, 压根没有权重。
#
# 用法(manager 或任意挂了 NFS 的节点, 先 scp 到 /root):
#   tmux new -s dl_h3 -d 'bash /root/download_minimax_h3.sh'          # 默认 nf4+fl2va+docs ≈ 182G
#   tail -f "$(dirname "${DEST:-/nfs-data/models}")/dl_minimax_h3.log" # 看速度+进度
#   MODELS="nf4_fl2va docs" bash /root/download_minimax_h3.sh          # 最快出片路径, 只 32G
#   MODELS="fl2va ref2va" bash /root/download_minimax_h3.sh            # bf16 双任务, 去重后 196G
#   MODELS="diffusers" bash /root/download_minimax_h3.sh               # 追加 diffusers 格式
#   MODELS="fl2va ref2va" DEDUP=0 bash /root/download_minimax_h3.sh    # 关去重, 各目录独立实体(268.4G)
#   VERIFY=0 bash /root/download_minimax_h3.sh                         # 跳过末尾逐文件大小审计
# 中断后重跑本脚本会自动续传。
# =============================================================================
set -u
# DEST 自动适配:计算节点有 /nfs-data 软链;manager 节点用真身 /nfs-models/wuhanjisuan894/models
DEST="${DEST:-$([ -d /nfs-data/models ] && echo /nfs-data/models || echo /nfs-models/wuhanjisuan894/models)}"
LOG="${LOG:-$(dirname "$DEST")/dl_minimax_h3.log}"
MODELS="${MODELS:-nf4 fl2va docs}"
DEDUP="${DEDUP:-1}"          # 1=同 sha 目录整目录软链复用 FL2VA, 0=每份都真下
VERIFY="${VERIFY:-1}"        # 1=末尾拉官方清单逐文件比对大小
MS_REPO="${MS_REPO:-MiniMax/MiniMax-H3}"
HF_REPO="${HF_REPO:-MiniMaxAI/MiniMax-H3}"   # 注意 HF 上是 MiniMaxAI 组织, 与 MS 的 MiniMax 不同
NF4_REPO="${NF4_REPO:-DiffSynth-Studio/MiniMax-H3-NF4}"
W4A16_REPO="${W4A16_REPO:-Ar4ikov/MiniMax-H3-transformer-W4A16-RTN}"   # 只在 HF 上, 没有魔搭镜像
ROOT="$DEST/MiniMax-H3"
NF4_ROOT="$DEST/MiniMax-H3-NF4"
W4A16_ROOT="$DEST/MiniMax-H3-W4A16"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$ROOT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "==== [$(date +%T)] 目标=$DEST | 模型=$MODELS | 去重=$DEDUP | 主源=ModelScope, 回退=hf-mirror"

# ---- 落盘空间预检(只警告, NFS 配额未必反映在 df 上) ----
need=0
for m in $MODELS; do case "$m" in
  nf4)       need=$((need+49));;
  nf4_fl2va) need=$((need+33));;
  w4a16)     need=$((need+36));;
  fl2va)     need=$((need+135));;
  ref2va)    if [ "$DEDUP" = 1 ]; then need=$((need+62)); else need=$((need+135)); fi;;
  diffusers) if [ "$DEDUP" = 1 ]; then need=$((need+134)); else need=$((need+196)); fi;;
esac; done
avail=$(df -Pk "$DEST" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
echo ">>> 预计需 ${need}G, 当前可用 ${avail:-未知}G"
[ -n "${avail:-}" ] && [ "$avail" -lt "$need" ] && echo "  !! 空间可能不够, 建议先删无用权重或分标签分批下"

if ! python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >/tmp/dl_pip.log 2>&1; then
  echo "pip/依赖安装失败(看 /tmp/dl_pip.log), 试 apt install python3-pip 后重跑"; tail -5 /tmp/dl_pip.log; exit 2
fi

# ---- 实时聚合速度监视器 ----
mon(){
  local base prev prevt
  base=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0); prev=$base; prevt=$(date +%s)
  while true; do
    sleep 15
    local now nowt; now=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0); nowt=$(date +%s)
    awk -v b="$base" -v p="$prev" -v n="$now" -v dt="$((nowt-prevt))" -v ts="$(date +%T)" 'BEGIN{
      if(dt<=0)dt=1; printf "[%s] ▼ 本次已下 %.1f GB | 当前 %.0f MB/s\n", ts, (n-b)/1073741824, (n-p)/dt/1048576 }'
    prev=$now; prevt=$nowt
  done
}
mon & MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

# ---- pattern -> 真实文件名 ----
# 新版 ms CLI 的位置参数只吃**字面文件名**, 传 "FL2VA/*" 会被当成真实路径去 GET, 直接 404
# (E3020 获取模型文件失败) 然后白白掉到 hf-mirror。所以先用清单 API 把 pattern 展开。
# HF 那条回退路径走 allow_patterns, 本来就吃 fnmatch, 不用展开。
expand(){ # $1=MS仓 $2..=fnmatch pattern(留空=整仓);逐行输出文件名
  python3 - "$@" <<'PY' 2>/dev/null
import fnmatch, json, sys, urllib.request
repo, pats = sys.argv[1], sys.argv[2:]
url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)
files = [f["Path"] for f in d["Data"]["Files"] if f["Type"] == "blob"]
print("\n".join(p for p in files if not pats or any(fnmatch.fnmatch(p, q) for q in pats)))
PY
}

# ---- 下载封装: ModelScope 优先, 失败回退 HF。$1=标签 $2=MS仓 $3=HF仓 $4=落盘目录 $5..=pattern ----
dl(){
  local tag=$1 msrepo=$2 hfrepo=$3 dest=$4; shift 4
  local pats=("$@") files=()
  mkdir -p "$dest"
  mapfile -t files < <(expand "$msrepo" ${pats[@]+"${pats[@]}"})
  if [ "${#files[@]}" -gt 0 ]; then
    echo ">>> [$(date +%T)] [$tag] ModelScope: $msrepo -> $dest (${#files[@]} 个文件)"
    if modelscope download --model "$msrepo" "${files[@]}" --local_dir "$dest"; then
      echo "  OK(MS) $tag"; return 0
    fi
    echo "  !! ModelScope 下载失败, 回退 hf-mirror: $hfrepo"
  else
    echo "  !! 清单展开为空(API 不通或 pattern 没命中), 直接走 hf-mirror: $hfrepo"
  fi
  if python3 - "$hfrepo" "$dest" ${pats[@]+"${pats[@]}"} <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
pats = sys.argv[3:] or None
snapshot_download(repo_id=repo, local_dir=dest, max_workers=4, allow_patterns=pats)
print("OK")
PY
  then echo "  OK(HF) $tag"; else echo "  !! 两源都失败: $tag"; FAILED="$FAILED $tag"; return 1; fi
}

# ---- 整目录软链去重: $1=链接位置(相对 $ROOT) $2=目标(相对 $ROOT) ----
# 只在目标目录已就位时才链;否则返回 1, 调用方退回真实下载。
link_dir(){
  local from="$ROOT/$1" to="$ROOT/$2"
  [ -d "$to" ] || { echo "  !! 去重源缺失 $2, 改为真实下载 $1"; return 1; }
  if [ -e "$from" ] && [ ! -L "$from" ]; then
    echo "  跳过软链: $1 已是实体目录(要换成软链先手动删)"; return 0
  fi
  mkdir -p "$(dirname "$from")"
  # 用相对路径链, NFS 挂载点换了也不断
  local rel; rel=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],os.path.dirname(sys.argv[2])))" "$to" "$from")
  ln -sfn "$rel" "$from" && echo "  软链 $1 -> $rel" || { echo "  !! 软链失败 $1"; FAILED="$FAILED link:$1"; }
}

for m in $MODELS; do
case "$m" in
  # ---- NF4 量化(A100-40G 首选): DiffSynth-Studio 出品, 单卡可跑 ----
  # DiffSynth 的 processor_config 指向原仓 FL2VA/processor/, 顺手一起拉(几 MB), 免得跑的时候再联网
  nf4)       dl nf4 "$NF4_REPO" "$NF4_REPO" "$NF4_ROOT" "*.safetensors" "*.json" "README.md"
             dl nf4_proc "$MS_REPO" "$HF_REPO" "$ROOT" "FL2VA/processor/*" "FL2VA/tokenizer/*" ;;
  nf4_fl2va) dl nf4_fl2va "$NF4_REPO" "$NF4_REPO" "$NF4_ROOT" \
               "minimax-h3-fl2va-nf4.safetensors" "minimax-h3-text-encoder-nf4.safetensors" \
               "video_vae_nf4.safetensors" "audio_vae_nf4.safetensors" "*.json" "README.md"
             dl nf4_proc "$MS_REPO" "$HF_REPO" "$ROOT" "FL2VA/processor/*" "FL2VA/tokenizer/*" ;;

  # ---- W4A16 auto-round(Plan B): 只有 HF 有源, 走 hf-mirror;只含 DiT, TE 仍要原仓 bf16 ----
  w4a16)     dl w4a16 "$W4A16_REPO" "$W4A16_REPO" "$W4A16_ROOT"
             echo "  提示: 这份只有 transformer, 跑起来还要原仓 bf16 的 text_encoder(62G, MODELS=fl2va)" ;;

  # ---- FL2VA(bf16 原始 checkpoint): t2va + fl2va, 唯一的"全量实体", 后面两个标签都从它软链 ----
  fl2va)     dl fl2va "$MS_REPO" "$HF_REPO" "$ROOT" "FL2VA/*" ;;

  # ---- Ref2VA: 只有 transformer 与 FL2VA 不同, 其余整目录软链 ----
  ref2va)    if [ "$DEDUP" = 1 ]; then
               dl ref2va "$MS_REPO" "$HF_REPO" "$ROOT" "Ref2VA/transformer/*" "Ref2VA/model_index.json"
               ok=1
               for d in text_encoder video_vae audio_vae processor tokenizer; do
                 link_dir "Ref2VA/$d" "FL2VA/$d" || ok=0
               done
               [ "$ok" = 1 ] || dl ref2va_full "$MS_REPO" "$HF_REPO" "$ROOT" "Ref2VA/*"
             else
               dl ref2va "$MS_REPO" "$HF_REPO" "$ROOT" "Ref2VA/*"
             fi ;;

  # ---- diffusers 模块化格式(根目录): transformer/transformer_ref/vae 是独立副本, 必须真下 ----
  diffusers) if [ "$DEDUP" = 1 ]; then
               dl diffusers "$MS_REPO" "$HF_REPO" "$ROOT" \
                  "transformer/*" "transformer_ref/*" "vae/*" "audio_vae/*" \
                  "scheduler/*" "audio_scheduler/*" "modular_model_index.json" "configuration.json"
               ok=1
               for d in text_encoder tokenizer processor; do
                 link_dir "$d" "FL2VA/$d" || ok=0
               done
               [ "$ok" = 1 ] || dl diffusers_te "$MS_REPO" "$HF_REPO" "$ROOT" \
                  "text_encoder/*" "tokenizer/*" "processor/*"
             else
               dl diffusers "$MS_REPO" "$HF_REPO" "$ROOT" \
                  "transformer/*" "transformer_ref/*" "vae/*" "audio_vae/*" \
                  "scheduler/*" "audio_scheduler/*" "text_encoder/*" "tokenizer/*" "processor/*" \
                  "modular_model_index.json" "configuration.json"
             fi ;;

  # ---- 文档 / 复现脚本 / demo ----
  docs)      dl docs "$MS_REPO" "$HF_REPO" "$ROOT" "README.md" "LICENSE" "configuration.json" "docs/*" "scripts/*" ;;
  assets)    dl assets "$MS_REPO" "$HF_REPO" "$ROOT" "assets/*" ;;
  *) echo "!! 未知标签: $m"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 下载结束, 校对 ===="

check(){ # $1=绝对路径 $2=最小字节(0=只判存在)
  local f="$1" sz
  sz=$(du -sbL "$f" 2>/dev/null | cut -f1 || echo 0)
  local mark=""; [ -L "$f" ] && mark=" (软链)"
  if [ -e "$f" ] && [ "$sz" -ge "$2" ]; then
    echo "  ✓ ${f#$DEST/} ($(numfmt --to=iec "$sz" 2>/dev/null || echo "${sz}B"))$mark"; return 0
  else echo "  ✗ 缺失或不完整: ${f#$DEST/}"; return 1; fi
}
miss=0
for m in $MODELS; do
case "$m" in
  nf4)       check "$NF4_ROOT/minimax-h3-fl2va-nf4.safetensors"        15000000000 || miss=1
             check "$NF4_ROOT/minimax-h3-ref2va-nf4.safetensors"       15000000000 || miss=1
             check "$NF4_ROOT/minimax-h3-text-encoder-nf4.safetensors" 14000000000 || miss=1
             check "$NF4_ROOT/video_vae_nf4.safetensors"                1400000000 || miss=1
             check "$NF4_ROOT/audio_vae_nf4.safetensors"                 250000000 || miss=1
             check "$ROOT/FL2VA/processor"                                       0 || miss=1 ;;
  nf4_fl2va) check "$NF4_ROOT/minimax-h3-fl2va-nf4.safetensors"        15000000000 || miss=1
             check "$NF4_ROOT/minimax-h3-text-encoder-nf4.safetensors" 14000000000 || miss=1
             check "$NF4_ROOT/video_vae_nf4.safetensors"                1400000000 || miss=1
             check "$NF4_ROOT/audio_vae_nf4.safetensors"                 250000000 || miss=1
             check "$ROOT/FL2VA/processor"                                       0 || miss=1 ;;
  w4a16)     check "$W4A16_ROOT/config.json"                          0 || miss=1
             check "$W4A16_ROOT/quantization_config.json"             0 || miss=1
             n=$(ls "$W4A16_ROOT"/diffusion_pytorch_model-*.safetensors 2>/dev/null | wc -l)
             [ "$n" = 10 ] && echo "  ✓ W4A16 分片 10/10" || { echo "  ✗ W4A16 分片 $n/10"; miss=1; } ;;
  fl2va)     check "$ROOT/FL2VA/transformer"   60000000000 || miss=1
             check "$ROOT/FL2VA/text_encoder"  60000000000 || miss=1
             check "$ROOT/FL2VA/video_vae"      9000000000 || miss=1
             check "$ROOT/FL2VA/audio_vae"       500000000 || miss=1
             check "$ROOT/FL2VA/model_index.json"        0 || miss=1 ;;
  ref2va)    check "$ROOT/Ref2VA/transformer"  60000000000 || miss=1
             check "$ROOT/Ref2VA/text_encoder" 60000000000 || miss=1
             check "$ROOT/Ref2VA/video_vae"     9000000000 || miss=1
             check "$ROOT/Ref2VA/audio_vae"      500000000 || miss=1
             check "$ROOT/Ref2VA/model_index.json"       0 || miss=1 ;;
  diffusers) check "$ROOT/transformer"         60000000000 || miss=1
             check "$ROOT/transformer_ref"     60000000000 || miss=1
             check "$ROOT/vae"                  9000000000 || miss=1
             check "$ROOT/audio_vae"             500000000 || miss=1
             check "$ROOT/text_encoder"        60000000000 || miss=1
             check "$ROOT/modular_model_index.json"      0 || miss=1 ;;
  docs)      check "$ROOT/README.md" 0 || miss=1 ;;
esac
done

# ---- 逐文件大小审计:拿官方清单跟本地 stat 对, 揪出"下了一半"的分片 ----
if [ "$VERIFY" = 1 ]; then
  echo "---- [$(date +%T)] 逐文件大小审计(对 ModelScope 官方清单) ----"
  # 退出码约定: 0=全对 3=有文件大小不符 4=清单拉不到(不判失败)
  python3 - "$ROOT" "$NF4_ROOT" "$MS_REPO" "$NF4_REPO" $MODELS <<'PY'
import fnmatch, json, os, sys, urllib.request
root, nf4_root, ms_repo, nf4_repo = sys.argv[1:5]
tags = set(sys.argv[5:])

def manifest(repo):
    url = (f"https://www.modelscope.cn/api/v1/models/{repo}"
           "/repo/files?Revision=master&Recursive=true")
    with urllib.request.urlopen(url, timeout=60) as r:
        return [f for f in json.load(r)["Data"]["Files"] if f["Type"] == "blob"]

pref = {"fl2va": ("FL2VA/",), "ref2va": ("Ref2VA/",), "assets": ("assets/",),
        "docs": ("docs/", "scripts/", "README.md", "LICENSE"),
        "nf4": ("FL2VA/processor/", "FL2VA/tokenizer/"),
        "nf4_fl2va": ("FL2VA/processor/", "FL2VA/tokenizer/"),
        "diffusers": ("transformer/", "transformer_ref/", "vae/", "audio_vae/", "scheduler/",
                      "audio_scheduler/", "text_encoder/", "tokenizer/", "processor/",
                      "modular_model_index.json")}
nf4_pat = {"nf4": ("*.safetensors",),
           "nf4_fl2va": ("minimax-h3-fl2va-nf4.safetensors",
                         "minimax-h3-text-encoder-nf4.safetensors",
                         "video_vae_nf4.safetensors", "audio_vae_nf4.safetensors")}

jobs = []
want = tuple(p for t in tags for p in pref.get(t, ()))
if want:
    jobs.append((ms_repo, root, lambda p, w=want: p.startswith(w)))
npat = tuple(p for t in tags for p in nf4_pat.get(t, ()))
if npat:
    jobs.append((nf4_repo, nf4_root,
                 lambda p, w=npat: any(fnmatch.fnmatch(p, q) for q in w)))

bad = ok = 0
for repo, base, sel in jobs:
    try:
        files = manifest(repo)
    except Exception as e:
        print(f"  (审计跳过 {repo}:清单拉取失败 {e}, 不影响已下文件)")
        sys.exit(4)
    for f in files:
        if not sel(f["Path"]):
            continue
        lp = os.path.join(base, f["Path"])
        have = os.path.getsize(lp) if os.path.exists(lp) else -1   # 跟随软链
        if have == f["Size"]:
            ok += 1
        else:
            bad += 1
            print(f"  ✗ {base}/{f['Path']}: 本地 {have} != 远端 {f['Size']}")
print(f"  审计: {ok} 个文件大小一致, {bad} 个异常" + ("" if bad else " —— 全部完整 ✅"))
sys.exit(3 if bad else 0)
PY
  rc=$?
  [ "$rc" = 3 ] && miss=1
fi

if [ -n "$FAILED" ] || [ "$miss" -ne 0 ]; then
  echo "!! 未完成, 重跑本脚本会续传:$FAILED"; exit 1
fi
echo "完成, 全部成功。总占用:"
du -sh "$ROOT" "$NF4_ROOT" 2>/dev/null
cat <<EOF

下一步 A —— NF4 单卡验证(推荐先走这条, ~32G 一张 A100 装得下):
  git clone https://github.com/modelscope/DiffSynth-Studio.git && cd DiffSynth-Studio
  pip install -e ".[quant]"     # bitsandbytes;鲲鹏 ARM 上可用(HunyuanImage3 那次已验)
  # 权重走本地 NFS, 别让它联网:ModelConfig 传 path=$NF4_ROOT/minimax-h3-fl2va-nf4.safetensors
  #                             processor_config 指向 $ROOT/FL2VA/processor
  # 完整示例见 $NF4_ROOT/README.md;offload_device 有内存就用 "cpu"(比 disk 快)
  # ⚠️ NF4 是 weight-only, 只省显存不提速;ref2va 属编辑类任务, 画质要跟 bf16 对照着看

下一步 B —— bf16 原始权重(画质基线, 4×A100-40G 装不下, 需 TP/offload):
  sglang serve --model-path $ROOT --num-gpus 4 --ulysses-degree 4 \\
    --performance-mode speed --host 0.0.0.0 --port 30010 --model-variant fl2va
  # ulysses 是序列并行, 每卡都要放全量 61.7G 权重 —— 直接起会 OOM, 先算好显存再试
  # 复现用例请求体见 $ROOT/scripts/readme/reproducible-768p-*.sh

通用:
  # 提示词写法(H3-Context-IR 没开源, 输入质量全靠它)见 $ROOT/docs/VIDEO_PROMPT_WRITING_GUIDE_*.md
  # 2K 要走官方 H3-Regenerate-2K API, 本地只能出 768p
EOF
