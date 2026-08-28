#!/usr/bin/env bash
# =============================================================================
# 下载 Wan-Animate-2(角色动画, 14B 蒸馏)与 SwiftVR(实时一步式视频修复)到 NFS。
# 沿用 download_minimax_h3.sh 的套路:ModelScope 主源(国内快)→ hf-mirror 回退,
# 清单 API 展开 pattern + 实时测速 + 断点续传 + 整目录软链去重 + 逐文件大小审计。
# 只写新目录, 不碰已有模型。
#
# ---------------------------------------------------------------------------
# Wan-Animate-2(仓 Wan-AI/Wan2.2-Animate-2-14B, 全量 76.9G)
# ---------------------------------------------------------------------------
#   wan_animate_2/wan_animate_2_bf16_distillation.safetensors  30.5G  蒸馏版 DiT ← 上游配置用的就是它
#   wan_animate_2/wan_animate_2_bf16.safetensors               30.5G  非蒸馏版, 默认不下
#   videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth          10.6G  ← 与 Animate 一代同文件, 默认软链
#   videomodel/Wan-AI/models_clip_open-clip-...-vit-huge-14.pth 4.4G  ← 同上
#   videomodel/Wan-AI/vae.pth                                  0.73G  必须真下:一代的 Wan2.1_VAE.pth
#                                                                     只有 507609928 字节, 不是同一个
#   videomodel/Wan-AI/{umt5-xxl,xlm-roberta-large}/*           ~40M   分词器
#
#   去重(DEDUP=1, 默认开)会**先算本地 sha256 再跟远端清单比**, 一致才软链, 不一致自动退回真下载。
#   两个候选各 11.4G / 4.8G, 校验各需一两分钟(NFS 顺序读 ~270MB/s)。DEDUP=0 直接全量真下。
#   去重后蒸馏路径实际落盘约 31.3G(全量 76.9G → 省 46G:16.1G 软链 + 30.5G 不下非蒸馏版)。
#
#   ⚠️ 上游 LightX2V 已原生支持(PR #1430), runner=wan_animate2_runner, model_cls=
#      "wan2.2_animate2_distilled", task=animate, 配置 configs/wan22/wan_animate2_distill.json。
#      **它和一代不是同一条路**:一代靠 pose/face adapter 预处理, 二代 runner 明确
#      "intentionally registered separately", 只吃 --image_path(参考图) + --video_path(驱动视频)。
#   ⚠️ 官方配置在我们 ARM 镜像上跑不了, 下完要改两处, 见文末提示。
#
# ---------------------------------------------------------------------------
# SwiftVR(仓 H-oliday/SwiftVR, 推理必需 4 个文件 ~20.2G)
# ---------------------------------------------------------------------------
#   transformer/diffusion_pytorch_model.safetensors  19.9G  DiT(骨架 Wan2.2-TI2V-5B)
#   transformer/config.json                           495B
#   reae.safetensors                                  156M  Restoration-aware AE
#   prompt_embedding.safetensors                      4.0M  空提示词固定 embedding
#   assets/(官方 demo 175M) 默认不下, 要做主观对照再加 swiftvr_assets 标签
#
#   ⚠️ 官方是 diffusers key 布局, LightX2V 要 wan_dit 布局, 下完必须转一道, 见文末提示。
#
# 标签(默认 "animate2 swiftvr"):
#   animate2         ~31G   蒸馏版 DiT + vae + 分词器(+ T5/CLIP 软链自一代)
#   animate2_base    +30.5G 追加非蒸馏 bf16 DiT(画质基线对照用, 默认不下)
#   swiftvr          ~20.2G SwiftVR 推理必需 4 件
#   swiftvr_assets   0.17G  SwiftVR 官方 demo 视频/对比图
#
# 用法(manager 或任意挂了 NFS 的节点, 先 scp 到 /root):
#   tmux new -s dl_a2sv -d 'bash /root/download_animate2_swiftvr.sh'
#   tail -f "$(dirname "${DEST:-/nfs-data/models}")/dl_animate2_swiftvr.log" | grep --line-buffered ▼
#   MODELS="animate2" bash /root/download_animate2_swiftvr.sh        # 只下 Animate-2
#   MODELS="animate2 animate2_base" bash ...                          # 连非蒸馏版一起
#   DEDUP=0 bash ...                                                  # 关去重, T5/CLIP 各下一份
#   VERIFY=0 bash ...                                                 # 跳过末尾逐文件大小审计
# 中断后重跑本脚本会自动续传。
# =============================================================================
set -u
# DEST 自动适配:计算节点有 /nfs-data 软链;manager 节点用真身 /nfs-models/wuhanjisuan894/models
DEST="${DEST:-$([ -d /nfs-data/models ] && echo /nfs-data/models || echo /nfs-models/wuhanjisuan894/models)}"
LOG="${LOG:-$(dirname "$DEST")/dl_animate2_swiftvr.log}"
MODELS="${MODELS:-animate2 swiftvr}"
DEDUP="${DEDUP:-1}"
VERIFY="${VERIFY:-1}"

A2_MS="${A2_MS:-Wan-AI/Wan2.2-Animate-2-14B}"
A2_HF="${A2_HF:-Wan-AI/Wan2.2-Animate-2-14B}"
A2_ROOT="$DEST/Wan2.2-Animate-2-14B"
A1_ROOT="$DEST/Wan2.2-Animate-14B"        # 一代, 去重源

SV_MS="${SV_MS:-H-oliday/SwiftVR}"
SV_HF="${SV_HF:-H-oliday/SwiftVR}"
SV_ROOT="$DEST/SwiftVR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "==== [$(date +%T)] 目标=$DEST | 模型=$MODELS | 去重=$DEDUP | 主源=ModelScope, 回退=hf-mirror"

# ---- 落盘空间预检(只警告, NFS 配额未必反映在 df 上) ----
need=0
for m in $MODELS; do case "$m" in
  animate2)       if [ "$DEDUP" = 1 ]; then need=$((need+32)); else need=$((need+48)); fi;;
  animate2_base)  need=$((need+31));;
  swiftvr)        need=$((need+21));;
  swiftvr_assets) need=$((need+1));;
esac; done
avail=$(df -Pk "$DEST" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
echo ">>> 预计需 ${need}G, 当前可用 ${avail:-未知}G"
[ -n "${avail:-}" ] && [ "$avail" -lt "$need" ] && echo "  !! 空间可能不够"

# ---- 依赖:先探测, 已装就别碰 pip ----
# 踩坑(manager 节点):直连 PyPI 不通时 `pip install -U` 会在 do_poll 上无限干等,
# 日志一行不动看着像死了, 而 modelscope/huggingface_hub 其实早装好了。
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
if python3 -c "import modelscope, huggingface_hub" >/dev/null 2>&1; then
  echo "  依赖已就绪(跳过 pip)"
else
  echo "  缺依赖, 安装中(源=$PIP_INDEX, 超时 300s)..."
  if ! timeout 300 python3 -m pip install -q -i "$PIP_INDEX" --timeout 20 --retries 2 \
        modelscope "huggingface_hub[cli]" >/tmp/dl_pip.log 2>&1; then
    echo "  !! 依赖安装失败/超时(看 /tmp/dl_pip.log)。换源:PIP_INDEX=https://mirrors.aliyun.com/pypi/simple"
    tail -5 /tmp/dl_pip.log; exit 2
  fi
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
# 新版 ms CLI 的位置参数只吃**字面文件名**, 传 "wan_animate_2/*" 会被当成真实路径去 GET
# 直接 404, 然后白白掉到 hf-mirror。所以先用清单 API 把 pattern 展开。
# HF 那条回退路径走 allow_patterns, 本来就吃 fnmatch, 不用展开。
expand(){ # $1=MS仓 $2..=fnmatch pattern(留空=整仓)
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

# ---- 校验过的软链去重 ----
# $1=远端相对路径(在 A2 仓里) $2=本地已有的候选实体(绝对路径)
# 先算本地 sha256 跟清单比, 一致才软链。不一致/缺失返回 1, 调用方退回真实下载。
# 只比大小是不够的:同名不同版的 Wan 权重字节数撞车过。
link_verified(){
  local rel="$1" src="$2" dst="$A2_ROOT/$1"
  [ -f "$src" ] || { echo "  去重源不存在 $src → 改为真实下载 $rel"; return 1; }
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    echo "  跳过软链: $rel 已是实体文件(要换成软链先手动删)"; return 0
  fi
  local want; want=$(expand_sha "$rel")
  [ -n "$want" ] || { echo "  拿不到远端 sha($rel) → 改为真实下载"; return 1; }
  echo "  校验去重候选 $rel ← $(basename "$src") ($(numfmt --to=iec "$(stat -c%s "$src")" 2>/dev/null))..."
  local got; got=$(sha256sum "$src" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "  sha 不符(本地 ${got:0:12} != 远端 ${want:0:12}) → 改为真实下载 $rel"; return 1
  fi
  mkdir -p "$(dirname "$dst")"
  # 用相对路径链, NFS 挂载点换了也不断
  local link; link=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],os.path.dirname(sys.argv[2])))" "$src" "$dst")
  ln -sfn "$link" "$dst" && echo "  ✓ sha 一致, 软链 $rel -> $link" || { echo "  !! 软链失败 $rel"; return 1; }
}

expand_sha(){ # $1=远端相对路径 -> 输出 Sha256
  python3 - "$A2_MS" "$1" <<'PY' 2>/dev/null
import json, sys, urllib.request
repo, want = sys.argv[1], sys.argv[2]
url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
with urllib.request.urlopen(url, timeout=60) as r:
    for f in json.load(r)["Data"]["Files"]:
        if f.get("Path") == want:
            print(f.get("Sha256", "")); break
PY
}

for m in $MODELS; do
case "$m" in

  # ---- Wan-Animate-2 蒸馏版 ----
  animate2)
    dl animate2_dit "$A2_MS" "$A2_HF" "$A2_ROOT" \
       "wan_animate_2/wan_animate_2_bf16_distillation.safetensors"
    dl animate2_rest "$A2_MS" "$A2_HF" "$A2_ROOT" \
       "videomodel/Wan-AI/vae.pth" \
       "videomodel/Wan-AI/umt5-xxl/*" \
       "videomodel/Wan-AI/xlm-roberta-large/*" \
       "README.md" "configuration.json"
    if [ "$DEDUP" = 1 ]; then
      link_verified "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth" \
                    "$A1_ROOT/models_t5_umt5-xxl-enc-bf16.pth" \
        || dl animate2_t5 "$A2_MS" "$A2_HF" "$A2_ROOT" "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth"
      link_verified "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
                    "$A1_ROOT/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        || dl animate2_clip "$A2_MS" "$A2_HF" "$A2_ROOT" "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    else
      dl animate2_te "$A2_MS" "$A2_HF" "$A2_ROOT" \
         "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth" \
         "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    fi ;;

  # ---- 非蒸馏 bf16(画质基线对照, 默认不下) ----
  animate2_base)
    dl animate2_base "$A2_MS" "$A2_HF" "$A2_ROOT" "wan_animate_2/wan_animate_2_bf16.safetensors" ;;

  # ---- SwiftVR 推理必需 4 件 ----
  swiftvr)
    dl swiftvr "$SV_MS" "$SV_HF" "$SV_ROOT" \
       "transformer/config.json" \
       "transformer/diffusion_pytorch_model.safetensors" \
       "reae.safetensors" "prompt_embedding.safetensors" \
       "configuration.json" "README.md" ;;

  swiftvr_assets)
    dl swiftvr_assets "$SV_MS" "$SV_HF" "$SV_ROOT" "assets/*" ;;

  *) echo "!! 未知标签: $m (支持: animate2 animate2_base swiftvr swiftvr_assets)"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 下载结束, 校对 ===="

check(){ # $1=绝对路径 $2=最小字节(0=只判存在)
  local f="$1" sz mark=""
  sz=$(du -sbL "$f" 2>/dev/null | cut -f1 || echo 0)
  [ -L "$f" ] && mark=" (软链)"
  if [ -e "$f" ] && [ "$sz" -ge "$2" ]; then
    echo "  ✓ ${f#$DEST/} ($(numfmt --to=iec "$sz" 2>/dev/null || echo "${sz}B"))$mark"; return 0
  else echo "  ✗ 缺失或不完整: ${f#$DEST/}"; return 1; fi
}
miss=0
for m in $MODELS; do
case "$m" in
  animate2)      check "$A2_ROOT/wan_animate_2/wan_animate_2_bf16_distillation.safetensors" 30000000000 || miss=1
                 check "$A2_ROOT/videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth"          11000000000 || miss=1
                 check "$A2_ROOT/videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" 4000000000 || miss=1
                 check "$A2_ROOT/videomodel/Wan-AI/vae.pth"                                    700000000 || miss=1
                 check "$A2_ROOT/videomodel/Wan-AI/umt5-xxl"                                           0 || miss=1 ;;
  animate2_base) check "$A2_ROOT/wan_animate_2/wan_animate_2_bf16.safetensors"               30000000000 || miss=1 ;;
  swiftvr)       check "$SV_ROOT/transformer/diffusion_pytorch_model.safetensors"            19000000000 || miss=1
                 check "$SV_ROOT/reae.safetensors"                                             150000000 || miss=1
                 check "$SV_ROOT/prompt_embedding.safetensors"                                   4000000 || miss=1
                 check "$SV_ROOT/transformer/config.json"                                              0 || miss=1 ;;
esac
done

# ---- 逐文件大小审计:拿官方清单跟本地 stat 对, 揪出"下了一半"的分片 ----
if [ "$VERIFY" = 1 ]; then
  echo "---- [$(date +%T)] 逐文件大小审计 ----"
  # 退出码: 0=全对 3=有文件大小不符 4=清单拉不到(不判失败)
  python3 - "$A2_MS" "$A2_ROOT" "$SV_MS" "$SV_ROOT" $MODELS <<'PY'
import json, os, sys, urllib.request
a2_repo, a2_root, sv_repo, sv_root = sys.argv[1:5]
tags = set(sys.argv[5:])

def manifest(repo):
    url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [f for f in json.load(r)["Data"]["Files"] if f["Type"] == "blob"]

jobs = []
a2_want = []
if "animate2" in tags:
    a2_want += ["wan_animate_2/wan_animate_2_bf16_distillation.safetensors", "videomodel/"]
if "animate2_base" in tags:
    a2_want += ["wan_animate_2/wan_animate_2_bf16.safetensors"]
if a2_want:
    jobs.append((a2_repo, a2_root, tuple(a2_want)))
sv_want = []
if "swiftvr" in tags:
    sv_want += ["transformer/", "reae.safetensors", "prompt_embedding.safetensors"]
if "swiftvr_assets" in tags:
    sv_want += ["assets/"]
if sv_want:
    jobs.append((sv_repo, sv_root, tuple(sv_want)))

bad = ok = 0
for repo, base, want in jobs:
    try:
        files = manifest(repo)
    except Exception as e:
        print(f"  (审计跳过 {repo}:清单拉取失败 {e}, 不影响已下文件)")
        sys.exit(4)
    for f in files:
        if not f["Path"].startswith(want):
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
  [ $? = 3 ] && miss=1
fi

if [ -n "$FAILED" ] || [ "$miss" -ne 0 ]; then
  echo "!! 未完成, 重跑本脚本会续传:$FAILED"; exit 1
fi
echo "完成, 全部成功。总占用:"
du -shL "$A2_ROOT" "$SV_ROOT" 2>/dev/null

cat <<EOF

==== Wan-Animate-2 后处理(必做) ====
权重不用转格式, 但**官方配置在我们 ARM 镜像上跑不了**, 拷一份改两处:
  cp configs/wan22/wan_animate2_distill.json configs/wan22/a100/wan_animate2_distill_a100.json
  # 1) "rope_type": "flashinfer_rope"  ->  "torch_real_rope"
  #    镜像里没装 flashinfer(Z-Image / SwiftVR 都踩过同一个坑)
  # 2) "rms_norm_type": "sgl-kernel"   ->  "torch"
  #    镜像里没有 sgl_kernel;sageattention / triton 是有的, 所以
  #    self_attn_1_type=sage_attn2、modulate_type=triton、layer_norm_type=Triton 可以留着
  # 3) 路径字段指到本地:
  #      dit_original_ckpt : $A2_ROOT/wan_animate_2/wan_animate_2_bf16_distillation.safetensors
  #      t5_original_ckpt  : $A2_ROOT/videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth
  #      t5_tokenizer_path : $A2_ROOT/videomodel/Wan-AI/umt5-xxl
  #      clip_original_ckpt: $A2_ROOT/videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
  #      vae_path          : $A2_ROOT/videomodel/Wan-AI/vae.pth
跑法(单卡带 block offload;8 卡 ulysses 用 wan_animate2_distill_8gpu.json):
  python -m lightx2v.infer --model_cls wan2.2_animate2_distilled --task animate \\
    --model_path $A2_ROOT \\
    --config_json <改好的配置> \\
    --image_path <参考图> --video_path <驱动视频> \\
    --save_result_path <输出.mp4>
⚠️ 默认 720x1280 / 81 帧, 40G 单卡是压线的, 先量显存再压时长。

==== SwiftVR 后处理(必做) ====
官方是 diffusers key 布局, LightX2V 要 wan_dit 布局, 跑推理前先转:
  python tools/convert/examples/convert_swiftvr.py \\
      --source $SV_ROOT --output ${SV_ROOT}_lightx2v
  # --output 目录必须不存在;转完自动校验 tensor 数/shape/dtype 与源一致
  # 转换要 torch, 在 lightx2v 容器里跑(需 --gpus, 它 import 时会检查 CUDA 设备)
然后用 configs/swiftvr/a800/swiftvr.json 起推理(A100 用 a800 档), 同样把
rope_type 从 flashinfer_rope 改成 torch_real_rope。
EOF
