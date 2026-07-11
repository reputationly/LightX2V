#!/usr/bin/env bash
# =============================================================================
# 下载 InfiniteTalk 四件套到 NFS(音频数字人, MeiGen-AI, Wan2.1 底座, 有 4 步蒸馏)。
# 只写新目录, 不碰已有模型。总量 ~90G。
#
# 四件套(标签 | 大小 | 仓库):
#   base     ~43G  Wan-AI/Wan2.1-I2V-14B-480P(底座: DiT+umT5+CLIP+VAE)
#   adapter  ~10G  MeiGen-AI/InfiniteTalk(音频 adapter, single+multi)
#   wav2vec  ~1.4G TencentGameMate/chinese-wav2vec2-base(音频编码器)
#   distill  ~33G  lightx2v/Wan2.1-Distill-Models 只取
#                  wan2.1_i2v_480p_lightx2v_4step.safetensors(4步蒸馏 DiT, bf16)
#
# 用法(服务器上):
#   tmux new -s dl_it -d 'bash /root/download_infinitetalk.sh'
#   tail -f /nfs-models/wuhanjisuan894/dl_infinitetalk.log
#   MODELS="adapter wav2vec" bash /root/download_infinitetalk.sh   # 只下指定
# 中断重跑自动续传。
# =============================================================================
set -u
DEST="${DEST:-/nfs-models/wuhanjisuan894/models}"
LOG="${LOG:-/nfs-models/wuhanjisuan894/dl_infinitetalk.log}"
MODELS="${MODELS:-base adapter wav2vec distill}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "==== [$(date +%T)] 目标=$DEST | 模型=$MODELS | 主源=ModelScope, 回退=hf-mirror"
python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >/tmp/dl_pip.log 2>&1 || { echo "pip 失败(看 /tmp/dl_pip.log)"; exit 2; }

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

dl(){ # $1=MS仓 $2=目标子目录 $3=HF回退仓 [$4..=具体文件(留空=整仓)]
  local msid=$1 sub=$2 hfid=$3; shift 3
  local dest="$DEST/$sub" files=("$@")
  echo ">>> [$(date +%T)] ModelScope: $msid -> $dest"
  if modelscope download --model "$msid" ${files[@]+"${files[@]}"} --local_dir "$dest"; then
    echo "  OK(MS) $msid"; return
  fi
  echo "  !! ModelScope 失败, 回退 hf-mirror: $hfid"
  if python3 - "$hfid" "$dest" ${files[@]+"${files[@]}"} <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
pats = sys.argv[3:] or None
snapshot_download(repo_id=repo, local_dir=dest, max_workers=4, allow_patterns=pats)
print("OK")
PY
  then echo "  OK(HF) $hfid"; else echo "  !! 两源都失败: $msid"; FAILED="$FAILED $msid"; fi
}

for m in $MODELS; do
case "$m" in
  base)    dl Wan-AI/Wan2.1-I2V-14B-480P          "Wan2.1-I2V-14B-480P"                    Wan-AI/Wan2.1-I2V-14B-480P ;;
  adapter) dl MeiGen-AI/InfiniteTalk               "MeiGen-AI/InfiniteTalk"                 MeiGen-AI/InfiniteTalk ;;
  wav2vec) dl TencentGameMate/chinese-wav2vec2-base "TencentGameMate/chinese-wav2vec2-base" TencentGameMate/chinese-wav2vec2-base ;;
  distill) dl lightx2v/Wan2.1-Distill-Models       "Wan2.1-Distill-Models"                  lightx2v/Wan2.1-Distill-Models \
             wan2.1_i2v_480p_lightx2v_4step.safetensors ;;
  distill720) dl lightx2v/Wan2.1-Distill-Models    "Wan2.1-Distill-Models"                  lightx2v/Wan2.1-Distill-Models \
             wan2.1_i2v_720p_lightx2v_4step.safetensors ;;
  distill480_int8) dl lightx2v/Wan2.1-Distill-Models "Wan2.1-Distill-Models"                lightx2v/Wan2.1-Distill-Models \
             wan2.1_i2v_480p_int8_lightx2v_4step.safetensors ;;
  *) echo "!! 未知标签: $m"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 下载结束 ===="

check(){ # $1=相对路径 $2=最小字节
  local f="$DEST/$1" sz
  sz=$(du -sb "$f" 2>/dev/null | cut -f1 || echo 0)
  if [ "$sz" -ge "$2" ]; then echo "  ✓ $1 ($(numfmt --to=iec $sz 2>/dev/null || echo ${sz}B))"; return 0
  else echo "  ✗ 缺失或不完整: $1"; return 1; fi
}
miss=0
for m in $MODELS; do
case "$m" in
  base)    check "Wan2.1-I2V-14B-480P" 40000000000 || miss=1 ;;
  adapter) check "MeiGen-AI/InfiniteTalk" 4000000000 || miss=1 ;;
  wav2vec) check "TencentGameMate/chinese-wav2vec2-base" 300000000 || miss=1 ;;
  distill) check "Wan2.1-Distill-Models/wan2.1_i2v_480p_lightx2v_4step.safetensors" 32000000000 || miss=1 ;;
esac
done
if [ -n "$FAILED" ] || [ "$miss" -ne 0 ]; then echo "!! 未完成, 重跑会续传:$FAILED"; exit 1; fi
echo "完成, 全部成功。"
