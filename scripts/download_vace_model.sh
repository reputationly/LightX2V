#!/usr/bin/env bash
# =============================================================================
# 下载 Wan2.2-VACE-Fun-A14B(可控编辑 VACE, bf16 全套 ~81G)到 NFS。
# 只写入新目录 $DEST/Wan2.2-VACE-Fun-A14B, 不触碰其他已有模型。
#
# 仓库(已核实, 2026-07-09):
#   ModelScope 主源: PAI/Wan2.2-VACE-Fun-A14B
#   HF 回退:        alibaba-pai/Wan2.2-VACE-Fun-A14B(走 hf-mirror)
#   内容: high_noise_model/(34.7G) + low_noise_model/(34.7G)
#         + models_t5_umt5-xxl-enc-bf16.pth(11.4G) + Wan2.1_VAE.pth(0.5G)
#         + google/umt5-xxl tokenizer
#   注: 魔搭无官方 INT8 版, int8 需之后离线量化(参考 Wan2.2-I2V 的做法)
#
# 用法(服务器上, 先 scp 到 /data):
#   tmux new -s dl_vace -d 'bash /data/download_vace_model.sh'
#   tail -f /nfs-models/wuhanjisuan894/dl_vace.log
# 中断后重跑会自动续传。
# =============================================================================
set -u
DEST="${DEST:-/nfs-models/wuhanjisuan894/models}"
SUB="Wan2.2-VACE-Fun-A14B"
LOG="${LOG:-/nfs-models/wuhanjisuan894/dl_vace.log}"
MS_REPO="PAI/Wan2.2-VACE-Fun-A14B"
HF_REPO="alibaba-pai/Wan2.2-VACE-Fun-A14B"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST/$SUB" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "==== [$(date +%T)] 目标=$DEST/$SUB | 主源=ModelScope:$MS_REPO, 回退=hf-mirror:$HF_REPO"
python3 -m pip --version >/dev/null 2>&1 || { echo "pip 缺失, apt 安装 python3-pip..."; apt-get install -y python3-pip >/tmp/dl_pip.log 2>&1; }
if ! python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >>/tmp/dl_pip.log 2>&1; then
  echo "pip/依赖安装失败(看 /tmp/dl_pip.log)"; tail -8 /tmp/dl_pip.log; exit 2
fi

# ---- 实时聚合速度监视器(只统计本模型目录) ----
mon(){
  local base prev prevt
  base=$(du -sb "$DEST/$SUB" 2>/dev/null | cut -f1 || echo 0); prev=$base; prevt=$(date +%s)
  while true; do
    sleep 15
    local now nowt; now=$(du -sb "$DEST/$SUB" 2>/dev/null | cut -f1 || echo 0); nowt=$(date +%s)
    awk -v b="$base" -v p="$prev" -v n="$now" -v dt="$((nowt-prevt))" -v ts="$(date +%T)" 'BEGIN{
      if(dt<=0)dt=1; printf "[%s] ▼ 本次已下 %.1f GB | 当前 %.0f MB/s\n", ts, (n-b)/1073741824, (n-p)/dt/1048576 }'
    prev=$now; prevt=$nowt
  done
}
mon & MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

ok=0
echo ">>> [$(date +%T)] ModelScope: $MS_REPO -> $DEST/$SUB"
if modelscope download --model "$MS_REPO" --local_dir "$DEST/$SUB"; then
  echo "  OK(MS) $MS_REPO"; ok=1
else
  echo "  !! ModelScope 失败, 回退 hf-mirror: $HF_REPO"
  if python3 - "$HF_REPO" "$DEST/$SUB" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=4)
print("OK")
PY
  then echo "  OK(HF) $HF_REPO"; ok=1; fi
fi

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 下载结束 ===="
du -sh "$DEST/$SUB" 2>/dev/null

# ---- 完整性自检: 关键文件都在且大小合理才算成功 ----
check(){ # $1=相对路径 $2=最小字节数
  local f="$DEST/$SUB/$1" sz
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$2" ]; then echo "  ✓ $1 ($(numfmt --to=iec $sz 2>/dev/null || echo ${sz}B))"; return 0
  else echo "  ✗ 缺失或不完整: $1"; return 1; fi
}
miss=0
check "high_noise_model/diffusion_pytorch_model.safetensors" 34000000000 || miss=1
check "low_noise_model/diffusion_pytorch_model.safetensors"  34000000000 || miss=1
check "models_t5_umt5-xxl-enc-bf16.pth"                      11000000000 || miss=1
check "Wan2.1_VAE.pth"                                       500000000   || miss=1
check "google/umt5-xxl/tokenizer.json"                       10000000    || miss=1

if [ "$ok" -ne 1 ] || [ "$miss" -ne 0 ]; then
  echo "!! 未完成, 重跑本脚本会自动续传"; exit 1
fi
echo "完成, 全部成功: $DEST/$SUB"
