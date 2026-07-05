#!/usr/bin/env bash
# =============================================================================
# 下载 lightx2v 组织 2026-06/07 新模型到 NFS(A100 视角:只收 bf16/LoRA,跳 NVFP4)。
# ModelScope 优先(国内快), 失败回退 hf-mirror。实时测速 + 断点续传 + 失败追踪。
#
# 默认下(A100 能用 + 你 NFS 上还没有的新模型):
#   qwen_2512_lora   Qwen-Image-2512-Lightning       文生图 8 步蒸馏 LoRA(配已有 Qwen-Image base)
#   qwen_lightning   Qwen-Image-Lightning            Qwen 蒸馏 LoRA
#   qwen_edit_lora   Qwen-Image-Edit-2511-Lightning  图生图/编辑 蒸馏 LoRA
#   qwen_vae         qwen-image-vae                  Qwen 快速 VAE(810M)
#   wan21_official   Wan2.1-Official-Models          Wan2.1 base(你只有 1.3B, 没 14B; A100 轻量 dense)
#   wan21_13b_distill  Wan2.1-T2V-1.3B-Distill-Models 1.3B 蒸馏
#   wan21_13b_longcat  Wan2.1-T2V-1.3B-longcat-step500 1.3B longcat LoRA(175M)
#   qwen_2512_base   Qwen/Qwen-Image-2512            Qwen-Image-2512 基座(~58G; 配 qwen_2512_lora)
#   qwen_edit_base   Qwen/Qwen-Image-Edit-2511       图生图基座(~58G; 配 qwen_edit_lora)
#   qwen_2512_lora   Qwen-Image-2512-Lightning       2512 的 8 步蒸馏 LoRA
#   qwen_edit_lora   Qwen-Image-Edit-2511-Lightning  图生图/编辑 蒸馏 LoRA
#
# 可选(默认不下, 用 MODELS=... 显式触发):
#   wan22_official   Wan2.2-Official-Models          ⚠️ 内容已有(Wan-AI/Wan2.2-T2V-A14B), 一般不用
#   encoders         Encoders                        ⚠️ 编码器随模型自带, 冗余(~22G)
#   encoders_lx2v    Encoders-Lightx2v               ⚠️ 同上
#
# 不提供(A100 sm_80 硬件不支持, 加载即报错):
#   Wan2.2-NVFP4-Sparse / Self-Forcing-NVFP4  —— NVFP4 要 Blackwell
#
# 用法(在挂了 NFS 的机器上, 先 scp 本脚本过去):
#   tmux new -s dl -d 'bash download_new_models.sh'                 # 挂后台(默认全下上面 7 个)
#   tail -f /nfs-models/wuhanjisuan894/dl_new_models.log            # 看速度+进度
#   MODELS="qwen_2512_lora qwen_vae" bash download_new_models.sh    # 只下指定
# 中断后重跑会自动续传。
# =============================================================================
set -u
DEST="${DEST:-/nfs-models/wuhanjisuan894/models}"      # 与现有模型同一根(qwen/z-image/wan 都在这)
LOG="${LOG:-/nfs-models/wuhanjisuan894/dl_new_models.log}"
MODELS="${MODELS:-qwen_lightning qwen_vae qwen_2512_base qwen_2512_lora qwen_edit_base qwen_edit_lora wan21_official wan21_13b_distill wan21_13b_longcat}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "目标=$DEST | 模型=$MODELS | 主源=ModelScope, 回退=hf-mirror"
# 全新系统可能没有 pip:先确保 pip 可用(走 apt),再装下载工具
python3 -m pip --version >/dev/null 2>&1 || { echo "pip 缺失, apt 安装 python3-pip..."; apt-get install -y python3-pip >/tmp/dl_pip.log 2>&1; }
if ! python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >>/tmp/dl_pip.log 2>&1; then
  echo "pip/依赖安装失败(看 /tmp/dl_pip.log)"; tail -8 /tmp/dl_pip.log; exit 2
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

# ---- 下载封装: ModelScope 优先, 失败回退 HF。参数: $1=MS仓 $2=目标子目录 $3=HF回退仓 [$4..=文件名(留空=整仓)] ----
dl(){
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
  # ---- Qwen 文生图/图生图 增强(配已有的 Qwen-Image base + int8) ----
  qwen_2512_lora)    dl lightx2v/Qwen-Image-2512-Lightning      "loras/Qwen-Image-2512-Lightning"      lightx2v/Qwen-Image-2512-Lightning ;;
  qwen_lightning)    dl lightx2v/Qwen-Image-Lightning           "loras/Qwen-Image-Lightning"           lightx2v/Qwen-Image-Lightning ;;
  qwen_edit_lora)    dl lightx2v/Qwen-Image-Edit-2511-Lightning "loras/Qwen-Image-Edit-2511-Lightning" lightx2v/Qwen-Image-Edit-2511-Lightning ;;
  qwen_vae)          dl lightx2v/qwen-image-vae                 "Qwen-Image/qwen-image-vae"            lightx2v/qwen-image-vae ;;
  # ---- Qwen 新基座(~58G/个; 各自的 Lightning LoRA 需要配对基座) ----
  qwen_2512_base)    dl Qwen/Qwen-Image-2512                    "Qwen-Image-2512"                      Qwen/Qwen-Image-2512 ;;
  qwen_edit_base)    dl Qwen/Qwen-Image-Edit-2511               "Qwen-Image-Edit-2511"                 Qwen/Qwen-Image-Edit-2511 ;;
  # ---- Wan2.1 轻量(A100 上比 Wan2.2 MoE 省心) ----
  wan21_official)    dl lightx2v/Wan2.1-Official-Models         "Wan2.1-Official-Models"               lightx2v/Wan2.1-Official-Models ;;
  wan21_13b_distill) dl lightx2v/Wan2.1-T2V-1.3B-Distill-Models "Wan2.1-T2V-1.3B-Distill-Models"       lightx2v/Wan2.1-T2V-1.3B-Distill-Models ;;
  wan21_13b_longcat) dl lightx2v/Wan2.1-T2V-1.3B-longcat-step500 "loras/Wan2.1-T2V-1.3B-longcat-step500" lightx2v/Wan2.1-T2V-1.3B-longcat-step500 ;;
  # ---- 可选(默认不在 MODELS 里; 已有/冗余) ----
  wan22_official)    dl lightx2v/Wan2.2-Official-Models         "Wan2.2-Official-Models"               lightx2v/Wan2.2-Official-Models ;;
  encoders)          dl lightx2v/Encoders                       "Encoders"                             lightx2v/Encoders ;;
  encoders_lx2v)     dl lightx2v/Encoders-Lightx2v              "Encoders-Lightx2v"                    lightx2v/Encoders-Lightx2v ;;
  *) echo "!! 未知标签: $m"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 全部完成 ===="
for m in $MODELS; do
  case "$m" in
    qwen_2512_lora)    d="loras/Qwen-Image-2512-Lightning" ;;
    qwen_lightning)    d="loras/Qwen-Image-Lightning" ;;
    qwen_edit_lora)    d="loras/Qwen-Image-Edit-2511-Lightning" ;;
    qwen_vae)          d="Qwen-Image/qwen-image-vae" ;;
    qwen_2512_base)    d="Qwen-Image-2512" ;;
    qwen_edit_base)    d="Qwen-Image-Edit-2511" ;;
    wan21_official)    d="Wan2.1-Official-Models" ;;
    wan21_13b_distill) d="Wan2.1-T2V-1.3B-Distill-Models" ;;
    wan21_13b_longcat) d="loras/Wan2.1-T2V-1.3B-longcat-step500" ;;
    wan22_official)    d="Wan2.2-Official-Models" ;;
    encoders)          d="Encoders" ;;
    encoders_lx2v)     d="Encoders-Lightx2v" ;;
    *) continue ;;
  esac
  du -sh "$DEST/$d" 2>/dev/null || echo "  (缺) $DEST/$d"
done
if [ -n "$FAILED" ]; then echo "!! 以下有失败, 需重跑(会续传):$FAILED"; exit 1; fi
echo "完成, 全部成功。"
