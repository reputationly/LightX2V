#!/usr/bin/env bash
# =============================================================================
# 下载新模型到 NFS —— 优先 ModelScope(国内 ~50MB/s, 比 hf-mirror 快 ~37x),
# 失败自动回退 hf-mirror。实时速度 + 断点续传 + 失败追踪。
#
# 模型(标签 | 大小 | 主源 ModelScope 仓库):
#   hy15        ~80G  HunyuanVideo1.5 T2V = Tencent-Hunyuan/HunyuanVideo-1.5
#                      + Qwen2.5-VL/byt5/Glyph 编码器 + lightx2v 蒸馏DiT(16.7G)
#   wan_i2v     114G  lightx2v/Wan2.2-Distill-Models(只取 720p_260412 high+low 两个DiT)
#   wan_animate  72G  Wan-AI/Wan2.2-Animate-14B(角色动画)
#   wan_audio    49G  Wan-AI/Wan2.2-S2V-14B(音频驱动数字人, bf16)
#   qwen_image   58G  Qwen/Qwen-Image(文生图)
#   z_image      33G  Tongyi-MAI/Z-Image-Turbo(文生图, 快)
#   全下 ≈ 406G(NFS 还有 ~2.2T)
#   --- 以下为可选, 不在默认全下里, 用 MODELS=... 显式触发 ---
#   seedvr_3b   ~15G  魔搭 bytedance-community/SeedVR2-3B(超分; DiT+VAE+emb 全, 可直接跑)
#   seedvr_7b   ~66G  魔搭 bytedance-community/SeedVR2-7B(取普通版+锐化版两个 DiT 对比)
#                      ⚠️ 7B 仓库缺 pos/neg_emb, 必须连 seedvr_3b 一起下(借 3B 的 vae+emb)
#                      源: 魔搭社区镜像(快), 回退 HF 官方 ByteDance-Seed/*
#
# 用法(服务器上, 先 scp 到 /data):
#   tmux new -s dl -d 'bash /data/download_models.sh'   # 挂后台(默认全下)
#   tail -f /nfs-data/dl_models.log                     # 看速度+进度
#   MODELS="wan_audio wan_i2v" bash /data/download_models.sh   # 只下指定
#   MODELS="seedvr_3b seedvr_7b" bash /data/download_models.sh # 下 SeedVR2 超分(3B+7B)
# 中断后重跑本脚本会自动续传。
# =============================================================================
set -u
DEST="${DEST:-/nfs-data/models}"
LOG=/nfs-data/dl_models.log
MODELS="${MODELS:-hy15 wan_i2v wan_animate wan_audio qwen_image z_image}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # HF 回退用的镜像
export HF_HUB_DISABLE_XET=1                                   # 禁 Xet(否则慢+超时)
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1   # 全部输出写 $LOG, tail -f 可看
FAILED=""

echo "目标=$DEST | 模型=$MODELS | 主源=ModelScope, 回退=hf-mirror"
# 装工具(modelscope 为主, huggingface_hub 作回退)
if ! python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >/tmp/dl_pip.log 2>&1; then
  echo "pip 安装失败(看 /tmp/dl_pip.log), 试 apt install python3-pip 后重跑"; tail -5 /tmp/dl_pip.log; exit 2
fi

# ---- 实时聚合速度监视器: 每15s 打印 本次已下GB + 当前MB/s ----
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

# ---- 下载封装: ModelScope 优先, 失败回退 HF ----
# 参数: $1=ModelScope仓库 $2=目标子目录 $3=HF回退仓库 [$4..=具体文件名(留空=整仓)]
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
  hy15)
    HY="hunyuanvideo-1.5"
    dl Tencent-Hunyuan/HunyuanVideo-1.5 "$HY"                             tencent/HunyuanVideo-1.5
    dl Qwen/Qwen2.5-VL-7B-Instruct      "$HY/text_encoder/llm"            Qwen/Qwen2.5-VL-7B-Instruct
    dl AI-ModelScope/byt5-small         "$HY/text_encoder/byt5-small"     google/byt5-small
    dl AI-ModelScope/Glyph-SDXL-v2      "$HY/text_encoder/Glyph-SDXL-v2"  AI-ModelScope/Glyph-SDXL-v2
    dl lightx2v/Hy1.5-Distill-Models    "$HY/distill_models/480p_t2v"     lightx2v/Hy1.5-Distill-Models \
       hy1.5_t2v_480p_lightx2v_4step.safetensors
    # 配置(hunyuan_video_t2v_480p_distill.json)加载 distill_model.safetensors, 加软链对齐(留原文件, 不破坏续传)
    ln -sf hy1.5_t2v_480p_lightx2v_4step.safetensors \
       "$DEST/$HY/distill_models/480p_t2v/distill_model.safetensors" 2>/dev/null || true
    ;;
  wan_i2v)
    dl lightx2v/Wan2.2-Distill-Models   "Wan2.2-Distill-Models"           lightx2v/Wan2.2-Distill-Models \
       wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors \
       wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors
    ;;
  wan_animate) dl Wan-AI/Wan2.2-Animate-14B "Wan2.2-Animate-14B" Wan-AI/Wan2.2-Animate-14B ;;
  wan_audio)   dl Wan-AI/Wan2.2-S2V-14B     "Wan2.2-S2V-14B"     Wan-AI/Wan2.2-S2V-14B ;;
  qwen_image)  dl Qwen/Qwen-Image           "Qwen-Image"         Qwen/Qwen-Image ;;
  z_image)     dl Tongyi-MAI/Z-Image-Turbo  "Z-Image-Turbo"      Tongyi-MAI/Z-Image-Turbo ;;
  # SeedVR2 视频超分: 走魔搭社区镜像 bytedance-community/*(快), HF官方 ByteDance-Seed/* 作回退
  seedvr_3b)   # 3B 整仓: DiT(seedvr2_ema_3b.pth)+ ema_vae.pth + pos/neg_emb.pt, 可直接跑
    dl bytedance-community/SeedVR2-3B "ByteDance-Seed/SeedVR2-3B" ByteDance-Seed/SeedVR2-3B ;;
  seedvr_7b)   # 7B 缺 emb, 取普通版+锐化版两个 DiT 对比; 跑时 model_path 指 3B 目录, config dit_original_ckpt 指其一
    dl bytedance-community/SeedVR2-7B "ByteDance-Seed/SeedVR2-7B" ByteDance-Seed/SeedVR2-7B seedvr2_ema_7b.pth seedvr2_ema_7b_sharp.pth ;;
  *) echo "!! 未知模型标签: $m (支持: hy15 wan_i2v wan_animate wan_audio qwen_image z_image seedvr_3b seedvr_7b)"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 全部完成 ===="
for m in $MODELS; do
  case "$m" in
    hy15)        d="hunyuanvideo-1.5" ;;
    wan_i2v)     d="Wan2.2-Distill-Models" ;;
    wan_animate) d="Wan2.2-Animate-14B" ;;
    wan_audio)   d="Wan2.2-S2V-14B" ;;
    qwen_image)  d="Qwen-Image" ;;
    z_image)     d="Z-Image-Turbo" ;;
    seedvr_3b)   d="ByteDance-Seed/SeedVR2-3B" ;;
    seedvr_7b)   d="ByteDance-Seed/SeedVR2-7B" ;;
    *)           continue ;;
  esac
  du -sh "$DEST/$d" 2>/dev/null || echo "  (缺) $DEST/$d"
done
if [ -n "$FAILED" ]; then echo "!! 以下有失败, 需重跑(会续传):$FAILED"; exit 1; fi
echo "完成, 全部成功。"
