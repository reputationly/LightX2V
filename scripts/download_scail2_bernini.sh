#!/usr/bin/env bash
# =============================================================================
# 下载 SCAIL-2(智谱, Wan2.1-I2V-14B 骨架, 角色动画)+ Bernini(字节, Wan2.2 渲染器,
# 视频生成/编辑)到 NFS。规划见 docs/SCAIL2-Bernini-接入规划方案.md。
# 沿用 download_models.sh 的套路:ModelScope 主源(国内快)→ hf-mirror 回退,
# 实时测速 + 断点续传 + 失败追踪。只写新目录, 不碰已有模型。
#
# 模型(标签 | 约大小 | 主源→回退):
#   ---- 默认下(DiT 主体 + LightX2V 小 LoRA, 都是 NFS 上还没有的) ----
#   scail2          ~28-40G  zai-org/SCAIL-2(sat 格式 DiT, ⚠️下完需 convert.py 转 safetensors, 见文末)
#   scail2_lx2v     小       LightX2V Wan2.1-I2V 4步蒸馏 LoRA(加速核心, ⚠️很可能 NFS 已有→先查, 见文末)
#   bernini_r_14b   ~28G     ByteDance/Bernini-R-Diffusers(Wan2.2 渲染器, 先接这个)
#   bernini_r_13b   ~3G      ByteDance/Bernini-R-1.3B-Diffusers(轻量档, 40G 单卡快验用)
#   bernini_lx2v    小       rzgar/Bernini-R-LightX2V-4step-loras(high/low noise 两个, LightX2V 出品, 仅 HF)
#
#   ---- 可选(默认不下, 用 MODELS=... 显式触发) ----
#   scail2_dpo      小       Comfy-Org/SCAIL-2(DPO LoRA, 提质, 仅 *.safetensors)
#   bernini_full    大       ByteDance/Bernini-Diffusers(planner+renderer 全量, 二期再用)
#   qwen25vl        ~16G     Qwen/Qwen2.5-VL-7B-Instruct(full Bernini 的语义规划器;
#                            ⚠️ hy15 里已下过一份到 hunyuanvideo-1.5/text_encoder/llm, 大概率能复用→先查)
#
# ⚠️ 仓库 ID 说明:MS(ModelScope)id 多为最佳猜测;猜错会自动回退 hf-mirror(HF id 已核对)。
#    若已知准确 MS id, 改下面 case 里对应行的第 1 个参数即可。
#
# 用法(服务器上, 先 scp 到 /data 或 NFS):
#   tmux new -s dl_sb -d 'bash /data/download_scail2_bernini.sh'   # 挂后台(默认下 5 个)
#   tail -f /nfs-data/dl_scail2_bernini.log                        # 看速度+进度
#   MODELS="bernini_r_13b" bash download_scail2_bernini.sh         # 只下指定(建议先拿 1.3B 快验)
#   MODELS="qwen25vl bernini_full" bash download_scail2_bernini.sh # 二期 full Bernini
# 中断后重跑本脚本会自动续传。先跑文末的「NFS 已有核对」命令, 把结果给我再决定下哪些。
# =============================================================================
set -u
# DEST 自动适配:计算节点有 /nfs-data 软链;manager 节点用真身 /nfs-models/wuhanjisuan894/models
DEST="${DEST:-$([ -d /nfs-data/models ] && echo /nfs-data/models || echo /nfs-models/wuhanjisuan894/models)}"
LOG="${LOG:-$(dirname "$DEST")/dl_scail2_bernini.log}"
MODELS="${MODELS:-scail2 scail2_lx2v bernini_r_14b bernini_r_13b bernini_lx2v}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "==== [$(date +%T)] 目标=$DEST | 模型=$MODELS | 主源=ModelScope, 回退=hf-mirror"
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

# ---- 下载封装: ModelScope 优先, 失败回退 HF。$1=MS仓 $2=目标子目录 $3=HF回退仓 [$4..=文件/pattern(留空=整仓)] ----
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
  # ---- SCAIL-2(Wan2.1-I2V-14B 骨架) ----
  scail2)        dl ZhipuAI/SCAIL-2  "SCAIL-2"  zai-org/SCAIL-2 ;;
  scail2_dpo)    dl Comfy-Org/SCAIL-2 "loras/SCAIL-2-DPO" Comfy-Org/SCAIL-2 "*.safetensors" ;;
  # LightX2V Wan2.1-I2V 4步蒸馏 LoRA(加速核心, SCAIL-2 是 i2v 底座用 i2v 版)
  scail2_lx2v)   dl lightx2v/Wan2.1-Distill-Loras "loras/Wan2.1-Distill-Loras" lightx2v/Wan2.1-Distill-Loras \
                    wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors ;;

  # ---- Bernini(Wan2.2 渲染器) ----
  bernini_r_14b) dl ByteDance/Bernini-R-Diffusers      "Bernini-R-Diffusers"      ByteDance/Bernini-R-Diffusers ;;
  bernini_r_13b) dl ByteDance/Bernini-R-1.3B-Diffusers "Bernini-R-1.3B-Diffusers" ByteDance/Bernini-R-1.3B-Diffusers ;;
  # LightX2V 蒸馏 LoRA 对(high/low noise)—— 社区仅 HF, MS 大概率没有(会走回退)
  bernini_lx2v)  dl rzgar/Bernini-R-LightX2V-4step-loras "loras/Bernini-R-LightX2V-4step" rzgar/Bernini-R-LightX2V-4step-loras ;;
  # 二期: full Bernini(含 MLLM planner)
  bernini_full)  dl ByteDance/Bernini-Diffusers "Bernini-Diffusers" ByteDance/Bernini-Diffusers ;;
  qwen25vl)      dl Qwen/Qwen2.5-VL-7B-Instruct "Qwen2.5-VL-7B-Instruct" Qwen/Qwen2.5-VL-7B-Instruct ;;

  *) echo "!! 未知标签: $m (支持: scail2 scail2_lx2v scail2_dpo bernini_r_14b bernini_r_13b bernini_lx2v bernini_full qwen25vl)"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 全部完成 ===="
for m in $MODELS; do
  case "$m" in
    scail2)        d="SCAIL-2" ;;
    scail2_dpo)    d="loras/SCAIL-2-DPO" ;;
    scail2_lx2v)   d="loras/Wan2.1-Distill-Loras" ;;
    bernini_r_14b) d="Bernini-R-Diffusers" ;;
    bernini_r_13b) d="Bernini-R-1.3B-Diffusers" ;;
    bernini_lx2v)  d="loras/Bernini-R-LightX2V-4step" ;;
    bernini_full)  d="Bernini-Diffusers" ;;
    qwen25vl)      d="Qwen2.5-VL-7B-Instruct" ;;
    *) continue ;;
  esac
  du -sh "$DEST/$d" 2>/dev/null || echo "  (缺) $DEST/$d"
done
if [ -n "$FAILED" ]; then echo "!! 以下有失败, 需重跑(会续传):$FAILED"; exit 1; fi

echo
echo "==== SCAIL-2 后处理提醒 ===="
echo "SCAIL-2 权重是 sat 格式(model/1/fsdp2_rank_0000_checkpoint.pt), 接入 wan 分支前需转 safetensors:"
echo "  git clone https://github.com/zai-org/SCAIL-2 && cd SCAIL-2  # 用 wan-scail2 分支的 convert.py"
echo "  python convert.py --scail-dir $DEST/SCAIL-2 --save-path $DEST/SCAIL-2/SCAIL-2.safetensors  # Python 3.10-3.12"
echo "完成, 全部成功。"
