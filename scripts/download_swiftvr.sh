#!/usr/bin/env bash
# =============================================================================
# 下载 SwiftVR(H-oliday, 实时一步式生成式视频修复/超分, 骨架=Wan2.2-TI2V-5B)到 NFS。
# 上游 LightX2V 已原生支持(PR #1400/#1406/#1409/#1421/#1427/#1438), 有 a800 配置,
# 但需要先把官方 diffusers 权重转成 LightX2V 的 wan_dit key 布局(见文末转换提醒)。
# 沿用 download_scail2_bernini.sh 的套路:ModelScope 主源(国内快)→ hf-mirror 回退,
# 实时测速 + 断点续传 + 失败追踪。只写新目录, 不碰已有模型。
#
# 模型(标签 | 约大小 | 主源→回退):
#   swiftvr        ~20.2G  H-oliday/SwiftVR —— 只取推理必需 4 个文件:
#                          transformer/diffusion_pytorch_model.safetensors  ~19.9G (DiT)
#                          transformer/config.json                            495B
#                          reae.safetensors                                 ~156M (Restoration-aware AE)
#                          prompt_embedding.safetensors                     ~4.0M (空提示词固定 embedding)
#   swiftvr_assets ~175M   官方 demo 视频/图(assets/), 默认不下, 要看效果对比再下
#
# ⚠️ ModelScope CLI 不吃 glob, 所以这里全部写死文件名(踩过坑, 见 minimax-h3 记录)。
#
# 用法(服务器上, 先 scp 到 /data 或 NFS):
#   tmux new -s dl_swiftvr -d 'bash /data/download_swiftvr.sh'   # 挂后台
#   tail -f /nfs-data/dl_swiftvr.log                             # 看速度+进度
#   MODELS="swiftvr_assets" bash download_swiftvr.sh             # 只下 demo 素材
# 中断后重跑本脚本会自动续传。
# =============================================================================
set -u
# DEST 自动适配:计算节点有 /nfs-data 软链;manager 节点用真身 /nfs-models/wuhanjisuan894/models
DEST="${DEST:-$([ -d /nfs-data/models ] && echo /nfs-data/models || echo /nfs-models/wuhanjisuan894/models)}"
LOG="${LOG:-$(dirname "$DEST")/dl_swiftvr.log}"
MODELS="${MODELS:-swiftvr}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "==== [$(date +%T)] 目标=$DEST | 模型=$MODELS | 主源=ModelScope, 回退=hf-mirror"
# ---- 依赖:先探测, 已装就别碰 pip ----
# 踩坑(2026-08-27, manager 节点):直连 PyPI 不通, `pip install -U` 会在 do_poll 上无限干等,
# 日志一行不动看着像死了;而 modelscope/huggingface_hub 其实早就装好了。所以:
#   1) 能 import 就直接跳过;2) 真要装才走国内源, 并且给死超时, 装不上就报错而不是挂着。
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
if python3 -c "import modelscope, huggingface_hub" >/dev/null 2>&1; then
  echo "  依赖已就绪: $(python3 -c 'import modelscope,huggingface_hub;print("modelscope",modelscope.__version__,"| huggingface_hub",huggingface_hub.__version__)')(跳过 pip)"
else
  echo "  缺依赖, 安装中(源=$PIP_INDEX, 超时 300s)..."
  if ! timeout 300 python3 -m pip install -q -i "$PIP_INDEX" --timeout 20 --retries 2 \
        modelscope "huggingface_hub[cli]" >/tmp/dl_pip.log 2>&1; then
    echo "  !! 依赖安装失败/超时(看 /tmp/dl_pip.log)。换源重试: PIP_INDEX=https://mirrors.aliyun.com/pypi/simple bash $0"
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
  # ---- SwiftVR 官方 diffusers 权重(推理必需文件, 不含 demo 素材) ----
  swiftvr)        dl H-oliday/SwiftVR "SwiftVR" H-oliday/SwiftVR \
                     transformer/config.json \
                     transformer/diffusion_pytorch_model.safetensors \
                     reae.safetensors \
                     prompt_embedding.safetensors \
                     configuration.json \
                     README.md ;;
  # 官方 demo 视频/对比图, 只在要做主观评测参照时下
  swiftvr_assets) dl H-oliday/SwiftVR "SwiftVR" H-oliday/SwiftVR \
                     assets/demo_1.mp4 assets/demo_2.mp4 assets/demo_3.mp4 \
                     assets/qualitative.png assets/teaser.avif ;;

  *) echo "!! 未知标签: $m (支持: swiftvr swiftvr_assets)"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 全部完成 ===="
du -sh "$DEST/SwiftVR" 2>/dev/null || echo "  (缺) $DEST/SwiftVR"
ls -l "$DEST/SwiftVR/transformer/diffusion_pytorch_model.safetensors" 2>/dev/null || echo "  (缺) DiT 主权重"
if [ -n "$FAILED" ]; then echo "!! 以下有失败, 需重跑(会续传):$FAILED"; exit 1; fi

echo
echo "==== SwiftVR 后处理提醒(必做) ===="
echo "官方权重是 diffusers key 布局, LightX2V 要 wan_dit 布局, 跑推理前先转:"
echo "  python tools/convert/examples/convert_swiftvr.py \\"
echo "      --source $DEST/SwiftVR --output $DEST/SwiftVR_lightx2v"
echo "  # 转换脚本在上游 PR #1400 起引入, 我们 fork 还没合, 需先同步上游代码"
echo "  # --output 目录必须不存在;转完会自动校验 tensor 数量/shape/dtype 与源一致"
echo "然后用 configs/swiftvr/a800/swiftvr.json 起推理(A100 用 a800 档)。"
echo "⚠️ 该配置写的是 rope_type=flashinfer_rope, 我们 ARM 镜像没装 flashinfer,"
echo "   实测 Z-Image 时踩过同样的坑, 大概率要改成 rope_type=torch。"
echo "完成, 全部成功。"
