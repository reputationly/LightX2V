#!/usr/bin/env bash
# 画质调查第二批(独立文件, 避免覆盖运行中的 run_batch.sh)
# 用法: tmux new -s tcam -d 'bash /data/smoke/run_batch2.sh tcamera'
set -u
export LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
SMOKE=/data/smoke
OUTD=/data/outputs

case "${1:?用法: run_batch2.sh tcamera}" in
  tcamera)  # 运镜真实感 A/B: 同seed同摄影级提示词, 固定机位 vs 前进运镜(t2v 单卡 triton)
    BASE_P="实拍风格，一条瀑布从青灰色岩壁倾泻而下，水雾弥漫，阳光在水汽中形成光斑，岩石表面湿润反光，苔藓细节清晰，自然光，纪录片质感，35mm胶片，真实物理水流"
    for arm in static motion; do
      if [ "$arm" = "static" ]; then
        P="$BASE_P，固定机位，三脚架拍摄，画面稳定无移动"
      else
        P="$BASE_P，镜头缓慢向前推进，前进运镜"
      fi
      exec > >(tee -a "$OUTD/tcam720_${arm}.log") 2>&1
      NAME="tcam-$arm" MODEL_CLS=wan2.2_moe_distill TASK=t2v NP=4 GPUS="0,1,2,3" PORT=8100 STEPS=4 SEED=42 HEALTH_TO=1800 \
      MODEL_PATH=/nfs-data/models/Wan-AI/Wan2.2-T2V-A14B \
      CFG="$SMOKE/w1c_t2v_triton.json" \
      PROMPT="$P" \
      OUT="$OUTD/tcam720_${arm}.mp4" \
      bash $SMOKE/test_model.sh
      sleep 15
    done
    ;;
  *) echo "未知任务: $1"; exit 1 ;;
esac
