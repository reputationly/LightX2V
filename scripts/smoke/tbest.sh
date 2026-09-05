#!/usr/bin/env bash
# t2v 真实感天花板组合: 摄影级提示词+静态机位 → 720p 4卡 triton → SeedVR2 3B 收尾
# 用法: tmux new -s tbest -d 'bash /data/smoke/tbest.sh'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
SMOKE=/data/smoke
OUTD=/data/outputs
P="实拍风格，一条瀑布从青灰色岩壁倾泻而下，水雾弥漫，阳光在水汽中形成光斑，岩石表面湿润反光，苔藓细节清晰，自然光，纪录片质感，35mm胶片，真实物理水流，固定机位，三脚架拍摄，画面稳定无移动"
exec > >(tee -a "$OUTD/tbest.log") 2>&1

echo "########## [$(date +%T)] 第一步: t2v 720p 4卡 triton ##########"
LX_IMG=$LX_IMG NAME=tbest-gen MODEL_CLS=wan2.2_moe TASK=t2v NP=4 GPUS="0,1,2,3" PORT=8100 STEPS=4 SEED=42 HEALTH_TO=1800 \
MODEL_PATH=/nfs-data/models/Wan-AI/Wan2.2-T2V-A14B \
CFG="$SMOKE/w1c_t2v_triton.json" \
PROMPT="$P" \
OUT="$OUTD/tbest_720p.mp4" \
bash $SMOKE/test_model.sh

sleep 15
echo "########## [$(date +%T)] 第二步: SeedVR2 3B 超分收尾 ##########"
LX_IMG=$LX_IMG NAME=tbest-sr MODEL_CLS=seedvr2 TASK=sr NP=1 GPUS=0 PORT=8100 STEPS=1 HEALTH_TO=1800 \
EXTRA_VOL="-v $SMOKE/seedvr_runner.py:/opt/LightX2V/lightx2v/models/runners/seedvr/seedvr_runner.py:ro" \
MODEL_PATH=/nfs-data/models/ByteDance-Seed/SeedVR2-3B \
CFG="$SMOKE/seedvr2_3b_seg121.json" \
VIDEO_PATH="$OUTD/tbest_720p.mp4" \
PROMPT="video super resolution" \
OUT="$OUTD/tbest_1080p.mp4" \
bash $SMOKE/test_model.sh
echo "########## [$(date +%T)] 完成: tbest_720p.mp4(原) / tbest_1080p.mp4(终) ##########"
