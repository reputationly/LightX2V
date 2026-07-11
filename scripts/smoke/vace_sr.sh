#!/usr/bin/env bash
# VACE 生产配方终验: 480p 4步LoRA 产物 → SeedVR2 3B 超分(sr_ratio 2.0 → 960x1664)
# 用法: tmux new -s vace_sr -d 'bash /data/smoke/vace_sr.sh'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
SMOKE=/data/smoke
exec > >(tee -a /data/outputs/vace_sr.log) 2>&1
LX_IMG=$LX_IMG NAME=vace-sr MODEL_CLS=seedvr2 TASK=sr NP=1 GPUS=0 PORT=8100 STEPS=1 HEALTH_TO=1800 \
EXTRA_VOL="-v $SMOKE/seedvr_runner.py:/opt/LightX2V/lightx2v/models/runners/seedvr/seedvr_runner.py:ro" \
MODEL_PATH=/nfs-data/models/ByteDance-Seed/SeedVR2-3B \
CFG="$SMOKE/seedvr2_3b_seg121.json" \
VIDEO_PATH="/data/outputs/vace_ab_lora4.mp4" \
PROMPT="video super resolution" \
OUT="/data/outputs/vace_lora_sr.mp4" \
bash $SMOKE/test_model.sh
echo "########## [$(date +%T)] 完成: vace_lora_sr.mp4 ##########"
