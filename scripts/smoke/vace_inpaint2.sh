#!/usr/bin/env bash
# VACE inpaint 修正实验: mask区域灰底填充(官方预处理做法), 提示词改黄裙(反差明显)
# 用法: bash /data/smoke/vace_inpaint2.sh prep
#       tmux new -s vace_inp2 -d 'bash /data/smoke/vace_inpaint2.sh run'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
DOCKER_BASE="docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -v /data/smoke/vace_processor.py:/opt/LightX2V/lightx2v/models/input_encoders/hf/vace/vace_processor.py:ro \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $LX_IMG"

case "${1:?用法: vace_inpaint2.sh prep|run}" in
  prep)
    $DOCKER_BASE python /data/smoke/vace_prep_inpaint2.py
    ;;
  run)
    exec > >(tee -a /data/outputs/vace_inpaint2.log) 2>&1
    $DOCKER_BASE python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8.json --seed 42 \
      --prompt "一位女子在海边漫步，她穿着一条鲜黄色的连衣裙，长发随风飘动，阳光洒在海面上" \
      --src_video /data/vace_inputs/inpaint2_src.mp4 \
      --src_mask /data/vace_inputs/inpaint_mask.mp4 \
      --save_result_path /data/outputs/vace_inpaint2_int8.mp4
    ;;
  *) echo "未知任务: $1"; exit 1 ;;
esac
