#!/usr/bin/env bash
# VACE 收官: LoRA 编辑能力确认(720p 4卡case已删——三审判死, 单卡720p见 vace_720p2.sh)
#   cny_lora: 4步LoRA + canny控制重绘(验编辑能力在蒸馏下保全, 480p单卡 ~3min)
# 用法: tmux new -s vace_f1 -d 'bash /data/smoke/vace_final.sh cny_lora'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
PATCH="-v /data/smoke/vace_model.py:/opt/LightX2V/lightx2v/models/networks/wan/vace_model.py:ro \
  -v /data/smoke/vace_processor.py:/opt/LightX2V/lightx2v/models/input_encoders/hf/vace/vace_processor.py:ro"

case "${1:?用法: vace_final.sh cny_lora}" in
  cny_lora)  # 与40步canny同输入同seed同提示词, 唯一变量=4步LoRA
    exec > >(tee -a /data/outputs/vace_cny_lora.log) 2>&1
    docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
      -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V $PATCH \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
      python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8_lora4step.json \
      --prompt "水墨画风格，一位女子在海边漫步，长发随风飘动，写意笔触，宣纸质感，留白意境" \
      --src_video /data/vace_inputs/canny_src.mp4 \
      --save_result_path /data/outputs/vace_cny_lora.mp4 --seed 42
    ;;
  *) echo "未知任务: $1"; exit 1 ;;
esac
