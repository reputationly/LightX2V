#!/usr/bin/env bash
# VACE V2V 重跑: 带 decord→PyAV 兜底补丁(vace_processor.py 挂载覆盖)
# 用法: tmux new -s vace4 -d 'bash /data/smoke/vace_v2v2.sh'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
exec > >(tee -a /data/outputs/vace_v2v.log) 2>&1
docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -v /data/smoke/vace_processor.py:/opt/LightX2V/lightx2v/models/input_encoders/hf/vace/vace_processor.py:ro \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
  python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
  --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
  --config_json /data/smoke/wan22_moe_vace_a100_int8.json \
  --prompt "水墨画风格，一位女子在海边漫步，长发随风飘动，写意笔触，宣纸质感" \
  --src_video /data/outputs/vace_r2v_int8.mp4 \
  --save_result_path /data/outputs/vace_v2v_int8.mp4 --seed 42
