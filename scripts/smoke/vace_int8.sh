#!/usr/bin/env bash
# VACE int8-triton 首跑(R2V 纯参考图 smoke)
# 用法: tmux new -s vace2 -d 'bash /data/smoke/vace_int8.sh'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
exec > >(tee -a /data/outputs/vace_int8.log) 2>&1
docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
  python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
  --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
  --config_json /data/smoke/wan22_moe_vace_a100_int8.json \
  --prompt "一位女子在海边漫步，长发随风飘动，阳光洒在海面上，电影感十足" \
  --src_ref_images /opt/LightX2V/assets/inputs/imgs/girl.png \
  --save_result_path /data/outputs/vace_r2v_int8.mp4 --seed 42
