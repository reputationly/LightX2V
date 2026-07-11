#!/usr/bin/env bash
# VACE 蒸馏LoRA 干净A/B: 写实参考图(woman.jpeg) + 写实提示词, 同seed
#   臂1 lora:   4步 LoRA 单卡(~5min)
#   臂2 base40: 40步 4卡对照(~10min, 复用 ul4 配置)
# 用法: tmux new -s vace_ab1 -d 'bash /data/smoke/vace_ab.sh lora'    (某节点)
#       tmux new -s vace_ab2 -d 'bash /data/smoke/vace_ab.sh base40'  (另一节点)
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
P="一位女子在海边漫步，长发随风飘动，阳光洒在海面上，实拍风格，自然光，电影感十足"
REF=/opt/LightX2V/assets/inputs/imgs/woman.jpeg

case "${1:?用法: vace_ab.sh lora|base40}" in
  lora)
    exec > >(tee -a /data/outputs/vace_ab_lora.log) 2>&1
    docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
      -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
      -v /data/smoke/vace_model.py:/opt/LightX2V/lightx2v/models/networks/wan/vace_model.py:ro \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
      python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8_lora4step.json \
      --prompt "$P" --src_ref_images "$REF" \
      --save_result_path /data/outputs/vace_ab_lora4.mp4 --seed 42
    ;;
  base40)
    exec > >(tee -a /data/outputs/vace_ab_base40.log) 2>&1
    docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
      -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
      torchrun --nproc_per_node=4 -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8_ul4.json \
      --prompt "$P" --src_ref_images "$REF" \
      --save_result_path /data/outputs/vace_ab_base40.mp4 --seed 42
    ;;
  *) echo "未知: $1"; exit 1 ;;
esac
