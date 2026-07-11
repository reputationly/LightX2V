#!/usr/bin/env bash
# VACE 剩余模式测试: t2v / inpaint / outpaint / extend(单卡 int8, 各~21min)
# 先构造输入(任一节点跑一次即可, 秒级):
#   bash /data/smoke/vace_modes.sh prep
# 然后四个case分节点发车:
#   tmux new -s vace_t2v -d 'bash /data/smoke/vace_modes.sh t2v'
#   tmux new -s vace_inp -d 'bash /data/smoke/vace_modes.sh inpaint'
#   tmux new -s vace_out -d 'bash /data/smoke/vace_modes.sh outpaint'
#   tmux new -s vace_ext -d 'bash /data/smoke/vace_modes.sh extend'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
DOCKER_BASE="docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -v /data/smoke/vace_processor.py:/opt/LightX2V/lightx2v/models/input_encoders/hf/vace/vace_processor.py:ro \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $LX_IMG"
INFER="python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
  --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
  --config_json /data/smoke/wan22_moe_vace_a100_int8.json --seed 42"

case "${1:?用法: vace_modes.sh prep|t2v|inpaint|outpaint|extend}" in
  prep)
    $DOCKER_BASE python /data/smoke/vace_prep_inputs.py
    ;;
  t2v)  # 全空输入: 验纯文生视频路径
    exec > >(tee -a /data/outputs/vace_t2v.log) 2>&1
    $DOCKER_BASE $INFER \
      --prompt "一只橘猫在洒满阳光的草地上奔跑，毛发细节清晰，实拍风格，自然光" \
      --save_result_path /data/outputs/vace_t2v_int8.mp4
    ;;
  inpaint)  # 局部重绘: 中央区域重画
    exec > >(tee -a /data/outputs/vace_inpaint.log) 2>&1
    $DOCKER_BASE $INFER \
      --prompt "一位女子在海边漫步，她穿着鲜红色的连衣裙，长发随风飘动，阳光洒在海面上" \
      --src_video /data/vace_inputs/inpaint_src.mp4 \
      --src_mask /data/vace_inputs/inpaint_mask.mp4 \
      --save_result_path /data/outputs/vace_inpaint_int8.mp4
    ;;
  outpaint)  # 扩画布: 60%画面向四周扩展
    exec > >(tee -a /data/outputs/vace_outpaint.log) 2>&1
    $DOCKER_BASE $INFER \
      --prompt "广阔的海滩全景，一位女子在海边漫步，远处是蔚蓝大海和天空，海鸥飞翔，阳光明媚" \
      --src_video /data/vace_inputs/outpaint_src.mp4 \
      --src_mask /data/vace_inputs/outpaint_mask.mp4 \
      --save_result_path /data/outputs/vace_outpaint_int8.mp4
    ;;
  extend)  # 首帧续写: 从首帧生成后续80帧
    exec > >(tee -a /data/outputs/vace_extend.log) 2>&1
    $DOCKER_BASE $INFER \
      --prompt "一位女子在海边漫步，镜头缓慢向前推进，长发随风飘动，阳光洒在海面上，电影感十足" \
      --src_video /data/vace_inputs/extend_src.mp4 \
      --src_mask /data/vace_inputs/extend_mask.mp4 \
      --save_result_path /data/outputs/vace_extend_int8.mp4
    ;;
  *) echo "未知任务: $1"; exit 1 ;;
esac
