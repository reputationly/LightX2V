#!/usr/bin/env bash
# VACE LoRA vs 40步 扩展A/B: 三对新题材(夜景人物/动物动作/无脸场景), 同seed同提示词同参考图
# 用法: bash /data/smoke/vace_ab2.sh <b|c|d> <lora|base>
#   b=夜景男子仙女棒(img_1)  c=墨镜猫冲浪(img_0)  d=烹饪手部(img_2)
# 部署建议(节点并行):
#   0015: tmux new -s ab_l -d 'for p in b c d; do bash /data/smoke/vace_ab2.sh $p lora; sleep 10; done'
#   0010: tmux new -s ab_b1 -d 'bash /data/smoke/vace_ab2.sh b base'
#   0013: tmux new -s ab_b2 -d 'bash /data/smoke/vace_ab2.sh c base'
#   0014: tmux new -s ab_b3 -d 'bash /data/smoke/vace_ab2.sh d base'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
PAIR="${1:?用法: vace_ab2.sh <b|c|d> <lora|base>}"
ARM="${2:?用法: vace_ab2.sh <b|c|d> <lora|base>}"

case "$PAIR" in
  b) REF=/opt/LightX2V/assets/inputs/imgs/img_1.jpg
     P="一位年轻男子在夜晚的建筑前挥舞仙女棒烟花，火花闪烁，夜景实拍风格，自然光影" ;;
  c) REF=/opt/LightX2V/assets/inputs/imgs/img_0.jpg
     P="一只戴墨镜的白猫站在冲浪板上冲浪，海浪飞溅，阳光明媚，实拍风格" ;;
  d) REF=/opt/LightX2V/assets/inputs/imgs/img_2.jpg
     P="厨房里一双手正在用木铲翻炒锅中的蔬菜，蒸汽升腾，窗外阳光洒入，实拍风格" ;;
  *) echo "未知pair: $PAIR"; exit 1 ;;
esac

exec > >(tee -a "/data/outputs/vace_ab_${PAIR}_${ARM}.log") 2>&1
if [ "$ARM" = "lora" ]; then
  docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
    -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
    -v /data/smoke/vace_model.py:/opt/LightX2V/lightx2v/models/networks/wan/vace_model.py:ro \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
    python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
    --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
    --config_json /data/smoke/wan22_moe_vace_a100_int8_lora4step.json \
    --prompt "$P" --src_ref_images "$REF" \
    --save_result_path "/data/outputs/vace_ab_${PAIR}_lora.mp4" --seed 42
else
  docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
    torchrun --nproc_per_node=4 -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
    --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
    --config_json /data/smoke/wan22_moe_vace_a100_int8_ul4.json \
    --prompt "$P" --src_ref_images "$REF" \
    --save_result_path "/data/outputs/vace_ab_${PAIR}_base.mp4" --seed 42
fi
