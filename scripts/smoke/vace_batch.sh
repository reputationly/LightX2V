#!/usr/bin/env bash
# VACE 第二批: 4卡提速 / V2V 控制视频模式(独立新文件, 不覆盖 vace_int8.sh)
# 用法: tmux new -s vace3 -d 'bash /data/smoke/vace_batch.sh ul4'
#       tmux new -s vace4 -d 'bash /data/smoke/vace_batch.sh v2v'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
case "${1:?用法: vace_batch.sh ul4|v2v}" in

  ul4)  # 4卡 ulysses int8, 与 vace_r2v_int8 同 seed/prompt/参考图 —— 测多卡提速比(禁offload/禁load_from_rank0)
    exec > >(tee -a /data/outputs/vace_ul4.log) 2>&1
    docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
      -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
      torchrun --nproc_per_node=4 -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8_ul4.json \
      --prompt "一位女子在海边漫步，长发随风飘动，阳光洒在海面上，电影感十足" \
      --src_ref_images /opt/LightX2V/assets/inputs/imgs/girl.png \
      --save_result_path /data/outputs/vace_r2v_ul4.mp4 --seed 42
    ;;

  v2v)  # V2V 控制视频模式: 拿 R2V 产物当 src_video, 提示词换风格 —— 验证视频重绘链路
    exec > >(tee -a /data/outputs/vace_v2v.log) 2>&1
    docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
      -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
      python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8.json \
      --prompt "水墨画风格，一位女子在海边漫步，长发随风飘动，写意笔触，宣纸质感" \
      --src_video /data/outputs/vace_r2v_int8.mp4 \
      --save_result_path /data/outputs/vace_v2v_int8.mp4 --seed 42
    ;;

  *) echo "未知任务: $1"; exit 1 ;;
esac
