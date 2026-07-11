#!/usr/bin/env bash
# VACE V2V 真控制信号实验: 原片→canny边缘视频→水墨重绘(独立文件, 不碰运行中的 vace_modes.sh)
# 用法: bash /data/smoke/vace_canny.sh prep   (秒级, 生成canny控制视频)
#       tmux new -s vace_cny -d 'bash /data/smoke/vace_canny.sh run'
set -u
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
DOCKER_BASE="docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -v /data/smoke/vace_processor.py:/opt/LightX2V/lightx2v/models/input_encoders/hf/vace/vace_processor.py:ro \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $LX_IMG"

case "${1:?用法: vace_canny.sh prep|run}" in
  prep)
    $DOCKER_BASE python - <<'PYEOF'
import av, numpy as np, cv2
c = av.open("/data/outputs/vace_r2v_int8.mp4"); s = c.streams.video[0]
frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]; c.close()
out = av.open("/data/vace_inputs/canny_src.mp4", "w")
st = out.add_stream("libx264", rate=16); st.height, st.width = frames[0].shape[:2]
st.pix_fmt = "yuv420p"; st.options = {"crf": "12"}
for f in frames:
    edge = cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 100, 200)
    rgb = np.stack([edge]*3, axis=-1)
    for pkt in st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")): out.mux(pkt)
for pkt in st.encode(): out.mux(pkt)
out.close(); print("写出 canny_src.mp4", len(frames), "帧")
PYEOF
    ;;
  run)
    exec > >(tee -a /data/outputs/vace_canny.log) 2>&1
    $DOCKER_BASE python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100_int8.json --seed 42 \
      --prompt "水墨画风格，一位女子在海边漫步，长发随风飘动，写意笔触，宣纸质感，留白意境" \
      --src_video /data/vace_inputs/canny_src.mp4 \
      --save_result_path /data/outputs/vace_canny_int8.mp4
    ;;
  *) echo "未知任务: $1"; exit 1 ;;
esac
