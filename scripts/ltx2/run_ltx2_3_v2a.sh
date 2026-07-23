#!/bin/bash

# LTX-2.3 纯配音(V2A):画面逐帧不变,只为输入视频生成配套音轨。
# 输出 mp4 = 原视频画面(-c:v copy 零损失)+ AI 生成音轨。
# prompt 里描述想要的声音:音效 / 环境音 / BGM(需显式点名) / 台词(引号内逐字)。
#
# 时长:默认对【整条】源视频配音(--target_video_length 对 v2a 无效)。
#   只想配前 N 帧时加 --reference_video_frame_cap N —— 注意 cap 之外的尾巴会是静音。
# 显存:整条视频一次性 VAE 编码 + 联合注意力,过长(>15s)可能 OOM,建议先用短片验证。
# 同步:模型按 24fps 训练,喂 ≈24fps 素材同步效果最佳(非 24fps 会打 warning 并按真实时长对齐)。

# set path and first
lightx2v_path=/path/to/LightX2V
model_path=Lightricks/LTX-2
VIDEO_PATH=          # 待配音的源视频(mp4)

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls ltx2 \
  --task v2a \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/ltx2/ltx2_3_v2a.json \
  --video_path "${VIDEO_PATH}" \
  --prompt "Footsteps on a wooden floor and quiet room ambience, matching the person's movements." \
  --negative_prompt "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_v2a.mp4" \
