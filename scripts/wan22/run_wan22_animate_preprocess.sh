#!/bin/bash

# set path firstly
lightx2v_path=
model_path=
video_path=
refer_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# animate preprocess without trim trailing blank
python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
    --ckpt_path ${model_path}/process_checkpoint \
    --video_path $video_path  \
    --refer_path $refer_path \
    --save_path ${lightx2v_path}/save_results/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag

# # animate preprocess with trim trailing blank
# python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
#     --ckpt_path ${model_path}/process_checkpoint \
#     --video_path $video_path  \
#     --refer_path $refer_path \
#     --save_path ${lightx2v_path}/save_results/animate/process_results \
#     --resolution_area 1280 720 \
#     --retarget_flag \
#     --trim_trailing_blank

# # replace preprocess without trim trailing blank
# python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
#     --ckpt_path ${model_path}/process_checkpoint \
#     --video_path $video_path  \
#     --refer_path $refer_path \
#     --save_path ${lightx2v_path}/save_results/replace/process_results \
#     --resolution_area 1280 720 \
#     --iterations 3 \
#     --k 7 \
#     --w_len 1 \
#     --h_len 1 \
#     --replace_flag

# # replace preprocess with trim trailing blank
# python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
#     --ckpt_path ${model_path}/process_checkpoint \
#     --video_path $video_path  \
#     --refer_path $refer_path \
#     --save_path ${lightx2v_path}/save_results/replace/process_results \
#     --resolution_area 1280 720 \
#     --iterations 3 \
#     --k 7 \
#     --w_len 1 \
#     --h_len 1 \
#     --replace_flag \
#     --trim_trailing_blank
