#!/usr/bin/env bash
# Bernini-R 720p / 4 卡 ulysses 单条测试(单容器 torchrun 4 进程,全 4 卡)。
# 用法(节点上):tmux new -s u4 -d 'bash /nfs-models/_transfer/run_720p_ulysses4.sh'; tail -f /nfs-output/u4.log
# 若 NCCL 卡死:把下面 NCCL_* 两行的注释去掉(强制 host 中转,慢但稳)。
set -u
IMG="${IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
MODEL_PATH="/nfs-models/wuhanjisuan894/models/Wan2.1-I2V-14B-480P"
CONFIG="/nfs-models/_transfer/bernini_r_14b_t2v_720p_ulysses4_int8.json"
OUT="/nfs-output/bernini_720p_ulysses4.mp4"
PROMPT="A cinematic shot of a golden retriever running through a field of sunflowers at sunset, warm rim light, 35mm film"
NEG="blurry, low quality, distorted, overexposed"

docker run --rm --runtime nvidia --gpus all --memory 200g --shm-size 16g \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
  -e PYTHONPATH=/opt/LightX2V \
  `# -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1   # NCCL 卡死时去注释` \
  "$IMG" \
  torchrun --nproc_per_node=4 -m lightx2v.infer \
    --model_cls wan2.2_moe \
    --task t2v \
    --model_path "$MODEL_PATH" \
    --config_json "$CONFIG" \
    --prompt "$PROMPT" \
    --negative_prompt "$NEG" \
    --save_result_path "$OUT"

echo "==== done: $OUT ===="
ls -lh "$OUT" 2>/dev/null || echo "(没出片,看上面报错)"
