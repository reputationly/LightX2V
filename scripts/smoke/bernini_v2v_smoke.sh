#!/usr/bin/env bash
# Bernini-R v2v(in-context 视频编辑)冒烟 —— 补丁挂载方式跑未发布代码。
# 验证目标(照 VACE v2v 判据):结构保留 + 提示词重绘("变成了水墨风格, 内容跟原始的差不多")。
#
# 前置(Mac → 节点):把 7 个改动 py + v2v config 放到 /nfs-models/_transfer/v2v_patch/,保持仓库相对路径:
#   rsync -avR lightx2v/infer.py \
#     lightx2v/models/networks/wan/infer/{module_io,pre_infer}.py \
#     lightx2v/models/networks/wan/model.py \
#     lightx2v/models/runners/default_runner.py \
#     lightx2v/models/runners/wan/wan_runner.py \
#     lightx2v/utils/input_info.py \
#     configs/wan22_bernini/bernini_r_14b_v2v_int8.json \
#     configs/wan22_bernini/bernini_r_14b_v2v_ulysses4_int8.json \
#     root@<238>:/nfs-models/_transfer/v2v_patch/
#
# 用法(节点):
#   bash bernini_v2v_smoke.sh                                    # 单卡 480p(先验正确性)
#   NP=4 bash bernini_v2v_smoke.sh                               # 4卡 ulysses(同seed对比单卡,验SP不错位)
#   SRC=/nfs-output/xxx.mp4 PROMPT="..." bash bernini_v2v_smoke.sh
set -u
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
NP="${NP:-1}"
MODEL_PATH=/nfs-models/wuhanjisuan894/models/Wan2.1-I2V-14B-480P
P=/nfs-models/_transfer/v2v_patch
# 源片默认拿一条已验证的 t2v 480p 出片(结构清晰、有主体运动,适合验编辑)
SRC="${SRC:-/nfs-output/bernini_bench/t2v/dev-gpustack-a100-0021/p01.mp4}"
PROMPT="${PROMPT:-Convert the video into a traditional Chinese ink-wash painting style, monochrome, flowing brush strokes, rice paper texture}"
NEG="${NEG:-blurry, low quality, distorted}"
SEED="${SEED:-42}"
OUT="${OUT:-/nfs-output/bernini_v2v_smoke_np${NP}_s${SEED}.mp4}"

if [ "$NP" -gt 1 ]; then
  CFG="$P/configs/wan22_bernini/bernini_r_14b_v2v_ulysses4_int8.json"
  RUNCMD="torchrun --nproc_per_node=$NP -m lightx2v.infer"
  SHM="--shm-size 16g"
else
  CFG="$P/configs/wan22_bernini/bernini_r_14b_v2v_int8.json"
  RUNCMD="python -m lightx2v.infer"
  SHM=""
fi

# shellcheck disable=SC2086
docker run --rm --runtime nvidia --gpus all --memory 200g $SHM \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
  -v "$P/lightx2v/infer.py":/opt/LightX2V/lightx2v/infer.py:ro \
  -v "$P/lightx2v/models/networks/wan/infer/module_io.py":/opt/LightX2V/lightx2v/models/networks/wan/infer/module_io.py:ro \
  -v "$P/lightx2v/models/networks/wan/infer/pre_infer.py":/opt/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py:ro \
  -v "$P/lightx2v/models/networks/wan/model.py":/opt/LightX2V/lightx2v/models/networks/wan/model.py:ro \
  -v "$P/lightx2v/models/runners/default_runner.py":/opt/LightX2V/lightx2v/models/runners/default_runner.py:ro \
  -v "$P/lightx2v/models/runners/wan/wan_runner.py":/opt/LightX2V/lightx2v/models/runners/wan/wan_runner.py:ro \
  -v "$P/lightx2v/utils/input_info.py":/opt/LightX2V/lightx2v/utils/input_info.py:ro \
  -e PYTHONPATH=/opt/LightX2V \
  "$IMG" $RUNCMD \
    --model_cls wan2.2_moe \
    --task v2v \
    --model_path "$MODEL_PATH" \
    --config_json "$CFG" \
    --src_video "$SRC" \
    --prompt "$PROMPT" \
    --negative_prompt "$NEG" \
    --seed "$SEED" \
    --save_result_path "$OUT"

echo "==== done: $OUT ===="
ls -lh "$OUT" 2>/dev/null || echo "(没出片,看上面报错)"
