#!/usr/bin/env bash
# =============================================================================
# bench_bernini.sh — Bernini-R int8 提示词扫测,单节点 4 卡并发。
# 每张 GPU 串行跑分到它名下的一批提示词,4 卡之间并行(每卡同时只 1 个实例,防 OOM)。
# 输出与日志落 /nfs-output(共享),按 hostname 分目录,方便多节点汇总。
#
# 用法(在某个 GPU 节点上,或 238 上 `ssh -n root@<node> 'bash /root/bench_bernini.sh'`):
#   TASK=t2v bash bench_bernini.sh                         # 文生视频扫测(默认)
#   TASK=t2i bash bench_bernini.sh                         # 文生图(1 帧;用 t2i config)
#   GPUS="0 1" TASK=t2v bash bench_bernini.sh              # 只用 2 卡
#   RES=720 TASK=t2v bash bench_bernini.sh                 # 720p(改 config 里分辨率, 见下)
# 环境变量可覆盖 IMG/MODEL_PATH/CONFIG/PROMPTS/OUT。
# 汇总:tail 各 .log 的 "RUN pipeline cost";视频在 $OUT/*.mp4。
# =============================================================================
set -u
TASK="${TASK:-t2v}"
GPUS="${GPUS:-0 1 2 3}"
IMG="${IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
MODEL_PATH="${MODEL_PATH:-/nfs-models/wuhanjisuan894/models/Wan2.1-I2V-14B-480P}"
CONFIG="${CONFIG:-/nfs-models/_transfer/bernini_r_14b_${TASK}_int8.json}"
PROMPTS="${PROMPTS:-/nfs-models/_transfer/bernini_prompts.txt}"
NEG="${NEG:-blurry, low quality, distorted, extra fingers, overexposed, static, watermark}"
# LightX2V 的 task:图像用 t2v 路径出 1 帧(config 里 target_video_length=1),故 model task 恒 t2v
LX_TASK="t2v"
OUT="${OUT:-/nfs-output/bernini_bench/${TASK}/$(hostname)}"
mkdir -p "$OUT"

[ -f "$PROMPTS" ] || { echo "✗ 提示词文件不存在: $PROMPTS(先 scp bernini_prompts.txt 到 _transfer)"; exit 2; }
[ -f "$CONFIG" ]  || { echo "✗ config 不存在: $CONFIG(先 scp bernini_r_14b_${TASK}_int8.json 到 _transfer)"; exit 2; }

mapfile -t P < <(grep -vE '^[[:space:]]*(#|$)' "$PROMPTS")
gpus=($GPUS); ng=${#gpus[@]}
echo "==== [$(date +%T)] TASK=$TASK | ${#P[@]} 条提示词 | GPU: ${GPUS} | 输出 $OUT"

run_one() { # $1=idx $2=gpu
  local idx=$1 g=$2 prompt="${P[$idx]}" id; id=$(printf '%02d' "$idx")
  local out="$OUT/${id}.mp4" log="$OUT/${id}.log"
  docker run --rm --runtime nvidia --gpus "\"device=${g}\"" --memory 60g \
    -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output -e PYTHONPATH=/opt/LightX2V "$IMG" \
    python -m lightx2v.infer --model_cls wan2.2_moe_distill --task "$LX_TASK" \
    --model_path "$MODEL_PATH" --config_json "$CONFIG" \
    --prompt "$prompt" --negative_prompt "$NEG" \
    --save_result_path "$out" > "$log" 2>&1
  local cost; cost=$(grep -oE 'RUN pipeline cost [0-9.]+ seconds' "$log" | tail -1)
  if [ -f "$out" ]; then echo "  ✅ [$id gpu$g] ${cost:-?} -> ${id}.mp4"
  else echo "  ❌ [$id gpu$g] 失败(tail: $(tail -n1 "$log"))"; fi
}

# 轮转分组:第 i 条 -> gpu[i % ng];每卡一个队列串行,队列间并行
declare -A Q
for i in "${!P[@]}"; do Q[$((i % ng))]+="$i "; done
for k in "${!gpus[@]}"; do
  ( for idx in ${Q[$k]}; do run_one "$idx" "${gpus[$k]}"; done ) &
done
wait

echo "==== [$(date +%T)] 完成。耗时汇总:"
grep -h -oE 'RUN pipeline cost [0-9.]+ seconds' "$OUT"/*.log 2>/dev/null | sort | uniq -c
echo "视频: ls $OUT/*.mp4 ; 拉回本地看画质"
