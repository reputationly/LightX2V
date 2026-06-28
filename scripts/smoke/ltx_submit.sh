#!/usr/bin/env bash
# =============================================================================
# 向常驻 LTX server 提交一条生成(任意 prompt), 轮询+记录显存峰值+落盘。
# 提交体与 ltx_one.sh 完全一致(infer_steps=8, seed 可控), 画质与基线可比。
#
# 用法(服务器 edt-vpn 上, 需先 bash /data/ltx_serve.sh 起好 server):
#   bash /data/ltx_submit.sh hummingbird            # 用预设 prompt(默认121帧)
#   bash /data/ltx_submit.sh hummingbird 49         # 第2参=帧数
#   bash /data/ltx_submit.sh "你自己的一句提示词"      # 直接给文本
#   FRAMES=49 SEED=7 NAME=test1 bash /data/ltx_submit.sh horse_beach
# 预设: night_market | hummingbird | horse_beach | mountain_drone | coffee_rain
# 输出: /data/outputs/ltx_<NAME>.mp4 (NAME 默认取预设名或 'custom')
# =============================================================================
set -u
API=http://localhost:8000
DATA=/data; OUTDIR="$DATA/outputs"; mkdir -p "$OUTDIR"
ARG="${1:-night_market}"
FRAMES="${2:-${FRAMES:-121}}"   # 第2参 > 环境变量 FRAMES > 默认121
SEED="${SEED:-42}"

# 预设 prompt(运动重的几条最能看顺滑度)
case "$ARG" in
  night_market)  P="A bustling night market street with glowing neon signs, people walking past food stalls, reflections on wet pavement, cinematic, shallow depth of field"; NM=night_market ;;
  hummingbird)   P="A hummingbird hovering and rapidly beating its wings beside a bright red flower, extreme close-up, droplets of nectar in the air, cinematic, shallow depth of field"; NM=hummingbird ;;
  horse_beach)   P="A galloping horse running along a beach at sunset, spray of sand and water, dynamic side tracking shot, cinematic, motion blur"; NM=horse_beach ;;
  mountain_drone)P="Aerial drone shot flying fast over snowy mountain peaks at golden hour, sweeping camera movement, cinematic vista"; NM=mountain_drone ;;
  coffee_rain)   P="A cup of coffee on a wooden table by a window, rain streaming down the glass, steam rising, warm cozy light, cinematic"; NM=coffee_rain ;;
  *)             P="$ARG"; NM=custom ;;   # 非预设 -> 当作字面 prompt
esac
NAME="${NAME:-$NM}"
OUT="$OUTDIR/ltx_${NAME}.mp4"

# health 预检
code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health" 2>/dev/null || echo 000)
[ "$code" = "200" ] || { echo "server 未就绪(health=$code), 先 bash /data/ltx_serve.sh"; exit 1; }

echo "prompt[$NM]: $P"
echo "帧数=$FRAMES seed=$SEED -> $OUT"
rm -f "$OUT"
BODY=$(P="$P" OUT="$OUT" NF="$FRAMES" SD="$SEED" python3 -c "import json,os; print(json.dumps({'prompt':os.environ['P'],'negative_prompt':'','save_result_path':os.environ['OUT'],'target_video_length':int(os.environ['NF']),'infer_steps':8,'seed':int(os.environ['SD'])}))")
T0=$(date +%s)
RESP=$(curl -sS -m 30 -X POST "$API/v1/tasks/video/" -H "Content-Type: application/json" -d "$BODY")
TID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['task_id'])" 2>/dev/null)
[ -z "${TID:-}" ] && { echo "提交失败, 返回: $RESP"; exit 1; }
echo "submit tid=$TID"
PEAK=0; MISS=0; MAXWAIT="${MAXWAIT:-1800}"   # 最长等待秒数(server 挂/卡死的总兜底)
while true; do
  sleep 5
  EL=$(( $(date +%s)-T0 ))
  if [ "$EL" -gt "$MAXWAIT" ]; then echo "!! 超过 ${MAXWAIT}s 未完成, 放弃(server 可能挂了/卡死; 可设 MAXWAIT 调长)"; exit 1; fi
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 0)
  [ "$M" -gt "$PEAK" ] && PEAK=$M
  HC=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API/health" 2>/dev/null || echo 000)   # 探活
  ST=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
  echo "t=${EL}s status=${ST:-?} gpu=${M} peak=${PEAK} health=${HC}"
  case "$ST" in
    completed) break ;;
    failed) curl -sS "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys; e=json.load(sys.stdin).get('error') or ''; print('ERR:', e[:200])" 2>/dev/null; exit 1 ;;
    "") MISS=$((MISS+1))   # 取不到状态: 若同时 server 不健康且连续 3 次, 判定挂了, 不再死等
        if [ "$HC" != "200" ] && [ "$MISS" -ge 3 ]; then echo "!! server 不响应(health=$HC, 状态连续空 ${MISS} 次), 判定挂了"; exit 1; fi ;;
    *) MISS=0 ;;   # processing/pending 等正常状态, 重置计数, 继续等(受 MAXWAIT 兜底)
  esac
done
SZ=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
echo "DONE elapsed=$(( $(date +%s)-T0 ))s peak=${PEAK}MiB size=$SZ -> $OUT"
command -v ffprobe >/dev/null && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of default=noprint_wrappers=1 "$OUT" 2>/dev/null | sed 's/^/  /'
