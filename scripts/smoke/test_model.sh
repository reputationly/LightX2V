#!/usr/bin/env bash
# =============================================================================
# 通用模型测试 harness —— 起容器 + 提交(官方图/音频/提示词)+ 记录耗时/显存峰值/规格
# 用于 int8 vs bf16 对比测试, 各模型复用。
#
# 必填环境变量:
#   NAME       容器名(如 wan-i2v-int8)
#   MODEL_CLS  --model_cls(wan2.2_moe / wan2.2_animate / qwen_image / z_image / ltx2 / hunyuan_video ...)
#   TASK       --task(t2v / i2v / s2v / t2i / animate ...)
#   MODEL_PATH --model_path(基座目录, 提供 VAE/T5 等)
#   CFG        --config_json(配置文件)
#   PROMPT     提示词
#   OUT        产物保存路径(/data/outputs/xxx.mp4 或 .png)
# 选填:
#   NP=1            卡数(>1 走 torchrun ulysses)
#   IMAGE=...       输入图(i2v/s2v/animate, 容器内可达路径, 如 /opt/LightX2V/assets/inputs/imgs/girl.png)
#   AUDIO=...       输入音频(s2v)
#   FRAMES=81       帧数
#   SEED=42
#   STEPS=4         infer_steps
#   EXTRA_ENV="-e KEY=VAL ..."   额外容器环境变量
#   HEALTH_TO=900   health 超时秒
#   KEEP=1          测完不停容器(便于复用/调试)
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
: "${NAME:?需设 NAME}"; : "${MODEL_CLS:?需设 MODEL_CLS}"; : "${TASK:?需设 TASK}"
: "${MODEL_PATH:?需设 MODEL_PATH}"; : "${CFG:?需设 CFG}"; : "${PROMPT:?需设 PROMPT}"; : "${OUT:?需设 OUT}"
NP="${NP:-1}"; FRAMES="${FRAMES:-81}"; SEED="${SEED:-42}"; STEPS="${STEPS:-4}"; HEALTH_TO="${HEALTH_TO:-900}"
API=http://localhost:8000
G=$'\e[32m'; R=$'\e[31m'; B=$'\e[36m'; N=$'\e[0m'
log(){ printf '%s[%s]%s %s\n' "$B" "$(date +%T)" "$N" "$*"; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"

# 起容器(单卡 python / 多卡 torchrun ulysses)
DEVS=$(seq -s, 0 $((NP-1)))
if [ "$NP" -gt 1 ]; then RUNCMD="torchrun --nproc_per_node=$NP --master_port=29533 -m lightx2v.server"; SHM="--shm-size=32g"; else RUNCMD="python -m lightx2v.server"; SHM=""; fi
log "起容器 $NAME (cls=$MODEL_CLS task=$TASK np=$NP) 配置=$CFG"
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --memory="${MEM:-240g}" --memory-swap="${MEM:-240g}" $SHM -p 8000:8000 -p 8001:8001 -v /data:/data -v /nfs-data:/nfs-data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES="$DEVS" ${EXTRA_ENV:-} \
  "$IMG" $RUNCMD --model_cls "$MODEL_CLS" --task "$TASK" \
  --model_path "$MODEL_PATH" --config_json "$CFG" --host 0.0.0.0 --port 8000 >/dev/null \
  || { printf '%s容器启动失败%s\n' "$R" "$N"; exit 2; }

# 等 health(同时记录加载耗时)
T0=$(date +%s); code=000
while [ "$(( $(date +%s)-T0 ))" -lt "$HEALTH_TO" ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
  sleep 10; printf '%s  加载中 %ss (health=%s)%s\r' "$B" "$(( $(date +%s)-T0 ))" "$code" "$N"
done; printf '\n'
[ "$code" = "200" ] || { printf '%shealth 超时, 末尾日志:%s\n' "$R" "$N"; docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /'; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
LOAD=$(( $(date +%s)-T0 )); log "${G}ready 加载 ${LOAD}s${N}, 提交生成..."

# 组装提交体(按需带 image_path/audio_path)
BODY=$(P="$PROMPT" O="$OUT" NF="$FRAMES" SD="$SEED" ST="$STEPS" IMG_P="${IMAGE:-}" LF="${LAST_FRAME:-}" AUD="${AUDIO:-}" RSZ="${RESIZE_MODE:-}" NP_="${NEG_PROMPT:-}" python3 -c '
import json,os
d={"prompt":os.environ["P"],"negative_prompt":os.environ.get("NP_",""),"save_result_path":os.environ["O"],
   "target_video_length":int(os.environ["NF"]),"infer_steps":int(os.environ["ST"]),"seed":int(os.environ["SD"])}
if os.environ.get("IMG_P"): d["image_path"]=os.environ["IMG_P"]
if os.environ.get("LF"): d["last_frame_path"]=os.environ["LF"]   # flf2v 尾帧
if os.environ.get("AUD"): d["audio_path"]=os.environ["AUD"]
rsz=os.environ.get("RSZ","")   # resize_mode: "null"->JSON null(触发 wan i2v 的 max_area=target_h*w 分辨率路径), 其余非空->原样字符串, 空->不传(默认adaptive走480p)
if rsz.lower() in ("null","none"): d["resize_mode"]=None
elif rsz: d["resize_mode"]=rsz
print(json.dumps(d))')

G0=$(date +%s)
case "$TASK" in t2i|i2i|*image*) EP=image ;; *) EP=video ;; esac   # 图像任务走 image 端点, 视频走 video
RESP=$(curl -sS -m 30 -X POST "$API/v1/tasks/$EP/" -H "Content-Type: application/json" -d "$BODY")
TID=$(echo "$RESP" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null)
[ -z "${TID:-}" ] && { printf '%s提交失败: %s%s\n' "$R" "$RESP" "$N"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }

# 轮询(带超时 + 探活, 不死循环)
PEAK=0; MISS=0; MAXW=$(( 60*60 )); ST=""
while true; do
  sleep 5; EL=$(( $(date +%s)-G0 ))
  [ "$EL" -gt "$MAXW" ] && { printf '%s超 %ss 未完成, 放弃%s\n' "$R" "$MAXW" "$N"; break; }
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)   # 所有卡取最大(多卡峰值才准)
  [ "$M" -gt "$PEAK" ] && PEAK=$M
  HC=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API/health" 2>/dev/null || echo 000)
  ST=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
  printf "  t=%ss status=%s gpu=%sMiB peak=%sMiB\n" "$EL" "${ST:-?}" "$M" "$PEAK"
  case "$ST" in
    completed) break ;;
    failed) curl -sS "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print('ERR:',(json.load(sys.stdin).get('error') or '')[:200])" 2>/dev/null; break ;;
    "") MISS=$((MISS+1)); [ "$HC" != "200" ] && [ "$MISS" -ge 3 ] && { printf '%sserver 挂了(health=%s)%s\n' "$R" "$HC" "$N"; break; } ;;
    *) MISS=0 ;;
  esac
done
GEN=$(( $(date +%s)-G0 ))

# 汇总
echo "============================================="
RC=0
if [ "$ST" != "completed" ]; then
  # 只认 completed: 失败/超时/server挂 一律失败, 不被残留或部分写的文件掩盖(否则上层/自动化误判成功)
  printf '%s任务未成功(status=%s) — 失败, 末尾日志:%s\n' "$R" "${ST:-超时/无响应}" "$N"
  docker logs --tail 30 "$NAME" 2>&1 | grep -iE 'error|oom|traceback|assert' | tail -10 | sed 's/^/    /'
  RC=1
elif [ -f "$OUT" ]; then
  SZB=$(stat -c%s "$OUT" 2>/dev/null || echo 0); SZ=$((SZB/1024))
  log "${G}完成${N} | 加载${LOAD}s | 生成${GEN}s | 峰值${PEAK}MiB | ${SZ}KB | $OUT"
  command -v ffprobe >/dev/null && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of default=noprint_wrappers=1 "$OUT" 2>/dev/null | sed 's/^/  /'
  if [ "$SZB" -lt 51200 ]; then printf '%s⚠️ 产物仅 %sKB, 疑似空/黑屏/损坏(生成没真跑通)%s\n' "$R" "$SZ" "$N"; RC=1; fi
else
  printf '%s任务报 completed 但无产物文件 — 失败%s\n' "$R" "$N"; RC=1
fi
[ "${KEEP:-0}" = "1" ] || docker rm -f "$NAME" >/dev/null 2>&1 || true
exit $RC
