#!/usr/bin/env bash
# 发一个视频 SR 请求并盯到终态;顺带采显存峰值、记录 phase/phase_progress 序列。
#
#   sr.sh <port> <input.mp4> <output.mp4> [额外 JSON 字段,形如 '"sr_ratio":4.0'] [cancel_after_s]
#
# 判据落盘到 /data/verify-logs/<output 名>.json:
#   http_code / wall_s / status / phase 序列是否单调 / 完成时 phase_progress==100 / vram_peak
set -u
PORT=${1:?port}
SRC=${2:?input}
OUT=${3:?output}
EXTRA=${4:-}
CANCEL_AFTER=${5:-0}
LOGD=/data/verify-logs
mkdir -p "$LOGD" "$(dirname "$OUT")"
NAME=$(basename "$OUT" .mp4)

BODY=$(printf '{"video_path":"%s","save_result_path":"%s","seed":42%s}' "$SRC" "$OUT" "${EXTRA:+,$EXTRA}")
T0=$(date +%s.%N)
RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/tasks/video/" -H 'Content-Type: application/json' -d "$BODY" --max-time 60)
TID=$(printf '%s' "$RESP" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("task_id",""))' 2>/dev/null)
if [ -z "$TID" ]; then echo "[sr] POST failed: $RESP"; exit 1; fi
echo "[sr] task=$TID body=$BODY"

# 显存采样(每 0.3s 一次,取每卡峰值)
: > "$LOGD/vram.$NAME.log"
( while :; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null >> "$LOGD/vram.$NAME.log"
    sleep 0.3
  done ) &
SAMPLER=$!

PHASES=(); PROGRESS=(); STATUS=""
CANCELLED=0
while :; do
  S=$(curl -s "http://127.0.0.1:$PORT/v1/tasks/$TID/status" 2>/dev/null)
  STATUS=$(printf '%s' "$S" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("status",""))' 2>/dev/null)
  PH=$(printf '%s' "$S" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("phase"),d.get("phase_progress"))' 2>/dev/null)
  PHASES+=("$PH")
  case "$STATUS" in
    completed|failed|cancelled) break ;;
  esac
  if [ "$CANCEL_AFTER" != 0 ] && [ "$CANCELLED" = 0 ]; then
    NOW=$(date +%s.%N)
    if python3 -c "import sys;sys.exit(0 if $NOW-$T0 >= $CANCEL_AFTER else 1)"; then
      echo "[sr] DELETE at +${CANCEL_AFTER}s"
      curl -s -X DELETE "http://127.0.0.1:$PORT/v1/tasks/$TID" >/dev/null
      CANCELLED=1
    fi
  fi
  sleep 1
done
T1=$(date +%s.%N)
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

PEAK=$(awk -F', ' '{if ($2+0 > m[$1]) m[$1]=$2+0} END {for (i in m) printf "%s=%sMiB ", i, m[i]}' "$LOGD/vram.$NAME.log")
python3 - "$NAME" "$TID" "$STATUS" "$T0" "$T1" "$PEAK" "${PHASES[@]}" <<'PY'
import json, sys
name, tid, status, t0, t1, peak, *phases = sys.argv[1:]
seq = []
for p in phases:
    ph, pr = p.split()
    seq.append({"phase": None if ph == "None" else ph, "progress": None if pr == "None" else float(pr)})
denoise = [s["progress"] for s in seq if s["phase"] == "denoise" and s["progress"] is not None]
monotonic = all(b >= a for a, b in zip(denoise, denoise[1:]))
last = seq[-1] if seq else {}
out = {
    "name": name, "task_id": tid, "status": status,
    "wall_s": round(float(t1) - float(t0), 1),
    "vram_peak": peak.strip(),
    "phase_samples": len(seq),
    "denoise_monotonic": monotonic,
    "denoise_max": max(denoise) if denoise else None,
    "final_phase": last.get("phase"),
    "final_progress": last.get("progress"),
}
with open(f"/data/verify-logs/{name}.json", "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
PY
