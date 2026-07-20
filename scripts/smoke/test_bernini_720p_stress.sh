#!/usr/bin/env bash
# =============================================================================
# Bernini-R 720p 4卡ulysses 压测 —— 单容器复用, 连发 N 次, 丢首发(triton预热), 取稳态。
# 一个脚本覆盖 E0(热耗时)/E1(RIFE插帧)/E2(扫帧数)/E6(长稳)。
# 全程采样 GPU显存/利用率/每卡显存/容器CPU/容器内存/宿主可用内存 → CSV。
# 照 test_infinitetalk_stress.sh 规范改写;差异见 ⚠️。
#
# 用法(计算节点, 脚本和配置都在 /nfs-models/_transfer/):
#   bash test_bernini_720p_stress.sh                          # 4卡 ulysses, 81帧, N=6 → E0 热态基线
#   TARGET_FPS=32 bash test_bernini_720p_stress.sh            # E1 插帧(需 RIFE 权重 + rife config)
#   FRAMES=121 bash test_bernini_720p_stress.sh               # E2 扫帧数(改 121/161/201)
#   N=80 bash test_bernini_720p_stress.sh                     # E6 长稳(~1h+)
#   GPUS="0,1" bash test_bernini_720p_stress.sh               # E4 并行度阶梯(seq_p=2)
#
# env: GPUS(默认0,1,2,3) N(默认6) STEPS(默认4) PORT(默认8300) FRAMES(默认81) TARGET_FPS(默认0=不插帧) KEEP
#      CFG_BASE(默认 ulysses4 int8;TARGET_FPS>0 时建议换 ..._rife_int8.json)
# 输出: /nfs-output/bernini_720p_stress_<tag>/  (iter_*.mp4 + monitor.csv + 汇总打印)
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
GPUS="${GPUS:-0,1,2,3}"; N="${N:-6}"; STEPS="${STEPS:-4}"; PORT="${PORT:-8300}"
FRAMES="${FRAMES:-81}"; TARGET_FPS="${TARGET_FPS:-0}"
CFG_BASE="${CFG_BASE:-/nfs-models/_transfer/bernini_r_14b_t2v_720p_ulysses4_int8.json}"
MODEL_PATH=/nfs-models/wuhanjisuan894/models/Wan2.1-I2V-14B-480P
PROMPT="A cinematic shot of a golden retriever running through a field of sunflowers at sunset, warm rim light, 35mm film"
NEG="blurry, low quality, distorted, overexposed"
NP=$(awk -F, '{print NF}' <<<"$GPUS")
API="http://localhost:$PORT"
TAG="np${NP}_g${GPUS//,/}_f${FRAMES}_fps${TARGET_FPS}_s${STEPS}"
NAME="bernini720-stress-$TAG"
OUTDIR="/nfs-output/bernini_720p_stress_${TAG}"; CFG="/nfs-output/cfg_bernini720_${TAG}.json"
CSV="$OUTDIR/monitor.csv"
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'
# Bernini 40 head, ulysses 需被卡数整除
[ "$NP" -gt 1 ] && [ $((40 % NP)) -ne 0 ] && { echo "${R}40 head 不被 $NP 整除, ulysses 不可用${N0}"; exit 2; }
mkdir -p "$OUTDIR"

# ---- 由基础配置生成本次配置(注入 ulysses/帧数/插帧;⚠️禁 load_from_rank0) ----
python3 - "$CFG_BASE" "$CFG" "$NP" "$FRAMES" "$TARGET_FPS" <<'PY'
import json, sys
src, dst, np_, frames, tfps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = json.load(open(src))
c["target_video_length"] = frames        # 须 4n+1;set_config 会再对齐 VAE stride
if np_ > 1:
    p = c.get("parallel") if isinstance(c.get("parallel"), dict) else {}
    p.setdefault("seq_p_attn_type", "ulysses"); p["seq_p_size"] = np_
    c["parallel"] = p
else:
    c.pop("parallel", None)               # 单卡:去 parallel(720p单卡大概率OOM,仅E4间接对照用)
# ⚠️ Bernini = Wan2.2 MoE 双专家,禁 load_from_rank0(会破坏 high/low 专家路由,W1/VACE 铁律)。
c.pop("load_from_rank0", None)
if tfps > 0:                              # E1:开 RIFE 插帧(需 CFG_BASE 带 video_frame_interpolation 或此处补)
    vfi = c.get("video_frame_interpolation") if isinstance(c.get("video_frame_interpolation"), dict) else \
          {"algo": "rife", "model_path": "/nfs-models/wuhanjisuan894/models/rife/flownet.pkl"}
    vfi["target_fps"] = tfps
    c["video_frame_interpolation"] = vfi
    c.setdefault("fps", 16)
json.dump(c, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"config -> {dst} (parallel={c.get('parallel')}, frames={frames}, vfi={c.get('video_frame_interpolation')})")
PY

echo "${B}###### Bernini-R 720p 压测 | GPUS=$GPUS(NP=$NP) | frames=$FRAMES | rife_fps=$TARGET_FPS | steps=$STEPS | N=$N ######${N0}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$NP" -gt 1 ]; then RUNCMD="torchrun --nproc_per_node=$NP --master_port=$((29000 + PORT % 1000)) -m lightx2v.server"; SHM="--shm-size=32g"; else RUNCMD="python -m lightx2v.server"; SHM=""; fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --memory=240g --memory-swap=240g $SHM -p "$PORT":8000 \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  "$IMG" $RUNCMD --model_cls wan2.2_moe_distill --task t2v --model_path "$MODEL_PATH" --config_json "$CFG" \
  --host 0.0.0.0 --port 8000 >/dev/null || { echo "${R}容器启动失败${N0}"; exit 2; }

# ---- 后台监控采样器: 每 2s 一行 CSV ----
echo "ts,phase,gpu_mem_max_mib,gpu_util_max,percard_mem,cpu_pct,ctr_mem_mib,host_avail_gb" > "$CSV"
PHASE_F="$OUTDIR/.phase"; echo "load" > "$PHASE_F"
to_mib(){ awk -v s="$1" 'BEGIN{n=s+0; u=tolower(s); m=(index(u,"gi")||index(u,"gb"))?n*1024:(index(u,"ki")||index(u,"kb"))?n/1024:n; printf "%.0f", m}'; }
mon(){
  while true; do
    sleep 2
    local MM UU PC DS CPU MEMU HAV
    MM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
    UU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
    PC=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd'|' - || echo 0)
    DS=$(docker stats --no-stream --format '{{.CPUPerc}}|{{.MemUsage}}' "$NAME" 2>/dev/null || echo "0%|0MiB")
    CPU=${DS%%|*}; CPU=${CPU%\%}; MEMU=${DS#*|}; MEMU=${MEMU%% *}
    HAV=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    echo "$(date +%s),$(cat "$PHASE_F" 2>/dev/null || echo ?),$MM,$UU,$PC,${CPU:-0},$(to_mib "${MEMU:-0}"),$HAV" >> "$CSV"
  done
}
mon & MON=$!
trap 'kill $MON 2>/dev/null; [ "${KEEP:-0}" = "1" ] || docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

# ---- 等 health(720p 4卡 NFS 冷加载 ~350s,给足 1800s) ----
T0=$(date +%s); code=000
while [ "$(( $(date +%s)-T0 ))" -lt 1800 ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break; sleep 5; printf "  加载中 %ss\r" "$(( $(date +%s)-T0 ))"
done; printf '\n'
[ "$code" = "200" ] || { echo "${R}health 超时, 末尾日志:${N0}"; docker logs --tail 40 "$NAME" 2>&1 | tail -15 | sed 's/^/    /'; exit 1; }
LOAD=$(( $(date +%s)-T0 )); echo "${G}ready 加载 ${LOAD}s${N0}, 连发 $N 条(首条 triton 预热不计)..."

# ---- 连发 N 条 ----
TIMES=(); PROBE=""
for i in $(seq 1 "$N"); do
  echo "iter$i" > "$PHASE_F"
  OUT="$OUTDIR/iter_${i}.mp4"; rm -f "$OUT"
  BODY=$(P="$PROMPT" NG="$NEG" O="$OUT" ST="$STEPS" FR="$FRAMES" TF="$TARGET_FPS" SD="$((1000+i))" python3 -c '
import json,os
b={"prompt":os.environ["P"],"negative_prompt":os.environ["NG"],"save_result_path":os.environ["O"],
   "infer_steps":int(os.environ["ST"]),"seed":int(os.environ["SD"]),"target_video_length":int(os.environ["FR"])}
tf=int(os.environ["TF"])
if tf>0: b["target_fps"]=tf
print(json.dumps(b))')
  TID=$(curl -sS -m 30 -X POST "$API/v1/tasks/video/" -H "Content-Type: application/json" -d "$BODY" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null)
  [ -z "${TID:-}" ] && { echo "  ${R}iter$i 提交失败${N0}"; continue; }
  t0=$(date +%s); ST_=""
  while true; do
    sleep 3
    ST_=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
    [ "$ST_" = "completed" ] && break
    [ "$ST_" = "failed" ] && { echo "  ${R}iter$i 失败(可能OOM,server优雅failed不崩,继续下一条)${N0}"; break; }
    [ "$(( $(date +%s)-t0 ))" -gt 1200 ] && { echo "  ${R}iter$i 超时${N0}"; break; }
  done
  [ "$ST_" != "completed" ] && continue
  s=$(( $(date +%s)-t0 )); SZ=$(( $(stat -c%s "$OUT" 2>/dev/null || echo 0)/1024 ))
  PK=$(awk -F, -v ph="iter$i" '$2==ph{ if($3>m)m=$3; if($4>u)u=$4; if($6>c)c=$6; if($7>me)me=$7; if(h==""||$8<h)h=$8 } END{printf "%s,%s,%s,%s,%s", m+0,u+0,c+0,me+0,h+0}' "$CSV")
  IFS=, read -r pk_gmem pk_util pk_cpu pk_cmem min_havail <<<"$PK"
  tag=""; [ "$i" = "1" ] && tag=" (预热, 不计)"
  printf "  iter%-2s %4ss | 显存峰 %sMiB | GPU利用峰 %s%% | 容器内存峰 %sMiB | 宿主可用最低 %sGB | %sKB%s\n" \
    "$i" "$s" "$pk_gmem" "$pk_util" "$pk_cmem" "$min_havail" "$SZ" "$tag"
  [ "$i" = "1" ] && PROBE="$OUT"      # 首条留作分辨率核实
  [ "$i" != "1" ] && TIMES+=("$s")
done
echo "done" > "$PHASE_F"

# ---- ⚠️ 分辨率核实(I2V §8 坑:出片可能非 720×1280)----
if [ -n "$PROBE" ] && [ -f "$PROBE" ]; then
  RES=$(docker run --rm -v /nfs-output:/nfs-output --entrypoint ffprobe "$IMG" -v error -select_streams v:0 \
        -show_entries stream=width,height,nb_frames -of csv=p=0 "$PROBE" 2>/dev/null || echo "?")
  echo "  出片规格(w,h,frames): $RES  ← 期望 1280,720,${FRAMES};若不符看 I2V报告§8(resize_mode)"
fi

# ---- 汇总 ----
echo "============================================="
if [ "${#TIMES[@]}" -gt 0 ]; then
  printf '%s\n' "${TIMES[@]}" | sort -n | awk -v g="$GPUS" -v np="$NP" -v fr="$FRAMES" -v tf="$TARGET_FPS" -v st="$STEPS" -v ld="$LOAD" '
    {a[NR]=$1; s+=$1}
    END{n=NR; mean=s/n; med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
      printf "  [GPUS=%s NP=%s frames=%s rife_fps=%s steps=%s] 加载 %ss | 稳态 %d 条: 均值 %.1fs | 中位 %ss | 最小 %ss | 最大 %ss | 吞吐 %.2f 条/分\n",
        g,np,fr,tf,st,ld,n,mean,med,a[1],a[n],60/mean}'
  GPEAK=$(awk -F, 'NR>1 && $2!="load"{if($3>m)m=$3} END{print m+0}' "$CSV")
  MPEAK=$(awk -F, 'NR>1{if($7>m)m=$7} END{print m+0}' "$CSV")
  HMIN=$(awk -F, 'NR>1{if(h==""||$8<h)h=$8} END{print h+0}' "$CSV")
  echo "  全程峰值: 显存 ${GPEAK}MiB | 容器内存 ${MPEAK}MiB | 宿主可用内存最低 ${HMIN}GB"
  echo "  监控明细: $CSV (每2s一行, 含每卡显存列)"
  RC=0
else
  echo "  ${R}无有效样本 — 全部失败/超时(720p 4卡若首条就 failed,多半是 OOM 或 NCCL)${N0}"; RC=1
fi
echo "  产物: $OUTDIR"
exit $RC
