#!/usr/bin/env bash
# =============================================================================
# InfiniteTalk 压测 —— 单容器复用, 连发 N 次, 丢首发(预热), 取稳态均值。
# 支持 单卡单实例 / 多卡单实例(ulysses), 全程采样 GPU显存/GPU利用率/容器CPU/容器内存/宿主可用内存 → CSV。
#
# 用法(计算节点, 脚本和配置都在 /data/smoke/):
#   bash /data/smoke/test_infinitetalk_stress.sh                       # 单卡(GPU2), 蒸馏4步, N=6
#   GPUS="0,1,2,3" PORT=8100 bash /data/smoke/test_infinitetalk_stress.sh   # 4卡 ulysses
#   NOOFL=1 bash /data/smoke/test_infinitetalk_stress.sh               # 去掉 cpu_offload(常驻显存形态)
#   N=10 STEPS=4 bash /data/smoke/test_infinitetalk_stress.sh
#
# env: GPUS(默认2) N(默认6) STEPS(默认4) PORT(默认8200) NOOFL(默认0)
#      CFG_BASE(默认 /data/smoke/infinitetalk_480p_single_distilled.json) KEEP
# 输出: /data/outputs/it_stress_<tag>/  (iter_*.mp4 + monitor.csv + 汇总打印)
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
GPUS="${GPUS:-2}"; N="${N:-6}"; STEPS="${STEPS:-4}"; PORT="${PORT:-8200}"; NOOFL="${NOOFL:-0}"
CFG_BASE="${CFG_BASE:-/data/smoke/infinitetalk_480p_single_distilled.json}"
MODEL_PATH=/nfs-data/models/Wan2.1-I2V-14B-480P
IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png
AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3
PROMPT="让角色根据音频内容自然说话"
PATCH="/data/smoke/it_transformer_infer.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/infinitetalk/transformer_infer.py:ro"
NP=$(awk -F, '{print NF}' <<<"$GPUS")
API="http://localhost:$PORT"
TAG="np${NP}_g${GPUS//,/}_$( [ "$NOOFL" = "1" ] && echo res || echo ofl )_s${STEPS}"
NAME="it-stress-$TAG"
OUTDIR="/data/outputs/it_stress_${TAG}"; CFG="/data/outputs/cfg_it_stress_${TAG}.json"
CSV="$OUTDIR/monitor.csv"
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'
[ "$NP" -gt 1 ] && [ $((40 % NP)) -ne 0 ] && { echo "${R}40 head 不被 $NP 整除, ulysses 不可用${N0}"; exit 2; }
mkdir -p "$OUTDIR"

# ---- 由基础配置生成本次配置(NOOFL 剥 offload; NP>1 注入 ulysses) ----
python3 - "$CFG_BASE" "$CFG" "$NOOFL" "$NP" <<'PY'
import json, sys
src, dst, noofl, np_ = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4])
c = json.load(open(src))
if noofl:
    c["cpu_offload"] = False; c.pop("offload_granularity", None)
if np_ > 1:
    # 配置里已带 parallel 块(含 fp8_comm 等调优 flags)则只补 size, 不覆盖
    p = c.get("parallel") if isinstance(c.get("parallel"), dict) else {}
    p.setdefault("seq_p_attn_type", "ulysses")
    p["seq_p_size"] = np_
    c["parallel"] = p
    # 4 rank 各自在 CPU 侧展开全量权重(~35G×4)+shmem 会打爆 256G(实测 global OOM 险僵机);
    # 只让 rank0 读盘、broadcast 给其余卡。
    c["load_from_rank0"] = True
json.dump(c, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"config -> {dst} (cpu_offload={c.get('cpu_offload')}, parallel={c.get('parallel')})")
PY

echo "${B}###### InfiniteTalk 压测 | GPUS=$GPUS(NP=$NP) | ${NOOFL:+NOOFL=$NOOFL }steps=$STEPS | N=$N ######${N0}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$NP" -gt 1 ]; then RUNCMD="torchrun --nproc_per_node=$NP --master_port=$((29000 + PORT % 1000)) -m lightx2v.server"; SHM="--shm-size=32g"; else RUNCMD="python -m lightx2v.server"; SHM=""; fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --memory=240g --memory-swap=240g $SHM -p "$PORT":8000 \
  -v /data:/data -v /nfs-data:/nfs-data -v "$PATCH" \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  "$IMG" $RUNCMD --model_cls infinitetalk --task s2v --model_path "$MODEL_PATH" --config_json "$CFG" \
  --host 0.0.0.0 --port 8000 >/dev/null || { echo "${R}容器启动失败${N0}"; exit 2; }

# ---- 后台监控采样器: 每 2s 一行 CSV(测完 awk 出各阶段峰值/曲线) ----
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

# ---- 等 health ----
T0=$(date +%s); code=000
while [ "$(( $(date +%s)-T0 ))" -lt 1800 ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break; sleep 5; printf "  加载中 %ss\r" "$(( $(date +%s)-T0 ))"
done; printf '\n'
[ "$code" = "200" ] || { echo "${R}health 超时, 末尾日志:${N0}"; docker logs --tail 40 "$NAME" 2>&1 | tail -15 | sed 's/^/    /'; exit 1; }
LOAD=$(( $(date +%s)-T0 )); echo "${G}ready 加载 ${LOAD}s${N0}, 连发 $N 条(首条预热不计)..."

# ---- 连发 N 条 ----
TIMES=()
for i in $(seq 1 "$N"); do
  echo "iter$i" > "$PHASE_F"
  OUT="$OUTDIR/iter_${i}.mp4"; rm -f "$OUT"
  BODY=$(P="$PROMPT" O="$OUT" ST="$STEPS" IM="$IMAGE" AU="$AUDIO" python3 -c "import json,os;print(json.dumps({'prompt':os.environ['P'],'negative_prompt':'','save_result_path':os.environ['O'],'infer_steps':int(os.environ['ST']),'seed':42,'image_path':os.environ['IM'],'audio_path':os.environ['AU']}))")
  TID=$(curl -sS -m 30 -X POST "$API/v1/tasks/video/" -H "Content-Type: application/json" -d "$BODY" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null)
  [ -z "${TID:-}" ] && { echo "  ${R}iter$i 提交失败${N0}"; continue; }
  t0=$(date +%s); ST_=""
  while true; do
    sleep 3
    ST_=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
    [ "$ST_" = "completed" ] && break
    [ "$ST_" = "failed" ] && { echo "  ${R}iter$i 失败${N0}"; break; }
    [ "$(( $(date +%s)-t0 ))" -gt 1200 ] && { echo "  ${R}iter$i 超时${N0}"; break; }
  done
  [ "$ST_" != "completed" ] && continue
  s=$(( $(date +%s)-t0 )); SZ=$(( $(stat -c%s "$OUT" 2>/dev/null || echo 0)/1024 ))
  # 本 iter 阶段内各项峰值(从 CSV 聚合)
  PK=$(awk -F, -v ph="iter$i" '$2==ph{ if($3>m)m=$3; if($4>u)u=$4; if($6>c)c=$6; if($7>me)me=$7; if(h==""||$8<h)h=$8 } END{printf "%s,%s,%s,%s,%s", m+0,u+0,c+0,me+0,h+0}' "$CSV")
  IFS=, read -r pk_gmem pk_util pk_cpu pk_cmem min_havail <<<"$PK"
  tag=""; [ "$i" = "1" ] && tag=" (预热, 不计)"
  printf "  iter%-2s %4ss | 显存峰 %sMiB | GPU利用峰 %s%% | CPU峰 %s%% | 容器内存峰 %sMiB | 宿主可用最低 %sGB | %sKB%s\n" \
    "$i" "$s" "$pk_gmem" "$pk_util" "$pk_cpu" "$pk_cmem" "$min_havail" "$SZ" "$tag"
  [ "$i" != "1" ] && TIMES+=("$s")
done
echo "done" > "$PHASE_F"

# ---- 汇总 ----
echo "============================================="
if [ "${#TIMES[@]}" -gt 0 ]; then
  printf '%s\n' "${TIMES[@]}" | sort -n | awk -v g="$GPUS" -v np="$NP" -v no="$NOOFL" -v st="$STEPS" -v ld="$LOAD" '
    {a[NR]=$1; s+=$1}
    END{n=NR; mean=s/n; med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
      printf "  [GPUS=%s NP=%s NOOFL=%s steps=%s] 加载 %ss | 稳态 %d 条: 均值 %.1fs | 中位 %ss | 最小 %ss | 最大 %ss | 吞吐 %.2f 条/分\n",
        g,np,no,st,ld,n,mean,med,a[1],a[n],60/mean}'
  GPEAK=$(awk -F, 'NR>1 && $2!="load"{if($3>m)m=$3} END{print m+0}' "$CSV")
  CPEAK=$(awk -F, 'NR>1{if($6>m)m=$6} END{print m+0}' "$CSV")
  MPEAK=$(awk -F, 'NR>1{if($7>m)m=$7} END{print m+0}' "$CSV")
  HMIN=$(awk -F, 'NR>1{if(h==""||$8<h)h=$8} END{print h+0}' "$CSV")
  echo "  全程峰值: 显存 ${GPEAK}MiB | 容器CPU ${CPEAK}% | 容器内存 ${MPEAK}MiB | 宿主可用内存最低 ${HMIN}GB"
  echo "  监控明细: $CSV (每2s一行, 含每卡显存列, 可画曲线)"
  RC=0
else
  echo "  ${R}无有效样本 — 全部失败/超时${N0}"; RC=1
fi
echo "  产物: $OUTDIR"
exit $RC
