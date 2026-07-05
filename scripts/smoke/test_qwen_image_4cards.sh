#!/usr/bin/env bash
# =============================================================================
# Qwen-Image N 单卡实例 并发吞吐压测(仿 test_z_image_4cards.sh)
#   每卡一个实例(CUDA_VISIBLE_DEVICES=i, 端口 8000+i);
#   ① 每实例预热 1 张(焊 autotune 缓存, 丢弃不计);
#   ② 并发 REQS 张轮换 prompt 分发到各实例, 测墙钟 → img/s;
#   ③ 采样每实例容器内存峰值 + 主机已用峰值(Qwen offload 权重驻 CPU, 内存是重点)。
#
# 默认 MODE=merged8(8步快版, 生产吞吐主力)。每实例 cpu_offload=true → 约 40G 主机内存/实例。
#
# 用法(服务器, 先 scp 到 /data/):
#   bash /data/test_qwen_image_4cards.sh                 # 4 实例 merged8, 并发 16 张
#   REQS=32 NINST=4 bash /data/test_qwen_image_4cards.sh
#   MODE=base bash /data/test_qwen_image_4cards.sh       # 25步基座(慢)
# 选填 env: MODE(merged8|base) NINST(默认4) REQS(默认16) ASPECT(默认1:1) SEED MEM(每实例内存,默认64g) KEEP
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
MODE="${MODE:-merged8}"; NINST="${NINST:-4}"; REQS="${REQS:-16}"; ASPECT="${ASPECT:-1:1}"; SEED="${SEED:-42}"; MEM="${MEM:-64g}"
STAGGER="${STAGGER:-0}"   # 每起一个实例后等 N 秒再起下一个,降 SFS 同时读 152G 的争抢(建议 45-60)
CFGDIR=/data/lightx2v_configs; MODEL_BASE=/data/models/Qwen-Image-2512
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'

TASK=t2i
case "$MODE" in
  merged8) CFG=$CFGDIR/qwen_2512_a100_lightning_merged.json;      MP=$MODEL_BASE;;
  base)    CFG=$CFGDIR/qwen_2512_a100_base.json;                  MP=$MODEL_BASE;;
  edit)    CFG=$CFGDIR/qwen_edit_2511_a100_base.json;             MP=/data/models/Qwen-Image-Edit-2511; TASK=i2i;;
  edit_m)  CFG=$CFGDIR/qwen_edit_2511_a100_lightning_merged.json; MP=/data/models/Qwen-Image-Edit-2511; TASK=i2i;;
  edit_int8) CFG=$CFGDIR/qwen_edit_2511_a100_int8_offload.json;      MP=/data/models/Qwen-Image-Edit-2511; TASK=i2i;;
  *) echo "${R}MODE ∈ merged8|base|edit|edit_m|edit_int8${N0}"; exit 2;;
esac
IMAGE="${IMAGE:-/data/_editsrc/src_1_16x9.png}"   # i2i 输入图(edit/edit_m 用)
[ "$TASK" = i2i ] && { [ -f "$IMAGE" ] || { echo "${R}i2i 需 IMAGE=<存在的输入图>,当前: $IMAGE${N0}"; exit 2; }; }
[ -f "$CFG" ] || { echo "${R}配置不存在: $CFG${N0}"; exit 2; }
OUTDIR="/data/outputs/qwen_4cards_${MODE}"; mkdir -p "$OUTDIR"
to_mib(){ awk -v s="$1" 'BEGIN{n=s+0; u=tolower(s); m=(index(u,"gi")||index(u,"gb"))?n*1024:(index(u,"ki")||index(u,"kb"))?n/1024:n; printf "%.0f", m}'; }
PROMPTS=(
  "a cozy bookstore cafe with warm lighting, photorealistic, highly detailed"
  "a serene mountain lake at sunrise, mist over water, pine forest"
  "a vintage red sports car on a wet city street at night, neon reflections"
  "a Bengal tiger walking through golden grass, cinematic wildlife"
  "a futuristic city skyline at dusk, flying cars, cyberpunk"
  "a bowl of ramen with steam, chopsticks, wooden table, top view"
  "an astronaut cat floating in space, stars and nebula"
  "a watercolor painting of cherry blossoms by a river"
)
NAMES=$(for i in $(seq 0 $((NINST-1))); do echo "qwen-inst-$i"; done)

echo "${B}###### Qwen $NINST 单卡实例 | MODE=$MODE | 并发 $REQS 张 | aspect=$ASPECT ######${N0}"
# ---- 起 N 个单卡实例 ----
for i in $(seq 0 $((NINST-1))); do
  nm="qwen-inst-$i"; port=$((8000+i)); docker rm -f "$nm" >/dev/null 2>&1 || true
  docker run -d --name "$nm" --gpus all --init --memory="$MEM" -p "$port":8000 \
    -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES="$i" \
    "$IMG" python -m lightx2v.server --model_cls qwen_image --task "$TASK" --model_path "$MP" \
    --config_json "$CFG" --host 0.0.0.0 --port 8000 >/dev/null \
    && echo "  起 $nm (GPU$i, 端口$port)" || echo "  ${R}$nm 启动失败${N0}"
  # 错峰:除最后一个,起完等 STAGGER 秒再起下一个(降 SFS 峰值争抢)
  [ "$STAGGER" -gt 0 ] && [ "$i" -lt "$((NINST-1))" ] && { echo "  ...错峰等 ${STAGGER}s"; sleep "$STAGGER"; }
done
# ---- 等全部 ready(qwen 用 queue/status;4 实例同时读 SFS,给 1800s)----
echo "  等 $NINST 实例就绪(offload+SFS 冷启动可能几分钟)..."; T0=$(date +%s)
for i in $(seq 0 $((NINST-1))); do
  port=$((8000+i)); ok=0
  while [ "$(( $(date +%s)-T0 ))" -lt 1800 ]; do
    if ! docker ps --format '{{.Names}}' | grep -qx "qwen-inst-$i"; then
      echo "  ${R}实例$i 崩了${N0}"; docker logs --tail 25 "qwen-inst-$i" 2>&1 | sed 's/^/    /'; break
    fi
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/v1/tasks/queue/status" 2>/dev/null || echo 000)" = "200" ] && { ok=1; break; }; sleep 4
  done
  [ "$ok" = "1" ] && echo "  ${G}实例$i 就绪${N0}" || echo "  ${R}实例$i 超时/失败${N0}"
done
echo "  加载耗时 $(( $(date +%s)-T0 ))s"

post_one(){ # $1=port $2=prompt $3=out -> task_id
  local body; body=$(P="$2" O="$3" AR="$ASPECT" SD="$SEED" TK="$TASK" IMG_IN="$IMAGE" python3 -c "import json,os
d={'prompt':os.environ['P'],'save_result_path':os.environ['O'],'aspect_ratio':os.environ['AR'],'seed':int(os.environ['SD'])}
if os.environ['TK']=='i2i': d['image_path']=os.environ['IMG_IN']
print(json.dumps(d))")
  curl -sS -m 30 -X POST "http://localhost:$1/v1/tasks/image/" -H "Content-Type: application/json" -d "$body" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null
}
status_of(){ curl -sS -m 10 "http://localhost:$1/v1/tasks/$2/status" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status") or "")' 2>/dev/null; }

declare -a MEMPK; for i in $(seq 0 $((NINST-1))); do MEMPK[$i]=0; done; HOSTPK=0
sample_mem(){
  while read -r nm mem; do
    local idx=${nm#qwen-inst-}; local mib; mib=$(to_mib "$mem")
    [ "${mib:-0}" -gt "${MEMPK[$idx]:-0}" ] 2>/dev/null && MEMPK[$idx]=$mib
  done < <(docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' $NAMES 2>/dev/null | awk '{print $1, $2}')
  local hu; hu=$(free -m | awk '/^Mem:/{print $3}')
  [ "${hu:-0}" -gt "$HOSTPK" ] 2>/dev/null && HOSTPK=$hu
}

# ---- 预热: 每实例 1 张(焊 autotune, 不计入吞吐)----
echo "  ${B}预热: 每实例 1 张...${N0}"
declare -a WP WT
for i in $(seq 0 $((NINST-1))); do
  WP[$i]="$((8000+i))"; WT[$i]=$(post_one $((8000+i)) "warmup" "$OUTDIR/warm_$i.png")
done
W0=$(date +%s); wdone=0
while [ "$wdone" -lt "$NINST" ]; do
  sleep 2; wdone=0; sample_mem
  for i in $(seq 0 $((NINST-1))); do
    [ "${WT[$i]}" = "DONE" ] && { wdone=$((wdone+1)); continue; }
    [ -z "${WT[$i]}" ] && { WT[$i]=DONE; wdone=$((wdone+1)); continue; }
    st=$(status_of "${WP[$i]}" "${WT[$i]}")
    { [ "$st" = completed ] || [ "$st" = failed ]; } && { WT[$i]=DONE; wdone=$((wdone+1)); }
  done
  printf "    预热 %s/%s\r" "$wdone" "$NINST"
  [ "$(( $(date +%s)-W0 ))" -gt 900 ] && { printf '\n%s预热超时%s\n' "$R" "$N0"; break; }
done; printf '\n'

# ---- 并发负载: REQS 张轮询分发 ----
echo "  ${B}并发: $REQS 张分发到 $NINST 实例...${N0}"
declare -a PORTS TIDS FINAL; WALL0=$(date +%s%3N)
for r in $(seq 1 "$REQS"); do
  i=$(( (r-1) % NINST )); pr="${PROMPTS[$(( (r-1) % ${#PROMPTS[@]} ))]}"
  PORTS[$r]="$((8000+i))"; TIDS[$r]=$(post_one $((8000+i)) "$pr" "$OUTDIR/req_${r}_gpu${i}.png")
done
done_cnt=0; POLL0=$(date +%s)
while [ "$done_cnt" -lt "$REQS" ]; do
  sleep 1; done_cnt=0; sample_mem
  for r in $(seq 1 "$REQS"); do
    if [ -n "${FINAL[$r]:-}" ]; then done_cnt=$((done_cnt+1)); continue; fi
    if [ -z "${TIDS[$r]:-}" ]; then FINAL[$r]="failed"; done_cnt=$((done_cnt+1)); continue; fi
    st=$(status_of "${PORTS[$r]}" "${TIDS[$r]}")
    { [ "$st" = completed ] || [ "$st" = failed ]; } && { FINAL[$r]="$st"; done_cnt=$((done_cnt+1)); }
  done
  printf "    完成 %s/%s\r" "$done_cnt" "$REQS"
  [ "$(( $(date +%s)-POLL0 ))" -gt 3600 ] && { printf '\n%s轮询超时%s\n' "$R" "$N0"; break; }
done
WALL=$(( $(date +%s%3N)-WALL0 )); printf '\n'; sample_mem

# ---- 汇总: 吞吐 + 内存 ----
SUCC=0; FAIL=0
for r in $(seq 1 "$REQS"); do [ "${FINAL[$r]:-failed}" = "completed" ] && SUCC=$((SUCC+1)) || FAIL=$((FAIL+1)); done
echo "============================================="
RC=0
if [ "$SUCC" -eq 0 ]; then printf '%s全部失败(0/%s)— 查日志%s\n' "$R" "$REQS" "$N0"; RC=1
else
  awk -v succ="$SUCC" -v reqs="$REQS" -v wall="$WALL" -v n="$NINST" 'BEGIN{sec=wall/1000;tput=succ/sec;
    printf "  %d 实例 | 成功 %d/%d | 墙钟 %.1fs | 吞吐 %.3f img/s | 单实例 %.3f img/s\n",n,succ,reqs,sec,tput,tput/n}'
  [ "$FAIL" -gt 0 ] && { printf '%s⚠️ %s 张失败%s\n' "$R" "$FAIL" "$N0"; RC=1; }
fi
echo "  ---- 内存实测(峰值)----"
tot=0
for i in $(seq 0 $((NINST-1))); do mb=${MEMPK[$i]}; tot=$((tot+mb)); printf "  实例$i(GPU$i): %s MiB (%.2f GB)\n" "$mb" "$(awk "BEGIN{print $mb/1024}")"; done
awk -v t="$tot" -v h="$HOSTPK" -v n="$NINST" 'BEGIN{
  printf "  %d 实例容器内存合计: %d MiB (%.2f GB)\n", n, t, t/1024
  printf "  主机已用峰值(free): %d MiB (%.2f GB)\n", h, h/1024 }'
ok=$(ls "$OUTDIR"/req_*.png 2>/dev/null | wc -l); echo "  产物: $OUTDIR ($ok 张)"
echo "  清理: docker rm -f \$(for i in \$(seq 0 $((NINST-1))); do echo qwen-inst-\$i; done)"
[ "${KEEP:-0}" = "1" ] || for i in $(seq 0 $((NINST-1))); do docker rm -f "qwen-inst-$i" >/dev/null 2>&1 || true; done
exit $RC
