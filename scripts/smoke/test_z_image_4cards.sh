#!/usr/bin/env bash
# =============================================================================
# Z-Image N 单卡实例 并发吞吐 + 内存实测 + 全分辨率预热验证
#   每卡一个实例(CUDA_VISIBLE_DEVICES=i, 端口 8000+i);
#   ① 启动时每实例预热【全部 7 种分辨率】→ 验证"7 个分辨率都加载"+ 焊死 autotune 缓存;
#   ② 并发负载轮换分辨率/提示词(真实混流量);
#   ③ 全程采样【每实例容器内存峰值 + 主机内存用量峰值】,把"8GB 外推"坐实成实测。
#
# 用法(服务器, 先 scp 到 /data/):
#   bash /data/test_z_image_4cards.sh                  # 4 实例 bf16, 预热7比例, 并发16张
#   REQS=32 bash /data/test_z_image_4cards.sh
# 选填 env: PREC NINST REQS STEPS SEED ASPECTS
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
PREC="${PREC:-bf16}"; NINST="${NINST:-4}"; REQS="${REQS:-16}"; STEPS="${STEPS:-9}"; SEED="${SEED:-42}"
ASPECTS="${ASPECTS:-16:9 9:16 1:1 4:3 3:4 3:2 2:3}"
BF16_PATH=/nfs-data/models/Z-Image-Turbo; INT8_CKPT=/nfs-data/models-int8/Z-Image-Turbo-int8
OUTDIR="/data/outputs/z_4cards_${PREC}"; CFG="/data/cfg_z_4cards_${PREC}.json"
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'
NASP=$(echo $ASPECTS | wc -w | tr -d ' ')
mkdir -p "$OUTDIR"
to_mib(){ awk -v s="$1" 'BEGIN{n=s+0; u=tolower(s); m=(index(u,"gi")||index(u,"gb"))?n*1024:(index(u,"ki")||index(u,"kb"))?n/1024:n; printf "%.0f", m}'; }
PROMPTS=(
  "Young Chinese woman in red Hanfu, golden phoenix headdress, soft night light"
  "A serene mountain lake at sunrise, mist over water, pine forest"
  "A vintage red sports car on a wet city street at night, neon reflections"
  "A wooden sign with carved text, rustic background, warm light"
  "A Bengal tiger walking through golden grass, cinematic wildlife"
  "Abstract fluid art, swirling blue and gold paint, macro"
  "A cozy coffee shop interior, warm lighting, plants, books"
  "An astronaut cat floating in space, stars and nebula"
)

# 配置(单卡, 无 parallel; aspect_ratio 由请求体逐张指定)
{
  echo "{"
  echo "  \"num_channels_latents\": 16, \"infer_steps\": $STEPS, \"attn_type\": \"sage_attn2\","
  echo "  \"enable_cfg\": false, \"sample_guide_scale\": 0.0, \"rope_type\": \"torch\", \"patch_size\": 2"
  [ "$PREC" = "int8" ] && echo "  ,\"dit_quantized\": true, \"dit_quant_scheme\": \"int8-torchao\", \"dit_quantized_ckpt\": \"$INT8_CKPT\""
  echo "}"
} > "$CFG"

echo "${B}###### Z-Image $NINST 单卡实例 | PREC=$PREC | 预热 $NASP 比例 | 并发 $REQS 张 ######${N0}"
# ---- 起 N 个单卡实例 ----
for i in $(seq 0 $((NINST-1))); do
  nm="z-inst-$i"; port=$((8000+i)); docker rm -f "$nm" >/dev/null 2>&1 || true
  docker run -d --name "$nm" --gpus all --memory=60g --memory-swap=60g -p "$port":8000 \
    -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES="$i" \
    "$IMG" python -m lightx2v.server --model_cls z_image --task t2i --model_path "$BF16_PATH" \
    --config_json "$CFG" --host 0.0.0.0 --port 8000 >/dev/null \
    && echo "  起 $nm (GPU$i, 端口$port)" || echo "  ${R}$nm 启动失败${N0}"
done
# ---- 等全部 health ----
echo "  等 $NINST 实例就绪..."; T0=$(date +%s)
for i in $(seq 0 $((NINST-1))); do
  port=$((8000+i)); ok=0
  while [ "$(( $(date +%s)-T0 ))" -lt 900 ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" 2>/dev/null || echo 000)" = "200" ] && { ok=1; break; }; sleep 3
  done
  [ "$ok" = "1" ] && echo "  ${G}实例$i 就绪${N0}" || { echo "  ${R}实例$i health 超时${N0}"; docker logs --tail 20 "z-inst-$i" 2>&1 | sed 's/^/    /'; }
done
echo "  全部就绪, 加载 $(( $(date +%s)-T0 ))s"

post_one(){ # $1=port $2=prompt $3=out $4=aspect -> task_id
  local body; body=$(P="$2" O="$3" AR="$4" ST="$STEPS" SD="$SEED" python3 -c "import json,os;print(json.dumps({'prompt':os.environ['P'],'negative_prompt':' ','aspect_ratio':os.environ['AR'],'save_result_path':os.environ['O'],'infer_steps':int(os.environ['ST']),'seed':int(os.environ['SD'])}))")
  curl -sS -m 30 -X POST "http://localhost:$1/v1/tasks/image/" -H "Content-Type: application/json" -d "$body" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null
}
status_of(){ curl -sS -m 10 "http://localhost:$1/v1/tasks/$2/status" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status") or "")' 2>/dev/null; }

# ---- 内存采样: 每实例容器内存峰值(MiB) + 主机已用峰值(MiB) ----
declare -a MEMPK; for i in $(seq 0 $((NINST-1))); do MEMPK[$i]=0; done; HOSTPK=0
names=$(for i in $(seq 0 $((NINST-1))); do echo "z-inst-$i"; done)
sample_mem(){
  while read -r nm mem; do
    local idx=${nm#z-inst-}; local mib; mib=$(to_mib "$mem")
    [ "${mib:-0}" -gt "${MEMPK[$idx]:-0}" ] 2>/dev/null && MEMPK[$idx]=$mib
  done < <(docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' $names 2>/dev/null | awk '{print $1, $2}')
  local hu; hu=$(free -m | awk '/^Mem:/{print $3}')
  [ "${hu:-0}" -gt "$HOSTPK" ] 2>/dev/null && HOSTPK=$hu
}

# ---- 预热: 每实例 × 全部 ASPECTS(并行: 全提交再轮询), 验证 7 个分辨率都加载 ----
echo "  ${B}预热: 每实例跑全部 $NASP 种分辨率...${N0}"
declare -a WP WT; declare -a WARMOK; for i in $(seq 0 $((NINST-1))); do WARMOK[$i]=0; done
k=0
for i in $(seq 0 $((NINST-1))); do
  for ar in $ASPECTS; do
    safe="${ar/:/_}"; tid=$(post_one $((8000+i)) "warmup $ar" "$OUTDIR/warm_${i}_${safe}.png" "$ar")
    WP[$k]="$((8000+i)):$i"; WT[$k]="$tid"; k=$((k+1))
  done
done
W0=$(date +%s); wdone=0
while [ "$wdone" -lt "$k" ]; do
  sleep 2; wdone=0; sample_mem
  for j in $(seq 0 $((k-1))); do
    port="${WP[$j]%%:*}"; idx="${WP[$j]##*:}"; tid="${WT[$j]}"
    [ -z "$tid" ] && { wdone=$((wdone+1)); continue; }
    [ "${WT[$j]}" = "DONE" ] && { wdone=$((wdone+1)); continue; }
    st=$(status_of "$port" "$tid")
    if [ "$st" = "completed" ]; then WARMOK[$idx]=$(( ${WARMOK[$idx]} + 1 )); WT[$j]="DONE"; wdone=$((wdone+1));
    elif [ "$st" = "failed" ]; then WT[$j]="DONE"; wdone=$((wdone+1)); fi
  done
  printf "    预热 %s/%s\r" "$wdone" "$k"
  [ "$(( $(date +%s)-W0 ))" -gt 900 ] && { printf '\n%s预热超时%s\n' "$R" "$N0"; break; }
done; printf '\n'
sample_mem
WARM_RC=0
for i in $(seq 0 $((NINST-1))); do
  c=${WARMOK[$i]}; [ "$c" -eq "$NASP" ] && tag="${G}✓${N0}" || { tag="${R}✗${N0}"; WARM_RC=1; }
  echo "  实例$i 预热 $c/$NASP 分辨率 $tag"
done
# 预热没全部成功 → 中止基准(实例未完全预热 + 残留预热任务会污染吞吐),不再往下跑
if [ "$WARM_RC" = "1" ]; then
  echo "${R}✗ 预热未全 $NASP/$NASP, 实例未完全预热 → 中止基准测试(避免吞吐被污染)${N0}"
  [ "${KEEP:-0}" = "1" ] || for i in $(seq 0 $((NINST-1))); do docker rm -f "z-inst-$i" >/dev/null 2>&1 || true; done
  exit 1
fi

# ---- 并发负载: REQS 张轮换 aspect+prompt 分发到实例 ----
echo "  ${B}并发: $REQS 张(轮换 $NASP 分辨率)分发到 $NINST 实例...${N0}"
declare -a PORTS TIDS FINAL; WALL0=$(date +%s%3N)
for r in $(seq 1 "$REQS"); do
  i=$(( (r-1) % NINST )); ar=$(echo $ASPECTS | cut -d' ' -f$(( (r-1) % NASP + 1 ))); pr="${PROMPTS[$(( (r-1) % ${#PROMPTS[@]} ))]}"
  tid=$(post_one $((8000+i)) "$pr" "$OUTDIR/req_${r}_gpu${i}.png" "$ar"); PORTS[$r]="$((8000+i))"; TIDS[$r]="$tid"
done
done_cnt=0; POLL0=$(date +%s)
while [ "$done_cnt" -lt "$REQS" ]; do
  sleep 1; done_cnt=0; sample_mem
  for r in $(seq 1 "$REQS"); do
    if [ -n "${FINAL[$r]:-}" ]; then done_cnt=$((done_cnt+1)); continue; fi
    if [ -z "${TIDS[$r]:-}" ]; then FINAL[$r]="failed"; done_cnt=$((done_cnt+1)); continue; fi
    st=$(status_of "${PORTS[$r]}" "${TIDS[$r]}")
    { [ "$st" = "completed" ] || [ "$st" = "failed" ]; } && { FINAL[$r]="$st"; done_cnt=$((done_cnt+1)); }
  done
  printf "    完成 %s/%s\r" "$done_cnt" "$REQS"
  [ "$(( $(date +%s)-POLL0 ))" -gt 1800 ] && { printf '\n%s轮询超时%s\n' "$R" "$N0"; break; }
done
WALL=$(( $(date +%s%3N)-WALL0 )); printf '\n'; sample_mem

# ---- 汇总: 吞吐(按成功数)+ 内存实测 ----
SUCC=0; FAIL=0
for r in $(seq 1 "$REQS"); do [ "${FINAL[$r]:-failed}" = "completed" ] && SUCC=$((SUCC+1)) || FAIL=$((FAIL+1)); done
echo "============================================="
RC=0
if [ "$SUCC" -eq 0 ]; then printf '%s全部失败(0/%s)— 查日志%s\n' "$R" "$REQS" "$N0"; RC=1
else
  awk -v succ="$SUCC" -v reqs="$REQS" -v wall="$WALL" -v n="$NINST" 'BEGIN{sec=wall/1000;tput=succ/sec;
    printf "  %d 实例 | 成功 %d/%d | 墙钟 %.1fs | 吞吐 %.3f img/s | 单实例 %.3f img/s\n",n,succ,reqs,sec,tput,tput/n}'
  [ "$FAIL" -gt 0 ] && { printf '%s⚠️ %s 张失败, 吞吐按成功数计%s\n' "$R" "$FAIL" "$N0"; RC=1; }
fi
echo "  ---- 内存实测(峰值)----"
tot=0
for i in $(seq 0 $((NINST-1))); do
  mb=${MEMPK[$i]}; tot=$((tot+mb))
  printf "  实例$i(GPU$i): %s MiB (%.2f GB)\n" "$mb" "$(awk "BEGIN{print $mb/1024}")"
done
awk -v t="$tot" -v h="$HOSTPK" 'BEGIN{
  printf "  %d 实例容器内存合计: %d MiB (%.2f GB)\n", '"$NINST"', t, t/1024
  printf "  主机已用峰值(free): %d MiB (%.2f GB) / 256GB\n", h, h/1024
}'
echo "  预热验证: 每实例应 $NASP/$NASP 分辨率(见上方 ✓/✗)"
ok=$(ls "$OUTDIR"/req_*.png 2>/dev/null | wc -l); echo "  产物: $OUTDIR ($ok 张)"
echo "  清理: docker rm -f \$(for i in \$(seq 0 $((NINST-1))); do echo z-inst-\$i; done)"
[ "${KEEP:-0}" = "1" ] || for i in $(seq 0 $((NINST-1))); do docker rm -f "z-inst-$i" >/dev/null 2>&1 || true; done
exit $RC
