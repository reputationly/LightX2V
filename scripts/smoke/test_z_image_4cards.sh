#!/usr/bin/env bash
# =============================================================================
# Z-Image 4 单卡实例 并发吞吐压测 —— 每卡一个实例(CUDA_VISIBLE_DEVICES=i, 端口 8000+i)
# 验证生产假设: "4×单卡实例" 吞吐 >> "2×双卡实例"(因 ulysses 单图只 1.2×, 占卡不划算)
#   预期: 4 实例 ≈ 4/7.64s ≈ 0.52 img/s ; 而 2×双卡 ≈ 2/6.3s ≈ 0.32 img/s
#
# 用法(服务器, 先 scp 到 /data/):
#   bash /data/test_z_image_4cards.sh                  # 4 实例 bf16, 共发 16 张
#   REQS=32 bash /data/test_z_image_4cards.sh
# 选填 env: PREC(bf16/int8) NINST(实例数, 默认4) REQS(总请求, 默认16) STEPS ASPECT SEED
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
PREC="${PREC:-bf16}"; NINST="${NINST:-4}"; REQS="${REQS:-16}"; STEPS="${STEPS:-9}"; ASPECT="${ASPECT:-16:9}"; SEED="${SEED:-42}"
BF16_PATH=/nfs-data/models/Z-Image-Turbo; INT8_CKPT=/nfs-data/models-int8/Z-Image-Turbo-int8
OUTDIR="/data/outputs/z_4cards_${PREC}"; CFG="/data/cfg_z_4cards_${PREC}.json"
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'
mkdir -p "$OUTDIR"
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

# 配置(单卡, 无 parallel)
{
  echo "{"
  echo "  \"aspect_ratio\": \"$ASPECT\", \"num_channels_latents\": 16, \"infer_steps\": $STEPS,"
  echo "  \"attn_type\": \"sage_attn2\", \"enable_cfg\": false, \"sample_guide_scale\": 0.0,"
  echo "  \"rope_type\": \"torch\", \"patch_size\": 2"
  [ "$PREC" = "int8" ] && echo "  ,\"dit_quantized\": true, \"dit_quant_scheme\": \"int8-torchao\", \"dit_quantized_ckpt\": \"$INT8_CKPT\""
  echo "}"
} > "$CFG"

echo "${B}###### Z-Image $NINST 单卡实例吞吐 | PREC=$PREC | 共 $REQS 张 ######${N0}"
# ---- 起 N 个单卡实例(每卡一个, host 端口 8000+i, 只映射 API 口, 不映 8001 metrics)----
for i in $(seq 0 $((NINST-1))); do
  nm="z-inst-$i"; port=$((8000+i))
  docker rm -f "$nm" >/dev/null 2>&1 || true
  docker run -d --name "$nm" --gpus all --memory=60g --memory-swap=60g -p "$port":8000 \
    -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES="$i" \
    "$IMG" python -m lightx2v.server --model_cls z_image --task t2i --model_path "$BF16_PATH" \
    --config_json "$CFG" --host 0.0.0.0 --port 8000 >/dev/null \
    && echo "  起 $nm (GPU$i, 端口$port)" || echo "  ${R}$nm 启动失败${N0}"
done

# ---- 等全部 health ----
echo "  等 $NINST 个实例就绪..."
T0=$(date +%s)
for i in $(seq 0 $((NINST-1))); do
  port=$((8000+i)); ok=0
  while [ "$(( $(date +%s)-T0 ))" -lt 900 ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" 2>/dev/null || echo 000)" = "200" ] && { ok=1; break; }
    sleep 3
  done
  [ "$ok" = "1" ] && echo "  ${G}实例$i 就绪${N0}" || { echo "  ${R}实例$i health 超时${N0}"; docker logs --tail 20 "z-inst-$i" 2>&1 | sed 's/^/    /'; }
done
echo "  全部就绪, 加载 $(( $(date +%s)-T0 ))s"

post_one(){ # $1=port $2=prompt $3=out -> 回显 task_id
  local body; body=$(P="$2" O="$3" AR="$ASPECT" ST="$STEPS" SD="$SEED" python3 -c "import json,os;print(json.dumps({'prompt':os.environ['P'],'negative_prompt':' ','aspect_ratio':os.environ['AR'],'save_result_path':os.environ['O'],'infer_steps':int(os.environ['ST']),'seed':int(os.environ['SD'])}))")
  curl -sS -m 30 -X POST "http://localhost:$1/v1/tasks/image/" -H "Content-Type: application/json" -d "$body" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null
}

# ---- 预热每个实例各 1 张(不计入;带超时, 空tid/failed 跳过不阻塞)----
echo "  预热 $NINST 实例..."
for i in $(seq 0 $((NINST-1))); do
  tid=$(post_one $((8000+i)) "warmup" "$OUTDIR/warmup_$i.png")
  if [ -z "$tid" ]; then echo "  ${R}实例$i 预热提交失败(空 tid), 跳过${N0}"; continue; fi
  w0=$(date +%s)
  while true; do
    wst=$(curl -sS -m 10 "http://localhost:$((8000+i))/v1/tasks/$tid/status" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status") or "")' 2>/dev/null)
    [ "$wst" = "completed" ] && break
    [ "$wst" = "failed" ] && { echo "  ${R}实例$i 预热 failed${N0}"; break; }
    [ "$(( $(date +%s)-w0 ))" -gt 300 ] && { echo "  ${R}实例$i 预热超时(300s)${N0}"; break; }
    sleep 1
  done
done

# ---- 并发发 REQS 张(轮询分配到 N 实例), 测墙钟时间 ----
echo "  ${B}开始并发: $REQS 张分发到 $NINST 实例...${N0}"
declare -a PORTS TIDS
WALL0=$(date +%s%3N)
for r in $(seq 1 "$REQS"); do
  i=$(( (r-1) % NINST )); port=$((8000+i)); pr="${PROMPTS[$(( (r-1) % ${#PROMPTS[@]} ))]}"
  tid=$(post_one "$port" "$pr" "$OUTDIR/req_${r}_gpu${i}.png")
  PORTS[$r]="$port"; TIDS[$r]="$tid"
done
# 轮询直到全部 finalize(completed/failed/空tid→failed), 记录每请求最终态; 全局超时防卡死
declare -a FINAL; done_cnt=0; POLL0=$(date +%s)   # 索引数组(下标为整数 1..REQS), 不依赖 bash4 关联数组
while [ "$done_cnt" -lt "$REQS" ]; do
  sleep 1; done_cnt=0
  for r in $(seq 1 "$REQS"); do
    if [ -n "${FINAL[$r]:-}" ]; then done_cnt=$((done_cnt+1)); continue; fi          # 已定型, 不再查
    if [ -z "${TIDS[$r]:-}" ]; then FINAL[$r]="failed"; done_cnt=$((done_cnt+1)); continue; fi  # 空tid=提交失败
    st=$(curl -sS -m 10 "http://localhost:${PORTS[$r]}/v1/tasks/${TIDS[$r]}/status" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status") or "")' 2>/dev/null)
    { [ "$st" = "completed" ] || [ "$st" = "failed" ]; } && { FINAL[$r]="$st"; done_cnt=$((done_cnt+1)); }
  done
  printf "    完成 %s/%s\r" "$done_cnt" "$REQS"
  [ "$(( $(date +%s)-POLL0 ))" -gt 1800 ] && { printf '\n%s轮询超 1800s, 放弃(未定型按失败计)%s\n' "$R" "$N0"; break; }
done
WALL=$(( $(date +%s%3N)-WALL0 )); printf '\n'

# ---- 统计成功/失败, 吞吐按【成功数】算(失败请求不计入分子, 防虚高)----
SUCC=0; FAIL=0
for r in $(seq 1 "$REQS"); do [ "${FINAL[$r]:-failed}" = "completed" ] && SUCC=$((SUCC+1)) || FAIL=$((FAIL+1)); done
echo "============================================="
RC=0
if [ "$SUCC" -eq 0 ]; then
  printf '%s全部失败(0/%s 成功), 无有效吞吐 — 检查实例日志%s\n' "$R" "$REQS" "$N0"; RC=1
else
  awk -v succ="$SUCC" -v reqs="$REQS" -v wall="$WALL" -v n="$NINST" 'BEGIN{
    sec=wall/1000; tput=succ/sec;
    printf "  %d 实例 | 成功 %d/%d | 墙钟 %.1fs | 吞吐 %.3f img/s | 单实例 %.3f img/s\n", n, succ, reqs, sec, tput, tput/n
  }'
  [ "$FAIL" -gt 0 ] && { printf '%s⚠️ 有 %s 张失败, 吞吐已按成功数计, 但数据慎用(查实例日志)%s\n' "$R" "$FAIL" "$N0"; RC=1; }
fi
ok=$(ls "$OUTDIR"/req_*.png 2>/dev/null | wc -l)
echo "  产物: $OUTDIR ($ok 张)"
echo "  清理: docker rm -f \$(for i in \$(seq 0 $((NINST-1))); do echo z-inst-\$i; done)"
[ "${KEEP:-0}" = "1" ] || for i in $(seq 0 $((NINST-1))); do docker rm -f "z-inst-$i" >/dev/null 2>&1 || true; done
exit $RC
