#!/usr/bin/env bash
# =============================================================================
# Z-Image 分辨率扫描 + 多提示词压测 —— 单容器, 请求体传 aspect_ratio(每请求覆盖)
# 验证: ① 各 aspect_ratio 是否生效(ffprobe 量实际 W×H) ② 已知 runner 转置 bug(返回(w,h)被按(h,w)解包→横竖反)
#        ③ 不同提示词的稳定性/耗时
#
# 支持比例(runner default_aspect_ratios, 标注的是"请求标签 → 实际输出 W×H 因转置而反"):
#   16:9→928x1664  9:16→1664x928  1:1→1328x1328  4:3→1104x1472  3:4→1472x1104  3:2→1056x1584  2:3→1584x1056
#
# 用法(服务器, 先 scp 到 /data/):
#   bash /data/test_z_image_sweep.sh                       # bf16 单卡, 扫全部 7 比例, 每个换提示词
#   PASSES=2 bash /data/test_z_image_sweep.sh              # 每比例跑 2 遍取均值
#   ASPECTS="16:9 1:1" GPUS=0 bash /data/test_z_image_sweep.sh
# 选填 env: PREC(bf16/int8) GPUS(默认0) ASPECTS PASSES(默认1) STEPS SEED
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
GPUS="${GPUS:-0}"; PREC="${PREC:-bf16}"; PASSES="${PASSES:-1}"; STEPS="${STEPS:-9}"; SEED="${SEED:-42}"
ASPECTS="${ASPECTS:-16:9 9:16 1:1 4:3 3:4 3:2 2:3}"
BF16_PATH=/nfs-data/models/Z-Image-Turbo; INT8_CKPT=/nfs-data/models-int8/Z-Image-Turbo-int8
NP=$(awk -F, '{print NF}' <<<"$GPUS"); API=http://localhost:8000
NAME="z-sweep-${PREC}-${GPUS//,/_}"; OUTDIR="/data/outputs/z_sweep_${PREC}_${GPUS//,/_}"; CFG="/data/cfg_z_sweep_${PREC}.json"
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'
[ "$NP" -gt 1 ] && [ $((30 % NP)) -ne 0 ] && { echo "${R}30 head 不被 $NP 整除${N0}"; exit 2; }
mkdir -p "$OUTDIR"

# 提示词: 传了 PROMPT 就固定用它(阶段②: 变分辨率/同提示词); 不传则轮换下列多样化提示词(阶段③: 变分辨率/变提示词)
if [ -n "${PROMPT:-}" ]; then
  PROMPTS=("$PROMPT")
else
  PROMPTS=(
    "Young Chinese woman in red Hanfu, golden phoenix headdress, intricate embroidery, soft night light"
    "A serene mountain lake at sunrise, mist over water, pine forest, photorealistic landscape"
    "A vintage red sports car parked on a wet city street at night, neon reflections"
    "A wooden sign with the text 'LIGHTX2V' carved in bold letters, rustic background"
    "A majestic Bengal tiger walking through tall golden grass, cinematic wildlife photo"
    "Abstract fluid art, swirling blue and gold paint, macro photography, high detail"
    "A cozy coffee shop interior, warm lighting, plants, books, steam rising from a cup"
    "An astronaut cat floating in space, wearing a tiny helmet, stars and nebula behind"
  )
fi

# 配置(不含 aspect_ratio, 由请求体逐张指定)
{
  echo "{"
  echo "  \"num_channels_latents\": 16, \"infer_steps\": $STEPS, \"attn_type\": \"sage_attn2\","
  echo "  \"enable_cfg\": false, \"sample_guide_scale\": 0.0, \"rope_type\": \"torch\", \"patch_size\": 2"
  [ "$PREC" = "int8" ] && echo "  ,\"dit_quantized\": true, \"dit_quant_scheme\": \"int8-torchao\", \"dit_quantized_ckpt\": \"$INT8_CKPT\""
  [ "$NP" -gt 1 ] && echo "  ,\"parallel\": {\"seq_p_size\": $NP, \"seq_p_attn_type\": \"ulysses\"}"
  echo "}"
} > "$CFG"

echo "${B}###### Z-Image 分辨率扫描 | PREC=$PREC GPUS=$GPUS(NP=$NP) | 比例: $ASPECTS | 每比例×$PASSES ######${N0}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$NP" -gt 1 ]; then RUNCMD="torchrun --nproc_per_node=$NP --master_port=29535 -m lightx2v.server"; SHM="--shm-size=32g"; else RUNCMD="python -m lightx2v.server"; SHM=""; fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --memory=240g --memory-swap=240g $SHM -p 8000:8000 -p 8001:8001 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES="$GPUS" \
  "$IMG" $RUNCMD --model_cls z_image --task t2i --model_path "$BF16_PATH" --config_json "$CFG" \
  --host 0.0.0.0 --port 8000 >/dev/null || { echo "${R}容器启动失败${N0}"; exit 2; }

T0=$(date +%s); code=000
while [ "$(( $(date +%s)-T0 ))" -lt 900 ]; do code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health" 2>/dev/null || echo 000); [ "$code" = "200" ] && break; sleep 5; printf "  加载中 %ss\r" "$(( $(date +%s)-T0 ))"; done; printf '\n'
[ "$code" = "200" ] || { echo "${R}health 超时${N0}"; docker logs --tail 30 "$NAME" 2>&1 | sed 's/^/    /'; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
echo "${G}ready 加载 $(( $(date +%s)-T0 ))s${N0}"

submit_one(){  # $1=aspect $2=prompt $3=outpath  -> 回显 "ms WxH status"
  local ar="$1" pr="$2" out="$3" tid t0 ms st wh
  rm -f "$out"
  local body; body=$(AR="$ar" P="$pr" O="$out" ST="$STEPS" SD="$SEED" python3 -c "import json,os;print(json.dumps({'prompt':os.environ['P'],'negative_prompt':' ','aspect_ratio':os.environ['AR'],'save_result_path':os.environ['O'],'infer_steps':int(os.environ['ST']),'seed':int(os.environ['SD'])}))")
  tid=$(curl -sS -m 30 -X POST "$API/v1/tasks/image/" -H "Content-Type: application/json" -d "$body" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null)
  [ -z "${tid:-}" ] && { echo "0 ? submitfail"; return; }
  t0=$(date +%s%3N)
  while true; do sleep 1; st=$(curl -sS -m 10 "$API/v1/tasks/$tid/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
    [ "$st" = "completed" ] && break; [ "$st" = "failed" ] && { echo "0 ? failed"; return; }
    [ "$(( $(date +%s%3N)-t0 ))" -gt 300000 ] && { echo "0 ? timeout"; return; }; done
  ms=$(( $(date +%s%3N)-t0 ))
  wh=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$out" 2>/dev/null || echo "?")
  echo "$ms $wh completed"
}

# 预热(丢弃)
echo "  预热中..."; submit_one "1:1" "${PROMPTS[0]}" "$OUTDIR/warmup.png" >/dev/null

echo; printf "  %-6s %-12s %-12s %8s\n" "比例" "请求标签" "实际W×H" "耗时"
echo "  ---------------------------------------------"
idx=0; FAIL=0
for ar in $ASPECTS; do
  sumtimes=""; lastwh="?"
  for p in $(seq 1 "$PASSES"); do
    pr="${PROMPTS[$((idx % ${#PROMPTS[@]}))]}"; idx=$((idx+1))
    safe_ar="${ar/:/_}"
    read -r ms wh stt < <(submit_one "$ar" "$pr" "$OUTDIR/${safe_ar}_p${p}.png")
    [ "$stt" != "completed" ] && { printf "  %-6s %-12s %-12s %8s  ${R}%s${N0}\n" "$ar" "$ar" "-" "-" "$stt"; FAIL=$((FAIL+1)); continue; }
    lastwh="$wh"; sumtimes="$sumtimes $ms"
  done
  [ -z "$sumtimes" ] && continue
  avg=$(echo "$sumtimes" | awk '{for(i=1;i<=NF;i++)s+=$i; printf "%.0f", s/NF}')
  printf "  %-6s %-12s %-12s %6sms\n" "$ar" "$ar" "$lastwh" "$avg"
done

echo "  ---------------------------------------------"
echo "  ${B}注:实际 W×H 与请求标签横竖相反 = runner 转置 bug(get_input_target_shape 返回(w,h)被按(h,w)解包)${N0}"
echo "  产物: $OUTDIR"
[ "$FAIL" -gt 0 ] && printf '%s⚠️ 有 %s 次失败, 扫描不完整(检查实例日志)%s\n' "$R" "$FAIL" "$N0"
[ "${KEEP:-0}" = "1" ] || docker rm -f "$NAME" >/dev/null 2>&1 || true
exit $(( FAIL > 0 ? 1 : 0 ))
