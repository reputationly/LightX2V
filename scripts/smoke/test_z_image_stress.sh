#!/usr/bin/env bash
# =============================================================================
# Z-Image 2卡 NUMA / 吞吐压测 —— 单容器复用, 连续提交 N 次, 丢首次预热, 取稳态均值
# 专用于"同 NUMA vs 跨 NUMA"对比(消除冷启动 kernel autotune 干扰, 才能看清 PCIe/QPI 差异)
#
# 本机拓扑(nvidia-smi topo -m): GPU0,1=PHB@NUMA0 / GPU2,3=PHB@NUMA2 / 跨组=SYS
#   同 NUMA:  GPUS="0,1" 或 "2,3"
#   跨 NUMA:  GPUS="0,2"(或 0,3/1,2/1,3)
#
# 用法(服务器, 先 scp 本脚本到 /data/):
#   GPUS="0,1" bash /data/test_z_image_stress.sh      # 同 NUMA 基线
#   GPUS="0,2" bash /data/test_z_image_stress.sh      # 跨 NUMA
#   PREC=int8 N=8 GPUS="0,1" bash /data/test_z_image_stress.sh
#
# 选填 env: GPUS(必看, 默认0,1) PREC(bf16/int8) N(出图次数, 默认6) STEPS ASPECT ATTN SEED
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
GPUS="${GPUS:-0,1}"; PREC="${PREC:-bf16}"; N="${N:-6}"
STEPS="${STEPS:-9}"; ASPECT="${ASPECT:-16:9}"; ATTN="${ATTN:-sage_attn2}"; SEED="${SEED:-42}"
PROMPT="${PROMPT:-Young Chinese woman in red Hanfu, intricate embroidery. Elaborate golden phoenix headdress, red flowers, beads. Holds round folding fan. Neon lightning-bolt lamp, bright yellow glow. Soft-lit outdoor night, silhouetted pagoda, blurred colorful lights.}"
BF16_PATH=/nfs-data/models/Z-Image-Turbo
INT8_CKPT=/nfs-data/models-int8/Z-Image-Turbo-int8
NP=$(awk -F, '{print NF}' <<<"$GPUS")        # 卡数 = GPUS 里逗号分隔的个数
API=http://localhost:8000
NAME="z-stress-${PREC}-${GPUS//,/_}"
OUTDIR="/data/outputs/z_stress_${PREC}_${GPUS//,/_}"; CFG="/data/cfg_z_stress_${PREC}_${GPUS//,/_}.json"
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'
[ "$NP" -gt 1 ] && [ $((30 % NP)) -ne 0 ] && { echo "${R}z_image 30 head 不被 $NP 整除, ulysses 不可用(用 2/3/5/6)${N0}"; exit 2; }
mkdir -p "$OUTDIR"

# ---- 生成配置(bf16/int8, NP>1 加 ulysses)----
{
  echo "{"
  echo "  \"aspect_ratio\": \"$ASPECT\", \"num_channels_latents\": 16, \"infer_steps\": $STEPS,"
  echo "  \"attn_type\": \"$ATTN\", \"enable_cfg\": false, \"sample_guide_scale\": 0.0,"
  echo "  \"rope_type\": \"torch\", \"patch_size\": 2"
  [ "${PREC}" = "int8" ] && echo "  ,\"dit_quantized\": true, \"dit_quant_scheme\": \"int8-torchao\", \"dit_quantized_ckpt\": \"$INT8_CKPT\""
  [ "$NP" -gt 1 ] && echo "  ,\"parallel\": {\"seq_p_size\": $NP, \"seq_p_attn_type\": \"ulysses\"}"
  echo "}"
} > "$CFG"

echo "${B}###### Z-Image 压测 | PREC=$PREC | GPUS=$GPUS (NP=$NP) | N=$N | steps=$STEPS ######${N0}"
echo "  config: $CFG"
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$NP" -gt 1 ]; then RUNCMD="torchrun --nproc_per_node=$NP --master_port=29534 -m lightx2v.server"; SHM="--shm-size=32g"; else RUNCMD="python -m lightx2v.server"; SHM=""; fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --memory=240g --memory-swap=240g $SHM -p 8000:8000 -p 8001:8001 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES="$GPUS" \
  "$IMG" $RUNCMD --model_cls z_image --task t2i --model_path "$BF16_PATH" --config_json "$CFG" \
  --host 0.0.0.0 --port 8000 >/dev/null || { echo "${R}容器启动失败${N0}"; exit 2; }

# 等 health
T0=$(date +%s); code=000
while [ "$(( $(date +%s)-T0 ))" -lt 900 ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$API/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break; sleep 5; printf "  加载中 %ss\r" "$(( $(date +%s)-T0 ))"
done; printf '\n'
[ "$code" = "200" ] || { echo "${R}health 超时${N0}"; docker logs --tail 30 "$NAME" 2>&1 | sed 's/^/    /'; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
echo "${G}ready 加载 $(( $(date +%s)-T0 ))s${N0}, 连续出 $N 张(第1张预热, 不计入均值)..."

# ---- 连续提交 N 次, 每次测生成 ms + GPU/显存峰值 ----
TIMES=()
for i in $(seq 1 "$N"); do
  OUT="$OUTDIR/iter_${i}.png"; rm -f "$OUT"
  BODY=$(P="$PROMPT" O="$OUT" ST="$STEPS" SD="$SEED" python3 -c "import json,os;print(json.dumps({'prompt':os.environ['P'],'negative_prompt':' ','save_result_path':os.environ['O'],'infer_steps':int(os.environ['ST']),'seed':int(os.environ['SD'])}))")
  TID=$(curl -sS -m 30 -X POST "$API/v1/tasks/image/" -H "Content-Type: application/json" -d "$BODY" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])" 2>/dev/null)
  [ -z "${TID:-}" ] && { echo "  ${R}iter$i 提交失败${N0}"; continue; }
  t0=$(date +%s%3N); UP=0; ST=""
  while true; do
    sleep 1
    UU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
    [ "${UU:-0}" -gt "$UP" ] 2>/dev/null && UP=$UU
    ST=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
    [ "$ST" = "completed" ] && break
    [ "$ST" = "failed" ] && { echo "  ${R}iter$i 失败${N0}"; break; }
    [ "$(( $(date +%s%3N)-t0 ))" -gt 300000 ] && { echo "  ${R}iter$i 超时${N0}"; break; }
  done
  [ "$ST" != "completed" ] && continue
  ms=$(( $(date +%s%3N)-t0 )); SZ=$(( $(stat -c%s "$OUT" 2>/dev/null || echo 0)/1024 ))
  tag=""; [ "$i" = "1" ] && tag=" (预热, 不计)"
  printf "  iter%-2s 生成 %5sms (%.1fs) | GPU利用峰 %s%% | %sKB%s\n" "$i" "$ms" "$(awk "BEGIN{print $ms/1000}")" "$UP" "$SZ" "$tag"
  [ "$i" != "1" ] && TIMES+=("$ms")
done

# ---- 汇总: 稳态(去预热)均值/中位/最小 ----
echo "============================================="
if [ "${#TIMES[@]}" -gt 0 ]; then
  printf '%s\n' "${TIMES[@]}" | sort -n | awk -v g="$GPUS" -v p="$PREC" -v np="$NP" '
    {a[NR]=$1; s+=$1}
    END{n=NR; mean=s/n; med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
      printf "  [PREC=%s GPUS=%s NP=%s] 稳态 %d 张: 均值 %.0fms(%.2fs) | 中位 %.0fms | 最小 %.0fms | 最大 %.0fms\n", p,g,np,n,mean,mean/1000,med,a[1],a[n]}'
  RC=0
else
  echo "  ${R}无有效样本(全部提交失败/超时/failed)— 检查实例日志${N0}"; RC=1
fi
echo "  产物: $OUTDIR"
echo "  对比: 同 NUMA(GPUS=0,1) vs 跨 NUMA(GPUS=0,2) 的稳态均值之差 = QPI/UPI 跨 NUMA 惩罚"
# 默认清理容器(同 8000 端口, 不清下一次会冲突); KEEP=1 保留调试
[ "${KEEP:-0}" = "1" ] || docker rm -f "$NAME" >/dev/null 2>&1 || true
exit $RC
