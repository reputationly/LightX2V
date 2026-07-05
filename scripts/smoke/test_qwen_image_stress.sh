#!/usr/bin/env bash
# =============================================================================
# Qwen-Image 热态稳态压测 —— 单容器复用, 连发 N 张, 丢首张预热, 取稳态均值 + GPU util 峰 + 显存峰
# 仿 test_z_image_stress.sh。复用已验证的配置文件(attn_type=torch_sdpa 等)作源,按参数二次注入
# 并行 / offload,避免手抄模板。
#
# 关键对照(Qwen base 58G 单卡 40G 必须 offload,offload 搬 ~38G/前向 = I/O 瓶颈):
#   单卡 offload  : GPUS=0 OFFLOAD=1            (基线;看 GPU util 是否偏低 = 证 I/O-bound)
#   多卡 ulysses  : GPUS=0,1,2,3 PTYPE=ulysses  (权重仍复制 → 仍需 offload;只切序列计算)
#   多卡 TP 去off : GPUS=0,1,2,3 PTYPE=tp OFFLOAD=0 (权重分片 → 或可去 offload → 可能大提速)
#
# 用法(服务器, 先 scp 到 /data/):
#   MODE=base    GPUS=0        bash /data/test_qwen_image_stress.sh
#   MODE=merged8 GPUS=0        bash /data/test_qwen_image_stress.sh
#   MODE=base    GPUS=0,1,2,3 PTYPE=ulysses bash /data/test_qwen_image_stress.sh
#   MODE=base    GPUS=0,1,2,3 PTYPE=tp OFFLOAD=0 bash /data/test_qwen_image_stress.sh
#
# env: MODE(base|merged8|edit) GPUS(默认0) OFFLOAD(1/0) PTYPE(ulysses|tp) N(默认6) ASPECT(默认1:1)
#      SEED STEPS IMAGE(i2i 输入图) KEEP(1=不清容器) LX_IMG(镜像)
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest}"
GPUS="${GPUS:-0}"; MODE="${MODE:-base}"; N="${N:-6}"; SEED="${SEED:-42}"; ASPECT="${ASPECT:-1:1}"
OFFLOAD="${OFFLOAD:-1}"; PTYPE="${PTYPE:-ulysses}"
QUANT_CKPT="${QUANT_CKPT:-}"; SCHEME="${SCHEME:-int8-torchao}"; STEPS="${STEPS:-}"
PROMPT="${PROMPT:-a cozy bookstore cafe with warm lighting, photorealistic, highly detailed}"
CFGDIR=/data/lightx2v_configs; MODEL_BASE=/data/models/Qwen-Image-2512
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N0=$'\e[0m'

# MODE: base/merged8/edit=Qwen-Image-2512; bf16orig/int8orig=原版 Qwen-Image(用于干净的 int8 消融)
case "$MODE" in
  base)     SRC=$CFGDIR/qwen_2512_a100_base.json;             MP=$MODEL_BASE;                      TASK=t2i;;
  merged8)  SRC=$CFGDIR/qwen_2512_a100_lightning_merged.json; MP=$MODEL_BASE;                      TASK=t2i;;
  edit)     SRC=$CFGDIR/qwen_edit_2511_a100_base.json;             MP=/data/models/Qwen-Image-Edit-2511; TASK=i2i;;
  edit_m)   SRC=$CFGDIR/qwen_edit_2511_a100_lightning_merged.json; MP=/data/models/Qwen-Image-Edit-2511; TASK=i2i;;
  edit_int8) SRC=$CFGDIR/qwen_edit_2511_a100_int8_offload.json;      MP=/data/models/Qwen-Image-Edit-2511; TASK=i2i;;
  bf16orig) SRC=$CFGDIR/qwen_2512_a100_base.json;             MP=/data/models/Qwen-Image;          TASK=t2i;;
  int8orig) SRC=$CFGDIR/qwen_2512_a100_base.json;             MP=/data/models/Qwen-Image;          TASK=t2i
            QUANT_CKPT="${QUANT_CKPT:-/data/models-int8/Qwen-Image-int8}";;
  *) echo "${R}MODE ∈ base|merged8|edit|edit_m|edit_int8|bf16orig|int8orig${N0}"; exit 2;;
esac
[ -f "$SRC" ] || { echo "${R}配置不存在: $SRC${N0}"; exit 2; }
NP=$(awk -F, '{print NF}' <<<"$GPUS")
[ "$NP" -gt 1 ] && [ $((24 % NP)) -ne 0 ] && { echo "${R}Qwen 24 head 不整除 $NP(用 2/3/4/6/8)${N0}"; exit 2; }
case "$MODE" in edit|edit_m|edit_int8) [ -z "${IMAGE:-}" ] && { echo "${R}MODE=$MODE 需 IMAGE=<输入图路径>${N0}"; exit 2; };; esac

API=http://localhost:8000
TAG="${MODE}-${GPUS//,/_}-off${OFFLOAD}-${PTYPE}"
NAME="qwen-stress-$TAG"; OUTDIR="/data/outputs/qwen_stress_$TAG"; CFG="/data/cfg_qwen_stress_$TAG.json"
mkdir -p "$OUTDIR"

# ---- 从源配置二次注入 aspect / parallel / offload ----
SRC="$SRC" DST="$CFG" ASPECT="$ASPECT" NP="$NP" OFFLOAD="$OFFLOAD" PTYPE="$PTYPE" QCKPT="$QUANT_CKPT" SCHEME="$SCHEME" STEPS="$STEPS" python3 - <<'PY'
import json, os
c = json.load(open(os.environ["SRC"]))
c["aspect_ratio"] = os.environ["ASPECT"]
if os.environ.get("STEPS"):
    c["infer_steps"] = int(os.environ["STEPS"])
np = int(os.environ["NP"]); off = os.environ["OFFLOAD"] == "1"
c["cpu_offload"] = off
if off:
    c.setdefault("offload_granularity", "block")
else:
    c.pop("offload_granularity", None)
if np > 1:
    c["parallel"] = {"tensor_p_size": np} if os.environ["PTYPE"] == "tp" else {"seq_p_size": np, "seq_p_attn_type": "ulysses"}
else:
    c.pop("parallel", None)
qckpt = os.environ.get("QCKPT", "")
if qckpt:
    c["dit_quantized"] = True
    c["dit_quant_scheme"] = os.environ["SCHEME"]
    c["dit_quantized_ckpt"] = qckpt
# 注:QCKPT 空时不删源配置自带的 quant 字段(edit_int8 等靠配置文件带 int8)
json.dump(c, open(os.environ["DST"], "w"), ensure_ascii=False, indent=2)
print("[cfg]", os.environ["DST"], "| parallel=", c.get("parallel"), "| cpu_offload=", c["cpu_offload"], "| quant=", c.get("dit_quant_scheme"))
PY

echo "${B}###### Qwen 压测 | MODE=$MODE | GPUS=$GPUS(NP=$NP) | offload=$OFFLOAD | ptype=$PTYPE | N=$N | aspect=$ASPECT ######${N0}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$NP" -gt 1 ]; then RUN="torchrun --nproc_per_node=$NP --master_port=29541 -m lightx2v.server"; SHM="--shm-size=32g"; else RUN="python -m lightx2v.server"; SHM=""; fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --init --memory=240g $SHM -p 8000:8000 -p 8001:8001 \
  -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES="$GPUS" \
  "$IMG" $RUN --model_cls qwen_image --task "$TASK" --model_path "$MP" --config_json "$CFG" \
  --host 0.0.0.0 --port 8000 >/dev/null || { echo "${R}容器启动失败${N0}"; exit 2; }

# ---- 等 ready(qwen 无 /health,用 queue/status)----
T0=$(date +%s); code=000
while [ "$(( $(date +%s)-T0 ))" -lt "${READY_TO:-1200}" ]; do
  # 容器崩溃检测:不再傻等已死的服务
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "${R}容器已退出(崩溃/OOM),末 40 行日志:${N0}"; docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /'
    docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "$API/v1/tasks/queue/status" 2>/dev/null || echo 000)
  [ "$code" = 200 ] && break; sleep 5; printf "  加载中 %ss\r" "$(( $(date +%s)-T0 ))"
done; printf '\n'
[ "$code" = 200 ] || { echo "${R}ready 超时${N0}"; docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /'; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
echo "${G}ready 加载 $(( $(date +%s)-T0 ))s${N0}, 连出 $N 张(第1张预热不计)..."

# ---- 连发 N 次, 每次测生成 ms + GPU util 峰 + 显存峰 ----
TIMES=(); UPEAK=0; MPEAK=0
for i in $(seq 1 "$N"); do
  OUT="$OUTDIR/iter_$i.png"; rm -f "$OUT"
  BODY=$(P="$PROMPT" O="$OUT" A="$ASPECT" SD="$SEED" IMG_IN="${IMAGE:-}" python3 -c '
import json,os
d={"prompt":os.environ["P"],"save_result_path":os.environ["O"],"aspect_ratio":os.environ["A"],"seed":int(os.environ["SD"])}
if os.environ.get("IMG_IN"): d["image_path"]=os.environ["IMG_IN"]
print(json.dumps(d))')
  TID=$(curl -sS -m 30 -X POST "$API/v1/tasks/image/" -H 'Content-Type: application/json' -d "$BODY" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("task_id",""))' 2>/dev/null)
  [ -z "${TID:-}" ] && { echo "  ${R}iter$i 提交失败${N0}"; continue; }
  t0=$(date +%s%3N); UP=0; MU=0; ST=""
  while true; do
    sleep 1
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
    m=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
    [ "${u:-0}" -gt "$UP" ] 2>/dev/null && UP=$u
    [ "${m:-0}" -gt "$MU" ] 2>/dev/null && MU=$m
    ST=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status") or "")' 2>/dev/null)
    [ "$ST" = completed ] && break
    [ "$ST" = failed ] && { echo "  ${R}iter$i failed${N0}"; docker logs --tail 15 "$NAME" 2>&1 | sed 's/^/      /'; break; }
    [ "$(( $(date +%s%3N)-t0 ))" -gt 600000 ] && { echo "  ${R}iter$i 超时${N0}"; break; }
  done
  [ "$ST" != completed ] && continue
  ms=$(( $(date +%s%3N)-t0 )); SZ=$(( $(stat -c%s "$OUT" 2>/dev/null || echo 0)/1024 ))
  [ "$UP" -gt "$UPEAK" ] && UPEAK=$UP; [ "$MU" -gt "$MPEAK" ] && MPEAK=$MU
  tag=""; [ "$i" = 1 ] && tag=" (预热,不计)"
  bl=""; [ "$SZ" -lt 50 ] && bl=" ${R}⚠️<50KB疑黑图${N0}"
  printf "  iter%-2s %6sms (%.1fs) | GPU峰 %s%% | 显存峰 %sMiB | %sKB%s%s\n" "$i" "$ms" "$(awk "BEGIN{print $ms/1000}")" "$UP" "$MU" "$SZ" "$tag" "$bl"
  [ "$i" != 1 ] && TIMES+=("$ms")
done

# ---- 汇总 ----
echo "============================================="
if [ "${#TIMES[@]}" -gt 0 ]; then
  printf '%s\n' "${TIMES[@]}" | sort -n | awk -v t="$TAG" -v up="$UPEAK" -v mp="$MPEAK" '
    {a[NR]=$1; s+=$1}
    END{n=NR; mean=s/n; med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
      printf "  [%s] 稳态 %d 张: 均值 %.2fs | 中位 %.2fs | 最小 %.2fs | 最大 %.2fs | GPU util峰 %s%% | 显存峰 %sMiB\n",
        t,n,mean/1000,med/1000,a[1]/1000,a[n]/1000,up,mp}'
  RC=0
else
  echo "  ${R}无有效样本 — 看容器日志${N0}"; RC=1
fi
echo "  产物: $OUTDIR"
[ "${KEEP:-0}" = "1" ] || docker rm -f "$NAME" >/dev/null 2>&1 || true
exit $RC
