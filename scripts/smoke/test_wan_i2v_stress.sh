#!/usr/bin/env bash
# =============================================================================
# Wan2.2-I2V int8 4卡 —— 时长压力测试 + i2v vs flf2v 对比(单容器复用)
#
# 对每个 TASK(i2v 单图 / flf2v 首尾帧):起一次 int8 4卡 server, 循环不同
# target_video_length 提交, 记录 生成耗时 / GPU峰值 / 容器CPU内存峰值 / CPU%峰值。
# 末尾出对比表, 看首尾帧比单图多花多少 CPU / 时间。
#   时长 = 帧数 / 16fps; 帧数须 = 4n+1。  81→5s 121→7.5s 161→10s 201→12.5s 241→15s
#
# 用法(服务器, 仅需本脚本):
#   bash /data/test_wan_i2v_stress.sh                    # 480p, i2v+flf2v, 扫81~241
#   RES=720 bash /data/test_wan_i2v_stress.sh            # 720p
#   TASKS=flf2v FRAMES_LIST="81 161" bash /data/test_wan_i2v_stress.sh
# =============================================================================
set -uo pipefail
RES="${RES:-480}"
TASKS="${TASKS:-i2v flf2v}"
FRAMES_LIST="${FRAMES_LIST:-81 121 161 201 241}"
SEED="${SEED:-504166}"
# 默认用蓝鸟首/尾帧, 两种任务同首帧 -> 公平对比(i2v 忽略尾帧)
IMAGE="${IMAGE:-/opt/LightX2V/assets/inputs/imgs/flf2v_input_first_frame-fs8.png}"
LAST_FRAME="${LAST_FRAME:-/opt/LightX2V/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}"
PROMPT="${PROMPT:-CG animation style, a small blue bird takes off from the ground, flapping its wings, blue sky with white clouds, bright sunshine, low-angle close-up, cinematic.}"
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
NFS=/nfs-data/models; NFS8=/nfs-data/models-int8
BASE="$NFS/Wan-AI/Wan2.2-T2V-A14B"
INT8="$NFS8/Wan2.2-I2V-720p-int8"
CFGDIR=/data/lightx2v_configs; API=http://localhost:8000
mkdir -p "$CFGDIR"
RSZ=""; [ "$RES" = "720" ] && RSZ=null

# ---- int8 4卡配置(i2v/flf2v 共用, task 在起容器时指定)----
CFG="$CFGDIR/wan_stress_int8_ul4.json"
python3 - "$CFG" "$INT8" <<'PY'
import json,sys
cfg,int8=sys.argv[1:3]
c={"infer_steps":4,"target_video_length":81,"text_len":512,
   "target_height":720,"target_width":1280,
   "self_attn_1_type":"sage_attn2","cross_attn_1_type":"sage_attn2","cross_attn_2_type":"sage_attn2",
   "sample_guide_scale":[3.5,3.5],"sample_shift":5.0,"enable_cfg":False,
   "cpu_offload":False,"t5_cpu_offload":True,"vae_cpu_offload":False,
   "use_image_encoder":False,"boundary_step_index":2,
   "denoising_step_list":[1000,750,500,250],"rope_type":"torch",
   "dit_quantized":True,"dit_quant_scheme":"int8-torchao",
   "high_noise_quantized_ckpt":f"{int8}/high_noise",
   "low_noise_quantized_ckpt":f"{int8}/low_noise",
   "parallel":{"seq_p_size":4,"seq_p_attn_type":"ulysses"}}
json.dump(c,open(cfg,"w"),indent=2); print(cfg)
PY

peak_gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0; }
cpu_mem_mib(){ docker stats --no-stream --format '{{.MemUsage}}' "$1" 2>/dev/null | awk '{u=$1; if(u~/GiB/){sub(/GiB/,"",u);print int(u*1024)} else if(u~/MiB/){sub(/MiB/,"",u);print int(u)} else print 0}'; }
cpu_perc(){ docker stats --no-stream --format '{{.CPUPerc}}' "$1" 2>/dev/null | tr -d '%' | awk '{print int($1)}'; }

SUMMARY=""
for TASK in $TASKS; do
  NAME="wan-stress-$TASK"; OUT="/data/outputs/wan_stress_${TASK}_${RES}p"; mkdir -p "$OUT"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo; echo "=========================================================="
  echo " TASK=$TASK | RES=${RES}p | 起 int8 4卡 server(只起一次)"
  echo "=========================================================="
  docker run -d --name "$NAME" --gpus all --memory="${MEM:-240g}" --memory-swap="${MEM:-240g}" --shm-size=32g \
    -p 8000:8000 -p 8001:8001 -v /data:/data -v /nfs-data:/nfs-data \
    -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$IMG" torchrun --nproc_per_node=4 --master_port=29533 -m lightx2v.server \
    --model_cls wan2.2_moe_distill --task "$TASK" --model_path "$BASE" --config_json "$CFG" \
    --host 0.0.0.0 --port 8000 >/dev/null || { echo "[$TASK] 容器启动失败"; continue; }

  # 等 health
  T0=$(date +%s); ok=0
  while :; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API/health" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { ok=1; break; }
    el=$(( $(date +%s)-T0 )); [ "$el" -gt 600 ] && { echo "[$TASK] 600s 未就绪"; docker logs --tail 20 "$NAME"; break; }
    printf '\r  加载中 %ss (health=%s)' "$el" "$code"; sleep 5
  done
  [ "$ok" != 1 ] && { docker rm -f "$NAME" >/dev/null 2>&1; continue; }
  echo; echo "[$(date +%T)] [$TASK] ready 加载 $(( $(date +%s)-T0 ))s"

  for f in $FRAMES_LIST; do
    if [ $(( (f - 1) % 4 )) -ne 0 ]; then echo "!! FRAMES=$f 非4n+1, 跳过"; continue; fi
    dur=$(python3 -c "print(round($f/16,2))")
    out="$OUT/f${f}_s${SEED}.mp4"; rm -f "$out"
    echo; echo "######## [$TASK] FRAMES=$f (~${dur}s) RES=${RES}p ########"
    BODY=$(P="$PROMPT" O="$out" NF="$f" SD="$SEED" IMG_P="$IMAGE" LF="$LAST_FRAME" TK="$TASK" RSZ="$RSZ" python3 -c '
import json,os
d={"prompt":os.environ["P"],"negative_prompt":"","save_result_path":os.environ["O"],
   "target_video_length":int(os.environ["NF"]),"infer_steps":4,"seed":int(os.environ["SD"]),
   "image_path":os.environ["IMG_P"]}
if os.environ["TK"]=="flf2v" and os.environ.get("LF"): d["last_frame_path"]=os.environ["LF"]
if os.environ.get("RSZ","").lower() in ("null","none"): d["resize_mode"]=None
print(json.dumps(d))')
    RESP=$(curl -sS -m 30 -X POST "$API/v1/tasks/video/" -H "Content-Type: application/json" -d "$BODY")
    TID=$(echo "$RESP" | python3 -c "import json,sys;print(json.load(sys.stdin).get('task_id',''))" 2>/dev/null)
    [ -z "$TID" ] && { echo "  提交失败: $RESP"; SUMMARY="$SUMMARY\n  [$TASK] ${f}帧(~${dur}s): ❌ 提交失败"; continue; }
    G0=$(date +%s); PGPU=0; PCPU=0; PPCT=0; ST=""
    while :; do
      sleep 5; EL=$(( $(date +%s)-G0 ))
      g=$(peak_gpu); [ "$g" -gt "$PGPU" ] && PGPU=$g
      cm=$(cpu_mem_mib "$NAME"); [ "${cm:-0}" -gt "$PCPU" ] && PCPU=$cm
      cp=$(cpu_perc "$NAME"); [ "${cp:-0}" -gt "$PPCT" ] && PPCT=$cp
      HC=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API/health" 2>/dev/null || echo 000)
      ST=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
      echo "  t=${EL}s status=${ST:-?} gpu=${PGPU}MiB cpu=${PCPU}MiB cpu%=${PPCT} health=${HC}"
      case "$ST" in
        completed) break ;;
        failed) ERR=$(curl -sS "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print((json.load(sys.stdin).get('error') or '')[:100])" 2>/dev/null); echo "  ERR: $ERR"; break ;;
        "") [ "$HC" != "200" ] && { echo "  server 不响应(可能OOM杀进程)"; ST=failed; break; } ;;
      esac
      [ "$EL" -gt 1200 ] && { echo "  超1200s放弃"; ST=timeout; break; }
    done
    GEN=$(( $(date +%s)-G0 ))
    if [ "$ST" = "completed" ] && [ -s "$out" ]; then
      res=$(command -v ffprobe >/dev/null && ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$out" 2>/dev/null)
      echo "  ✅ 生成${GEN}s GPU峰值${PGPU}MiB CPU峰值${PCPU}MiB CPU%峰值${PPCT} ${res}"
      SUMMARY="$SUMMARY\n  [$TASK] ${f}帧(~${dur}s): ✅ 生成${GEN}s GPU${PGPU} CPU${PCPU}MiB CPU%${PPCT} ${res}"
    else
      echo "  ❌ 失败(status=$ST) GPU峰值${PGPU}MiB —— 大概率 OOM"
      SUMMARY="$SUMMARY\n  [$TASK] ${f}帧(~${dur}s): ❌ ${ST} GPU${PGPU}MiB"
      HC=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API/health" 2>/dev/null || echo 000)
      [ "$HC" != "200" ] && { echo "!! [$TASK] server 已挂, 终止后续档"; break; }
    fi
  done
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "[$(date +%T)] [$TASK] 清理容器"
done

echo; echo "=========================================================="
echo " 压测对比汇总 (RES=${RES}p, int8 4卡):"
echo -e "$SUMMARY"
echo "=========================================================="
echo "产物: /data/outputs/wan_stress_{i2v,flf2v}_${RES}p/"
echo "对比看点: 同帧数下 flf2v(首尾帧) vs i2v(单图) 的 生成耗时 / CPU内存 / CPU% 差距"
