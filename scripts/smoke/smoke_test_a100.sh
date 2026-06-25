#!/usr/bin/env bash
# =============================================================================
# LightX2V ARM64 A100 镜像冒烟回归 —— 三用例统一脚本
#
# 覆盖 docs/LTX-2.3-单卡A100-POC交接.md §15 的三个验收用例:
#   用例1  LTX2.3        单卡 bf16        1280x768 / 121帧
#   用例2  Wan2.2 int8   单卡            832x480  / 49帧
#   用例3  Wan2.2 int8   4卡 ulysses     1280x720 / 49帧
#
# 每个用例自动: 清旧容器 -> 起 server -> 等 health -> 提交生成 -> 验证产物
#               (规格 / 抽帧体积查雪花 / 黑屏) -> 停容器 -> 记录结果
# 结束打印汇总表, 全绿则 exit 0。
#
# 用法:
#   bash smoke_test_a100.sh [IMAGE]          # IMAGE 可选, 默认下方 DEFAULT_IMG
#   CASES="1 3" bash smoke_test_a100.sh      # 只跑用例 1 和 3
#   LX_IMG=<img> bash smoke_test_a100.sh     # 或用环境变量指定镜像
# 开关(环境变量):
#   SKIP_PULL=1        跳过镜像拉取(离线/已确认本地即目标镜像时); 默认总是 pull 刷新
#   SKIP_PREFLIGHT=1   跳过前置镜像 smoke(import 检查); 默认执行
#   SKIP_GPU_GUARD=1   跳过 GPU 空闲检查; 默认检查(有残留计算进程则终止)
#
# 前提: 在 A100 服务器(edt-vpn)上运行, /data 下有权重/配置, 且有
#       /data/ltx_one.sh 与 /data/wan_verify.sh 两个提交脚本。
# 全部使用绝对路径, 放在服务器任意目录都可运行。
# =============================================================================
set -uo pipefail

DEFAULT_IMG="crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721"
IMG="${1:-${LX_IMG:-$DEFAULT_IMG}}"
CASES="${CASES:-1 2 3}"
DATA=/data
OUT="$DATA/outputs"
mkdir -p "$OUT"

# ---- 颜色 / 日志 ----
G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[36m'; N=$'\e[0m'
ts(){ date +%T; }
log(){  printf '%s[%s]%s %s\n'        "$B" "$(ts)" "$N" "$*"; }
ok(){   printf '%s[%s] OK  %s%s\n'    "$G" "$(ts)" "$*" "$N"; }
bad(){  printf '%s[%s] FAIL %s%s\n'   "$R" "$(ts)" "$*" "$N"; }
warn(){ printf '%s[%s] WARN %s%s\n'   "$Y" "$(ts)" "$*" "$N"; }
hr(){   printf '%s--------------------------------------------------------------%s\n' "$B" "$N"; }

declare -A RES SPEC FRAME ELAPSED PEAK LABEL

# ---- 工具函数 ----
rm_container(){ docker kill "$1" >/dev/null 2>&1 || true; docker rm -f "$1" >/dev/null 2>&1 || true; }

wait_health(){  # $1=timeout_sec ; 返回 0=ready 1=timeout
  local to=$1 t=0 code
  while [ "$t" -lt "$to" ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { printf '\n'; return 0; }
    sleep 10; t=$((t+10))
    printf '%s[%s]%s   加载中 %ss / %ss (health=%s)\r' "$B" "$(ts)" "$N" "$t" "$to" "$code"
  done
  printf '\n'; return 1
}

run_submit(){  # $1=cmd  $2=case ; 实时透传输出并解析 elapsed/peak
  local cmd=$1 cs=$2 lf="/tmp/smoke_c${cs}.log"
  bash -c "$cmd" 2>&1 | tee "$lf"
  ELAPSED[$cs]=$(grep -oE 'elapsed=[0-9]+s'  "$lf" | tail -1 | sed 's/elapsed=//')
  PEAK[$cs]=$(   grep -oE 'peak=[0-9]+MiB'   "$lf" | tail -1 | sed 's/peak=//')
  grep -qE 'status=completed|DONE' "$lf"   # 返回码反映是否完成
}

verify(){  # $1=file $2=W $3=H $4=F $5=snow_kb $6=case ; 返回 0=通过
  local f=$1 ew=$2 eh=$3 ef=$4 snow=$5 cs=$6
  [ -f "$f" ] || { bad "用例$cs 产物不存在: $f"; return 1; }
  local w h nf
  w=$( ffprobe -v error -select_streams v:0 -show_entries stream=width      -of csv=p=0 "$f" 2>/dev/null)
  h=$( ffprobe -v error -select_streams v:0 -show_entries stream=height     -of csv=p=0 "$f" 2>/dev/null)
  nf=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames  -of csv=p=0 "$f" 2>/dev/null)
  SPEC[$cs]="${w}x${h}/${nf}帧"
  # 抽第20帧, 用体积做雪花旁证 (高熵噪声压缩后异常大)
  local png="$OUT/smoke_c${cs}_frame20.png" kb=0
  ffmpeg -y -i "$f" -vf "select=eq(n\,20)" -vframes 1 "$png" >/dev/null 2>&1 || true
  [ -f "$png" ] && kb=$(( $(stat -c%s "$png" 2>/dev/null || echo 0) / 1024 ))
  FRAME[$cs]="${kb}KB"
  # 黑屏检测
  local black
  black=$(ffmpeg -v info -i "$f" -vf blackdetect=d=0.1:pix_th=0.10 -an -f null - 2>&1 | grep -c blackdetect || true)
  # 判定
  local pass=1 notes=""
  if [ "$w" != "$ew" ] || [ "$h" != "$eh" ] || [ "$nf" != "$ef" ]; then
    pass=0; notes="$notes 规格不符(期望 ${ew}x${eh}/${ef}帧);"
  fi
  if [ "$black" != "0" ]; then pass=0; notes="$notes 检出黑屏${black}处;"; fi
  if [ "$kb" -gt "$snow" ]; then pass=0; notes="$notes 抽帧${kb}KB > 阈值${snow}KB(疑雪花);"; fi
  if [ "$pass" = "1" ]; then
    ok  "用例$cs 验证通过  规格 ${w}x${h}/${nf}帧 | 抽帧 ${kb}KB | 黑屏 无 | 抽帧图 $png"
    return 0
  fi
  bad "用例$cs 验证失败 -${notes}  (抽帧图 $png 可肉眼复核)"
  return 1
}

gpu_guard(){  # 确认要用的卡空闲: 容器预清理后仍有残留计算进程则列出并终止 (SKIP_GPU_GUARD=1 跳过)
  command -v nvidia-smi >/dev/null 2>&1 || { warn "无 nvidia-smi, 跳过 GPU 空闲检查"; return 0; }
  local procs
  procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  if [ -n "$procs" ]; then
    bad "GPU 上有残留计算进程, 可能 OOM/冲突, 终止 (确认无碍可 SKIP_GPU_GUARD=1 跳过):"
    echo "$procs" | sed 's/^/    /'
    warn "如确认是你自己的旧容器, 先 docker ps 找到并 docker rm -f, 或 kill 对应 PID 后重跑"
    exit 3
  fi
  ok "GPU 空闲检查通过 (无残留计算进程, 4 卡可用)"
}

preflight(){  # 镜像级 smoke: 秒级确认依赖/server 可 import, 坏镜像立即终止, 不浪费每用例 5~10min 加载
  hr; log "${Y}前置检查  镜像 smoke (import 依赖 + server)${N}"
  if docker run --rm --gpus all "$IMG" python -c "import torch,flash_attn,lightx2v" >/dev/null 2>&1; then
    ok "依赖 import 正常 (torch / flash_attn / lightx2v)"
  else
    bad "依赖 import 失败 —— 镜像可能损坏, 终止 (设 SKIP_PREFLIGHT=1 可跳过)"; exit 2
  fi
  if docker run --rm --gpus all "$IMG" python -c "import lightx2v.server" >/dev/null 2>&1; then
    ok "lightx2v.server import 正常"
  else
    bad "lightx2v.server import 失败 —— 终止 (设 SKIP_PREFLIGHT=1 可跳过)"; exit 2
  fi
}

start_and_verify(){  # 通用流程: $1=容器名 $2=health超时 $3=提交cmd $4=产物源 $5=产物存档名 $6=W $7=H $8=F $9=snowKB $10=case
  local name=$1 hto=$2 cmd=$3 src=$4 dst=$5 W=$6 H=$7 F=$8 snow=$9 cs=${10}
  log "等待 server 就绪 (<= ${hto}s)..."
  if ! wait_health "$hto"; then
    bad "用例$cs health 超时, 末尾日志:"; docker logs --tail 40 "$name" 2>&1 | sed 's/^/    /'
    RES[$cs]=FAIL; rm_container "$name"; return
  fi
  ok "server ready, 提交生成任务..."
  rm -f "$src" "$dst"   # 清掉历史残留, 确保只验证本次新产物 (防陈旧文件假 PASS)
  if ! run_submit "$cmd" "$cs"; then
    bad "用例$cs 生成未完成 (日志无 completed 标记), 判 FAIL"; RES[$cs]=FAIL; rm_container "$name"; sleep 3; return
  fi
  if [ ! -f "$src" ]; then
    bad "用例$cs 标记完成但未产出新文件 $src, 判 FAIL"; RES[$cs]=FAIL; rm_container "$name"; sleep 3; return
  fi
  cp -f "$src" "$dst"
  if verify "$dst" "$W" "$H" "$F" "$snow" "$cs"; then RES[$cs]=PASS; else RES[$cs]=FAIL; fi
  rm_container "$name"
  sleep 3
}

# ---- 三个用例 ----
case1(){
  LABEL[1]="LTX2.3 单卡 bf16"
  hr; log "${Y}用例1  ${LABEL[1]}  (1280x768/121帧, 期望 ~86s, 峰值 ~18.9GB)${N}"
  rm_container lightx2v-ltx-new
  docker run -d --name lightx2v-ltx-new --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data \
    -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES=0 -e LTX_GEMMA_ON_CPU=1 -e LTX_GEMMA_LAYERWISE_GPU=1 \
    -e LTX_VAE_SPATIAL_TILE=256 -e LTX_VAE_SPATIAL_OVERLAP=32 -e LTX_VAE_TEMPORAL_TILE=16 -e LTX_VAE_TEMPORAL_OVERLAP=8 \
    "$IMG" python -m lightx2v.server --model_cls ltx2 --task t2av \
    --model_path /data/models/Lightricks/LTX-2.3 \
    --config_json /data/lightx2v_configs/ltx2_3_distill_v11_hq.json --host 0.0.0.0 --port 8000 >/dev/null \
    || { bad "用例1 容器启动失败"; RES[1]=FAIL; return; }
  start_and_verify lightx2v-ltx-new 600 "bash /data/ltx_one.sh 121" \
    "$OUT/ltx_test_probe.mp4" "$OUT/smoke_c1_ltx.mp4" 1280 768 121 3000 1
}

case2(){
  LABEL[2]="Wan2.2 int8 单卡 480p"
  hr; log "${Y}用例2  ${LABEL[2]}  (832x480/49帧, 期望 ~56s, 峰值 ~34.8GB)${N}"
  rm_container lightx2v-wan-int8
  docker run -d --name lightx2v-wan-int8 --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data \
    -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=0 \
    "$IMG" python -m lightx2v.server --model_cls wan2.2_moe --task t2v \
    --model_path /data/models/Wan-AI/Wan2.2-T2V-A14B \
    --config_json /data/lightx2v_configs/test_480p_int8_prequant.json --host 0.0.0.0 --port 8000 >/dev/null \
    || { bad "用例2 容器启动失败"; RES[2]=FAIL; return; }
  start_and_verify lightx2v-wan-int8 600 "bash /data/wan_verify.sh" \
    "$OUT/wan_verify.mp4" "$OUT/smoke_c2_wan480.mp4" 832 480 49 1500 2
}

case3(){
  LABEL[3]="Wan2.2 int8 4卡 720p"
  hr; log "${Y}用例3  ${LABEL[3]}  (1280x720/49帧, 期望 ~51s, 峰值 ~34.6GB/卡)${N}"
  rm_container lightx2v-wan-ul4
  docker run -d --name lightx2v-wan-ul4 --gpus all --shm-size=32g -p 8000:8000 -p 8001:8001 -v /data:/data \
    -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$IMG" torchrun --nproc_per_node=4 --master_port=29524 -m lightx2v.server \
    --model_cls wan2.2_moe --task t2v --model_path /data/models/Wan-AI/Wan2.2-T2V-A14B \
    --config_json /data/lightx2v_configs/cmp_720p_seko.json --host 0.0.0.0 --port 8000 >/dev/null \
    || { bad "用例3 容器启动失败"; RES[3]=FAIL; return; }
  start_and_verify lightx2v-wan-ul4 900 "bash /data/wan_verify.sh" \
    "$OUT/wan_verify.mp4" "$OUT/smoke_c3_wan720.mp4" 1280 720 49 2200 3
}

# ---- 主流程 ----
START=$(date +%s)
hr
log "镜像: $IMG"
log "用例: $CASES   产物目录: $OUT"
hr
# 默认总是 pull, 确保验的是最新发布的镜像 (可变 tag 如 *latest 关键; 不可变 tag 无新层秒回)。
# 离线 / 已确认本地即目标镜像时, 用 SKIP_PULL=1 跳过。
if [ "${SKIP_PULL:-0}" = "1" ]; then
  docker image inspect "$IMG" >/dev/null 2>&1 || { bad "SKIP_PULL=1 但本地无镜像 $IMG, 退出"; exit 2; }
  log "SKIP_PULL=1, 使用本地已有镜像 (不刷新)"
else
  log "拉取/刷新镜像 $IMG ..."
  docker pull "$IMG" || { bad "镜像拉取失败, 退出"; exit 2; }
fi
# 前置镜像 smoke (可用 SKIP_PREFLIGHT=1 跳过)
[ "${SKIP_PREFLIGHT:-0}" = "1" ] || preflight

# 预清理所有相关容器, 避免 8000 端口冲突
for c in lightx2v-ltx-new lightx2v-wan-int8 lightx2v-wan-ul4 lightx2v-ltx-server lightx2v-server; do rm_container "$c"; done

# 清理后检查 GPU 是否真的空闲 (排除别的容器/裸进程占卡; 可用 SKIP_GPU_GUARD=1 跳过)
[ "${SKIP_GPU_GUARD:-0}" = "1" ] || gpu_guard

for c in $CASES; do
  case "$c" in
    1) case1 ;;
    2) case2 ;;
    3) case3 ;;
    *) warn "未知用例编号: $c (跳过)";;
  esac
done

# ---- 汇总 ----
END=$(date +%s)
echo; printf '%s================== 冒烟结果汇总 ==================%s\n' "$B" "$N"
printf '%-5s %-24s %-16s %-9s %-12s %s\n' "用例" "模型/配置" "规格" "耗时" "峰值" "结果"
allpass=1
for c in $CASES; do
  r="${RES[$c]:-未跑}"
  [ "$r" = "PASS" ] && col="$G" || { col="$R"; allpass=0; }
  printf '%-5s %-24s %-16s %-9s %-12s %s%s%s\n' \
    "$c" "${LABEL[$c]:-?}" "${SPEC[$c]:--}" "${ELAPSED[$c]:--}" "${PEAK[$c]:--}" "$col" "$r" "$N"
done
hr
log "总耗时 $((END-START))s"
if [ "$allpass" = "1" ]; then ok "全部用例通过, 镜像验收成功 ✅"; exit 0; else bad "存在失败用例, 请复核上方日志 ❌"; exit 1; fi
