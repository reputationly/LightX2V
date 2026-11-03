#!/usr/bin/env bash
# §8.2 取消回归:每个取消点都要「4 rank 在当前段内退出 + 锁释放 + 紧接着的请求能完成」。
#
#   cancel_regression.sh <port> <用哪个 profile 起的服务> [容器名]
#
# 覆盖:372f(4 段) 在 5s / 20s / 60s 取消、denoise 阶段取消、300f(3 段,段数不是 4 的倍数)取消。
# 每例之后紧接一个 480p/124f 小请求,必须 completed —— 这是"锁释放"的直接证据。
set -u
PORT=${1:?port}
PROFILE=${2:?profile}
CT=${3:-lx2v_new}
A=/nfs-data/lx2v_test/assets
V=/nfs-data/lx2v_test/verify-out
L=/data/verify-logs
SR=$(dirname "$0")/sr.sh
mkdir -p "$V" "$L"
OUT="$L/cancel_regression.$(date +%H%M%S).log"
: > "$OUT"

log() { echo "$*" | tee -a "$OUT"; }

post_probe() {  # 取消后的小请求:唯一的"服务还活着"判据
  local tag=$1
  local out="$V/cancelreg_after_${tag}.mp4"
  local before
  before=$(date +%s)
  bash "$SR" "$PORT" "$A/sr_input_5s.mp4" "$out" '"sr_ratio":4.0' >/dev/null 2>&1
  local status
  status=$(python3 -c "import json;print(json.load(open('$L/cancelreg_after_${tag}.json'))['status'])" 2>/dev/null || echo "MISSING")
  local wall=$(( $(date +%s) - before ))
  if [ "$status" = "completed" ]; then
    log "  [probe $tag] 后续请求 completed,${wall}s  ✅ 锁已释放"
  else
    log "  [probe $tag] 后续请求 status=$status,${wall}s  ❌ 服务未恢复"
  fi
}

run_case() {  # run_case <tag> <src> <cancel_after_s>
  local tag=$1 src=$2 after=$3
  log "== $tag:$(basename "$src") cancel@${after}s"
  bash "$SR" "$PORT" "$src" "$V/cancelreg_${tag}.mp4" '"sr_ratio":4.0' "$after" >/dev/null 2>&1
  local status vram
  status=$(python3 -c "import json;print(json.load(open('$L/cancelreg_${tag}.json'))['status'])" 2>/dev/null || echo MISSING)
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
  log "  状态=$status ; 取消后显存=$vram"
  post_probe "$tag"
}

log "### 取消回归开始 $(date -Is) profile=$PROFILE container=$CT"
run_case cancel_5s  "$A/sr_768p_15s.mp4" 5
run_case cancel_20s "$A/sr_768p_15s.mp4" 20
run_case cancel_60s "$A/sr_768p_15s.mp4" 60
run_case cancel_denoise "$A/sr_768p_15s.mp4" 45
run_case cancel_300f "${THREEF:-/nfs-data/lx2v_test/verify-src/src_768p_300.mp4}" 20

log "### 无害检查:服务日志里不应出现 tail 超时"
if docker logs "$CT" 2>&1 | grep -qE "boundary frames never arrived|_SR_TAIL_TIMEOUT|peer rank failed"; then
  log "  ❌ 日志出现 tail 超时/peer failed"
else
  log "  ✅ 日志无 tail 超时"
fi
log "### 结束 $(date -Is) ; 明细 $OUT"
