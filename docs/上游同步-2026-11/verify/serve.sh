#!/usr/bin/env bash
# 起一个生产形态的服务:GPUSTACK launcher -> torchrun -> lightx2v.server。
#
#   serve.sh <new|old> <profile> <gpus> <port> [container]
#
#   new = 用 /data/lx2v-sync 的代码(挂到 /opt/LightX2V,含 decord 垫片)
#   old = 镜像自带代码(现网 digest cdd5271fe686),作为对照组
#
# 两种模式走同一个 launcher 路径,唯一差别是 /opt/LightX2V 与 /opt/decord_shim 的来源 ——
# 这样对比的是"同一镜像 + 换代码",不是两套环境。
set -u
MODE=${1:?new|old}
PROFILE=${2:?profile name}
GPUS=${3:?gpu count}
PORT=${4:?port}
CT=${5:-lx2v_$MODE}
IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
SYNC=/data/lx2v-sync
LOGD=/data/verify-logs
mkdir -p "$LOGD"

case "$PROFILE" in
  seedvr2*) MODEL=/nfs-data/models/ByteDance-Seed/SeedVR2-3B ;;
  swiftvr*) MODEL=/nfs-data/models/SwiftVR_lightx2v ;;
  *) echo "unknown profile $PROFILE"; exit 2 ;;
esac

MOUNTS=(-v /nfs-data:/nfs-data)
if [ "$MODE" = new ]; then
  MOUNTS+=(-v "$SYNC":/opt/LightX2V -v "$SYNC"/dockerfiles/decord_shim:/opt/decord_shim)
fi

docker rm -f "$CT" >/dev/null 2>&1
docker run -d --name "$CT" --gpus all --ipc=host --shm-size=32g --network host \
  "${MOUNTS[@]}" -e "CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))" \
  "$IMG" python3 /opt/LightX2V/deploy/gpustack-lx2v-launcher/gpustack_lx2v_launcher.py \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" --profile "$PROFILE" >/dev/null

echo "[serve] $MODE profile=$PROFILE gpus=$GPUS port=$PORT container=$CT"
READY=0
for i in $(seq 1 90); do
  sleep 5
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/ready" 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "[serve] ready after $((i * 5))s"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CT}$"; then echo "[serve] !! container exited"; break; fi
done
SAFE=$(printf '%s' "$PROFILE" | tr '/' '_')
docker logs "$CT" > "$LOGD/serve.$MODE.$SAFE.log" 2>&1
if [ "$READY" != 1 ]; then
  echo "[serve] !! not ready; see $LOGD/serve.$MODE.$SAFE.log"
  tail -40 "$LOGD/serve.$MODE.$SAFE.log"
  exit 1
fi
