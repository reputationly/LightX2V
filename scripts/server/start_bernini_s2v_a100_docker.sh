#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-lightx2v:bernini-s2v-a100}
CONTAINER_NAME=${CONTAINER_NAME:-lightx2v-bernini-s2v}
HOST_PORT=${HOST_PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NFS_ROOT=${NFS_ROOT:-/nfs-data}

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    --shm-size=32g \
    --memory=240g \
    --memory-swap=240g \
    -p "${HOST_PORT}:8000" \
    -v "${NFS_ROOT}:${NFS_ROOT}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -e LIGHTX2V_PATH=/opt/LightX2V \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    "${IMAGE}" \
    bash /opt/LightX2V/scripts/server/start_bernini_s2v_a100.sh

echo "Bernini-R-S2V service started: ${CONTAINER_NAME} on port ${HOST_PORT}"
