#!/usr/bin/env bash
# =============================================================================
# LTX2.3 server 常驻启动(供反复提交/调试用, 不自动关)
#   默认起 1080p bf16 单卡; 可换配置/卡数调试。
#
# 用法(服务器 edt-vpn 上):
#   bash /data/ltx_serve.sh                         # 默认 1080p 单卡常驻
#   CFG=/data/lightx2v_configs/ltx2_3_distill_v11_hq.json bash /data/ltx_serve.sh   # 换 768p
#   NP=4 CFG=<ul4配置> bash /data/ltx_serve.sh       # 4卡 ulysses
# 起好后用 /data/ltx_submit.sh 提交任意 prompt。改完配置重跑本脚本即可(会先清旧容器)。
# =============================================================================
set -uo pipefail
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
DATA=/data
CFG="${CFG:-$DATA/lightx2v_configs/ltx2_3_distill_v11_hq.json}"   # 默认用已验证的基准配置(必存在); 1080p 用 CFG=...1080p.json 显式指定
MODEL_PATH="$DATA/models/Lightricks/LTX-2.3"
NP="${NP:-1}"
NAME=lightx2v-ltx-serve
B=$'\e[36m'; G=$'\e[32m'; R=$'\e[31m'; N=$'\e[0m'

[ -f "$CFG" ] || { printf '%s未找到配置: %s%s\n' "$R" "$CFG" "$N"; exit 2; }
docker kill "$NAME" >/dev/null 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true

DEVS=$(seq -s, 0 $((NP-1)))
printf '%s启动 LTX server: 配置=%s 卡数=%s%s\n' "$B" "$CFG" "$NP" "$N"
if [ "$NP" -gt 1 ]; then
  RUNCMD="torchrun --nproc_per_node=$NP --master_port=29531 -m lightx2v.server"
  EXTRA="--shm-size=32g"
else
  RUNCMD="python -m lightx2v.server"
  EXTRA=""
fi
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all $EXTRA -p 8000:8000 -p 8001:8001 -v /data:/data -v /nfs-data:/nfs-data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES="$DEVS" -e LTX_GEMMA_ON_CPU=1 -e LTX_GEMMA_LAYERWISE_GPU=1 \
  -e LTX_VAE_SPATIAL_TILE=256 -e LTX_VAE_SPATIAL_OVERLAP=32 -e LTX_VAE_TEMPORAL_TILE=16 -e LTX_VAE_TEMPORAL_OVERLAP=8 \
  "$IMG" $RUNCMD --model_cls ltx2 --task t2av \
  --model_path "$MODEL_PATH" --config_json "$CFG" --host 0.0.0.0 --port 8000 >/dev/null \
  || { printf '%s容器启动失败%s\n' "$R" "$N"; exit 2; }

to=900; t=0
while [ "$t" -lt "$to" ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
  sleep 10; t=$((t+10)); printf '%s  加载中 %ss/%ss (health=%s)%s\r' "$B" "$t" "$to" "$code" "$N"
done; printf '\n'
if [ "$code" != "200" ]; then
  printf '%shealth 超时, 末尾日志:%s\n' "$R" "$N"; docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /'; exit 1
fi
printf '%sserver ready ✅  常驻中(容器 %s)。提交: bash /data/ltx_submit.sh "<prompt或预设>" [帧数]%s\n' "$G" "$NAME" "$N"
printf '%s停止: docker rm -f %s%s\n' "$B" "$NAME" "$N"
