#!/usr/bin/env bash
# =============================================================================
# LTX2.3 1080p 基线生成 —— 满血 bf16 单卡, 5s(121帧/24fps)
#
# 目的: 出一条 1080p 基线片给人眼看画质/流畅度, 作为后续调配置的对照。
#   - 用已验证的 bf16 单卡配置(ltx2_3_distill_v11_hq.json), 只把分辨率
#     从 768x1280 抬到 1088x1920(标准1080p; 高度1080->1088 满足 /64 整除)。
#   - 满血 bf16(非 int8): 画质=模型真实水平, 可直接和 768p 基线对比。
#   - 同 prompt(night_market)/同帧数(121), 和 POC 基线唯一变量是分辨率。
#
# ⚠️ 显存: 768p/121 实测峰值 18.9GB; 1080p(~2x面积)预估 ~23-26GB, 单卡40GB
#    应能装下, 但 upsampler 阶段是峰值, 有 OOM 风险 -> 脚本会捕获并提示退路。
#
# 用法(服务器 edt-vpn 上):
#   bash ltx_1080p_baseline.sh
#   FRAMES=49 bash ltx_1080p_baseline.sh    # 想先快速验证能不能跑, 用短帧
# =============================================================================
set -uo pipefail

IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
DATA=/data
CFGDIR="$DATA/lightx2v_configs"
OUT="$DATA/outputs"
BASE_CFG="$CFGDIR/ltx2_3_distill_v11_hq.json"   # 已验证 bf16 单卡基准配置
MODEL_PATH="$DATA/models/Lightricks/LTX-2.3"
FRAMES="${FRAMES:-121}"
H=1088; W=1920
mkdir -p "$OUT" "$CFGDIR"

G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[36m'; N=$'\e[0m'
ts(){ date +%T; }
log(){ printf '%s[%s]%s %s\n' "$B" "$(ts)" "$N" "$*"; }
ok(){  printf '%s[%s] OK  %s%s\n' "$G" "$(ts)" "$*" "$N"; }
bad(){ printf '%s[%s] FAIL %s%s\n' "$R" "$(ts)" "$*" "$N"; }
warn(){ printf '%s[%s] WARN %s%s\n' "$Y" "$(ts)" "$*" "$N"; }

[ -f "$BASE_CFG" ] || { bad "基准配置不存在: $BASE_CFG"; exit 2; }
[ -f "$DATA/ltx_one.sh" ] || { bad "$DATA/ltx_one.sh 不存在(提交脚本)"; exit 2; }

# 派生 1080p 配置: 只改分辨率, 其余(8步/upsampler/tiling/gemma offload/帧数)全继承
CFG="$CFGDIR/ltx2_3_distill_v11_hq_1080p.json"
python3 - "$BASE_CFG" "$CFG" "$H" "$W" <<'PY'
import json, sys
base, out, H, W = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(base))
d["target_height"] = H
d["target_width"]  = W
json.dump(d, open(out, "w"), indent=2, ensure_ascii=False)
print(out)
PY
ok "已生成 1080p 配置: $CFG  (${W}x${H}, ${FRAMES}帧/24fps)"

NAME=lightx2v-ltx-1080p
docker kill "$NAME" >/dev/null 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true

log "启动 bf16 单卡 server(1080p)..."
docker run -d --name "$NAME" --gpus all -p 8000:8000 -p 8001:8001 -v /data:/data -v /nfs-data:/nfs-data \
  -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES=0 -e LTX_GEMMA_ON_CPU=1 -e LTX_GEMMA_LAYERWISE_GPU=1 \
  -e LTX_VAE_SPATIAL_TILE=256 -e LTX_VAE_SPATIAL_OVERLAP=32 -e LTX_VAE_TEMPORAL_TILE=16 -e LTX_VAE_TEMPORAL_OVERLAP=8 \
  "$IMG" python -m lightx2v.server --model_cls ltx2 --task t2av \
  --model_path "$MODEL_PATH" --config_json "$CFG" --host 0.0.0.0 --port 8000 >/dev/null \
  || { bad "容器启动失败"; exit 2; }

# 等 health
to=600; t=0
while [ "$t" -lt "$to" ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
  sleep 10; t=$((t+10)); printf '%s[%s]%s   加载中 %ss/%ss (health=%s)\r' "$B" "$(ts)" "$N" "$t" "$to" "$code"
done; printf '\n'
if [ "$code" != "200" ]; then
  bad "health 超时, 末尾日志:"; docker logs --tail 50 "$NAME" 2>&1 | sed 's/^/    /'
  docker rm -f "$NAME" >/dev/null 2>&1 || true; exit 1
fi
ok "server ready, 提交生成(night_market, ${FRAMES}帧)..."

PROBE="$OUT/ltx_test_probe.mp4"; rm -f "$PROBE"
SLOG="/tmp/ltx_1080p_server.log"
t0=$(date +%s)
bash "$DATA/ltx_one.sh" "$FRAMES" 2>&1 | tee /tmp/ltx_1080p_submit.log
t1=$(date +%s)
docker logs "$NAME" > "$SLOG" 2>&1 || true

EL=$((t1-t0))
PEAK=$(grep -oE 'peak=[0-9]+MiB' /tmp/ltx_1080p_submit.log | tail -1 | sed 's/peak=//')
DST="$OUT/ltx_1080p_baseline.mp4"

echo
if [ -f "$PROBE" ]; then
  cp -f "$PROBE" "$DST"
  WW=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$DST" 2>/dev/null)
  HH=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$DST" 2>/dev/null)
  NF=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$DST" 2>/dev/null)
  FR=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$DST" 2>/dev/null)
  ok "1080p 基线已生成: $DST"
  log "规格 ${WW}x${HH} / ${NF}帧 / ${FR}fps | 耗时 ${EL}s | 峰值 ${PEAK:-?}"
  log "下载到本地播放器(VLC/QuickTime)看, 别只在网页预览框看(预览端可能丢帧造成'卡'的错觉)"
else
  bad "未产出文件 —— 大概率 1080p 单卡 OOM 或失败, 看日志:"
  grep -iE 'out of memory|CUDA error|RuntimeError|Error' "$SLOG" | tail -10 | sed 's/^/    /'
  warn "退路: ① 先 FRAMES=49 验证管线; ② 仍 OOM 则上多卡(ulysses)或退 768p"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit 1   # 失败要非零退出, 否则 tmux/自动化会误判成功
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
