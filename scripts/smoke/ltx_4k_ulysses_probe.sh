#!/usr/bin/env bash
# =============================================================================
# LTX2.3 高分辨率多卡(ulysses)提速探针 —— 4K / 2K, int8, ul2 vs ul4
#
# 目的: 验证 "分辨率拉高(token 多)后, ulysses 多卡能否像 Wan2.2 那样把 DiT
#       近线性降下来"。当前 768x1280 实测多卡不划算(token 太少, 通信吃掉收益);
#       本探针在 2K/4K 规格上用 ul2 vs ul4 的 DiT 耗时比直接回答这个问题。
#
# ⚠️ 只测速度/可行性, 不看画质 —— 用的是已知"画质崩"的 int8 权重(全块量化)。
#    画质是另一条线(skip 敏感块), 与本探针无关。
#
# 安全: 跑前强制单线程预热 int8 权重进 page cache, 避免上次那种
#       4-rank 冷读慢盘 -> IO 风暴 -> 内核软锁死(需硬重启)的事故。
#
# 用法(服务器 edt-vpn 上):
#   bash ltx_4k_ulysses_probe.sh                 # 默认: 2K 与 4K 各跑 ul2+ul4, 49帧
#   RES="4k" NP="4" bash ltx_4k_ulysses_probe.sh # 只跑 4K 的 ul4
#   FRAMES=121 bash ltx_4k_ulysses_probe.sh      # 4K 跑通后再上 121 帧
#   SKIP_PREWARM=1 ...                           # 已预热过, 跳过(否则每次都热, 但已缓存秒回)
# 开关:
#   RES   : "2k" | "4k" | "2k 4k"(默认两者, 2K 先跑垫脚)
#   NP    : "2 4"(默认, 同分辨率 ul2 与 ul4 都跑做对比) | "4"(只 ul4)
#   FRAMES: 49(默认, 先短帧降风险) -> 验证能跑后再 121
# =============================================================================
set -uo pipefail

IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
DATA=/data
CFGDIR="$DATA/lightx2v_configs"
OUT="$DATA/outputs"
INT8_DIR="$DATA/models/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1-int8"
BASE_INT8_CFG="$CFGDIR/ltx2_int8_ul2.json"   # 现成的 int8 ulysses-2 配置, 作派生基准
MODEL_PATH="$DATA/models/Lightricks/LTX-2.3"

RES="${RES:-2k 4k}"
NP="${NP:-2 4}"
FRAMES="${FRAMES:-49}"
mkdir -p "$OUT" "$CFGDIR"

G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[36m'; N=$'\e[0m'
ts(){ date +%T; }
log(){  printf '%s[%s]%s %s\n' "$B" "$(ts)" "$N" "$*"; }
ok(){   printf '%s[%s] OK  %s%s\n' "$G" "$(ts)" "$*" "$N"; }
bad(){  printf '%s[%s] FAIL %s%s\n' "$R" "$(ts)" "$*" "$N"; }
warn(){ printf '%s[%s] WARN %s%s\n' "$Y" "$(ts)" "$*" "$N"; }
hr(){   printf '%s--------------------------------------------------------------%s\n' "$B" "$N"; }

declare -A REL   # 结果: key=res_npNP -> "elapsed=.. peak=.. perstep=.. status=.."

# 分辨率: target 是最终输出(开 upsampler 时 stage1=target/2); 必须 64 整除
res_hw(){ case "$1" in
  2k) echo "1536 2560" ;;   # stage1 768x1280(=已验证全分辨率), upsample-> 1536x2560
  4k) echo "2304 3840" ;;   # 768x1280 的 3x 等比, ~4K 宽
  *)  echo "" ;;
esac; }

rm_container(){ docker kill "$1" >/dev/null 2>&1 || true; docker rm -f "$1" >/dev/null 2>&1 || true; }

prewarm(){  # 单线程顺序读 int8 权重进 page cache —— 防 IO 风暴内核锁死
  [ "${SKIP_PREWARM:-0}" = "1" ] && { warn "SKIP_PREWARM=1, 跳过预热(确认已缓存才跳)"; return 0; }
  [ -d "$INT8_DIR" ] || { bad "int8 权重目录不存在: $INT8_DIR"; exit 2; }
  local sz; sz=$(du -sh "$INT8_DIR" 2>/dev/null | cut -f1)
  log "预热 int8 权重进内存(单线程, 防多 rank 冷读打爆慢盘): $INT8_DIR ($sz)"
  local n=0
  for f in "$INT8_DIR"/*.safetensors "$INT8_DIR"/*.json; do
    [ -f "$f" ] || continue
    cat "$f" > /dev/null 2>&1 && n=$((n+1))
  done
  ok "预热完成: $n 个文件已读入 page cache(26GB << 256GB 内存, 会常驻缓存)"
}

monitor_start(){  # 后台记录 host load / dmesg, 出现软锁死征兆时留痕(不自动 kill, 你盯着可 Ctrl-C)
  local mf="$1"
  ( while true; do
      printf '%s load=%s mem_avail=%s\n' "$(date +%T)" \
        "$(cut -d' ' -f1-3 /proc/loadavg)" \
        "$(awk '/MemAvailable/{print $2/1024/1024"G"}' /proc/meminfo)"
      sleep 5
    done ) > "$mf" 2>&1 &
  echo $!
}

gen_cfg(){  # $1=res标签 $2=H $3=W $4=seq_p -> 写出派生配置, 回显路径
  local rl=$1 H=$2 W=$3 sp=$4
  local out="$CFGDIR/ltx2_int8_${rl}_ul${sp}.json"
  python3 - "$BASE_INT8_CFG" "$out" "$H" "$W" "$sp" <<'PY'
import json, sys
base, out, H, W, sp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = json.load(open(base))
d["target_height"] = H
d["target_width"]  = W
d.setdefault("parallel", {})
d["parallel"]["seq_p_size"] = sp
d["parallel"].setdefault("seq_p_attn_type", "ulysses")
d["cpu_offload"] = False           # int8 ~20GB/卡 全驻留
d["gemma_cpu_offload"] = True      # 否则 gemma 也上 GPU -> OOM
json.dump(d, open(out, "w"), indent=2, ensure_ascii=False)
print(out)
PY
}

wait_health(){ local to=$1 t=0 code; while [ "$t" -lt "$to" ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { printf '\n'; return 0; }
  sleep 10; t=$((t+10)); printf '%s[%s]%s   加载中 %ss/%ss (health=%s)\r' "$B" "$(ts)" "$N" "$t" "$to" "$code"
done; printf '\n'; return 1; }

run_one(){  # $1=res标签 $2=seq_p
  local rl=$1 sp=$2 key="${rl}_ul${sp}"
  read -r H W <<<"$(res_hw "$rl")"
  [ -z "${H:-}" ] && { warn "未知分辨率 $rl, 跳过"; return; }
  hr; log "${Y}== $rl ($W x $H, ${FRAMES}帧) | int8 ulysses-${sp} ==${N}"
  local cfg; cfg=$(gen_cfg "$rl" "$H" "$W" "$sp") || { bad "$key 配置生成失败"; REL[$key]="status=cfg_fail"; return; }
  log "配置: $cfg"

  local name="ltx-probe-$key" devs; devs=$(seq -s, 0 $((sp-1)))
  rm_container "$name"
  local slog="/tmp/ltxprobe_${key}_server.log"
  docker run -d --name "$name" --gpus all --shm-size=32g -p 8000:8000 -p 8001:8001 -v /data:/data -v /nfs-data:/nfs-data \
    -e PYTHONPATH=/opt/LightX2V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES="$devs" -e LTX_GEMMA_ON_CPU=1 -e LTX_GEMMA_LAYERWISE_GPU=1 \
    -e LTX_VAE_SPATIAL_TILE=256 -e LTX_VAE_SPATIAL_OVERLAP=32 -e LTX_VAE_TEMPORAL_TILE=16 -e LTX_VAE_TEMPORAL_OVERLAP=8 \
    "$IMG" torchrun --nproc_per_node="$sp" --master_port=29531 -m lightx2v.server \
    --model_cls ltx2 --task t2av --model_path "$MODEL_PATH" \
    --config_json "$cfg" --host 0.0.0.0 --port 8000 >/dev/null \
    || { bad "$key 容器启动失败"; REL[$key]="status=launch_fail"; return; }

  log "等待 server 就绪(<=900s, 4K 加载+预热较久)..."
  if ! wait_health 900; then
    bad "$key health 超时, 末尾日志:"; docker logs --tail 50 "$name" 2>&1 | sed 's/^/    /'
    docker logs "$name" > "$slog" 2>&1 || true
    REL[$key]="status=health_timeout"; rm_container "$name"; return
  fi
  ok "server ready, 提交 1 条生成(${FRAMES}帧)..."

  local probe="$OUT/ltx_test_probe.mp4"; rm -f "$probe"
  local t0 t1 el; t0=$(date +%s)
  bash "$DATA/ltx_one.sh" "$FRAMES" 2>&1 | tee "/tmp/ltxprobe_${key}_submit.log"
  t1=$(date +%s); el=$((t1-t0))
  docker logs "$name" > "$slog" 2>&1 || true

  local peak perstep status="ok"
  peak=$(grep -oE 'peak=[0-9]+MiB' "/tmp/ltxprobe_${key}_submit.log" | tail -1 | sed 's/peak=//')
  # 尝试从 server 日志抽 DiT 每步耗时(格式可能因版本不同, 抽不到就看 $slog)
  perstep=$(grep -oiE '([0-9]+\.[0-9]+)s/?(it|step)' "$slog" | tail -3 | tr '\n' ' ')
  if grep -qiE 'out of memory|CUDA error|RuntimeError' "$slog"; then status="OOM/err"; fi
  if [ ! -f "$probe" ]; then status="no_output"; bad "$key 未产出文件(可能 OOM/失败), 见 $slog"; fi

  REL[$key]="status=$status elapsed=${el}s peak=${peak:-?} perstep=[${perstep:-看$slog}]"
  log "$key 结果: ${REL[$key]}"
  rm_container "$name"; sleep 3
}

# ---- 主流程 ----
hr; log "镜像: $IMG"; log "分辨率: $RES | 卡数: $NP | 帧数: $FRAMES"
log "判读: 同分辨率下 ul4 的 DiT 若 ≈ ul2 的一半 -> ulysses 在该规格划算; 若 ≈ 持平 -> 仍通信瓶颈"
hr
[ -f "$BASE_INT8_CFG" ] || { bad "基准 int8 配置不存在: $BASE_INT8_CFG (确认 int8 ul2 配置路径)"; exit 2; }
[ -f "$DATA/ltx_one.sh" ] || { bad "$DATA/ltx_one.sh 不存在(提交脚本)"; exit 2; }

prewarm
MF="/tmp/ltxprobe_monitor.log"; MPID=$(monitor_start "$MF")
log "host 监控已起(PID $MPID -> $MF), 留意 load 飙升/MemAvailable 骤降"

for rl in $RES; do
  for sp in $NP; do
    run_one "$rl" "$sp"
  done
done

kill "$MPID" >/dev/null 2>&1 || true

echo; printf '%s============== LTX 高分辨率多卡探针 结果 ==============%s\n' "$B" "$N"
printf '%-12s %s\n' "规格_卡数" "指标"
for rl in $RES; do for sp in $NP; do
  k="${rl}_ul${sp}"; printf '%-12s %s\n' "$k" "${REL[$k]:-未跑}"
done; done
hr
log "对照基线: 单卡 768x1280/121帧 ≈ 86s; 2K stage1 正好=768x1280"
log "完整 server 日志: /tmp/ltxprobe_*_server.log  (DiT 每步耗时在里面)"
log "⚠️ 本探针只验速度/可行性; int8 画质崩是已知的, 不在本轮范围"
