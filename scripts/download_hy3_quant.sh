#!/usr/bin/env bash
# =============================================================================
# 下载 HuggingFace 社区量化仓到 NFS —— 专治 hf-mirror 对个别大分片 504 的老毛病。
#
# 为什么不用 download_new_models.sh:那个走 ModelScope 主源,而 HunyuanImage-3.0
#   的 NF4/INT8 量化仓(EricRollei / jamesw767)ModelScope 上没有,只能 HF;且
#   hf-mirror 会对某些大 safetensors 分片持续 504,必须逐文件 curl 断点续传兜底,
#   并支持切到 origin(huggingface.co,可挂代理)。
#
# 特性:
#   - 从 HF tree API 拉真实文件清单(含 LFS 真实大小)
#   - 逐文件:本地大小 == 远端 => 秒跳过;否则 curl -C - 断点续传
#   - 双源:hf-mirror 主,origin 回退(origin 可用 PROXY 走 Mac 的 Shadowrocket)
#   - 504/5xx 狂重试;下完按大小校验,不符自动换源重下
#   - 幂等:中断后重跑 = 只补没下全的文件
#
# 用法(manager 或任意挂了 /nfs-models 的节点):
#   bash download_hy3_quant.sh                                  # 默认下 NF4-v2
#   REPO=jamesw767/HunyuanImage-3-Instruct-Distil-INT8 bash download_hy3_quant.sh
#   PROXY=http://127.0.0.1:8899 bash download_hy3_quant.sh      # origin 回退走代理
#   DEST=/some/path bash download_hy3_quant.sh                  # 自定义落盘位置
#
# 配合反向隧道走 Mac 代理(origin 直连被墙时):
#   # Mac 上:ssh -R 8899:127.0.0.1:<Shadowrocket的HTTP端口> root@<本节点>
#   # 然后:PROXY=http://127.0.0.1:8899 bash download_hy3_quant.sh
# =============================================================================
set -u

REPO="${REPO:-EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4-v2}"
DEST="${DEST:-/nfs-models/wuhanjisuan894/models/$(basename "$REPO")}"
MIRROR="${MIRROR:-https://hf-mirror.com}"
ORIGIN="${ORIGIN:-https://huggingface.co}"
PROXY="${PROXY:-}"                       # 例:http://127.0.0.1:8899;空=不用代理
REV="${REV:-main}"
LOG="${LOG:-$(dirname "$DEST")/dl_$(basename "$REPO").log}"
RETRY="${RETRY:-50}"

mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "==== $(date '+%F %T') 下载 $REPO"
echo "     -> $DEST | mirror=$MIRROR | proxy=${PROXY:-无}"

command -v curl  >/dev/null || { echo "!! 缺 curl";  exit 3; }
command -v python3 >/dev/null || { echo "!! 缺 python3"; exit 3; }

# ---- 拉文件清单(含真实大小):mirror 优先,失败切 origin(带代理) ----
fetch_tree(){
  local host="$1"; local px=()
  [ "$host" = "$ORIGIN" ] && [ -n "$PROXY" ] && px=(-x "$PROXY")
  curl -sSL --fail --connect-timeout 30 "${px[@]}" \
    "$host/api/models/$REPO/tree/$REV?recursive=1"
}
API_JSON=""
for host in "$MIRROR" "$ORIGIN"; do
  echo ">>> 拉文件清单 @ $host"
  API_JSON=$(fetch_tree "$host") && [ -n "$API_JSON" ] && break
  API_JSON=""
done
[ -z "$API_JSON" ] && { echo "!! 清单拉取失败(mirror+origin 都不通),检查网络/代理"; exit 2; }

mapfile -t FILES < <(python3 - "$API_JSON" <<'PY'
import sys, json
for e in json.loads(sys.argv[1]):
    if e.get("type") == "file":
        size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
        print(f'{size}\t{e["path"]}')
PY
)
[ "${#FILES[@]}" -eq 0 ] && { echo "!! 清单为空,REPO/REV 是否正确?"; exit 2; }
echo ">>> 清单共 ${#FILES[@]} 个文件"

# ---- 后台聚合测速(照 download_new_models.sh 的 mon 简版) ----
mon(){
  local prev now; prev=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0)
  while sleep 15; do
    now=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0)
    awk -v p="$prev" -v n="$now" -v ts="$(date +%T)" 'BEGIN{
      printf "[%s] ▼ 已下 %.1f GB | 近15s %.0f MB/s\n", ts, n/1073741824, (n-p)/15/1048576 }'
    prev=$now
  done
}
mon & MON=$!; trap 'kill $MON 2>/dev/null || true' EXIT

# ---- 逐文件下载:大小匹配跳过,否则断点续传;mirror 失败切 origin ----
dlfile(){
  local size="$1" path="$2" out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  local have=0; [ -f "$out" ] && have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$size" != "0" ] && [ "$have" = "$size" ]; then echo "  跳过(完整) $path"; return 0; fi
  echo "  下载 $path (远端 ${size}B / 本地 ${have}B)"
  local host px
  for host in "$MIRROR" "$ORIGIN"; do
    px=(); [ "$host" = "$ORIGIN" ] && [ -n "$PROXY" ] && px=(-x "$PROXY")
    if curl -L --fail -C - --retry "$RETRY" --retry-delay 5 --retry-all-errors \
            --connect-timeout 30 "${px[@]}" \
            -o "$out" "$host/$REPO/resolve/$REV/$path"; then
      local now; now=$(stat -c%s "$out" 2>/dev/null || echo 0)
      if [ "$size" = "0" ] || [ "$now" = "$size" ]; then echo "    OK($host) $path"; return 0; fi
      echo "    !! 大小不符($now != $size),换源重下"
    else
      echo "    !! $host 下载失败,换源"
    fi
  done
  echo "  !!! 两源都失败: $path"; return 1
}

FAIL=0
for line in "${FILES[@]}"; do
  dlfile "${line%%$'\t'*}" "${line#*$'\t'}" || FAIL=1
done

kill $MON 2>/dev/null || true
echo "==== $(date '+%F %T') 校验 ===="
du -sh "$DEST"
echo "safetensors 分片数: $(ls "$DEST"/model-*.safetensors 2>/dev/null | wc -l)"
ls "$DEST" | grep -E 'config.json|index.json|modeling_' || true
if [ "$FAIL" = 0 ]; then echo "全部完成 ✅"; else echo "有文件失败,重跑本脚本会续传 ❌"; exit 1; fi
