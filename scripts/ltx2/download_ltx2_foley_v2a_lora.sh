#!/usr/bin/env bash
# =============================================================================
# 下载 LTX-2.3 V2A Foley LoRA 到 NFS(照 Bernini/scripts/download_bernini_models.sh 范式)。
# v2a(纯配音)裸基座音频条件化很弱 → 必须挂官方 Foley LoRA 才能让音效跟画面/提示词对上。
#
# 下载清单(二选一,auto 自动回退):
#   ① 官方  Lightricks/LTX-2.3-22b-LoRA-Foley-V2A
#            文件: ltx-2.3-22b-lora-foley-v2a-1.0.safetensors (~227MB) + ltx-2.3-foley-v2a.json
#            ★ GATED:必须先在 HF 同意 license 并拿 token,否则 403。设 HF_TOKEN=hf_xxx 用它。
#   ② 社区  FuzzPuppy/LTX-2.3-Foley-LoRA  (un-gated,无需 token)
#            文件: ltx-2.3-foley-400-steps.safetensors
#
# 源:ModelScope 主源(国内快)+ hf-mirror 回退,断点续传,只取 *.safetensors/*.json(跳过示例 mp4)。
#
# 用法(管理节点,联网):
#   # 有官方权限:
#   HF_TOKEN=hf_xxxx tmux new -s dl_foley -d 'bash /nfs-models/_transfer/download_ltx2_foley_v2a_lora.sh'
#   # 没权限直接用社区版:
#   SOURCE=community tmux new -s dl_foley -d 'bash /nfs-models/_transfer/download_ltx2_foley_v2a_lora.sh'
#   tail -f /nfs-models/wuhanjisuan894/dl_foley.log
# SOURCE=auto(默认,先官方后社区) | official(只官方) | community(只社区)。中断后重跑自动续传。
# =============================================================================
set -u
DEST="${DEST:-/nfs-models/wuhanjisuan894/models/loras/foley-v2a}"
LOG="${LOG:-/nfs-models/wuhanjisuan894/dl_foley.log}"
SOURCE="${SOURCE:-auto}"
HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "==== [$(date +%T)] LTX-2.3 Foley V2A LoRA 下载 → $DEST | SOURCE=$SOURCE ===="
python3 -m pip --version >/dev/null 2>&1 || { echo "装 pip..."; apt-get install -y python3-pip >/tmp/dl_pip.log 2>&1; }
if ! python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >>/tmp/dl_pip.log 2>&1; then
  echo "pip/依赖安装失败(看 /tmp/dl_pip.log)"; tail -8 /tmp/dl_pip.log; exit 2
fi

# 只下模型权重与配置,跳过 comparisons/examples 里的示例 mp4。
PATTERNS='["*.safetensors","*.json"]'

# 官方(gated):ModelScope 主源 → hf-mirror(带 token)回退。
dl_official(){
  echo ""
  echo ">>> [$(date +%T)] 官方 Lightricks/LTX-2.3-22b-LoRA-Foley-V2A (ModelScope 优先)"
  if modelscope download --model "Lightricks/LTX-2.3-22b-LoRA-Foley-V2A" --local_dir "$DEST" \
       --include "*.safetensors" --include "*.json" 2>/dev/null; then
    echo "  OK(MS) 官方"; return 0
  fi
  echo "  !! ModelScope 无此库/失败,回退 hf-mirror(需 HF_TOKEN)"
  [ -n "$HF_TOKEN" ] || { echo "  ✗ 官方 GATED 且未设 HF_TOKEN → 跳过官方"; return 1; }
  HF_TOKEN="$HF_TOKEN" python3 - <<PY
import sys, time
from huggingface_hub import snapshot_download
for a in range(1, 41):
    try:
        snapshot_download(repo_id="Lightricks/LTX-2.3-22b-LoRA-Foley-V2A",
                          local_dir="$DEST", token="$HF_TOKEN",
                          allow_patterns=$PATTERNS, max_workers=2, etag_timeout=30)
        print("OK"); break
    except Exception as e:
        print(f"  [retry {a}/40] {type(e).__name__}: {str(e)[:90]} — 15s 后续传", flush=True)
        time.sleep(15)
else:
    raise SystemExit(1)
PY
}

# 社区(un-gated):无需 token。
dl_community(){
  echo ""
  echo ">>> [$(date +%T)] 社区 FuzzPuppy/LTX-2.3-Foley-LoRA (un-gated, hf-mirror)"
  python3 - <<PY
import sys, time
from huggingface_hub import snapshot_download
for a in range(1, 41):
    try:
        snapshot_download(repo_id="FuzzPuppy/LTX-2.3-Foley-LoRA",
                          local_dir="$DEST",
                          allow_patterns=$PATTERNS, max_workers=2, etag_timeout=30)
        print("OK"); break
    except Exception as e:
        print(f"  [retry {a}/40] {type(e).__name__}: {str(e)[:90]} — 15s 后续传", flush=True)
        time.sleep(15)
else:
    raise SystemExit(1)
PY
}

ok=0
case "$SOURCE" in
  official)  dl_official  && ok=1 ;;
  community) dl_community && ok=1 ;;
  auto)      dl_official && ok=1 || { echo "  → 官方拿不到,自动回退社区版"; dl_community && ok=1; } ;;
  *) echo "未知 SOURCE=$SOURCE(auto|official|community)"; exit 2 ;;
esac

# configs/ltx2/ltx2_3_v2a.json 里 lora_configs 写死的是官方文件名。社区回退版的文件名
# 不同(ltx-2.3-foley-400-steps.safetensors),不规范化就会让 config 的 safe_open 找不到
# 文件、加载失败。故:任何源成功后,若规范名缺失,把实下的 .safetensors 软链到规范名,
# 保证 config 引用的路径永远可加载。
CANON_NAME="ltx-2.3-22b-lora-foley-v2a-1.0.safetensors"

echo ""
echo "==== [$(date +%T)] 结果核对 ===="
found=$(find "$DEST" -maxdepth 2 -iname "*.safetensors" 2>/dev/null)
if [ "$ok" -eq 1 ] && [ -n "$found" ]; then
  if [ ! -e "$DEST/$CANON_NAME" ]; then
    src=$(echo "$found" | head -1)
    if [ -n "$src" ]; then
      ln -sf "$src" "$DEST/$CANON_NAME"
      echo "  ↪ 社区回退:已软链 $DEST/$CANON_NAME → $(basename "$src")(供 ltx2_3_v2a.json 按官方名加载)"
    fi
  fi
  echo "$found" | while read -r f; do echo "  ✓ $(du -h "$f" | cut -f1)  $f"; done
  echo "✅ 完成。config(ltx2_3_v2a.json)按官方名 $CANON_NAME 加载,strength 0.8~1.0。"
else
  echo "  ✗ 未拿到权重。"
  echo "    - 官方 GATED:去 https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Foley-V2A 申请 access,拿 token 后 HF_TOKEN=hf_xxx 重跑;"
  echo "    - 或直接 SOURCE=community 用社区 un-gated 版重跑。"
  exit 1
fi
