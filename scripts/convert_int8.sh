#!/usr/bin/env bash
# =============================================================================
# 批量把下好的 bf16 DiT 量化成 int8(docker 内跑 LightX2V converter)
# 官方原生 model_type, 内置 key 映射, 无需手填 target-keys。
#
# 敏感层策略(重要):
#   - 用原生 model_type, converter 内置 target_keys 只量化 attn/mlp, 自动把 embedder/
#     norm/输入输出投影留 bf16; wan_dit 还额外 ignore ca/audio。→ 层级敏感已护住。
#   - 块级敏感(整块不量化)只有 LTX 需要([0,43-47]), Wan 全块量化 proven 好。
#   - Qwen/Z-Image/Hunyuan 块级敏感未知: 本脚本先按内置转, 测出崩了再用 conv 的第6参
#     传 --ignore-quant-keys(如 'transformer_blocks.0.,transformer_blocks.N.')重转。
#
# int8 把握:
#   Wan 三个(I2V/S2V/Animate)—— int8 proven 好(你们 wan22_t2v_int8 同源)
#   Qwen-Image / Z-Image / Hunyuan —— 机制支持, 画质待验(转了和 bf16 对比测)
#
# 标签(默认全转, Wan 优先排前面):
#   wan_i2v_high wan_i2v_low wan_s2v wan_animate qwen_image z_image hunyuan_t2v
#
# 用法(服务器, 先 scp 到 /data):
#   tmux new -s cv -d 'bash /data/convert_int8.sh'      # 挂后台(转~280G, 小时级)
#   tail -f /nfs-data/convert_int8.log
#   MODELS="wan_i2v_high wan_i2v_low" bash /data/convert_int8.sh   # 只转 Wan I2V
# 输出: /nfs-data/models-int8/<模型>/  (int8 block 格式, 推理 dit_quantized_ckpt 指它)
# converter 跑在 docker 内: import 需 --gpus all, 计算走 --device cpu。
# =============================================================================
set -u
IMG="${LX_IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-20260625-0256-2cde4721}"
SRC="${SRC:-/nfs-data/models}"
OUT="${OUT:-/nfs-data/models-int8}"
LOG=/nfs-data/convert_int8.log
MODELS="${MODELS:-wan_i2v_high wan_i2v_low wan_s2v wan_animate qwen_image z_image hunyuan_t2v}"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "镜像=$IMG"
echo "源=$SRC | 输出=$OUT | 模型=$MODELS"

conv(){  # $1=标签 $2=源(文件/目录) $3=model_type $4=输出子目录 $5=output_name
         # $6=可选: --ignore-quant-keys(csv), 跳过这些块/层的量化(测出块敏感后填, 如 LTX 的 transformer_blocks.0.,...)
  local tag=$1 src=$2 mt=$3 sub=$4 name=$5 iq="${6:-}" dst="$OUT/$4"
  echo "---------------------------------------------"
  echo ">>> [$(date +%T)] [$tag] $mt ${iq:+(跳过量化: $iq)}"
  echo "    源: $src"
  echo "    出: $dst"
  if [ ! -e "$src" ]; then echo "  !! 源不存在, 跳过"; FAILED="$FAILED $tag"; return; fi
  if [ -f "$dst/.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  跳过 [$tag]: 已转过(FORCE=1 可强制重转) -> $(du -sh "$dst" 2>/dev/null | cut -f1)"; return
  fi
  rm -rf "$dst"; mkdir -p "$dst"   # 清掉上次崩留下的半成品碎块, 从干净目录重转
  sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true   # 清 page cache, 防多文件读写累积压内存
  local extra=(); [ -n "$iq" ] && extra+=(--ignore-quant-keys "$iq")
  # 内存上限: 超了 cgroup 杀容器(干净失败), 不会拖垮宿主内核; --no-parallel 单线程降峰值
  if docker run --rm --gpus all --memory="${MEM:-180g}" --memory-swap="${MEM:-180g}" \
       -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V "$IMG" \
       python /opt/LightX2V/tools/convert/converter.py \
       --source "$src" --output "$dst" --output_name "$name" \
       --model_type "$mt" --linear_type int8 --quantized --save_by_block --no-parallel --device cpu \
       ${extra[@]+"${extra[@]}"}; then
    echo "  OK [$tag]  -> $(du -sh "$dst" 2>/dev/null | cut -f1)"; touch "$dst/.done"
  else
    echo "  !! 转换失败 [$tag](见上方 converter 报错)"; FAILED="$FAILED $tag"
  fi
}

for m in $MODELS; do
case "$m" in
  # ---- Wan 三个: int8 proven ----
  wan_i2v_high) conv wan_i2v_high \
    "$SRC/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors" \
    wan_dit "Wan2.2-I2V-720p-int8/high_noise" wan_i2v_high_int8 ;;
  wan_i2v_low)  conv wan_i2v_low \
    "$SRC/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors" \
    wan_dit "Wan2.2-I2V-720p-int8/low_noise" wan_i2v_low_int8 ;;
  wan_s2v)      conv wan_s2v     "$SRC/Wan2.2-S2V-14B"             wan_dit         "Wan2.2-S2V-14B-int8"     wan_s2v_int8 ;;
  wan_animate)  conv wan_animate "$SRC/Wan2.2-Animate-14B"        wan_animate_dit "Wan2.2-Animate-14B-int8" wan_animate_int8 ;;
  # ---- 图像/Hunyuan: 机制支持, 画质待验 ----
  qwen_image)   conv qwen_image  "$SRC/Qwen-Image/transformer"    qwen_image_dit  "Qwen-Image-int8"         qwen_image_int8 ;;
  z_image)      conv z_image     "$SRC/Z-Image-Turbo/transformer" z_image_dit     "Z-Image-Turbo-int8"      z_image_int8 ;;
  # 转蒸馏 DiT(distill 配置实际加载的那个), 不是基座 transformer
  hunyuan_t2v)  conv hunyuan_t2v \
    "$SRC/hunyuanvideo-1.5/distill_models/480p_t2v/hy1.5_t2v_480p_lightx2v_4step.safetensors" \
    hunyuan_dit "hunyuanvideo-1.5-int8/480p_t2v_distill" hunyuan_t2v_int8 ;;
  *) echo "!! 未知模型标签: $m"; FAILED="$FAILED $m" ;;
esac
done

echo "============================================="
echo "==== [$(date +%T)] 全部完成 ===="
du -sh "$OUT"/* 2>/dev/null
if [ -n "$FAILED" ]; then echo "!! 以下失败:$FAILED"; exit 1; fi
echo "完成, 全部成功。"
