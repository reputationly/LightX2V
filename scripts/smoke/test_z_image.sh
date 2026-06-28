#!/usr/bin/env bash
# =============================================================================
# Z-Image-Turbo 文生图(t2i)测试 —— bf16/int8 × 单卡/4卡 ulysses 四宫格
# 生成配置 + 用 test_model.sh 跑, 对比能否出图 / 画质 / 速度 / 显存。(参考 test_wan_i2v.sh 写法)
#
# 验证矩阵(本机 4×A100):
#   bf16      单卡   ①能不能出
#   bf16_ul2  2卡    ②bf16 多卡(ulysses)
#   int8      单卡   ③int8 单卡行不行(5.8G, 必装得下)
#   int8_ul2  2卡    ④int8 多卡 ulysses
#   ⚠️ z_image 有 30 个 attn head, ulysses 要求 seq_p_size 整除 30 → 只能 2/3/5/6/...,
#      **4 卡不行**(30÷4 除不尽, 会报错)。本机最多用 ul2 或 ul3。还有 bf16_ul3/int8_ul3。
#
# 三个已规避的坑:
#   1. 官方 config attn_type=flash_attn3 是 Hopper 专属 → A100 改 sage_attn2(此镜像 Wan 已验证)。
#   2. int8: model_path 仍指 bf16 目录(借 text_encoder/tokenizer/vae), transformer 用
#      dit_quantized_ckpt 指 int8 目录; scheme=int8-torchao(本机 proven)。
#   3. turbo 用 9 步; test_model.sh 的 body infer_steps 覆盖 config, 故 STEPS 默认 9。
#      (image 端点多收的 target_video_length 被 pydantic 忽略, 无害。)
#   4. rope_type 默认 flashinfer, 但本镜像没装 flashinfer(apply_rope_with_cos_sin_cache_inplace=None)
#      → 推理时 'NoneType' object is not callable。改 rope_type=torch(纯 torch RoPE, 无依赖)。
#
# 用法(服务器上, 先 scp 本脚本 + test_model.sh 到 /data/):
#   bash /data/test_z_image.sh                              # 默认: bf16 + int8 单卡(最简单先验)
#   CASES="bf16 bf16_ul2 int8 int8_ul2" bash /data/test_z_image.sh   # 四宫格(多卡用 2 卡)
#   CASES="int8_ul2 int8_ul3" bash /data/test_z_image.sh   # int8 2卡 vs 3卡
#   PROMPT="a cat astronaut" SEED=42 ASPECT="1:1" bash /data/test_z_image.sh
#
# 选填 env: CASES PROMPT NEG_PROMPT SEED STEPS ASPECT ATTN
# =============================================================================
set -uo pipefail
DATA=/data; CFGDIR="$DATA/lightx2v_configs"; OUT="$DATA/outputs"; HARNESS="$DATA/test_model.sh"
BF16_PATH="${BF16_PATH:-/nfs-data/models/Z-Image-Turbo}"            # 提供 transformer/text_encoder/tokenizer/vae
INT8_CKPT="${INT8_CKPT:-/nfs-data/models-int8/Z-Image-Turbo-int8}"  # int8 transformer 目录(non_block.safetensors)
# 官方默认提示词(scripts/z_image/z_image_turbo_t2i.sh): 含文字渲染⚡️+复杂场景, 好的考验图
PROMPT="${PROMPT:-Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.}"
NEG_PROMPT="${NEG_PROMPT:- }"
SEED="${SEED:-42}"; STEPS="${STEPS:-9}"; ASPECT="${ASPECT:-16:9}"; ATTN="${ATTN:-sage_attn2}"
CASES="${CASES:-bf16 int8}"                       # 默认先跑最简单的两个单卡; 全矩阵见用法
RUN="${RUN:-$(date +%m%d_%H%M)}"; RUNDIR="$OUT/z_image_$RUN"
mkdir -p "$CFGDIR" "$RUNDIR"
[ -f "$HARNESS" ] || { echo "缺 $HARNESS (先 scp test_model.sh 到 /data)"; exit 2; }

# ---- 生成配置(bf16/int8, 单卡/多卡)----
gen_cfg(){  # $1=变体 $2=卡数 -> 回显配置路径
  local v=$1 np=$2 cfg="$CFGDIR/z_image_${v}.json"
  {
    echo "{"
    echo "  \"aspect_ratio\": \"$ASPECT\","
    echo "  \"num_channels_latents\": 16,"
    echo "  \"infer_steps\": $STEPS,"
    echo "  \"attn_type\": \"$ATTN\","
    echo "  \"enable_cfg\": false,"
    echo "  \"sample_guide_scale\": 0.0,"
    echo "  \"rope_type\": \"torch\","
    echo "  \"patch_size\": 2"
    if [ "${v#int8}" != "$v" ]; then   # int8*
      echo "  ,\"dit_quantized\": true"
      echo "  ,\"dit_quant_scheme\": \"int8-torchao\""
      echo "  ,\"dit_quantized_ckpt\": \"$INT8_CKPT\""
    fi
    if [ "$np" -gt 1 ]; then            # 多卡: seq_p_size>1 时 set_config 自动开 seq_parallel
      echo "  ,\"parallel\": {\"seq_p_size\": $np, \"seq_p_attn_type\": \"ulysses\"}"
    fi
    echo "}"
  } > "$cfg"
  echo "$cfg"
}

FAILED=""
run(){  # $1=变体 $2=卡数
  local v=$1 np=$2 cfg; cfg=$(gen_cfg "$v" "$np")
  echo; echo "######### Z-Image t2i [$v] (np=$np, steps=$STEPS, attn=$ATTN) #########"
  echo "  config: $cfg"
  if ! NAME="z-image-$v" MODEL_CLS=z_image TASK=t2i MODEL_PATH="$BF16_PATH" CFG="$cfg" \
    PROMPT="$PROMPT" NEG_PROMPT="$NEG_PROMPT" OUT="$RUNDIR/${v}_s${SEED}.png" \
    NP="$np" SEED="$SEED" STEPS="$STEPS" \
    bash "$HARNESS"; then
    echo "!! 用例 [$v] 失败"; FAILED="$FAILED $v"
  fi
}

for c in $CASES; do
  case "$c" in
    bf16)     run bf16 1 ;;
    bf16_ul2) run bf16_ul2 2 ;;
    bf16_ul3) run bf16_ul3 3 ;;
    int8)     run int8 1 ;;
    int8_ul2) run int8_ul2 2 ;;
    int8_ul3) run int8_ul3 3 ;;
    bf16_ul4|int8_ul4) echo "跳过 [$c]: z_image 30 head 不被 4 整除, ulysses 4 卡不可用(用 ul2/ul3)"; FAILED="$FAILED $c";;
    *) echo "未知用例: $c (支持 bf16 bf16_ul2 bf16_ul3 int8 int8_ul2 int8_ul3)"; FAILED="$FAILED $c";;
  esac
done

echo; echo "===== 产物($RUNDIR, 下载整目录对比画质)====="
ls -lh "$RUNDIR"/*.png 2>/dev/null
echo "对比要点: int8 vs bf16 画质有无差 / 4卡比单卡快多少 / 各自显存峰值 / bf16能否4卡(z_image权重小, 预期可)"
[ -n "$FAILED" ] && { echo "!! 失败用例:$FAILED"; exit 1; }
echo "全部用例完成。"
