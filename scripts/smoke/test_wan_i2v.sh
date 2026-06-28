#!/usr/bin/env bash
# =============================================================================
# Wan2.2-I2V 720p 测试 —— bf16 vs int8(单卡) vs int8(4卡 ulysses)
# 生成三套配置 + 用 test_model.sh 跑三连测, 对比画质/速度/显存。
#
# 输入:官方人像图 + i2v 提示词(可用 IMAGE/PROMPT 覆盖)。
# 基座(VAE/T5):Wan2.2-T2V-A14B;蒸馏 DiT(bf16):Wan2.2-Distill-Models 720p;
# int8:/nfs-data/models-int8/Wan2.2-I2V-720p-int8/{high,low}_noise(int8-torchao 块格式)。
#
# 用法(服务器, test_model.sh 也要在 /data):
#   bash /data/test_wan_i2v.sh                 # 三个都测
#   CASES="bf16 int8" bash /data/test_wan_i2v.sh
#   IMAGE=/opt/LightX2V/assets/inputs/imgs/woman.jpeg PROMPT="..." bash /data/test_wan_i2v.sh
# =============================================================================
set -uo pipefail
DATA=/data; CFGDIR="$DATA/lightx2v_configs"; OUT="$DATA/outputs"; HARNESS="$DATA/test_model.sh"
NFS=/nfs-data/models; NFS8=/nfs-data/models-int8
BASE="$NFS/Wan-AI/Wan2.2-T2V-A14B"                       # VAE/T5 基座
DISTILL="$NFS/Wan2.2-Distill-Models"
INT8="$NFS8/Wan2.2-I2V-720p-int8"
IMAGE="${IMAGE:-/opt/LightX2V/assets/inputs/imgs/girl.png}"
PROMPT="${PROMPT:-A young woman gently turns her head and smiles, hair softly moving in the breeze, warm natural light, cinematic, shallow depth of field}"
SEED="${SEED:-42}"; FRAMES="${FRAMES:-81}"
TASK="${TASK:-i2v}"                # i2v=单图; flf2v=首尾帧(需配 LAST_FRAME 尾帧图)
LAST_FRAME="${LAST_FRAME:-}"       # flf2v 尾帧图路径(i2v 留空)
NEG_PROMPT="${NEG_PROMPT:-}"
RUN="${RUN:-$(date +%m%d_%H%M)}"; RUNDIR="$OUT/wan_i2v_$RUN"   # 每轮独立时间戳目录, 不和旧结果混
RES="${RES:-480}"   # 480=默认(480×832, 单卡也能跑); 720=匹配在线版832×1104(传 resize_mode=null 触发 max_area 路径)
if [ "$RES" = "720" ]; then
  RESIZE_MODE="${RESIZE_MODE:-null}"     # 触发 wan i2v 的 max_area=target_h×w 分辨率路径
  CASES="${CASES:-int8_ul4}"             # 720p 默认只跑 int8 4卡(87s/35.3G有余量); 实测 bf16单卡险过(40.3G/183s无余量), int8单卡 OOM —— 要复现单卡: CASES="bf16 int8 int8_ul4"
else
  CASES="${CASES:-bf16 int8 int8_ul4}"   # bf16_ul4 已验证必 CPU OOM(每rank复制57G→4×276G>256G, 加载阶段 SIGKILL), 默认不跑; CASES=bf16_ul4 可复现(harness 240G上限护宿主)
fi
mkdir -p "$CFGDIR" "$RUNDIR"
[ -f "$HARNESS" ] || { echo "缺 $HARNESS (先 scp test_model.sh 到 /data)"; exit 2; }

# ---- 生成三套配置(基于你们能跑通的 int8-torchao 配方 + 官方 i2v 结构)----
gen_cfg(){  # $1=变体 -> 回显配置路径
  local v=$1 out="$CFGDIR/wan_i2v_720p_${v}.json"
  python3 - "$v" "$out" "$DISTILL" "$INT8" <<'PY'
import json,sys
v,out,distill,int8=sys.argv[1:5]
c={
 "infer_steps":4,"target_video_length":81,"text_len":512,
 "target_height":720,"target_width":1280,   # 720p面积; 仅当请求体传 resize_mode=null(RES=720)时才走这条 max_area 路径算出832×1104, 否则server默认adaptive=480p
 "self_attn_1_type":"sage_attn2","cross_attn_1_type":"sage_attn2","cross_attn_2_type":"sage_attn2",
 "sample_guide_scale":[3.5,3.5],"sample_shift":5.0,"enable_cfg":False,
 "cpu_offload":False,"t5_cpu_offload":True,"vae_cpu_offload":False,
 "use_image_encoder":False,"boundary_step_index":2,
 "denoising_step_list":[1000,750,500,250],"rope_type":"torch",
}
if v.startswith("bf16"):
    # bf16 装不下40G -> offload; Wan MoE 必须用 "model" 粒度(一次换一个28.5G专家); "block" 会黑屏
    c["cpu_offload"]=True; c["offload_granularity"]="model"
    c["high_noise_original_ckpt"]=f"{distill}/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors"
    c["low_noise_original_ckpt"] =f"{distill}/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors"
    if v=="bf16_ul4":
        c["parallel"]={"seq_p_size":4,"seq_p_attn_type":"ulysses"}   # ulysses每卡放权重+offload换专家
else:  # int8 / int8_ul4 (28G, 单卡直接装下)
    c["dit_quantized"]=True; c["dit_quant_scheme"]="int8-torchao"
    c["high_noise_quantized_ckpt"]=f"{int8}/high_noise"
    c["low_noise_quantized_ckpt"] =f"{int8}/low_noise"
    if v=="int8_ul4":
        c["parallel"]={"seq_p_size":4,"seq_p_attn_type":"ulysses"}
json.dump(c,open(out,"w"),indent=2)
print(out)
PY
}

FAILED=""
run(){  # $1=变体 $2=卡数
  local v=$1 np=$2 cfg; cfg=$(gen_cfg "$v")
  echo; echo "######### Wan2.2-I2V 720p [$v] (np=$np) #########"
  if ! NAME="wan-$TASK-$v" MODEL_CLS=wan2.2_moe_distill TASK="$TASK" MODEL_PATH="$BASE" CFG="$cfg" \
    PROMPT="$PROMPT" NEG_PROMPT="$NEG_PROMPT" IMAGE="$IMAGE" LAST_FRAME="$LAST_FRAME" OUT="$RUNDIR/${v}_s${SEED}.mp4" \
    NP="$np" FRAMES="$FRAMES" SEED="$SEED" STEPS=4 RESIZE_MODE="${RESIZE_MODE:-}" \
    bash "$HARNESS"; then
    echo "!! 用例 [$v] 失败"; FAILED="$FAILED $v"
  fi
}

for c in $CASES; do
  case "$c" in
    bf16)     run bf16 1 ;;
    bf16_ul4) run bf16_ul4 4 ;;
    int8)     run int8 1 ;;
    int8_ul4) run int8_ul4 4 ;;
    *) echo "未知用例: $c"; FAILED="$FAILED $c";;
  esac
done

echo; echo "===== 产物($RUNDIR, 下载整个目录对比画质)====="
ls -lh "$RUNDIR"/*.mp4 2>/dev/null
echo "对比要点: int8 vs bf16 画质崩没崩 / 4卡比单卡快多少 / 显存峰值"
[ -n "$FAILED" ] && { echo "!! 失败用例:$FAILED"; exit 1; }
echo "全部用例完成。"
