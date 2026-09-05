#!/usr/bin/env bash
# =============================================================================
# 本轮测试批任务封装(避免超长命令粘贴断行)。用法:
#   tmux new -s srloop -d 'bash /data/smoke/run_batch.sh srloop'
#   tmux new -s s2v    -d 'bash /data/smoke/run_batch.sh s2v'
#   tmux new -s seg121 -d 'bash /data/smoke/run_batch.sh seg121'
# 日志: /data/outputs/<任务名>.log
# =============================================================================
set -u
export LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
SMOKE=/data/smoke
OUTD=/data/outputs
PATCH="-v $SMOKE/seedvr_runner.py:/opt/LightX2V/lightx2v/models/runners/seedvr/seedvr_runner.py:ro"
SEEDVR_MP=/nfs-data/models/ByteDance-Seed/SeedVR2-3B

case "${1:?用法: run_batch.sh srloop|s2v|seg121}" in
  srloop)  # SeedVR2 3版 × 3素材(GPU 1)
    exec > >(tee -a "$OUTD/srloop.log") 2>&1
    for V in graded_horse_beach cmp_lx2v_morning_720p121 wan22_lightning_fp8_480P_5s_8step; do
      for C in 3b 7b 7b_sharp; do
        echo "########## $C x $V ##########"
        NAME=seedvr2-$C MODEL_CLS=seedvr2 TASK=sr NP=1 GPUS=1 STEPS=1 \
        EXTRA_VOL="$PATCH" MODEL_PATH=$SEEDVR_MP \
        CFG=$SMOKE/seedvr2_$C.json \
        VIDEO_PATH=/nfs-data/outputs/$V.mp4 \
        PROMPT="video super resolution" \
        OUT=$OUTD/sr_${C}_${V}.mp4 \
        bash $SMOKE/test_model.sh
      done
    done
    ;;
  s2v)     # Wan2.2-S2V int8 单卡(GPU 2, 端口 8200)
    exec > >(tee -a "$OUTD/s2v_int8.log") 2>&1
    NAME=wan-s2v-int8 MODEL_CLS=wan2.2_s2v TASK=s2v NP=1 GPUS=2 PORT=8200 STEPS=40 \
    MODEL_PATH=/nfs-data/models/Wan2.2-S2V-14B \
    CFG=$SMOKE/wan_s2v_int8.json \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    VIDEO_DURATION=5 HEALTH_TO=1800 \
    PROMPT="一个人对着镜头自然地说话，表情生动" \
    OUT=$OUTD/wan_s2v_int8_1card.mp4 \
    bash $SMOKE/test_model.sh
    ;;
  seg121)  # SeedVR2 3B 单段无缝版(GPU 3, 端口 8300)
    exec > >(tee -a "$OUTD/seg121.log") 2>&1
    NAME=seedvr2-3b-seg121 MODEL_CLS=seedvr2 TASK=sr NP=1 GPUS=3 PORT=8300 STEPS=1 FRAMES=121 \
    EXTRA_VOL="$PATCH" MODEL_PATH=$SEEDVR_MP \
    CFG=$SMOKE/seedvr2_3b_seg121.json \
    VIDEO_PATH=/nfs-data/outputs/graded_horse_beach.mp4 \
    PROMPT="video super resolution" \
    OUT=$OUTD/sr_3b_seg121_graded_horse_beach.mp4 \
    bash $SMOKE/test_model.sh
    ;;
  s2v_bf16)  # Wan2.2-S2V bf16 单卡(GPU 2, 端口 8200; DiT block offload + s2v_runner 设备错位补丁)
    exec > >(tee -a "$OUTD/s2v_bf16.log") 2>&1
    NAME=wan-s2v-bf16 MODEL_CLS=wan2.2_s2v TASK=s2v NP=1 GPUS=2 PORT=8200 STEPS=40 \
    EXTRA_VOL="-v $SMOKE/wan_s2v_runner.py:/opt/LightX2V/lightx2v/models/runners/wan/wan_s2v_runner.py:ro -v $SMOKE/wan_ops.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/s2v/wan_ops.py:ro -v $SMOKE/tensor.py:/opt/LightX2V/lightx2v/common/ops/tensor/tensor.py:ro -v $SMOKE/s2v_transformer_infer.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/s2v/transformer_infer.py:ro -v $SMOKE/s2v_model.py:/opt/LightX2V/lightx2v/models/networks/wan/s2v_model.py:ro" \
    MODEL_PATH=/nfs-data/models/Wan2.2-S2V-14B \
    CFG=$SMOKE/wan_s2v_bf16.json \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    VIDEO_DURATION=5 HEALTH_TO=1800 \
    PROMPT="一个人对着镜头自然地说话，表情生动" \
    OUT=$OUTD/wan_s2v_bf16_1card.mp4 \
    bash $SMOKE/test_model.sh
    ;;
  conv_s2v)  # 重转 S2V int8: 保留 ca/audio key(bf16 不量化), 只量化 attn/ffn。输出到新目录, 不碰旧的
    exec > >(tee -a "$OUTD/conv_s2v.log") 2>&1
    docker run --rm --gpus all --memory=180g --memory-swap=180g \
      -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V "$LX_IMG" \
      python /opt/LightX2V/tools/convert/converter.py \
      --source /nfs-data/models/Wan2.2-S2V-14B \
      --output /nfs-data/models-int8/Wan2.2-S2V-14B-int8-full \
      --output_name wan_s2v_int8 \
      --model_type wan_dit --linear_type int8 --quantized --save_by_block --no-parallel --device cpu \
      --ignore-keys none --ignore-quant-keys ca,audio
    ;;
  conv_s2v_v2)  # 再转 v2: self_attn 也保 bf16(S2V wan_ops 自定义算子不支持量化 self_attn), 只量化 cross_attn+ffn
    exec > >(tee -a "$OUTD/conv_s2v_v2.log") 2>&1
    docker run --rm --gpus all --memory=180g --memory-swap=180g \
      -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V "$LX_IMG" \
      python /opt/LightX2V/tools/convert/converter.py \
      --source /nfs-data/models/Wan2.2-S2V-14B \
      --output /nfs-data/models-int8/Wan2.2-S2V-14B-int8-v2 \
      --output_name wan_s2v_int8 \
      --model_type wan_dit --linear_type int8 --quantized --save_by_block --no-parallel --device cpu \
      --ignore-keys none --ignore-quant-keys ca,audio,self_attn
    ;;
  s2v_int8_v2)  # v2 int8 单卡(GPU 3, 端口 8300; cross_attn+ffn int8, self_attn/audio bf16)
    exec > >(tee -a "$OUTD/s2v_int8_v2.log") 2>&1
    NAME=wan-s2v-int8-v2 MODEL_CLS=wan2.2_s2v TASK=s2v NP=1 GPUS=3 PORT=8300 STEPS=40 \
    EXTRA_VOL="-v $SMOKE/wan_s2v_runner.py:/opt/LightX2V/lightx2v/models/runners/wan/wan_s2v_runner.py:ro" \
    MODEL_PATH=/nfs-data/models/Wan2.2-S2V-14B \
    CFG=$SMOKE/wan_s2v_int8_v2.json \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    VIDEO_DURATION=5 HEALTH_TO=1800 \
    PROMPT="一个人对着镜头自然地说话，表情生动" \
    OUT=$OUTD/wan_s2v_int8_v2_1card.mp4 \
    bash $SMOKE/test_model.sh
    ;;
  s2v_int8_full)  # 重转完成后: int8 单卡(指向新目录 -full)
    exec > >(tee -a "$OUTD/s2v_int8_full.log") 2>&1
    NAME=wan-s2v-int8 MODEL_CLS=wan2.2_s2v TASK=s2v NP=1 GPUS=2 PORT=8200 STEPS=40 \
    MODEL_PATH=/nfs-data/models/Wan2.2-S2V-14B \
    CFG=$SMOKE/wan_s2v_int8_full.json \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    VIDEO_DURATION=5 HEALTH_TO=1800 \
    PROMPT="一个人对着镜头自然地说话，表情生动" \
    OUT=$OUTD/wan_s2v_int8_1card.mp4 \
    bash $SMOKE/test_model.sh
    ;;
  it_distill)  # InfiniteTalk 4步蒸馏 单人(GPU 2, 端口 8200)
    exec > >(tee -a "$OUTD/it_distill.log") 2>&1
    NAME=infinitetalk-distill MODEL_CLS=infinitetalk TASK=s2v NP=1 GPUS=2 PORT=8200 STEPS=4 \
    EXTRA_VOL="-v $SMOKE/it_transformer_infer.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/infinitetalk/transformer_infer.py:ro" \
    MODEL_PATH=/nfs-data/models/Wan2.1-I2V-14B-480P \
    CFG=$SMOKE/infinitetalk_480p_single_distilled.json \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    HEALTH_TO=1800 \
    PROMPT="让角色根据音频内容自然说话" \
    NEG_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    OUT=$OUTD/infinitetalk_distill_1card.mp4 \
    bash $SMOKE/test_model.sh
    ;;
  it_run)  # 通用 InfiniteTalk 变体: $2=配置名(不带.json) $3=输出名 [$4=步数,默认4] [$5=GPU,默认2] [$6=端口,默认8200]
    CFGN="${2:?用法: it_run <配置名> <输出名> [steps] [gpu] [port]}"; OUTN="${3:?需要输出名}"
    ST="${4:-4}"; G="${5:-2}"; P="${6:-8200}"
    exec > >(tee -a "$OUTD/${OUTN}.log") 2>&1
    NAME="it-$OUTN" MODEL_CLS=infinitetalk TASK=s2v NP=1 GPUS="$G" PORT="$P" STEPS="$ST" \
    EXTRA_VOL="-v $SMOKE/it_transformer_infer.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/infinitetalk/transformer_infer.py:ro" \
    MODEL_PATH=/nfs-data/models/Wan2.1-I2V-14B-480P \
    CFG="$SMOKE/${CFGN}.json" \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    HEALTH_TO=1800 \
    PROMPT="让角色根据音频内容自然说话" \
    NEG_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    OUT="$OUTD/${OUTN}.mp4" \
    bash $SMOKE/test_model.sh
    ;;
  it_multi)  # InfiniteTalk 多人对话: $2=配置名 $3=输出名 [$4=GPU,默认1] [$5=端口,默认8210]
    CFGN="${2:?用法: it_multi <配置名> <输出名> [gpu] [port]}"; OUTN="${3:?需要输出名}"
    G="${4:-1}"; P="${5:-8210}"
    exec > >(tee -a "$OUTD/${OUTN}.log") 2>&1
    NAME="it-$OUTN" MODEL_CLS=infinitetalk TASK=s2v NP=1 GPUS="$G" PORT="$P" STEPS=4 \
    EXTRA_VOL="-v $SMOKE/it_transformer_infer.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/infinitetalk/transformer_infer.py:ro" \
    MODEL_PATH=/nfs-data/models/Wan2.1-I2V-14B-480P \
    CFG="$SMOKE/${CFGN}.json" \
    IMAGE=/opt/LightX2V/assets/inputs/audio/multi_person/seko_input.png \
    AUDIO="/opt/LightX2V/assets/inputs/audio/multi_person/p1.mp3,/opt/LightX2V/assets/inputs/audio/multi_person/p2.mp3" \
    HEALTH_TO=1800 \
    PROMPT="The video features a man and a woman standing by a bench in the park, their expressions tense and voices raised as they argue. The man gestures with both hands while the woman stands with her hands on her waist, brows furrowed in frustration." \
    NEG_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    OUT="$OUTD/${OUTN}.mp4" \
    bash $SMOKE/test_model.sh
    ;;
  it_hot)  # 热态稳态: 容器常驻(首发=冷态), 再连发 2 次热态提交计时。$2=配置名(默认蒸馏) $3=GPU $4=端口
    CFGN="${2:-infinitetalk_480p_single_distilled}"; G="${3:-2}"; P="${4:-8200}"
    exec > >(tee -a "$OUTD/it_hot.log") 2>&1
    KEEP=1 NAME=it-hot MODEL_CLS=infinitetalk TASK=s2v NP=1 GPUS="$G" PORT="$P" STEPS=4 \
    EXTRA_VOL="-v $SMOKE/it_transformer_infer.py:/opt/LightX2V/lightx2v/models/networks/wan/infer/infinitetalk/transformer_infer.py:ro" \
    MODEL_PATH=/nfs-data/models/Wan2.1-I2V-14B-480P \
    CFG="$SMOKE/${CFGN}.json" \
    IMAGE=/opt/LightX2V/assets/inputs/audio/seko_input.png \
    AUDIO=/opt/LightX2V/assets/inputs/audio/seko_input.mp3 \
    HEALTH_TO=1800 \
    PROMPT="让角色根据音频内容自然说话" \
    OUT="$OUTD/it_hot_1.mp4" \
    bash $SMOKE/test_model.sh
    API="http://localhost:$P"
    for i in 2 3; do
      T0=$(date +%s)
      BODY=$(python3 -c "import json;print(json.dumps({'prompt':'让角色根据音频内容自然说话','negative_prompt':'','save_result_path':'$OUTD/it_hot_$i.mp4','infer_steps':4,'seed':42,'image_path':'/opt/LightX2V/assets/inputs/audio/seko_input.png','audio_path':'/opt/LightX2V/assets/inputs/audio/seko_input.mp3'}))")
      TID=$(curl -sS -m 30 -X POST "$API/v1/tasks/video/" -H "Content-Type: application/json" -d "$BODY" | python3 -c "import json,sys;print(json.load(sys.stdin)['task_id'])")
      echo "hot run $i submitted tid=$TID"
      while true; do
        sleep 5
        ST_=$(curl -sS -m 10 "$API/v1/tasks/$TID/status" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or '')" 2>/dev/null)
        [ "$ST_" = "completed" ] && { echo "★ 热态第 $i 次: $(( $(date +%s)-T0 ))s"; break; }
        [ "$ST_" = "failed" ] && { echo "!! 热态第 $i 次失败"; break; }
        [ $(( $(date +%s)-T0 )) -gt 1200 ] && { echo "!! 热态第 $i 次超时"; break; }
      done
    done
    docker rm -f it-hot >/dev/null 2>&1
    ;;
  it_queue)  # A期收尾串行队列(单容器独占宿主, 干净数据): base40 → long22 → step8 → noofl → 720p
    exec > >(tee -a "$OUTD/it_queue.log") 2>&1
    for spec in \
      "infinitetalk_480p_single_base:it_base40:40:8110" \
      "infinitetalk_480p_single_distilled_long:it_long22:4:8120" \
      "infinitetalk_480p_single_distilled:it_step8:8:8130" \
      "infinitetalk_480p_single_distilled_noofl:it_noofl:4:8140" \
      "infinitetalk_720p_single_distilled:it_720p:4:8150"; do
      IFS=: read -r cfgn outn st port <<<"$spec"
      echo "########## [$(date +%T)] 队列: $outn (cfg=$cfgn steps=$st port=$port) ##########"
      bash "$SMOKE/run_batch.sh" it_run "$cfgn" "$outn" "$st" 1 "$port"
      sleep 15
    done
    echo "########## [$(date +%T)] 队列全部结束 ##########"
    ;;
  w1_run)  # W1 内部通用: $2=配置名 $3=输出名 $4=任务(t2v/i2v) $5=NP $6=STEPS [$7=RESIZE_MODE]
    CFGN="$2"; OUTN="$3"; TK="$4"; NPn="$5"; ST="$6"; RSZ="${7:-}"
    exec > >(tee -a "$OUTD/${OUTN}.log") 2>&1
    IMG_ARG=""; [ "$TK" = "i2v" ] && IMG_ARG="/opt/LightX2V/assets/inputs/imgs/girl.png"
    GP="0"; [ "$NPn" -gt 1 ] && GP="0,1,2,3"
    NAME="$OUTN" MODEL_CLS=wan2.2_moe TASK="$TK" \
    NP="$NPn" GPUS="$GP" PORT=8100 STEPS="$ST" SEED=42 RESIZE_MODE="$RSZ" HEALTH_TO=1800 \
    MODEL_PATH=/nfs-data/models/Wan-AI/Wan2.2-T2V-A14B \
    CFG="$SMOKE/${CFGN}.json" \
    IMAGE="$IMG_ARG" \
    PROMPT=$( [ "$TK" = "i2v" ] && echo "一位女子面对镜头微笑，微风拂动头发，背景是海边黄昏，电影感" || echo "夜晚的城市街道，霓虹灯闪烁，一辆汽车驶过湿润的路面，倒影清晰，电影感十足" ) \
    OUT="$OUTD/${OUTN}.mp4" \
    bash $SMOKE/test_model.sh
    ;;
  w1a)  # i2v 480p 单卡 A/B(torchao vs triton, 串行)
    bash "$SMOKE/run_batch.sh" w1_run w1_i2v_torchao  w1a_torchao i2v 1 4
    sleep 15
    bash "$SMOKE/run_batch.sh" w1_run w1_i2v_triton   w1a_triton  i2v 1 4
    ;;
  w1b)  # i2v 720p 4卡 A/B(生产形态)
    bash "$SMOKE/run_batch.sh" w1_run w1_i2v_torchao_ul4 w1b_torchao i2v 4 4 null
    sleep 15
    bash "$SMOKE/run_batch.sh" w1_run w1_i2v_triton_ul4  w1b_triton  i2v 4 4 null
    ;;
  w1c)  # t2v 生产 A/B(线上配置原样 vs 换triton)
    bash "$SMOKE/run_batch.sh" w1_run w1c_t2v_torchao w1c_torchao t2v 4 4
    sleep 15
    bash "$SMOKE/run_batch.sh" w1_run w1c_t2v_triton  w1c_triton  t2v 4 4
    ;;
  base40)  # t2v 40步 bf16 官方基线(画质上限, 同seed同prompt)
    bash "$SMOKE/run_batch.sh" w1_run t2v_base40_bf16 t2v_base40 t2v 1 40
    ;;
  vace1)  # VACE 首跑(R2V 纯参考图 smoke, 离线CLI)
    exec > >(tee -a "$OUTD/vace_smoke.log") 2>&1
    docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=0 \
      -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
      python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
      --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
      --config_json /data/smoke/wan22_moe_vace_a100.json \
      --prompt "一位女子在海边漫步，长发随风飘动，阳光洒在海面上，电影感十足" \
      --src_ref_images /opt/LightX2V/assets/inputs/imgs/girl.png \
      --save_result_path /data/outputs/vace_r2v_smoke.mp4 --seed 42
    ;;
  tprompt)  # 提示词真实感 A/B: 同seed同配置, 素句 vs 摄影级描述(t2v 单卡 triton)
    for arm in plain photo; do
      if [ "$arm" = "plain" ]; then
        P="山间瀑布，水流倾泻而下"
      else
        P="实拍风格，一条瀑布从青灰色岩壁倾泻而下，水雾弥漫，阳光在水汽中形成光斑，岩石表面湿润反光，苔藓细节清晰，长焦镜头，自然光，纪录片质感，35mm胶片，浅景深，真实物理水流"
      fi
      exec > >(tee -a "$OUTD/tprompt_${arm}.log") 2>&1
      NAME="tprompt-$arm" MODEL_CLS=wan2.2_moe TASK=t2v NP=1 GPUS=0 PORT=8100 STEPS=4 SEED=42 HEALTH_TO=1800 \
      MODEL_PATH=/nfs-data/models/Wan-AI/Wan2.2-T2V-A14B \
      CFG="$SMOKE/w1d_t2v_triton_1card.json" \
      PROMPT="$P" \
      OUT="$OUTD/tprompt_${arm}.mp4" \
      bash $SMOKE/test_model.sh
      sleep 15
    done
    ;;
  *) echo "未知任务: $1"; exit 1 ;;
esac
