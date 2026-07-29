#!/usr/bin/env bash
# =============================================================================
# 下载 Bernini-R-S2V(rzgar, 在 Bernini-R/Wan2.2 上加语音驱动对口型 = 官方 Wan2.2-S2V
# 结构的微调) 到 NFS, 用于经 LightX2V 原生 wan2.2_s2v 适配。规划见对话记录。
# 沿用 download_scail2_bernini.sh 的套路:ModelScope 主源(国内快)→ hf-mirror 回退,
# 实时测速 + 断点续传 + 失败追踪。只写新目录, 不碰已有模型。
#
# ⚠️ A100(sm_80)硬件不支持 FP8 tensor core, 所以:
#   - 只下 fp16(bf16 跑, 或后续用 LightX2V converter 量化到 int8-torchao);
#   - 跳过 rzgar 的 fp8_scaled(A100 无加速) 和 int8-convrot(ComfyUI 方案, 与 torchao 不兼容);
#   - ComfyUI 自定义节点 / 示例视频 一律不下(我们走 LightX2V, 不走 ComfyUI)。
#
# 模型(标签 | 约大小 | 说明):
#   ---- 默认下 ----
#   s2v_fp16      ~65G(32.6G×2)  high+low noise 双专家 DiT(fp16, 含 audio_injector/
#                                casual_audio_encoder/cond_encoder/frame_packer, key 已对齐 LightX2V)
#   s2v_wav2vec   ~0(软链复用)   wav2vec2 音频前端。LightX2V 走 HF 目录加载(Wav2Vec2Processor/
#                                Wav2Vec2ForCTC.from_pretrained), 需 config.json/preprocessor_config.json/
#                                vocab.json + 权重一整套; rzgar 的单文件 wav2vec2_large_english_fp16.
#                                safetensors 是 ComfyUI 打包格式(同一份权重的 fp16, 正好半个大小),
#                                from_pretrained 读不了, 故不下 —— 改为复用官方 Wan2.2-S2V-14B 的
#                                wav2vec2-large-xlsr-53-english/ 目录(缺了才整目录拉)。
#
#   ---- 可选(默认不下; A100 不推荐, 仅备查) ----
#   s2v_fp8       ~33G           rzgar 的 fp8_scaled(A100 无 fp8 加速, 只省显存且带反量化开销, 不建议)
#
# 复用(GPU 机上 base Bernini 部署已有, S2V 直接共用, 无需重下):
#   VAE(Wan2.1_VAE.pth) / T5 文本编码器(umt5)  —— 不在本脚本内, 转换后在 lx2v 目录里软链
#   wav2vec2-large-xlsr-53-english/            —— 由本脚本的 s2v_wav2vec 负责软链
#
# 用法(服务器上, 先 scp 脚本到 /root; 下载产物一律落 NFS 的 $DEST):
#   tmux new -s dl_s2v -d 'bash /root/download_bernini_s2v.sh'    # 挂后台(默认下 fp16 + wav2vec)
#   tail -f "$(dirname "${DEST:-/nfs-data/models}")/dl_bernini_s2v.log"  # 看速度+进度
#   MODELS="s2v_wav2vec" bash /root/download_bernini_s2v.sh       # 只挂音频前端(软链/补齐)
#   MODELS="s2v_fp8" bash /root/download_bernini_s2v.sh           # (不推荐)只下 fp8
# 中断后重跑本脚本会自动续传。
# =============================================================================
set -u
# DEST 自动适配:计算节点有 /nfs-data 软链;manager 节点用真身 /nfs-models/wuhanjisuan894/models
DEST="${DEST:-$([ -d /nfs-data/models ] && echo /nfs-data/models || echo /nfs-models/wuhanjisuan894/models)}"
LOG="${LOG:-$(dirname "$DEST")/dl_bernini_s2v.log}"
MODELS="${MODELS:-s2v_fp16 s2v_wav2vec}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
FAILED=""

echo "==== [$(date +%T)] 目标=$DEST | 模型=$MODELS | 主源=ModelScope, 回退=hf-mirror"
if ! python3 -m pip install -q -U modelscope "huggingface_hub[cli]" >/tmp/dl_pip.log 2>&1; then
  echo "pip/依赖安装失败(看 /tmp/dl_pip.log), 试 apt install python3-pip 后重跑"; tail -5 /tmp/dl_pip.log; exit 2
fi

# ---- 实时聚合速度监视器 ----
mon(){
  local base prev prevt
  base=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0); prev=$base; prevt=$(date +%s)
  while true; do
    sleep 15
    local now nowt; now=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0); nowt=$(date +%s)
    awk -v b="$base" -v p="$prev" -v n="$now" -v dt="$((nowt-prevt))" -v ts="$(date +%T)" 'BEGIN{
      if(dt<=0)dt=1; printf "[%s] ▼ 本次已下 %.1f GB | 当前 %.0f MB/s\n", ts, (n-b)/1073741824, (n-p)/dt/1048576 }'
    prev=$now; prevt=$nowt
  done
}
mon & MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

# ---- 下载封装: ModelScope 优先, 失败回退 HF。$1=MS仓 $2=目标子目录 $3=HF回退仓 [$4..=文件/pattern(留空=整仓)] ----
dl(){
  local msid=$1 sub=$2 hfid=$3; shift 3
  local dest="$DEST/$sub" files=("$@")
  echo ">>> [$(date +%T)] ModelScope: $msid -> $dest  files=[${files[*]:-整仓}]"
  if modelscope download --model "$msid" ${files[@]+"${files[@]}"} --local_dir "$dest"; then
    echo "  OK(MS) $msid"; return
  fi
  echo "  !! ModelScope 失败, 回退 hf-mirror: $hfid"
  if python3 - "$hfid" "$dest" ${files[@]+"${files[@]}"} <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
pats = sys.argv[3:] or None
snapshot_download(repo_id=repo, local_dir=dest, max_workers=4, allow_patterns=pats)
print("OK")
PY
  then echo "  OK(HF) $hfid"; else echo "  !! 两源都失败: $msid"; FAILED="$FAILED $msid"; fi
}

for m in $MODELS; do
case "$m" in
  # ---- 默认: fp16 双专家 DiT + wav2vec2 音频前端 ----
  s2v_fp16)     dl rzgar/Bernini-R-S2V "Bernini-R-S2V" rzgar/Bernini-R-S2V \
                   "Bernini-R-S2V-FP16/wan2.2_bernini_r_high_noise_fp16_s2v.safetensors" \
                   "Bernini-R-S2V-FP16/wan2.2_bernini_r_low_noise_fp16_s2v.safetensors" ;;
  # LightX2V 的 WanS2VRunner.load_audio_encoder 拼的是 <model_path>/wav2vec2-large-xlsr-53-english,
  # 再交给 Wav2Vec2Processor/Wav2Vec2ForCTC.from_pretrained —— 必须是完整 HF 目录, 单文件读不了。
  # 官方 Wan2.2-S2V-14B 里那份就是同一模型(fp32), 有就软链复用, 没有才整目录拉。
  s2v_wav2vec)  W2V_SRC="$DEST/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"
                W2V_DST="$DEST/Bernini-R-S2V/wav2vec2-large-xlsr-53-english"
                if [ -f "$W2V_SRC/preprocessor_config.json" ]; then
                  echo ">>> [$(date +%T)] 复用已有 wav2vec 目录, 软链 $W2V_DST -> $W2V_SRC"
                  mkdir -p "$(dirname "$W2V_DST")" && ln -sfn "$W2V_SRC" "$W2V_DST" \
                    || { echo "  !! 软链失败"; FAILED="$FAILED s2v_wav2vec"; }
                else
                  echo ">>> [$(date +%T)] 未见官方 wav2vec 目录, 整目录拉取"
                  dl Wan-AI/Wan2.2-S2V-14B "Bernini-R-S2V" Wan-AI/Wan2.2-S2V-14B \
                     "wav2vec2-large-xlsr-53-english/*"
                fi ;;
  # ---- 可选(A100 不推荐, 仅备查) ----
  s2v_fp8)      dl rzgar/Bernini-R-S2V "Bernini-R-S2V" rzgar/Bernini-R-S2V \
                   "Bernini-R-S2V-FP8/wan2.2_bernini_r_high_noise_fp8_scaled_s2v.safetensors" \
                   "Bernini-R-S2V-FP8/wan2.2_bernini_r_low_noise_fp8_scaled_s2v.safetensors" ;;
  *) echo "!! 未知标签: $m"; FAILED="$FAILED $m";;
esac
done

kill $MON 2>/dev/null || true
echo "==== [$(date +%T)] 全部完成, 校对大小 ===="
for f in \
  "Bernini-R-S2V/Bernini-R-S2V-FP16/wan2.2_bernini_r_high_noise_fp16_s2v.safetensors" \
  "Bernini-R-S2V/Bernini-R-S2V-FP16/wan2.2_bernini_r_low_noise_fp16_s2v.safetensors" \
  "Bernini-R-S2V/wav2vec2-large-xlsr-53-english/preprocessor_config.json" \
  "Bernini-R-S2V/Bernini-R-S2V-FP8/wan2.2_bernini_r_high_noise_fp8_scaled_s2v.safetensors" \
  "Bernini-R-S2V/Bernini-R-S2V-FP8/wan2.2_bernini_r_low_noise_fp8_scaled_s2v.safetensors" ; do
  [ -e "$DEST/$f" ] && du -sh "$DEST/$f" 2>/dev/null
done
if [ -n "$FAILED" ]; then echo "!! 以下有失败, 需重跑(会续传):$FAILED"; exit 1; fi
echo "完成, 全部成功。"
echo "下一步: 用 converter 产出 lx2v 目录后, 把四件套软链进去(runner 的 model_path 指向 lx2v 目录,"
echo "        音频前端读的是 <model_path>/wav2vec2-large-xlsr-53-english, 少了会在初始化处直接失败):"
echo "        for d in $DEST/Bernini-R-S2V-lx2v-{high,low}; do"
echo "          for a in Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth google wav2vec2-large-xlsr-53-english; do"
echo "            ln -sfn $DEST/Wan2.2-S2V-14B/\$a \$d/\$a; done; done"
echo "        要 A100 int8 提速再用 LightX2V tools/convert/converter.py 从 fp16 量化到 int8-torchao。"
