#!/usr/bin/env bash
# =============================================================================
# Wan2.2-Lightning 4步蒸馏 LoRA(T2V-A14B Seko-V2.0, high/low 各~1.23G)
# 两步走: 下载在 238(有 modelscope 环境), 转换在计算节点(要 arm64 容器)
#
# 用法:
#   [238]      bash /nfs-models/wuhanjisuan894/smoke/download_vace_lora.sh dl
#   [计算节点]  bash /data/smoke/download_vace_lora.sh convert
# =============================================================================
set -u
SUB="Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0"

case "${1:?用法: download_vace_lora.sh dl(238上)|convert(计算节点上)}" in
  dl)
    DEST=/nfs-models/wuhanjisuan894/models
    mkdir -p "$DEST/Wan2.2-Lightning"
    echo "==== [$(date +%T)] ModelScope 下载 lightx2v/Wan2.2-Lightning (Seko-V2.0, ~2.5G)"
    modelscope download --model "lightx2v/Wan2.2-Lightning" \
      "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors" \
      "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors" \
      --local_dir "$DEST/Wan2.2-Lightning" || { echo "!! 下载失败"; exit 1; }
    for f in high_noise_model low_noise_model; do
      sz=$(stat -c %s "$DEST/$SUB/$f.safetensors" 2>/dev/null || echo 0)
      [ "$sz" -ge 1000000000 ] || { echo "✗ $f 缺失或不完整"; exit 1; }
      echo "✓ $f ($(numfmt --to=iec $sz))"
    done
    echo "==== 下载完成, 去任一计算节点跑: bash /data/smoke/download_vace_lora.sh convert"
    ;;
  convert)
    LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
    echo "==== [$(date +%T)] 容器内转换为 lightx2v 格式"
    docker run --rm -v /nfs-data:/nfs-data -v /data:/data -e PYTHONPATH=/opt/LightX2V "$LX_IMG" bash -c "
      python /opt/LightX2V/tools/extract/convert_lightning_to_x2v_lora.py \
        --input-lora /nfs-data/models/$SUB/high_noise_model.safetensors \
        --output-lora /nfs-data/models/$SUB/high_noise_model_x2v.safetensors --to-bf16 && \
      python /opt/LightX2V/tools/extract/convert_lightning_to_x2v_lora.py \
        --input-lora /nfs-data/models/$SUB/low_noise_model.safetensors \
        --output-lora /nfs-data/models/$SUB/low_noise_model_x2v.safetensors --to-bf16"
    ls -lh "/nfs-data/models/$SUB/"
    echo "==== [$(date +%T)] 完成, 可发车: tmux new -s vace_lora -d 'bash /data/smoke/vace_lora.sh'"
    ;;
  *) echo "未知: $1"; exit 1 ;;
esac
