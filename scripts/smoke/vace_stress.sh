#!/usr/bin/env bash
# VACE 编辑档压测: N 个单卡实例并发(每实例绑一张卡), 每实例连跑 R 轮 R2V LoRA
# 监控: 2s 采样 host可用内存/每卡显存 → CSV;每轮耗时入 log
# 用法: tmux new -s vst -d 'bash /data/smoke/vace_stress.sh 1'   # 单实例基线
#       tmux new -s vst -d 'bash /data/smoke/vace_stress.sh 2'   # 2并发
#       tmux new -s vst -d 'bash /data/smoke/vace_stress.sh 4'   # 4并发(形态B可行性)
# 要求: 安静宿主(无其他任务)
set -u
N="${1:?用法: vace_stress.sh <并发数 1|2|4> [轮数,默认3]}"
R="${2:-3}"
LX_IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/lightx2v:arm64-a100-latest
TAG="vst_n${N}_$(date +%H%M)"
OUTD=/data/outputs
CSV="$OUTD/${TAG}.csv"
exec > >(tee -a "$OUTD/${TAG}.log") 2>&1
echo "==== [$(date +%T)] VACE压测 并发=$N 轮数=$R tag=$TAG"

# ---- 监控(2s采样) ----
echo "ts,host_avail_mb,$(seq -s, 0 3 | sed 's/[0-9]*/gpu&_mb/g')" > "$CSV"
( while true; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    gpus=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,)
    echo "$(date +%T),$avail,$gpus" >> "$CSV"
    sleep 2
  done ) & MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

# ---- N 实例并发, 各绑一卡, 各跑 R 轮(错峰点火120s/台, 还原生产启动纪律) ----
for i in $(seq 0 $((N-1))); do
  (
    sleep $((i*120))
    for r in $(seq 1 $R); do
      t0=$(date +%s)
      docker run --rm --gpus all --memory=240g --memory-swap=240g -e CUDA_VISIBLE_DEVICES=$i \
        -v /data:/data -v /nfs-data:/nfs-data -e PYTHONPATH=/opt/LightX2V \
        -v /data/smoke/vace_model.py:/opt/LightX2V/lightx2v/models/networks/wan/vace_model.py:ro \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$LX_IMG" \
        python -m lightx2v.infer --model_cls wan2.2_moe_vace --task vace \
        --model_path /nfs-data/models/Wan2.2-VACE-Fun-A14B \
        --config_json /data/smoke/wan22_moe_vace_a100_int8_lora4step_720p_rife.json \
        --prompt "一位女子在海边漫步，长发随风飘动，阳光洒在海面上，实拍风格" \
        --src_ref_images /opt/LightX2V/assets/inputs/imgs/woman.jpeg \
        --save_result_path "$OUTD/${TAG}_g${i}_r${r}.mp4" --seed $((42+r)) \
        > "$OUTD/${TAG}_g${i}_r${r}.runlog" 2>&1
      t1=$(date +%s)
      echo "[$(date +%T)] 实例g$i 第${r}轮 结束 耗时$((t1-t0))s rc=$?"
    done
  ) &
done
wait
kill $MON 2>/dev/null || true

echo "==== [$(date +%T)] 全部结束, 汇总 ===="
grep "结束 耗时" "$OUTD/${TAG}.log" | sort
awk -F, 'NR>1{if($2<m||m==0)m=$2} END{printf "host可用内存最低: %d MB\n", m}' "$CSV"
for g in 0 1 2 3; do
  awk -F, -v c=$((g+3)) 'NR>1{if($c>x)x=$c} END{if(x>0)printf "GPU'"$g"' 显存峰值: %d MB\n", x}' "$CSV"
done
ls -lh "$OUTD/${TAG}"_g*.mp4 2>/dev/null | wc -l | xargs echo "成功出片数:"
