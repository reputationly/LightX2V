#!/usr/bin/env python3
# =============================================================================
# MiniMax-H3 NF4 单卡冒烟测试(DiffSynth-Studio 后端)
#
# 目的:在 A100-40G + 鲲鹏 ARM 上把"能不能跑"这件事变成三个数 —— 显存峰值 / 单条耗时 / 画质。
#
# 为什么不照抄模型卡的配方:官方 README 给的是 disk/cpu offload("6G 显存就能跑"),
#   但那意味着每步都要把 16G 的 NF4 权重从内存搬进显存 —— 50 步 = 800GB 传输,
#   本机是 A100 PCIE 无 NVLink(实测 ~20GB/s), 纯传输就 30+ 分钟, 计算全被淹没。
#   NF4 单路权重才 32G(DiT 16.0 + TE 14.3 + video_vae 1.5 + audio_vae 0.26),
#   40G 卡还剩 ~8G 给激活, 480p/124帧 约 12k token 大概率够 → **默认全常驻(--offload none)**,
#   OOM 了再退 --offload cpu。别一上来就 offload。
#
# 权重全部指向 NFS 本地路径, 不联网(HF/MS 都设了 offline)。
#
# 用法(GPU 节点上, 先装好 DiffSynth-Studio):
#   git clone https://github.com/modelscope/DiffSynth-Studio.git && cd DiffSynth-Studio
#   pip install -e ".[quant]"
#   # 环境自检(不加载权重, 秒回):
#   python smoke_minimax_h3_nf4.py --dry-run
#   # 第一测:480p 短片, 全常驻
#   CUDA_VISIBLE_DEVICES=0 python smoke_minimax_h3_nf4.py --out /root/h3_t2va.mp4
#   # OOM 了再退一步(慢很多, 但能出片):
#   CUDA_VISIBLE_DEVICES=0 python smoke_minimax_h3_nf4.py --offload cpu
#   # 想先快速看通不通, 把步数砍到 10(画质会差, 只验流程):
#   CUDA_VISIBLE_DEVICES=0 python smoke_minimax_h3_nf4.py --steps 10
#
# ⚠️ 这个脚本是照模型卡文档的 API 写的 + 运行时对 ModelConfig 做签名自省来适配参数名。
#    DiffSynth 主线在快速迭代, 首跑若报 TypeError, 看脚本打印的 "解析出的 ModelConfig kwargs"
#    对照 diffsynth/pipelines/minimax_h3_audio_video.py 改一行即可, 不用重写。
# =============================================================================
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

# 不联网:权重都在 NFS 上, 任何一次隐式下载都会在无外网的节点上卡死
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

DEFAULT_PROMPT = (
    "A young woman sits by a rain-streaked cafe window at dusk, warm tungsten light from a "
    "table lamp raking across her face. She looks up from her book, smiles softly, and says "
    'in English: "It finally stopped raining." Shallow depth of field, 50mm lens, '
    "handheld micro-movement, cinematic color grading, film grain. "
    "Ambient soundscape: gentle rain tapering off outside, muffled cafe chatter, "
    "a ceramic cup set down on a saucer."
)


def find_models_root() -> Path:
    """DEST 自动适配:计算节点有 /nfs-data 软链, manager 用真身。与下载脚本保持一致。"""
    for p in ("/nfs-data/models", "/nfs-models/wuhanjisuan894/models"):
        if Path(p).is_dir():
            return Path(p)
    return Path("/nfs-data/models")


def human(n: float) -> str:
    return f"{n:.1f}"


def preflight(args) -> dict:
    """环境自检:把 ARM + A100 上最容易崩的几处提前暴露出来, 而不是等加载完 32G 才炸。"""
    info = {}
    print("=" * 72)
    print("环境自检")
    print("=" * 72)

    import torch

    info["torch"] = torch.__version__
    info["platform"] = f"{os.uname().sysname}-{os.uname().machine}"
    print(f"  torch          : {torch.__version__}  ({info['platform']})")

    if not torch.cuda.is_available():
        print("  !! CUDA 不可用, 后面必崩")
        info["cuda"] = False
        return info
    info["cuda"] = True
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    info["gpu"] = name
    info["sm"] = f"sm_{cap[0]}{cap[1]}"
    info["vram_total_gb"] = round(total, 1)
    print(f"  GPU            : {name}  {info['sm']}  {total:.1f} GB  ×{torch.cuda.device_count()} 可见")
    if cap[0] < 8:
        print("  !! sm < 80, bf16 支持不完整, 这条路基本走不通")

    # bitsandbytes:NF4 反量化的 CUDA kernel 靠它。ARM 上装不上是最常见的一堵墙。
    try:
        import bitsandbytes as bnb

        info["bitsandbytes"] = bnb.__version__
        print(f"  bitsandbytes   : {bnb.__version__} ✓")
    except Exception as e:
        info["bitsandbytes"] = f"FAIL: {e}"
        print(f"  bitsandbytes   : ✗ {e}")
        print("     → NF4 反量化没有 kernel, 必崩。ARM 上装:pip install bitsandbytes --no-build-isolation")

    # attention 后端:ARM 上 flash-attn 基本装不上, 会静默退化到 SDPA(更慢)或直接崩。
    backends = []
    for mod in ("flash_attn", "sageattention", "xformers"):
        try:
            __import__(mod)
            backends.append(mod)
        except Exception:
            pass
    backends.append("torch-sdpa")
    info["attn_backends"] = backends
    print(f"  attention 可用 : {', '.join(backends)}")
    if backends == ["torch-sdpa"]:
        print("     → 只有 SDPA, 长序列会明显更慢;这是 ARM 上的预期状态, 不是错误")

    try:
        import diffsynth

        info["diffsynth"] = getattr(diffsynth, "__version__", "unknown")
        print(f"  DiffSynth      : {info['diffsynth']} ✓")
    except Exception as e:
        info["diffsynth"] = f"FAIL: {e}"
        print(f"  DiffSynth      : ✗ {e}")
        print("     → git clone https://github.com/modelscope/DiffSynth-Studio && pip install -e '.[quant]'")

    # 权重就位检查
    print(f"  权重目录       : {args.nf4_dir}")
    need = {
        "DiT": Path(args.nf4_dir) / f"minimax-h3-{args.task_family}-nf4.safetensors",
        "TextEncoder": Path(args.nf4_dir) / "minimax-h3-text-encoder-nf4.safetensors",
        "VideoVAE": Path(args.nf4_dir) / "video_vae_nf4.safetensors",
        "AudioVAE": Path(args.nf4_dir) / "audio_vae_nf4.safetensors",
        "Processor": Path(args.processor_dir),
    }
    ok = True
    for k, p in need.items():
        if p.exists():
            sz = p.stat().st_size / 2**30 if p.is_file() else 0
            print(f"    ✓ {k:12s} {p.name}" + (f"  ({sz:.1f} GB)" if sz else ""))
        else:
            ok = False
            print(f"    ✗ {k:12s} 缺失: {p}")
    info["weights_ready"] = ok
    if not ok:
        print('     → 先跑 scripts/download_minimax_h3.sh (MODELS="nf4_fl2va docs")')

    # 显存账
    est_weights = 16.0 + 14.3 + 1.5 + 0.3
    lat_t = max(1, args.frames // 4)
    tokens = lat_t * (args.height // 32) * (args.width // 32)
    info["est_weight_gb"] = est_weights
    info["est_tokens"] = tokens
    print(f"  显存预估       : 权重 ~{est_weights:.1f} GB / 卡上 {info.get('vram_total_gb', '?')} GB → 余 ~{info.get('vram_total_gb', 0) - est_weights:.1f} GB 给激活")
    print(f"  序列长度预估   : {lat_t}(latent帧) × {args.height // 32} × {args.width // 32} ≈ {tokens:,} token")
    if tokens > 40000 and args.offload == "none":
        print("     !! token 数偏大, 全常驻大概率 OOM;先把 --height/--width/--frames 压下来")
    print()
    return info


def build_model_configs(ModelConfig, args, torch):
    """
    照模型卡的 vram_config 组装, 但先对 ModelConfig 做签名自省 —— DiffSynth 主线在动,
    参数名对不上时宁可自动剔除, 也别让脚本在加载完 30G 之后才抛 TypeError。
    """
    sig = set(inspect.signature(ModelConfig.__init__).parameters)

    if args.offload == "none":
        # 全常驻:权重一次性进显存, 不做逐层搬运。这是本机唯一可能有可用速度的模式。
        vram = dict(
            offload_device="cuda",
            offload_dtype=torch.bfloat16,
            onload_device="cuda",
            onload_dtype=torch.bfloat16,
            preparing_device="cuda",
            preparing_dtype=torch.bfloat16,
            computation_device="cuda",
            computation_dtype=torch.bfloat16,
        )
    elif args.offload == "cpu":
        # 权重常驻内存(256G 够), 逐层上卡。比 disk 快, 但每步都吃 PCIe 带宽。
        vram = dict(
            offload_device="cpu",
            offload_dtype=torch.bfloat16,
            onload_device="cpu",
            onload_dtype=torch.bfloat16,
            preparing_device="cuda",
            preparing_dtype=torch.bfloat16,
            computation_device="cuda",
            computation_dtype=torch.bfloat16,
        )
    else:  # disk —— 最省显存也最慢, 本机走 NFS 更慢, 只在前两者都不行时用
        vram = dict(
            offload_device="disk",
            offload_dtype="disk",
            onload_device="cpu",
            onload_dtype=torch.bfloat16,
            preparing_device="cuda",
            preparing_dtype=torch.bfloat16,
            computation_device="cuda",
            computation_dtype=torch.bfloat16,
        )

    dropped = [k for k in vram if k not in sig]
    vram = {k: v for k, v in vram.items() if k in sig}
    if dropped:
        print(f"  (ModelConfig 不认这些参数, 已剔除: {dropped})")

    nf4 = Path(args.nf4_dir)
    files = [
        nf4 / f"minimax-h3-{args.task_family}-nf4.safetensors",
        nf4 / "minimax-h3-text-encoder-nf4.safetensors",
        nf4 / "video_vae_nf4.safetensors",
        nf4 / "audio_vae_nf4.safetensors",
    ]

    # 本地文件:优先 path=;老版本没有 path 就退回 model_id+origin_file_pattern(会走缓存目录)
    key = "path" if "path" in sig else None
    if key is None:
        print("  !! ModelConfig 没有 path 参数, 退回 model_id 模式 —— 需要外网或已填充的 MODELSCOPE_CACHE")
        cfgs = [ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern=f.name, **vram) for f in files]
    else:
        cfgs = [ModelConfig(path=str(f), **vram) for f in files]

    proc_kw = {"path": str(args.processor_dir)} if key else {"model_id": "MiniMax/MiniMax-H3", "origin_file_pattern": "FL2VA/processor/"}
    proc = ModelConfig(**proc_kw)

    print(f"  解析出的 ModelConfig kwargs: {sorted(vram)} + {key or 'model_id'}")
    return cfgs, proc


def main():
    root = find_models_root()
    ap = argparse.ArgumentParser(description="MiniMax-H3 NF4 单卡冒烟测试")
    ap.add_argument("--nf4-dir", default=str(root / "MiniMax-H3-NF4"))
    ap.add_argument("--processor-dir", default=str(root / "MiniMax-H3" / "FL2VA" / "processor"))
    ap.add_argument("--task", choices=["t2va", "fl2va"], default="t2va", help="t2va=纯文生;fl2va=带首/尾帧(需 --first-frame/--last-frame)")
    ap.add_argument("--first-frame", default=None)
    ap.add_argument("--last-frame", default=None)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    # 默认对齐模型卡示例(约 12k token), 这是 40G 卡上最有希望全常驻的一档。
    # pipeline 默认是 768×1344(≈92k token), 在 40G 上必爆, 所以这里显式压下来。
    # 注意 num_frames 会被 pipeline 向上吸附到最近的 17n+5(124 = 17×7+5, 正好不用调)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--offload", choices=["none", "cpu", "disk"], default="none", help="none=全常驻(默认, 最快);cpu=逐层上卡;disk=最省显存最慢")
    ap.add_argument("--out", default="/root/h3_smoke.mp4")
    ap.add_argument("--metrics", default=None, help="指标 JSON 落盘路径(默认 <out>.json)")
    ap.add_argument("--dry-run", action="store_true", help="只做环境自检, 不加载权重")
    args = ap.parse_args()
    args.task_family = "fl2va"  # NF4 仓里 t2va 和 fl2va 共用同一份 DiT
    args.metrics = args.metrics or (args.out + ".json")

    m = preflight(args)
    if args.dry_run:
        print("--dry-run: 到此为止, 未加载权重。")
        return 0
    if not m.get("cuda") or not m.get("weights_ready"):
        print("!! 自检未通过, 中止(要强行试就把缺的补上再来)")
        return 2

    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from diffsynth.utils.data.audio_video import write_video_audio

    print("=" * 72)
    print(f"加载权重 (offload={args.offload})")
    print("=" * 72)
    cfgs, proc = build_model_configs(ModelConfig, args, torch)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    kw = dict(torch_dtype=torch.bfloat16, device="cuda", model_configs=cfgs, processor_config=proc)
    # vram_limit 是逐层调度的阈值;全常驻模式下不该设, 设了反而会触发分层搬运
    if args.offload != "none":
        kw["vram_limit"] = torch.cuda.mem_get_info()[1] / 2**30 - 2
    try:
        pipe = MiniMaxH3Pipeline.from_pretrained(**kw)
    except torch.cuda.OutOfMemoryError:
        print("\n!! 加载阶段就 OOM —— 权重放不进单卡。改用 --offload cpu 重试。")
        return 3
    load_s = time.time() - t0
    peak_load = torch.cuda.max_memory_allocated() / 2**30
    print(f"  加载耗时 {load_s:.1f}s | 加载后显存峰值 {peak_load:.1f} GB")

    print("=" * 72)
    print(f"推理 {args.width}×{args.height} / {args.frames}帧 / {args.steps}步 / seed={args.seed}")
    print("=" * 72)
    call = dict(prompt=args.prompt, height=args.height, width=args.width, num_frames=args.frames, num_inference_steps=args.steps, seed=args.seed)
    if args.task == "fl2va":
        # 真实签名是 keyframes=[PIL] + keyframe_indices=[0|-1](0=首帧, -1=尾帧), 不是 input_image/end_image
        from PIL import Image

        kfs, idxs = [], []
        if args.first_frame:
            kfs.append(Image.open(args.first_frame).convert("RGB"))
            idxs.append(0)
        if args.last_frame:
            kfs.append(Image.open(args.last_frame).convert("RGB"))
            idxs.append(-1)
        if kfs:
            call["keyframes"] = kfs
            call["keyframe_indices"] = idxs
        else:
            print("  !! --task fl2va 但没给首/尾帧, 等价于 t2va")

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    try:
        video, audio = pipe(**call)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"\n!! 推理 OOM (峰值 {peak:.1f} GB)。按性价比依次试:")
        print("   1) 降分辨率/帧数: --width 640 --height 384 --frames 61")
        print("   2) 退到逐层上卡: --offload cpu   (慢很多, 每步都吃 PCIe 带宽)")
        return 3
    except TypeError as e:
        print(f"\n!! pipe() 参数对不上: {e}")
        print("   → 看 diffsynth/pipelines/minimax_h3_audio_video.py 的 __call__ 签名, 改上面的 call 字典")
        return 4
    infer_s = time.time() - t1
    peak_infer = torch.cuda.max_memory_allocated() / 2**30

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(video=video, audio=audio, output_path=args.out, fps=args.fps, audio_sample_rate=32000)

    lat_t = max(1, args.frames // 4)
    tokens = lat_t * (args.height // 32) * (args.width // 32)
    metrics = {
        "env": m,
        "config": {k: getattr(args, k) for k in ("task", "offload", "height", "width", "frames", "steps", "seed")},
        "est_tokens": tokens,
        "load_s": round(load_s, 1),
        "infer_s": round(infer_s, 1),
        "s_per_step": round(infer_s / args.steps, 2),
        "peak_vram_load_gb": round(peak_load, 1),
        "peak_vram_infer_gb": round(peak_infer, 1),
        "out": args.out,
    }
    Path(args.metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("=" * 72)
    print("完成")
    print("=" * 72)
    print(f"  出片        : {args.out}")
    print(f"  指标        : {args.metrics}")
    print(f"  加载耗时    : {load_s:.1f} s")
    print(f"  推理耗时    : {infer_s:.1f} s  ({infer_s / args.steps:.2f} s/step)")
    print(f"  显存峰值    : 加载 {peak_load:.1f} GB / 推理 {peak_infer:.1f} GB (卡上 {m.get('vram_total_gb')} GB)")
    print(f"  序列长度    : ~{tokens:,} token")
    print()
    print("  接下来判断三件事:")
    print("   1) 画质 —— 本地只有 768p 且没有 H3-Context-IR(没开源), 效果注定低于海螺网页版,")
    print("      看这个'打折后'的效果是否还在可用线上;音画同步和口型也一并看。")
    print("   2) 速度 —— s/step 乘以生产要的步数, 对比 InfiniteTalk 93.8s 那条线还有多远。")
    print("   3) 余量 —— 显存峰值离 40G 还剩多少, 决定能不能往 768p / 更长帧数走。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
